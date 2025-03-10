
#include "asyncc.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

class RaftNode { 
    uint32_t port;
    uint16_t id;
public:

    uint16_t getID() {
        return id;
    }

    int getPort() {
        return port;
    }

    RaftNode(uint32_t port, uint16_t id) : port(port), id(id) {}

    Task<Res<int>> connect() {
        int fd = try_await(open_socket(AF_INET, SOCK_STREAM, 0, 0));
        defer_err_async {
            co_await close_file(fd);
        };

        sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;
        try_await(connect_socket(fd, (sockaddr*)&addr, sizeof(addr)));

        co_return fd;
    }

    Task<Res<int>> send_message(uint8_t* buffer, size_t len) {
        int fd = try_await(connect());
        defer_async {
            co_await close_file(fd);
        };

        size_t sent = 0;
        while (sent < len) {
            int bytes = try_await(send_socket_direct(fd, buffer + sent, len - sent, 0));
            sent += bytes;
        }

        co_return 0;
    }

    Task<Res<int>> send_message_with_retry(uint8_t* buffer, size_t len) {
        int ret = 1;
        while (true) {
            auto res = co_await send_message(buffer, len);
            if(res.is_ok()) {
                co_return res.getValue();
            }   

            int timeout = ret * 200;
            if (ret < 100) {
                ret++;
            }

            try_await(waitMS(timeout));
        }
    }
};

#include <random>


#define REQUEST_VOTE_PACKET 0
struct RequestVotePacket {
    uint32_t packet_size = sizeof(RequestVotePacket);
    uint32_t packet_type = REQUEST_VOTE_PACKET;
    uint64_t term;
    int candidateId;
    uint64_t lastLogIndex;
    uint64_t lastLogTerm;
};

struct RequestVoteResponsePacket {
    uint64_t term;
    bool voteGranted;
};


enum class RaftRole {
    Follower = 0,
    Candidate = 1,
    Leader = 2
};
struct RaftState {
    uint64_t currentTerm = 0;
    int votedFor = -1;
    std::vector<uint64_t> log;
    RaftRole role = RaftRole::Follower;
};
class RaftServer {
private:
    std::mt19937 random_gen;
    std::uniform_int_distribution<int> election_timeout_dist;
    std::optional<TimeoutCancelToken> election_timeout;


    int server_fd;
    int port;
    std::vector<RaftNode> peers;
    RaftState state;
    int id;

public:
    RaftServer(int id, int port, std::vector<RaftNode> peers) {
        this->id = id;
        this->peers = peers;
        this->port = port;
        this->state = RaftState();

        this->election_timeout = std::nullopt;
        this->random_gen = std::mt19937(std::random_device()());
        this->election_timeout_dist = std::uniform_int_distribution<int>(150, 300);
    }

    Fiber client_handler(int fd) {
        defer_async {
            co_await close_file_direct(fd);
        };

        while(loop.running) {
            char buffer[1024];
            int bytes = try_await(recv_socket_direct(fd, buffer, 1024, 0));

            if (bytes < 4) {
                std::cerr << "Invalid packet received" << std::endl;
                co_return;
            }
        
            uint32_t packetLen = *(uint32_t*)buffer;
            if (packetLen > 1020) {
                std::cerr << "Malformed packet received (len > 1020)" << std::endl;
                co_return;
            }
        
            uint32_t packetType = *(uint32_t*)(buffer + 4);
        
            switch (packetType) {
                case REQUEST_VOTE_PACKET: {
                    if (packetLen != sizeof(RequestVotePacket)) {
                        std::cerr << "Malformed packet received (len != sizeof(RequestVotePacket))" << std::endl;
                        co_return;
                    }
        
                    RequestVotePacket* packet = (RequestVotePacket*)buffer;
                    RequestVoteResponsePacket response = {
                        .term = state.currentTerm,
                        .voteGranted = false
                    };
        
                    if (packet->term < state.currentTerm) {
                        response.voteGranted = false;
                    } else if (state.votedFor == -1 || state.votedFor == packet->candidateId) {
                        response.voteGranted = true;
                        state.votedFor = packet->candidateId;
                    }

                    std::cout << "[ RaftNode " << port << "] Voted for " << packet->candidateId << " in term " << packet->term << " with voteGranted = " << response.voteGranted << std::endl;
        
                    co_await send_socket_direct(fd, (uint8_t*)&response, sizeof(response), 0);
                    co_return;
                   
                }
                default: {
                    
                    std::cerr << "Unknown packet type received" << std::endl;
                    break;
                }
            }
        }
    }

    Res<void> setElectionTimeout() {
        if (election_timeout.has_value()) {
            return Error("Election timeout already set, THIS IS AN INVALID STATE and should never happen");
        }
        
        int timeout = election_timeout_dist(random_gen);
        setTimeout(timeout, [this]() -> Fiber {
            std::cout << "[RaftNode " << port << "] Election timeout triggered" << std::endl;
            election_timeout = std::nullopt;

            // start election
            this->state.currentTerm++;
            this->state.votedFor = this->id;
            this->state.role = RaftRole::Candidate;

            RequestVotePacket packet = {
                .term = this->state.currentTerm,
                .candidateId = this->id,
                .lastLogIndex = this->state.log.size(),
                .lastLogTerm = this->state.log.size() > 0 ? this->state.log.back() : 0
            };

            std::vector<std::shared_ptr<Task<Res<int>>>> tasks;

            for (auto& peer : peers) {
                // Use std::shared_ptr to manage the lifetime of the task
                tasks.push_back(std::make_shared<Task<Res<int>>>(peer.send_message_with_retry((uint8_t*)&packet, sizeof(packet))));
            }
            

            for (auto& task : tasks) {
                co_await *task;
            }

            co_return;
        });
        
        return std::nullopt;
    }


    Task<Res<uint8_t>> run() {
        server_fd = try_await(server_socket(port));
        defer_async {
            co_await close_file(server_fd);
        };

        setElectionTimeout();
 

        std::cout << "[RaftNode " << port << "] Server started on port " << port << std::endl;
        while(loop.running) {
            int client_fd = try_await(accept_socket_direct(server_fd, nullptr, nullptr));
            client_handler(client_fd);
        }

        co_return 0;
    }   
 
};

Fiber _main(int id, int port, std::vector<RaftNode> peers) {
    RaftServer server(id, port, peers);
    try_await(server.run());
    co_return;
}

int main() {
    auto res = loop.init(1024);
    if(!res.is_ok()) {
        std::cerr << "Error initializing event loop: " << res.getError().message << std::endl;
        return 1;
    }

    std::vector<RaftNode>nodes = { RaftNode{8080, 0}, RaftNode{8081, 1}, RaftNode{8082, 2}, RaftNode{8083, 3} };

    for (auto& node : nodes) {
        std::vector<RaftNode> peers = {};
        for (auto& peer : nodes) {
            if (peer.getPort() != node.getPort()) {
                peers.push_back(peer);
            }
        }

        _main(node.getID(), node.getPort(), peers);
        
    }

    std::cout << "Starting event loop" << std::endl;
    loop.loop();
    return 0;
}