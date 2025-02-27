
#include "asyncc.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

class RaftNode { 
    uint32_t port;
    uint16_t id;
public:

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

enum class RaftRole {
    Follower = 0,
    Candidate = 1,
    Leader = 2
};
struct RaftState {
    int currentTerm = 0;
    int votedFor = -1;
    std::vector<int> log;
};
class RaftServer {
private:
    int server_fd;
    int port;
    std::vector<RaftNode> peers;
    RaftState state;

public:
    RaftServer(int port, std::vector<RaftNode> peers) {
        this->peers = peers;
        this->port = port;
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
                case 0: {
                    try_await(send_socket_direct(fd, buffer, bytes, 0));
                    break;
                }
                default: {
                    std::cerr << "Unknown packet type received" << std::endl;
                    break;
                }
            }
        }
    }

    Task<Res<uint8_t>> run() {
        server_fd = try_await(server_socket(port));

        std::cout << "[RaftNode " << port << "] Server started on port " << port << std::endl;
        while(loop.running) {
            int client_fd = try_await(accept_socket_direct(server_fd, nullptr, nullptr));
            client_handler(client_fd);
        }

        co_return 0;
    }   
 
};

Fiber _main(int port, std::vector<RaftNode> peers) {
    RaftServer server(port, peers);
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

        _main(node.getPort(), peers);
        
    }
    
    std::cout << "Starting event loop" << std::endl;
    loop.loop();
    return 0;
}