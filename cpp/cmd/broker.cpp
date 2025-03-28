#include <core.hpp>

#include <map>



struct RemoteServer {
    int fd;
};

std::map<std::string, RemoteServer&> servers;

#define MIN_CHANNEL_NAME_LEN 1

ErrTask client_setup(int fd) {
    uint32_t len;
    try_await(recv_all_direct(fd, (uint8_t*)&len, sizeof(len)), {
        co_await close_file_direct(fd);
    });

    uint8_t client_type;
    uint16_t channel_name_len;
    if (len < sizeof(client_type) + sizeof(channel_name_len) + MIN_CHANNEL_NAME_LEN) {
        co_await close_file_direct(fd);
        co_return_void;
    }

    uint8_t* buf = co_propagate(try_alloc(len), {
        co_await close_file_direct(fd);
    });

    try_await(recv_all_direct(fd, buf, len), {
        co_await close_file_direct(fd);
        delete[] buf;
    });


    client_type = buf[0];
    channel_name_len = *(uint16_t*)(buf + sizeof(client_type));

    if (len < sizeof(client_type) + sizeof(channel_name_len) + channel_name_len) {
        co_await close_file_direct(fd);
        delete[] buf;
        co_return_void;
    }

    std::string channel_name((char*)(buf + sizeof(client_type) + sizeof(channel_name_len)), channel_name_len);
    delete[] buf;

    if(client_type == 0) {
        // client is opening a new channel
        // ensure channel name is not already in use
        if (servers.find(channel_name) != servers.end()) {
            co_await close_file_direct(fd);
            co_return_void;
        };

        servers[channel_name] = RemoteServer(fd);
    } else {
        auto server = servers.find(channel_name);
        if (server == servers.end()) {
            co_await close_file_direct(fd);
            co_return_void;
        }

        // send server a connection request
        uint8_t conn_req = 1;
        


    }

    co_return_void;
}

ErrTask init(int argc, char** argv) {
    auto fd = try_await(server_socket(8888));

    while (true) {
        auto client = try_await(accept_socket_direct(fd, nullptr, nullptr));
        client_setup(client);
    }

    co_return_void;
}   
