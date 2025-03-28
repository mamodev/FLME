#pragma once


#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#include "../results.hpp"
#include "../defer.hpp"

#include "../corutines/task.hpp"
#include "../engine/io_uring.hpp"


Task<Res<int>> server_socket(int port) {
    int fd = try_await(open_socket(AF_INET, SOCK_STREAM, 0, 0));

    int opt = 1;
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int)) < 0)
        co_return Error("Error setting socket options SO_REUSEADDR");

    if (setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(int)) < 0)
        co_return Error("Error setting socket options SO_REUSEPORT");

    try_await(bind_socket(fd, (sockaddr *)&addr, sizeof(addr)), {
        co_await close_file(fd);
    });

    try_await(listen_socket(fd, 10), {
        co_await close_file(fd);
    });

    co_return fd;
}

Task<Res<int>> client_socket(std::string host, int port) {
    addrinfo hints = {0}, *res, *p;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) {
        co_return Error("Error getting address info");
    }


    int socke_fd  = -1;

    for (p = res; p != nullptr; p = p->ai_next) {
        auto res = co_await open_socket(p->ai_family, p->ai_socktype, p->ai_protocol, 0);
        if (!res.is_ok()) 
            continue;   

        int _sockfd = res.getValue();
        res = co_await connect_socket(_sockfd, p->ai_addr, p->ai_addrlen);
        if (res.is_ok()) {
            socke_fd = _sockfd;
            break;
        }

        co_await close_file(_sockfd);
    }

    freeaddrinfo(res);

    if(socke_fd != -1) {
        co_return socke_fd;
    }
        
    co_return Error("Error connecting to socket");
}


Task<Res<int>> recv_all(int fd, uint8_t* buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        auto res = try_await(recv_socket(fd, buf + total, len - total, 0));
        total += res;
    }

    co_return total;
}

Task<Res<int>> server_socket_direct(int port) {
    auto fd = try_await(server_socket(port));
    co_propagate(loop.register_file(fd), {
        co_await close_file(fd);
    });

    co_return fd;
}

Task<Res<int>> recv_all_direct(int fd, uint8_t* buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        auto res = try_await(recv_socket_direct(fd, buf + total, len - total, 0));
        total += res;
    }

    co_return total;
}