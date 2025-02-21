#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <iostream>
#include <coroutine>    

#include "async.hpp"

volatile bool running = 1;
void singal_handler(int sig) {
    running = 0;
}

rutine client_handler(int fd) {
    defer([&]() -> rutine {
        std::cout << "Closing client handler socket" << std::endl;
        co_await io::close_file(fd);
        std::cout << "Client handler socket closed" << std::endl;
    });
    
    while(running) {
        char buf[1024];
        int nbytes = try_await(io::recv(fd, buf, sizeof(buf), 0));
        if (nbytes <= 0) {
            break;
        }

        try_await(io::send(fd, buf, nbytes, 0));
    }
}

rutine server(int PORT, WaitGroup& wg, int backlog = 10) {
    int sock_fd = try_await(io::socket(AF_INET, SOCK_STREAM, 0, 0));
    defer([&]() -> rutine {
        std::cout << "Closing server socket" << std::endl;
        co_await io::close_file(sock_fd);
        std::cout << "Server socket closed" << std::endl;
    });

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    try_await(io::set_socket_reuse_address(sock_fd));
    try_await(io::set_socket_reuse_port(sock_fd));
    try_await(io::bind(sock_fd, (struct sockaddr*) &addr, sizeof(addr)));
    try_await(io::listen(sock_fd, backlog));
    wg.done();
    std::cout << "Server started on port " << PORT << std::endl;

    while(running) {
        int client_fd = try_await(io::accept(sock_fd, nullptr, nullptr));
        std::cout << "New client connected: " << client_fd << std::endl;
        client_handler(client_fd);
    }
}

rutine client(int PORT, WaitGroup& wg) {
    co_await wg;

    int fd = try_await(io::socket(AF_INET, SOCK_STREAM, 0, 0));
    defer([&]() -> rutine {
        std::cout << "Closing client socket" << std::endl;
        co_await io::close_file(fd);
    });

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    
    try_await(io::connect(fd, (struct sockaddr*) &addr, sizeof(addr)));
    std::cout << "Client connected to server" << std::endl;

    char buf[1024];
    while (running) {
        std::cout << "Next message: " << std::flush;
        try_await(io::read_file(0, buf, sizeof(buf), 0));
        std::cout << "Sending message" << std::endl;
        try_await(io::send(fd, buf, sizeof(buf), 0));
        try_await(io::recv(fd, buf, sizeof(buf), 0));
        std::cout << "Received message: " << buf << std::endl;

        struct __kernel_timespec ts;
        ts.tv_sec = 1;
        ts.tv_nsec = 0;

        try_await(io::timeout(&ts, 1, 0));
    }
}


int main () {   
    signal(SIGINT, singal_handler);
    signal(SIGTERM, singal_handler);
    int res = io::thread_initialize_async_engine();
    if (res < 0) {
        std::cerr << "io_uring_init failed" << std::endl;
        return 1;
    }   

    auto wg = WaitGroup(1);
    server(8080, wg);
    client(8080, wg);
    
    while ((running && corutines_active > 0) || !(io::io_uring.Idle())) {
        if(!running) {
            io::io_uring.cancel();
        }

        io::io_uring.handle_cqe();
        // std::cout << "Active corutines: " << corutines_active << std::endl;
    }
    return 0;
}