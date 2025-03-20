#define DEBUG

#include <core.hpp>



struct TestCtx {
    int server_fd;
    int server_port;

    WaitGroup server_wg;

    uint64_t bytes;
    uint64_t packets;
    WaitGroup wg;
};


ErrTask client(TestCtx& t) {
    t.wg.add(1);

    auto fd = try_await(client_socket("localhost", t.server_port));
    defer_async {
        co_await close_file(fd);
    };  


    uint8_t* buf = co_propagate(try_alloc(t.bytes));

    for (int i = 0; i < t.packets; i++) {
        try_await(send_socket(fd, buf, t.bytes, 0));
    }


    buf[0] = 1;
    try_await(send_socket(fd, buf, 1, 0));

    try_await(recv_socket(fd, buf, 1, 0));
    try_await(send_socket(fd, buf, 1, 0));
    t.wg.done();

    co_return_void;
}        

ErrTask server(TestCtx& t) {
    t.server_wg.add(1);

    auto fd = try_await(server_socket_direct(t.server_port));
    defer_async {
        co_await close_file_direct(fd);
    };

    t.server_wg.done();

    uint8_t* buf = co_propagate(try_alloc(t.bytes));

    std::cout << "server started" << std::endl;
    while (!t.wg.isDone())
    {
        std::cout << "waiting for connection" << std::endl;
        auto cfd = try_await(accept_socket_direct(fd, nullptr, nullptr));
        defer_async {
            std::cout << "closing connection" << std::endl;
            co_await close_file(cfd);
        };


        std::cout << "connection accepted" << std::endl;
        while (true) {
            auto stop = try_await(recv_socket_direct(cfd, buf, 1, 0));
            if (stop == 1 && buf[0] == 1) {
                std::cout << "stopping" << std::endl;
                try_await(send_socket_direct(cfd, buf, 1, 0));
                std::cout << "sent" << std::endl;
                try_await(recv_socket_direct(cfd, buf, 1, 0));
                std::cout << "received" << std::endl;
                break;
            }

            auto res = try_await(recv_all_direct(cfd, buf, t.bytes - 1));
            if (res != t.bytes - 1) {
                co_return Error("error reading from socket " + std::to_string(res) + " " + std::to_string(t.bytes));
            }
        }
    }

    co_return_void;
}

ErrTask init(int argc, char** argv) {
    TestCtx t = { 
        .server_port = 8080,
        .bytes = 10,
        .packets = 1,
        .wg = WaitGroup()
    };

    auto t_server = server(t);

    co_await t.server_wg.wait();

    std::vector<ErrTask> clients;
    for (int i = 0; i < 5; i++) {
        clients.push_back(client(t));
    }

    try_await(t_server);


    for (auto& c : clients) {
        try_await(c);
    }


    co_return_void;
}