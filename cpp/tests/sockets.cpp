#define DEBUG

#include <core.hpp>
#include <cstring>  

struct TestCtx {
    int server_fd;
    int server_port;

    WaitGroup server_wg;

    uint64_t bytes;
    uint64_t packets;
    WaitGroup wg;
};


ErrTask client_normal(TestCtx& t) {
    t.wg.add(1);

    auto fd = try_await(client_socket("localhost", t.server_port));
    //std::cout << "[C] Sending " << t.packets << " packets of " << t.bytes << " bytes" << std::endl;


    defer_async {
        co_await close_file(fd);
    };  

    uint8_t* buf = co_propagate(try_alloc(t.bytes));

    for (int i = 0; i < t.packets; i++) {
        try_await(send_socket(fd, buf, t.bytes, 0));
    }

    //std::cout << "Done sending" << std::endl;


    buf[0] = 1;
    try_await(send_socket(fd, buf, 1, 0));
    try_await(recv_socket(fd, buf, 1, 0));
    try_await(send_socket(fd, buf, 1, 0));
    t.wg.done();

    delete[] buf;

    co_return_void;
}        

ErrTask server_normal(TestCtx& t) {

    t.server_wg.add(1);

    auto fd = try_await(server_socket(t.server_port));
    defer_async {
        co_await close_file(fd);
    };

    t.server_wg.done();

    uint8_t* buf = co_propagate(try_alloc(t.bytes));

    std::memset(buf, 0, t.bytes);
    // --partial-loads-ok=yes

    while (!t.wg.isDone())
    {
        auto cfd = try_await(accept_socket(fd, nullptr, nullptr));
        defer_async {
            co_await close_file(cfd);
        };

        //std::cout << "[S] Accepted connection" << std::endl;

        while (true) {
            auto stop = try_await(recv_socket(cfd, buf, 1, 0));
            if (stop == 1 && buf[0] == 1) {
                //std::cout << "[S] Stopping" << std::endl;
                try_await(send_socket(cfd, buf, 1, 0));
                //std::cout << "[S] Stopped" << std::endl;
                try_await(recv_socket(cfd, buf, 1, 0));
                //std::cout << "[S] Stopped ack" << std::endl;
                break;
            }

            //std::cout << "[S] Reading " << t.bytes - 1 << " bytes" << std::endl;
            
            auto res = try_await(recv_all(cfd, buf, t.bytes - 1));
            if (res != t.bytes - 1) {
                co_return Error("error reading from socket " + std::to_string(res) + " " + std::to_string(t.bytes));
            }

            //std::cout << "[S] Read " << t.bytes - 1 << " bytes" << std::endl;
        }
    }

    //std::cout << "[S] Server done" << std::endl;

    delete[] buf;

    co_return_void;
}

ErrTask init(int argc, char** argv) {
    TestCtx t = {
        .server_port = 8080,
        .bytes = 1032,
        .packets = 120,
        .wg = WaitGroup()
    };

    auto t_server = server_normal(t);
    co_await t.server_wg.wait();

    std::vector<ErrTask> clients;
    for (int i = 0; i < 2; i++) {
        clients.push_back(client_normal(t));
    }

    //std::cout << "Waiting for clients to finish" << std::endl;

    try_await(t_server);

    for (auto& c : clients) {
        try_await(c);
    }

    co_return_void;
}