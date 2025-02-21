
#include "asyncc.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


Task<void> client(int PORT)
{
    int fd = try_await(open_socket(AF_INET, SOCK_STREAM, 0, 0));
    defer_async
    {
        co_await close_file(fd);
    };

    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    try_await(connect_socket(fd, (sockaddr *)&addr, sizeof(addr)));

    std::string message = "Hello from client";
    try_await(send_socket(fd, message.data(), message.size(), 0));

    char buffer[1024];
    int n = try_await(recv_socket(fd, buffer, 1024, 0));
    buffer[n] = '\0';

    co_return;
}

Fiber client_test(int PORT)
{   
    try_await(waitMS(1000));

    while (true)
    {
        co_await client(PORT);
        // try_await(waitMS(200));
    }

    co_return;
}

Fiber client_handle(int fd)
{
    defer_async
    {
        co_await close_file_direct(fd);
        std::cout << "Closed client: " << fd << std::endl;
    };

    std::cout << "Fiber client: " << fd << std::endl;

    char buffer[1024];

    int n = try_await(recv_socket_direct(fd, buffer, 1024, 0));
    buffer[n] = '\0';

    try_await(send_socket_direct(fd, buffer, n, 0));
    co_return;
}

Fiber server(int PORT)
{
    // co_await defer_tests();
    int fd = try_await(open_socket(AF_INET, SOCK_STREAM, 0, 0));
    defer_async
    {
        co_await close_file(fd);
    };

    int opt = 1;
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int)) < 0)
    {
        std::cerr << "Error setting socket options SO_REUSEADDR: " << errno << std::endl;
        co_return;
    }

    if (setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(int)) < 0)
    {
        std::cerr << "Error setting socket options SO_REUSEPORT: " << errno << std::endl;
        co_return;
    }

    try_await(bind_socket(fd, (sockaddr *)&addr, sizeof(addr)));
    try_await(listen_socket(fd, 10));

    while (true)
    {
        int client_fd = try_await(accept_socket_direct(fd, nullptr, nullptr));
        
        std::cout << "Accepted client: " << client_fd << std::endl;
        client_handle(client_fd);
    }
    co_return;
}



#include <sched.h>
#include <unistd.h>

void pinProcessToCPU(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);        
    CPU_SET(cpu_id, &cpuset);  // Set the CPU core

    pid_t pid = getpid();  // Get current process ID

    if (sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Error setting process affinity\n";
    } else {
        std::cout << "Process pinned to CPU " << cpu_id << "\n";
    }
}


int process(int n)
{
    std::cout << "PID: " << getpid() << std::endl;
    pinProcessToCPU(n);
    auto s = loop.init(100);
    if(!s.is_ok())
    {
        std::cerr << "Error initializing loop: " << s.getError().message << std::endl;
        return 1;
    }

    if (n == 0)
        server(8080);
    else    
        client_test(8080);


    loop.loop();

    std::cout << "Exiting main" << std::endl;
    return 0;
}

int main() {
    int n = 5;  // Number of child processes to spawn
    pid_t pid;

    // Loop to create n child processes
    for (int i = 1; i <= n; i++) {
        pid = fork();

        if (pid == -1) {
            perror("fork failed");
            return 1;
        } else if (pid == 0) {
            process(i);  // Child process
            return 0;  // Exit the child process after printing
        }
        // Parent process continues to fork new child without printing
    }

    process(0);  // Parent process

    for (int i = 1; i <= n; i++) {
        wait(NULL); 
    }

    printf("Parent: All child processes have finished.\n");
    return 0;
}
