// #define DEBUG
// #define INCLUDE_DEBUG_TASK
// #include <set>
// #include <string>
// std::set<std::string> included_tasks = {"Task<int> producer(CoQueue<int>&)"};
#include <core.hpp>


Task<int> producer(CoQueue<int>& queue) {
    for (int i = 0; i < 10; i++) {
        co_await waitMS(100);
        // queue.push(i);
    }

    co_return 0;
}

Task<int> consumer(CoQueue<int>& queue) {
    for (int i = 0; i < 10; i++) {
        // int value = co_await queue.pop();
        int value = 0;
        
        std::cout << "Consumer got value: " << value << std::endl;
    }

    co_return 0;
}


ErrTask init(int argc, char** argv) {
    std::cout << "Hello, world!" << std::endl;

    CoQueue<int> queue;

    Task<int> producer_task = producer(queue);
    Task<int> consumer_task = consumer(queue);

    co_await producer_task;
    co_await consumer_task;



    co_return_void;
}   
