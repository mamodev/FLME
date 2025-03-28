#pragma once

#include <cstdint>  
#include <vector>

#include "../futures.hpp"
#include "../engine/io_uring.hpp"

struct WaitGroup {
    uint32_t count = 0;
    std::vector<FutureHandler<int>*> futures;
    
    void add(int n) {
        count += n;
    }

    void done() {
        if (count == 0) {
            return;
        }

        count--;

        if (count == 0) {
            std::vector<FutureHandler<int>*> newFutures;
            std::swap(futures, newFutures);


            // std::cout << "Done called, resuming " << newFutures.size() << " futures" << std::endl;

            for (auto &f : newFutures) {
                f->set(newFutures.size());
            }
        }
    }

    bool isDone() {
        return count == 0;
    }

    Future<int> wait() {
        if (count == 0) {
            return Future<int>(0);
        }

        FutureHandler<int> *handler = new FutureHandler<int>();
        handler->set_deallocator(delete_ptr<FutureHandler<int>>, handler);
        
        futures.push_back(handler);
        return Future<int>(handler);
    }
};


template <typename T>
struct CoQueue {
    std::vector<T> data_buffer;
    std::vector<FutureHandler<T>*> waiters;

    CoQueue() {
        data_buffer.reserve(10);
        waiters.reserve(10);
    }

    Future<T> pop() {
        if (data_buffer.empty()) {
            FutureHandler<T>* waiter = new FutureHandler<T>();
            waiter->set_deallocator(delete_ptr<FutureHandler<T>>, waiter);

            waiters.push_back(waiter);
            return Future<T>(waiter);
        } else {
            T value = data_buffer.back();
            data_buffer.pop_back();
            return value;
        }
    }

    void push(T value) {
        if (waiters.empty()) {
            data_buffer.push_back(value);
        } else {
            FutureHandler<T>* waiter = waiters.back();
            waiters.pop_back();
            waiter->set(value);
        }
    }

    CoQueue(const CoQueue&) = delete;
    CoQueue& operator=(const CoQueue&) = delete;
};
