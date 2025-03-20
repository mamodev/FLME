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