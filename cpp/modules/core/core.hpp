#pragma once

// 3 type of functions:
// 1. fn that return a value or an error code: Result<T, Error>
// 2. fn that return void or an error: Optional<Error>
// 3. fn that return value no error code: T

// All of this can be a future result, a Future<T> wich can be co_awaited to get T
// The complication of this is that we don't want to add overhead where it's not needed
// So if a Future could be resolved synchronously, we don't want to allocate memory for it

// To solve this we Always return Promise<T> as copy, but inside the Promise has an union of T and a Ptr to T
// Or better a Ptr to FutureHandler<T> so that the handler can be allocated manually and keeped for async resolution


// Memory management of Futures:
// 1. If the Future is resolved synchronously, the value is stored in the Future itself so it use the move/copy constructor to pass value T;
// 2. If the Future is resolved asynchronously, the value is stored in a FutureHandler<T> that is allocated somewhere else and the Future has a pointer to it
//    so the FutureHandler<T> is not copied or moved, just the pointer is copied
//    for this reason the FutureHandler<T> Must in some way deallocated.

#include "results.hpp"
#include "futures.hpp"

#include "defer.hpp"

#include "engine/io_uring.hpp"

#include "network/utils.hpp"

#include "corutines/task.hpp"
#include "corutines/fiber.hpp"
#include "corutines/sync.hpp"

using ErrTask = Task<Res<void>>;

Task<Res<void>> init(int argc, char** argv); 

int exitCode = -1;

Fiber __main__fiber(int argc, char** argv) { 
    auto res = co_await init(argc, argv); 
    if(!res.is_ok()) { 
        std::cout << "Error: " << res.getError().message << std::endl; 
        exitCode = 1;
    } 

    exitCode = 0;
    co_return;
} 

int main(int argc, char** argv) { 
    auto res = loop.init(1024); 
    if(!res.is_ok()) { 
        std::cerr << "Error initializing event loop: " << res.getError().message << std::endl; \
        return 1; 
    } 
    __main__fiber(argc, argv); 
    loop.loop(); 

    if(exitCode == -1) {
        std::cerr << "No exit code set, this should never happen" << std::endl;
        return 2;
    }

    return exitCode;
}