
#include <core.hpp>

uint64_t count = 0;
Fiber f1 () {
    count++;
    co_return;
}

Task<int> tm() {
    co_await waitMS(1);
    count++;
    co_return 1;    
}

Task<int> t() {
    count++;
    co_return 1;
}

Task<void> tv () {
    count++;
    co_return;
}

Task<void> tvm () {
    co_await waitMS(1);
    count++;
    co_return;
}

ErrTask init(int argc, char **argv) {
    int N = 1000;

    count = 0;
    for(int i = 0; i < N; i++) {
        f1();
    }

    if (count != N) {
        co_return Error("count != " + std::to_string(N) + " but " + std::to_string(count));
    }

    count = 0;
    for (int i = 0; i < N; i++) {
        auto a = t();
    }

    if (count != N) {
        co_return Error("count != " + std::to_string(N) + " but " + std::to_string(count));
    }

    count = 0;
    for (int i = 0; i < N; i++) {
        auto a = co_await tm();
    }   

    if (count != N) {
        co_return Error("count != " + std::to_string(N) + " but " + std::to_string(count));
    }

    count = 0;
    for (int i = 0; i < N; i++) {
        auto a = tm();
    }   

    while(count != N) {
       auto a = co_await waitMS(1);
    }

    // Task<void>
    count = 0;
    for (int i = 0; i < N; i++) {
        auto a = tv();
    }

    if (count != N) {
        co_return Error("count != " + std::to_string(N) + " but " + std::to_string(count));
    }

    count = 0;
    for (int i = 0; i < N; i++) {
        co_await tvm();
    }

    if (count != N) {
        co_return Error("count != " + std::to_string(N) + " but " + std::to_string(count));
    }

    count = 0;
    for (int i = 0; i < N; i++) {
        auto a = tvm();
    }

    while(count != N) {
       auto a = co_await waitMS(1);
    }

    co_return_void;
}