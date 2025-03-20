#define DEBUG

#include <set>  
#include <string>

// #define INCLUDE_DEBUG_TASK
// std::set<std::string> included_tasks = {"Task<Res<int> > t3()"};

#include <core.hpp>

Task<int> t4 () {
    try_await(waitMS(1));    
    co_return 1;
}

Task<int> t3 () {
    co_await t4();
    co_return 1;
}

Task<int> t2 () {
    co_await t3();
    co_return 1;
}

Task<int> t () {
    co_await t2();
    co_return 1;
}

Task<int> t4_imm() {
    co_return 1;
}

Task<int> t3_imm() {
    co_await t4_imm();
    co_return 1;
}

Task<int> t2_imm() {
    co_await t3_imm();
    co_return 1;
}

Task<int> t_imm() {
    co_await t2_imm();
    co_return 1;
}

Task<int> t_mix10() {
    try_await(waitMS(1));
};

Task<int> t_mix9() {
    co_await t_mix10();
};

Task<int> t_mix8() {
    try_await(waitMS(1));
    co_await t_mix9();
};

Task<int> t_mix7() {
    co_await t_mix8();
};

Task<int> t_mix6() {
    try_await(waitMS(1));
    co_await t_mix7();
};

Task<int> t_mix5() {
    co_await t_mix6();
};

Task<int> t_mix4() {
    try_await(waitMS(1));
    // co_await t_mix5();
    co_return 1;
};

Task<int> t_mix3() {
    co_await t_mix4();
    co_return 1;

};

Task<int> t_mix2() {
    // try_await(waitMS(1));
    co_await t_mix4();
    co_return 1;
};

Task<int> t_mix1() {
    co_await t_mix2();
    co_return 1;

};

Task<int> t_mix() {
    co_await t_mix1();
    co_return 1;
};

Task<Res<void>> init(int argc, char **argv) {

    int i = 1;
    // {
    //     i = co_await t();
    // }

    // {
    //     i = co_await t_imm();
    // }

    {
        i = co_await t_mix();
    }

    co_return_void;
}

// struct co_gethandle
// {
//     std::coroutine_handle<> _handle;

//     bool await_ready() const noexcept { return false; }
//     bool await_suspend(std::coroutine_handle<> handle) noexcept { _handle = handle; return false; }
//     std::coroutine_handle<> await_resume() noexcept { return _handle; }
// };