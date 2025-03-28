#pragma once

struct __co_tag_get_handle_t {};
#define get_current_handle() co_yield __co_tag_get_handle_t{}

#define ENABLE_GET_CURRENT_HANDLE \
auto yield_value(__co_tag_get_handle_t) { \
    struct __co_return_handle { \
        std::coroutine_handle<promise_type> handle; \
        bool await_ready() noexcept { return true; } \
        void await_suspend(std::coroutine_handle<>) noexcept {} \
        auto await_resume() noexcept { \
            return handle; \
        } \
    }; \
    return __co_return_handle{.handle = std::coroutine_handle<promise_type>::from_promise(*this)}; \
}
