#pragma once

#include <coroutine>

#include "./corutines/defer.hpp"
#include "./utils/fun_type.hpp"

// #include <functional>
// #include <vector>

// #include "results.hpp"

// #include "corutines/task.hpp"
// #include "corutines/fiber.hpp"


// struct defer_scope {
//     bool has_async_error = false;
//     bool has_async = false;
//     bool deferred = false;

//     struct defer_entry {
//         std::variant<std::function<void()>, std::function<Task<void>()>> f;
//         bool error;
//     };

//     std::vector<defer_entry> defers;


//     defer_scope() {}
//     defer_scope(const defer_scope& other) = delete;
    
//     template <typename F>
//     defer_scope& operator +(F&& f) {
//         add_defer(std::forward<F>(f), false);
//         return *this;
//     }
    
//     template <typename F>
//     defer_scope& operator - (F&& f) {
//         add_defer(std::forward<F>(f), true);
//         return *this;
//     }

//     template <typename F>
//     void add_defer(F&& f, bool error) {
//          if constexpr (std::is_invocable_r_v<Task<void>, F>) {
//             if (error) 
//                 has_async_error = true;
//             else 
//                 has_async = true;

//             defers.push_back({std::function<Task<void>()>(std::forward<F>(f)), error});

//         } else if constexpr (std::is_invocable_r_v<void, F>) {
//             defers.push_back({std::function<void()>(std::forward<F>(f)), error});
           
//         }  else {
//             static_assert(false, "Invalid type for defer");
//         }
//     }

//     Task<void> sync_defer() {
//         bool errors = std::uncaught_exceptions() > 0;

//         for (auto& defer : defers) {
//             if (!defer.error || errors) {
//                 if (std::holds_alternative<std::function<Task<void>()>>(defer.f)) {
//                     co_await std::get<std::function<Task<void>()>>(defer.f)();
//                 } else {
//                     std::get<std::function<void()>>(defer.f)();
//                 }
//             }
//         }

//         deferred = true;

//         co_return;
//     }

//     Fiber async_defer(bool exec_error) {
//         std::vector<defer_entry> defers_copy = defers;

//         for (auto& defer : defers_copy) {
//             if (!defer.error || exec_error) {
//                 if (std::holds_alternative<std::function<Task<void>()>>(defer.f)) {
//                     co_await std::get<std::function<Task<void>()>>(defer.f)();
//                 } else {
//                     std::get<std::function<void()>>(defer.f)();
//                 }
//             }
//         }

//         co_return;
//     }

//     ~defer_scope() {
//         if (deferred) {
//             return;
//         }

//         bool errors = std::uncaught_exceptions() > 0;

//         if (has_async || errors && has_async_error) {
//             async_defer(errors);
//         } else {
//             auto _we_can_ignore_becouse_task_will_never_suspend = sync_defer();
//         }

//         deferred = true;
//     }
// };

// #define UNIQUE_NAME(PREFIX) \
//     _UNIQUE_NAME_CONCAT(PREFIX, __LINE__)

// #define _UNIQUE_NAME_CONCAT(a, b) _UNIQUE_NAME_CONCAT2(a, b)
// #define _UNIQUE_NAME_CONCAT2(a, b) a##_##b



// struct defer_single_scope {
//     std::variant<std::function<void()>, std::function<Task<void>()>> f;
//     bool error;

//     template <typename F>
//     defer_single_scope(F&& f, bool error) : error(error) {
//         if constexpr (std::is_invocable_r_v<Task<void>, F>) {
//             this->f = std::function<Task<void>()>(std::forward<F>(f));
//         } else if constexpr (std::is_invocable_r_v<void, F>) {
//             this->f = std::function<void()>(std::forward<F>(f));
//         } else {
//             static_assert(false, "Invalid type for defer");
//         }
//     }


//     ~defer_single_scope() {
//         if (error && std::uncaught_exceptions() == 0) {
//             return;
//         }
        
//         if (std::holds_alternative<std::function<Task<void>()>>(f)) {
//             std::get<std::function<Task<void>()>>(f)();
//         } else {
//             std::get<std::function<void()>>(f)();
//         }
//     }
// };

// struct dummy_defer_scope {
//     template <typename F>
//     defer_single_scope operator+ (F&& f) {
//         return defer_single_scope(std::forward<F>(f), false);
//     }


//     template <typename F>
//     defer_single_scope operator- (F&& f) {
//         return defer_single_scope(std::forward<F>(f), true);
//     }
// };

// dummy_defer_scope __defer__scope__;


// #define defer decltype(auto) \
//     UNIQUE_NAME(__DEFER__) = __defer__scope__ + [&]() -> void

// #define defer_err decltype(auto) \
//     UNIQUE_NAME(__DEFER__) = __defer__scope__ - [&]() -> void

// #define defer_async \
//     decltype(auto) UNIQUE_NAME(__DEFER__) = __defer__scope__ + [=]() -> Task<void>

// #define defer_err_async \
//     decltype(auto) UNIQUE_NAME(__DEFER__) = __defer__scope__ - [=]() -> Task<void>

// #define squential_defer defer_scope __defer__scope__ = defer_scope();

// #define await_defer __defer__scope__.sync_defer();

// #define co_return_void co_return std::nullopt;