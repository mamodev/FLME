#pragma once

#include <optional>
#include <coroutine>
#include <exception>
#include <stdexcept>    

#include <iostream>

#include <utility>
#include <vector>
#include "../utils.hpp"


template <typename T, template <typename> class Template>
struct is_specialization_of : std::false_type {};

// Specialization: true if T is Template<U> for any U
template <typename U, template <typename> class Template>
struct is_specialization_of<Template<U>, Template> : std::true_type {};

// Helper variable template for convenience
template <typename T, template <typename> class Template>
inline constexpr bool is_specialization_of_v = is_specialization_of<T, Template>::value;

#define COL_RESET  "\033[0m"  // Reset
#define COL_BOLD   "\033[1m"  // Bold
#define COL_BLUE   "\033[34m" // Blue
#define COL_GREEN  "\033[32m" // Green
#define COL_YELLOW "\033[33m" // Yellow
#define COL_GRAY   "\033[90m" // Gray
#define COL_RED    "\033[31m" // Red

#ifdef EXCLUDE_DEBUG_TASK
extern std::set<std::string> excluded_tasks;

inline bool is_excluded_task(const std::string& function_name) {
    return excluded_tasks.find(function_name) != excluded_tasks.end();
}
#else
inline bool is_excluded_task(const std::string&) {
    return false;
}
#endif

#ifdef INCLUDE_DEBUG_TASK
extern std::set<std::string> included_tasks;

inline bool is_included_task(const std::string& function_name) {
    return included_tasks.find(function_name) != included_tasks.end();
}
#else
inline bool is_included_task(const std::string&) {
    return true;  // If INCLUDE_DEBUG_TASK is not defined, allow all tasks.
}
#endif

            // debug(COL_BOLD COL_BLUE "[" << function_name << ", " \
            //       << COL_GREEN << this->loc.file_name() << ":" \
            //       << COL_YELLOW << this->loc.line() \
            //       << COL_BLUE << "] " << this->handle.address() << COL_RESET << " " << stmnt << "[this: " << this << "]"); \

#define tdebug(stmnt) \
    do { \
        std::string function_name = this->loc.function_name(); \
        if (!is_excluded_task(function_name) && is_included_task(function_name)) { \
            debug( \
                COL_BLUE COL_BOLD <<  __PRETTY_FUNCTION__ << COL_RESET COL_GRAY ", " << __FILE__ << ":" << __LINE__ \
                << "\n  ↳ " << COL_GREEN COL_BOLD << function_name << COL_RESET COL_GRAY ", " \
                << "defined in file: " << this->loc.file_name() << ":" \
                << this->loc.line() << COL_RESET COL_GRAY ", this: " COL_RED COL_BOLD << this << COL_RESET COL_GRAY ", handle: " COL_RED COL_BOLD << this->handle.address() \
                << COL_GRAY "\n\t ↳ " << stmnt << COL_RESET \
            ); \
        } \
    } while (0)

#include <source_location>

template <typename T = void>
struct [[nodiscard]] Task {
    std::source_location loc;
    
    struct promise_type;
    std::coroutine_handle<promise_type> handle;
    bool* discarded = nullptr;

    Task(std::coroutine_handle<promise_type> handle, bool* discarded, std::source_location loc) : handle(handle), discarded(discarded), loc(loc) {
        tdebug("Task<T>::Task(handle, discarded, loc)");

    }

    Task() : handle(nullptr) {
        tdebug("Task<T>::Task()");
    }

    struct final_suspend_aw {
        promise_type* promise;
        std::source_location loc;
        std::coroutine_handle<> handle;

        final_suspend_aw(promise_type* promise) : promise(promise), loc(promise->loc), handle(promise->handle) {
            tdebug("final_suspend_aw::final_suspend_aw");
        }

        bool await_ready() noexcept {
            tdebug("discarded: " << promise->discarded << " has_continuation: " << promise->continuation.has_value());
            if (promise->discarded) {
                return true;
            }

            return false;
        }

        void await_suspend(std::coroutine_handle<> __ub_if_resumed) noexcept {
            tdebug("final_suspend_aw::await_suspend");

            if (promise->continuation.has_value()) {

                tdebug("final_suspend_aw::await_suspend, resuming continuation");
                promise->continuation.value().resume();
            }
        }

        void await_resume()  noexcept {
            tdebug("final_suspend_aw::await_resume");
            // // std::cout << "final_suspend_aw::await_resume" << std::endl;
            // if (promise->continuation.has_value()) {
            //     tdebug("final_suspend_aw::await_resume, resuming continuation");
            //     promise->continuation.value().resume();
            // }

            // tdebug("final_suspend_aw::await_resume, after resume")
        }

        ~final_suspend_aw() {
            tdebug("final_suspend_aw::~final_suspend_aw");
        }

    };

    struct promise_type {
        bool discarded = false;
        bool detached = false;
        std::optional<T> value;
        std::optional<std::coroutine_handle<>> continuation;

        std::coroutine_handle<promise_type> handle;
        std::source_location loc;

        promise_type(std::source_location loc = std::source_location::current()) 
        : loc(loc) , handle(std::coroutine_handle<promise_type>::from_promise(*this)) {
            // tdebug("Primise created at " << loc.file_name() << ":" << loc.line());
        }

        Task get_return_object() {
            this->value = std::nullopt;
            this->continuation = std::nullopt;
            this->discarded = false;
            return Task(handle, &discarded, loc);
        }

        std::suspend_never initial_suspend() { return {}; }

        final_suspend_aw final_suspend() noexcept {
            tdebug("promise_type::final_suspend");
            // std::cout << "task<T>::promise_type::final_suspend" << std::endl;
            return final_suspend_aw{this};
        }

        void unhandled_exception() {
            // std::cout << "task<T>::promise_type::unhandled_exception" << std::endl;
            const char * err = nullptr;
            try {
                std::rethrow_exception(std::current_exception());
            } catch (const UnwrapException& e) {
                err = e.what();
            } catch (...) {
                err = "Some exception occured";
            }

            if constexpr (is_specialization_of_v<T, Res>) {
                value = Error(err);
            } else {
                tdebug("Unhandled exception inside task, that should never fail");
                std::cerr << "Unhandled exception inside task, that should never fail" << std::endl;
                std::cerr << err << std::endl;
                std::terminate();
            }
        }

        void return_value(T v) {
            this->value = v;
            tdebug("task<T>::promise_type::return_value, has_value");
        }

        ~promise_type() {
            tdebug("task<T>::promise_type::~promise_type");
        }
    };

    bool await_ready() {
        if(!handle) {
            std::cerr << "Task<T>::await_ready() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

        auto has_value = handle.promise().value.has_value();

        tdebug("has_value: " << has_value);

        return has_value;
    }

    void await_suspend(std::coroutine_handle<> h) {
        if(!handle) {
            std::cerr << "Task<T>::await_suspend() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

        tdebug("task<T>::await_suspend");
        if (handle.promise().continuation.has_value()) {
            throw std::runtime_error("Awaiting task in invalid state, this should never happen");
        }

        handle.promise().continuation = h;
    }

    T await_resume() {
        tdebug("task<T>::await_resume");

        if(!handle) {
            std::cerr << "Task<T>::await_resume() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

        // std::cout << "task<T>::await_resume" << std::endl;
        if (!handle.promise().value.has_value()) {
            throw std::runtime_error("Resuming task in invalid state, this should never happen, the promise has no value");
        }

        return handle.promise().value.value();
    }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    Task& operator=(Task&& other) {
        if(this == &other) 
            return *this;

        
        loc = other.loc;
        handle = std::exchange(other.handle, {});
        discarded = std::exchange(other.discarded, nullptr);

        return *this;
    }

    Task(Task&& other) {
        handle = std::exchange(other.handle, {});
        discarded = std::exchange(other.discarded, nullptr);
        loc = other.loc;
    }

    ~Task() {
        if(handle) {
            tdebug("task<T>::~Task " << (discarded != nullptr) << " this ptr: " << this);

            if(discarded != nullptr) {
                *discarded = true;
            }

            if(handle && !handle.promise().detached) {
                auto a = handle.address();
                handle.destroy();
            }
        }
    }   
};

template <>
struct [[nodiscard]] Task<void> {
    std::source_location loc;
    struct promise_type;
    std::coroutine_handle<promise_type> handle;
    bool* discarded = nullptr;

    Task(std::coroutine_handle<promise_type> handle, bool* discarded, std::source_location loc = std::source_location::current()) 
        : handle(handle), discarded(discarded), loc(loc) {
        tdebug("Task<void>::Task(handle, discarded, loc)");
    }

    Task() : handle(nullptr) {
        tdebug("Task<void>::Task()");
    }

    struct final_suspend_aw {
        promise_type* promise;
        std::source_location loc;
        std::coroutine_handle<> handle;

        final_suspend_aw(promise_type* promise) : promise(promise), loc(promise->loc), handle(promise->handle) {
            tdebug("final_suspend_aw::final_suspend_aw");
        }

        bool await_ready() noexcept {
            tdebug("discarded: " << promise->discarded << " has_continuation: " << promise->continuation.has_value());
            return promise->discarded;
        }

        void await_suspend(std::coroutine_handle<> __ub_if_resumed) noexcept {
            tdebug("final_suspend_aw::await_suspend");

            if (promise->continuation.has_value()) {
                tdebug("final_suspend_aw::await_suspend, resuming continuation");
                promise->continuation.value().resume();
            }
        }

        void await_resume() noexcept {
            tdebug("final_suspend_aw::await_resume");
        }

        ~final_suspend_aw() {
            tdebug("final_suspend_aw::~final_suspend_aw");
        }
    };

    struct promise_type {
        bool discarded = false;
        bool detached = false;
        std::optional<std::coroutine_handle<>> continuation;
        std::coroutine_handle<promise_type> handle;
        std::source_location loc;

        promise_type(std::source_location loc = std::source_location::current())
            : loc(loc), handle(std::coroutine_handle<promise_type>::from_promise(*this)) {}

        Task get_return_object() {
            this->continuation = std::nullopt;
            this->discarded = false;
            return Task(handle, &discarded, loc);
        }

        std::suspend_never initial_suspend() { return {}; }

        final_suspend_aw final_suspend() noexcept {
            tdebug("promise_type::final_suspend");
            return final_suspend_aw{this};
        }

        void unhandled_exception() {

            const char * err = nullptr;
            try {
                std::rethrow_exception(std::current_exception());
            } catch (const UnwrapException& e) {
                err = e.what();
            } catch (...) {
                err = "Some exception occured";
            }

            tdebug("Unhandled exception inside task, that should never fail");
            std::cerr << "Unhandled exception inside task, that should never fail" << std::endl;
            std::cerr << err << std::endl;
            std::terminate();
        }

        void return_void() {
            tdebug("Task<void>::promise_type::return_void");
        }

        ~promise_type() {
            tdebug("Task<void>::promise_type::~promise_type");
        }
    };

    bool await_ready() {
        if (!handle) {
            std::cerr << "Task<void>::await_ready() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

        return handle.done();
    }

    void await_suspend(std::coroutine_handle<> h) {
        if (!handle) {
            std::cerr << "Task<void>::await_suspend() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

        tdebug("Task<void>::await_suspend");
        if (handle.promise().continuation.has_value()) {
            throw std::runtime_error("Awaiting Task<void> in invalid state, this should never happen");
        }

        handle.promise().continuation = h;
    }

    void await_resume() {
        tdebug("Task<void>::await_resume");

        if (!handle) {
            std::cerr << "Task<void>::await_resume() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

        if (!handle.done()) {
            throw std::runtime_error("Resuming Task<void> in invalid state, this should never happen");
        }
    }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    Task& operator=(Task&& other) {
        if (this == &other)
            return *this;

        loc = other.loc;
        handle = std::exchange(other.handle, {});
        discarded = std::exchange(other.discarded, nullptr);

        return *this;
    }

    Task(Task&& other) {
        handle = std::exchange(other.handle, {});
        discarded = std::exchange(other.discarded, nullptr);
        loc = other.loc;
    }

    ~Task() {
        if (handle) {
            tdebug("Task<void>::~Task " << (discarded != nullptr) << " this ptr: " << this);

            if (discarded != nullptr) {
                *discarded = true;
            }

            if (handle && !handle.promise().detached) {
                std::cout << "Destroying handle" << std::endl;


                handle.destroy();
            } else {
                std::cout << "Detached handle" << std::endl;
            }
        }
    }
};


// template <>
// struct [[nodiscard]] Task<void> {
//     struct promise_type;
//     std::coroutine_handle<promise_type> handle;

//     bool* discarded = nullptr;

//     Task(std::coroutine_handle<promise_type> handle, bool* discarded) : handle(handle), discarded(discarded) {}


//     struct final_suspend_aw {
//         promise_type& promise;
//         final_suspend_aw(promise_type& promise) : promise(promise) {}

//         bool await_ready() noexcept {
//             return promise.continuation.has_value() || promise.discarded;
//         }

//         void await_suspend(std::coroutine_handle<> h) noexcept {
//         }

//         void await_resume()  noexcept {
//             if (promise.continuation.has_value()) {
//                 promise.continuation.value().resume();
//             }
//         }
//     };

//     struct promise_type {
//         bool done = false;
//         bool discarded = false;
//         bool detached = false;
//         std::optional<std::coroutine_handle<>> continuation;

//         Task get_return_object() {
//             this->discarded = false;    
//             this->done = false;
//             this->continuation = std::nullopt;
//             return Task{std::coroutine_handle<promise_type>::from_promise(*this), &discarded};
//         }


//         std::suspend_never initial_suspend() { return {}; }

//         final_suspend_aw final_suspend() noexcept {
//             return final_suspend_aw{*this};
//         }

//         void unhandled_exception() {};

        
//         void return_void() {
//             done = true;
//         }
//     };

//     bool await_ready() {
//         return handle.promise().done;
//     }

//     void await_suspend(std::coroutine_handle<> h) {
//         if (handle.promise().continuation.has_value()) {
//             throw std::runtime_error("Awaiting task in invalid state, this should never happen");
//         }

//         handle.promise().continuation = h;
//     }

//     void await_resume() {
//         if (!handle.promise().done) {
//             throw std::runtime_error("Resuming task in invalid state, this should never happen");
//         }

//         return;
//     }

//     Task(const Task&) = delete;
//     Task& operator=(const Task&) = delete;

//     ~Task() {
//         *discarded = true;

//         if (!handle.promise().detached) {
//             handle.destroy();
//         }

//     }
// };
