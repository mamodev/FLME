#pragma once

#include <optional>
#include <coroutine>
#include <exception>
#include <stdexcept>    

#include <iostream>

#include <utility>
#include <vector>

#include "../utils.hpp"
#include "shared.hpp"
#include "defer.hpp"
#include "../results.hpp" 

#ifdef DEBUG
#include <source_location>

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
#else
#define tdebug(stmnt) do {} while (0)
#endif


// HELPER FUNCTIONS
template <typename T, template <typename> class Template>
struct is_specialization_of : std::false_type {};

template <typename U, template <typename> class Template>
struct is_specialization_of<Template<U>, Template> : std::true_type {};

template <typename T, template <typename> class Template>
inline constexpr bool is_specialization_of_v = is_specialization_of<T, Template>::value;
// HELPER FUNCTIONS


template <typename T>
struct Task;

struct TaskFinalSuspend {
    bool discarded;
    std::optional<std::coroutine_handle<>> continuation;

    TaskFinalSuspend() : discarded(false), continuation(std::nullopt) {}

    #ifdef DEBUG
    std::source_location loc;
    std::coroutine_handle<> handle;
    TaskFinalSuspend(std::source_location loc, std::coroutine_handle<> handle) : discarded(false), continuation(std::nullopt), loc(loc), handle(handle) {
        tdebug("constructor");
    }
    #endif
 

    bool await_ready() noexcept {
        tdebug("discarded: " << discarded << " has_continuation: " << continuation.has_value());
        return discarded;
    }

    void await_suspend(std::coroutine_handle<> __ub_if_resumed) noexcept {
        tdebug("suspend, has_continuation: " << continuation.has_value());

        if (continuation.has_value()) {
            continuation.value().resume();

        }
    }

    void await_resume()  noexcept {
        tdebug("resuming, and auto destroying handle");
    }
};

template <typename T = void>
struct Task {
    struct promise_type;

    std::coroutine_handle<promise_type> handle;
    
    Task() : handle(nullptr) { tdebug("Task<T>::Task()"); }

    #ifdef DEBUG
    std::source_location loc;
    Task(std::coroutine_handle<promise_type> handle, std::source_location loc) : handle(handle), loc(loc) {
        tdebug("Task<T>::Task(handle, discarded, loc)");
    }
    #else
    Task(std::coroutine_handle<promise_type> handle) : handle(handle) {}
    #endif

    struct promise_type {
        ENABLE_GET_CURRENT_HANDLE

        TaskFinalSuspend fs;
        std::optional<T> value;

        #ifdef DEBUG
        std::coroutine_handle<promise_type> handle;
        std::source_location loc;

        promise_type(std::source_location loc = std::source_location::current()) {
            this->loc = loc;
            this->handle = std::coroutine_handle<promise_type>::from_promise(*this);
            this->fs = TaskFinalSuspend(loc, this->handle);
        }

        Task get_return_object() {
            return Task(std::coroutine_handle<promise_type>::from_promise(*this), loc);
        }

        #else
        promise_type() : fs() {}   

        Task get_return_object() {
            return Task(std::coroutine_handle<promise_type>::from_promise(*this));
        }    
        #endif

        std::suspend_never initial_suspend() { return {}; }

        TaskFinalSuspend final_suspend() noexcept {
            tdebug("promise_type::final_suspend");
            return fs;
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
    };

    bool await_ready() {
        if(!handle) {
            std::cerr << "Task<T>::await_ready() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

        tdebug("has_value: " << handle.promise().value.has_value());
        return handle.promise().value.has_value();
    }

    void await_suspend(std::coroutine_handle<> h) {
        if(!handle) {
            std::cerr << "Task<T>::await_suspend() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

        tdebug("task<T>::await_suspend");
        if (handle.promise().fs.continuation.has_value()) {
            throw std::runtime_error("Awaiting task in invalid state, this should never happen");
        }

        handle.promise().fs.continuation = h;
    }

    T await_resume() {
        tdebug("task<T>::await_resume");
        if(!handle) {
            std::cerr << "Task<T>::await_resume() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

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

        
        #ifdef DEBUG
        loc = other.loc;
        #endif
        handle = std::exchange(other.handle, {});

        return *this;
    }

    Task(Task&& other) {
        #ifdef DEBUG
        loc = other.loc;
        #endif
        handle = std::exchange(other.handle, {});
    }

    ~Task() {
        tdebug("task<T>::~Task ");
        if(!handle) {
            return;
        }

        // this should not destroy the handle if promise not completed yet
        if (!handle.promise().value.has_value()) {
            handle.promise().fs.discarded = true;
        } else {
            handle.destroy();
        }
    }   
};

#define co_return_void co_return std::nullopt;

template <>
struct Task<void> {
    struct promise_type;
    std::coroutine_handle<promise_type> handle;

    Task() : handle(nullptr) {tdebug("Task<void>::Task()");}

    #ifdef DEBUG
    std::source_location loc;
    Task(std::coroutine_handle<promise_type> handle, std::source_location loc) : handle(handle), loc(loc) {
        tdebug("Task<void>::Task(handle, loc)");
    }
    #else 
    Task(std::coroutine_handle<promise_type> handle) : handle(handle) {}
    #endif

    struct promise_type {
        bool done;  //this replace the Task<T> optional value
        TaskFinalSuspend fs;

        #ifdef DEBUG
        std::coroutine_handle<promise_type> handle;
        std::source_location loc;
        promise_type(std::source_location loc = std::source_location::current()) {
            this->loc = loc;
            this->handle = std::coroutine_handle<promise_type>::from_promise(*this);
            this->fs = TaskFinalSuspend(loc, this->handle);
        }

        Task get_return_object() {
            return Task(std::coroutine_handle<promise_type>::from_promise(*this), loc);
        }

        #else
        promise_type() : fs(), done(false) {}
        Task get_return_object() {
            return Task(std::coroutine_handle<promise_type>::from_promise(*this));
        }
        #endif

  
        std::suspend_never initial_suspend() { return {}; }

        TaskFinalSuspend final_suspend() noexcept {
            tdebug("promise_type::final_suspend");
            return fs;
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
            done = true;
            tdebug("Task<void>::promise_type::return_void");
        }
    };

    bool await_ready() {
        if (!handle) {
            std::cerr << "Task<void>::await_ready() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

        return handle.promise().done;
    }

    void await_suspend(std::coroutine_handle<> h) {
        if (!handle) {
            std::cerr << "Task<void>::await_suspend() Awaiting invalid task, this should never happen" << std::endl;
            exit(1);
        }

        tdebug("Task<void>::await_suspend");
        if (handle.promise().fs.continuation.has_value()) {
            throw std::runtime_error("Awaiting Task<void> in invalid state, this should never happen");
        }

        handle.promise().fs.continuation = h;
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


        #ifdef DEBUG
        loc = other.loc;
        #endif
        handle = std::exchange(other.handle, {});

        return *this;
    }

    Task(Task&& other) {
        #ifdef DEBUG
        loc = other.loc;
        #endif
        handle = std::exchange(other.handle, {});
    }

    ~Task() {
        tdebug("Task<void>::~Task");
        if (!handle) {
           return;
        }

        if (!handle.promise().done) {
            handle.promise().fs.discarded = true;
        } else {
            handle.destroy();
        }
    }
};
