#pragma once

#include <iostream>
#include <coroutine>
#include "shared.hpp"

#ifdef DEBUG
#define __db_print(x) std::cout << "[" << this->ID << "] " << x << std::endl;
#else
#define __db_print(x)
#endif

int DEFER_ID = 0;
struct c_defer {
    struct promise_type;

    struct defer_manager {
        promise_type *head;
        promise_type *tail;
        std::coroutine_handle<> h;

        defer_manager() : head(nullptr), tail(nullptr), h(nullptr) {}

        void print_queue() {
            std::cout << "\tdefer queue: ";
            auto p = head;
            while (p) {
                std::cout << p->ID << " ";
                p = p->next;
            }
            std::cout << std::endl;
        }
    
        void add(promise_type *d) {
            std::cout << "defer_manager::add(" << d->ID << ")" << std::endl;
            if (!head) {
                std::cout << "\tdefer_manager::add: head is nullptr" << std::endl;
                head = d;
                tail = head;
            } else {
                std::cout << "\tdefer_manager::add: head is not nullptr, adding after " << tail->ID << std::endl;
                tail->next = d;
                d->prev = tail;
                tail = d;
            }

            print_queue();
        }

        void remove(promise_type &p) {
            std::cout << "defer_manager::remove(" << p.ID << ")" << std::endl;
            if (head == &p) {
                head = p.next;
            }

            if (tail == &p) {
                tail = p.prev;
            }

            if (p.prev) {
                p.prev->next = p.next;
            }

            if (p.next) {
                p.next->prev = p.prev;
            }

            p.next = nullptr;
            p.prev = nullptr;
            print_queue();
        }
    };

        
    std::coroutine_handle<promise_type> h;
    int ID;
    
    struct promise_type {
        ENABLE_GET_CURRENT_HANDLE
        
        int ID;
        bool blocked_once = false;
        bool added = false;
        
        defer_manager *dm;
        promise_type *next = nullptr;
        promise_type *prev = nullptr;

        c_defer get_return_object() {
            this->ID = DEFER_ID++;
            __db_print("promise_type::get_return_object");
            return c_defer{.h = std::coroutine_handle<promise_type>::from_promise(*this), .ID = ID};
        }

        std::suspend_always initial_suspend() { return {}; }

        template<typename AW>
        auto await_transform(AW&& awaitable) {
            if(!blocked_once && !added) {
                dm->add(std::addressof(*this));
                added = true;
            }

            blocked_once = true;
            return std::forward<AW>(awaitable);
        }

        auto final_suspend() noexcept {
            __db_print("promise_type::final_suspend");
            struct final_awaitable {
                int ID;
                promise_type &p;

                bool await_ready() noexcept {
                    if (p.dm->head == nullptr) {
                        return !p.added;
                    }

                    return false;
                }
                
                void await_suspend(std::coroutine_handle<> h) noexcept {
                    
                    if (p.dm->head == nullptr && p.dm->h) {
                        p.dm->h.resume();
                    } else {
                        std::coroutine_handle<promise_type>::from_promise(*p.dm->head).resume();
                    }

                    h.destroy();
                }

                void await_resume() noexcept {
                    __db_print("final_awaitable::await_resume, next: " << p.next << ", prev: " << p.prev << " h: " << (bool) p.dm->h);
                }
            };

            if(added) {
                dm->remove(*this);
            }

            return final_awaitable{.ID = ID, .p = *this};
        }

        void return_void() {
            __db_print("promise_type::return_void");
        }

        void unhandled_exception() {
            __db_print("promise_type::unhandled_exception");
        }
    };

    ~c_defer() {
        __db_print("c_defer::~c_defer");
        if(!h && !h.promise().dm) {
            std::cout << "c_defer::~c_defer: destroying c_defer in invalid state" << std::endl;
            exit(1);
        }


        bool should_resume = h.promise().dm->head == nullptr;
        if (should_resume) {
            __db_print("c_defer::resume");
            h.resume();
            __db_print("c_defer::resume done");
        } else {
            //todo maby i should set to blocked_once = true here
            h.promise().added = true;
            h.promise().dm->add(std::addressof(h.promise()));
        }
    }
};

// create a type hepler to check if an hanlde has a promise_type member 
// which contain a member called fs wich has a member of type defer_manager
template <typename T>
concept has_defer_manager = requires(T t) {
    typename T::promise_type::defer_manager;
};