#pragma once

#include <coroutine>
#include <cstdint>
#include <exception>

#include <iostream>
#include "../utils.hpp"

extern thread_local uint64_t ACTIVE_FIBERS;

struct Fiber {
    struct promise_type {
        Fiber get_return_object() {
            ACTIVE_FIBERS++;
            debug("Fiber::get_return_object active fibers: " << ACTIVE_FIBERS);
            return Fiber{};
        }

        std::suspend_never initial_suspend() {
            // std::cout << "Fiber initial_suspend" << std::endl;
            return {}; }

        std::suspend_never final_suspend() noexcept {
            // std::cout << "Fiber final_suspend" << std::endl
            ACTIVE_FIBERS--;
            debug("Fiber::final_suspend active fibers: " << ACTIVE_FIBERS);
            return {}; }

        void return_void() {
            // std::cout << "Fiber return_void" << std::endl;
        }

        void unhandled_exception() {
            try{
                std::rethrow_exception(std::current_exception());
            } catch (const std::exception& e) {
                std::cerr << "Unhandled exception inside fiber: " << e.what() << std::endl;
            }

        }
    };
};