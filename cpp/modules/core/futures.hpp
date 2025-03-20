#pragma once

#include <coroutine>
#include <variant>
#include <optional>

#include <functional>

#include <iostream>

#include "utils.hpp"

template <typename T, typename = void>
struct has_detached : std::false_type {};

template <typename T>
struct has_detached<T, std::void_t<decltype(std::declval<T>().detached)>> : std::true_type {};

template <typename T>
constexpr bool has_detached_v = has_detached<T>::value;

template <typename T>
struct FutureHandler {
private:
    std::optional<T> value;
    std::optional<std::coroutine_handle<>> handle;

    std::function<void(void*)> deallocator;
    void* deallocator_data;

public:
    FutureHandler() : value(std::nullopt), handle(std::nullopt), deallocator(nullptr), deallocator_data(nullptr) {}

    void set_deallocator(std::function<void(void*)> deallocator, void* data) {
        this->deallocator = deallocator;
        this->deallocator_data = data;
    }

    bool is_ready() {
        return value.has_value();
    }

    void set(T value) {
        this->value = value;
        if (handle.has_value()) {
            auto h = handle.value();
            if(!h) {
                throw std::runtime_error("Setting future value in invalid state, this should never happen, the handle does not rapresent a valid coroutine");
            }

            if(h.done()) {
                throw std::runtime_error("Setting future value in invalid state, this should never happen, the handle is already done");
            }


            h.resume();
        }
    }

    void set(std::coroutine_handle<> handle) {
        if (value.has_value()) {
            handle.resume();
        } else {
            this->handle = handle;
        }
    }

    T get() {
        if (!value.has_value()) {
            throw std::runtime_error("Getting future value in invalid state, this should never happen");
        }

        T v = value.value();
        value = std::nullopt;

        if (deallocator != nullptr) {
            deallocator(deallocator_data);
        }

        return v;
    }
};


template <typename T>
struct Future {
    std::variant<
        T,
        FutureHandler<T>*
    > value;

    template <typename U>
    Future(U&& value) : value(std::forward<U>(value)) {
    }
    
    bool is_ready() {
        // std::cout << "Future::is_ready" << std::endl;
        if (std::holds_alternative<T>(value)) { // if is not a ptr the value is resolved
            return true;
        } else if (std::holds_alternative<FutureHandler<T>*>(value)) {
            return std::get<FutureHandler<T>*>(value)->is_ready();
        } else {
            throw std::runtime_error("Future::is_ready, variant is in invalid state, this should never happen");
        }
    }

    std::string getType() {
        if (std::holds_alternative<T>(value)) {
            return "T";
        } else if (std::holds_alternative<FutureHandler<T>*>(value)) {
            return "FutureHandler<T>*";
        } else {
            return "Unknown";
        }
    }

    T get() {
        // std::cout << "Future::get" << std::endl;
        if (std::holds_alternative<T>(value)) {
            return std::get<T>(value);
        } else if (std::holds_alternative<FutureHandler<T>*>(value)) {
            return std::get<FutureHandler<T>*>(value)->get();
        } else {
            throw std::runtime_error("Getting future value in invalid state, this should never happen");
        }
    }

    bool await_ready() {
        // std::cout << "Future::await_ready id: " << name << std::endl;
        return is_ready();
    }


    template <typename PromiseType>
    void await_suspend(std::coroutine_handle<PromiseType> handle) {
        // std::cout << "Future::await_suspend id: " << name << std::endl;

        if (std::holds_alternative<T>(value)) {
            handle.resume();
        } else if (std::holds_alternative<FutureHandler<T>*>(value)) {
            
            if constexpr (has_detached_v<PromiseType>) {  
                // debug("Future::await_suspend");
                auto handleAddress = handle.address();
                auto typedHandle = std::coroutine_handle<PromiseType>::from_address(handleAddress);
                typedHandle.promise().detached = true;  // Mark ownership transfer
            }

            std::get<FutureHandler<T>*>(value)->set(handle);

        } else {
            throw std::runtime_error("Awaiting future in invalid state, this should never happen");
        }


    }

    T await_resume() {
        // std::cout << "Future::await_resume: " << name <<  " type: " << getType() << std::endl;
        if (!is_ready()) {
            throw std::runtime_error("Resuming from future in invalid state, this should never happen, T: " + getType());
        }
        return get();
    }
};