#pragma once

#include <string_view>
#include <array>
#include <algorithm>

#include "_ctre.hpp"

enum class fn_t {
    Fiber,
    VoidTask,
    ErrTask,
    Task,
    NormalFunction,
};

constexpr auto decode_fn_t (fn_t t) {
    switch (t) {
        case fn_t::Fiber:
            return "Fiber";
        case fn_t::VoidTask:
            return "VoidTask";
        case fn_t::ErrTask:
            return "ErrTask";
        case fn_t::Task:
            return "Task";
        case fn_t::NormalFunction:
            return "NormalFunction";
    }
}
template <size_t N>
struct StringLiteral {
    consteval StringLiteral(const char (&str)[N]) {
        std::copy_n(str, N, value.begin());
    }
    std::array<char, N> value;
};

template <StringLiteral str>
consteval fn_t get_fn_t() {
    if constexpr (ctre::match<R"(^\s*Fiber\s+\w+\s*\(.*\)\s*)">(std::string_view{str.value.data()})) {
        return fn_t::Fiber;
    } else if constexpr (ctre::match<R"(^\s*Task\s*(?:<\s*(?:void)?\s*>)?\s+\w+\s*\(.*\)\s*)">(std::string_view{str.value.data()})) {
        return fn_t::VoidTask;
    } else if constexpr (ctre::match<R"(^\s*(?:ErrTask|Task\s*<\s*Res<void>\s*>)\s+\w+\s*\(.*\)\s*)">(std::string_view{str.value.data()})) {
        return fn_t::ErrTask;
    } else if constexpr (ctre::match<R"(^\s*Task\s*<.*>\s+\w+\s*\(.*\)\s*)">(std::string_view{str.value.data()})) {
        return fn_t::Task;
    } else {
        return fn_t::NormalFunction;
    }
}

