#pragma once

#include <variant>
#include <optional>
#include <string>
#include <cstdint>

#define ERR_OUT_OF_MEMORY Error("Out of memory")
#define ERR_UNINITIALIZED Error("Uninitialized")
#define ERR_UNKNOWN Error("Unknown error")

struct Error {
    std::string message;
    // constexpr Error() : message("Unknown error") {}
    constexpr Error(const char* msg) : message(msg) {}
    Error(const std::string& msg) : message(msg) {}
};


template <typename T = void>
struct Res {
    std::variant<T, Error> value;

    // Res() : value(ERR_UNINITIALIZED) {}
    Res(T value) : value(value) {}
    Res(Error value) : value(value) {}

    bool is_ok() {
        return std::holds_alternative<T>(value);
    }

    T getValue() {
        return std::get<T>(value);
    }

    // const T& getValue() const {
    //     return std::get<T>(value);
    // }

    Error getError() {
        return std::get<Error>(value);
    }
};


template<>
struct Res<void> {
    std::optional<Error> value;

    Res() = default;
    Res(Error error) : value(std::move(error)) {}
    Res(std::nullopt_t) : value(std::nullopt) {}

    bool is_ok() {
        return !value.has_value();
    }

    Error getError() {
        return value.value();
    }

    void getValue() {}
};

class UnwrapException : public std::exception {
    Error error;
public:
    UnwrapException(Error error) : error(error) {}
    const char* what() const noexcept override {
        return error.message.c_str();
    }
};

template<typename T>
T unwrap_or_excep(Res<T> res) {
    if (!res.is_ok()) {
        throw UnwrapException(res.getError());
    }

    if constexpr (std::is_same_v<T, void>) {
        return;
    } else {
        return res.getValue();
    }
}

#define propagate(res) ({ \
    auto _r = res; \
    if (!_r.is_ok()) { \
        return _r.getError(); \
    } \
    _r.getValue(); \
})


// WARNING: propagate in a coroutine will not trigger defer_error statements,
// if you need this behavioe use unwrap_or_excep
#define co_propagate(res) ({ \
    auto _r = res; \
    if (!_r.is_ok()) { \
        co_return _r.getError(); \
    } \
    _r.getValue(); \
})


#define try_await(x) unwrap_or_excep(co_await x)


// Res<uint8_t *> try_alloc(size_t size) {
//     auto ptr = new (std::nothrow) uint8_t[size];
//     if (ptr == nullptr) {
//         return ERR_OUT_OF_MEMORY;
//     }
// }




template<typename T = uint8_t>
static inline Res<T*> try_alloc(size_t size = 1) {
    auto ptr = new (std::nothrow) T[size];
    if (ptr == nullptr) {
        return ERR_OUT_OF_MEMORY;
    }

    return ptr;
}