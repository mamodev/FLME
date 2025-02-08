#include <variant>
#include <coroutine>
#include <functional>
#include <type_traits>



class go {
    public:
        struct promise;
        using promise_type=promise;

        std::coroutine_handle<go::promise> handle;

        go(std::coroutine_handle<go::promise> h) {
            this->handle = h;
        }

        struct promise {

            go get_return_object() {
                auto c = go(
                    std::coroutine_handle<go::promise>::from_promise(*this)
                );

                return c;
            }

            void unhandled_exception() noexcept { }
            void return_void() noexcept { }

            std::suspend_never initial_suspend() noexcept {
                return {};
            }

            std::suspend_always final_suspend() noexcept { 
                return {};
            }
        };
};


using Error = const char*;
Error ERR_GENERIC = "Something went wrong";
Error ERR_NEW_FAILED = "new operator failed";

template<typename T>
struct Res {
    using StoredType = std::conditional_t<
        std::is_reference_v<T>,
        std::reference_wrapper<std::remove_reference_t<T>>,
        T
    >;

    std::variant<std::monostate, StoredType, Error> data;


    Res(T val)
    {
        if constexpr (std::is_reference_v<T>) {
            data = std::ref(val);
        } else {
            data = val;
        }
    }

    template<typename E, typename = std::enable_if_t<std::is_same_v<E, Error>>>
    Res(E err)
    {
        data = err;
    }


    inline T getValue() {
        if constexpr (std::is_reference_v<T>) {
            return std::get<std::reference_wrapper<std::remove_reference_t<T>>>(data).get();
        } else {
            return std::get<T>(data);
        }
    }

    inline Error getError() {
        return std::get<Error>(data);
    }

    inline bool is_ok() {
        return std::holds_alternative<StoredType>(data);
    }

};


template<typename T>
struct PromiseHandle {
    std::coroutine_handle<> handle;
    bool resolved = false;
    Res<T> value;
};

template<typename T>
class PromiseImpl {
public:
    PromiseHandle<T>* handle;
    PromiseImpl(PromiseHandle<T>* handle) {
        this->handle = handle;
    }

    bool is_resolved() {
        return this->handle->resolved;
    }

    Res<T> get() {
        return this->handle->value;
    }

   auto operator co_await() noexcept {
        struct Awaiter {
            Promise<T>* promise;

            bool await_ready() noexcept {
                return promise->is_resolved();
            }

            Res<T> await_resume() noexcept {
                return promise->get();
            }

            void await_suspend(std::coroutine_handle<> handle) {
                promise->handle->handle = handle;
            }
        };

        return Awaiter{ this };
    }
};


template<typename T>
struct Promise
{
    std::variant<PromiseImpl<T>, Res<T>> __inner
};


go prova (Promise<int> p) {

}


int main() {
    return 0;
}