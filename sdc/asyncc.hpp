#pragma once

#include <iostream>
#include <liburing.h>
#include <coroutine>
#include <variant>
#include <optional>
#include <functional>
#include <cassert>

// 3 type of functions:
// 1. fn that return a value or an error code: Result<T, Error>
// 2. fn that return void or an error: Optional<Error>
// 3. fn that return value no error code: T

// All of this can be a future result, a Future<T> wich can be co_awaited to get T
// The complication of this is that we don't want to add overhead where it's not needed
// So if a Future could be resolved synchronously, we don't want to allocate memory for it

// To solve this we Always return Promise<T> as copy, but inside the Promise has an union of T and a Ptr to T
// Or better a Ptr to FutureHandler<T> so that the handler can be allocated manually and keeped for async resolution

template <typename T>
struct FutureHandler {
private:
    std::optional<T> value;
    std::optional<std::coroutine_handle<>> handle;

public:
    FutureHandler() : value(std::nullopt), handle(std::nullopt) {}

    bool is_ready() {
        return value.has_value();
    }

    void set(T value) {
        // std::cout << "FutureHandler::set" << std::endl;
        this->value = value;
        if (handle.has_value()) {
            auto h = handle.value();
            // std::cout << "FutureHandler::set resuming" << std::endl;
            h.resume();
            // std::cout << "FutureHandler::set resumed" << std::endl;
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

        return value.value();
    }
};



thread_local uint64_t FutID = 0;
template <typename T>
struct Future {
    std::variant<
        T,
        FutureHandler<T>*
    > value;

    std::string name = "Future-" + std::to_string(FutID++);

    template <typename U>
    Future(U&& value) : value(std::forward<U>(value)) {
        // std::cout << "Future::Future ptr? " << std::is_pointer_v<U> << std::endl;
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

    void await_suspend(std::coroutine_handle<> handle) {
        // std::cout << "Future::await_suspend id: " << name << std::endl;

        if (std::holds_alternative<T>(value)) {
            handle.resume();
        } else if (std::holds_alternative<FutureHandler<T>*>(value)) {
            std::get<FutureHandler<T>*>(value)->set(handle);
        } else {
            throw std::runtime_error("Awaiting future in invalid state, this should never happen");
        }


    }

    T await_resume() {
        // std::cout << "Future::await_resume: " << name <<  " type: " << getType() << std::endl;
        if (!is_ready()) {
            throw std::runtime_error("Resuming from future " + name + " in invalid state, this should never happen, T: " + getType());
        }
        return get();
    }
};

// The Future mechanism is intended to be used with C++20 coroutines
// We mainly have two types of coroutines:
// 1. Root coroutines, this corutines are not awaitable, they just execute until they block on IO or other async operation
//     This are intended to be used as "Green Threads" or "Fibers"
// 2. Awaitable coroutines, this coroutines are intended to be used with the await keyword,
//    they can be awaited on other awaitable coroutines

thread_local uint64_t ACTIVE_FIBERS = 0;

struct Fiber {
    struct promise_type {
        Fiber get_return_object() {
            ACTIVE_FIBERS++;
            return Fiber{};
        }

        std::suspend_never initial_suspend() {
            // std::cout << "Fiber initial_suspend" << std::endl;
            return {}; }

        std::suspend_never final_suspend() noexcept {
            // std::cout << "Fiber final_suspend" << std::endl;
            ACTIVE_FIBERS--;
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


// Now, it is really common that an async function could have many possible errors
// So we need a way to return an error code. There are two common cases:
// 1 a function could return a value or an error code
// 2 a function could return void or an error code
// the second one is resolved with std::optional<Error>
// the first one is resolved with std::variant<T, Error>

//Whe should create some error type


struct Error {
    std::string message;
    constexpr Error(const char* msg) : message(msg) {}
    Error(const std::string& msg) : message(msg) {}
};

template <typename T = void>
struct Res {
    std::variant<T, Error> value;

    Res(T value) : value(value) {}
    Res(Error value) : value(value) {}

    bool is_ok() {
        return std::holds_alternative<T>(value);
    }

    T getValue() {
        return std::get<T>(value);
    }

    Error getError() {
        return std::get<Error>(value);
    }
};

template<>
struct Res<void> {
    std::optional<Error> value;
   
    // strip the reference
    template<typename T>
    Res(T&& value) {
        using U = std::remove_reference_t<T>;
        using V = std::remove_const_t<U>;

        if constexpr (std::is_same_v<V, Error>) {
            this->value = value;
        } else if constexpr (std::is_convertible_v<V, Error>) {
            this->value = Error(value);
        } else if constexpr (std::is_same_v<V, std::nullopt_t>) {
            this->value = std::nullopt;
        } else if constexpr (std::is_same_v<V, Res<void>>) {
            this->value = value.value;
        } else {
            static_assert(std::is_same_v<V, void>, "Invalid type for Res<void>");
        }
    }
    


    bool is_ok() {
        return !value.has_value();
    }

    Error getError() {
        return value.value();
    }
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

#define try_await(x) unwrap_or_excep(co_await x)

// Task is a coroutine that returns a value (or void Task<>), it can be awaited to get the value

template <typename T, template <typename> class Template>
struct is_specialization_of : std::false_type {};

// Specialization: true if T is Template<U> for any U
template <typename U, template <typename> class Template>
struct is_specialization_of<Template<U>, Template> : std::true_type {};

// Helper variable template for convenience
template <typename T, template <typename> class Template>
inline constexpr bool is_specialization_of_v = is_specialization_of<T, Template>::value;


template <typename T = void>
struct [[nodiscard]] Task {
    struct promise_type;
    std::coroutine_handle<promise_type> handle;

    Task(std::coroutine_handle<promise_type> handle) : handle(handle) {}


    struct final_suspend_aw {
        promise_type& promise;
        final_suspend_aw(promise_type& promise) : promise(promise) {}

        bool await_ready() noexcept {
            // std::cout << "final_suspend_aw::await_ready" << std::endl;
            return promise.continuation.has_value();
        }

        void await_suspend(std::coroutine_handle<> h) noexcept {
            // std::cout << "final_suspend_aw::await_suspend" << std::endl;
        }

        void await_resume()  noexcept {
            // std::cout << "final_suspend_aw::await_resume" << std::endl;
            if (promise.continuation.has_value()) {
                promise.continuation.value().resume();
            }
        }
    };

    struct promise_type {
        std::optional<T> value;
        std::optional<std::coroutine_handle<>> continuation;
        std::suspend_never initial_suspend() { return {}; }

        final_suspend_aw final_suspend() noexcept {
            // std::cout << "task<T>::promise_type::final_suspend" << std::endl;
            return final_suspend_aw{*this};
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
                std::cerr << "Unhandled exception inside task, that should never fail" << std::endl;
                std::cerr << err << std::endl;
                std::terminate();
            }
        }

        Task get_return_object() {
            this->value = std::nullopt;
            this->continuation = std::nullopt;
            // std::cout << "task<T>::promise_type::get_return_object" << std::endl;
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        void return_value(T value) {
            // std::cout << "task<T>::promise_type::return_value" << std::endl;
            this->value = value;
        }
    };

    bool await_ready() {
        // std::cout << "task<T>::await_ready" << std::endl;
        auto &p = handle.promise();
        auto &v = p.value;
        auto has_value = v.has_value();
        return has_value;
    }

    void await_suspend(std::coroutine_handle<> h) {
        // std::cout << "task<T>::await_suspend" << std::endl;
        if (handle.promise().continuation.has_value()) {
            throw std::runtime_error("Awaiting task in invalid state, this should never happen");
        }

        handle.promise().continuation = h;
    }

    T await_resume() {
        // std::cout << "task<T>::await_resume" << std::endl;
        if (!handle.promise().value.has_value()) {
            throw std::runtime_error("Resuming task in invalid state, this should never happen");
        }

        return handle.promise().value.value();
    }
};

template <>
struct [[nodiscard]] Task<void> {
    struct promise_type;
    std::coroutine_handle<promise_type> handle;
    Task(std::coroutine_handle<promise_type> handle) : handle(handle) {
    }

    struct final_suspend_aw {
        promise_type& promise;
        final_suspend_aw(promise_type& promise) : promise(promise) {}

        bool await_ready() noexcept {
            return promise.continuation.has_value();
        }

        void await_suspend(std::coroutine_handle<> h) noexcept {

        }

        void await_resume()  noexcept {
            if (promise.continuation.has_value()) {
                promise.continuation.value().resume();
            }
        }
    };

    struct promise_type {
        bool done = false;
        std::optional<std::coroutine_handle<>> continuation;
        std::suspend_never initial_suspend() { return {}; }

        final_suspend_aw final_suspend() noexcept {
            return final_suspend_aw{*this};
        }

        void unhandled_exception() {};

        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        void return_void() {
            done = true;
        }
    };

    bool await_ready() {
        return handle.promise().done;
    }

    void await_suspend(std::coroutine_handle<> h) {
        if (handle.promise().continuation.has_value()) {
            throw std::runtime_error("Awaiting task in invalid state, this should never happen");
        }

        handle.promise().continuation = h;
    }

    void await_resume() {
        if (!handle.promise().done) {
            throw std::runtime_error("Resuming task in invalid state, this should never happen");
        }

        return;
    }
};




// Now that we have some basics we can start implementing event loop
// General blocking cases:
// 1. IO operations
// 2. Sleep
// 3. yeleding to other fibers / tasks
enum class IORequestType {
    OPEN,
    READ,
    WRITE,
    CLOSE,
    SOCK_OPEN,
    SOCK_SEND,
    SOCK_RECV,
    SOCK_CONNECT,
    SOCK_ACCEPT,
    SOCK_BIND,
    SOCK_LISTEN,
    SOCK_SHUTDOWN,
    TIMEOUT,
    // ...
};

constexpr const char* encodeIORequestType(IORequestType type) {
    switch (type) {
        case IORequestType::OPEN: return "OPEN";
        case IORequestType::READ: return "READ";
        case IORequestType::WRITE: return "WRITE";
        case IORequestType::CLOSE: return "CLOSE";
        case IORequestType::SOCK_OPEN: return "SOCK_OPEN";
        case IORequestType::SOCK_SEND: return "SOCK_SEND";
        case IORequestType::SOCK_RECV: return "SOCK_RECV";
        case IORequestType::SOCK_CONNECT: return "SOCK_CONNECT";
        case IORequestType::SOCK_ACCEPT: return "SOCK_ACCEPT";
        case IORequestType::SOCK_BIND: return "SOCK_BIND";
        case IORequestType::SOCK_LISTEN: return "SOCK_LISTEN";
        case IORequestType::SOCK_SHUTDOWN: return "SOCK_SHUTDOWN";
        case IORequestType::TIMEOUT: return "TIMEOUT";
        default: return "UNKNOWN";
    }
}


struct io_uring_buf_ring *setup_buffer_ring(struct io_uring *ring, uint16_t BGID, uint32_t nbufs, unsigned int buf_size, uint8_t *bufs) {
	struct io_uring_buf_reg reg = { 0 };
	struct io_uring_buf_ring *br;
	int i;

    long PAGE_SIZE = sysconf(_SC_PAGESIZE);
	if (posix_memalign((void **) &br, PAGE_SIZE,
			   nbufs * sizeof(struct io_uring_buf_ring)))
		return NULL;

        

	/* assign and register buffer ring */
	reg.ring_addr = (uint64_t) br;
	reg.ring_entries = nbufs;
	reg.bgid = BGID;

	if (io_uring_register_buf_ring(ring, &reg, 0) != 0) {
        free(br);
        return NULL;
    }

	io_uring_buf_ring_init(br);
	for (i = 0; i < nbufs; i++) {
        void * buff_ptr = bufs + i * buf_size;

	    io_uring_buf_ring_add(br, buff_ptr, buf_size, i, io_uring_buf_ring_mask(nbufs), i);

	}

	/* we've supplied buffers, make them visible to the kernel */
	io_uring_buf_ring_advance(br, nbufs);
	return br;
}


class EventLoop {
    std::vector<std::coroutine_handle<>> yelded_coroutines;
    struct io_uring ring;

    uint64_t pending_io_requests = 0;

    io_uring_buf_ring *core_ring = nullptr;


    struct IORequest {
        IORequestType type;
        FutureHandler<Res<int>> handler;

        constexpr inline FutureHandler<Res<int>>* get_handler() {
            return &handler;
        }
    };

    const int max_files = 8192;
    const int nr_shared_buffers = 1024;
    const int shared_buffer_size = 4096;
    bool initialized = false;

public:
    volatile bool running = true;

    EventLoop() {}
    Res<void> init(unsigned entries) {
        int ret = io_uring_queue_init(entries, &ring, 0);
        if (ret < 0) {
            return Error("error creating io_uring, ERRNO: " + ret);
        }

        ret = io_uring_register_files_sparse(&ring, max_files);
		if (ret) {
            io_uring_queue_exit(&ring);
            return Error("error registering files, ERRNO: " + ret);
		}   

        ret = io_uring_register_ring_fd(&ring);
        if (ret != 1) {
            io_uring_queue_exit(&ring);
            return Error("error registering ring fd, ERRNO: " + std::to_string(ret));
        }

        core_ring = setup_buffer_ring(&ring, 0, nr_shared_buffers, shared_buffer_size, (uint8_t*)malloc(nr_shared_buffers * shared_buffer_size));
        if (core_ring == nullptr) {
            io_uring_queue_exit(&ring);
            return Error("error creating buffer ring");
        }

        initialized = true;
        return std::nullopt;
    }

    Res<struct io_uring_sqe*> get_sqe() {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        if (sqe == nullptr)
            return Error("error getting sqe, ERRNO: " + errno);

        return sqe;
    }

    Res<IORequest*> new_request(IORequestType type) {
        auto req = new (std::nothrow) IORequest();
        if (req == nullptr) {
            return Error("error creating new request, ERRNO: " + errno);
        }

        req->type = type;
        return req;
    }

    void lazy_submit() {
        this->pending_io_requests++;
        io_uring_submit(&ring);
    }

    void stop() {
        running = false;
    }

    void loop() {
        running = true;

        //instead of emptying the queue one element at a time, we can empty the queue until it's empty and
        // then execute the coroutines that are waiting (So we exploit cache locality and reduce size of pending queue)
        std::vector<std::pair<int, IORequest*>> results;
        io_uring_cqe* cqe;

        while(ACTIVE_FIBERS > 0 || pending_io_requests > 0) {
            while(io_uring_peek_cqe(&ring, &cqe) == 0) {
                void* data = io_uring_cqe_get_data(cqe);

                if(data != nullptr) {
                    IORequest* req = (IORequest*)data;
                    results.push_back({cqe->res, req});
                }

                io_uring_cqe_seen(&ring, cqe);
            }

            if (results.size() != 0) {
                for (auto& [res, data] : results) {
                    pending_io_requests--;

                    bool failed = res < 0;
                    if (data->type == IORequestType::TIMEOUT) {
                        failed = res != -ETIME;
                    }

                    // std::cout << "Request resolved with: " << res << std::endl;

                    if(failed) {
                        auto err = Error("IORequest of type " + std::string(encodeIORequestType(data->type)) + " failed with: " + std::to_string(res));
                        data->handler.set(Res<int>(err));
                    } else {
                        data->handler.set(Res<int>(res));
                    }

                }

                results.clear();
            }

            if (ACTIVE_FIBERS == 0 && pending_io_requests == 0) {
                std::cout << "No active fibers and no pending io requests, exiting" << std::endl;
                running = false;
                break;
            }

            int res = io_uring_wait_cqe(&ring, &cqe);
        }

    }

    ~EventLoop() {
        if (initialized) {
            io_uring_unregister_files(&ring);
            io_uring_unregister_ring_fd(&ring);
            io_uring_queue_exit(&ring);
            free(core_ring);
        }
    }
};

thread_local EventLoop loop = EventLoop();

Future<Res<int>> open_file(const char* path, int flags, int mode) {
    auto req = propagate(loop.new_request(IORequestType::OPEN));
    auto sqe = propagate(loop.get_sqe());
    #ifdef IORING_OP_OPEN
    io_uring_prep_open(sqe, path, flags, mode);
    #else
    io_uring_prep_openat(sqe, AT_FDCWD, path, flags, mode);
    #endif
    
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();

}

Future<Res<int>> read_file(int fd, void* buf, size_t count) {
    auto req = propagate(loop.new_request(IORequestType::READ));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_read(sqe, fd, buf, count, 0);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}

Future<Res<int>> write_file(int fd, void* buf, size_t count) {
    auto req = propagate(loop.new_request(IORequestType::WRITE));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_write(sqe, fd, buf, count, 0);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}

Future<Res<int>> close_file(int fd) {
    auto req = propagate(loop.new_request(IORequestType::CLOSE));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_close(sqe, fd);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}

Future<Res<int>> close_file_direct(int fd) {
    auto req = propagate(loop.new_request(IORequestType::CLOSE));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_close_direct(sqe, fd);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}

Future<Res<int>> timeout(__kernel_timespec *ts, unsigned int count, unsigned int flags) {
    auto req = propagate(loop.new_request(IORequestType::TIMEOUT));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_timeout(sqe, ts, count, flags);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}

Future<Res<int>> open_socket(int domain, int type, int protocol, unsigned flags) {
    auto req = propagate(loop.new_request(IORequestType::SOCK_OPEN));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_socket(sqe, domain, type, protocol, flags);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}


Future<Res<int>> connect_socket(int sockfd, struct sockaddr *addr, unsigned addrlen) {
    auto req = propagate(loop.new_request(IORequestType::SOCK_CONNECT));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_connect(sqe, sockfd, addr, addrlen);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}

Future<Res<int>> accept_socket(int sockfd, struct sockaddr *addr, unsigned *addrlen) {
    auto req = propagate(loop.new_request(IORequestType::SOCK_ACCEPT));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_accept(sqe, sockfd, addr, addrlen, 0);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();

    return req->get_handler();
}

Future<Res<int>> accept_socket_direct(int sockfd, struct sockaddr *addr, unsigned *addrlen) {
    auto req = propagate(loop.new_request(IORequestType::SOCK_ACCEPT));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_accept_direct(sqe, sockfd, addr, addrlen, 0, IORING_FILE_INDEX_ALLOC);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();

    return req->get_handler();
}

Future<Res<int>> bind_socket(int sockfd, struct sockaddr *addr, unsigned addrlen) {
    #ifdef IORING_OP_BIND
    auto req = propagate(loop.new_request(IORequestType::SOCK_BIND));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_bind(sqe, sockfd, addr, addrlen);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
    #else
    if(bind(sockfd, addr, addrlen) < 0) {
        return Error("error binding socket, ERRNO: " + errno);
    }    
    return 0;
    #endif
}

Future<Res<int>> listen_socket(int sockfd, int backlog) {
    #ifdef IORING_OP_LISTEN
    auto req = propagate(loop.new_request(IORequestType::SOCK_LISTEN));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_listen(sqe, sockfd, backlog);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
    #else
    if(listen(sockfd, backlog) < 0) {
        return Error("error listening socket, ERRNO: " + errno);
    }

    return 0;
    #endif
}

Future<Res<int>> recv_socket(int sockfd, void *buf, size_t len, int flags) {
    auto req = propagate(loop.new_request(IORequestType::SOCK_RECV));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_recv(sqe, sockfd, buf, len, flags);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}

Future<Res<int>> send_socket(int sockfd, void *buf, size_t len, int flags) {
    auto req = propagate(loop.new_request(IORequestType::SOCK_SEND));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_send(sqe, sockfd, buf, len, flags);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}

// create send and recv direct versions
Future<Res<int>> recv_socket_direct(int sockfd, void *buf, size_t len, int flags) {
    auto req = propagate(loop.new_request(IORequestType::SOCK_RECV));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_recv(sqe, sockfd, buf, len, flags);

    sqe->flags |= IOSQE_FIXED_FILE;

    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}

Future<Res<int>> send_socket_direct(int sockfd, void *buf, size_t len, int flags) {
    auto req = propagate(loop.new_request(IORequestType::SOCK_SEND));
    auto sqe = propagate(loop.get_sqe());
    io_uring_prep_send(sqe, sockfd, buf, len, flags);

    sqe->flags |= IOSQE_FIXED_FILE;

    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();
    return req->get_handler();
}

// Utility functions
Future<Res<int>> waitMS(int ms) {
    __kernel_timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    return timeout(&ts, 1000000, 0);
}

// Task<Res<void>> recv_all(int sockfd, void *buf, size_t len, int flags) {
//     size_t total = 0;
//     while (total < len) {
//         auto res = try_await(recv_socket(sockfd, (char*)buf + total, len - total, flags));
//         total += res;
//     }

//     co_return Res<void>(Error("Not implemented"));
// }

// Task<Res<void>> send_all(int sockfd, void *buf, size_t len, int flags) {
//     size_t total = 0;
//     while (total < len) {
//         auto res = try_await(write_file(sockfd, (char*)buf + total, len - total));
//         total += res;
//     }

    // return Error("Not implemented");
    // co_return Res<void>(std::nullopt);

    // Res<void> res = Res<void>();

    // co_return res;
// }

struct defer_scope {
    bool has_async_error = false;
    bool has_async = false;
    bool deferred = false;

    struct defer_entry {
        std::variant<std::function<void()>, std::function<Task<void>()>> f;
        bool error;
    };

    std::vector<defer_entry> defers;


    defer_scope() {}
    defer_scope(const defer_scope& other) = delete;
    
    template <typename F>
    defer_scope& operator +(F&& f) {
        add_defer(std::forward<F>(f), false);
        return *this;
    }
    
    template <typename F>
    defer_scope& operator - (F&& f) {
        add_defer(std::forward<F>(f), true);
        return *this;
    }

    template <typename F>
    void add_defer(F&& f, bool error) {
         if constexpr (std::is_invocable_r_v<Task<void>, F>) {
            if (error) 
                has_async_error = true;
            else 
                has_async = true;

            defers.push_back({std::function<Task<void>()>(std::forward<F>(f)), error});

        } else if constexpr (std::is_invocable_r_v<void, F>) {
            defers.push_back({std::function<void()>(std::forward<F>(f)), error});
           
        }  else {
            static_assert(false, "Invalid type for defer");
        }
    }

    Task<void> sync_defer() {
        bool errors = std::uncaught_exceptions() > 0;

        for (auto& defer : defers) {
            if (!defer.error || errors) {
                if (std::holds_alternative<std::function<Task<void>()>>(defer.f)) {
                    co_await std::get<std::function<Task<void>()>>(defer.f)();
                } else {
                    std::get<std::function<void()>>(defer.f)();
                }
            }
        }

        deferred = true;

        co_return;
    }

    Fiber async_defer(bool exec_error) {
        std::vector<defer_entry> defers_copy = defers;

        for (auto& defer : defers_copy) {
            if (!defer.error || exec_error) {
                if (std::holds_alternative<std::function<Task<void>()>>(defer.f)) {
                    co_await std::get<std::function<Task<void>()>>(defer.f)();
                } else {
                    std::get<std::function<void()>>(defer.f)();
                }
            }
        }

        co_return;
    }

    ~defer_scope() {
        if (deferred) {
            return;
        }

        bool errors = std::uncaught_exceptions() > 0;

        if (has_async || errors && has_async_error) {
            async_defer(errors);
        } else {
            auto _we_can_ignore_becouse_task_will_never_suspend = sync_defer();
        }

        deferred = true;
    }
};

#define UNIQUE_NAME(PREFIX) \
    _UNIQUE_NAME_CONCAT(PREFIX, __LINE__)

#define _UNIQUE_NAME_CONCAT(a, b) _UNIQUE_NAME_CONCAT2(a, b)
#define _UNIQUE_NAME_CONCAT2(a, b) a##_##b



struct defer_single_scope {
    std::variant<std::function<void()>, std::function<Task<void>()>> f;
    bool error = false;

    template <typename F>
    defer_single_scope(F&& f, bool error) : error(error) {
        if constexpr (std::is_invocable_r_v<Task<void>, F>) {
            this->f = std::function<Task<void>()>(std::forward<F>(f));
        } else if constexpr (std::is_invocable_r_v<void, F>) {
            this->f = std::function<void()>(std::forward<F>(f));
        } else {
            static_assert(false, "Invalid type for defer");
        }
    }

    Fiber async_defer() {
        std::function<Task<void>()> f = std::get<std::function<Task<void>()>>(this->f);
        co_await f();
        co_return;
    }

    ~defer_single_scope() {
        if (error && std::uncaught_exceptions() == 0) {
            return;
        }
        
        if (std::holds_alternative<std::function<Task<void>()>>(f)) {
            async_defer();
        } else {
            std::get<std::function<void()>>(f)();
        }
    }
};

struct dummy_defer_scope {
    template <typename F>
    defer_single_scope operator+ (F&& f) {
        return defer_single_scope(std::forward<F>(f), false);
    }


    template <typename F>
    defer_single_scope operator- (F&& f) {
        return defer_single_scope(std::forward<F>(f), true);
    }
};

dummy_defer_scope __defer__scope__;


#define defer decltype(auto) \
    UNIQUE_NAME(__DEFER__) = __defer__scope__ + [&]() -> void

#define defer_err decltype(auto) \
    UNIQUE_NAME(__DEFER__) = __defer__scope__ - [&]() -> void

#define defer_async \
    decltype(auto) UNIQUE_NAME(__DEFER__) = __defer__scope__ + [&]() -> Task<void>

#define defer_err_async \
    decltype(auto) UNIQUE_NAME(__DEFER__) = __defer__scope__ - [&]() -> Task<void>

#define squential_defer defer_scope __defer__scope__ = defer_scope();

#define await_defer __defer__scope__.sync_defer();

#define co_return_void co_return std::nullopt;

// #define ret_after_defer \
//         if (__defer__scope__.deferred) { \
//             co_return; \
//         } \
//         co_await __defer__scope__.sync_defer(); \
//         co_return;

Task<void> defer_test_1(bool error) {
    squential_defer;

    std::string prefix = error ? "[E] " : "[ ] ";

    defer {
        std::cout << prefix << "Defer 1" << std::endl;
    };

    defer_err {
        std::cout << prefix << "Defer 2 (err)" << std::endl;
    };

    defer_async {
        co_await waitMS(500);
        std::cout << prefix << "Defer 3 (async)" << std::endl;
    };

    defer_err_async {
        co_await waitMS(1000);
        std::cout << prefix << "Defer 4 (async )" << std::endl;
    };

    defer {
        std::cout << prefix << "Defer 5 (sync after async)" << std::endl;
    };

    defer_err {
        std::cout << prefix << "Defer 6 (sync after async DEFER_ERR)" << std::endl;
    };

    if (error) {
        throw std::runtime_error("Error");
    }

    co_return;
};

Task<int> defer_tests () {
    std::cout << "===== Before defer_test_1 : no error" << std::endl;
    co_await defer_test_1(false);
    std::cout << "===== After defer_test_1 : no error" << std::endl;

    std::cout << "===== Before defer_test_1 : error" << std::endl;
    co_await defer_test_1(true);
    std::cout << "===== After defer_test_1 : error" << std::endl;

    co_return 0;
}


#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

Task<Res<int>> server_socket(int port) {
    int fd = try_await(open_socket(AF_INET, SOCK_STREAM, 0, 0));
    defer_err_async
    {
        co_await close_file(fd);
    };

    int opt = 1;
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int)) < 0)
        co_return Error("Error setting socket options SO_REUSEADDR");

    if (setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(int)) < 0)
        co_return Error("Error setting socket options SO_REUSEPORT");

    try_await(bind_socket(fd, (sockaddr *)&addr, sizeof(addr)));
    try_await(listen_socket(fd, 10));


    co_return fd;
}



struct WaitGroup {
    uint32_t count = 0;
    std::vector<FutureHandler<int>*> futures;
    
    void add(int n) {
        count += n;
    }

    void done() {
        if (count == 0) {
            return;
        }

        count--;

        if (count == 0) {
            std::vector<FutureHandler<int>*> newFutures;
            std::swap(futures, newFutures);


            std::cout << "Done called, resuming " << newFutures.size() << " futures" << std::endl;

            for (auto &f : newFutures) {
                f->set(newFutures.size());
            }
        }
    }

    Future<int> wait() {
        if (count == 0) {
            return Future<int>(0);
        }

        FutureHandler<int> *handler = new FutureHandler<int>();
        futures.push_back(handler);
        return Future<int>(handler);
    }
};