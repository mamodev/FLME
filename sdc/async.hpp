#pragma once

#include <coroutine>
#include <iostream>
#include <liburing.h>
#include <vector>
#include <map>
#include <limits>
#include <variant>
#include <functional>
#include <type_traits>
#include <utility>
#include <string.h>

#include <netinet/in.h>
#include <arpa/inet.h>

struct Error { 
    uint16_t code;
    std::string message;
};

Error ERR_GENERIC = { 0, "Something went wrong" };
Error ERR_NEW_FAILED = { 1, "new operator failed" };

template<typename T>
struct Res {
    using StoredType = std::conditional_t<
        std::is_reference_v<T>,
        std::reference_wrapper<std::remove_reference_t<T>>,
        T
    >;

    std::variant<std::monostate, StoredType, Error> data;

    Res() {
    }

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

    inline T getValue() const {
        if constexpr (std::is_reference_v<T>) {
            return std::get<std::reference_wrapper<std::remove_reference_t<T>>>(data).get();
        } else {
            return std::get<T>(data);
        }
    }

    inline Error getError() const {
        return std::get<Error>(data);
    }

    inline bool is_ok() const {
        return std::holds_alternative<StoredType>(data);
    }

    inline bool is_setted() const { 
        return !std::holds_alternative<std::monostate>(data);
    }
};

#define ttry(res, ret_statement, ok_statement) \
    ({ \
        auto _r = res; \
        if (!_r.is_ok()) { \
            Error err = _r.getError(); \
            ret_statement; \
        } \
        auto _val = _r.getValue(); \
        ok_statement; \
    })

#define return_if_fail(res) ttry(res, return err, _val)
#define co_return_if_fail(res) ttry(res, co_return, _val)



thread_local int corutines_active = 0;

class ExitException : public std::exception {
    public:
        Error error;
        ExitException(Error e) {
            this->error = e;
        }
        ExitException() {}
};

class rutine {
    public:
        struct promise;
        using promise_type=promise;

        std::coroutine_handle<rutine::promise> handle;

        rutine(std::coroutine_handle<rutine::promise> h) {
            corutines_active++;
            this->handle = h;
        }

        ~rutine() {
            if (this->handle) {
                std::cout << "Destroying rutine" << std::endl;
                this->handle.destroy();
            }
        }

        struct promise {

            rutine get_return_object() {
                auto c = rutine(
                    std::coroutine_handle<rutine::promise>::from_promise(*this)
                );

                return c;
            }

            void unhandled_exception() noexcept { 
                
            }

            void return_void() noexcept { }

            std::suspend_never initial_suspend() noexcept {
                return {};
            }

            std::suspend_always final_suspend() noexcept { 

                corutines_active--;
                return {};
            }
        };
};


template<typename T>
struct PromiseHandle {
    Res<T> value;
    std::coroutine_handle<> handle;

    PromiseHandle() {
    }

    inline void resolve(T val) {
        this->value.data = val;
        std::cout<< "Resolving handle: " << this->value.is_setted() << std::endl;
        if(this->handle)  
            this->handle.resume();
    }

    inline void reject(Error err) {
        this->value.data = err;
        if(this->handle) 
            this->handle.resume();
    }

    inline void set_handle(std::coroutine_handle<> h) {
        std::cout << "Setting handle: " << this->value.is_setted() << std::endl;

        if(this->value.is_setted()) {
            h.resume();
        } else {
            std::cout << "Saving handle" << std::endl;
            this->handle = h;
        }
    }
};


template<typename T>
class Promise {
    public:
        std::variant<PromiseHandle<T>*, PromiseHandle<T>> handle;

        Promise() {
            this->handle = PromiseHandle<T>();  
        }

        Promise(PromiseHandle<T>* handle) {
            this->handle = handle;
        }

        Promise(Error err) {
            this->handle = PromiseHandle<T>();
            std::get<PromiseHandle<T>>(this->handle).reject(err);
        }

        Promise(T val) {
            this->handle = PromiseHandle<T>();
            std::get<PromiseHandle<T>>(this->handle).resolve(val);
        }

        constexpr bool is_ptr() {
            return std::holds_alternative<PromiseHandle<T>*>(this->handle);
        }

        constexpr bool is_value() {
            return std::holds_alternative<PromiseHandle<T>>(this->handle);
        }

        constexpr PromiseHandle<T>* get_ptr() {
            return std::get<PromiseHandle<T>*>(this->handle);
        }

        constexpr PromiseHandle<T> get_value() {
            return std::get<PromiseHandle<T>>(this->handle);
        }

        constexpr Res<T> get() {
            if (this->is_ptr()) 
                return this->get_ptr()->value;
            else if (this->is_value()) 
                return this->get_value().value;
            else 
                return ERR_GENERIC;
        }

        bool is_resolved() {
            if (this->is_ptr()) 
                return this->get_ptr()->value.is_setted();
            else if (this->is_value()) 
                return this->get_value().value.is_setted();
            else 
                return false;
        }

        void set_handle(std::coroutine_handle<> handle) {
            if (this->is_ptr()) 
                this->get_ptr()->set_handle(handle);
            else if (this->is_value()) 
                this->get_value().set_handle(handle);
            else 
                throw std::runtime_error("THIS SHOULD NEVER HAPPEN");
        }

        auto operator co_await() noexcept {
            struct Awaiter {
                Promise<T> &promise;

                bool await_ready() noexcept {
                    std::cout << "await_ready: " << promise.is_resolved() << std::endl;
                    return promise.is_resolved();
                }

                Res<T> await_resume() noexcept {
                    std::cout << "await_resume" << std::endl;   
                    return promise.get();
                }

                void await_suspend(std::coroutine_handle<> handle) {
                    std::cout << "await_suspend: " << this->promise.is_ptr() << std::endl;
                    promise.set_handle(handle);
                }
            };

            return Awaiter{ *this };
        }
};




class WaitGroup {
    public: 
        int count = 0;
        std::vector<std::coroutine_handle<>> handles;

        WaitGroup(unsigned n) {
            this->count = n;
        }

        void unlock() {
            for (auto handle : this->handles) {
                handle.resume();
            }
        }

        void add(int n) {
            this->count += n;
        }

        void done() {
            this->count--;
            if (this->count == 0) {
                this->unlock();
            }
        }

        auto operator co_await() noexcept { 
            struct Awaiter {
                WaitGroup &wg;

                bool await_ready() noexcept {
                    return wg.count == 0;
                }

                void await_suspend(std::coroutine_handle<> handle) {
                    wg.handles.push_back(handle);
                }

                void await_resume() noexcept {}
            };

            return Awaiter{ *this };
        }
};



template <typename T = void>
class [[nodiscard]] Task {
public:

    struct Promise;
    
    struct FinalAwaiter {
        bool await_ready() const noexcept { 
            return false; }

        void await_suspend(std::coroutine_handle<Promise> handle) noexcept
        {

            Promise& promise = handle.promise();
            if (promise.continuation) {
                return promise.continuation.resume();
            }
        }

        void await_resume() const noexcept { 
        }
    };

    struct Promise {
        std::coroutine_handle<> continuation;
        Res<T> result;

        Task get_return_object()
        {
            return Task { std::coroutine_handle<Promise>::from_promise(*this) };
        }

        void unhandled_exception() noexcept {
            auto curr = std::current_exception();
            if(!curr) {
                result = ERR_GENERIC;
                return;
            }

            try {
                std::rethrow_exception(curr);
            } catch (ExitException& e) {
                result = e.error;
            } catch (...) {
                result = ERR_GENERIC;
            }

        }

        void return_value(Res<T> res) noexcept { result = std::move(res); }
        std::suspend_never initial_suspend() noexcept { return {}; }
        FinalAwaiter final_suspend() noexcept {
            return {}; 
        }
    };

    using promise_type = Promise;


    Task() = default;
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    Task(Task&& other) : handle_(std::exchange(other.handle_, nullptr)) {}
    Task& operator=(Task&& other)
    {
        if (handle_) {
            handle_.destroy();
        }
        handle_ = std::exchange(other.handle_, nullptr);
        return *this;
    }

    ~Task() {
        if (handle_) 
            handle_.destroy();
    }

    struct Awaiter {
        std::coroutine_handle<Promise> handle;

        bool await_ready() const noexcept { 
            return !handle || handle.done(); }

        auto await_suspend(std::coroutine_handle<> calling) noexcept
        {
            handle.promise().continuation = calling;
            return handle;
        }

        Res<T> await_resume() noexcept { 
            return std::move(handle.promise().result); }
    };

    auto operator co_await() noexcept { return Awaiter { handle_ }; }

private:
    explicit Task(std::coroutine_handle<Promise> handle)
        : handle_(handle)
    {
    }

    std::coroutine_handle<Promise> handle_;
};


template<typename T>
T unwrap_or_excep(Res<T> res) {
    if (!res.is_ok()) {
        throw ExitException(res.getError());
    }

    return res.getValue();
}

#define try_await(x) unwrap_or_excep(co_await x)

// ================== Defer ==================
// ================== Defer ==================
// ================== Defer ==================

struct defer_dummy {};
template <class F> 
struct deferrer { 
    F f; 

    ~deferrer() { f(); } 
};

template <class F> deferrer<F> operator*(defer_dummy, F f) { return {f}; }
#define DEFER_(LINE) zz_defer##LINE
#define DEFER(LINE) DEFER_(LINE)
#define defer auto DEFER(__LINE__) = defer_dummy{} *[&]()

struct defer_error_dummy {};
template <class F> struct deferrer_error {
    F f;
    deferrer_error(F f) : f(f) {}
    ~deferrer_error() {
      if (std::uncaught_exceptions()) {
        f();
      }
    }
    deferrer_error& operator=(const deferrer_error&) = delete;
    deferrer_error(const deferrer_error&) = delete;
    deferrer_error(deferrer_error&&) = delete;
    deferrer_error& operator=(deferrer_error&&) = delete;
};

template <class F> deferrer_error<F> operator*(defer_error_dummy, F f) {
    return deferrer_error<F>(f);
}

#define DEFER_ERROR_(LINE) zz_defer_error##LINE
#define DEFER_ERROR(LINE) DEFER_ERROR_(LINE)
#define defer_error auto DEFER_ERROR(__LINE__) = defer_error_dummy{} *[&]()


// ================== Defer ==================
// ================== Defer ==================
// ================== Defer ==================

namespace io {

Error ERR_IOURING_INIT_FAILED = { 100, "io_uring_init failed" };
Error ERR_IOURING_CANCELD = { 101, "io_uring was canceld, app is shutting down" };

class IOUring {
public:
    enum class Type {
        COULD_INDEFINELY_BLOCK,
        EVENTUALLY_UNBLOCKED,
        TIMEOUT,
    };

    struct io_request {
        Type type;
        void* promise_handle;
    };


    int init(unsigned entries, unsigned flags) {
        this->ring_ptr = &this->ring;
        return io_uring_queue_init(entries, this->ring_ptr, flags);
    }

    ~IOUring() {
        if (this->ring_ptr == nullptr) {
            io_uring_queue_exit(this->ring_ptr);
        }
    }

    Res<struct io_uring_sqe*> get_sqe() {
        struct io_uring_sqe *sqe = io_uring_get_sqe(this->ring_ptr);
        if (sqe == nullptr) {
            return ERR_IOURING_INIT_FAILED;
        }

        return sqe;
    }

   

    template<typename T>
    Promise<T> lazy_submit(struct io_uring_sqe *sqe, Type type=Type::EVENTUALLY_UNBLOCKED) {
        if (this->canceld) {
            return ERR_IOURING_CANCELD;
        }

        PromiseHandle<T>* p_handle = new PromiseHandle<T>();
        if (p_handle == nullptr) 
            return ERR_NEW_FAILED;

        uint64_t next_id = this->next_id();
        // io_uring_sqe_set_data(sqe, (void*) next_id);
        io_uring_sqe_set_data64(sqe,  next_id);

        // std::cout << "Submitting request id: " << next_id << " With ptr: " << p_handle << std::endl;

        this->pending_requests[next_id] = {
            .type = type,
            .promise_handle = p_handle
        };

        this->n_pending_req++;
        this->submit();
        return p_handle;
    }

    uint64_t inline next_id() {
        return this->next_request_id++ % (std::numeric_limits<uintptr_t>::max() - this->reserved_ids);
    }

    uint64_t inline get_reserved_id(uint64_t position) {
        return std::numeric_limits<uintptr_t>::max() - this->reserved_ids + position;
    }

    bool inline is_reserved_id(uint64_t id) {
        return id >= std::numeric_limits<uintptr_t>::max() - this->reserved_ids;
    }

    void submit() {
        io_uring_submit(this->ring_ptr);
    }

    bool Idle () {
        return this->n_pending_req == 0;
    }

    // This function is safe to call multiple times, it will only cancel the requests once
    // this is intended to be called when shutting down the application
    void cancel() { 
        if (this->canceld) {
            return;
        }

        this->canceld = true;
        uint64_t cancel_id = get_reserved_id(0);

        uint64_t c = 0;
        for (auto& [id, req] : this->pending_requests) {
            if (req.type == Type::EVENTUALLY_UNBLOCKED) 
                continue;   

            struct io_uring_sqe *sqe = io_uring_get_sqe(this->ring_ptr);
            if (sqe == nullptr) {
                throw std::runtime_error("io_uring_get_sqe in handle_cqe, this should be handled better but is an hack :)");
            }

            // std::cout << "Cancelling request " << id << std::endl;
            io_uring_prep_cancel64(sqe, id, 0);
            io_uring_sqe_set_data64(sqe, cancel_id);
            c++;
        }   

        if (c > 0) {
            this->submit();              
        }

    }

    int handle_cqe() {
        struct io_uring_cqe *cqe;
        uint64_t cancel_id = get_reserved_id(0);

        int res = io_uring_peek_cqe(this->ring_ptr, &cqe);
        if (res == -EAGAIN) {
            // std::cout << "EAGAIN, waiting" << std::endl;
            res = io_uring_wait_cqe(this->ring_ptr, &cqe);
        }

        if (res == -EINTR) {
            return 0;  
        }

        if (res < 0) {
            return res;
        }

        // std::cout << "Handling CQE res: " << res << std::endl;

        uint64_t request_id = io_uring_cqe_get_data64(cqe);
        if (request_id == cancel_id) {
            io_uring_cqe_seen(this->ring_ptr, cqe);
            std::cout << "cancel response" << std::endl;
            return 0;
        }

       
        if (this->pending_requests.find(request_id) == this->pending_requests.end()) {
            throw std::runtime_error("Request id not found in pending_requests, this should never happen");
        }

        // std::cout << "Handelig Request id: " << request_id << std::endl;
        // print_pending_req();
        
        io_request req = this->pending_requests[request_id];
        if (req.promise_handle == nullptr) {
            throw std::runtime_error("io_request promise_handle is null, this should never happen");
        }

        PromiseHandle<int>* handle = (PromiseHandle<int>*) req.promise_handle;
        io_uring_cqe_seen(this->ring_ptr, cqe);
        this->n_pending_req--;

        bool failed = cqe->res < 0;
        if (req.type == Type::TIMEOUT) {
            failed = cqe->res != -ETIME;
        }

        if (failed) {
            // std::cout << "Request id: " << request_id << " rejected " << cqe->res << std::endl;
            // char *err = strerror(-cqe->res);
            // if (err != nullptr) {
            //     std::cout << "Error: " << err << std::endl;
            // }

            handle->reject(ERR_GENERIC);
        } else {
            std::cout << "Request id: " << request_id << " resolvaed" << std::endl;
            handle->resolve(cqe->res);
            std::cout << "SEGFAULT" << std::endl;
            // std::cout << "Request id: " << request_id << " resolvaed" << std::endl;
        }

        
        delete (PromiseHandle<int>*) handle;
        this->pending_requests.erase(request_id);

        return 0;
    }

uint64_t n_pending_req = 0;

private:

    void print_pending_req() {
        std::cout << "Pending requests: " << std::endl;
        for (auto& [id, req] : this->pending_requests) {
            std::cout << "\tRequest id: " << id << std::endl;
        }
    }

    bool canceld = false;
    uint64_t reserved_ids = 1;
    uint64_t next_request_id = 0;
    std::map<uint64_t, io_request> pending_requests;

    struct io_uring ring;
    struct io_uring *ring_ptr = nullptr;
};

thread_local IOUring io_uring = IOUring();
int thread_initialize_async_engine() {
    return io_uring.init(100, 0); 
}

Promise<int> open_file(const char* path, int flags, mode_t mode) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_open(sqe, path, flags, mode); 
    return io_uring.lazy_submit<int>(sqe);
}

Promise<int> close_file(int fd) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_close(sqe, fd);
    return io_uring.lazy_submit<int>(sqe);
}

Promise<int> read_file(int fd, void* buf, unsigned nbytes, __u64 offset) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_read(sqe, fd, buf, nbytes, offset);
    return io_uring.lazy_submit<int>(sqe, IOUring::Type::COULD_INDEFINELY_BLOCK);
}

Promise<int> write_file(int fd, const void* buf, unsigned nbytes, __u64 offset) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_write(sqe, fd, buf, nbytes, offset);
    return io_uring.lazy_submit<int>(sqe);
}

Promise<int> socket(int domain, int type, int protocol, unsigned flags) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_socket(sqe, domain, type, protocol, flags);
    return io_uring.lazy_submit<int>(sqe);
};

Promise<int> bind(int sockfd, struct sockaddr *addr, unsigned addrlen) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_bind(sqe, sockfd, addr, addrlen);
    return io_uring.lazy_submit<int>(sqe);
};

Promise<int> set_socket_opt(int sockfd, int level, int optname, const void *optval, unsigned optlen) {
    int res = setsockopt(sockfd, level, optname, optval, optlen);
    if (res < 0) {
        return ERR_GENERIC;
    }
    
    return 0;
};

Promise<int> listen(int sockfd, int backlog) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_listen(sqe, sockfd, backlog);
    return io_uring.lazy_submit<int>(sqe);
};

Promise<int> connect(int sockfd, struct sockaddr *addr, unsigned addrlen) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_connect(sqe, sockfd, addr, addrlen);
    return io_uring.lazy_submit<int>(sqe);
};

Promise<int> accept(int sockfd, struct sockaddr *addr, unsigned *addrlen) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_accept(sqe, sockfd, addr, addrlen, 0);
    return io_uring.lazy_submit<int>(sqe, IOUring::Type::COULD_INDEFINELY_BLOCK);
};

Promise<int> recv(int sockfd, void *buf, size_t len, int flags) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_recv(sqe, sockfd, buf, len, flags);
    return io_uring.lazy_submit<int>(sqe, IOUring::Type::COULD_INDEFINELY_BLOCK);
};

Promise<int> send(int sockfd, const void *buf, size_t len, int flags) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_send(sqe, sockfd, buf, len, flags);
    return io_uring.lazy_submit<int>(sqe, IOUring::Type::COULD_INDEFINELY_BLOCK);
};

Promise<int> timeout(__kernel_timespec *ts, unsigned count, unsigned flags) {
    struct io_uring_sqe *sqe = return_if_fail(io_uring.get_sqe());
    io_uring_prep_timeout(sqe, ts, count, flags);
    return io_uring.lazy_submit<int>(sqe, IOUring::Type::TIMEOUT);
};

// IO_URING Utils
Promise<int> waitMS(int ms) {
    __kernel_timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    return timeout(&ts, 1, 0);
}

Promise<int> set_socket_reuse_address(int sockfd) {
    int optval = 1;
    return set_socket_opt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
}

Promise<int> set_socket_reuse_port(int sockfd) {
    int optval = 1;
    return set_socket_opt(sockfd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval));
}

};  