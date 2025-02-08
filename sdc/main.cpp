#include <coroutine>
#include <iostream>
#include <liburing.h>
#include <vector>
#include <map>
#include <limits>
#include <variant>


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
#define reject_if_fail(res) ttry(res, return Promise<int>::rejected(err), _val)




template<typename T>
class Promise {
    public:
        bool resolved;
        // T value;
        Res<T> value;
        std::coroutine_handle<> handle = nullptr;

        Promise() {
            this->resolved = false;
        }

        static Promise<T> resolved (T value) {
            Promise<T> p;
            p.resolve(value);
            return p;
        }

        static Promise<T> rejected (Error err) {
            Promise<T> p;
            p.reject(err);
            return p;
        }

        void resolve(T value) {
            this->value = value;
            this->resolved = true;
            if (this->handle) {
                this->handle.resume();
            }
        }

        void reject(Error err) {
            this->value = err;
            this->resolved = true;
            if (this->handle) {
                this->handle.resume();
            }
        }

        Res<T> get() {
            return this->value;
        }

        bool is_resolved() {
            return this->resolved;
        }

        void set_handle(std::coroutine_handle<> handle) {
            if (this->handle) {
                throw new std::exception();
            }

            this->handle = handle;

            if(this->resolved) {
                this->handle.resume();
            }
        }

        auto operator co_await() noexcept {
            struct Awaiter {
                Promise<T> &promise;

                bool await_ready() noexcept {
                    return promise.is_resolved();
                }

                Res<T> await_resume() noexcept {
                    return promise.get();
                }

                void await_suspend(std::coroutine_handle<> handle) {
                    promise.set_handle(handle);
                }
            };

            return Awaiter{ *this };
        }
};


namespace io {

Error ERR_IOURING_INIT_FAILED = "io_uring_init failed";

class IOUring {
public:
    int init(unsigned entries, unsigned flags) {
        this->ring_ptr = &this->ring;
        return io_uring_queue_init(entries, this->ring_ptr, flags);
    }

    ~IOUring() {
        if (this->ring_ptr == nullptr) {
            io_uring_queue_exit(this->ring_ptr);
        }
    }

    // IO URING UTILS
    Res<Promise<int>&> waitMS(unsigned ms) {
        struct __kernel_timespec ts = {
            .tv_sec = ms / 1000,
            .tv_nsec = (ms % 1000) * 1000000
        };

        return this->timeout(&ts);
    }

   
    Res<struct io_uring_sqe*> get_sqe() {
        struct io_uring_sqe *sqe = io_uring_get_sqe(this->ring_ptr);
        if (sqe == nullptr) {
            return ERR_IOURING_INIT_FAILED;
        }

        return sqe;
    }

    Promise<int>& open_file(const char* path, int flags, mode_t mode) {
        struct io_uring_sqe *sqe = return_if_fail(this->get_sqe());
        io_uring_prep_open(sqe, path, flags, mode); 

        
        return this->lazy_submit<int>(sqe);
    }

    Res<Promise<int>&> close_file(int fd) {
        struct io_uring_sqe *sqe = return_if_fail(this->get_sqe());
        io_uring_prep_close(sqe, fd);
        return this->lazy_submit<int>(sqe);
    }

    Res<Promise<int>&> read_file(int fd, void* buf, unsigned nbytes, __u64 offset) {
        struct io_uring_sqe *sqe = return_if_fail(this->get_sqe());
        io_uring_prep_read(sqe, fd, buf, nbytes, offset);
        return this->lazy_submit<int>(sqe);
    }

    Res<Promise<int>&> write_file(int fd, const void* buf, unsigned nbytes, __u64 offset) {
        struct io_uring_sqe *sqe = return_if_fail(this->get_sqe());
        io_uring_prep_write(sqe, fd, buf, nbytes, offset);
        return this->lazy_submit<int>(sqe);
    }

    Res<Promise<int>&> timeout(struct __kernel_timespec* ts) {
        struct io_uring_sqe *sqe = return_if_fail(this->get_sqe());
        io_uring_prep_timeout(sqe, ts, 0, 0);
        return this->lazy_submit<int>(sqe);
    }


    template<typename T>
    Res<Promise<T>&> lazy_submit(struct io_uring_sqe *sqe) {

        Promise<T>* p = new (std::nothrow) Promise<T>();
        if (p == nullptr) 
            return ERR_NEW_FAILED;

        auto next_id = this->next_request_id++ % std::numeric_limits<uint64_t>::max();
        io_uring_sqe_set_data(sqe, (void*) next_id);
        this->pending_requests[next_id] = (void*) p;
        this->n_pending_req++;
        this->submit();

        return *p;
    }

    void submit() {
        io_uring_submit(this->ring_ptr);
    }

    bool Idle () {
        std::cout << this << std::endl;

        return this->n_pending_req == 0;
    }

    int handle_cqe() {
        struct io_uring_cqe *cqe;
        int res = io_uring_peek_cqe(this->ring_ptr, &cqe);
        if (res == -EAGAIN) {
            res = io_uring_wait_cqe(this->ring_ptr, &cqe);
        }

        if (res < 0) {
            return res;
        }

        auto request_id = (uint64_t) io_uring_cqe_get_data(cqe);
        auto promise = this->pending_requests[request_id];

        io_uring_cqe_seen(this->ring_ptr, cqe);
        this->n_pending_req--;
        this->pending_requests.erase(request_id);
        if (cqe->res < 0) {
            ((Promise<int>*) promise)->resolve(cqe->res);
        } else {
            ((Promise<int>*) promise)->resolve(cqe->res);
        }

        //TODO THIS PROMISE SHOULD BE CLEANED UP

        return 0;
    }

uint64_t n_pending_req = 0;

private:
    uint64_t next_request_id = 0;
    std::map<int, void*> pending_requests;

    struct io_uring ring;
    struct io_uring *ring_ptr = nullptr;
};

thread_local IOUring io_uring = IOUring();
int thread_initialize_async_engine() {
    return io_uring.init(100, 0);
}

};



go prova() {

    auto r = ttry(io::io_uring.waitMS(5000), co_return, auto __res = co_await _val; __res);

    auto res = co_await ttry(io::io_uring.waitMS(5000), co_return); 
    auto r = co_await res;
    std::cout << "waitMS: " << r << std::endl;




    // auto fd = co_await *io_uring->open_file("test.txt", O_CREAT | O_RDWR, 0666);
    // auto fd = io::open_file()
    // std::cout << "open_file: " << fd << std::endl;

    // co_await io_uring->waitMS(5000);

    // auto written = co_await *io_uring->write_file(fd, "hello world", 11, 0);
    // std::cout << "write_file: " << written << std::endl;

    // auto closed = co_await *io_uring->close_file(fd);
    // std::cout << "close_file: " << closed << std::endl;


    co_return;
}


volatile bool running = 1;
void singal_handler(int sig) {
    running = 0;
}

int main () {   
    signal(SIGINT, singal_handler);
    signal(SIGTERM, singal_handler);

    std::cout << "starting" << std::endl;

    int res = io::thread_initialize_async_engine();
    if (res < 0) {
        std::cerr << "io_uring_init failed" << std::endl;
        return 1;
    }

    prova();
    prova();
    prova();

    io::io_uring.Idle();

    std::cout << "submitting" << std::endl;


    while (running || !(io::io_uring.Idle())) {
        io::io_uring.handle_cqe();
    }

    std::cout << "done" << std::endl;

    return 0;
}