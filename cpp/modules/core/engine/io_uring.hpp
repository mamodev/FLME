#pragma once

#include <liburing.h>
#include <vector>
#include <coroutine>

#include "../results.hpp"
#include "../futures.hpp"
#include "../corutines/fiber.hpp"

#include "../utils.hpp"

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

template <typename T>
void delete_ptr(void* ptr) {
    delete static_cast<T*>(ptr);
}


constexpr size_t POOL_CHUNK_SIZE = 1024;

struct IORequest {
    IORequestType type;
    FutureHandler<Res<int>> handler;
    bool canceled = false;

    constexpr inline FutureHandler<Res<int>>* get_handler() {
        return &handler;
    }
};

class IORequestPool {
    struct Chunk {
        std::vector<IORequest> objects;
        std::vector<IORequest*> free_list;

        Chunk() {
            objects.resize(POOL_CHUNK_SIZE);
            for (auto& obj : objects) {
                free_list.push_back(&obj);
            }
        }

        IORequest* allocate() {
            if (free_list.empty()) return nullptr;
            IORequest* obj = free_list.back();
            free_list.pop_back();
            return obj;
        }

        void deallocate(IORequest* obj) {
            free_list.push_back(obj);
        }

        bool is_empty() const {
            return free_list.size() == objects.size();
        }
    };

    std::vector<Chunk*> chunks;

public:
    ~IORequestPool() {
        for (auto* chunk : chunks) delete chunk;
    }

    IORequest* allocate() {
        for (auto* chunk : chunks) {
            IORequest* obj = chunk->allocate();
            if (obj) return obj;
        }

        // No free space, allocate a new chunk
        auto* new_chunk = new Chunk();
        chunks.push_back(new_chunk);
        return new_chunk->allocate();
    }

    void unsafe_deallocate(void *obj) {
        deallocate(static_cast<IORequest*>(obj));
    }

    void deallocate(IORequest* obj) {
        for (auto it = chunks.begin(); it != chunks.end(); ++it) {
            auto* chunk = *it;
            if (&chunk->objects[0] <= obj && obj < &chunk->objects[POOL_CHUNK_SIZE]) {
                chunk->deallocate(obj);
    
                // If chunk is empty and we have more than one chunk, remove it
                if (chunk->is_empty() && chunks.size() > 1) {
                    delete chunk; // Free the chunk memory
                    it = chunks.erase(it); // Erase and update iterator
                    --it; // Adjust iterator to avoid skipping the next element
                }
                return;
            }
        }
    }
};

class EventLoop {
    std::vector<std::coroutine_handle<>> yelded_coroutines;
    struct io_uring ring;

    uint64_t pending_io_requests = 0;


    uint8_t *core_ring_mem = nullptr;
    io_uring_buf_ring *core_ring = nullptr;

    IORequestPool request_pool;
 

    const int max_files = 8192;
    const int nr_shared_buffers = 1024;
    const int shared_buffer_size = 4096;
    bool initialized = false;

public:
    volatile bool running = true;

    Res<void> register_file(int fd) {
        int ret = io_uring_register_files_update(&ring, 0, &fd, 1);
        if (ret != 1) {
            return Error("error registering file, ERRNO: " + std::to_string(ret));
        }

        return std::nullopt;
    }

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

        core_ring_mem = (uint8_t*)malloc(nr_shared_buffers * shared_buffer_size);
        if (core_ring_mem == nullptr) {
            io_uring_queue_exit(&ring);
            return Error("error allocating buffer ring memory");
        }

        core_ring = setup_buffer_ring(&ring, 0, nr_shared_buffers, shared_buffer_size, core_ring_mem);
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

        req->handler.set_deallocator(delete_ptr<IORequest>, req);

        // auto req = request_pool.allocate();

        // auto cl = [this](void* ptr) {
        //     request_pool.unsafe_deallocate(ptr);
        // };

        // req->handler.set_deallocator(cl, req);

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

                    if (data->canceled) {
                        continue;
                    }

                    debug("Request of type " << encodeIORequestType(data->type) << " resolved with: " << res << " failed: " << failed);
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
            free(core_ring_mem);
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

Future<Res<int>> open_socket_direct(int domain, int type, int protocol, unsigned flags) {
    auto req = propagate(loop.new_request(IORequestType::SOCK_OPEN));
    auto sqe = propagate(loop.get_sqe());
   
    io_uring_prep_socket_direct_alloc(sqe, domain, type, protocol, flags);
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

#include <memory>
using TimeoutCancelToken = std::shared_ptr<bool>;

template <typename Callback, typename = std::enable_if_t<std::is_invocable_r_v<void, Callback> || std::is_invocable_r_v<Fiber, Callback>>>
Fiber __setTimeout_Fiber(Future<Res<int>> fut, Callback cb, TimeoutCancelToken ctoken) {
    auto res = co_await fut;
    if (res.is_ok() && !*ctoken) {
        cb();
        *ctoken = true;
    }

    co_return;
}

// template <typename Callback, typename = std::enable_if_t<std::is_invocable_r_v<void, Callback> || std::is_invocable_r_v<Fiber, Callback>>>
template<typename Callback>
Res<TimeoutCancelToken> setTimeout( int ms, Callback &&cb) {
    __kernel_timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;

    auto req = propagate(loop.new_request(IORequestType::TIMEOUT));
    auto sqe = propagate(loop.get_sqe());
   
    io_uring_prep_timeout(sqe, &ts, 1000000, 0);
    io_uring_sqe_set_data(sqe, (void*)req);
    loop.lazy_submit();

  
    FutureHandler<Res<int>>* handler = req->get_handler();

    //TODO make it no throw
    TimeoutCancelToken cancel = std::make_shared<bool>(false);
    __setTimeout_Fiber(Future<Res<int>>(handler), std::forward<Callback>(cb), cancel);

    return cancel;
}

bool clearTimeout(TimeoutCancelToken token) {
    if(token) {
        if (!*token) {
            *token = true;
            return true;
        }
    }

    return false;
}
