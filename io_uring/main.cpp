//This program create a simple server socket

#include <iostream>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include <liburing.h>

// include std::array, std::size_t
#include <array>
#include <stdexcept>


#define PORT 8081
#define SOCKET_URING_ENTRIES 1000


// struct uring_request {
//     uint8_t type; // 0: accept, 1: read, 2: write
// } 

typedef struct uring_request {
    uint8_t type; // 0: accept, 1: read, 2: write
    int fd;
} uring_request_t;

volatile bool running = true;

void signal_handler(int signal) {
        running = false;
}



template <typename T>
struct PoolItem {
    T item;
    uint8_t flags; // 0: free, 1: in use
};

#define POOL_ITEM_FREE 0
#define POOL_ITEM_IN_USE 1

template <typename T, std::size_t Size>
class StaticPool {
public:
    StaticPool() : free_head(0), free_tail(Size - 1), used_count(0) {
        for (std::size_t i = 0; i < Size; ++i) {
            free_indices[i] = i;
        }
    }

    T* allocate() {
        size_t remaining = this->remaining();
        if (remaining == 0) {
            throw std::out_of_range("Pool is full, cannot allocate more objects.");
        }

        std::size_t index = free_indices[free_head];
        
        if (buffer[index].flags == POOL_ITEM_IN_USE) {
            throw std::invalid_argument("Object is already allocated. this should never happen.");  
        }   

        buffer[index].flags = POOL_ITEM_IN_USE;

        free_head = (free_head + 1) % Size;
        ++used_count;

        return &(buffer[index].item);
    }

    void deallocate(T* obj) {
        std::size_t index = ((PoolItem<T>*) obj) - buffer.data();
        if (buffer[index].flags == POOL_ITEM_FREE) {
            throw std::invalid_argument("Object is already deallocated.");
        }

        if (used_count == 0) {
            throw std::invalid_argument("Pool is empty, cannot deallocate objects.");
        }


        if (index >= Size || index < 0) {
            throw std::invalid_argument("Pointer does not belong to this pool.");
        }

        // Add the index back to the circular free index buffer
        free_tail = (free_tail + 1) % Size;
        free_indices[free_tail] = index;
        --used_count;
    }

    std::size_t remaining() const {
        return Size - used_count;
    }

private:
    std::array<PoolItem<T>, Size> buffer;                 // The statically allocated buffer for objects
    std::array<std::size_t, Size> free_indices; // Circular buffer for free indices
    std::size_t free_head;                      // Head of the free index circular buffer
    std::size_t free_tail;                      // Tail of the free index circular buffer
    std::size_t used_count;                     // Number of currently allocated objects
};

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



#define SOCK_BUFF_SIZE 4096
#define SOCK_BUFF_COUNT 128

uint8_t sock_buffs[SOCK_BUFF_COUNT][SOCK_BUFF_SIZE];


#define SOCEKT_TRY_MULTIPLE_PORTS 1
int setup_socket(uint16_t port, int backlog, uint8_t flags) {

    int server_fd, socket_opt;  
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        return -1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;

    int attempts = 0, max_attempts = flags & SOCEKT_TRY_MULTIPLE_PORTS ? 10 : 1;
    while (attempts < max_attempts) {
        address.sin_port = htons(port + attempts);
        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) >= 0) {
            break;
        }

        attempts++;
    }

    if (attempts == max_attempts) {
        close(server_fd);
        return -1;
    }


    if (listen(server_fd, backlog) < 0) {
        close(server_fd);
        return -1;
    }

    return server_fd;
}



int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    int server_fd = setup_socket(PORT, 10, SOCEKT_TRY_MULTIPLE_PORTS);
    if (server_fd < 0) {
        std::cerr << "Failed to setup server socket" << std::endl;
        return 1;
    }

    std::cout << "Server listening on port " << PORT << std::endl;

    struct io_uring ring;

    if (io_uring_queue_init(SOCKET_URING_ENTRIES, &ring, 0) < 0) {
        std::cerr << "io_uring_queue_init failed" << std::endl;
        close(server_fd);
        return 1;
    }


    StaticPool<uring_request_t, SOCKET_URING_ENTRIES> request_pool;


    int BGID = 0;

    struct io_uring_buf_ring * br = setup_buffer_ring(&ring, BGID, SOCK_BUFF_COUNT, SOCK_BUFF_SIZE, (uint8_t *) sock_buffs);
    if(br == NULL) {
        std::cerr << "setup_buffer_ring failed" << std::endl;
        close(server_fd);
        return 1;
    }


    struct io_uring_sqe *tmp_sqe = NULL;
    uring_request_t *tmp_request = NULL;

    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    
    
    tmp_sqe = io_uring_get_sqe(&ring);
    uring_request_t accept_request = {0};
    io_uring_prep_multishot_accept(tmp_sqe, server_fd, (struct sockaddr *) &client_addr, &client_addr_len, 0);
    io_uring_sqe_set_data(tmp_sqe, &accept_request);
    io_uring_submit(&ring);

    struct io_uring_cqe *cqe;

    

    int recv_multishot_flags = 0 | IOSQE_BUFFER_SELECT;
    while (running)
    {
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) {
            std::cerr << "io_uring_wait_cqe failed" << std::endl;
            break;
        }

        uring_request_t *request = (uring_request_t *) io_uring_cqe_get_data(cqe);

        if (request->type == 0) {
            int client_fd = cqe->res;  
            std::cout << "Accept request new fd: " << client_fd << std::endl;

            tmp_sqe = io_uring_get_sqe(&ring);
            tmp_sqe->flags = recv_multishot_flags;
            tmp_sqe->buf_group = BGID;
            tmp_request = request_pool.allocate();
            tmp_request->type = 1;
            tmp_request->fd = client_fd;
        
            io_uring_prep_recv_multishot(tmp_sqe, client_fd, NULL, 0, 0);
            io_uring_sqe_set_data(tmp_sqe, tmp_request);
            io_uring_submit(&ring);
        } else if (request->type == 1) {

            int bytes_received = cqe->res;
            int buffer_id = -1;
            if(cqe->flags & IORING_CQE_F_BUFFER) {
                buffer_id = cqe->flags >> IORING_CQE_BUFFER_SHIFT;
            }

            if (cqe->flags & IORING_CQE_F_BUF_MORE) {
                throw std::runtime_error("Buffer more flag is not supported");
            }

            #ifdef INVARIANT_CHECKS
            if((buffer_id < 0) == (bytes_received <= 0)) {
                std::cerr << "Buffer id should be present if bytes received is positive" << std::endl;
            }

            if(buffer_id >= 0 && buffer_id < SOCK_BUFF_COUNT) {
                std::cerr << "Buffer id out of range" << std::endl;
            }
            #endif

            if (bytes_received > 0) {
                sock_buffs[buffer_id][bytes_received] = '\0';
                std::cout << "Received message: " << sock_buffs[buffer_id] << std::endl;
           
            } else {
                if (bytes_received == -ENOBUFS) {
                    break;  
                }

                std::cout << "Closing connection, exit code: " << bytes_received << std::endl;
                close(request->fd);
                request_pool.deallocate(request);

            }


            if (buffer_id != -1) {
                io_uring_buf_ring_add(br, sock_buffs[buffer_id], SOCK_BUFF_SIZE, 0, io_uring_buf_ring_mask(SOCK_BUFF_COUNT), 0);
                io_uring_buf_ring_advance(br, 1);
            }

        } else if (request->type == 2) {
            std::cout << "Write request" << std::endl;
        } else {
            std::cerr << "Unknown request type" << std::endl;
        }

        io_uring_cqe_seen(&ring, cqe);
    }
    

    io_uring_queue_exit(&ring);
    close(server_fd);

    return 0;
}
