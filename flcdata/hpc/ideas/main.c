#include <fcntl.h>      // O_CREAT, O_RDWR
#include <mqueue.h>     // mq_open, mq_send, mq_receive, mq_close, mq_unlink
#include <signal.h>     // SIGCHLD
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     // strncpy
#include <sys/stat.h>   // S_IRUSR, S_IWUSR
#include <sys/types.h>  // pid_t
#include <sys/wait.h>   // waitpid
#include <unistd.h>     // ftruncate, fork, execlp, sleep, _exit
#include <sys/mman.h>   // shm_open, shm_unlink
#include <stdint.h>     // uint32_t
#include <stdbool.h>
#include <netdb.h>
#include <pthread.h>   // pthread_create, pthread_join


// Network
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>

#define MAX_CONCURRENT_MODELS 1024
#define MAX_CONCURRENT_EVENTS 1024

volatile sig_atomic_t running = 1;

void signal_handler(int signum) {
    running = 0;
}

#pragma pack(push, 1)
typedef struct {
    uint32_t model_version;

    uint32_t client_id;
    uint32_t partition_id;
    
    uint32_t data_offset;
    uint32_t data_size;

    uint32_t targets_offset;
    uint32_t targets_size;

    uint32_t ephochs;
    uint32_t batch_size;
    
    float learning_rate;
    float momentum;
    float weight_decay;
    bool shuffle;

} train_msg_t;
#pragma pack(push, 1)

uint64_t hash_client(uint32_t client_id, uint32_t partition_id) {
    return ((uint64_t)client_id << 32) | partition_id;
}

uint32_t client_id_from_hash(uint64_t hash) {
    return (uint32_t)(hash >> 32);
}

uint32_t partition_id_from_hash(uint64_t hash) {
    return (uint32_t)(hash & 0xFFFFFFFF);
}


typedef struct datset_allocation {
    uint32_t start;
    uint32_t end;
} dataset_allocation_t;

#define DEF_VECT(type) \
    typedef struct { \
        type *data; \
        size_t size; \
        size_t capacity; \
    } vect_##type; \
    \
    int vect_##type##_init(vect_##type *v) { \
        v->data = malloc(100 * sizeof(type)); \
        if (v->data == NULL) { \
            return -1; \
        } \
        v->size = 0; \
        v->capacity = 100; \
        return 0; \
    } \
    \
    int vect_##type##_push_back(vect_##type *v, type value) { \
        if (v->data == NULL) { \
            if (vect_##type##_init(v) != 0) { \
                return -1; \
            } \
        } \
        if (v->size == v->capacity) { \
            size_t new_capacity = v->capacity + 100; \
            type *new_data = realloc(v->data, new_capacity * sizeof(type)); \
            if (new_data == NULL) { \
                return -1; \
            } \
            v->data = new_data; \
            v->capacity = new_capacity; \
        } \
        v->data[v->size++] = value; \
        return 0; \
    } \
    \
    void vect_##type##_free(vect_##type *v) { \
        free(v->data); \
        v->data = NULL; \
        v->size = 0; \
        v->capacity = 0; \
    } \
    \
    void vect_##type##_remove_at(vect_##type *v, size_t index) { \
        if (index < v->size) { \
            for (size_t i = index; i < v->size - 1; ++i) { \
                v->data[i] = v->data[i + 1]; \
            } \
            v->size--; \
        } \
    }

DEF_VECT(dataset_allocation_t)
DEF_VECT(train_msg_t)
DEF_VECT(size_t)

uint32_t model_size = 0;
uint32_t dataset_size = 0;


struct shared_memory {
    int globals;
    int dataset;
    int results;
    
    uint64_t globals_size;
    uint64_t dataset_size;
    uint64_t results_size;

    void *globals_ptr;
    void *dataset_ptr;
    void *results_ptr;

    pthread_mutex_t dataset_allocation_mutex;
    vect_dataset_allocation_t dataset_allocations;

    int32_t globals_allocations[MAX_CONCURRENT_MODELS];
    int64_t results_allocations[MAX_CONCURRENT_EVENTS];
};


struct shared_memory SHM = {
    .globals = -1,
    .dataset = -1,
    .results = -1,

    .globals_size = 0,
    .dataset_size = 0,
    .results_size = 0,

    .dataset_allocation_mutex = PTHREAD_MUTEX_INITIALIZER,
    .globals_allocations = {0},

    .globals_ptr = NULL,
    .dataset_ptr = NULL,
    .results_ptr = NULL
};


int is_dsrange_alloc(uint32_t start, uint32_t end) {
    pthread_mutex_lock(&SHM.dataset_allocation_mutex);

    vect_dataset_allocation_t *v = &SHM.dataset_allocations;

    // Check if the range [start, end] is already allocated
    for (size_t i = 0; i < v->size; ++i) {
        if (v->data[i].start <= end && v->data[i].end >= start) {
            pthread_mutex_unlock(&SHM.dataset_allocation_mutex);
            return 1;
        }
    }
    
    pthread_mutex_unlock(&SHM.dataset_allocation_mutex);
    return 0;
}


void ds_add_alloc(uint32_t start, uint32_t end) {
    pthread_mutex_lock(&SHM.dataset_allocation_mutex);

    vect_dataset_allocation_t *v = &SHM.dataset_allocations;

    // Find the position to insert and check for merging
    size_t i = 0;
    while (i < v->size && v->data[i].end < start - 1) {
        i++;
    }

    // Now, i is the first range that might overlap or be adjacent to [start, end]
    uint32_t new_start = start;
    uint32_t new_end = end;

    // Merge all overlapping or adjacent ranges
    size_t merge_start = i;
    while (i < v->size && v->data[i].start <= end + 1) {
        if (v->data[i].start < new_start) new_start = v->data[i].start;
        if (v->data[i].end > new_end) new_end = v->data[i].end;
        i++;
    }

    // Remove the merged ranges
    size_t merge_end = i;
    if (merge_start < merge_end) {
        // Shift left to remove merged ranges
        size_t num_to_remove = merge_end - merge_start;
        for (size_t j = merge_start + num_to_remove; j < v->size; ++j) {
            v->data[j - num_to_remove] = v->data[j];
        }
        v->size -= num_to_remove;
    }

    // Insert the new merged range at merge_start
    // Make space for the new range
    if (vect_dataset_allocation_t_push_back(v, (dataset_allocation_t){0, 0}) != 0) {
        // handle allocation failure
        pthread_mutex_unlock(&SHM.dataset_allocation_mutex);
        return;
    }
    for (size_t j = v->size - 1; j > merge_start; --j) {
        v->data[j] = v->data[j - 1];
    }
    v->data[merge_start].start = new_start;
    v->data[merge_start].end = new_end;

    pthread_mutex_unlock(&SHM.dataset_allocation_mutex);
}

void ds_alloc_print(void) {
    pthread_mutex_lock(&SHM.dataset_allocation_mutex);

    vect_dataset_allocation_t *v = &SHM.dataset_allocations;

    printf("Dataset allocations:\n");
    for (size_t i = 0; i < v->size; ++i) {
        printf("  [%u, %u]\n", v->data[i].start, v->data[i].end);
    }

    pthread_mutex_unlock(&SHM.dataset_allocation_mutex);
}

void* get_gmodel_ptr(int32_t version) {
    for (int i = 0; i < MAX_CONCURRENT_MODELS; i++) {
        if (SHM.globals_allocations[i] == version) {
            return SHM.globals_ptr + i * model_size;
        }
    }
    return NULL;
}

void* next_free_gmodel(int32_t version) {
    for (int i = 0; i < MAX_CONCURRENT_MODELS; i++) {
        if (SHM.globals_allocations[i] == -1) {
            SHM.globals_allocations[i] = version;
            return SHM.globals_ptr + i * model_size;
        }
    }
    return NULL;
}

void* next_free_result(int64_t version) {
    for (int i = 0; i < MAX_CONCURRENT_EVENTS; i++) {
        if (SHM.results_allocations[i] == -1) {
            SHM.results_allocations[i] = version;
            return SHM.results_ptr + i * model_size;
        }
    }
    return NULL;
}

void* get_result_ptr(int64_t version) {
    for (int i = 0; i < MAX_CONCURRENT_EVENTS; i++) {
        if (SHM.results_allocations[i] == version) {
            return SHM.results_ptr + i * model_size;
        }
    }
    return NULL;
}

void free_result(int64_t version) {
    for (int i = 0; i < MAX_CONCURRENT_EVENTS; i++) {
        if (SHM.results_allocations[i] == version) {
            SHM.results_allocations[i] = -1;
            return;
        }
    }
}

mqd_t mq_in = (mqd_t)-1, mq_out = (mqd_t)-1;

int sock = -1;

pthread_t thread_sender = (pthread_t)0;
pthread_t thread_receiver = (pthread_t)0;

#define MAX_WORKERS 2
pid_t python_workers[MAX_WORKERS] = { -1 };

// 2 types of messages
// 1. C => Python (Train model xxx with data yyy and hyperparameters zzz, and put the result in ppp)
// 2. Python => C (Model xxx trained)

#pragma pack(push, 1)
struct ptc_msg {
    uint64_t id;
    int32_t err_code;

    uint64_t model_offset;
    uint64_t results_offset;
};
#pragma pack(pop)

#define MAX_SHAPE_LEN 16
#pragma pack(push, 1)
struct shape {
    uint32_t dim;
    uint32_t shape[MAX_SHAPE_LEN];
};
#pragma pack(pop)
// python struct format for shape: I16I

#pragma pack(push, 1)
struct ctp_msg {
    uint64_t id;

    uint64_t model_offset;
    uint64_t results_offset;
    uint32_t model_size;

    uint64_t data_offset;
    struct shape data_shape;

    uint64_t targets_offset;
    struct shape targets_shape;


    uint32_t ephochs;
    uint32_t batch_size;
    float learning_rate;
    float momentum;
    float weight_decay;
    bool shuffle;
};
#pragma pack(pop)

// python struct format for ctp_msg: QQQI QI16I QI16I
// python struct format for ptc_msg: Ii

#define IN_QUEUE  "/in_queue"
#define OUT_QUEUE "/out_queue"

#define MAX_MQ_MESSAGES      10
#define PTC_MESSAGE_SIZE  sizeof(struct ptc_msg)
#define CTP_MESSAGE_SIZE  sizeof(struct ctp_msg)

static void unlink_shm(void) {
    if (SHM.globals != -1) {
        shm_unlink("/globals");
        SHM.globals = -1;
    }
    if (SHM.dataset != -1) {
        shm_unlink("/dataset");
        SHM.dataset = -1;
    }
    if (SHM.results != -1) {
        shm_unlink("/results");
        SHM.results = -1;
    }
}

static void unlink_qm(void) {
    if (mq_in != (mqd_t)-1) {
        mq_close(mq_in);
        mq_unlink(IN_QUEUE);
        mq_in = (mqd_t)-1;
    }
    if (mq_out != (mqd_t)-1) {
        mq_close(mq_out);
        mq_unlink(OUT_QUEUE);
        mq_out = (mqd_t)-1;
    }
}

static void kill_workers(void) {
    for (int i = 0; i < MAX_WORKERS; i++) {
        if (python_workers[i] > 0) {
            kill(python_workers[i], SIGTERM);
            waitpid(python_workers[i], NULL, 0);
            python_workers[i] = -1;
        }
    }
}

static void close_all_sockets(void) {
    if (sock != -1) {
        close(sock);
        sock = -1;
    }
 
}

static void wait_threads(void) {
    if (thread_sender != (pthread_t)0) {
        pthread_cancel(thread_sender);
        pthread_join(thread_sender, NULL);
        thread_sender = (pthread_t)0;
    }
    if (thread_receiver != (pthread_t)0) {
        pthread_cancel(thread_receiver);
        pthread_join(thread_receiver, NULL);
        thread_receiver = (pthread_t)0;
    }
}

static void clean_up(void) {
    printf("Cleaning up...\n");
    running = 0;
    
    printf("Killing workers...\n");
    kill_workers();

    printf("Waiting for threads to finish...\n");
    wait_threads();

    printf("Unlinking message queues...\n");
    unlink_qm();
    
    printf("Unlinking shared memory...\n");
    unlink_shm();

    printf("Closing sockets...\n");
    close_all_sockets();
}

int init_shm(void) {
    for (int i = 0; i < MAX_CONCURRENT_MODELS; i++) {
        SHM.globals_allocations[i] = -1;
    }

    for (int i = 0; i < MAX_CONCURRENT_EVENTS; i++) {
        SHM.results_allocations[i] = -1;
    }

    SHM.globals = shm_open("/globals", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (SHM.globals == -1) {
        perror("shm_open globals");
        return -1;
    }

    SHM.dataset = shm_open("/dataset", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (SHM.dataset == -1) {
        unlink_shm();
        perror("shm_open dataset");
        return -1;
    }

    SHM.results = shm_open("/results", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (SHM.results == -1) {
        unlink_shm();
        perror("shm_open results");
        return -1;
    }

    return 0;
}

int allocate_shm(uint32_t model_size, uint32_t dataset_size) {
    if (ftruncate(SHM.globals, model_size * MAX_CONCURRENT_MODELS) == -1) {
        unlink_shm();
        perror("ftruncate globals");
        return -1;
    }

    if (ftruncate(SHM.dataset, dataset_size) == -1) {
        unlink_shm();
        perror("ftruncate dataset");
        return -1;
    }

    if (ftruncate(SHM.results, model_size * MAX_CONCURRENT_EVENTS) == -1) {
        unlink_shm();
        perror("ftruncate results");
        return -1;
    }

    SHM.globals_size = model_size * MAX_CONCURRENT_MODELS;
    SHM.dataset_size = dataset_size;
    SHM.results_size = model_size * MAX_CONCURRENT_EVENTS;

    return 0;
}

int mmap_shm(uint32_t model_size, uint32_t dataset_size) {
    SHM.globals_ptr = mmap(NULL, model_size * MAX_CONCURRENT_MODELS,
                           PROT_READ | PROT_WRITE,
                           MAP_SHARED, SHM.globals, 0);

    if (SHM.globals_ptr == MAP_FAILED) {
        unlink_shm();
        perror("mmap globals");
        return -1;
    }

    SHM.dataset_ptr = mmap(NULL, dataset_size,
                           PROT_READ | PROT_WRITE,
                           MAP_SHARED, SHM.dataset, 0);

    if (SHM.dataset_ptr == MAP_FAILED) {
        unlink_shm();
        perror("mmap dataset");
        return -1;
    }

    SHM.results_ptr = mmap(NULL, model_size * MAX_CONCURRENT_EVENTS,
                           PROT_READ | PROT_WRITE,
                           MAP_SHARED, SHM.results, 0);

    if (SHM.results_ptr == MAP_FAILED) {
        unlink_shm();
        perror("mmap results");
        return -1;
    }

    return 0;
}

int init_and_map_shm(uint32_t model_size, uint32_t dataset_size) {
    if (init_shm() == -1) {
        fprintf(stderr, "Failed to initialize shared memory\n");
        return -1;
    }

    if (allocate_shm(model_size, dataset_size) == -1) {
        fprintf(stderr, "Failed to allocate shared memory\n");
        return -1;
    }

    if (mmap_shm(model_size, dataset_size) == -1) {
        fprintf(stderr, "Failed to map shared memory\n");
        return -1;
    }

    return 0;
}

int init_mq_queues (void) {
    struct mq_attr attr_in = {
        .mq_flags   = 0,
        .mq_maxmsg  = MAX_MQ_MESSAGES,
        .mq_msgsize = PTC_MESSAGE_SIZE,
        .mq_curmsgs = 0
    };

    struct mq_attr attr_out = {
        .mq_flags   = 0,
        .mq_maxmsg  = MAX_MQ_MESSAGES,
        .mq_msgsize = CTP_MESSAGE_SIZE,
        .mq_curmsgs = 0
    };

    mq_in = mq_open(IN_QUEUE,
                    O_CREAT | O_RDWR,
                    S_IRUSR | S_IWUSR,
                    &attr_in);

    if (mq_in == (mqd_t)-1) {
        return -1;
    }

    mq_out = mq_open(OUT_QUEUE,
                     O_CREAT | O_RDWR,
                     S_IRUSR | S_IWUSR,
                     &attr_out);

    if (mq_out == (mqd_t)-1) {
        mq_close(mq_in);
        mq_unlink(IN_QUEUE);
        mq_in = (mqd_t)-1;
        return -1;
    }

    return 0;
}

pid_t spawn_worker(void) {

    char shm_global_size[64] = {0};
    char shm_dataset_size[64] = {0};
    char shm_results_size[64] = {0};

    snprintf(shm_global_size, sizeof(shm_global_size), "%lu", SHM.globals_size);
    snprintf(shm_dataset_size, sizeof(shm_dataset_size), "%lu", SHM.dataset_size);
    snprintf(shm_results_size, sizeof(shm_results_size), "%lu", SHM.results_size);

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return -1;
    }
    if (pid == 0) {

        //unlink stdin
        int fd = open("/dev/null", O_RDWR);
        if (fd == -1) {
            perror("open /dev/null");
            _exit(1);
        }
        if (dup2(fd, STDIN_FILENO) == -1) {
            perror("dup2");
            _exit(1);
        }
        
        // child
        execlp(
            "python3",        // program
            "python3",        // argv[0]
            "worker.py",      // script name
            "--in-queue", OUT_QUEUE,
            "--out-queue", IN_QUEUE,
            "--shm-models", "/globals",
            "--shm-dataset", "/dataset",
            "--shm-results", "/results",
            "--shm-models-size", shm_global_size,
            "--shm-dataset-size", shm_dataset_size,
            "--shm-results-size", shm_results_size,
            (char*)NULL
        );
        // if we get here, exec failed
        perror("execlp");
        _exit(1);
    }

    // parent returns child's pid
    return pid;
}

int init_workers(void) {
    for (int i = 0; i < MAX_WORKERS; i++) {
       python_workers[i] = spawn_worker();
        if (python_workers[i] < 0) {
            kill_workers();
            return -1;
        }
    }

    return 0;
}

int resolve(const char *hostname, struct sockaddr_in *out_addr) {
    struct addrinfo hints, *res, *p;
    int status;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; // Allow IPv4 or IPv6
    hints.ai_socktype = SOCK_STREAM;

    if ((status = getaddrinfo(hostname, NULL, &hints, &res)) != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(status));
        return -1;
    }

    for (p = res; p != NULL; p = p->ai_next) {
        void *addr;
        char *ipver;

        if (p->ai_family == AF_INET) { // IPv4
            memcpy(out_addr, p->ai_addr, sizeof(struct sockaddr_in));
            break;
        }
    }

    freeaddrinfo(res);
    return 0;
}

int host_connect(const char *hostname, int port) {
    struct sockaddr_in addr;
    if (resolve(hostname, &addr) == -1) {
        fprintf(stderr, "Failed to resolve hostname\n");
        return -1;
    }

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        perror("socket");
        return -1;
    }

    int size = 16 * 1024 * 1024;
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size)) == -1) {
        perror("setsockopt");
        close(sockfd);
        return -1;
    }

    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)) == -1) {
        perror("setsockopt");
        close(sockfd);
        return -1;
    }

    addr.sin_port = htons(port);

    if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("connect");
        close(sockfd);
        return -1;
    }

    return sockfd;
}

int recv_all(int sockfd, void *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        size_t chunk_size = len - total;
        ssize_t bytes_received = recv(sockfd, (char *)buf + total, chunk_size, 0);
        if (bytes_received == -1) {
            perror("recv");
            return -1;
        }
        if (bytes_received == 0) {
            perror("recv: connection closed");
            return -1;
        }

        total += bytes_received;
    }

    return 0;
}

int send_all(int sockfd, const void *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t bytes_sent = send(sockfd, (const char *)buf + total, len - total, 0);
        if (bytes_sent == -1) {
            perror("send");
            return -1;
        }
        total += bytes_sent;
    }

    return 0;
}

int sock_auth(int sockfd, uint32_t worker_id) {
    if (send(sockfd, &worker_id, sizeof(worker_id), 0) == -1) {
        perror("send");
        return -1;
    }

    return 0;
}

int recv_dataset_slice(int sockfd) {
    uint32_t data_offset, data_size, targets_offset, targets_size;
    if (recv_all(sockfd, &data_offset, sizeof(data_offset)) == -1) {
        perror("recv");
        return -1;
    }
    if (recv_all(sockfd, &data_size, sizeof(data_size)) == -1) {
        perror("recv");
        return -1;
    }
    if (recv_all(sockfd, &targets_offset, sizeof(targets_offset)) == -1) {
        perror("recv");
        return -1;
    }
    if (recv_all(sockfd, &targets_size, sizeof(targets_size)) == -1) {
        perror("recv");
        return -1;
    }

    if (recv_all(sockfd, SHM.dataset_ptr + data_offset, data_size) == -1) {
        perror("recv");
        return -1;
    }

    if (recv_all(sockfd, SHM.dataset_ptr + targets_offset, targets_size) == -1) {
        perror("recv");
        return -1;
    }

    ds_add_alloc(data_offset, data_offset + data_size);
    ds_add_alloc(targets_offset, targets_offset + targets_size);
    return 0;
}

int recv_model(int sockfd) {
    int32_t version = 0;
    if (recv_all(sockfd, &version, sizeof(version)) == -1) {
        perror("recv");
        return -1;
    }   

    if (get_gmodel_ptr(version) != NULL) {
        fprintf(stderr, "Model version %d already exists\n", version);
        return -1;
    }

    void* next_free_slot = next_free_gmodel(version);
    if (next_free_slot == NULL) {
        fprintf(stderr, "No free slot for model version %d\n", version);
        return -1;
    }

    if (recv_all(sockfd, next_free_slot, model_size) == -1) {
        perror("recv");
        return -1;
    }

    return 0;
}

pthread_mutex_t train_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
vect_train_msg_t train_queue = {0};

int train_queue_advance() {
    pthread_mutex_lock(&train_queue_mutex);

    int error = 0;

    vect_size_t free_slots = {0};

    for (size_t i = 0; i < train_queue.size; ++i) {
        train_msg_t *train_msg = &train_queue.data[i];

        // Check if the dataset range is already allocated
        if (!is_dsrange_alloc(train_msg->data_offset, train_msg->data_offset + train_msg->data_size)) {
            // printf("Dataset range [%u, %u] is not allocated\n",
            //        train_msg->data_offset, train_msg->data_offset + train_msg->data_size);
            continue;
        }

        void* model_ptr = get_gmodel_ptr(train_msg->model_version);
        if (model_ptr == NULL) {
            // printf("Model version %d not found\n", train_msg->model_version);
            continue;
        }
    
        // IT'S READY TO TRAIN
        uint64_t id = hash_client(train_msg->client_id, train_msg->partition_id);
        if (get_result_ptr(id) != NULL) {
            // There is probably a prev version of the client training
            // printf("Client %u, partition %u is already training\n",
                //    train_msg->client_id, train_msg->partition_id);
            continue;
        }

        void* result_ptr = next_free_result(id);
        if (result_ptr == NULL) {
            // No free slot for result, let's wait for a free slot
            // printf("No free slot for client %u, partition %u\n",
            //        train_msg->client_id, train_msg->partition_id);
            continue;
        }

        struct ctp_msg ctp_msg = {0};
        ctp_msg.id = id;
        ctp_msg.model_offset = (uint64_t)(model_ptr - SHM.globals_ptr);
        ctp_msg.results_offset = (uint64_t)(result_ptr - SHM.results_ptr);
        ctp_msg.model_size = model_size;

        ctp_msg.data_offset = train_msg->data_offset;
        ctp_msg.data_shape.dim = 2;
        ctp_msg.data_shape.shape[0] = train_msg->data_size / sizeof(float) / 100; //TODO: get real shape from dataset header
        ctp_msg.data_shape.shape[1] = 100; //TODO: get real shape from dataset header

        ctp_msg.targets_offset = train_msg->targets_offset;
        ctp_msg.targets_shape.dim = 1;
        ctp_msg.targets_shape.shape[0] = train_msg->targets_size / sizeof(uint64_t);

        ctp_msg.ephochs = train_msg->ephochs;
        ctp_msg.batch_size = train_msg->batch_size;
        ctp_msg.learning_rate = train_msg->learning_rate;
        ctp_msg.momentum = train_msg->momentum;
        ctp_msg.weight_decay = train_msg->weight_decay;
        ctp_msg.shuffle = train_msg->shuffle;


        // printf("Starting training for client %u, partition %u: hash_id = %lu, model_version = %d\n",
        //        train_msg->client_id, train_msg->partition_id, id, train_msg->model_version);


        if (mq_send(mq_out, (const char*)&ctp_msg, CTP_MESSAGE_SIZE, 0) == -1) {
            perror("mq_send");
            error = -1;
            break;
        }

        vect_size_t_push_back(&free_slots, i);
    }

    for (size_t i = free_slots.size; i > 0; --i) {
        size_t index = free_slots.data[i - 1];
        // i use -1 becouse i is the size of the vector

        vect_train_msg_t_remove_at(&train_queue, index);
    }


    pthread_mutex_unlock(&train_queue_mutex);
    return error;
}

void* t_data_receiver(void *arg) {
    int sockfd = (int)(intptr_t)arg;
    while(running) {
        uint32_t type = 0;
        if (recv_all(sockfd, &type, sizeof(type)) == -1) {
            perror("recv");
           break;
        }


        if (type == 1) {
            if (recv_model(sockfd) == -1) {
                perror("recv_model");
               break;
            }
        } else if (type == 0) {
            if (recv_dataset_slice(sockfd) == -1) {
                perror("recv_dataset_slice");
               break;
            }
        } else if (type == 2) {
            train_msg_t train_msg = {0};
            if (recv_all(sockfd, &train_msg, sizeof(train_msg)) == -1) {
                perror("recv_train_msg");
               break;
            }

            pthread_mutex_lock(&train_queue_mutex);
            if (vect_train_msg_t_push_back(&train_queue, train_msg) != 0) {
                fprintf(stderr, "Failed to push train message to queue\n");
                pthread_mutex_unlock(&train_queue_mutex);
               break;
            }
            pthread_mutex_unlock(&train_queue_mutex);

            // printf("Received train message for client %u, partition %u, model_version %d\n",
            //        train_msg.client_id, train_msg.partition_id, train_msg.model_version);
        } else {
            fprintf(stderr, "Unknown type: %u\n", type);
           break;
        }

        if (train_queue_advance() == -1) {
            fprintf(stderr, "Failed to advance train queue\n");
           break;
        }
    }

    printf("Receiver thread exiting...\n");

    running = 0;
}

void* t_data_sender(void *arg) {
    int sockfd = (int)(intptr_t)arg;

    while(running) {
        struct ptc_msg ptc_msg = {0};

        if (mq_receive(mq_in, (char*)&ptc_msg, PTC_MESSAGE_SIZE, NULL) == -1) {
            perror("mq_receive");
            break;
        }

        if (ptc_msg.err_code != 0) {
            fprintf(stderr, "Error in mq result: %d\n", ptc_msg.err_code);
            break;
        }

        uint32_t client_id = client_id_from_hash(ptc_msg.id);
        uint32_t partition_id = partition_id_from_hash(ptc_msg.id);
        void* result_ptr = get_result_ptr(ptc_msg.id);

        // printf("Received message from client %u, partition %u, hash_id = %lu\n",
        //        client_id, partition_id, ptc_msg.id);

        void * model_ptr = SHM.globals_ptr + ptc_msg.model_offset;
        void * results_ptr = SHM.results_ptr + ptc_msg.results_offset;

        if (memcmp(model_ptr, results_ptr, model_size) == 0) {
            fprintf(stderr, "This is strange, model and results are the same. there is some training error\n");
            break;
        }



        if (result_ptr == NULL) {
            fprintf(stderr, "No result pointer for client %u, partition %u\n", client_id, partition_id);
            break;
        }

        if (send_all(sockfd, &client_id, sizeof(client_id)) == -1) {
            perror("send");
            break;
        }

        if (send_all(sockfd, &partition_id, sizeof(partition_id)) == -1) {
            perror("send");
            break;
        }

        if (send_all(sockfd, result_ptr, model_size) == -1) {
            perror("send");
            break;
        }

        free_result(ptc_msg.id);
        if (train_queue_advance() == -1) {
            fprintf(stderr, "Failed to advance train queue\n");
            break;
        }

    }

    running = 0;
}

int real_main(void) {
    if(pthread_create(&thread_sender, NULL, t_data_sender, (void*)(intptr_t)sock) != 0) {
        thread_sender = (pthread_t)0;
        fprintf(stderr, "Failed to create sender thread\n");
        return EXIT_FAILURE;
    }

    if(pthread_create(&thread_receiver, NULL, t_data_receiver, (void*)(intptr_t)sock) != 0) {
        thread_receiver = (pthread_t)0;
        fprintf(stderr, "Failed to create receiver thread\n");
        return EXIT_FAILURE;
    }

    
    while (running) {
        sleep(1);
    }

    fflush(stdout);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <worker_id>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // parse worker_id it should be an uint32_t
    char *endptr;
    uint32_t worker_id = strtoul(argv[1], &endptr, 10);
    if (*endptr != '\0') {
        fprintf(stderr, "Invalid worker_id: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    // wait workerid seconds
    sleep(worker_id);

    atexit(clean_up);
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGQUIT, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    sock = host_connect("mamodeh.ddns.net", 6969);
    if (sock == -1) {
        fprintf(stderr, "Failed to connect to server\n");
        return EXIT_FAILURE;
    }

    if (sock_auth(sock, worker_id) == -1) {
        fprintf(stderr, "Failed to authenticate socket 1\n");
        return EXIT_FAILURE;
    }

    // from socket 1 receive model size and dataset size
    if (recv_all(sock, &model_size, sizeof(model_size)) == -1) {
        perror("recv");
        return EXIT_FAILURE;
    }

    if (recv_all(sock, &dataset_size, sizeof(dataset_size)) == -1) {
        perror("recv");
        return EXIT_FAILURE;
    }

    printf("Model size: %u\n", model_size);
    printf("Dataset size: %u\n", dataset_size);

    if (init_and_map_shm(model_size, dataset_size) == -1) {
        fprintf(stderr, "Failed to initialize and map shared memory\n");
        return EXIT_FAILURE;
    }

    printf("globals_ptr: %p\n", SHM.globals_ptr);
    printf("dataset_ptr: %p\n", SHM.dataset_ptr);
    printf("results_ptr: %p\n", SHM.results_ptr);

    if (init_mq_queues() == -1) {
        fprintf(stderr, "Failed to initialize message queues\n");
        return EXIT_FAILURE;
    }

    signal(SIGCHLD, SIG_IGN);
    if (init_workers() == -1) {
        fprintf(stderr, "Failed to initialize workers\n");
        return EXIT_FAILURE;
    }

    // load dataset header
    uint32_t dataset_header_size = 0;
    if (recv_all(sock, &dataset_header_size, sizeof(dataset_header_size)) == -1) {
        perror("recv");
        return EXIT_FAILURE;
    }

    if (dataset_header_size > dataset_size) {
        fprintf(stderr, "Dataset header size is larger than dataset size\n");
        return EXIT_FAILURE;
    }

    if (recv_all(sock, SHM.dataset_ptr, dataset_header_size) == -1) {
        perror("recv");
        return EXIT_FAILURE;
    }
   
    printf("Dataset header size: %u\n", dataset_header_size);
   
    return real_main();
}
