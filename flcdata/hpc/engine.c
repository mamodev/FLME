
// USAGE: <exect> <model.bin> <dataset.bin> <timeline.bin> [...<worker>]
// were worker is a string of type <type>:<id>
// example: "cpu:0" or "gpu:1"

// General STD imports
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     // strncpy
#include <stdint.h>     // uint32_t
#include <stdbool.h>
#include <signal.h>     // sig_atomic_t
#include <errno.h>
#include <sys/types.h>  // pid_t
#include <sys/wait.h>   // waitpid

// SHM related imports
#include <unistd.h>     // ftruncate, fork, execlp, sleep, _exit
#include <sys/mman.h>
#include <sys/stat.h>   // S_IRUSR, S_IWUSR
#include <mqueue.h>

#include "clib/timeline.h" // simulation_t

#define MAX_CONCURRENT_MODELS 1024
#define MAX_CONCURRENT_EVENTS 1024

#define MAX_WRK_MQ_MESSAGES 10
#define WRK_IN_QUEUE "/wrk_in"
#define WRK_OUT_QUEUE "/wrk_out"
#define MAX_WORKERS 30

volatile sig_atomic_t running = 1;

void signal_handler(int signum) {
    write(STDERR_FILENO, "Signal handler called\n", 22);
    running = 0;
}

void setup_signals() {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; // <--- NO SA_RESTART

    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);

    signal(SIGPIPE, SIG_IGN); // This is fine
}

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
   
    int32_t globals_allocations[MAX_CONCURRENT_MODELS];
    int64_t results_allocations[MAX_CONCURRENT_EVENTS];
};

#define MAX_SHAPE_LEN 16
#pragma pack(push, 1)
struct shape {
    uint32_t dim;
    uint32_t shape[MAX_SHAPE_LEN];
};
#pragma pack(pop)

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

#pragma pack(push, 1)
struct ptc_msg {
    uint64_t id;
    int32_t err_code;

    uint64_t model_offset;
    uint64_t results_offset;
};
#pragma pack(pop)

typedef char worker_id_t[64];

#define PTC_MESSAGE_SIZE  sizeof(struct ptc_msg)
#define CTP_MESSAGE_SIZE  sizeof(struct ctp_msg)

// GLOBAL VARIABLES
uint32_t model_size = 0;
uint32_t dataset_size = 0;
simulation_t simulation = {0};

u_int32_t num_workers = 0; 
worker_id_t worker_ids[MAX_WORKERS];
pid_t worker_pids[MAX_WORKERS];

mqd_t wrk_in = (mqd_t)-1, wrk_out = (mqd_t)-1;

struct shared_memory SHM = {
    .globals = -1,
    .dataset = -1,
    .results = -1,

    .globals_size = 0,
    .dataset_size = 0,
    .results_size = 0,


    .globals_ptr = NULL,
    .dataset_ptr = NULL,
    .results_ptr = NULL,

    .globals_allocations = {0},
    .results_allocations = {0},
};

void cleanup(void) {
    for (int i = 0; i < MAX_WORKERS; i++) {
        if (worker_pids[i] != -1) {
            kill(worker_pids[i], SIGKILL);
        }
    }

    for (int i = 0; i < MAX_WORKERS; i++) {
        if (worker_pids[i] != -1) {
            waitpid(worker_pids[i], NULL, 0);
            printf("Worker[%d] %s terminated\n", i, worker_ids[i]);
            worker_pids[i] = -1;
        }
    }

    if (wrk_in != (mqd_t)-1) {
        mq_close(wrk_in);
        mq_unlink(WRK_IN_QUEUE);
        wrk_in = (mqd_t)-1;
    }

    if (wrk_out != (mqd_t)-1) {
        mq_close(wrk_out);
        mq_unlink(WRK_OUT_QUEUE);
        wrk_out = (mqd_t)-1;
    }

    if (SHM.globals_ptr != NULL) {
        munmap(SHM.globals_ptr, SHM.globals_size);
        SHM.globals_ptr = NULL;
    }
    if (SHM.dataset_ptr != NULL) {
        munmap(SHM.dataset_ptr, SHM.dataset_size);
        SHM.dataset_ptr = NULL;
    }
    if (SHM.results_ptr != NULL) {
        munmap(SHM.results_ptr, SHM.results_size);
        SHM.results_ptr = NULL;
    }

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

// INITIALIZATION FUNCTIONS

int init_shm(void) {
    for (int i = 0; i < MAX_CONCURRENT_MODELS; i++) 
        SHM.globals_allocations[i] = -1;

    for (int i = 0; i < MAX_CONCURRENT_EVENTS; i++) 
        SHM.results_allocations[i] = -1;

        SHM.globals = shm_open("/globals", O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (SHM.globals == -1) {
        perror("shm_open");
        return -1;
    }

    SHM.dataset = shm_open("/dataset", O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (SHM.dataset == -1) {
        perror("shm_open");
        return -1;
    }

    SHM.results = shm_open("/results", O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (SHM.results == -1) {
        perror("shm_open");
        return -1;
    }

    return 0;
}

int allocate_shm(uint32_t model_size, uint32_t dataset_size) {
    SHM.dataset_size = dataset_size;
    SHM.globals_size = model_size * MAX_CONCURRENT_MODELS;
    SHM.results_size = model_size * MAX_CONCURRENT_EVENTS;

    if (ftruncate(SHM.globals, model_size * MAX_CONCURRENT_MODELS) == -1) {
        perror("ftruncate globals");
        return -1;
    }

    if (ftruncate(SHM.dataset, dataset_size) == -1) {
        perror("ftruncate dataset");
        return -1;
    }

    if (ftruncate(SHM.results, model_size * MAX_CONCURRENT_EVENTS) == -1) {
        perror("ftruncate results");
        return -1;
    }
    
    return 0;
}

int mmap_shm(uint32_t model_size, uint32_t dataset_size) {
    
    SHM.globals_ptr = mmap(NULL, model_size * MAX_CONCURRENT_MODELS,
                           PROT_READ | PROT_WRITE,
                           MAP_SHARED, SHM.globals, 0);

    if (SHM.globals_ptr == MAP_FAILED) {
        SHM.globals_ptr = NULL;
        perror("mmap globals");
        return -1;
    }

    SHM.dataset_ptr = mmap(NULL, dataset_size,
                           PROT_READ | PROT_WRITE,
                           MAP_SHARED, SHM.dataset, 0);

    if (SHM.dataset_ptr == MAP_FAILED) {
        SHM.dataset_ptr = NULL;
        perror("mmap dataset");
        return -1;
    }

    SHM.results_ptr = mmap(NULL, model_size * MAX_CONCURRENT_EVENTS,
                           PROT_READ | PROT_WRITE,
                           MAP_SHARED, SHM.results, 0);

    if (SHM.results_ptr == MAP_FAILED) {
        SHM.results_ptr = NULL;
        perror("mmap results");
        return -1;
    }

    return 0;
}

int init_mq_queues (void) {
    struct mq_attr attr_in = {
        .mq_flags   = 0,
        .mq_maxmsg  = MAX_WRK_MQ_MESSAGES,
        .mq_msgsize = PTC_MESSAGE_SIZE,
        .mq_curmsgs = 0
    };

    struct mq_attr attr_out = {
        .mq_flags   = 0,
        .mq_maxmsg  = MAX_WRK_MQ_MESSAGES,
        .mq_msgsize = CTP_MESSAGE_SIZE,
        .mq_curmsgs = 0
    };

    wrk_in = mq_open(WRK_IN_QUEUE,
                    O_CREAT | O_RDWR,
                    S_IRUSR | S_IWUSR,
                    &attr_in);

    if (wrk_in == (mqd_t)-1) {
        perror("mq_open :: wrk_in");
        return -1;
    }

    wrk_out = mq_open(WRK_OUT_QUEUE,
                     O_CREAT | O_RDWR,
                     S_IWUSR | S_IRUSR,
                     &attr_out);

    if (wrk_out == (mqd_t)-1) {
        perror("mq_open :: wrk_out");
        return -1;
    }

    return 0;
}

int init_channels(void) {
    if (model_size == 0 || dataset_size == 0) {
        fprintf(stderr, "Model size or dataset size is zero\n");
        return -1;
    }

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

    if (init_mq_queues() == -1) {
        fprintf(stderr, "Failed to initialize message queues\n");
        return -1;
    }

    return 0;
}

int load_sizes(const char *model, const char *ds) {
    int model_fd = open(model, O_RDONLY);
    if (model_fd == -1) {
        perror("open model");
        return -1;
    }

    if (read(model_fd, &model_size, sizeof(model_size)) != sizeof(model_size)) {
        perror("read model size");
        close(model_fd);
        return -1;
    }

    int dataset_fd = open(ds, O_RDONLY);
    if (dataset_fd == -1) {
        perror("open dataset");
        close(model_fd);
        return -1;
    }

    if (read(dataset_fd, &dataset_size, sizeof(dataset_size)) != sizeof(dataset_size)) {
        perror("read dataset size");
        close(model_fd);
        close(dataset_fd);
        return -1;
    }

    model_size += sizeof(uint32_t); //and uint32_t size
    dataset_size += sizeof(uint32_t); //and uint32_t size

    close(model_fd);
    close(dataset_fd);

    return 0;
}

int load_data_into_shm(const char *model, const char *ds) {
    int model_fd = open(model, O_RDONLY);
    if (model_fd == -1) {
        perror("open model");
        return -1;
    }

    if (lseek(model_fd, 0, SEEK_SET) == -1) {
        perror("lseek model");
        close(model_fd);
        return -1;
    }
    if (read(model_fd, SHM.globals_ptr, model_size) != model_size) {
        perror("read model");
        close(model_fd);
        return -1;
    }

    int dataset_fd = open(ds, O_RDONLY);
    if (dataset_fd == -1) {
        perror("open dataset");
        close(model_fd);
        return -1;
    }

    if (lseek(dataset_fd, 0, SEEK_SET) == -1) {
        perror("lseek dataset");
        close(dataset_fd);
        return -1;
    }
    if (read(dataset_fd, SHM.dataset_ptr, dataset_size) != dataset_size) {
        perror("read dataset");
        close(dataset_fd);
        return -1;
    }

    SHM.globals_allocations[0] = 1;

    close(model_fd);
    close(dataset_fd);

    return 0;
}

pid_t spawn_worker(worker_id_t device) {

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
            "--device", device,
            "--in-queue", WRK_OUT_QUEUE,
            "--out-queue", WRK_IN_QUEUE,
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

int parse_args(int argc, char *argv[], char **model, char **ds, char **timeline) {
    for (int i = 0; i < MAX_WORKERS; i++) {
        worker_pids[i] = -1;
        memset(worker_ids[i], 0, sizeof(worker_id_t));
    }

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <modelpath> <datasetpath> <timelinepath> [<worker_id>...]\n", argv[0]);
        return -1;
    }

    num_workers = argc - 4;

    if (num_workers > MAX_WORKERS) {
        fprintf(stderr, "Too many worker ids, max is 10\n");
        return -1;
    }

    for (int i = 4; i < argc; i++) {
        if (strlen(argv[i]) > sizeof(worker_id_t) - 1) {
            fprintf(stderr, "Worker id too long: %s\n", argv[i]);
            return -1;
        }
        strncpy(worker_ids[i - 4], argv[i], sizeof(worker_id_t));
    }

    *model = argv[1];
    *ds = argv[2];
    *timeline = argv[3];

    return 0;
}


struct fetch_event {
    uint32_t client_id;
    uint32_t partition_id;
    uint32_t gmodel_version;
};

uint64_t hash_client(uint32_t client_id, uint32_t partition_id) {
    return (uint64_t)client_id << 32 | (uint64_t)partition_id;
}

int find_gmodel(uint32_t version) {
    for (int i = 0; i < MAX_CONCURRENT_MODELS; i++) {
        if (SHM.globals_allocations[i] == version) {
            return i;
        }
    }
    return -1;
}

void *get_gmodel_ptr(uint32_t version) {
    for (int i = 0; i < MAX_CONCURRENT_MODELS; i++) {
        if (SHM.globals_allocations[i] == version) {
            return SHM.globals_ptr + i * model_size;
        }
    }
    return NULL;
}

void* next_free_gmodel(int32_t version) {
    for (int i = 0; i < MAX_CONCURRENT_MODELS; i++) {
        if (SHM.globals_allocations[i] == version) {
            perror("next_free_gmodel :: already allocated");
            return NULL;
        }
        if (SHM.globals_allocations[i] == -1) {
            SHM.globals_allocations[i] = version;
            return SHM.globals_ptr + i * model_size;
        }
    }

    perror("next_free_gmodel :: no free slots");
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

void free_result(int64_t version) {
    for (int i = 0; i < MAX_CONCURRENT_EVENTS; i++) {
        if (SHM.results_allocations[i] == version) {
            SHM.results_allocations[i] = -1;
            return;
        }
    }
}

int get_partition_slice(uint32_t client_id, uint32_t partition_id,
    uint64_t *data_offset, struct shape *data_shape,
    uint64_t *targets_offset, struct shape *targets_shape) {
 
    uint32_t n_partitions = simulation.n_partitions;
    uint32_t n_clients = simulation.clients_per_partition[partition_id];

    if (client_id > n_clients) {
        fprintf(stderr, "Client id %u is out of range [0, %u)\n", client_id, n_clients);
        return -1;
    }

    if (partition_id > n_partitions) {
        fprintf(stderr, "Partition id %u is out of range [0, %u)\n", partition_id, n_partitions);
        return -1;
    }        

    uint32_t ds_partitions = ((uint32_t*)SHM.dataset_ptr)[1];
    if (ds_partitions != n_partitions) {
        fprintf(stderr, "Dataset partitions %u is different from simulation partitions %u\n", ds_partitions, n_partitions);
        return -1;
    }

    uint32_t header_len = sizeof(uint32_t) * 4;
    uint32_t prev_partition_len = 0;

    void *ds = SHM.dataset_ptr;
    ds += header_len;

    size_t dsize[2] = { sizeof(float), sizeof(uint64_t) };

    uint64_t debug_size = header_len;
    
    for (uint32_t p = 0; p < n_partitions; p++) {
        // printf("Partition %u\n", p);
        for (int type = 0; type < 2; type++) {
            uint32_t acc = 1;
            uint32_t dim = ((uint32_t*)ds)[0];
            
            if (dim > MAX_SHAPE_LEN) {
                // fprintf(stderr, "Dimension %u is out of range [0, %u]\n", dim, MAX_SHAPE_LEN);
                return -1;
            }
            
            ds += sizeof(uint32_t);

            if (p == partition_id && type == 0)
                data_shape->dim = dim;
            else if (p == partition_id && type == 1)
                targets_shape->dim = dim;

            // printf("\tType %u, Dim %u\n",  type, dim);
            
            for (uint32_t i = 0; i < dim; i++) {
                
                // printf("\t\tType %u, Dim %u: %u\n", type, i, ((uint32_t*)ds)[i]);
                // fflush(stdout);
                acc *= ((uint32_t*)ds)[i];
                if (p == partition_id && type == 0)
                    data_shape->shape[i] = ((uint32_t*)ds)[i];
                else if (p == partition_id && type == 1)
                    targets_shape->shape[i] = ((uint32_t*)ds)[i];
            }

            if (p < partition_id) {
                prev_partition_len += acc * dsize[type];
            }

            debug_size += acc * dsize[type] + sizeof(uint32_t) + dim * sizeof(uint32_t);
  
            ds += sizeof(uint32_t) * dim;
        }
    }

    if (debug_size != SHM.dataset_size) {
        fprintf(stderr, "Debug size %lu is different from dataset size %lu\n", debug_size, SHM.dataset_size);
        return -1;
    }

    header_len = (uint32_t)(ds - SHM.dataset_ptr);

    uint32_t psamples = targets_shape->shape[0];
    uint32_t samples_per_client = psamples / n_clients;

    if (samples_per_client == 0) {
        fprintf(stderr, "Samples per client is zero choose a bigger dataset or different partitioning\n");
        return -1;
    }


    uint32_t sample_dim = 1;
    for (uint32_t i = 1; i < data_shape->dim; i++) {
        sample_dim *= data_shape->shape[i];
    }

    uint32_t target_dim = 1;
    for (uint32_t i = 1; i < targets_shape->dim; i++) {
        target_dim *= targets_shape->shape[i];
    }

    targets_shape->shape[0] = samples_per_client;
    data_shape->shape[0] = samples_per_client;

    uint32_t target_bytes = target_dim * dsize[1];
    uint32_t sample_bytes = sample_dim * dsize[0];
    uint32_t poffset = header_len + prev_partition_len;

    // uint32_t current_partition_prev_client_d_len = samples_per_client * client_id * sample_bytes;
    // uint32_t current_partition_prev_client_t_len = samples_per_client * client_id * sample_bytes;
    // uint32_t current_partition_data_len = psamples * sample_dim * dsize[0];

    *data_offset = poffset + (client_id * samples_per_client * sample_bytes);
    *targets_offset = poffset + (psamples * sample_bytes) + (client_id * samples_per_client * target_bytes);

    // printf("Partition %u, Client %u\n", partition_id, client_id);
    // printf("\tpsamples: %u, samples_per_client: %u\n", psamples, samples_per_client);
    // printf("\tTarget DIM: %u, Sample DIM: %u\n", target_dim, sample_dim);
    // printf("\tData shape: ");
    // for (uint32_t i = 0; i < data_shape->dim; i++) {
    //     printf("%u ", data_shape->shape[i]);
    // }
    // printf("\n");
    // printf("\tTargets shape: ");
    // for (uint32_t i = 0; i < targets_shape->dim; i++) {
    //     printf("%u ", targets_shape->shape[i]);
    // }
    // printf("\n");
    // printf("\tHeader len: %u\n", header_len);
    // printf("\tPartition Start: %u, Partition End: %u\n", poffset, poffset + psamples * (sample_bytes + target_bytes));
    // printf("\tData (%u, %u), Targets (%u, %u)\n", poffset, poffset + psamples * sample_bytes, poffset + psamples * sample_bytes, poffset + psamples * (sample_bytes + target_bytes));
    // printf("\tData offset: %lu, Targets offset: %lu\n", *data_offset, *targets_offset);
    // fflush(stdout);


    // add some assertions
    if (*data_offset + samples_per_client * sample_bytes > SHM.dataset_size) {
        fprintf(stderr, "Data offset %lu + data len %u > dataset size %lu\n", *data_offset, sample_dim * samples_per_client * dsize[0], SHM.dataset_size);
        return -1;
    }

    if (*targets_offset + samples_per_client * target_bytes  > SHM.dataset_size) {
        fprintf(stderr, "Targets offset %lu + targets len %u > dataset size %lu\n", *targets_offset, target_dim * samples_per_client * dsize[1], SHM.dataset_size);
        return -1;
    }   
    
    return 0;
}


ssize_t get_model_header_size(void * model_ptr) {
    void* __save = model_ptr;
    int32_t size = *(int32_t*)model_ptr;
    model_ptr += sizeof(int32_t);
    int32_t nlayers = *(int32_t*)model_ptr;
    model_ptr += sizeof(int32_t);

    for (int i = 0; i < nlayers; i++) {
        int32_t keylen = *(int32_t*)model_ptr;
        model_ptr += sizeof(int32_t);

        model_ptr += keylen;
        
        int32_t shape_len = *(int32_t*)model_ptr;
        model_ptr += sizeof(int32_t);
        model_ptr += shape_len * sizeof(int32_t);
    }

    return model_ptr - __save;
}


#define MAX_FETCH_EVENTS 1024
int sim_loop(void) {

    printf("Simulation loop started\n");

    ssize_t model_layers_offset = get_model_header_size(SHM.globals_ptr);

    int t = 0;
    uint32_t gmodel_version = 1;
    uint32_t next_agg = simulation.aggregations[0];

    struct fetch_event fetch_events[MAX_FETCH_EVENTS] = {0};
    memset(fetch_events, 0, sizeof(fetch_events));

    uint32_t pending_updates_count = 0;

    while (t < simulation.n_ticks && running) {
        timeline_tick_t *tick = &simulation.timeline[t];
        for (uint32_t e = 0; e < tick->event_count; e++) {
            event_t *event = &tick->events[e];

            if (event->type == 0) {
                //fetch
                int i = 0;
                while(i < MAX_FETCH_EVENTS && fetch_events[i].gmodel_version) i++;
                if (i >= MAX_FETCH_EVENTS || fetch_events[i].gmodel_version) {
                    perror("fetch_events :: not enough space");
                    goto end;
                }

                fetch_events[i].client_id = event->client_id;
                fetch_events[i].partition_id = event->partition_id;
                fetch_events[i].gmodel_version = gmodel_version;

            } else if (event->type == 1) {
                //train
                // ignore for semplicity
            } else if (event->type == 2) {

                int i = 0;
                while(i < MAX_FETCH_EVENTS 
                        && (!fetch_events[i].gmodel_version
                        || fetch_events[i].client_id != event->client_id 
                        || fetch_events[i].partition_id != event->partition_id)) i++;
                
                if (i >= MAX_FETCH_EVENTS || 
                        !fetch_events[i].gmodel_version ||
                        fetch_events[i].client_id != event->client_id || 
                        fetch_events[i].partition_id != event->partition_id) {
                    perror("event: send, fetch_events :: not found");
                    return -1;
                }


                void *model_ptr = get_gmodel_ptr(fetch_events[i].gmodel_version);
                if (model_ptr == NULL) {
                    perror("event: send, get_gmodel_ptr :: not found");
                    return -1;
                }

                void *results_ptr = next_free_result(hash_client(event->client_id, event->partition_id));
                if (results_ptr == NULL) {
                    perror("event: send, next_free_result :: not found");
                    return -1;
                }

                fetch_events[i].gmodel_version = 0; //FREE THE SLOT

                struct ctp_msg ctp_msg = {0};
                ctp_msg.id = hash_client(event->client_id, event->partition_id);
                ctp_msg.model_offset = (uint64_t)(model_ptr - SHM.globals_ptr);
                ctp_msg.results_offset = (uint64_t)(results_ptr - SHM.results_ptr);

                ctp_msg.model_size = model_size;

                if (get_partition_slice(event->client_id, event->partition_id,
                        &ctp_msg.data_offset, &ctp_msg.data_shape,
                        &ctp_msg.targets_offset, &ctp_msg.targets_shape) == -1) {
                    perror("event: send, get_partition_slice :: not found");
                    return -1;
                }
              
                ctp_msg.ephochs = 20;
                ctp_msg.batch_size = 1024;
                ctp_msg.learning_rate = 0.01;
                ctp_msg.momentum = 0.9;
                ctp_msg.weight_decay = 0.0001;
                ctp_msg.shuffle = false;

                printf("Sending message to worker %u:%u\n", event->client_id, event->partition_id);

                if (mq_send(wrk_out, (const char*)&ctp_msg, CTP_MESSAGE_SIZE, 0) == -1) {
                    perror("mq_send");
                    return -1;
                }

                pending_updates_count++;
            } 
        }

        if (t == next_agg) {

            gmodel_version++;

            void *new_gmodel_ptr = next_free_gmodel(gmodel_version);
            if (new_gmodel_ptr == NULL) {
                goto end;
            }

            
            memcpy(new_gmodel_ptr, SHM.globals_ptr, model_layers_offset);
           	
	        printf("Waiting for %u models", pending_updates_count++);

            for (int u = 0; u < pending_updates_count; u++) {
                struct ptc_msg ptc_msg = {0};
                if (mq_receive(wrk_in, (char*)&ptc_msg, PTC_MESSAGE_SIZE, NULL) == -1) {
                    perror("mq_receive");
                    goto end;
                }

                uint64_t id = ptc_msg.id;
                int32_t err_code = ptc_msg.err_code;
                
                for (int i =  model_layers_offset; i < model_size; i++) {
                    ((float*)new_gmodel_ptr)[i] += ((float*)SHM.results_ptr)[i];
                }   

                free_result(id);

                if (err_code != 0) {
                    fprintf(stderr, "Error in worker: %d\n");
                    continue;
                }
            }

            for (int i =  model_layers_offset; i < model_size; i++) {
                ((float*)new_gmodel_ptr)[i] /= pending_updates_count;
            }

            pending_updates_count = 0;
     

            // should aggregate the models



            if (gmodel_version > simulation.n_aggregations) {
                break;  
            }

            next_agg = simulation.aggregations[gmodel_version - 1];
        }

        t++;
    }

end:
    printf("Simulation loop terminated t: [%d/%d]\n", t, simulation.n_ticks);
    return 0;
}



int main(int argc, char *argv[]) {
    char *model = NULL, *ds = NULL, *timeline = NULL;
    if (parse_args(argc, argv, &model, &ds, &timeline) == -1) {
        fprintf(stderr, "Failed to parse arguments\n");
        return -1;

    }

    setup_signals();
    atexit(cleanup);

    if (read_simulation(timeline, &simulation) == -1) {
        fprintf(stderr, "Failed to read simulation\n");
        return -1;
    }

    if (load_sizes(model, ds) == -1) {
        fprintf(stderr, "Failed to load sizes\n");
        return -1;
    }

    if (init_channels() == -1) {
        fprintf(stderr, "Failed to initialize channels\n");
        return -1;
    }

    if (load_data_into_shm(model, ds) == -1) {
        fprintf(stderr, "Failed to load data into shared memory\n");
        return -1;
    }

    for (int i = 0; i < num_workers; i++) {
        worker_pids[i] = spawn_worker(worker_ids[i]);
        if (worker_pids[i] == -1) {
            fprintf(stderr, "Failed to spawn worker %s\n", worker_ids[i]);
            return -1;
        }
    }

    return sim_loop();
}
