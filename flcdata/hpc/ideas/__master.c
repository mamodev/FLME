#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     // strncpy
#include <unistd.h>     // ftruncate, fork, execlp, sleep, _exit
#include <stdint.h>

// Network
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <signal.h>

volatile int running = 1;

void sigint_handler(int signum) {
    running = 0;
}

typedef struct {
    uint8_t type;
    uint32_t client_id;
    uint32_t partition_id;
} event_t;

typedef struct {
    uint32_t event_count;
    event_t *events;
} timeline_tick_t;

typedef struct {
    uint32_t n_partitions;
    uint32_t n_aggregations;
    uint32_t n_ticks;

    uint32_t *clients_per_partition; // size: n_partitions
    uint32_t *aggregations;          // size: n_aggregations
    timeline_tick_t *timeline;       // size: n_ticks
} simulation_t;

void free_simulation(simulation_t *sim) {
    if (!sim) return;
    if (sim->clients_per_partition) free(sim->clients_per_partition);
    if (sim->aggregations) free(sim->aggregations);
    if (sim->timeline) {
        for (uint32_t i = 0; i < sim->n_ticks; ++i) {
            if (sim->timeline[i].events) free(sim->timeline[i].events);
        }
        free(sim->timeline);
    }
}

int read_simulation(const char *filename, simulation_t *sim) {
    FILE *f = fopen(filename, "rb");
    if (!f) return -1;

    uint32_t total_len;
    if (fread(&total_len, sizeof(uint32_t), 1, f) != 1) goto fail;

    if (fread(&sim->n_partitions, sizeof(uint32_t), 1, f) != 1) goto fail;
    if (fread(&sim->n_aggregations, sizeof(uint32_t), 1, f) != 1) goto fail;
    if (fread(&sim->n_ticks, sizeof(uint32_t), 1, f) != 1) goto fail;

    sim->clients_per_partition = malloc(sim->n_partitions * sizeof(uint32_t));
    sim->aggregations = malloc(sim->n_aggregations * sizeof(uint32_t));
    sim->timeline = malloc(sim->n_ticks * sizeof(timeline_tick_t));
    if (!sim->clients_per_partition || !sim->aggregations || !sim->timeline) goto fail;

    for (uint32_t i = 0; i < sim->n_partitions; ++i) {
        if (fread(&sim->clients_per_partition[i], sizeof(uint32_t), 1, f) != 1) goto fail;
    }
    for (uint32_t i = 0; i < sim->n_aggregations; ++i) {
        if (fread(&sim->aggregations[i], sizeof(uint32_t), 1, f) != 1) goto fail;
    }

    for (uint32_t t = 0; t < sim->n_ticks; ++t) {
        if (fread(&sim->timeline[t].event_count, sizeof(uint32_t), 1, f) != 1) goto fail;
        uint32_t ec = sim->timeline[t].event_count;
        sim->timeline[t].events = malloc(ec * sizeof(event_t));
        if (!sim->timeline[t].events && ec > 0) goto fail;
        for (uint32_t e = 0; e < ec; ++e) {
            if (fread(&sim->timeline[t].events[e].type, sizeof(uint8_t), 1, f) != 1) goto fail;
            if (fread(&sim->timeline[t].events[e].client_id, sizeof(uint32_t), 1, f) != 1) goto fail;
            if (fread(&sim->timeline[t].events[e].partition_id, sizeof(uint32_t), 1, f) != 1) goto fail;
        }
    }

    fclose(f);
    return 0;

fail:
    fclose(f);
    free_simulation(sim);
    return -1;
}

void main_loop() {
    simulation_t sim = {0};
    if (read_simulation("timeline.bin", &sim) != 0) {
        fprintf(stderr, "Failed to read timeline.bin\n");
    }

    uint32_t global_model_version = 1;
    uint32_t next_agg = sim.aggregations[0];

    for (uint32_t t = 0; t < sim.n_ticks; ++t) {
        for (uint32_t e = 0; e < sim.timeline[t].event_count; ++e) {
            event_t *event = &sim.timeline[t].events[e];
            switch (event->type) {
                case 0: // fetch event
                    fetch_client(event->client_id, event->partition_id, global_model_version);
                    break;
                case 1: // train event
                    train_client(event->client_id, event->partition_id, global_model_version);
                    break;
                case 2: // send event
                    break;
                default:
                    fprintf(stderr, "Unknown event type: %u\n", sim.timeline[t].events[e].type);
                    break;
            }

        }

        if (t <= next_agg) {
            global_model_version++;

            if (global_model_version >= sim.n_aggregations) {
                break;
            }

            next_agg = sim.aggregations[global_model_version];
        }
    }

    printf("Final global model version: %u\n", global_model_version);


 
 
    free_simulation(&sim);
}

void *client_handler(void *arg) {
    int client_fd = (int)(intptr_t)arg;
    close(client_fd);
    return NULL;
}

#define MAX_CLIENTS 1024


int main(void) {
    main_loop();
    return 0;

    int server_fd;

    signal(SIGINT, sigint_handler);

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(6969);

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
        perror("setsockopt");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)) == -1) {
        perror("setsockopt");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("bind");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, SOMAXCONN) == -1) {
        perror("listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    
    while (running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &addr_len);
        if (client_fd == -1) {
            perror("accept");
            continue;
        }


        uint32_t type;
        if (recv(client_fd, &type, sizeof(type), 0) == -1) {
            perror("recv");
            close(client_fd);
            continue;
        }
        

      
    }

    close(server_fd);
}