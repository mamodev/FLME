#ifndef TIMELINE_H
#define TIMELINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>     // strncpy
#include <stdio.h>     // FILE, fopen, fread, fclose





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

    sim->clients_per_partition = (uint32_t *) malloc(sim->n_partitions * sizeof(uint32_t));
    sim->aggregations = (uint32_t *) malloc(sim->n_aggregations * sizeof(uint32_t));
    sim->timeline = (timeline_tick_t *) malloc(sim->n_ticks * sizeof(timeline_tick_t));
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
        sim->timeline[t].events = (event_t *)  malloc(ec * sizeof(event_t));
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





#endif