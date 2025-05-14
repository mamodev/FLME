import json
import os
import struct

# struct event_t {
#     uint8_t type;
#     uint32_t client_id;
#     uint32_t partition_id;
# };

# struct timeline_tick_t {
#     uint32_t event_count;
#     struct event_t *events;
# };

# struct simulation_t {
#     uint32_t n_partitions;
#     uint32_t n_aggregations;

#     uint32_t *aggregations;
#     uint32_t *clients_per_partition;

#     struct timeline_tick_t *timeline;

#     uint32_t n_ticks;
# }

EVNT_TYPE_MAP = {
    "fetch": 0,
    "train": 1,
    "send": 2,
}

tm_path = "../.timelines/ss.json"
with open(tm_path, 'r') as f:
    tm = json.load(f)
    f.close()


# print all

sim = tm["sim"]
n_partitions = sim["npartitions"]
aggregations = tm["aggregations"]
n_aggregations = len(aggregations)
clients_per_partition = sim["client_per_partition"]
timeline = tm["timeline"]
n_ticks = len(timeline)

buff = b''
buff += struct.pack('I', n_partitions)
buff += struct.pack('I', n_aggregations)
buff += struct.pack('I', n_ticks)

for cpp in clients_per_partition:
    buff += struct.pack('I', cpp)

for agg in aggregations:
    buff += struct.pack('I', agg)

for tick in timeline:
    event_count = len(tick)
    buff += struct.pack('I', event_count)
    for event in tick:
        event_type = EVNT_TYPE_MAP[event["type"]]
        client_id = event["client"][1]
        partition_id = event["client"][0]
        
        buff += struct.pack('B', event_type)
        buff += struct.pack('I', client_id)
        buff += struct.pack('I', partition_id)

buff = struct.pack('I', len(buff)) + buff
with open('timeline.bin', 'wb') as f:
    f.write(buff)
    f.close()








