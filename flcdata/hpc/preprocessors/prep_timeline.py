def preprocess_timeline(path: str, out_path: str, verbose: bool = False):
    import os
    import json
    import struct

    assert type(path) == str and path != "", f"Path {path} is not a valid string"
    assert path.endswith(".json"), f"Path {path} is not a json file"
    assert os.path.exists(path), f"Path {path} does not exist"
    assert os.path.isfile(path), f"Path {path} is not a file"

    assert type(out_path) == str and out_path != "", f"Output path {out_path} is not a valid string"
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    assert os.path.exists(out_path), f"Output path {out_path} does not exist"
    assert os.path.isdir(out_path), f"Output path {out_path} is not a directory"

    EVNT_TYPE_MAP = {
        "fetch": 0,
        "train": 1,
        "send": 2,
    }

    with open(path, 'r') as f:
        tm = json.load(f)
        f.close()

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


    if verbose:
        print("timeline size: ", len(buff))

    buff = struct.pack('I', len(buff)) + buff
    with open(os.path.join(out_path, "timeline.bin"), 'wb') as f:
        f.write(buff)
        f.close()








