




# ensure_data_on_node(ds_name, partition, client_idx)

# ensure_global_model_on_node(ds_name, partition, model_version)

# fut_model = train_on_node(ds_name, partition, client_idx, model_version)


import asyncio
import socket
import time
import argparse
import sys
import numpy as np
import struct
from typing import Union, Any, Dict, List, Tuple
import json
import torch    
import io
import hashlib


from collections import defaultdict

class SmartScheduler:
    def __init__(self):
        self.worker_versions = defaultdict(set)
        self.worker_clients = defaultdict(set)
        self.worker_load = defaultdict(int)
        self.workers = set()

    def update_workers(self, workers):
        """Update the set of available workers."""
        self.workers = set(workers)
        for w in list(self.worker_versions):
            if w not in self.workers:
                del self.worker_versions[w]
                del self.worker_clients[w]
                del self.worker_load[w]
    
    def update_workers_versions(self, toKeep):
        for w in self.workers:
            for v in list(self.worker_versions[w]):
                if v not in toKeep:
                    self.worker_versions[w].discard(v)
         
  

    def mark_assignment(self, worker, client, version):
        """Mark that a client is assigned to a worker for a version."""
        self.worker_clients[worker].add(client)
        self.worker_versions[worker].add(version)

    async def release_load(self, worker):
        self.worker_load[worker] -= 1

    async def pick_node(self, client, version):
        while len(self.workers) == 0:
            print("[WARN] No workers available, waiting...")
            await asyncio.sleep(1)

        # 1. Prefer workers that have the client and the version
        candidates = [
            w for w in self.workers
            if client in self.worker_clients[w] and version in self.worker_versions[w]
        ]

        if not candidates:
            # 2. Prefer workers that have the client
            candidates = [
                w for w in self.workers
                if client in self.worker_clients[w]
            ]

        # if not candidates:
        #     # 3. Prefer workers that have the version
        #     candidates = [
        #         w for w in self.workers
        #         if version in self.worker_versions[w]
        #     ]
      
        if not candidates:
            # 4. Fallback: any worker
            candidates = list(self.workers)

        # 5. Among candidates, pick the one with the lowest load
        min_load = min(self.worker_load[w] for w in candidates)
        best_workers = [w for w in candidates if self.worker_load[w] == min_load]
        np.random.shuffle(best_workers)
        chosen = best_workers[0]
        self.mark_assignment(chosen, client, version)
        self.worker_load[chosen] += 1

        return chosen
    
class MemoryDataLoader:
    def __init__(
        self, data, targets, batch_size=1024, shuffle=False,
        n_partitions=1, partition_idx=0
    ):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_partitions = n_partitions
        self.partition_idx = partition_idx

        # Calculate partition indices
        total_len = len(self.data)
        part_size = total_len // n_partitions
        remainder = total_len % n_partitions

        # Compute start and end indices for this partition
        self.start_idx = partition_idx * part_size + min(partition_idx, remainder)
        self.end_idx = self.start_idx + part_size
        if partition_idx < remainder:
            self.end_idx += 1

        # Slice the data for this partition
        self.partition_data = self.data[self.start_idx:self.end_idx]
        self.partition_targets = self.targets[self.start_idx:self.end_idx]

    def __iter__(self):
        indices = list(range(len(self.partition_data)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long)
            yield (
                self.partition_data[batch_indices],
                self.partition_targets[batch_indices]
            )
            
    def __len__(self):
        return len(self.partition_data)

__SCHEDULER__ = SmartScheduler()

__WORKERS__ = {}

def chash(client):
    return f"{client[0]}_{client[1]}"

# async def pick_node(client, version):
#     wkeys = list(workers.keys())

#     while len(wkeys) == 0:
#         print("[WARN] No workers available, waiting...")
#         await asyncio.sleep(1)
#         wkeys = list(workers.keys())

#     np.random.shuffle(wkeys)
#     return wkeys[0]


async def drop_old_models(node, models_to_keep):
    payload = {
        "type": "drop_old_models",
        "models_to_keep": models_to_keep,
    }

    await worker_packet(node, payload)
  

async def ensure_file_on_node(node, local_path, remote_path):
    # check if file exists locally
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File {local_path} not found")


    sha256 = hashlib.sha256()
    with open(local_path, "rb") as f:
        while True:
            data = f.read(1024)
            if not data:
                break
            sha256.update(data)

    sha256_hash = sha256.hexdigest()

    payload = {
        "type": "ensure-file",
        "path": remote_path,
        "hash": sha256_hash,
    }

    res = await worker_packet(node, payload)
    if res == b"1":
        return
    
    # read all data from file

    with open(local_path, "rb") as f:
        data = f.read()

    await worker_packet(node, {
        "type": "dowload-file",
        "path": remote_path,
        "data": data,
    })

async def ensure_data_on_node(node, ds_name, partition, client_idx, fallbackData):
    payload = {
        "type": "ensure_data",
        "ds_name": ds_name,
        "partition": partition,
        "client_idx": client_idx,
    }

    res = await worker_packet(node, payload)
    if res == b"1":
        return 

    await worker_packet(node, {
        "type": "download-data",
        "ds_name": ds_name,
        "partition": partition,
        "client_idx": client_idx,
        "targets": fallbackData.targets,
        "data": fallbackData.data,
    })

    # print(f"Sent data to node {node} for client {client_idx} on partition {partition}")


async def ensure_global_model_on_node(node, model_version, fallbackData):
    payload = {
        "type": "ensure_global_model",
        "model_version": model_version,
    }

    res = await worker_packet(node, payload)
    if res == b"1":
        return 
    
    res = await worker_packet(node, fallbackData, ignoreSerialize=True)

async def train_on_node(node, ds_name, partition, client_idx, model_version, 
                            insize, outsize, lr, bs, momentum, epochs, shuffle):
    payload = {
        "type": "train",
        "ds_name": ds_name,
        "partition": partition,
        "client_idx": client_idx,
        "model_version": model_version,
        "insize": insize,
        "outsize": outsize,
        "learning_rate": lr,
        "batch_size": bs,
        "momentum": momentum,
        "ephocs": epochs,
        "shuffle": shuffle,
    }

    m = await worker_packet(node, payload)
    res = torch.load(io.BytesIO(m))
    return (res["model"], res["meta"])


NODE_TIMES = dict()

async def remote_fetch(params):
    if params["node"] not in NODE_TIMES:
        NODE_TIMES[params["node"]] = {
            "train": {
                "avg": 0,
                "count": 0,
            },
            "ensure_data": {
                "avg": 0,
                "count": 0,
            },
            "ensure_model":  {
                "avg": 0,
                "count": 0,
            },
            "serialization": {
                "avg": 0,
                "count": 0,
            },
        }

    # await ensure_file_on_node(
    #     params["node"],
    #     "./lib/flcdata.py",
    #     "./lib/flcdata.py",
    # )

    # print(f"Ensuring data on node {params['node']} for client {params['client']}")
    start_time = time.time()
    await ensure_data_on_node(
        params["node"],
        params["ds_folder"],
        params["partition"],
        params["client"],
        params["train_loader"],
    )
    end_time = time.time()
    NODE_TIMES[params["node"]]["ensure_data"]["avg"] += (end_time - start_time)
    NODE_TIMES[params["node"]]["ensure_data"]["count"] += 1

    # print(f"Ensuring global model on node {params['node']} for version {params['version']}")

    start_time = time.time()
    await ensure_global_model_on_node(
        params["node"],
        params["version"],
        params["global_model"],
    )
    end_time = time.time()
    NODE_TIMES[params["node"]]["ensure_model"]["avg"] += (end_time - start_time)
    NODE_TIMES[params["node"]]["ensure_model"]["count"] += 1


    return params

async def remote_train(params):
    # print(f"Training on node {params['node']} for client {params['client']}")

    start_time = time.time()
    res = await train_on_node(
        params["node"],
        params["ds_folder"],
        params["partition"],
        params["client"],
        params["version"],
        params["insize"],
        params["outsize"],
        params["lr"],
        params["batch_size"],
        params["momentum"],
        params["ephocs"],
        params["shuffle"],
    )
    end_time = time.time()

    NODE_TIMES[params["node"]]["train"]["avg"] += (end_time - start_time)
    NODE_TIMES[params["node"]]["train"]["count"] += 1


    await __SCHEDULER__.release_load(params["node"])
    return res



def aggregate_model(_mdl, updates):
    ref = updates[0][0]
    total_samples = sum([meta["train_samples"] for _, meta in updates]) 

    model = {k: sum([m[k] * meta["train_samples"] for m, meta in updates]) / total_samples for k in ref.keys()}

    return model, {
            "contributors": [
                {
                    "meta": meta,
                }
                for m, meta in updates
            ]
        }


async def run(args):
    sim, timeline, aggregations, ds_folder, repo_folder, args = args

    proportional_knowledge = "proportionalKnowledge" in sim and sim["proportionalKnowledge"]
    splt_per_partition = sim["client_per_partition"]
    if proportional_knowledge:
        max_client_pp = max(splt_per_partition)
        for pidx, _ in enumerate(sim["client_per_partition"]):
            splt_per_partition[pidx] = max_client_pp


    from lib.flcdata import FLCDataset, Model
    in_size, out_size = FLCDataset.LoadSize(ds_folder)
    train_dss = FLCDataset.LoadGroups(ds_folder=ds_folder, train=True)
    train_loaders = [
        [
            MemoryDataLoader(
                partition_ds.data,
                partition_ds.targets,
                batch_size=2048,
                shuffle=True,
                n_partitions=splt_per_partition[pidx],
                partition_idx=cidx,
            )
            for cidx in range(sim["client_per_partition"][pidx])
        ]
        for pidx, partition_ds in enumerate(train_dss)
    ]


    initial_model = Model(
        insize=in_size,
        outsize=out_size,
    ).state_dict()

    current_gmodel_version = 1

    global_models = {
        1: _serialize_to_bytes({
            "type": "download-global-model",
            'model': initial_model,
            'model_version': 1,
        }),
    }

    next_agg = aggregations[0]

    client_worker_map = dict()
    
    client_fetch_coros = dict()
    
    client_train_coros = dict()
    
    client_who_sent = set()
    UPDATES = []

    for t, events in enumerate(timeline):
        for event in events:
            ckey = chash(event['client'])
            if event['type'] == 'fetch':
                # print(f"<== {event['type']} {ckey}", flush=True)

                if ckey in client_fetch_coros:
                    raise ValueError(f"[FETCH] Client {ckey} already fetching data")
                if ckey in client_worker_map:
                    raise ValueError(f"[FETCH] Client {ckey} already has a worker assigned")
                if ckey in client_train_coros:
                    raise ValueError(f"[FETCH] Client {ckey} already training")

                worker = await __SCHEDULER__.pick_node(ckey, current_gmodel_version)
                client_worker_map[ckey] = worker
                client_fetch_coros[ckey] = asyncio.create_task(
                    remote_fetch(
                        {
                            "node": worker,
                            "client": event['client'][1],
                            "partition": event['client'][0],
                            "version": current_gmodel_version,
                            "ds_folder": ds_folder,
                            "train_loader": train_loaders[event['client'][0]][event['client'][1]],
                            "global_model": global_models[current_gmodel_version],
                        }
                    ))
                
            elif event['type'] == 'train':
                # print(f" T  {event['type']} {ckey}", flush=True)

                if ckey not in client_fetch_coros:
                    raise ValueError(f"Client {ckey} not found in fetch map")
                
                async def wait_and_train(task):
                    fetch_args = await task
                    return await remote_train(
                        {
                            "node": fetch_args["node"],
                            "client": fetch_args["client"],
                            "partition": fetch_args["partition"],
                            "version": fetch_args["version"],
                            "ds_folder": fetch_args["ds_folder"],
                            "insize": in_size,
                            "outsize": out_size,
                            "lr": args.lr_range[0],
                            "batch_size": 1024,
                            "momentum": 0.9,
                            "ephocs": 20,
                            "shuffle": False,
                        }
                    )
                
                client_train_coros[ckey] = asyncio.create_task(wait_and_train(client_fetch_coros[ckey]))
            
            elif event['type'] == 'send':
                # print(f"==> {event['type']} {ckey}", flush=True)
                if ckey not in client_train_coros:
                    raise ValueError(f"Client {ckey} not found in train map")

                UPDATES.append(client_train_coros[ckey])
                client_who_sent.add(ckey)

                del client_fetch_coros[ckey]
                del client_worker_map[ckey]
                del client_train_coros[ckey]
            else:
                raise ValueError(f"Unknown event type: {event['type']}")

        if next_agg == t:
        

            client_who_sent.clear()

            upts = await asyncio.gather(*UPDATES)
            UPDATES.clear()

            newModel, meta = aggregate_model(
                global_models[current_gmodel_version],
                [
                    (m, meta)
                    for m, meta in upts
                ]
            )

            current_gmodel_version += 1
            
            global_models[current_gmodel_version] = _serialize_to_bytes({
                "type": "download-global-model",
                'model': newModel,
                'model_version': current_gmodel_version,
            })

            # print()
            print(f"[INFO] GMODEL [{current_gmodel_version}/{len(aggregations)}]")
            # print(f"[INFO] AGGREGATED {len(upts)} UPDATES")
            # # for k, v in NODE_TIMES.items():
            # #     print(f"[INFO] {k} TRAIN: {v['train']['avg'] / v['train']['count']:.2f} s")
            # #     print(f"[INFO] {k} ENSURE_DATA: {v['ensure_data']['avg'] / v['ensure_data']['count']:.2f} s")
            # #     print(f"[INFO] {k} ENSURE_MODEL: {v['ensure_model']['avg'] / v['ensure_model']['count']:.2f} s")

            # # print avgs for all nodes
            # print(f"[INFO] AVG TRAIN: {sum([v['train']['avg'] for v in NODE_TIMES.values()]) / sum([v['train']['count'] for v in NODE_TIMES.values()]):.2f} s, count: {sum([v['train']['count'] for v in NODE_TIMES.values()])}")
            # print(f"[INFO] AVG ENSURE_DATA: {sum([v['ensure_data']['avg'] for v in NODE_TIMES.values()]) / sum([v['ensure_data']['count'] for v in NODE_TIMES.values()]):.2f} s, count: {sum([v['ensure_data']['count'] for v in NODE_TIMES.values()])}")
            # print(f"[INFO] AVG ENSURE_MODEL: {sum([v['ensure_model']['avg'] for v in NODE_TIMES.values()]) / sum([v['ensure_model']['count'] for v in NODE_TIMES.values()]):.2f} s, count: {sum([v['ensure_model']['count'] for v in NODE_TIMES.values()])}")
            # print(f"[INFO] AVG SERIALIZATION: {sum([v['serialization']['avg'] for v in NODE_TIMES.values()]) / sum([v['serialization']['count'] for v in NODE_TIMES.values()]):.2f} s, count: {sum([v['serialization']['count'] for v in NODE_TIMES.values()])}")

            if current_gmodel_version >= len(aggregations):
                break

            next_agg = aggregations[current_gmodel_version]


            # CLEAN UP OLD MODELS
            # to_keep = []
            # for k, v in client_fetch_map.items():
            #     to_keep.append(v)

            # to_keep.append(current_gmodel_version)

            # for k in list(global_models.keys()):
            #     if k not in to_keep:
            #         del global_models[k]

            # __SCHEDULER__.update_workers_versions(to_keep)
            # for node in __WORKERS__.keys():
            #     await drop_old_models(node, to_keep)

            # await asyncio.sleep(0.1)


            # OLD TRAIN

             # clients = [(key, client_fetch_map[key]) for key in clients]
            # client_nodes = [await __SCHEDULER__.pick_node(client, version) for client, version in clients]
            # clients = [(n, k, v) for (k, v), n in zip(clients, client_nodes)]

       
            # coros = []
            # for node, client, version in clients: 
            #     part, cidx = client.split("_")
            #     part = int(part)
            #     cidx = int(cidx)

            #     coros.append(
            #         remote_train(
            #             {
            #                 "node": node,
            #                 "client": cidx,
            #                 "partition": part,
            #                 "version": version,
            #                 "ds_folder": ds_folder,
            #                 "train_loader": train_loaders[part][cidx],
            #                 "global_model": global_models[version],
            #                 "insize": in_size,
            #                 "outsize": out_size,
            #                 "lr": args.lr_range[0],
            #                 "batch_size": 1024,
            #                 "momentum": 0.9,
            #                 "ephocs": 1,
            #                 "shuffle": False,
            #             }
            #         )
            #     )

            # upts = await asyncio.gather(*coros)

    
    print("Simulation finished")


import gzip
def _serialize_to_bytes(data: Any) -> bytes:
    buff = io.BytesIO()
    torch.save(data, buff)
    buff.seek(0)
    return gzip.compress(buff.read(), compresslevel=9)


async def worker_packet(name, data, ignoreSerialize=False) -> bytes:

    if not ignoreSerialize:
        start_time = time.time()
        loop = asyncio.get_running_loop()
        serialized: bytes = await loop.run_in_executor(
            None,         # use default ThreadPoolExecutor
            _serialize_to_bytes,
            data
        )
        end_time = time.time()

        NODE_TIMES[name]["serialization"]["avg"] += (end_time - start_time)
        NODE_TIMES[name]["serialization"]["count"] += 1
    else:
        serialized = data

    queue = __WORKERS__[name]["queue"]
    fut = asyncio.Future()
    await queue.put((serialized, lambda data, error: fut.set_result((data, error))))
    data, err = await fut
    if err is not None:
        raise err
    
    return data



def decode_auth_packet(packet: str) -> Union[None, dict]:
    packet_data = packet
    data = {}
    for line in packet_data.split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value

    if not "secret" in data:
        return None

    if data["secret"] != "asdasd":
        return None

    return data

def raw_packet(packet: str) -> bytes:
    packet = packet.encode("utf-8") if isinstance(packet, str) else packet
    packet_length = len(packet)
    packet_length_bytes = struct.pack("!I", packet_length)
    packet_with_length = packet_length_bytes + packet
    return packet_with_length

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    size = await reader.readexactly(4)
    size = struct.unpack("!I", size)[0]
    data = await reader.readexactly(size)
    str_data = data.decode("utf-8")
    auth = decode_auth_packet(str_data)
    if auth is None:
        print("Invalid auth packet", file=sys.stderr)
        writer.write(b"2")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        return  

    queue = asyncio.Queue()


    __WORKERS__[auth["name"]] = {
        "auth": auth,
        "queue": queue,
    }

    __SCHEDULER__.update_workers(__WORKERS__.keys())


    print(f"[INFO] New worker connected: {auth['name']} ({len(__WORKERS__)})")

    writer.write(b"1")
    await writer.drain()


    try:
        while True:
            req = await queue.get()
            if req is None:
                break

            data, cb = req

            try:
                writer.write(raw_packet(data))
                await writer.drain()

                size = await reader.readexactly(4)
                size = struct.unpack("!I", size)[0]
                data = await reader.readexactly(size)
                cb(data, None)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                cb(None, e)

    finally:
        del __WORKERS__[auth["name"]]
        __SCHEDULER__.update_workers(__WORKERS__.keys())

        writer.close()

async def start_server(host: str, port: int, run_task: asyncio.Task):
    server = await asyncio.start_server(handle_client, host, port)

    print(f"Serving on {host}:{port}")
    async with server:
        await asyncio.wait(
            [server.serve_forever(), run_task], return_when=asyncio.FIRST_COMPLETED
        )
        
    print("Server stopped")


EventType = str  # Literal["fetch", "train", "send"]
Client = Tuple[int, int]

EventFetchModel = Dict[str, Any]
EventTrainModel = Dict[str, Any]
EventSendModel = Dict[str, Any]

Event = Union[EventFetchModel, EventTrainModel, EventSendModel]
Timeline = List[List[Event]]

Simulation = Dict[str, Any]

SimExport = Dict[str, Any]


def parse_sim_export(json_file_path: str) -> SimExport:
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Basic validation
        if not isinstance(data, dict):
            raise ValueError("The JSON data must be a dictionary.")
        if "timeline" not in data:
            raise ValueError("The JSON data must contain a 'timeline' field.")
        if "aggregations" not in data:
            raise ValueError(
                "The JSON data must contain an 'aggregations' field."
            )
        if "sim" not in data:
            raise ValueError("The JSON data must contain a 'sim' field.")

        # Validation for aggregations
        if not isinstance(data["aggregations"], list):
            raise ValueError(
                "The 'aggregations' field must be a list of numbers."
            )
        for item in data["aggregations"]:
            if not isinstance(item, (int, float)):
                raise ValueError(
                    "Each element in 'aggregations' must be a number."
                )

        # Validation for timeline (Timeline) - more complex
        timeline = data["timeline"]
        if not isinstance(timeline, list):
            raise ValueError("The 'timeline' field must be a list (Timeline).")

        for time_step in timeline:
            if not isinstance(time_step, list):
                raise ValueError(
                    "Each time step in the Timeline must be a list of events."
                )
            for event in time_step:
                if not isinstance(event, dict):
                    raise ValueError("Each event must be a dictionary.")
                if "type" not in event:
                    raise ValueError("Each event must have a 'type' field.")
                if "client" not in event:
                    raise ValueError("Each event must have a 'client' field.")

                # Further validation based on event type can be added here

        if not isinstance(data["sim"], dict):
            raise ValueError("The 'sim' field must be a dictionary.")
        
        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except ValueError as e:
        raise ValueError(f"Error parsing JSON: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLCdata server")
    parser.add_argument("--port", type=int, help="Server port", default=6969)
    parser.add_argument("--host", type=str, help="Server host", default="0.0.0.0")

    parser.add_argument("--timeline", type=str, required=True, help="Path to the JSON file")
    parser.add_argument("--ds-folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--repo-folder", type=str, required=True, help="Path to the model repository folder")
    parser.add_argument("--nuke-repo", action="store_true", help="Delete the model repository before starting")
    
    parser.add_argument(
        "--lr-range",
        type=float,  # Apply float conversion to each argument
        nargs=2,     # Expect exactly 2 arguments following --lr-range
        default=[0.01, 0.01], # Default can be a list or tuple
        metavar=("LR_START", "LR_END"), # Optional: Improves help message
        help="Learning rate range (start end) for the simulation"
    )   

    
    parser.add_argument("--lr-interpolation", type=str, default="linear", help="Interpolation method for learning rate")
    parser.add_argument("--lr-end-interpolation", type=int, default=0, help="End interpolation for learning rate")


    args = parser.parse_args()
    port = args.port
    host = args.host

    if args.nuke_repo:
        import shutil
        import os
        if os.path.exists(args.repo_folder):
            shutil.rmtree(args.repo_folder)
            print(f"Deleted repository folder: {args.repo_folder}")


    sim_data = parse_sim_export(args.timeline)
    timeline = sim_data["timeline"]
    aggregations = sim_data["aggregations"]
    sim = sim_data["sim"]   


    args = (
        sim, timeline, aggregations, args.ds_folder, args.repo_folder, args
    )

    async def main(args):
        run_task = asyncio.create_task(run(args))
        await start_server(host, port, run_task)

    asyncio.run(main(args))
                

# python tm-server.py --timeline=.timelines/ss.json --repo-folder=.simulations/sim-ss-data --ds-folder=.splits/data --nuke-repo --lr-range 0.1 0.1