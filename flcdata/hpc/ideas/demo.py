

import socket
import struct
import threading    
import io
import queue
import json
import numpy as np
import itertools
from typing import List
import asyncio

target_workers = 7

data_path = "data.bin"
model_path = "model.bin"
sim = "../.timelines/ss.json"

with open(sim, 'r') as f:
    timeline_json = json.load(f)
    f.close()


with open(model_path, 'rb') as f:
    model = f.read()
    f.close()


# Decode model and keep vectors shapes and offsets
buff = io.BytesIO(model)
MODEL_SIZE = struct.unpack('i', buff.read(4))[0]
n_layers = struct.unpack('i', buff.read(4))[0]
layer_shapes = []
for i in range(n_layers):
    key_len = struct.unpack('i', buff.read(4))[0]
    key = buff.read(key_len).decode('ascii')

    shape_len = struct.unpack('i', buff.read(4))[0]
    shape = []
    for j in range(shape_len):
        dim = struct.unpack('i', buff.read(4))[0]
        shape.append(dim)

    layer_shapes.append((key, shape))

model_header_len = buff.tell()
base_offset = buff.tell()
layers = []
for key, shape in layer_shapes:
    layer_offset = base_offset
    layer_type = np.float32
    layer_size = np.prod(shape) * np.dtype(layer_type).itemsize
    layers.append((key, layer_offset, layer_size, shape))
    base_offset += layer_size

def avg_models(models: List[bytes], weights: List[float]):
    assert len(models) == len(weights)
    assert len(models) > 0

    buff = bytearray(MODEL_SIZE)
    buff[:model_header_len] = models[0][:model_header_len] 

    for i in range(1, len(models)):
        # assert models[i][:model_header_len] == models[0][:model_header_len], f"Model header mismatch"
        assert models[i][model_header_len:] != models[0][model_header_len:], f"Model body should be different"

    for L in range(n_layers):
        key, layer_offset, layer_size, shape = layers[L]
        acc = np.zeros(shape, dtype=np.float32)
        # acc = np.random.rand(*shape).astype(np.float32)  # Random initialization for testing

        for i in range(len(models)):
            m = models[i]
            w = weights[i]
            layer_data = np.frombuffer(m[layer_offset:layer_offset + layer_size], dtype=np.float32)
            layer_data = layer_data.reshape(shape)
            
            acc += layer_data * w
    

        acc /= sum(weights)

        acc = acc.astype(np.float32)
        acc = acc.tobytes()
        buff[layer_offset:layer_offset + layer_size] = acc



    return buff


with open(data_path, 'rb') as f:
    h1 = struct.unpack('IIII', f.read(16))
    dataset_size = h1[0] + 4
    n_partitions = h1[1]
    partition_shapes = []

    for p in range(n_partitions):
        data_shape_len = struct.unpack('I', f.read(4))[0]
        data_shape = []
        for i in range(data_shape_len):
            dim = struct.unpack('I', f.read(4))[0]
            data_shape.append(dim)

        targets_len = struct.unpack('I', f.read(4))[0]
        targets_shape = []
        for i in range(targets_len):
            dim = struct.unpack('I', f.read(4))[0]
            targets_shape.append(dim)

        partition_shapes.append((data_shape, targets_shape))

    header_len = f.tell()
    f.seek(0)
    header_data = f.read(header_len)

host = '0.0.0.0'
port = 6969

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024 * 10)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 10)

sock.bind((host, port))
sock.listen(100)

workers = {}

while len(workers) < target_workers:
    print(f"Waiting for {target_workers - len(workers)} workers to connect...")
    client_socket, addr = sock.accept()

    worker_id = struct.unpack('I', client_socket.recv(4))[0]
    if worker_id in workers:
        print(f"Worker {worker_id} already connected.")
        client_socket.close()
        continue
    else:
        workers[worker_id] = {}
    
    client_socket.send(struct.pack('II', len(model), dataset_size))
    client_socket.send(struct.pack('I', header_len))
    client_socket.send(header_data)

    workers[worker_id]["sock"] = client_socket

print(f"Worker {worker_id} connected from {addr}")

workers = [
            {
                "id": worker_id,
                "sock": workers[worker_id]["sock"],
                "sender_thread": None,
                "data_file": open(data_path, 'rb'),
                "data_sender_queue": queue.PriorityQueue(0),
            } 
        for worker_id in workers]


def send_file_slice_zero_copy(sock, ds_file, offset, size):
    remaining = size
    while remaining > 0:
        chunk = sock.sendfile(ds_file, offset, size)
        if chunk <= 0:
            raise Exception("Socket sendfile failed")

        offset += chunk
        size -= chunk
        remaining -= chunk

class DatasetPartition:
    def __init__(self, doffset, dsize, toffset, tsize):
        self.doffset = doffset
        self.dsize = dsize
        self.toffset = toffset
        self.tsize = tsize

    def send(self, sock, ds_file):
        sock.sendall(struct.pack('IIIII', 0, self.doffset, self.dsize, self.toffset, self.tsize))
        send_file_slice_zero_copy(sock, ds_file, self.doffset, self.dsize)
        send_file_slice_zero_copy(sock, ds_file, self.toffset, self.tsize)

def data_sender_thread(worker, sock: socket.socket, queue):
    while True:
        event = queue.get()
        priority, _, payload = event
        if payload is None:
            break

        if isinstance(payload, DatasetPartition):
            payload.send(sock, worker["data_file"])
        elif isinstance(payload, tuple):
            sock.sendmsg(payload)
        else:
            sock.sendall(payload)

def recv_all(sock, size):
    data = bytearray(size)
    received = 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            return None

        data[received:received + len(chunk)] = chunk
        received += len(chunk)

    return data

def data_receiver_thread(worker, client_socket, in_queue, out_queue):

    upds = []
    while True:
        models_to_get = in_queue.get()
        if models_to_get is None:
            break

        while True:
            got = 0
            for client_id, part_id in models_to_get:
                for i in range(len(upds)):
                    if upds[i][0] == client_id and upds[i][1] == part_id:
                        got += 1
                        break

            if got == len(models_to_get):
                break


            header = recv_all(client_socket, 8)
            if not header:
                break

            client_id, part_id = struct.unpack('II', header)
            m = recv_all(client_socket, len(model))
            if not m:
                break

            upds.append((client_id, part_id, m))

    
        outs = []
        for client_id, part_id in models_to_get:
            for c, p, m in upds:
                if c == client_id and p == part_id:
                    outs.append((client_id, part_id, m))
                    break

        

        out_queue.put(outs)


def get_client_partition_offsets(partition, client, splt_per_partition):
    dshape, tshape = partition_shapes[partition]

    psamples = dshape[0]
    psamples_per_client = psamples // splt_per_partition[partition]

    dshape = list(dshape)
    dshape[0] = psamples_per_client

    tshape = list(tshape)
    tshape[0] = psamples_per_client


    data_offset = 0
    targets_offset = 0 

    for p in range(len(splt_per_partition)):
        if p < partition:
            data_offset += 4 * np.prod(partition_shapes[p][0])
            targets_offset += 8 * np.prod(partition_shapes[p][1])


    data_size = psamples_per_client * np.prod(dshape[1:]) * 4
    targets_size = psamples_per_client * 8

    part_data_size = 4 * np.prod(partition_shapes[partition][0])
    prev_part_offs = header_len + data_offset + targets_offset

    data_offset = prev_part_offs + client * data_size

    targets_offset = prev_part_offs + part_data_size + client * targets_size


    nsamples = psamples_per_client

    data_offset = int(data_offset)
    targets_offset = int(targets_offset)
    data_size = int(data_size)
    targets_size = int(targets_size)

    return data_offset, data_size, targets_offset, targets_size, nsamples



def file_writer_thread(path, file_queue):
    import os
    import struct
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path + "info.json", 'w') as f:
        json.dump({}, f)
        f.close()



    while True:
        file_name, data = file_queue.get()
        if file_name is None:
            break


        buff = io.BytesIO(data)

        model_size = struct.unpack("i", buff.read(4))[0]
        # print(f"Python: model size = {model_size}")

        num_layers = struct.unpack("i", buff.read(4))[0]
        # print(f"Python: num layers = {num_layers}")

        model = {}
        layers = []

        for i in range(num_layers):
            key_len = struct.unpack("i", buff.read(4))[0]
            key = buff.read(key_len).decode('ascii')
            shape_len = struct.unpack("i", buff.read(4))[0]
            shape = []
            for j in range(shape_len):
                dim = struct.unpack("i", buff.read(4))[0]
                shape.append(dim)

            layers.append((key, shape))


        boffset = buff.tell()
        for i in range(num_layers):
            nparr = np.frombuffer(
                data,
                dtype=np.float32,
                count=np.prod(layers[i][1]),
                offset=boffset  
            )

            boffset += np.prod(layers[i][1]) * np.dtype(np.float32).itemsize

            nparr = nparr.reshape(layers[i][1])
            model[layers[i][0]] = torch.from_numpy(nparr)
            del nparr

        torch.save(model, path + file_name)

async def main(file_writer_queue):
    
    timeline = timeline_json["timeline"]
    sim = timeline_json["sim"]

    proportional_knowledge = "proportionalKnowledge" in sim and sim["proportionalKnowledge"]
    splt_per_partition = sim["client_per_partition"]
    if proportional_knowledge:
        max_client_pp = max(splt_per_partition)
        for pidx, _ in enumerate(sim["client_per_partition"]):
            splt_per_partition[pidx] = max_client_pp

    # upload all data in order  

    fetched_clients = set()

    client_worker = {}
    workerIdx = 0
    tie_br = itertools.count()

    for t, events in enumerate(timeline):   
        for event in events:
            if event["type"] == "fetch":
                partition = event["client"][0]
                client = event["client"][1]

                if (partition, client) in fetched_clients:
                    continue

                fetched_clients.add((partition, client))

                data_offset, data_size, targets_offset, targets_size, nsamples = get_client_partition_offsets(partition, client, splt_per_partition)

                queue = workers[workerIdx]["data_sender_queue"]

                queue.put((0, next(tie_br), DatasetPartition(data_offset, data_size, targets_offset, targets_size)))

                client_worker[(partition, client)] = workerIdx
                workerIdx  = (workerIdx + 1) % len(workers)

    del fetched_clients


    aggregations = timeline_json["aggregations"]
    global_model_version = 1
    next_agg = aggregations[0]

    workers_sent_models = [set() for _ in range(len(workers))]

    updates = []

    global_model = model

    client_samples = {}

    for t, events in enumerate(timeline):
        for event in events:
            if event["type"] == "fetch":
                workerIdx = client_worker[(event["client"][0], event["client"][1])]
                worker = workers[workerIdx]
                queue = worker["data_sender_queue"]
                
                if global_model_version not in workers_sent_models[workerIdx]:
                    queue.put((0, next(tie_br), (struct.pack('II', 1, global_model_version), global_model)))
                    workers_sent_models[workerIdx].add(global_model_version)


                ephocs = 12
                batch_size = 1024
                learning_rate = 0.1
                momentum = 0.9
                weight_decay = 0.0001
                shuffle = False



                data_offset, data_size, targets_offset, targets_size, nsamples = get_client_partition_offsets(event["client"][0], event["client"][1], splt_per_partition)
               
                buff = struct.pack('10Ifff?',
                    2, global_model_version, event["client"][1], event["client"][0],
                    data_offset, data_size, targets_offset, targets_size,
                    ephocs, batch_size, learning_rate, momentum, weight_decay, shuffle)

                client_samples[(event["client"][0], event["client"][1])] = nsamples

                queue.put((1, next(tie_br), buff))


            if event["type"] == "train":
                pass

            elif event["type"] == "send":
                updates.append(event["client"])

        if t >= next_agg:
            models = []
            for workerIdx, worker in enumerate(workers):
                wqueue = worker["receiver_wqueue"]
                wqueue.put(set([(client[1], client[0]) for client in updates if client_worker[(client[0], client[1])] == workerIdx]))

            for workerIdx, worker in enumerate(workers):
                rqueue = worker["receiver_rqueue"]
                models += rqueue.get()

            assert len(models) == len(updates), f"model received mismatch: {len(models)} != {len(updates)}"

            models = [ (c, p, m, client_samples[(p, c)]) for c, p, m in models]

            global_model = avg_models([m for c, p, m, nsamples in models], [nsamples for c, p, m, nsamples in models])
            
            updates = []

            global_model_version += 1

            file_writer_queue.put((f"{global_model_version}.model", global_model))

            print(f"New global model version: {global_model_version}")

            if global_model_version >= len(aggregations):
                break
            
            next_agg = aggregations[global_model_version]

try:
    for worker in workers:
        worker_id = worker["id"]
        client_socket = worker["sock"]
        q = worker["data_sender_queue"]
        t = threading.Thread(target=data_sender_thread, args=(worker, client_socket, q))
        t.start()
        worker["sender_thread"] = t


        rwrite = queue.Queue(0)
        rread = queue.Queue(0)

        worker["receiver_rqueue"] = rread
        worker["receiver_wqueue"] = rwrite
   
        t = threading.Thread(target=data_receiver_thread, args=(worker, client_socket, rwrite, rread))
        t.start()
        worker["receiver_thread"] = t

    file_writer_queue = queue.Queue(0)
    file_writer_thread = threading.Thread(target=file_writer_thread, args=("../.simulations/prova/", file_writer_queue))
    file_writer_thread.start()
    asyncio.run(main(file_writer_queue))
    print("Simulation finished.")

    

except KeyboardInterrupt:
    print("Simulation interrupted.")
finally:
    
    for worker in workers:
        worker["sock"].close()
        worker["data_sender_queue"].put((999, 1, None))
        worker["receiver_wqueue"].put(None)
        worker["sender_thread"].join()

    file_writer_queue.put((None, None))
    file_writer_thread.join()

    sock.close()

