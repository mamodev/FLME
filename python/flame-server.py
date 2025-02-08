# the server will spawn --nworkers subprocesses 
# after this it will create a socket server and listen on port --port
# the worker will connect to the server for communication

import os
import sys
import json
import torch
import numpy as np
import time 

import torch.optim as optim
import torch.nn.functional as F

import argparse
import socket
import subprocess
import io

import threading
import queue

from cubeds import CuboidDataset, CuboidDatasetMeta
from cubeds import VecClassifier

from flame_utils import dataset_to_device
from flame_evaluator import evaluator_thread

def train(model: torch.nn.Module, train_loader, optimizer, scheduler = None):
    model.train()
    
    gprameters = [val.detach().clone() for val in model.parameters()]

    mu = 1

    for data, target in train_loader:  
        optimizer.zero_grad()

        proximal_term = np.float32(0.0)
        for lp, p in zip(gprameters, model.parameters()):
            proximal_term += torch.square((lp - p).norm(2))

        output = model(data)
        loss = F.nll_loss(output, target) + (mu / 2) * proximal_term
        loss.backward()
        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()

    
def send_packet(socket, packet):
    assert type(packet) == bytes, f"Packet must be bytes: {type(packet)}"
    length = len(packet)
    length_buff = length.to_bytes(4, byteorder='big')
    socket.sendall(length_buff + packet)

def receive_packet(socket):
    length_buff = socket.recv(4)
    length = int.from_bytes(length_buff, byteorder='big')

    recevd = 0
    packet = b''
    while recevd < length:
        data = socket.recv(length - recevd)
        packet += data
        recevd += len(data)

    return packet

def send_int(socket, i):
    send_packet(socket, i.to_bytes(4, byteorder='big'))

def recv_int(socket):
    i_buff = receive_packet(socket)
    return int.from_bytes(i_buff, byteorder='big')

def receive_json(socket):
    packet = receive_packet(socket)
    return json.loads(packet)

def cpu_state_dict(state_dict):
    return {k: v.cpu().numpy() for k, v in state_dict.items()}

def send_json(socket, data):
    packet = json.dumps(data).encode()
    send_packet(socket, packet)

def send_state_dict(socket, state_dict):
    buffer = io.BytesIO()
    np.save(buffer, cpu_state_dict(state_dict))
    send_packet(socket, buffer.getvalue())

def receive_state_dict(socket):
    packet = receive_packet(socket)
    buffer = io.BytesIO(packet)
    sdict  = np.load(buffer, allow_pickle=True).item()
    return {k: torch.tensor(v) for k, v in sdict.items()}

parser = argparse.ArgumentParser(description='Flame server')
parser.add_argument('--port', type=int, default=8080, help='port number')
parser.add_argument('--worker', type=int, default=-1, help='worker flag (default: -1)')
parser.add_argument('--ds', type=str, default='', help='data store path')
parser.add_argument('--max-workers', type=int, default=4, help='max number of workers')
parser.add_argument('--partitioning', type=int, default=1, help='partitioning factor')
parser.add_argument('--rounds', type=int, default=10, help='number of rounds')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')



def deepClone(model):
    return {k: v.detach().clone() for k, v in model.items()}

def worker(args):
    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)

    wid = args.worker
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect(('localhost', args.port))
    print(f"Worker {wid} connected to server")

    wid_buff = wid.to_bytes(4, byteorder='big')
    server_socket.sendall(wid_buff)

    groups = receive_json(server_socket)

    get_ds_from_json = lambda g: CuboidDataset.LoadGroup(args.ds, g["group"], train=True, partitioning=g["partitioning"], partitioning_start=g["min"], partitioning_end=g["max"])

    dds = [(get_ds_from_json(g), g) for g in groups]
    # Flat dds: from List[Tuple[List[Dataset], Dict]] to List[Tuple[Dataset, Dict]]
    dds = [(ds, g) for ds, g in dds for ds in ds]

    dds_group = [g for _, g in dds]
    dds = [ds for ds, _ in dds]

    ddsl = [dataset_to_device(ds, torch.device("cuda"), batch_size=512, shuffle=True) for ds in dds]

    # devices = [Device(VecClassifier(3, 8), ds, None) for ds in dds]
    cuda_device = torch.device("cuda")
    net = VecClassifier(3, 8).to(cuda_device)

    samples = sum([len(ds) for ds in dds])
    send_int(server_socket, samples)


    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)


    while True:

        # model: Dict[str, torch.Tensor]
        try: 
            model = receive_state_dict(server_socket)
        except:
            break

        if len(model) == 0:
            break

        local_models = []
        for i, ds in enumerate(ddsl):
            net.load_state_dict(deepClone(model))
            optimizer.zero_grad()
            # scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
            scheduler = None

            torch.manual_seed(0)
            np.random.seed(0)
            for _ in range(5):
                train(net, ds, optimizer, scheduler)
            
            local_models.append(net.state_dict())
            
        send_int(server_socket, len(local_models))
        for i, local_model in enumerate(local_models):
            send_int(server_socket, dds_group[i]["group"])
            send_int(server_socket, len(dds[i]))            
            send_state_dict(server_socket, local_model)

    server_socket.close()

def server(args):
    meta = CuboidDatasetMeta.load(f"{args.ds}/META.json")
    n_groups = meta.n_groups

    nvdevices = n_groups * args.partitioning
    n_workers = min(args.max_workers, nvdevices)
    vdids = [(i,g)  for g in range(n_groups) for i in range(args.partitioning)]
    avg_device_per_worker = nvdevices // n_workers
    print(f"Partitioning {nvdevices} devices among {n_workers} workers, goal: {avg_device_per_worker} devices per worker")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    server_socket.bind(('localhost', args.port))
    server_socket.listen(n_workers)
    print(f"Server listening on port {args.port}")

    
    print(f"Starting server with {n_workers} workers")
    workers = []    
    for i in range(n_workers):
        worker = subprocess.Popen([sys.executable, __file__, f'--worker={i}', f'--ds={args.ds}', 
            f'--port={args.port}', f'--partitioning={args.partitioning}'])

        workers.append(worker)
    print("Workers started:")

    worker_sockets = []
    while len(worker_sockets) < n_workers:
        client_socket, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")
        worker_sockets.append([-1, client_socket])

    # receive the worker id from the workers

    for i, (_, worker_socket) in enumerate(worker_sockets):
        wid_buff = worker_socket.recv(4)
        wid = int.from_bytes(wid_buff, byteorder='big')
        worker_sockets[i][0] = wid

    print("All workers connected")
    worker_sockets.sort(key=lambda x: x[0])

    
    for i, (wid, worker_socket) in enumerate(worker_sockets):
        # n_devices = avg_device_per_worker   
        # if i == n_workers - 1:
        #     n_devices += int(nvdevices - (avg_device_per_worker * n_workers))

        startidx = i * avg_device_per_worker
        endidx = (i + 1) * avg_device_per_worker
        if i == n_workers - 1:
            endidx = nvdevices

        devices = vdids[startidx:endidx]
        groups = set([g for _, g in devices])
        json_groups = []
        
        for g in list(groups):
            # list of device id for group g
            devices_of_group = [i for i, group in devices if group == g]
            minIdx = min(devices_of_group)
            maxIdx = max(devices_of_group)
            n_if_contiguous = maxIdx - minIdx + 1
            assert n_if_contiguous == len(devices_of_group), f"Devices are not contiguous: {devices_of_group}"
            print(f"Worker {wid} has devices {minIdx} to {maxIdx} for group {g}")
            json_groups.append({"group": g, "min": minIdx + 1, "max": maxIdx + 1, "partitioning": args.partitioning})

        send_json(worker_socket, json_groups)


    nsampl = 0
    for (wid, worker_socket) in worker_sockets:
        n_samples = recv_int(worker_socket)
        print(f"Worker {wid} has {n_samples} samples")
        nsampl += n_samples

    print(f"All workser configured, Total samples: {nsampl}")


    model = VecClassifier(3, 8).state_dict()



    # Start evaluator thread
    model_queue = queue.Queue()
    stop_eval_event = threading.Event()

    gids = [g for g in range(n_groups) for _ in range(args.partitioning)]
    initial_local_models = [model for _ in range(n_groups * args.partitioning)]

    model_queue.put((model, initial_local_models, gids))

    def isStateDictInGPU(state_dict):
        for k, v in state_dict.items():
            if v.device.type != 'cuda':
                return False
        return True

    def aggregator(model_queue, worker_sockets, model, stop_event):
        ephocs = args.rounds
        for e in range(ephocs):
            
            if isStateDictInGPU(model):
                print("Model is in GPU")
                exit(1)

            local_models = []
            local_samples = []
            gids = []

            print(f"Epoch {e}/{ephocs} started")
            for (wid, worker_socket) in worker_sockets:
                send_state_dict(worker_socket, model)

            for (wid, worker_socket) in worker_sockets:
                n_models = recv_int(worker_socket)
                for i in range(n_models):
                    gid = recv_int(worker_socket)
                    n_samples = recv_int(worker_socket)
                    local_model = receive_state_dict(worker_socket)
                    local_models.append(local_model)
                    gids.append(gid)
                    local_samples.append(n_samples)

            for local_model in local_models:
                assert not isStateDictInGPU(local_model), "The model is recevied from socket how can it be in GPU"

            for k in local_models[0].keys():
                model[k] = sum([m[k] * n for m, n in zip(local_models, local_samples)]) / sum(local_samples)

            print(f"Epoch {e}/{ephocs} done")

            model_queue.put((model, local_models, gids))

        stop_event.set()

    aggregator_thread = threading.Thread(target=aggregator, args=(model_queue, worker_sockets, model, stop_eval_event))
    aggregator_thread.start()

    
    evaluator_thread(model_queue, stop_eval_event, args.ds, model)
    aggregator_thread.join()

    for wid, worker_socket in worker_sockets:
        worker_socket.close()

    server_socket.close()

if __name__ == '__main__':
    args = parser.parse_args()
    if args.worker != -1:
        worker(args)
    else:
        server(args)
