
import socket
import time
import json
import struct
import argparse
import sys
import os
from typing import Union
import io
import torch    

import hashlib



class MemoryDataLoader:
    def __init__(self, data_tensors, target_tensors, batch_size=1, shuffle=False):
        self.data = data_tensors
        self.targets = target_tensors
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            import random
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long)
            yield (
                self.data[batch_indices],
                self.targets[batch_indices]
            )

    
    def __len__(self):
        return len(self.data)



def create_auth_packet(host: str, port: int, name: str, role="slave") -> bytes:
    SECRET = "asdasd"
    curr_arch = os.uname().machine
    curr_os = os.uname().sysname
    curr_os_version = os.uname().release

    packet = (
        f"secret={SECRET}\n"
        f"host={host}\n"
        f"port={port}\n"
        f"name={name}\n"
        f"arch={curr_arch}\n"
        f"os={curr_os}\n"
        f"os_version={curr_os_version}\n"
        f"role={role}\n"
    ).encode("utf-8")

    return raw_packet(packet)


def raw_packet(packet: str) -> bytes:
    packet = packet.encode("utf-8") if isinstance(packet, str) else packet
    packet_length = len(packet)
    packet_length_bytes = struct.pack("!I", packet_length)
    packet_with_length = packet_length_bytes + packet
    return packet_with_length



parser = argparse.ArgumentParser(description="FLCdata slave")
parser.add_argument("--host", type=str, help="Host to connect to")
parser.add_argument("--port", type=int, help="Port to connect to")
parser.add_argument("--name", type=str, help="Name of the client")


args = parser.parse_args()

if args.host is None or args.port is None or args.name is None:
    print("Usage: <cmd.py> --host <host> --port <port> --name <name>")
    sys.exit(1)


def try_connect(host: str, port: int, name: str, role="slave") -> Union[None, socket.socket]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))

        s.sendall(create_auth_packet(host, port, name, role))
        print(f"Connected to {host}:{port} as {name} ({role})")

        response = s.recv(1)

        if response == b"1":
            print("Connection accepted")
            return s
        elif response == b"2":
            print("Invalid Auth")
            s.close()
            exit(1)
        else:
            print("Unknown response")
            s.close()
       
    except Exception as e:
        print(f"Connection failed: {e}")
        return None
    
def untill_connected(host: str, port: int, name: str, role="slave", interval: int = 2) -> socket.socket:
    while True:
        s = try_connect(host, port, name, role)
        if s is not None:
            return s
        print(f"Retrying connection to {host}:{port} in {interval} seconds...")
        time.sleep(interval)

def recv_all(s: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = s.recv(size - len(data))
        if not chunk:
            break
        data += chunk
        
    return data

def deep_clone_sdict(state_dict):
    return {k: v.clone() for k, v in state_dict.items()}

hash_cache = dict()

def get_file_hash(path: str) -> Union[str, None]:
    global hash_cache
    if path in hash_cache:
        return hash_cache[path]
    
    if os.path.exists(path) and os.path.isfile(path):
        sha256 = hashlib.sha256()
        data = b""
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024)
                if not chunk:
                    break
                data += chunk
                sha256.update(chunk)

        hh = sha256.hexdigest()
        hash_cache[path] = hh
        return hh   
    else:
        return None

import gzip

def main():
    s = untill_connected(args.host, args.port, args.name)

    loaders_cache = dict()
    globals_model_cache = dict()
    net = None
    while True:
        req_size = recv_all(s, 4)
        if len(req_size) == 0:
            print("Connection closed by server")
            break

        req_size = struct.unpack("!I", req_size)[0]
        req = recv_all(s, req_size)
 
        req = gzip.decompress(req)
        req = io.BytesIO(req)
        req = torch.load(req, weights_only=False)


        if req["type"] == "ensure_data":
            loader_key = f'{req["client_idx"]}_{req["partition"]}'

            if loader_key in loaders_cache:
                s.sendall(raw_packet(b"1"))
            else:
                s.sendall(raw_packet(b"0"))
       
        if req["type"] == "download-data":
            loader_key = f'{req["client_idx"]}_{req["partition"]}'
           
            loaders_cache[loader_key] = {
                "data": req["data"],
                "targets": req["targets"],
            }

            s.sendall(raw_packet(b"1"))

        elif req["type"] == "ensure_global_model":
            if req["model_version"] in globals_model_cache:
                s.sendall(raw_packet(b"1"))
            else:
             s.sendall(raw_packet(b"0"))

        elif req["type"] == "download-global-model":
            globals_model_cache[req["model_version"]] = req["model"]
            s.sendall(raw_packet(b"1"))

        elif req["type"] == "ensure-file":
            hash_sum = req["hash"]
            path = req["path"]
            hash = get_file_hash(path)
            if hash == hash_sum:
                s.sendall(raw_packet(b"1"))
            else:
                s.sendall(raw_packet(b"0"))

        elif req["type"] == "download-file":
            path = req["path"]
           
            folder = os.path.dirname(path)
            if folder != "" and not os.path.exists(folder):
                os.makedirs(folder)

            with open(path, "wb") as f:
                f.write(req["data"])
          
            s.sendall(raw_packet(b"1"))

        elif req["type"] == "train":
            loader_key = f'{req["client_idx"]}_{req["partition"]}'
            assert loader_key in loaders_cache, f"Loader {loader_key} not found in cache"
            assert req["model_version"] in globals_model_cache, f"Model {req['model_version']} not found in cache"

            if net is None:
                from lib.flcdata import  Model
                net = Model(insize=req["insize"], outsize=req["outsize"])

            net.load_state_dict(
                deep_clone_sdict(globals_model_cache[req["model_version"]])
            )

            ephocs = req["ephocs"]
            lr = req["learning_rate"]
            batch_size = req["batch_size"]
            momentum = req["momentum"]
            shuffle = req["shuffle"]
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

            loader = MemoryDataLoader(
                loaders_cache[loader_key]["data"],
                loaders_cache[loader_key]["targets"],
                batch_size=batch_size,
                shuffle=shuffle,
            )


            net.train()
            for i in range(ephocs):
                for data, target in loader:
                    optimizer.zero_grad()
                    output = net(data)
                    loss = torch.nn.functional.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()


            upt = net.state_dict()

            res = {
                "model": upt,
                "meta": {
                    "from_version": req["model_version"],
                    "client_idx": req["client_idx"],
                    "partition": req["partition"],
                    "shuffle": shuffle,
                    "batch_size": batch_size,
                    "momentum": momentum,
                    "learning_rate": lr,
                    "train_loss": 0.1,
                    "test_loss": 0.1,
                    "local_epoch": ephocs,
                    "train_samples": len(loader),
                }
            }


            buff = io.BytesIO()
            torch.save(res, buff)
            buff.seek(0)
            data = buff.read()
            s.sendall(raw_packet(data))

        elif req["type"] == "drop_old_models":
            to_keep = req["models_to_keep"]

            for k in list(globals_model_cache.keys()):
                if k not in to_keep:
                    del globals_model_cache[k]

            s.sendall(raw_packet(b"1"))



while True:
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)