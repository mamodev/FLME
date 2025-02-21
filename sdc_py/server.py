import asyncio
import socket
import struct
import argparse 
import importlib
from typing import List, Dict, Tuple
import numpy as np
from torch import nn
import os
import torch

from multiprocessing import shared_memory, SimpleQueue, Process, queues

import jwt  
import datetime

import protocol

SECRET_KEY = "some_secret_key"

# The fl_model python module should contain the following attributes:
# - Model : nn.Module

class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.split_models = {}

    def split(self, clusterID, intoClusterID, one, two):
        if clusterID not in self.models:
            raise Exception(f"Cluster {clusterID} not found")
        
        if intoClusterID  in self.models:
            raise Exception(f"Cluster {intoClusterID} already exists")
        
        self.models[clusterID]["split"] = intoClusterID
        self.models[clusterID][self.models[clusterID]["latest"]+1] = one
        
        self.models[intoClusterID] = { "latest": 1 }
        self.models[intoClusterID][1] = two


    def put(self, clusterID, version, model):
        if clusterID not in self.models:
            self.models[clusterID] = {}
            self.models[clusterID]["latest"] = version
            self.models[clusterID]["split"] = None

        if version > self.models[clusterID]["latest"]:
            self.models[clusterID]["latest"] = version

        self.models[clusterID][version] = model
    
    def get(self, clusterID, version):
        if clusterID not in self.models:
            return None
        
        if version == 0:
            version = self.models[clusterID]["latest"]
        
        if version not in self.models[clusterID]:
            return None
        
        return self.models[clusterID][version], version


class TCPServerWorker:
    def __init__(self, id=None, host='127.0.0.1', port=8888, update_queue=None, gmodel_update_queue=None):
        assert update_queue is not None, "update_queue must be provided"
        assert gmodel_update_queue is not None, "gmodel_update_queue must be provided"
        assert id is not None, "id must be provided"

        self.id = id
        self.update_queue = update_queue
        self.gmodel_update_queue = gmodel_update_queue
        self.host = host
        self.port = port

        self.registry = ModelRegistry()

    async def handle_packet(self, writer: asyncio.StreamWriter, packet_id: int, payload: bytes, auth: Dict):
        assert auth != None, "auth must be provided"
        if packet_id == protocol.GetModelPacketID:
            try:
                packet = protocol.GetModelPacket.from_buffer(payload)
            except Exception as e:
                writer.write(protocol.create_response_packet(400, str(e).encode()))
                return

            model, version = self.registry.get(auth['cluster'], packet.model_version)
            if model is None:
                writer.write(protocol.create_response_packet(404, b"Model not found"))
                return

            res = protocol.GetModelPacketResponse(version, protocol.ModelData(model))
            writer.write(protocol.create_response_packet(0, res.to_buffer()))
            return

        elif packet_id == protocol.PutModelPacketID:
            try:
                packet = protocol.PutModelPacket.from_buffer(payload)
            except Exception as e:
                writer.write(protocol.create_response_packet(400, str(e).encode()))
                return
    
            upd = (self.id, auth['cluster'], auth['key'], packet.meta, packet.data.to_buffer())

            writer.write(protocol.create_response_packet(0, b"OK"))
            self.update_queue.put(upd)
            return

        else:
            writer.write(protocol.create_response_packet(404, b"Unknown packet"))

    async def handle_auth(self, writer: asyncio.StreamWriter, token_buff: bytes): 
        try:
            auth = jwt.decode(token_buff, SECRET_KEY, algorithms=["HS256"])
            if "key" not in auth:
                raise jwt.InvalidTokenError("Token must contain a 'key' field")
            
            if "cluster" not in auth or type(auth["cluster"]) != int:   
                raise jwt.InvalidTokenError("Token must contain a 'cluster' field of type int")
            
            writer.write(protocol.create_response_packet(0, b"OK"))
            return auth
        except jwt.ExpiredSignatureError:
            writer.write(protocol.create_response_packet(401, b"Token expired"))
        except jwt.InvalidTokenError as e:
            writer.write(protocol.create_response_packet(401, str(e).encode()))
        return None

    async def read_packet(self, reader: asyncio.StreamReader) -> bool:
        packet_id_data = await reader.readexactly(2)   
        packet_id = struct.unpack("!H", packet_id_data)[0]
        packet_len_data = await reader.readexactly(4)
        packet_len = struct.unpack("!I", packet_len_data)[0]
        payload_data = await reader.readexactly(packet_len)
        return (packet_id, payload_data)
    

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        auth = None
        should_wait_close = True
        try:
            while True:
                packet_id, payload_data = await self.read_packet(reader)

                if not auth and packet_id != protocol.AuthPacketID:
                    writer.write(protocol.create_response_packet(401, "Unauthorized"))
                elif packet_id == protocol.AuthPacketID:
                    auth = await self.handle_auth(writer, payload_data)
                else:
                    await self.handle_packet(writer, packet_id, payload_data, auth)
                
                await writer.drain()

        except asyncio.IncompleteReadError:
            # print(f"Connection from {addr} closed unexpectedly")
            pass
        except ConnectionResetError:
            # print(f"Connection from {addr} reset by peer")
            should_wait_close = False
        finally:
            writer.close()
            if should_wait_close:
                await writer.wait_closed()

    async def poll_get(self, queue: SimpleQueue, interval: float = 0.1):
        while True:
            if not queue.empty():
                return queue.get()
            
            await asyncio.sleep(interval)

    async def gather_new_global_model(self):
        while True:
            try:
                # clusterID, id, model = await self.poll_get(self.gmodel_update_queue)
                update_type, data = await self.poll_get(self.gmodel_update_queue)
                if update_type == "model":
                    clusterID, id, model = data
                    self.registry.put(clusterID, id, model)

                elif update_type == "split":
                    clusterID, intoClusterID, one, two = data
                    self.registry.split(clusterID, intoClusterID, one, two)

            except asyncio.CancelledError:
                print(f"gather_new_global_model cancelled, exiting loop")
                return

    async def start_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((self.host, self.port))
        sock.listen(100)
        server = await asyncio.start_server(self.handle_client, sock=sock)

        asyncio.create_task(self.gather_new_global_model())

        print(f"Worker {self.id} started server on {self.host}:{self.port}")
        async with server:
            try:
                await server.serve_forever()
            except asyncio.CancelledError:
                pass

        print(f"Worker {self.id} stopped server")


class TCPServer:
    def __init__(self, host='127.0.0.1', port=8888, model_specs : nn.Module = None):
        assert model_specs is not None, "model_specs must be provided"
        self.Net = model_specs
        self.host = host
        self.port = port
        self.childs = []
    

    def start(self, num_workers: int = None):
        if num_workers is None:
            num_workers = os.cpu_count()

        upd_queue = SimpleQueue()
        gmodel_queues = [ SimpleQueue() for _ in range(num_workers) ]

        initial_model = self.Net().state_dict()
        for q in gmodel_queues:
            q.put(("model", (1, 1, initial_model)))

        def start_worker(id, h, p, uq, gmq):
            worker = TCPServerWorker(host=h,
                                        port=p,
                                        update_queue=uq,
                                        gmodel_update_queue=gmq,
                                        id=id)
            try:
                asyncio.run(worker.start_server())
            except KeyboardInterrupt:
                print(f"Worker {id} keyboard interrupted")

        self.childs = []
        for i in range(num_workers):
            p = Process(target=start_worker, args=(i, self.host, self.port, upd_queue, gmodel_queues[i]))
            p.start()
            self.childs.append(p)

        import aggregator

        def on_new_model(clusterID, version, model):
            for q in gmodel_queues:
                q.put(("model", (clusterID, version, model)))
        
        aggregators = [ aggregator.Aggregator(initial_model, on_new_model, 1)]

        try:
            while True:
                wid, cluster_id, client_key, meta, state_buffer = upd_queue.get()

                assert cluster_id <= len(aggregators), f"Cluster ID {cluster_id} not found"
                agg= aggregators[cluster_id - 1]

                split = agg.put(wid, client_key, meta, state_buffer)
                if split:
                    one, two = split
                    agg.gmodel = one
                    new_agg = aggregator.Aggregator(two, on_new_model, len(aggregators) + 1)
                    aggregators.append(new_agg)
                    print(f"Splitting cluster {cluster_id} into ({cluster_id}, {len(aggregators)})")

                    for q in gmodel_queues:
                        q.put(("split", (cluster_id, len(aggregators), one, two)))

        except KeyboardInterrupt:
            print("Parent process interrupted.")
        except Exception as e:
            print(f"Parent process exception: {e}")
        finally:
            for q in gmodel_queues:
                q.close()

            for p in self.childs:
                p.join()

        print("Parent process terminated.") 

def load_model_specs(model_path: str) -> nn.Module:
    if model_path.endswith(".py"):
        model_path = model_path[:-3]

    model_module = importlib.import_module(model_path)

    if not hasattr(model_module, "Model"):
        raise Exception(f"Model module {model_path} does not contain a 'Model' class")

    return model_module.Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start TCP Server')
    parser.add_argument('--port', type=int, default=8888, help='Port number')
    parser.add_argument('--model', type=str, help='Model path', required=True)

    args = parser.parse_args()
    module = load_model_specs(args.model)

    server = TCPServer(port=args.port, model_specs=module)
    server.start(num_workers=2)