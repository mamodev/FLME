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
        self.global_models = {}
        self.latest_global_model_version = None

    async def handle_packet(self, writer: asyncio.StreamWriter, packet_id: int, payload: bytes, auth: Dict):
        assert auth != None, "auth must be provided"
        if packet_id == protocol.GetModelPacketID:
            try:
                packet = protocol.GetModelPacket.from_buffer(payload)
            except Exception as e:
                writer.write(protocol.create_response_packet(400, str(e).encode()))
                return

            if packet.model_version == 0:
                    packet.model_version = self.latest_global_model_version
            elif not packet.model_version in self.global_models:
                writer.write(protocol.create_response_packet(404, b"Model not found"))
                return


            model_data = protocol.ModelData(self.global_models[packet.model_version])
            buff = struct.pack("!I", packet.model_version)
            buff += model_data.to_buffer()

            writer.write(protocol.create_response_packet(0, buff))

        elif packet_id == protocol.PutModelPacketID:
            try:
                packet = protocol.PutModelPacket.from_buffer(payload)
            except Exception as e:
                writer.write(protocol.create_response_packet(400, str(e).encode()))
                return
    
            upd = (self.id, auth['key'], packet.meta, packet.data.to_buffer())

            writer.write(protocol.create_response_packet(0, b"OK"))
            self.update_queue.put(upd)
            return

        else:
            writer.write(protocol.create_response_packet(404, b"Unknown packet"))

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        print(f"[{self.id}] Connection from {addr}")

        auth = None
        should_wait_close = True
        try:
            while True:
                # Read PacketID (1 byte)
                packet_id_data = await reader.readexactly(2)
                packet_id = struct.unpack("!H", packet_id_data)[0]
                if(packet_id == 0):
                    print(f"Connection closed by {addr} with EOF")
                    break

                # Read PacketLen (4 bytes)
                packet_len_data = await reader.readexactly(4)
                packet_len = struct.unpack("!I", packet_len_data)[0]

                # Read Payload
                payload_data = await reader.readexactly(packet_len)
                # print(f"Received Packet from {addr}: ID={packet_id}, Length={packet_len}, payload_len={len(payload_data)}")

                if not auth and packet_id != protocol.AuthPacketID:
                    writer.write(protocol.create_response_packet(401, "Unauthorized"))
                elif packet_id == protocol.AuthPacketID:
                    try:
                        auth = jwt.decode(payload_data, SECRET_KEY, algorithms=["HS256"])
                        if "key" not in auth:
                            raise jwt.InvalidTokenError("Invalid token")

                        print(f"Authenticated: {auth}")
                        writer.write(protocol.create_response_packet(0, b"OK"))
                    except jwt.ExpiredSignatureError:
                        writer.write(protocol.create_response_packet(401, b"Token expired"))
                    except jwt.InvalidTokenError:
                        writer.write(protocol.create_response_packet(401, b"Invalid token"))
                else:
                    await self.handle_packet(writer, packet_id, payload_data, auth)
                
                await writer.drain()

        except asyncio.IncompleteReadError:
            print(f"Connection from {addr} closed unexpectedly")
        except ConnectionResetError:
            print(f"Connection from {addr} reset by peer")
            should_wait_close = False
        finally:
            writer.close()
            if should_wait_close:
                await writer.wait_closed()



        print(f"Connection from {addr} finally closed")


    async def poll_get(self, queue: SimpleQueue, interval: float = 0.1):
        while True:
            if not queue.empty():
                return queue.get()
            
            await asyncio.sleep(interval)

    async def gather_new_global_model(self):
        while True:
            try:
                id, model = await self.poll_get(self.gmodel_update_queue)

                # print(f"{self.id}: Received new global model {id}")
                self.global_models[id] = model
                self.latest_global_model_version = id
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
            q.put((1, initial_model))

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

        try:
            # cerate a state dict of zeros tensors
            current_model_update = {k: torch.zeros_like(v) for k, v in initial_model.items()}
            current_model_version = 0
            current_accumulated_updates = 0
            current_accumulated_updates_weight = 0

            while True:
                wid, client_key, meta, state_buffer = upd_queue.get()

                model = protocol.ModelData.from_buffer(state_buffer).model_state

                w = meta.train_samples
                current_accumulated_updates += 1
                current_accumulated_updates_weight += w

                for k, v in model.items():
                    current_model_update[k] += w * v

                if current_accumulated_updates >= 3:
                    current_model_version += 1
                    print(f"Updating global model to version {current_model_version}")

                    for k in current_model_update:
                        current_model_update[k] /= current_accumulated_updates_weight

                    for q in gmodel_queues:
                        q.put((current_model_version, current_model_update))

                    current_model_update = {k: torch.zeros_like(v) for k, v in initial_model.items()}
                    current_accumulated_updates = 0
                    current_accumulated_updates_weight = 0

        except KeyboardInterrupt:
            print("Parent process interrupted.")
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