import os
import asyncio
import socket
import struct
from typing import Tuple, List
from multiprocessing import SimpleQueue, Process

async def safe_close(writer: asyncio.StreamWriter):
    """Safely closes an asyncio StreamWriter."""
    if writer.is_closing():
        return  # Already closing, no need to proceed
    
    writer.close()
    try:
        await writer.wait_closed()  # Ensure proper closure
    except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
        pass  # Ignore errors from a lost connection

class WorkerHandlers:
    async def handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, out_ch: SimpleQueue):
        raise NotImplementedError("handle_conn must be implemented")
    
    async def handle_in_channel(self, msg, out_ch: SimpleQueue):
        raise NotImplementedError("handle_worker_in_channel must be implemented")

class ServerHandlers:
    def init(self, out_chs: List[SimpleQueue]):
        pass

    def handle_main_in_channel(self, msg, out_chs: List[SimpleQueue]):
        raise NotImplementedError("handle_main_in_channel must be implemented")
    

async def poll_get(queue: SimpleQueue, interval: float = 0.1):
        while True:
            if not queue.empty():
                return queue.get()
            
            await asyncio.sleep(interval)

async def read_packet(reader: asyncio.StreamReader) -> Tuple[int, bytes]:
    packet_id_data = await reader.readexactly(2)   
    packet_id = struct.unpack("!H", packet_id_data)[0]
    packet_len_data = await reader.readexactly(4)
    packet_len = struct.unpack("!I", packet_len_data)[0]
    payload_data = await reader.readexactly(packet_len)
    return (packet_id, payload_data)

class ServerWorker:
    id: int
    host: str
    port: int
    in_channel: SimpleQueue
    out_channel: SimpleQueue

    def __init__(self, id, host, port, in_channel, out_channel, worker_handlers: "WorkerHandlers"):
        assert in_channel is not None, "update_queue must be provided"
        assert out_channel is not None, "gmodel_update_queue must be provided"
        assert id is not None, "id must be provided"
        assert host is not None, "host must be provided"
        assert port is not None, "port must be provided"
        assert worker_handlers is not None, "worker_handlers must be provided"

        self.id = id
        self.host = host
        self.port = port
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.worker_handlers = worker_handlers
    
    async def in_ch_reader(self):
        while True:
            try:
                in_msg = await poll_get(self.in_channel)
                await self.worker_handlers.handle_in_channel(in_msg, self.out_channel)

            except asyncio.CancelledError:
                print(f"in_ch_reader cancelled, exiting loop")
                return
            
    async def __handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        await self.worker_handlers.handle_conn(reader, writer, self.out_channel)

    async def start_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((self.host, self.port))
        sock.listen(100)
        server = await asyncio.start_server(self.__handle_conn, sock=sock)
        asyncio.create_task(self.in_ch_reader())

        self.out_channel.put({
            "type": "internal",
            "event": "started",
            "worker_id": self.id
        })

        async with server:
            try:
                await server.serve_forever()
            except asyncio.CancelledError:
                pass

        print(f"Worker {self.id} stopped server")

class Server:
    def __init__(self, host, port, server_handlers: "ServerHandlers", worker_handlers: "WorkerHandlers"):
        assert host is not None, "host must be provided"
        assert port is not None, "port must be provided"
        assert server_handlers is not None, "server_handlers must be provided"
        assert worker_handlers is not None, "worker_handlers must be provided"

        self.host = host
        self.port = port
        self.childs = []

        self.in_channel = None
        self.out_channels = None

        self.server_handlers = server_handlers
        self.worker_handlers = worker_handlers
        
    def __is_running(self):
        return len(self.childs) > 0
    
    """"
    Start the  server with the specified number of workers (default is the number of CPUs)
    Returns a tuple of the update queue and the global model update queues (one for each worker)
    Calling start on already started server will raise an exception
    """
    def start(self, num_workers: int = None):
        assert not self.__is_running(), "Server is already running"

        if num_workers is None:
            num_workers = os.cpu_count()

        self.in_channel = SimpleQueue()
        self.out_channels = [ SimpleQueue() for _ in range(num_workers) ]

        def start_worker(id, h, p, out_ch, in_ch, wh):
            worker = ServerWorker(id, h, p, out_ch, in_ch, wh)
            try:
                asyncio.run(worker.start_server())
            except KeyboardInterrupt:
                print(f"Worker {id} keyboard interrupted")

        self.childs = []
        for i in range(num_workers):
            p = Process(target=start_worker, args=(i, self.host, self.port, self.out_channels[i], self.in_channel, self.worker_handlers))
            p.start()
            self.childs.append(p)

        try:
            for _ in range(num_workers):
                msg = self.in_channel.get()
                assert msg["type"] == "internal" and msg["event"] == "started", "Worker did not start properly, unexpected message {msg}"

            print(f"Server started on {self.host}:{self.port} with {num_workers} workers")

            self.server_handlers.init(self.out_channels)
            while True:
                msg = self.in_channel.get()
                self.server_handlers.handle_main_in_channel(msg, self.out_channels)
        except KeyboardInterrupt:
            print("Main server keyboard interrupted")
        except Exception as e:
            print(f"Main server stopped with exception: {e}")
        finally:
            self.stop()

        print("Main server stopped")

    def stop(self):
        for out_ch in self.out_channels:
            out_ch.close()

        for p in self.childs:
            p.terminate()

        for p in self.childs:
            p.join()

        self.childs = []