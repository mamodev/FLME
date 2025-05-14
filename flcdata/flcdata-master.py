import socket
import time
import argparse
import sys
import os
import struct
from typing import  Union
import asyncio

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

# flcdata-masyer.py --port 6969

parser = argparse.ArgumentParser(description="FLCdata master")
parser.add_argument("--port", type=int, help="Port to connect to")
args = parser.parse_args()

if args.port is None:
    print("Usage: flcdata-master.py --port <port>")
    sys.exit(1)

port = args.port
host = "0.0.0.0"

clients = {

}


async def send_message():
    client_name = input("Enter client name: ")
    if client_name not in clients:
        print(f"Client {client_name} not found")
        return

    message = input("Enter message: ")
    queue = clients[client_name]["queue"]

    data = message.encode("utf-8")
    size = struct.pack("!I", len(data))

    fut = asyncio.Future()

    async def callback(data):
        fut.set_result(data.decode("utf-8"))
    
    queue.put_nowait((size + data, callback))


    res = await fut
    print(f"Response from {client_name}: {res}")


CMDS = {
    "help": ("Show this help message", lambda: print("\n".join([f"{k}: {v[0]}" for k, v in CMDS.items()]))),
    "exit": ("Exit the program", lambda: asyncio.get_event_loop().stop()),
    "list": ("List all connected clients", lambda: print("\n".join([f"{k}: {v['auth']}" for k, v in clients.items()]))),
    "clear": ("Clear the screen", lambda: os.system("cls" if os.name == "nt" else "clear")) ,
    "send": ("Send a message to a client", send_message),
}

async def stdio_loop():
    print("Enter commands (type 'help' for available commands, 'exit' to quit):")
    while True:
        command = await asyncio.to_thread(input, "> ")
        if command in CMDS:
            res =  CMDS[command][1]()
            # check if res should be awaited
            if asyncio.iscoroutinefunction(res) or asyncio.iscoroutine(res) or asyncio.isfuture(res) or isinstance(res, asyncio.Future):
                await res
        else:
            print(f"Unknown command {command}. Type 'help' for available commands.")


async def slave_loop(client_name: str, queue: asyncio.Queue, writer: asyncio.StreamWriter, reader: asyncio.StreamReader):
    try:
        while True:
            req = await queue.get()
            if req is None:
                break

            packet, callback = req
            writer.write(packet)
            await writer.drain()

            size = await reader.readexactly(4)
            size = struct.unpack("!I", size)[0]
            data = await reader.readexactly(size)

            asyncio.create_task(callback(data))
    finally:
        writer.close()
        await writer.wait_closed()


# start asyncio loop
async def start_server(host: str, port: int):
    server = await asyncio.start_server(handle_client, host, port)

    asyncio.create_task(stdio_loop())
    
    print(f"Serving on {host}:{port}")
    async with server:
        await server.serve_forever()


async def send_file(auth, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    print(f"Sending file")
    packet_size = await reader.readexactly(4)
    packet_size = struct.unpack("!I", packet_size)[0]
    print(f"Packet size: {packet_size}")
    packet = await reader.readexactly(packet_size)
    packet = packet.decode("utf-8")
    params = {}
    for line in packet.split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            params[key] = value

    print(f"Params: {params}")

    path = params["path"]

    # straeam file

    if not os.path.exists(path):
        #todo send error
        pass

    if not os.path.isfile(path):
        #todo send error
        pass
    
    if not os.access(path, os.R_OK):
        #todo send error
        pass

    file_size = os.path.getsize(path)

    print(f"Sending file {path} ({file_size} bytes)")

    file_size_bytes = struct.pack("!I", file_size)
    writer.write(file_size_bytes)
    await writer.drain()

    with open(path, "rb") as f:
        while True:
            data = f.read(1024)
            if not data:
                break
            writer.write(data)
            await writer.drain()

    writer.close()
    return


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


    writer.write(b"1")
    await writer.drain()


    if "role" in auth and auth["role"] == "slave":
        queue = asyncio.Queue()
        clients[auth["name"]] = {
            "auth": auth,
            "queue": queue,
        }

        asyncio.create_task(slave_loop(auth["name"], queue, writer, reader))
        return
    
    if "role" in auth and auth["role"] == "file-transfer":
        await send_file(auth, reader, writer)
    else:
        writer.write(b"2")
        await writer.drain()
        writer.close()


   
if __name__ == "__main__":
    asyncio.run(start_server(host, port))

    #transfer_file("tsconfig.json", "t.json")