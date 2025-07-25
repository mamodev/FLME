import socket
import time
import argparse
import sys
import os
import struct
from typing import  Union


def dict_packet(**kwargs) -> bytes:
    packet = ""
    for key, value in kwargs.items():
        packet += f"{key}={value}\n"
    return raw_packet(packet)

def raw_packet(packet: str) -> bytes:
    packet = packet.encode("utf-8") if isinstance(packet, str) else packet
    packet_length = len(packet)
    packet_length_bytes = struct.pack("!I", packet_length)
    packet_with_length = packet_length_bytes + packet
    return packet_with_length

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
    
def untill_connected(host: str, port: int, name: str, role="slave", interval: int = .5) -> socket.socket:
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


# flcdata-slave.py --host mamodeh.ddns.net --port 6969 --name slave1

parser = argparse.ArgumentParser(description="FLCdata slave")
parser.add_argument("--host", type=str, help="Host to connect to")
parser.add_argument("--port", type=int, help="Port to connect to")
parser.add_argument("--name", type=str, help="Name of the slave")
args = parser.parse_args()


if args.host is None or args.port is None or args.name is None:
    print("Usage: flcdata-slave.py --host <host> --port <port> --name <name>")
    sys.exit(1)


host = args.host
port = args.port
name = args.name


def transfer_file(remote_path: str, local_path: str) -> None:
    folder = os.path.dirname(local_path)
    if folder != "" and not os.path.exists(folder):
        os.makedirs(folder)

    s = untill_connected(host, port, name, role="file-transfer")
    s.sendall(dict_packet(path=remote_path))

    local_file = open(local_path, "wb")
    file_size = s.recv(4)
    file_size = struct.unpack("!I", file_size)[0]
    print(f"Receiving file {remote_path} ({file_size} bytes)")

    rcved = 0
    while rcved < file_size:    
        data = s.recv(file_size - rcved)
        if not data:
            break
        local_file.write(data)
        rcved += len(data)

    local_file.close()
    print(f"File {remote_path} received and saved to {local_path}")
    


while True:
    s = None
    __RESPONSE__ = "status=ok\n"
    try:
        s = untill_connected(host, port, name)
        size = s.recv(4)
        size = struct.unpack("!I", size)[0]
        data = recv_all(s, size)
        str_data = data.decode("utf-8")
        

        try:
            exec(str_data)
        except Exception as e:
            __RESPONSE__ = f"status=error\nerror={e}\n"
            pass

        res_len = len(__RESPONSE__)
        res_len_bytes = struct.pack("!I", res_len)
        s.sendall(res_len_bytes + __RESPONSE__.encode("utf-8"))

    except KeyboardInterrupt:
        print("Exiting...")
        if s:
            s.close()
        break
    except Exception as e:
        print(f"Error: {e}")
        if s:
            s.close()