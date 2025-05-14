#!/usr/bin/env python3

import argparse
import socket
import time
import sys
from typing import Tuple

DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 6969
DEFAULT_DURATION = 5.0  # seconds
START_BLOCK = 1024      # 1 KiB
MAX_BLOCK = 1024 * 1024 * 16  # 16 MiB


def timed_transfer_send(conn: socket.socket,
                        duration: float,
                        block_size: int) -> Tuple[int, float]:
    """
    Send as many bytes as possible in `duration` seconds in
    chunks of `block_size`. Returns (bytes_sent, elapsed).
    """
    buf = b'\0' * block_size
    end_time = time.monotonic() + duration
    total_sent = 0
    while time.monotonic() < end_time:
        conn.sendall(buf)
        total_sent += block_size
    elapsed = duration
    return total_sent, elapsed


def timed_transfer_recv(conn: socket.socket,
                        duration: float,
                        block_size: int) -> Tuple[int, float]:
    """
    Receive as many bytes as possible in `duration` seconds in
    chunks of `block_size`. Returns (bytes_received, elapsed).
    """
    end_time = time.monotonic() + duration
    total_recv = 0
    while time.monotonic() < end_time:
        to_read = block_size
        while to_read > 0:
            chunk = conn.recv(to_read)
            if not chunk:
                return total_recv, time.monotonic() + duration - end_time
            total_recv += len(chunk)
            to_read -= len(chunk)
    elapsed = duration
    return total_recv, elapsed


def run_server(host: str, port: int, duration: float) -> None:
    print(f"[server] Listening on {host}:{port}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        with conn:
            print(f"[server] Connection from {addr}")
            block_size = START_BLOCK
            while block_size <= MAX_BLOCK:
                # Phase 1: upload (server → client)
                sent, t1 = timed_transfer_send(conn, duration, block_size)
                mbps = sent / t1 / (1024*1024)
                print(f"[server] UP  block={block_size//1024}KiB "
                      f"bytes={sent}  time={t1:.2f}s  "
                      f"{mbps:.2f} MiB/s")

                # Phase 2: download (server ← client)
                rec, t2 = timed_transfer_recv(conn, duration, block_size)
                mbps2 = rec / t2 / (1024*1024)
                print(f"[server] DOWN block={block_size//1024}KiB "
                      f"bytes={rec}  time={t2:.2f}s  "
                      f"{mbps2:.2f} MiB/s")

                block_size *= 2


def run_client(host: str, port: int, duration: float) -> None:
    print(f"[client] Connecting to {host}:{port}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print("[client] Connected.")
        block_size = START_BLOCK
        while block_size <= MAX_BLOCK:
            # Phase 1: receive (server → client)
            rec, t1 = timed_transfer_recv(s, duration, block_size)
            mbps = rec / t1 / (1024*1024)
            print(f"[client] DOWN block={block_size//1024}KiB "
                  f"bytes={rec}  time={t1:.2f}s  "
                  f"{mbps:.2f} MiB/s")

            # Phase 2: send (client → server)
            sent, t2 = timed_transfer_send(s, duration, block_size)
            mbps2 = sent / t2 / (1024*1024)
            print(f"[client] UP  block={block_size//1024}KiB "
                  f"bytes={sent}  time={t2:.2f}s  "
                  f"{mbps2:.2f} MiB/s")

            block_size *= 2


def parse_args():
    p = argparse.ArgumentParser(
        description="Bandwidth benchmark: server or client mode, variable block size."
    )
    p.add_argument(
        "mode", choices=["server", "client"],
        help="Run as server (listen) or client (connect)."
    )
    p.add_argument(
        "--host", default=DEFAULT_HOST,
        help="Host to bind/listen (server) or connect to (client)."
    )
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help="TCP port number (default: 6969)."
    )
    p.add_argument(
        "--time", type=float, default=DEFAULT_DURATION,
        help="Duration in seconds for each test phase (default: 5s)."
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "server":
        run_server(args.host, args.port, args.time)
    else:
        run_client(args.host, args.port, args.time)


if __name__ == "__main__":
    main()
