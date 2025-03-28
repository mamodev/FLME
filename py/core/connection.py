import asyncio

import protocol.protocol as protocol
import protocol.tcp as tcp

from dataclasses import dataclass

@dataclass
class Connection: 
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    auth: None | protocol.Auth = None
    upgrated: bool = False

    async def send_event(self, evt: int) -> bool:
        try:
            self.writer.write(protocol.create_packet(evt, b""))
            await self.writer.drain()
            return True
        except Exception:
            return False
        
    def get_conn_name(self):
        if self.auth:
            return f"[{self.auth.key}] ({'Listener' if self.upgrated else 'Client'})"

        return f"[Unknown] ({self.upgrated})"

    def write(self, data: bytes):
        # print(f"{self.get_conn_name()} Writing {len(data)} bytes")
        self.writer.write(data)

    async def close(self):
        # print(f"{self.get_conn_name()} Closing")
        await tcp.safe_close(self.writer)

    async def drain(self):
        # print(f"{self.get_conn_name()} Draining")
        await self.writer.drain()

    def __hash__(self):
        return hash(self.writer)
    
    async def broadcast_event(evt: int, connections) -> int:
        count = 0
        for conn in connections:
            if await conn.send_event(evt):
                count += 1
        return count