import asyncio

import protocol.protocol as protocol
import protocol.tcp as tcp

from typing import Tuple

async def connect(host: str, port: int, auth: protocol.Auth) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    r, w = await asyncio.open_connection(host, port)
    w.write(protocol.create_packet(protocol.AuthPacketID, auth.encode()))
    await w.drain()
    return r, w
        

async def get_latest_model(w, r) -> protocol.GetModelPacketResponse:
    w.write(protocol.create_packet(protocol.GetModelPacketID, protocol.GetModelPacket(0).to_buffer()))
    await w.drain()
    packet_id, payload = await tcp.read_packet(r)
    if packet_id != 0:
        raise Exception(f"Failed to get model: {payload}")
    
    return protocol.GetModelPacketResponse.from_buffer(payload)

async def put_model(w, r, model: protocol.ModelData, meta: protocol.ModelMeta):
    putModelPacket = protocol.PutModelPacket(model, meta)
    w.write(protocol.create_packet(protocol.PutModelPacketID, putModelPacket.to_buffer()))
    await w.drain()
    
    status, msg = await tcp.read_packet(r)
    if status != 0:
        raise Exception(f"Failed to put model: {msg}")
    