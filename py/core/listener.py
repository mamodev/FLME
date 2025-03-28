import asyncio
from utils import logs
from protocol import tcp, protocol

class Listener:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.listeners = dict()

    async def signal(self, signal: int):
        future = asyncio.get_running_loop().create_future()

        if not signal in self.listeners:
            self.listeners[signal] = []

        self.listeners[signal].append(future)

        return await future
    
    async def listen(self):
        print("Listening")
        try:
            while True:
                packet_id, payload_data = await tcp.read_packet(self.reader)
                if packet_id in self.listeners:
                    for future in self.listeners[packet_id]:
                        future.set_result(packet_id)
                    self.listeners[packet_id] = []
        
        except asyncio.CancelledError:
            pass
        except asyncio.IncompleteReadError:
            pass
        except Exception as e:
            logs.print_except(e)
        finally:
            await tcp.safe_close(self.writer)
            print("Listener stopped")


    async def new_listener(auth: protocol.Auth, host: str, port: int):
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(protocol.create_packet(protocol.AuthPacketID, auth.encode()))
        writer.write(protocol.create_packet(protocol.UpgrateToListener, b""))
        await writer.drain()

        return Listener(reader, writer)