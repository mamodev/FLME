import asyncio

from core.connection import Connection
import protocol.protocol as protocol
import protocol.tcp as tcp
from utils.observable import ObservableSet
import utils.logs as logs



class Server:
    def __init__(self, host, port, strategy, repo):
        self.host = host
        self.port = port
        self.server = None
        self.repo = repo

        def __strat_wrapper(strat):
            async def __strat(s):
                try:
                    await strat(s)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logs.print_except(e)
            
            return __strat
        
        self.strategy = __strat_wrapper(strategy)

        self.listeners = ObservableSet()
        self.updates = ObservableSet()


    async def run(self):


        self.server = await asyncio.start_server(self.handle_connection, self.host, self.port)
    
        strategy_task = asyncio.create_task(self.strategy(self))
        await self.server.serve_forever()

        await asyncio.gather(strategy_task)

    
    async def handle_packet(self, conn: Connection, packet_id: int, payload: bytes):
            if packet_id == protocol.GetModelPacketID:
                try:
                    packet = protocol.GetModelPacket.from_buffer(payload)
                except Exception as e:
                    conn.write(protocol.create_response_packet(400, str(e).encode()))
                    return

                model, version, err = self.repo.get_model(packet.model_version)
                if err:
                    conn.write(protocol.create_response_packet(404, err.encode()))
                    return
                
                res = protocol.GetModelPacketResponse(version, protocol.ModelData(model))
                conn.write(protocol.create_response_packet(0, res.to_buffer()))
                return

            elif packet_id == protocol.PutModelPacketID:
                try:
                    packet = protocol.PutModelPacket.from_buffer(payload)
                except Exception as e:
                    conn.write(protocol.create_response_packet(400, str(e).encode()))
                    return

                self.updates.add(packet)
                conn.write(protocol.create_response_packet(0, b""))
                return
            else:
                conn.write(protocol.create_response_packet(404, b"Unknown packet"))

    async def handle_connection(self, _r: asyncio.StreamReader, _w: asyncio.StreamWriter):
        conn = Connection(_r, _w)
        try:
            while True:
                packet_id, payload_data = await tcp.read_packet(conn.reader)
                assert not conn.upgrated, "Connection already upgrated"

                try:
                    if not conn.auth and packet_id != protocol.AuthPacketID:
                        raise Exception("Unauthorized")

                    if packet_id == protocol.AuthPacketID:
                        conn.auth = protocol.Auth.decode(payload_data)
                    elif packet_id == protocol.UpgrateToListener:
                        conn.upgrated = True
                        self.listeners.add(conn)
                    else:
                        await self.handle_packet(conn, packet_id, payload_data)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    conn.write(protocol.create_response_packet(500, str(e).encode()))
                finally:
                    await conn.drain()

        except asyncio.CancelledError:
            pass
        except asyncio.IncompleteReadError:
            pass
        except ConnectionResetError:
            pass
        except BrokenPipeError:
            pass
        except Exception as e:
            logs.print_except(e)
        finally:
            if conn.upgrated:
                self.listeners.remove(conn)

            await conn.close()