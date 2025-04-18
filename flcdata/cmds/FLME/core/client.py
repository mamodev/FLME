
import asyncio
from core.listener import Listener
from protocol import rpc, tcp
from utils import logs

async def ClientTask(args, auth, __client, extra_args=None):
    try:
        listener_task = None
        r = None
        w = None
        
        r, w = await rpc.connect(args.host, args.port, auth)
        ls = await Listener.new_listener(auth, args.host, args.port)
        listener_task = asyncio.create_task(ls.listen())

        await __client(r, w, ls, args, extra_args)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logs.print_except(e)
    finally:
        if listener_task is not None:
            listener_task.cancel()
            await listener_task
        if w is not None:
            await tcp.safe_close(w) 

    
