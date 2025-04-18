
print(f"running {__file__}", flush=True)
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import time
import argparse
import asyncio

from torch.optim import SGD
from torch.nn import functional

from protocol import protocol, rpc
from core.client import ClientTask

import torch


from lib.flcdata import FLCDataset, Model

torch.set_num_threads(os.cpu_count())

parser = argparse.ArgumentParser(description="Run server")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8888)
parser.add_argument("--nclients", type=int, default=1)
parser.add_argument("--client_id", type=int, default=0)    
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--strategy", type=str, default="sync", choices=["sync", "async"], help="Synchronization strategy")

args = parser.parse_args()

class MemoryDataLoader:
    def __init__(self, data, targets, batch_size=1024, shuffle=False):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i:i + self.batch_size], self.targets[i:i + self.batch_size]

    def __len__(self):
        return len(self.data)


def dataset_to_device(dataset, device, batch_size=1024, shuffle=False):
    assert dataset.data.device != device, f"Dataset already on device: {dataset.data.device}"
    assert dataset.targets.device != device, f"Targets already on device: {dataset.targets.device}"

    dataset.data = dataset.data.to(device)
    dataset.targets = dataset.targets.to(device)

    return MemoryDataLoader(dataset.data, dataset.targets, batch_size=batch_size, shuffle=shuffle)


train_dss = FLCDataset.LoadGroups(ds_folder=args.dataset, train=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loaders = [dataset_to_device(ds, device) for ds in train_dss]


async def sync_client(r, w, ls, args, extra_args=None):
    momentum = 0.1
    lr = 0.01
    ephocs = 15
    net = Model(insize=train_dss[0].n_features, outsize=train_dss[0].n_classes)
    net.to(device)
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum)

    # from torch.amp import autocast, GradScaler
    # scaler = GradScaler("cuda")

    while True:
        res = await ls.signal(protocol.TrainEventID)
        if res is None:
            return

        model = await rpc.get_latest_model(w, r)
        model_data = model.model
        
        train_loader = train_loaders[args.client_id % len(train_loaders)]

        net.load_state_dict(model_data.model_state)
        net.to(device)

        for e in range(ephocs):
            net.train()
            for i, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()

                output = net(data)
                loss = functional.nll_loss(output, target)
                loss.backward()
                optimizer.step()

        model_meta = protocol.ModelMeta(
            momentum=momentum,
            learning_rate=lr,
            train_loss=0.1,
            test_loss=0.2,
            local_epoch=ephocs,
            train_samples=len(train_loader), 
        )

        net.to("cpu")
        model_data = net.state_dict()

        await rpc.put_model(w, r, protocol.ModelData(model_data), model_meta)



class RandomTicker:
    def __init__(self):
        self.waiters = []
    
    async def wait(self):
        asyncio.sleep(0)
        
        if len(self.waiters) == 0:
            return
        
        future = asyncio.get_running_loop().create_future()
        self.waiters.append(future)
        return await future
    
    def done(self):
        import numpy as np
        random_waiter = np.random.choice(self.waiters)
        self.waiters.remove(random_waiter)
        random_waiter.set_result(None)


async def async_client(r, w, ls, args, extra_args):
    momentum = 0.1
    lr = 0.01
    ephocs = 15
    net = Model(insize=train_dss[0].n_features, outsize=train_dss[0].n_classes)
    net.to(device)
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum)

    # from torch.amp import autocast, GradScaler
    # scaler = GradScaler("cuda")
    ticker = extra_args["ticker"]

    while True:
        await ticker.wait()
        model = await rpc.get_latest_model(w, r)
        model_data = model.model
        
        train_loader = train_loaders[args.client_id % len(train_loaders)]

        net.load_state_dict(model_data.model_state)
        net.to(device)

        for e in range(ephocs):
            net.train()
            for i, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()

                output = net(data)
                loss = functional.nll_loss(output, target)
                loss.backward()
                optimizer.step()

        model_meta = protocol.ModelMeta(
            momentum=momentum,
            learning_rate=lr,
            train_loss=0.1,
            test_loss=0.2,
            local_epoch=ephocs,
            train_samples=len(train_loader), 
        )

        net.to("cpu")
        model_data = net.state_dict()

        await rpc.put_model(w, r, protocol.ModelData(model_data), model_meta)
        ticker.done()

        await ls.signal(protocol.NewGlobalModelEventID)


# wait max 3 seconds for the server to start
async def wait_server(args):
    start_time = time.time()
    from protocol import tcp
    while time.time() - start_time < 5:
        try:
            r, w = await rpc.connect(args.host, args.port, protocol.Auth("client_0", 0, 0, 0))
            await tcp.safe_close(w)
            return True
            break
        except Exception as e:
            pass

        await asyncio.sleep(0.1)

    return False


async def __run(args):
    up = await wait_server(args)
    if not up:
        print("Server is not up, exiting!")
        return

    tasks = []
    
    import numpy as np
    clients = [c for c in range(args.nclients)]
    np.random.shuffle(clients)
    ticker = RandomTicker()

    extra_args = {
        "ticker": ticker,
    } if args.strategy == "async" else None

    for i in range(args.nclients):
        auth = protocol.Auth(f"client_{i}", 0, 0, 0)

        args = argparse.Namespace(**vars(args))
        args.client_id = i

        client_fn = sync_client if args.strategy == "sync" else async_client

        tasks.append(ClientTask(args, auth, client_fn,extra_args=extra_args))  # Directly append the coroutine

    await asyncio.gather(*tasks)  # Await all coroutines
    print("All tasks completed")

asyncio.run(__run(args))



