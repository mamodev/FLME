

import argparse
import asyncio

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import functional

from protocol import protocol, rpc
from core.client import ClientTask

from model import Model
from cubeds import CuboidDataset

path = ".data/cuboids/total-skew"

import torch

import os


torch.set_num_threads(os.cpu_count())


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

train_dss = CuboidDataset.LoadGroups(ds_folder=path, train=True)
test_ds = CuboidDataset.LoadMerged(ds_folder=path, train=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loaders = [dataset_to_device(ds, device) for ds in train_dss]
test_loader = dataset_to_device(test_ds, device)


async def client(r, w, ls, args):
    momentum = 0
    lr = 0.001
    ephocs = 3
    net = Model()
    net.to(device)
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum)

    # from torch.amp import autocast, GradScaler
    # scaler = GradScaler("cuda")

    while True:
        await ls.signal(protocol.TrainEventID)

        model = await rpc.get_latest_model(w, r)
        model_data = model.model
        
        train_loader = train_loaders[args.client_id % len(train_loaders)]

        # print size of ds
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

                # with autocast("cuda"):
                #     output = net(data)
                #     loss = functional.nll_loss(output, target)  

                # scaler.scale(loss).backward()   
                # scaler.step(optimizer)
                # scaler.update()

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run server")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--nclients", type=int, default=1)
    parser.add_argument("--client_id", type=int, default=0)    

    args = parser.parse_args()

    tasks = []
    for i in range(args.nclients):
        auth = protocol.Auth(f"client_{i}", 0, 0, 0)

        args = argparse.Namespace(**vars(args))
        args.client_id = i

        tasks.append(ClientTask(args, auth, client))  # Directly append the coroutine

    async def __run(tasks):
        await asyncio.gather(*tasks)  # Await all coroutines

    asyncio.run(__run(tasks))