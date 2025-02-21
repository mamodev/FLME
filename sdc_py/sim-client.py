import argparse
import time
import torch
import protocol
from torch_utils import dataset_to_device, test, train

parser = argparse.ArgumentParser(description='Client for SDC')
parser.add_argument('host', type=str, help='host to connect to, e.g. localhost:1234 or :1234')
parser.add_argument('-c', type=str, help='number of clusters to use example:  --clusters (<gid>,<partitions>,<start_from>,<end_at>)',
                     action='append',
                     nargs='+', required=True)

args = parser.parse_args()

if not ':' in args.host:
    args.host = f":{args.host}"

host = args.host.split(':')
port = int(host[1])
host = host[0]
clusters = args.c
assert all([len(c) == 4 for c in clusters]), "All clusters must have 4 values (gid, partitions, start_from, end_at)"

# SIM - IMPORTS
from model import Model
from cubeds import CuboidDataset
import numpy as np

# path = "./data/cuboids/cuboids"
path = "./data/cuboids/ilswap-skew"

train_dss = [CuboidDataset.LoadGroup(ds_folder=path, 
                                     g=int(c[0]), 
                                     train=True, 
                                     partitioning=int(c[1]), 
                                     partitioning_start=int(c[2]), 
                                     partitioning_end=int(c[3]))
                                     for c in clusters]

train_dss = [item for sublist in train_dss for item in sublist]

# train_dss = CuboidDataset.LoadGroups(ds_folder=path, train=True)
test_ds = CuboidDataset.LoadMerged(ds_folder=path, train=False)

train_loaders = [dataset_to_device(ds, 'cuda') for ds in train_dss]
test_loader = dataset_to_device(test_ds, 'cuda')

lr = 0.01
momentum = 0 
ephocs = 3

while True:
    idx = np.random.randint(0, len(train_dss))
    train_ds = train_dss[idx]
    train_loader = train_loaders[idx]

    net = Model()
    net.to('cuda')
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    s = protocol.connect_with_auth(host, port, train_ds.gid, train_ds.pid)
    model_version, model = protocol.get_latest_model(s)

    net.load_state_dict(model.model_state)
    train_loss = train(net, train_loader, optimizer, ephocs)
    test_loss, correct, total = test(net, test_loader)

    print(f"Test loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({100. * correct / total:.1f}%), Train loss: {train_loss:.4f}")

    model_meta = protocol.ModelMeta(
        momentum=momentum,
        learning_rate=lr,
        train_loss=train_loss,
        test_loss=test_loss,
        local_epoch=ephocs,
        train_samples=len(train_ds),
        derived_from=model_version
    )

    protocol.put_model(s, net.state_dict(), model_meta)
    # time.sleep(.2)