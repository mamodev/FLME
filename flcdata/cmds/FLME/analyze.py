print(f"running {__file__}", flush=True)
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import torch    
import json
from lib.flcdata import Model, FLCDataset

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

def test(net, loader):
    if len(loader) == 0:
        print("=="*20)
        print("Empty loader")
        print("=="*20)
        return 0

    net.eval()
    with torch.no_grad():
        correct = 0
        total = len(loader.data)


        for data, target in loader:
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        return correct / total


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze the data")
    parser.add_argument("--sim-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.sim_dir):
        raise ValueError(f"Path '{args.sim_dir}' does not exist")
    
    if not os.path.isdir(args.sim_dir):
        raise ValueError(f"Path '{args.sim_dir}' is not a directory")

    versions = []
    for file in os.listdir(args.sim_dir):
        if file.endswith(".model"):
            version = int(file.split(".")[0])
            versions.append(version)

    versions = sorted(versions)
    metrics = []
    
    print(f"Found {len(versions)} models ({min(versions)} -> {max(versions)})")

    dss = FLCDataset.LoadGroups(ds_folder=args.dataset_dir, train=False)
    ds = FLCDataset.LoadMerged(ds_folder=args.dataset_dir, train=False)
    net = Model(insize=ds.n_features, outsize=ds.n_classes)

    device = torch.device("cuda")
    loader = dataset_to_device(ds, device)
    loaders = [dataset_to_device(ds, device) for ds in dss]

    for v in versions:
        model_path = f"{args.sim_dir}/{v}.model"
        metrics_path = f"{args.sim_dir}/{v}.metrics"
        if os.path.exists(metrics_path):
            metrics.append(json.load(open(metrics_path)))
            continue

        _to = torch.load(model_path, weights_only=False)
        model = _to["model"]
        __metrics = _to["metrics"]

        net.load_state_dict(model)
        net.to(device)

        accuracy = test(net, loader)


        metric = {"version": v, "accuracy": accuracy, "groups": {},
                "contributors": __metrics["contributors"],
        }



        for ds, loader in zip(dss, loaders):
            accuracy = test(net, loader)
            metric["groups"][ds.dex] = accuracy

        metrics.append(metric)
        json.dump(metric, open(metrics_path,
                                "w"), indent=4, sort_keys=True)


    print(f"{__file__} - Finished analyzing models")