import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

import argparse
import os

import cubeds
import numpy as np
import torch    
from model import Model
import json


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
    parser.add_argument("path", help="Path to the simulation data")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise ValueError(f"Path '{args.path}' does not exist")
    
    if not os.path.isdir(args.path):
        raise ValueError(f"Path '{args.path}' is not a directory")

    versions = []
    for file in os.listdir(args.path):
        if file.endswith(".model"):
            version = int(file.split(".")[0])
            versions.append(version)

    versions = sorted(versions)
    metrics = []
    
    print(f"Found {len(versions)} models ({min(versions)} -> {max(versions)})")
    net = Model()
    ds_path = ".data/cuboids/total-skew"

    ds = cubeds.CuboidDataset.LoadMerged(ds_path, train=False)
    dss = cubeds.CuboidDataset.LoadGroups(ds_path, train=False)

    device = torch.device("cuda")
    loader = dataset_to_device(ds, device)
    loaders = [dataset_to_device(ds, device) for ds in dss]

    for v in versions:
        model_path = f"{args.path}/{v}.model"
        metrics_path = f"{args.path}/{v}.metrics"
        if os.path.exists(metrics_path):
            metrics.append(json.load(open(metrics_path)))
            continue

        dict = torch.load(model_path, weights_only=True)
        net.load_state_dict(dict)
        net.to(device)

        accuracy = test(net, loader)
        metric = {"version": v, "accuracy": accuracy, "groups": {}}

        for ds, loader in zip(dss, loaders):
            accuracy = test(net, loader)
            metric["groups"][ds.dex] = accuracy

        metrics.append(metric)
        json.dump(metric, open(metrics_path,
                                "w"), indent=4, sort_keys=True)


    print("Max accuracy:", max([m["accuracy"] for m in metrics]))
    print("Final accuracy:", metrics[-1]["accuracy"])

    fig, ax = plt.subplots(figsize=(10, 10))
    # plot on x the version number, on y the accuracy 
    ax.plot([m["version"] for m in metrics], [m["accuracy"] for m in metrics])
    for ds in dss:
        ax.plot([m["version"] for m in metrics], [m["groups"][ds.dex] for m in metrics], label=ds.dex, linestyle="--", marker="o")

    ax.set_xlabel("Version")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model accuracy over time")
    plt.show()








