import argparse
import torch    
import json
import math
import os

from flcdata import FLCDataset
from nets import MODELS

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


def dataset_to_device(dataset, device, batch_size=4096, shuffle=False):
    assert dataset.data.device != device, f"Dataset already on device: {dataset.data.device}"
    assert dataset.targets.device != device, f"Targets already on device: {dataset.targets.device}"

    dataset.data = dataset.data.to(device)
    dataset.targets = dataset.targets.to(device)

    return MemoryDataLoader(dataset.data, dataset.targets, batch_size=batch_size, shuffle=shuffle)

def test(net, loader):
    total = len(loader.data)

    if total == 0:
        print("=="*20)
        print("Empty loader")
        print("=="*20)
        return -1

    net.eval()
    with torch.no_grad():
        correct = 0
        t = 0

        for data, target in loader:
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            t += len(data)

        
        assert t == total, f"Total {total} != {t}"

        return correct / total

def sort_of_equal(a, b):
    return abs(a - b) < 0.001


def analyze(net_name, simdir, datasetdir, seed=0):
    Model = MODELS[net_name]
    print(f"Analyzing models in {simdir} with dataset {datasetdir} and seed {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if not os.path.exists(simdir):
        raise ValueError(f"Path '{simdir}' does not exist")
    
    if not os.path.isdir(simdir):
        raise ValueError(f"Path '{simdir}' is not a directory")

    versions = []
    for file in os.listdir(simdir):
        if file.endswith(".model"):
            version = int(file.split(".")[0])
            versions.append(version)

    versions = sorted(versions)
    metrics = []

    print(f"Found {len(versions)} models ({min(versions)} -> {max(versions)})")
    
    dss = FLCDataset.LoadGroups(ds_folder=datasetdir, train=False)

    # ds = FLCDataset.LoadMerged(ds_folder=datasetdir, train=False)
    # assert len(ds) == sum([len(ds) for ds in dss]), "Merged dataset size does not match the sum of group sizes"
    # print(f"Loaded {len(ds)} samples from {len(dss)} groups")


    net = Model(insize=dss[0].n_features, outsize=dss[0].n_classes)

    # create info.json contining various metadata
    info = {
        "dataset_path": datasetdir,
        "model_type": net_name,
        # "partition_nsamples": [len(ds) for ds in dss],
        # "partition_nsamples_per_class": [ [ len(ds.targets == c) for c in range(ds.n_classes) ] for ds in dss ]
        "partition_nsamples": {
            ds.dex: len(ds) for ds in dss
        },
        "partition_nsamples_per_class": {
            # ds.dex: [len(ds.targets == c) for c in range(ds.n_classes)] for ds in dss
            # len(ds.targets == c) not really working should use different indexing
            ds.dex: [len(ds.targets[ds.targets == c]) for c in range(ds.n_classes)] for ds in dss
        }
    }
    
    
    json.dump(info, open(f"{simdir}/info.json", "w"), indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = [dataset_to_device(ds, device) for ds in dss]
    
 

    for v in versions:
        model_path = f"{simdir}/{v}.model"
        metrics_path = f"{simdir}/{v}.metrics"
        if os.path.exists(metrics_path):
            metrics.append(json.load(open(metrics_path)))
            continue

        _to = torch.load(model_path, weights_only=False)

        model = _to
        __metrics = {
            "contributors": [],
        }

        # check if is a dict
        if isinstance(_to, dict) and "model" in _to:
            model = _to["model"]
            if "metrics" in _to:
                __metrics = _to["metrics"]

        net.load_state_dict(model)
        net.to(device)

        # accuracy = test(net, loader)

        metric = {"version": v, "accuracy": 0, "groups": {},
                "contributors": __metrics["contributors"],
        }


        acc2 = 0
        for _ds, _l in zip(dss, loaders):
            acc = test(net, _l)
            metric["groups"][_ds.dex] = acc
            acc2 += acc 

        acc2 /= len(dss)

        metric["accuracy"] = acc2

        # if not sort_of_equal(acc2, accuracy):
        #     print(f"Accuracy mismatch: {accuracy} != {acc2}")
        #     raise ValueError(f"Accuracy mismatch: {accuracy} != {acc2}")


        metrics.append(metric)
        json.dump(metric, open(metrics_path,
                                "w"), indent=4, sort_keys=True)


    print(f"{__file__} - Finished analyzing models")
    

if __name__ == "__main__":
    models_options=list(MODELS.keys())
    
    parser = argparse.ArgumentParser(description="Analyze the data")
    parser.add_argument("--net", type=str, choices=models_options, required=True, help="Network architecture to use")
    parser.add_argument("--sim-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--nuke-prev-metrics", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.nuke_prev_metrics:
        for file in os.listdir(args.sim_dir):
            if file.endswith(".metrics"):
                os.remove(os.path.join(args.sim_dir, file))

    analyze(args.net, args.sim_dir, args.dataset_dir, args.seed)


# python3 tools/simulation/analyze.py --sim-dir prova --dataset-dir data/a1812-06127_synth_00