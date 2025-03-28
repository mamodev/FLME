import json
import torch
import numpy as np
import torch.nn as nn

from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class CuboidDatasetMeta:
    n_groups: int
    n_labels: int
    n_void_groups: int
    space_size: int 
    n_samples_per_label: int
    test_prc: float
    groups_centers: List[List[Tuple[float, float, float]]]
    lable_swaps: Dict[int, Tuple[int, int]]
    features_skew: bool

    def get_name(self):
        out = f"g{self.n_groups}_l{self.n_labels}_v{self.n_void_groups}_s{self.space_size}_n{self.n_samples_per_label}_t{int(self.test_prc * 100)}"
        return out
    
    def save(self, path):
        META = {
            "ngroups": self.n_groups,
            "nlabels": self.n_labels,
            "nvoid": self.n_void_groups,
            "space_size": self.space_size,
            "nsamples": self.n_samples_per_label,
            "testprc": self.test_prc,
            "groups_centers": self.groups_centers,
            "lable_swaps": self.lable_swaps,
            "features_skew": self.features_skew
        }

        json.dump(META, open(path, "w"))    

    
    def load(path):   
        META = json.load(open(path, "r"))

        instance = CuboidDatasetMeta(
            n_groups=META["ngroups"],
            n_labels=META["nlabels"],
            n_void_groups=META["nvoid"],
            space_size=META["space_size"],
            n_samples_per_label=META["nsamples"],
            test_prc=META["testprc"],
            groups_centers=META["groups_centers"],
            lable_swaps=META["lable_swaps"],
            features_skew=META["features_skew"]
        )

        return instance


class VecClassifier(nn.Module):
    def __init__(self, input_dim, n_lables):
        super(VecClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_lables),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)
    
class CuboidDataset(torch.utils.data.Dataset):
    def __init__(self, lable_data, dex="Data", color='blue'):
        self.color = color  
        self.lables = 8
        assert len(lable_data) == 8, f"Provide a file for each label: provided {len(lable_data)} files"
        assert len(set([f[0] for f in lable_data])) == len(lable_data), "Labels must be unique"

        self.data = torch.tensor([])
        self.targets = torch.tensor([], dtype=torch.long)

        for label, vecs in lable_data:
            samples = []
            if type(vecs) != list:
                vecs = [vecs]

            for x in vecs:
                x = torch.tensor(x).float()
                samples.append(x)

            n_samples = sum([x.size(0) for x in samples])

            self.targets = torch.cat([self.targets, torch.tensor([label] * n_samples, dtype=torch.long)], dim=0)
            self.data = torch.cat([self.data, torch.cat(samples, dim=0)], dim=0)

            assert self.targets.size(0) == self.data.size(0), f"Data and targets size mismatch: {self.targets.size(0)} != {self.data.size(0)}"

        
        # Shuffle targers and data
        idx = torch.randperm(self.targets.size(0))
        self.targets = self.targets[idx]
        self.data = self.data[idx]
        self.dex = dex
        self.len = len(self.data)
 
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].long()
    
    def plot(self, ax):
        label_colors = ['blue', 'red', 'green', 'yellow', 'black', 'purple', 'orange', 'pink']

        # ax.set_xlim(-10, 10)
        # ax.set_ylim(-10, 10)
        # ax.set_zlim(-10, 10)
        ax.set_title(self.dex)
        idx = np.random.choice(self.len, 1000)
        pos = self.data[idx]
        lab = self.targets[idx]

        print(lab[0], type(lab[0]), lab[0].shape)
        colors = [label_colors[t] for t in lab]

        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=50)

    def info(self):
        s = f"CuboidDataset with {self.len} samples\n"
        return s
    


    def LoadGroup(ds_folder, g, train=True, partitioning=1, partitioning_start=None, partitioning_end=None):
        sub_folder = "train" if train else "test"
        META = CuboidDatasetMeta.load(f"{ds_folder}/META.json")

        assert g >= 0, f"Group {g} does not exist"
        assert g < META.n_groups, f"Group {g} does not exist"
        nlabels = META.n_labels

        assert partitioning > 0, f"Partitioning must be greater than 0"

        if partitioning_start is None:
            partitioning_start = 1

        if partitioning_end is None:
            partitioning_end = partitioning

        assert partitioning_start <= partitioning_end, f"Partitioning start must be less than partitioning end"
        assert partitioning_end <= partitioning, f"Partitioning end must be less than or equal to partitioning"
        assert partitioning_start > 0, f"Partitioning start must be greater than 0"
        assert partitioning_end > 0, f"Partitioning end must be greater than 0"

        vecs = [[] for _ in range(partitioning)]
        for l in range(nlabels):
            vec = np.load(f"{ds_folder}/{sub_folder}/g{g}_l{l}.npy")
            
            VEC_SIZE = len(vec)
            AVG_PVEC_SIZE = VEC_SIZE // partitioning

            for i in range(partitioning_start - 1, partitioning_end):
                startIdx = i * AVG_PVEC_SIZE
                endIdx = (i + 1) * AVG_PVEC_SIZE

                if i == partitioning - 1:
                    endIdx = VEC_SIZE

                slice = vec[startIdx: endIdx]
                vecs[i].append((l, slice))

            if partitioning_end == partitioning and partitioning_start == 1:
                total = sum([len(vecs[p][l][1]) for p in range(partitioning_start - 1, partitioning_end)])
                assert total == VEC_SIZE, f"Partitioning error: {total} != {VEC_SIZE}"
            

        ds = []
        for i in range(partitioning_start - 1, partitioning_end):
            ds.append(CuboidDataset(vecs[i], dex=f"{sub_folder}-g{g}-p{i}"))

        return ds
        

    # partitioning: 1 means no partitioning, 2 means that from 1 group 2 subdatasets are created
    def LoadGroups(ds_folder, train=True, partitioning=1, filter=None, partitioning_start=None, partitioning_end=None):
        sub_folder = "train" if train else "test"
        META = CuboidDatasetMeta.load(f"{ds_folder}/META.json")
        ngroups = META.n_groups
        nlabels = META.n_labels


        if partitioning_start is None:
            partitioning_start = 0

        if partitioning_end is None:
            partitioning_end = partitioning

        groups_colors = ['blue', 'red', 'green', 'yellow', 'black', 'purple', 'orange', 'pink']
        get_group_color = lambda g: groups_colors[g % len(groups_colors)]

        ds = []
        for g in range(ngroups):
            if filter is not None and g not in filter:
                continue

            vecs = [[] for _ in range(partitioning)]
            for l in range(nlabels):
                vec = np.load(f"{ds_folder}/{sub_folder}/g{g}_l{l}.npy")
                n = len(vec)
                n_per_ds = n // partitioning

                for i in range(partitioning_start, partitioning_end):
                   slice = vec[i * n_per_ds: (i + 1) * n_per_ds]
                   vecs[i].append((l, slice))
            
            for i in range(partitioning):
                ds.append(CuboidDataset(vecs[i], dex=f"{sub_folder}-g{g}-p{i}", color=get_group_color(g)))

        return ds

    def LoadMerged(ds_folder, train=True):
        sub_folder = "train" if train else "test"
        META = CuboidDatasetMeta.load(f"{ds_folder}/META.json")

        ngroups = META.n_groups
        nlabels = META.n_labels

        vecs = []
        for l in range(nlabels):
            label_vecs = []
            for g in range(ngroups):
                vec = np.load(f"{ds_folder}/{sub_folder}/g{g}_l{l}.npy")
                label_vecs.append(vec)

            vecs.append((l, label_vecs))

        return CuboidDataset(vecs)