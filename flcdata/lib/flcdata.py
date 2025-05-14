import numpy as np

from torch.utils.data import Dataset
import torch

class FLCDataset(Dataset):
    def __init__(self, data, targets, n_classes, dex="ds"):
        self.n_classes = n_classes
        self.n_features = data.shape[1]

        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.dex = dex


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def __load_npz__(ds_folder, train=True):
        name = "train" if train else "test"
        data = np.load(f"{ds_folder}/{name}.npz", allow_pickle=True)
      
        needed_keys = ["XX", "YY", "PP", "n_classes", "n_samples", "n_partitions"]
        file_keys = list(data.keys())

        # check if all needed keys are in the file
        for key in needed_keys:
            if key not in file_keys:
                raise ValueError(f"Key {key} not found in file. Available keys: {file_keys}")
            
        return {
            "XX": data["XX"],
            "YY": data["YY"],
            "PP": data["PP"],
            "n_classes": data["n_classes"],
            "n_samples": data["n_samples"],
            "n_partitions": data["n_partitions"],
        }
    

    def LoadGroups(ds_folder, train=True):
        data = FLCDataset.__load_npz__(ds_folder, train)
        dss = []
        for i in range(data["n_partitions"]):
            PP_indices = np.where(data["PP"] == i)[0]
            dss.append(
                FLCDataset(data["XX"][PP_indices], data["YY"][PP_indices], data["n_classes"], dex=f"ds-{i}")
            )
        return dss

    def LoadMerged(ds_folder, train=True):
        data = FLCDataset.__load_npz__(ds_folder, train)
        return FLCDataset(data["XX"], data["YY"], data["n_classes"], dex="ds-merged")
    
    def LoadSize(ds_folder):
        data = FLCDataset.__load_npz__(ds_folder, True)
        return (
            int(data["XX"].shape[1]),
            int(data["n_classes"]),
        )
    
from torch import nn
class SimpleModel(nn.Module):
    def __init__(self, insize=2, outsize=8):
        super(Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(insize, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, outsize),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)
    
class DeeperModel(nn.Module):
    def __init__(self, insize=2, outsize=8):
        super(DeeperModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(insize, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),  # Added layer
            nn.ReLU(),              # Added layer
            nn.Linear(256, 128),  # Added layer
            nn.ReLU(),              # Added layer
            nn.Linear(128, outsize),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)
    
class VeryDeepModel(nn.Module):
    def __init__(self, insize=2, outsize=8):
        super(VeryDeepModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(insize, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, outsize),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)
    
# Model = SimpleModel
Model = SimpleModel