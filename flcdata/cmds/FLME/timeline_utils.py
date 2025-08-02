
from typing import List, Dict, Any
import torch
from torch.utils.data import DataLoader

def chash(client):
    return f"{client[0]}_{client[1]}"

def deep_clone_sdict(state_dict):
    return {k: v.clone() for k, v in state_dict.items()}

class MemoryDataLoader:
    def __init__(
        self, data, targets, batch_size=1024, shuffle=False,
        n_partitions=1, partition_idx=0
    ):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_partitions = n_partitions
        self.partition_idx = partition_idx

        # Calculate partition indices
        total_len = len(self.data)
        part_size = total_len // n_partitions
        remainder = total_len % n_partitions

        # Compute start and end indices for this partition
        self.start_idx = partition_idx * part_size + min(partition_idx, remainder)
        self.end_idx = self.start_idx + part_size
        if partition_idx < remainder:
            self.end_idx += 1

        # Slice the data for this partition
        self.partition_data = self.data[self.start_idx:self.end_idx]
        self.partition_targets = self.targets[self.start_idx:self.end_idx]

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.partition_data)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long)
            yield (
                self.partition_data[batch_indices],
                self.partition_targets[batch_indices]
            )
            
    def __len__(self):
        return len(self.partition_data)

