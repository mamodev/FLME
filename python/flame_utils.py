
from torch.utils.data import DataLoader

def dataset_to_device(dataset, device, batch_size=1024, shuffle=False):
    assert dataset.data.device != device, f"Dataset already on device: {dataset.data.device}"
    assert dataset.targets.device != device, f"Targets already on device: {dataset.targets.device}"

    dataset.data = dataset.data.to(device)
    dataset.targets = dataset.targets.to(device)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def deepCloneSatetDict(model):
    return {k: v.detach().clone() for k, v in model.items()}