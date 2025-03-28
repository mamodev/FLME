from torch.utils.data import DataLoader
from torch import no_grad
from torch import nn
from torch.nn.functional import cross_entropy
from typing import Tuple

import importlib

def load_model_specs(model_path: str) -> nn.Module:
    if model_path.endswith(".py"):
        model_path = model_path[:-3]

    model_module = importlib.import_module(model_path)

    if not hasattr(model_module, "Model"):
        raise Exception(f"Model module {model_path} does not contain a 'Model' class")

    return model_module.Model

def deepCloneToCpu(model):
    return {k: v.detach().clone().cpu() for k, v in model.items()}

def dataset_to_device(dataset, device, batch_size=1024, shuffle=False):
    assert dataset.data.device != device, f"Dataset already on device: {dataset.data.device}"
    assert dataset.targets.device != device, f"Targets already on device: {dataset.targets.device}"

    dataset.data = dataset.data.to(device)
    dataset.targets = dataset.targets.to(device)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# reutrn test_loss, correct, total
def test(net, test_loader) -> Tuple[float, int, int]:
    net.eval()
    test_loss = 0
    correct = 0
    with no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, correct, len(test_loader.dataset)


def train(net, train_loader, optimizer, ephocs) -> float:
    net.train()
    running_loss = 0.0
    for e in range(ephocs):
        rl = 0.0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            rl += loss.item()

        running_loss += rl

    return running_loss / ephocs / len(train_loader.dataset)
        

    

