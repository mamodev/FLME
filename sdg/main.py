


from flcdata import FLCDataset, SimpleModel, DeeperModel, VeryDeepModel, SimpleBatchLoader
import torch
from torch.nn import functional as F


    
def deep_clone_state_dict(state_dict):
    return {k: v.clone() for k, v in state_dict.items()}

ds_path = ".data/cubeds-drift"

insize, outsize = FLCDataset.LoadSize(ds_path)

print(f"Input size: {insize}, Output size: {outsize}")

dss = FLCDataset.LoadGroupsAndLinearPartition(
    ds_path,
    10,
    shuffle=False,
)

test_dss = FLCDataset.LoadGroupsAndLinearPartition(
    ds_path,
    10,
    shuffle=False,
    train=False,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
for i in range(len(dss)):
    dss[i].to_device(device)
    test_dss[i].to_device(device)


loaders = [
    SimpleBatchLoader(ds.data, ds.targets, batch_size=2048)
    for ds in dss
]

test_loaders = [
    SimpleBatchLoader(ds.data, ds.targets, batch_size=2048)
    for ds in test_dss
]

Model = SimpleModel

nets = [
    Model(insize=insize, outsize=outsize).to(device)
    for _ in range(len(dss))
]

print(f"Total number of clients after partitioning: {len(dss)}")


start_model = nets[0].state_dict()
for i in range(1, len(nets)):
    nets[i].load_state_dict(deep_clone_state_dict(start_model))

rounds = 1000 
lr = 0.1
ephocs = 30
momentum = 0.9

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_ylim(0, 1)
plt.show(block=False)
plt.pause(0.1)


accuracies = [[] for _ in range(len(dss))]

for round in range(rounds):
    lr = 0.001

    models = []
    for i, ds in enumerate(dss):
        model = nets[i]
        model.train()
      
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        running_loss = 0.0
        num_batches = 0
        for epoch in range(ephocs):
            for x, y in loaders[i]:
                optimizer.zero_grad()
                output = model(x)
                loss = F.nll_loss(output, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

        avg_loss = running_loss / num_batches

    
        models.append((model.state_dict(), ds.n_samples, avg_loss))

    # Aggregate the models using FedAvg
    new_model = {}
    for k in models[0][0].keys():
        new_model[k] = torch.zeros_like(models[0][0][k])
        for model, n_samples, _ in models:
            new_model[k] += model[k] * (n_samples / sum([m[1] for m in models]))

    start_model = new_model

    test_accs = []

    for i, ds in enumerate(test_dss):
        model = nets[i]
        model.load_state_dict(deep_clone_state_dict(new_model))
        model.eval()
    
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for x, y in test_loaders[i]:
                output = model(x)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)


    net = SimpleModel(insize=insize, outsize=outsize).to(device)
    net.load_state_dict(new_model)
    total = 0
    correct = 0
    for i, ds in enumerate(test_dss):

        net.eval()
        with torch.no_grad():
            for x, y in test_loaders[i]:
                output = net(x)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

    oracle_acc = 100 * correct / total
    
    avg_test = sum(test_accs) / len(test_accs)
    avg_loss = sum([m[2] for m in models]) / len(models)

    var_test = sum([(x - avg_test) ** 2 for x in test_accs]) / len(test_accs)
    var_loss = sum([(x - avg_loss) ** 2 for x in [m[2] for m in models]]) / len(models)

    print(
        f"Round {round + 1}/{rounds} | "
        f"Avg Test Accuracy: {avg_test:.2f}% ({var_test:.2f}) | "
        f"Avg Loss: {avg_loss:.4f} ({var_loss:.4f}) | "
        f"Oracle Accuracy: {oracle_acc:.2f}% | "
    )   


    for i, acc in enumerate(test_accs):
        accuracies[i].append(acc / 100)

    ax.clear()

    ax.set_ylim(0, 1)
    for i, acc in enumerate(accuracies):
        ax.plot(range(len(acc)), acc, label=f"Client {i+1}", alpha=0.5)


    plt.draw()
    plt.pause(0.1)

plt.pause(0.1)
plt.show(block=True)