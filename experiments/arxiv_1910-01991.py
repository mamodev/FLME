import torch
import cubeds
import flame_utils
import os
import sys

# ds_folder = "data/cuboids/noskew-no-bias"
# ds_folder = "data/cuboids/fskew"

ds_folder = sys.argv[1]
if not os.path.exists(ds_folder):
    print(f"Folder {ds_folder} does not exist")
    sys.exit(1)


Partitioning = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meta = cubeds.CuboidDatasetMeta.load(f"{ds_folder}/META.json")
def partitioning(g):
    return g * 2 if g > 0 else 1

dss = [cubeds.CuboidDataset.LoadGroup(ds_folder, g, partitioning=partitioning(g)) for g in range(meta.n_groups)]
dss = [d for ds in dss for d in ds]

test_ds = cubeds.CuboidDataset.LoadMerged(ds_folder, train=False)
test_loader = flame_utils.dataset_to_device(test_ds, device, batch_size=2056, shuffle=False)

dss_loaders = [flame_utils.dataset_to_device(ds, device, batch_size=2056, shuffle=True) for ds in dss]
nets = [cubeds.VecClassifier(3, meta.n_labels).to(device) for _ in dss]
optimizers = [torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.0) for net in nets]

FedRounds = 50
LocalEpochs = 3

global_model = cubeds.VecClassifier(3, meta.n_labels).state_dict()

def flatten_dict(d):
    flats = []
    for k, v in d.items():
        flats.append(v.flatten())
    return torch.cat(flats)


dataset_name = os.path.basename(ds_folder) + "2"
if not os.path.exists("arxiv_res"):
    os.mkdir("arxiv_res")

csv_report_file = open(f"arxiv_res/{dataset_name}.csv", "w")
csv_report_file.write("Round,MaxNorm,MeanNorm,Loss\n")

for fed_round in range(FedRounds):
    avg_loss = 0.0
    models = []
    weights = []
    for i, (loader, net, optimizer) in enumerate(zip(dss_loaders, nets, optimizers)):
        net.load_state_dict(flame_utils.deepCloneSatetDict(global_model))
        running_loss = 0.0
        samples = 0
        for local_epoch in range(LocalEpochs):
            for x, y in loader:
                optimizer.zero_grad()
                y_pred = net(x)
                loss = torch.nn.functional.cross_entropy(y_pred, y)

                running_loss += loss.item()*x.size(0)
                samples += x.size(0)
                
                loss.backward()
                optimizer.step()

        running_loss = running_loss / samples
        avg_loss += running_loss
        # print(f"Round {fed_round}, Device {i}, Loss: {running_loss}")

        cpu_dict = {k: v.cpu() for k, v in net.state_dict().items()}

        models.append(cpu_dict)
        weights.append(samples)

    avg_loss = avg_loss / len(dss)
    # print(f"Round {fed_round}, Average Loss: {avg_loss}")


    diff = [ 
        {k: (m[k] - global_model[k]) for k in m.keys()} for m in models
    ]

    flat_diffs = [flatten_dict(d) for d in diff]

    norm_diffs = [torch.norm(d) for d in flat_diffs]
    max_norm = max(norm_diffs)
    mean_norm = sum(norm_diffs) / len(norm_diffs)

    norm_ratio = max_norm / mean_norm

    # Average the models but ensure that the dicts are in cpu
    global_model = {k: torch.stack([m[k] * w for m, w in zip(models, weights)]).sum(0) / sum(weights) for k in global_model.keys()}

    net = cubeds.VecClassifier(3, meta.n_labels).to(device)
    net.load_state_dict(global_model)

    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         y_pred = net(x)
    #         _, predicted = torch.max(y_pred.data, 1)
    #         total += y.size(0)
    #         correct += (predicted == y).sum().item()

    # print(f"Round {fed_round}, Accuracy: {100 * correct / total}")
    # print(f"Round {fed_round}, Max Norm: {max_norm}, Mean Norm: {mean_norm}")
    csv_report_file.write(f"{fed_round},{max_norm},{mean_norm},{avg_loss}\n")
    print(f"Round {fed_round}, Ratio: {norm_ratio} (Max: {max_norm}, Mean: {mean_norm})")









            






