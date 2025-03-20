#  Pythorch CNN model for image classification with MNIST dataset
#  Using CUDA and GPU for training

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import concurrent.futures


from datetime import datetime

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib
matplotlib.use('GTK3Agg')

import matplotlib.pyplot as plt

from sklearn.decomposition  import PCA

import numpy as np
import json



# class SpacialSytheticFeatureSkewedDataset(torch.utils.data.Dataset):
#     def __init__():
#         pass

#     def __len__(self):
#         return 1000
    
#     def __getitem__(self, idx):
#         x = torch.randn(1, 28, 28)
#         y = torch.randint(0, 10, (1,))
#         return x, y
      



def plot_cube(ax, center, size, color):
    half_size = size / 2
    vertices = np.array([[center[0] - half_size, center[1] - half_size, center[2] - half_size],
                         [center[0] + half_size, center[1] - half_size, center[2] - half_size],
                         [center[0] + half_size, center[1] + half_size, center[2] - half_size],
                         [center[0] - half_size, center[1] + half_size, center[2] - half_size],
                         [center[0] - half_size, center[1] - half_size, center[2] + half_size],
                         [center[0] + half_size, center[1] - half_size, center[2] + half_size],
                         [center[0] + half_size, center[1] + half_size, center[2] + half_size],
                         [center[0] - half_size, center[1] + half_size, center[2] + half_size]])

    faces = [[vertices[0], vertices[1], vertices[5], vertices[4]],
             [vertices[7], vertices[6], vertices[2], vertices[3]],
             [vertices[0], vertices[4], vertices[7], vertices[3]],
             [vertices[1], vertices[5], vertices[6], vertices[2]],
             [vertices[4], vertices[5], vertices[6], vertices[7]],
             [vertices[0], vertices[1], vertices[2], vertices[3]]]

    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='r', alpha=0.1))

class CuboidDataset(torch.utils.data.Dataset):
    def __init__(self, lable_data, dex="Data", color='blue'):
        self.color = color  
        self.lables = 8
        assert len(lable_data) == 8, f"Provide a file for each label: provided {len(lable_data)} files"
        assert len(set([f[0] for f in lable_data])) == len(lable_data), "Labels must be unique"

        self.data = torch.tensor([])
        self.targets = torch.tensor([]).int()

        for label, vecs in lable_data:
            samples = []
            if type(vecs) != list:
                vecs = [vecs]

            for x in vecs:
                x = torch.tensor(x).float()
                samples.append(x)

            n_samples = sum([x.size(0) for x in samples])

            self.targets = torch.cat([self.targets, torch.tensor([label] * n_samples).int()], dim=0)
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
    
    # partitioning: 1 means no partitioning, 2 means that from 1 group 2 subdatasets are created
    def LoadGroups(ds_folder, train=True, partitioning=1):
        sub_folder = "train" if train else "test"
        META = json.load(open(f"{ds_folder}/META.json", "r"))
        ngroups = META["ngroups"]
        nlabels = META["nlabels"]

        groups_colors = ['blue', 'red', 'green', 'yellow', 'black', 'purple', 'orange', 'pink']
        get_group_color = lambda g: groups_colors[g % len(groups_colors)]

        ds = []
        for g in range(ngroups):
            vecs = [[] for _ in range(partitioning)]
            for l in range(nlabels):
                vec = np.load(f"{ds_folder}/{sub_folder}/g{g}_l{l}.npy")
                n = len(vec)
                n_per_ds = n // partitioning

                for i in range(partitioning):
                   slice = vec[i * n_per_ds: (i + 1) * n_per_ds]
                   vecs[i].append((l, slice))
            
            for i in range(partitioning):
                ds.append(CuboidDataset(vecs[i], dex=f"{sub_folder} g{g} p{i}", color=get_group_color(g)))

        return ds

    def LoadMerged(ds_folder, train=True):
        sub_folder = "train" if train else "test"
        META = json.load(open(f"{ds_folder}/META.json", "r"))
        ngroups = META["ngroups"]
        nlabels = META["nlabels"]

        vecs = []
        for l in range(nlabels):
            label_vecs = []
            for g in range(ngroups):
                vec = np.load(f"{ds_folder}/{sub_folder}/g{g}_l{l}.npy")
                label_vecs.append(vec)

            vecs.append((l, label_vecs))

        return CuboidDataset(vecs)

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

class Evaluator:
    PlotCap = 1000
    def __init__(self, model, test_loader):

        self.device = torch.device("cuda")

        self.model = model
        self.test_loader = test_loader
        self.idx_permutation = np.random.permutation(len(test_loader.dataset))
        self.prev_correct_indices = np.zeros(len(test_loader.dataset), dtype=bool)

    def __eval(self):
        self.model.eval()
        with torch.no_grad():
            predicted_labels = []
            actual_labels = []
            points = []

            for data, target in self.test_loader:
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)

                pred = pred.squeeze(1)
                predicted_labels.append(pred.cpu().numpy())
                actual_labels.append(target.cpu().numpy())
                points.append(data.cpu().numpy())

            predicted_labels = np.concatenate(predicted_labels)
            actual_labels = np.concatenate(actual_labels)
            points = np.concatenate(points)

            return points, predicted_labels, actual_labels
        
    def eval_plot(self, fig, ax):
        points, predicted_labels, actual_labels = self.__eval()

        correct_indices = predicted_labels == actual_labels
        correct = correct_indices.sum()
        total = len(predicted_labels)
        accuracy = correct / total  

        prev_correct_but_now_incorrect = self.prev_correct_indices & ~correct_indices
        prev_incorrect_but_now_correct = ~self.prev_correct_indices & correct_indices
        self.prev_correct_indices = correct_indices

        points = points[self.idx_permutation]
        correct_indices = correct_indices[self.idx_permutation]
        prev_correct_but_now_incorrect = prev_correct_but_now_incorrect[self.idx_permutation]
        prev_incorrect_but_now_correct = prev_incorrect_but_now_correct[self.idx_permutation]

        points = points[:Evaluator.PlotCap]
        correct_indices = correct_indices[:Evaluator.PlotCap]
        prev_correct_but_now_incorrect = prev_correct_but_now_incorrect[:Evaluator.PlotCap]
        prev_incorrect_but_now_correct = prev_incorrect_but_now_correct[:Evaluator.PlotCap]

        colors = np.zeros((len(points), 4))
        colors[correct_indices] = [0, 0, 0, 1]
        colors[~correct_indices] = [0, 0, 0.8, .1]
        colors[prev_correct_but_now_incorrect] = [1, 0, 0, 1]
        colors[prev_incorrect_but_now_correct] = [0, 1, 0, 1]

        ax.clear()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=colors, s=50)

        return accuracy


def dataset_to_device(dataset, device, batch_size=1024, shuffle=False):
    d = dataset.data.to(device).float()
    t = dataset.targets.to(device).long()
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(d, t), batch_size=batch_size, shuffle=shuffle)

class Device:
    def __init__(self, model, train_ds, test_ds, batch_size=512):
        self.device = torch.device("cuda")
        self.test_ds = test_ds
        self.train_ds = train_ds

        self.train_loader = dataset_to_device(train_ds, self.device, batch_size=batch_size, shuffle=True)
        self.test_loader = dataset_to_device(test_ds, self.device, batch_size=batch_size, shuffle=False)

        self.model = model.to(self.device)
        self.optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.evaluator = Evaluator(model, self.test_loader)

    def train(self):
        self.model.train()
        for data, target in self.train_loader:  
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()

    def eval_plot(self, fig, ax):
        return self.evaluator.eval_plot(fig, ax)

def worker(device):
    device.train()
    return device.model.state_dict()

class FedLearnig:
    def __init__(self, devices, model, global_test_ds):
        self.devices = devices
        self.model = model
        self.global_test_ds = global_test_ds

        self.device = torch.device("cuda")
        self.model = model.to(self.device)

        self.test_loader = dataset_to_device(global_test_ds, self.device, batch_size=1024, shuffle=False)
        self.evaluator = Evaluator(model, self.test_loader)
        
    def train_round(self, workers=4):
        global_model = self.model.state_dict()

        for device in self.devices:
            device.model.load_state_dict(global_model)
            device.train()

        local_models = [device.model.state_dict() for device in self.devices]

        for k in local_models[0].keys():
           global_model[k] = sum([m[k] for m in local_models]) / len(local_models)

        self.model.load_state_dict(global_model)

    def eval_plot(self, fig, ax):
        return self.evaluator.eval_plot(fig, ax)


def plot_model_diff(ax, devices, from_model_dict):
    models_dicts = [device.model.state_dict() for device in devices]

    # model_dicts: list of model state_dicts
    # a state dict is a dictionary with keys as layer names and values as the layer weights (tensors)
    flat_models = []
    for model in models_dicts:
        flat_model = []
        # for layer in model.values():
        #     # bring flat_model to CPU if it is in GPU
        #     if layer.is_cuda:
        #         layer = layer.cpu()

            # flat_model.append(layer.flatten())

        for key, layer in model.items():
            if layer.is_cuda:
                layer = layer.cpu()

            ref_layer = from_model_dict[key]
            if ref_layer.is_cuda:
                ref_layer = ref_layer.cpu()

            flat_model.append((layer - ref_layer).flatten())
        
        flat_models.append(np.concatenate(flat_model))

    # flat_models = [np.concatenate([layer.flatten() for layer in model]) for model in models_dicts]
    mean = np.mean(flat_models, axis=0)
    differences = [model - mean for model in flat_models]

    import hdbscan
    import seaborn as sns

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
    clusterer.fit(differences)

    pca = PCA(n_components=3)
    parameters_3d = pca.fit_transform(differences)

    # colors = [dev.train_ds.color for dev in devices]
    palette = sns.color_palette()
    colors=[sns.desaturate(palette[col], sat) for col, sat in zip(clusterer.labels_, clusterer.probabilities_)]

    ax.clear()
    ax.set_title("PCA Visualization of NN Parameters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    
    plot_cube(ax, [0, 0, 0], .1, 'blue')
    xmin = np.min(parameters_3d[:, 0])
    xmax = np.max(parameters_3d[:, 0])
    ymin = np.min(parameters_3d[:, 1])
    ymax = np.max(parameters_3d[:, 1])
    zmin = np.min(parameters_3d[:, 2])
    zmax = np.max(parameters_3d[:, 2])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    ax.scatter(parameters_3d[:, 0], parameters_3d[:, 1], parameters_3d[:, 2], c=colors , s=50)

if __name__ == '__main__':

    print("Starting simulation")
    torch.manual_seed(0)
    np.random.seed(0)

    start_time = datetime.now()
    train_dss = CuboidDataset.LoadGroups("./cuboid_4_8_500", train=True, partitioning=8)
    test_ds = CuboidDataset.LoadMerged("./cuboid_4_8_500", train=False)
    end_time = datetime.now()

    print(f"Data loading time: {end_time - start_time}")

    start_time = datetime.now()
    devices = [Device(VecClassifier(3, 8), ds, ds) for ds in train_dss]
    fed = FedLearnig(devices, VecClassifier(3, 8), test_ds)

    end_time = datetime.now()
    print(f"Device creation time: {end_time - start_time}")

    plot_ds = False
    if plot_ds:
        fig_dst = plt.figure(figsize=(10, 8))

        drows = len(devices) // 4
        dcols = 4
        daxs = []

        for i, device in enumerate(devices):
            ax = fig_dst.add_subplot(drows, dcols, i + 1, projection='3d')
            daxs.append(ax)

        for device, ax in zip(devices, daxs):
            device.train_ds.plot(ax)
        
        plt.show()


    fig = plt.figure(figsize=(10, 8))
    # create two subplots one above the other
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')

    plt.show(block=False)
    fig.canvas.mpl_connect('close_event', lambda e: exit(0))

    # with torch.profiler.profile(
    #     with_stack=True,
    #     profile_memory=True,
    #     record_shapes=True,
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    # ) as prof:

    start_time = datetime.now()

    for epoch in range(0, 100):
        # if epoch > 1:
        #     input("Press Enter to continue")

        plt.draw()
        plt.pause(0.01)
        gmodel = fed.model.state_dict()
        fed.train_round()
        acc = fed.eval_plot(fig, ax1)
        plot_model_diff(ax2, devices, gmodel)

        print(f"Epoch {epoch}, Acc: {acc}")

    end_time = datetime.now()
    print(f"Training time: {end_time - start_time}")
        
    print("done")
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    #print table to trace.txt
    # trace = open("trace.txt", "w")
    # trace.write(prof.key_averages().table(sort_by="self_cuda_time_total"))
    # trace.close()


    # axs = []

    # device_per_row = 2
    # rows = len(devices) // device_per_row
    # cols = device_per_row
    
    # for i, device in enumerate(devices):
    #     ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
    #     axs.append(ax)


    # plt.show(block=False)
    # for epoch in range(0, 20):
    #     if epoch > 1:
    #         input("Press Enter to continue")

    #     plt.draw()
    #     plt.pause(0.01)

    #     for dix, device in enumerate(devices):
    #         if epoch != 0:
    #             device.train()

    #         acc = device.eval_plot(fig, axs[dix])
    #         print(f"Epoch {epoch}, Device {dix}, Acc: {acc}")


    # plt.show()
