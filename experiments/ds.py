import matplotlib
matplotlib.use('GTK3Agg')

import json

import os
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Create a synthetic dataset')
parser.add_argument('--nclients', type=int, default=4, help='Number of clients')
parser.add_argument('--npoints', type=int, default=500, help='Number of points per label')
parser.add_argument('--gui', action='store_true', help='Show the GUI', default=False)
parser.add_argument('--testprc', type=float, default=0.2, help='How many points to use for testing')
parser.add_argument('--void', type=int, default=0, help='How many points to use for testing')
parser.add_argument('--space-size', type=int, default=1, help='Size of the space')

args = parser.parse_args()
test_prc = args.testprc
gui = args.gui
n_clients = args.nclients + args.void
n = args.npoints
void = args.void
space_size = args.space_size


def fvoid(l):
    if void == 0:
        return l
    return l[:-void]

n_separations = 1
n_labels = (n_separations + 1)**3

colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 'gray']
markers = ['o', 'x', 's', 'v', 'p', 'P', '*', 'D', 'X', 'd']
lables_colors = [colors[i % len(colors)] for i in range(n_labels)]
labels_markers = [markers[i % len(markers)] for i in range(n_labels)]

# Create a 3D space
if gui:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-space_size, space_size)
    ax.set_ylim(-space_size, space_size)
    ax.set_zlim(-space_size, space_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def plot_plane(ax, origin, normal, color):
    # normal = normal / np.linalg.norm(normal)
    a, b, c = normal
    x0, y0, z0 = origin
    d = a * x0 + b * y0 + c * z0

    ls = np.linspace(-space_size, space_size, space_size)

    if c != 0:
        X, Y = np.meshgrid(ls, ls)

        xyz_points = [(x, y, 
            - (a * x + b * y + d) / c
                    ) for x in ls for y in ls]

        Z = np.array([z for x, y, z in xyz_points]).reshape(X.shape)

    elif b != 0:
        xyz_points = [(x, 
            - (a * x + d) / b, z
                    ) for x in ls for z in ls]
        
        X, Z = np.meshgrid(ls, ls)
        Y = np.array([y for x, y, z in xyz_points]).reshape(X.shape)
        
    else:
        xyz_points = [(
            - (b * y + d) / a, y, z
                    ) for y in ls for z in ls]

        Y, Z = np.meshgrid(ls, ls)
        X = np.array([x for x, y, z in xyz_points]).reshape(Y.shape)

    x, y, z = zip(*xyz_points)



    # ax.scatter(x, y, z, color=color, alpha=0.5)
    ax.plot_surface(X, Y, Z, color=color, alpha=0.1)
     


    # Plot the normal vector
    ax.quiver(
        origin[0], origin[1], origin[2], 
        normal[0], normal[1], normal[2], 
        color=color, arrow_length_ratio=0.1,
        length=space_size 
    )

def plot_xplane(ax, x, color):
    plot_plane(ax, (x, 0, 0), (1, 0, 0), color)

def plot_yplane(ax, y, color):
    plot_plane(ax, (0, y, 0), (0, 1, 0), color)

def plot_zplane(ax, z, color):
    plot_plane(ax, (0, 0, z), (0, 0, 1), color)


one_split_every = (space_size * 2) / (n_separations + 1)

plot_split = False
if gui and plot_split:
    for i in range(1, n_separations + 1):
        offset = -space_size + i * one_split_every

        plot_xplane(ax, offset, lables_colors[i])
        plot_yplane(ax, offset, lables_colors[i])
        plot_zplane(ax, offset, lables_colors[i])


subspaces = []
for i in range(n_separations + 1):
    for j in range(n_separations + 1):
        for k in range(n_separations + 1):

            startX = -space_size + i * one_split_every
            endX = -space_size + (i + 1) * one_split_every
            startY = -space_size + j * one_split_every
            endY = -space_size + (j + 1) * one_split_every

            startZ = -space_size + k * one_split_every
            endZ = -space_size + (k + 1) * one_split_every

            subspaces.append((startX, startY, startZ, endX, endY, endZ))


subspaces_points = []
for i, (sx, sy,sz, ex, ey, ez) in enumerate(subspaces):
    subspaces_points.append(
        np.array([
            (np.random.uniform(sx, ex), np.random.uniform(sy, ey), np.random.uniform(sz, ez))
            for _ in range(n)
        ])
    )


clients_subspaces_centers = []
for i in range(n_clients):
    subspaces_centers = []
    for i, (sx, sy,sz, ex, ey, ez) in enumerate(subspaces):
        subspaces_centers.append(
            (np.random.uniform(sx, ex), np.random.uniform(sy, ey), np.random.uniform(sz, ez))
        )

    clients_subspaces_centers.append(subspaces_centers)

if gui:
    for i, subspaces_centers in enumerate(fvoid(clients_subspaces_centers)):
        for j, center in enumerate(subspaces_centers):
            ax.scatter(center[0], center[1], center[2], c=colors[i], s=200, marker=labels_markers[j])


clients_points = [
    [ 
        None for _ in range(len(subspaces_points))
    ] for _ in range(n_clients)
]

for sub_index, sub_points in enumerate(subspaces_points):
    c_centers = [s[sub_index] for s in clients_subspaces_centers]
   
    distances = np.array([
        np.linalg.norm(sub_points - center, axis=1)
        for center in c_centers
    ])

    closest_centers = np.argmin(distances, axis=0)

    for cidx in range(n_clients):
        cp = sub_points[closest_centers == cidx]
        clients_points[cidx][sub_index] = cp

if gui:
    for i, points in enumerate(fvoid(clients_points)):
        for j, label_points in enumerate(points):
            ax.scatter(label_points[:, 0], label_points[:, 1], label_points[:, 2], c=colors[i], s=40, alpha=0.5, marker=labels_markers[j])


if gui:
    plt.show()


#DATASET CREATION

ds_name = f"cuboid_{n_clients - void}_{n_labels}_{n}"  
if os.path.exists(ds_name):
    res = input("Dataset already exists. Overwrite? [y/n]: ")
    if len(res) <= 0 or res[0].lower() != "y":
        exit(0)

# CREATE THE DATASET FOLDERS
os.system(f"rm -rf {ds_name}")
os.mkdir(ds_name)
os.mkdir(f"{ds_name}/test")
os.mkdir(f"{ds_name}/train")

META = {
    "ngroups": n_clients - void,
    "nlabels": n_labels,
    "npoints_per_label": n,
    "test_prc": test_prc,
    "groups_centers": fvoid(clients_subspaces_centers)
}

json.dump(META, open(f"{ds_name}/META.json", "w"))

clients_points = fvoid(clients_points)  
for cidx, points in enumerate(clients_points):
    for lidx, label_points in enumerate(points):
        n_test = int(len(label_points) * test_prc)
        np.random.shuffle(label_points)

        test_points = label_points[:n_test]
        train_points = label_points[n_test:]

        np.save(f"{ds_name}/test/g{cidx}_l{lidx}.npy", test_points)
        np.save(f"{ds_name}/train/g{cidx}_l{lidx}.npy", train_points)

print(f"Dataset {ds_name} created.")



# import torch
# class CuboidDataset(torch.utils.data.Dataset):
#     def __init__(self, files):
#         self.files = files
#         self.data = []

#         assert len(files) > 0, "No files provided"
#         assert len(set([f[0] for f in files])) == len(files), "Labels must be unique"

#         for label, paths in files:
#             if type(paths) == str:
#                 x = np.load(paths)
#                 y = torch.tensor([label] * len(x))
#                 xy = list(zip(x, y))
#                 self.data.extend(xy)
#             else:
#                 for path in paths:
#                     x = np.load(path)
#                     y = torch.tensor([label] * len(x))
#                     xy = list(zip(x, y))
#                     self.data.extend(xy)


#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]

#     def LoadGroups(ds_folder):
#         META = json.load(open(f"{ds_folder}/META.json", "r"))
#         ngroups = META["ngroups"]
#         nlabels = META["nlabels"]

#         ds = []
#         for g in range(ngroups):
#             files = []
#             for l in range(nlabels):
#                 files.append((l, f"{ds_folder}/g{g}_l{l}.npy"))

#             ds.append(CuboidDataset(files))

#         return ds