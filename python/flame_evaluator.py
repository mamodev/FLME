import torch
import queue

from cubeds import CuboidDataset
from cubeds import VecClassifier
from flame_utils import dataset_to_device

import matplotlib
matplotlib.use('GTK3Agg')

import matplotlib.pyplot as plt
from sklearn.decomposition  import PCA
import numpy as np

from sklearn import cluster as sk_cluster

import seaborn as sns

import json

from cubeds import CuboidDatasetMeta
import os
import time
from datetime import datetime

from threading import Lock
import plot_utils

import io

# def save_state_dict(d, path):
#     buffer = io.BytesIO()
#     np.save(buffer, d)
#     f = open(path, "wb")
#     f.write(buffer.getvalue())
#     f.close()

# def load_state_dict(path):
#     f = open(path, "rb")
#     buffer = io.BytesIO(f.read())
#     f.close()
#     return np.load(buffer, allow_pickle=True).item()



def __plot_model_diff(models_dicts, from_model_dict, gids):
    flat_models = []
    assert len(models_dicts) == len(gids), f"Models and gids have different lengths: {len(models_dicts)} != {len(gids)}"
    # shuffle_idx = np.random.permutation(len(models_dicts))
    # models_dicts = [models_dicts[i] for i in shuffle_idx]
    # gids = [gids[i] for i in shuffle_idx]

    start_time = time.time()

    for model in models_dicts:
        flat_model = []

        for _, layer in model.items():
            if layer.is_cuda:
                layer = layer.cpu()

            flat_model.append(layer.flatten())
        flat_models.append(flat_model)


    ref_flat = []
    for _, layer in from_model_dict.items():
        if layer.is_cuda:
            layer = layer.cpu()
        ref_flat.append(layer.flatten())


    print(f"Flattening time: {time.time() - start_time}")
    start_time = time.time()

    # layers_variance = []
    # for i in range(len(flat_models[0])):
    #     layer = []
    #     for j in range(len(flat_models)):
    #         layer.append(flat_models[j][i])

    #     layers_variance.append(layer)



    # layers_variance = [np.var(layer, axis=0) for layer in layers_variance]
    # layers_variance = [np.mean(layer) for layer in layers_variance]
    # max_variance_layer = np.argsort(layers_variance, axis=0)[-3:]

    # conc_model_layers = lambda model: np.concatenate([model[l] for l in max_variance_layer], axis=0)
    # PCA_vecs = [conc_model_layers(m) for m in flat_models] + [conc_model_layers(ref_flat)]

    ref_flat = np.concatenate(ref_flat)
    flat_models = [np.concatenate(flat_model) for flat_model in flat_models]
    flat_models = [flat_model - ref_flat for flat_model in flat_models]
    ref_flat = np.zeros_like(ref_flat)

    print(f"Variance time: {time.time() - start_time}")
    start_time = time.time()

    cluster_pca = PCA(n_components=3)
    cluster_pca = cluster_pca.fit_transform(flat_models)

    
    # cluster_pca = flat_models
    # clusterer = sk_cluster.DBSCAN(eps=0.5, min_samples=3)
    # clusterer = sk_cluster.HDBSCAN(min_cluster_size=3)
    # clusterer.fit(cluster_pca)
    # hdbscan_clusters = set(clusterer.labels_)
    # n_clusters = len(hdbscan_clusters) - (1 if -1 in hdbscan_clusters else 0)

    clusterer = sk_cluster.MiniBatchKMeans(n_clusters=3)
    clusterer.fit(cluster_pca)

    print(f"Clustering time: {time.time() - start_time}")
    start_time = time.time()

    pca = PCA(n_components=3)
    try:
        parameters_3d = pca.fit_transform(flat_models + [ref_flat])
    except:
        parameters_3d = np.zeros((len(flat_models) + 1, 3))


    print(f"PCA time: {time.time() - start_time}")

    ref_3d = parameters_3d[-1]
    parameters_3d = parameters_3d[:-1]

    # check if there are points on same coordinates with an epsilon of 1e-6
    # epsilon = 1e-6
    # unique_points = []
    # for point in parameters_3d:
    #     if not any(np.linalg.norm(point - up) < epsilon for up in unique_points):
    #         unique_points.append(point)

    # print(f"Unique points: {len(unique_points)} instead of {len(parameters_3d)}")

    print(f"Plotting {len(parameters_3d)} points")

    # colors = [dev.train_ds.color for dev in devices]

    # Is the clustering correct?
    clusters = set(clusterer.labels_)
    n_clusters = len(clusters)
    
    labels_groups = [list() for _ in range(n_clusters)]

    for i, label in enumerate(clusterer.labels_):
        labels_groups[label].append(gids[i])

    label_group_probability = [list() for _ in range(n_clusters)]
    
    for i, lg in enumerate(labels_groups):
        # prob contains the number of times each group appears in the cluster
        groups = list(set(lg))
        prob = [(g, lg.count(g) / len(lg)) for g in groups]
        label_group_probability[i].extend(prob)
       

    correctly_clustered_model = [
        # np.argmax(label_group_probability[clusterer.labels_[i]]) == gids[i]
    False   
        for i, g in enumerate(gids)
    ]

    return parameters_3d, ref_3d, clusterer, correctly_clustered_model, label_group_probability


def __plot_model_diff_draw(ax1, ax2, parameters_3d, ref_3d, clusterer, correctly_clustered_model, ngroups, label_probable_groups, gids, ignore_heatmap=False):
    
    # colors=[sns.desaturate(PALETTE[col], sat) for col, sat in zip(clusterer.labels_, clusterer.probabilities_)]
    nlables = len(label_probable_groups)
    # Most probable group for each label
    # lpg is a List of tuples (group, probability) for each label
    mlpg = [max(lpg, key=lambda x: x[1]) for lpg in label_probable_groups]

    colors = [
        "white"
        if l == -1
        else plot_utils.CLR_PLT[ mlpg[l][0] ]
        for l in clusterer.labels_
    ]

    ax1.cla()
    if not ignore_heatmap:
        ax2.cla()
    
    plt.pause(0.001)
    
    # set title of plot:
    ax1.set_title(f"N samples: {len(parameters_3d)}")
    ax1.text2D(-0.5, 0.95, f"Clusters: {len(set(clusterer.labels_))}", transform=ax1.transAxes)
    ax1.text2D(-0.5, 0.90, f"Outliers: {np.sum(clusterer.labels_ == -1)}", transform=ax1.transAxes)
    ax1.text2D(-0.5, 0.85, f"Correctly clustered: {np.sum(correctly_clustered_model)}", transform=ax1.transAxes)
    ax1.text2D(-0.5, 0.80, f"Original groups: {ngroups}", transform=ax1.transAxes)


    # ax1.scatter(parameters_3d[:, 0], parameters_3d[:, 1], parameters_3d[:, 2] , s=50, c=colors, marker=markers)

    for l in range(-1, nlables):
        idx = clusterer.labels_ == l
        
        xx = parameters_3d[idx]  
        cc = np.array(colors)[idx]

        if len(xx) == 0:
            continue

        ax1.scatter(xx[:, 0], xx[:, 1], xx[:, 2], s=50, c=cc, marker=plot_utils.MRK_PLT[l])            
   
    ax1.scatter(ref_3d[0], ref_3d[1], ref_3d[2], s=120, c='red')

    
    if not ignore_heatmap:
        prob_threshold = 0.1

        def get_cell_colors(idx):
            lidx = clusterer.labels_[idx]
            if lidx == -1:
                return ["white"]

            lmpg = label_probable_groups[lidx]
            lmpg = list(filter(lambda x: x[1] > prob_threshold, lmpg))
            lmpg = sorted(lmpg, key=lambda x: x[1], reverse=True)

            return [plot_utils.CLR_PLT[g] for g, _ in lmpg]
        
        def get_cell_real_gcolor(idx):
            return plot_utils.CLR_PLT[gids[idx]]
        
        def get_cluster(idx):
            l = clusterer.labels_[idx]
            if l == -1:
                return "Out"

            return f"C-{l}"
        

        cells = [plot_utils.TableCell(f"{get_cluster(i)}", get_cell_colors(i),
                                        text_background=get_cell_real_gcolor(i),
                                        text_alpha=1,
                                        text_fontsize=8
                                      ) for i in range(len(parameters_3d))]
        
        plot_utils.plt_table(ax2, cells, "auto")
        


def plot_model_diff(ax1, ax2, models_dicts, from_model_dict, gids, ngroups, groups_centers):
    parameters_3d, ref_3d, clusterer, correctly_clustered_model, label_probable_groups = __plot_model_diff(models_dicts, from_model_dict, gids)
    __plot_model_diff_draw(ax1, ax2, parameters_3d, ref_3d, clusterer, correctly_clustered_model, ngroups, label_probable_groups, gids)


def flat_state_dict(d: dict):
    flat = []

    for _, layer in d.items():
        flat.append(layer.flatten())

    return np.concatenate(flat)


def evaluator_thread(q, event, ds, initial_model):

    parts = ds.split("/")
    ds_folder = parts[-1]
    
    meta = CuboidDatasetMeta.load(f"{ds}/META.json")
    save_folder = f"./simulation-{ds_folder}"

    if os.path.exists(save_folder):
        os.system(f"rm -rf {save_folder}")

    os.makedirs(save_folder)

    meta.save(f"{save_folder}/META.json")

    net = VecClassifier(3, 8).to(torch.device("cuda"))
    test_ds = CuboidDataset.LoadMerged(ds, train=False)
    test_loader = dataset_to_device(test_ds, torch.device("cuda"), batch_size=1024, shuffle=False)

    midx = 0

    prev_model = initial_model
    prev_flat = flat_state_dict(prev_model)

    while not (event.is_set() and q.empty()):
        try:
            model, local_models, gids = q.get(timeout=1)  # Get item from queue

            model_flat = flat_state_dict(model)

            model_distance = np.linalg.norm(prev_flat - model_flat)
            
            net.load_state_dict(model)
            net.eval()

            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = net(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            sf = f"{save_folder}/model-{midx}"
            os.makedirs(sf)

            with open(f"{sf}/info.json", "w") as f:
                json.dump({
                    "gid": gids,
                    "correct": correct,
                }, f)

            np.save(f"{sf}/model.npy", model)
            for i, local_model in enumerate(local_models):
                np.save(f"{sf}/model-{i}.npy", local_model)

            print(f"Model {midx}: {correct}/{len(test_ds)} ({100. * correct / len(test_ds):.2f}%)")    
            print(f"Model distance: {model_distance}")
            
            midx = midx + 1

            q.task_done()
            prev_model = model
            prev_flat = model_flat

        except queue.Empty:
            continue


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate models')
    parser.add_argument('--folder', type=str, help='Folder containing the models', required=True)
    args = parser.parse_args()

    folder = args.folder

    META = CuboidDatasetMeta.load(f"{folder}/META.json")
    modelIdx = 4

    fig = plt.figure(figsize=(1,1))
    # fig.patch.set_facecolor('black')

    gs = matplotlib.gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1])


    cluster_3D_visualization = fig.add_subplot(gs[0], projection='3d')
    clustering_heatmap = fig.add_subplot(gs[1])


    diff_cache = {}

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0)  

    def polt_model(modelIdx, ignore_heatmap=False):
        nonlocal diff_cache
        model_folder = f"{folder}/model-{modelIdx}"
        
        info = json.load(open(f"{model_folder}/info.json"))
        model = np.load(f"{model_folder}/model.npy", allow_pickle=True).item()
        local_models = [np.load(f"{model_folder}/model-{i}.npy", allow_pickle=True).item() for i in range(len(info['gid']))]    


        if modelIdx in diff_cache:
            diff = diff_cache[modelIdx]
        else:
            diff = __plot_model_diff(local_models, model , info['gid'])
            diff_cache[modelIdx] = diff

        parameters_3d, ref_3d, clusterer, correctly_clustered_model, label_probable_groups = diff
        __plot_model_diff_draw(cluster_3D_visualization, clustering_heatmap, 
                               parameters_3d, ref_3d, clusterer, correctly_clustered_model, 
                               META.n_groups, label_probable_groups, gids=info['gid'],
                               ignore_heatmap=ignore_heatmap)


        plt.draw()
        plt.pause(0.001)


    def plot_info():    
        cluster_3D_visualization.cla()
        plt.pause(0.001)


        # create a table of NxN where N is the number of groups
        # do an heatmap of the distance between the groups
        
        pca = PCA(n_components=3)
        groups_centers_coord = [list() for _ in range(META.n_groups)]

        for i in range(META.n_groups):
            tuple_list = META.groups_centers[i]
            for x, y, z in tuple_list:
                groups_centers_coord[i].append(x)
                groups_centers_coord[i].append(y)
                groups_centers_coord[i].append(z)

        pca_gpoints = pca.fit_transform(groups_centers_coord)

        palette = PALETTE[:META.n_groups]
        cluster_3D_visualization.scatter(pca_gpoints[:, 0], pca_gpoints[:, 1], pca_gpoints[:, 2], s=50, c=palette)



        plt.draw()
        plt.pause(0.001)


    def on_key_press(event):
        nonlocal modelIdx

        etime = event.guiEvent.get_time()

        if event.key == 'right':
            modelIdx += 1
            polt_model(modelIdx)
        if event.key == 'up':
            plot_info()
        if event.key == 'down':
            polt_model(modelIdx, ignore_heatmap=True)

        elif event.key == 'left':
            if modelIdx == 0:
                return
            
            modelIdx -= 1
            polt_model(modelIdx)

    def on_exit(event):
        plt.close()
        exit(0)


    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('close_event', on_exit)
    plt.show()


if __name__ == "__main__":
    main()