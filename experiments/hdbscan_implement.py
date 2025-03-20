#https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html

import matplotlib
matplotlib.use('GTK3Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import heapq

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()


moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
data = np.vstack([moons, blobs])

# plt.scatter(data.T[0], data.T[1])
# plt.show()

def core_dist_knn(xx, k, x):
    distance = np.linalg.norm(xx - x, axis=1)
    distance.sort()

    return distance[k]

def mutual_reachability_distance(xx, k, x, y):
    return max(core_dist_knn(xx, k, x), core_dist_knn(xx, k, y), np.linalg.norm(x - y))

# import scipy for the minimum spanning tree
from scipy.sparse.csgraph import minimum_spanning_tree


k = 3
threshold = 1.0
drp = 0.1
n = len(data)

adj_matrix = np.inf * np.ones((n, n))
for i in range(n):
    for j in range(i+1, n):
        adj_matrix[i, j] = mutual_reachability_distance(data, k, data[i], data[j])
        adj_matrix[j, i] = adj_matrix[i, j]

output = minimum_spanning_tree(adj_matrix)

mst_edges = list(zip(*output.nonzero()))


max_edge = max([adj_matrix[u, v] for u, v in mst_edges])
min_edge = min([adj_matrix[u, v] for u, v in mst_edges])

for u, v in mst_edges:
    xx = [data[u][0], data[v][0]]
    yy = [data[u][1], data[v][1]]

    interp_dist = (adj_matrix[u, v] - min_edge) / (max_edge - min_edge)

    plt.plot(xx, yy, color=(interp_dist, 0, 1 - interp_dist))

plt.scatter(data.T[0], data.T[1])
plt.show()

