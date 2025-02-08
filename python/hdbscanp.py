import matplotlib
matplotlib.use('GTK3Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data

import hdbscan

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()


# moons, _ = data.make_moons(n_samples=50, noise=0.05)
# blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)

# make random 3d data 

# d = np.random.rand(100,3)

# create 3d data with 2 clusters
d = np.random.rand(100,3)
d[0:50,0] = d[0:50,0] + 1
d[0:50,1] = d[0:50,1] + 1
d[0:50,2] = d[0:50,2] + 1



clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(d)

palette = sns.color_palette()
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]

print(set(clusterer.labels_))

clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())

# make 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(d.T[0], d.T[1], d.T[2], c=cluster_colors, s=50, alpha=0.5)
plt.show()
