import matplotlib
matplotlib.use('GTK3Agg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# models: list of list of numpy arrays
def plot_models(models):
    flat_models = [np.concatenate([layer.flatten() for layer in model]) for model in models]
    mean = np.mean(flat_models, axis=0)
    differences = [model - mean for model in flat_models]

    # Reduce dimensions to 3D
    pca = PCA(n_components=3)
    parameters_3d = pca.fit_transform(differences)

    # Visualize in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(parameters_3d[:, 0], parameters_3d[:, 1], parameters_3d[:, 2], c='blue', s=50)
    ax.set_title("PCA Visualization of NN Parameters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()

np.random.seed(42)

N_MODELS = 10
LAYERS_SIZE = [1000, 1000, 10]

models = []
for _ in range(N_MODELS):
    model = []
    for i in range(len(LAYERS_SIZE) - 1):
        model.append(np.random.randn(LAYERS_SIZE[i], LAYERS_SIZE[i + 1]))
    models.append(model)

plot_models(models)

