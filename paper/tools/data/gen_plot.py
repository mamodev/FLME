"""
Visualization helpers for the low-rank MoG federated dataset.

Functions:
- plot_global_pca(...) : PCA scatter of sampled points colored by client or class.
- plot_client_mean_scatter(...) : PCA of per-client mean feature vectors.
- plot_class_cond_means(...) : Per-class, per-client mean scatter (shows class-conditional shifts).
- plot_pairwise_client_distances(...) : Histogram of pairwise client mean distances.
- plot_classifier_disagreement(...) : Histogram of disagreement fraction between
    global classifier and each client classifier (uses meta W_global/W_i).
"""
from typing import Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except Exception:
    umap = None


def _safe_cmap_for_n(n: int):
    # pick a categorical colormap and cycle if needed
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n)]
    return colors


def _sample_pool_for_plot(dataset: dict,
                          clients: Optional[Iterable[int]] = None,
                          per_client: int = 50,
                          rng: Optional[np.random.Generator] = None):
    """Return pooled X, client_ids, and class labels sampled from dataset.
    If clients is None sample from all clients (may be large)."""
    if rng is None:
        rng = np.random.default_rng(0)
    clients_all = [k for k in dataset.keys() if isinstance(k, int)]
    if clients is None:
        clients = clients_all
    else:
        clients = [c for c in clients if c in clients_all]
    Xs = []
    client_ids = []
    classes = []
    for c in clients:
        Xc = dataset[c]["X"]
        yc = dataset[c]["y"]
        n = Xc.shape[0]
        if n == 0:
            continue
        pick = rng.choice(n, size=min(per_client, n), replace=False)
        Xs.append(Xc[pick])
        client_ids.extend([c] * len(pick))
        classes.extend(list(yc[pick]))
    if len(Xs) == 0:
        return np.zeros((0, 0)), [], []
    X = np.vstack(Xs)
    return X, np.array(client_ids), np.array(classes)


def plot_global_pca(dataset: dict,
                    clients: Optional[Iterable[int]] = None,
                    per_client: int = 50,
                    color_by: str = "client",
                    method: str = "pca",
                    pca_components: int = 2,
                    random_state: int = 0):
    """PCA (or t-SNE / UMAP) scatter of pooled data points.
    color_by: "client" or "class"
    method: "pca" | "tsne" | "umap"
    """
    X, client_ids, classes = _sample_pool_for_plot(
        dataset, clients=clients, per_client=per_client,
        rng=np.random.default_rng(random_state)
    )
    if X.size == 0:
        raise ValueError("No samples found to plot.")
    if method == "pca":
        proj = PCA(n_components=pca_components, random_state=random_state)
        Z = proj.fit_transform(X)
    elif method == "tsne":
        Z = TSNE(n_components=2, random_state=random_state).fit_transform(X)
    elif method == "umap":
        if umap is None:
            raise RuntimeError("umap not installed. pip install umap-learn")
        Z = umap.UMAP(n_components=2, random_state=random_state).fit_transform(X)
    else:
        raise ValueError("method must be pca|tsne|umap")
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    if color_by == "client":
        unique_clients = np.unique(client_ids)
        colors = _safe_cmap_for_n(len(unique_clients))
        for i, c in enumerate(unique_clients):
            mask = client_ids == c
            ax.scatter(Z[mask, 0], Z[mask, 1], s=10, color=colors[i],
                       label=str(c), alpha=0.7)
        ax.set_title("Pooled samples colored by client (subset)")
    else:
        unique_classes = np.unique(classes)
        colors = _safe_cmap_for_n(len(unique_classes))
        for i, y in enumerate(unique_classes):
            mask = classes == y
            ax.scatter(Z[mask, 0], Z[mask, 1], s=10, color=colors[i],
                       label=str(y), alpha=0.7)
        ax.set_title("Pooled samples colored by class (subset)")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    # avoid huge legends when many clients; show at most 20 labels
    if (color_by == "client" and len(np.unique(client_ids)) <= 20) or \
       (color_by == "class" and len(np.unique(classes)) <= 20):
        ax.legend(fontsize="small", markerscale=1.2)
    plt.tight_layout()
    return fig, ax


def plot_client_mean_scatter(dataset: dict,
                             clients: Optional[Iterable[int]] = None,
                             method: str = "pca",
                             random_state: int = 0):
    """Compute per-client mean vectors and scatter them after PCA/UMAP/TSNE."""
    clients_all = [k for k in dataset.keys() if isinstance(k, int)]
    if clients is None:
        clients = clients_all
    else:
        clients = [c for c in clients if c in clients_all]
    means = []
    ids = []
    for c in clients:
        Xc = dataset[c]["X"]
        if Xc.shape[0] == 0:
            continue
        means.append(Xc.mean(axis=0))
        ids.append(c)
    if len(means) == 0:
        raise ValueError("No client data found.")
    Mx = np.vstack(means)
    if method == "pca":
        Z = PCA(n_components=2, random_state=random_state).fit_transform(Mx)
    elif method == "tsne":
        Z = TSNE(n_components=2, random_state=random_state).fit_transform(Mx)
    elif method == "umap":
        if umap is None:
            raise RuntimeError("umap not installed")
        Z = umap.UMAP(n_components=2, random_state=random_state).fit_transform(Mx)
    else:
        raise ValueError("method must be pca|tsne|umap")
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    colors = _safe_cmap_for_n(len(Z))
    for i in range(len(ids)):
        ax.scatter(Z[i, 0], Z[i, 1], color=colors[i], s=40, alpha=0.9)
        ax.text(Z[i, 0], Z[i, 1], str(ids[i]), fontsize=8)
    ax.set_title("Per-client mean vectors projected (clients numbered)")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    plt.tight_layout()
    return fig, ax


def plot_class_cond_means(dataset: dict,
                          class_id: int = 0,
                          clients: Optional[Iterable[int]] = None,
                          method: str = "pca",
                          random_state: int = 0):
    """For a chosen class, compute per-client empirical mean for that class
    and plot their projection. Useful to visualize class-conditional feature
    shifts across clients."""
    clients_all = [k for k in dataset.keys() if isinstance(k, int)]
    if clients is None:
        clients = clients_all
    else:
        clients = [c for c in clients if c in clients_all]
    means = []
    ids = []
    for c in clients:
        Xc = dataset[c]["X"]
        yc = dataset[c]["y"]
        mask = (yc == class_id)
        if mask.sum() == 0:
            continue
        means.append(Xc[mask].mean(axis=0))
        ids.append(c)
    if len(means) == 0:
        raise ValueError(f"No examples of class {class_id} found.")
    Mx = np.vstack(means)
    if method == "pca":
        Z = PCA(n_components=2, random_state=random_state).fit_transform(Mx)
    elif method == "tsne":
        Z = TSNE(n_components=2, random_state=random_state).fit_transform(Mx)
    elif method == "umap":
        if umap is None:
            raise RuntimeError("umap not installed")
        Z = umap.UMAP(n_components=2, random_state=random_state).fit_transform(Mx)
    else:
        raise ValueError("method must be pca|tsne|umap")
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    colors = _safe_cmap_for_n(len(Z))
    for i in range(len(ids)):
        ax.scatter(Z[i, 0], Z[i, 1], color=colors[i], s=40, alpha=0.9)
        ax.text(Z[i, 0], Z[i, 1], str(ids[i]), fontsize=8)
    ax.set_title(f"Per-client means for class {class_id}")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    plt.tight_layout()
    return fig, ax


def plot_pairwise_client_distances(dataset: dict, clients: Optional[List[int]] = None):
    """Histogram of pairwise euclidean distances between client mean vectors."""
    clients_all = [k for k in dataset.keys() if isinstance(k, int)]
    if clients is None:
        clients = clients_all
    else:
        clients = [c for c in clients if c in clients_all]
    means = []
    ids = []
    for c in clients:
        Xc = dataset[c]["X"]
        if Xc.shape[0] == 0:
            continue
        means.append(Xc.mean(axis=0))
        ids.append(c)
    Mx = np.vstack(means)
    # pairwise distances
    diffs = Mx[:, None, :] - Mx[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    # take upper triangle
    iu = np.triu_indices(dists.shape[0], k=1)
    vals = dists[iu]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(vals, bins=40, alpha=0.8)
    ax.set_title("Pairwise Euclidean distances between client means")
    ax.set_xlabel("distance")
    ax.set_ylabel("count")
    plt.tight_layout()
    return fig, ax


def plot_classifier_disagreement(dataset: dict,
                                 n_samples: int = 5000,
                                 random_state: int = 1):
    """Estimate disagreement fraction between W_global and each W_i by
    sampling x from the generator base (no per-client offsets) and checking
    argmax differences.

    Uses meta dict in dataset['_meta'] and returns fig, ax and disagreements.
    """
    meta = dataset.get("_meta", None)
    if meta is None:
        raise ValueError("dataset lacks '_meta' information required.")
    rng = np.random.default_rng(random_state)
    mu_y = meta["mu_y"]  # (K, d)
    A_y = meta["A_y"]    # (K, d, r)
    W_global = meta["W_global"]
    W_i = meta["W_i"]
    K, d = mu_y.shape[0], mu_y.shape[1]
    # Sample x from base mixture (ignore client offsets and noise)
    r = A_y.shape[2]
    Xs = []
    for _ in range(n_samples):
        y = rng.integers(0, K)
        z = rng.normal(size=(r,))
        x = mu_y[y] + A_y[y] @ z
        Xs.append(x)
    Xs = np.vstack(Xs)  # (n_samples, d)
    # compute global preds
    logits_global = Xs @ W_global.T  # (n_samples, K)
    preds_global = logits_global.argmax(axis=1)
    M = W_i.shape[0]
    disagreements = np.zeros(M)
    for m in range(M):
        logits_m = Xs @ W_i[m].T
        preds_m = logits_m.argmax(axis=1)
        disagreements[m] = (preds_m != preds_global).mean()
    # plot histogram
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(disagreements, bins=40, alpha=0.9)
    ax.set_title("Classifier disagreement fraction vs global classifier")
    ax.set_xlabel("fraction of differing predictions")
    ax.set_ylabel("number of clients")
    plt.tight_layout()
    return fig, ax, disagreements