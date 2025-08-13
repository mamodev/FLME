"""
High-dimensional synthetic generator based on low-rank MoG per class.

Key ideas:
- Each class y has a global mean mu_y (d,)
  and a low-rank factor A_y (d, r).
- Client i can have:
  - a global offset delta_i (feature skew, same for all classes)
  - per-class offsets delta_yi (class-conditional feature shift)
- Labels generated either via a shared classifier W_global (no concept drift)
  or client-specific W_i (concept drift). Softmax + sampling yields labels.

Usage example at bottom.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np


def make_rng(seed: Optional[int]):
    return np.random.default_rng(seed)


def l2_mean_pairwise_separation(means: np.ndarray) -> float:
    # means: (K, d)
    K = means.shape[0]
    if K < 2:
        return 0.0
    s = 0.0
    count = 0
    for i in range(K):
        for j in range(i + 1, K):
            s += np.linalg.norm(means[i] - means[j]) ** 2
            count += 1
    return (s / count) ** 0.5


def generate_global_class_components(
    rng: np.random.Generator,
    K: int,
    d: int,
    r: int,
    class_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create global class means and low-rank factors.

    Returns:
      mu_y: (K, d)
      A_y: (K, d, r)
    """
    # Random means, then scale to enforce separation roughly class_scale
    mu_y = rng.normal(size=(K, d))
    # normalize and scale so pairwise means have desired scale
    mu_norms = np.linalg.norm(mu_y, axis=1, keepdims=True) + 1e-8
    mu_y = mu_y / mu_norms * class_scale

    # Low-rank factors: sample gaussian then ortho columns optionally
    A_y = rng.normal(scale=0.5, size=(K, d, r))
    return mu_y, A_y


def generate_client_offsets(
    rng: np.random.Generator,
    M: int,
    d: int,
    K: int,
    feature_shift_scale: float = 0.1,
    class_cond_shift_scale: float = 0.0,
    rank_class_cond: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate client offsets:
      delta_i: (M, d)  -- global per-client offset (feature skew)
      delta_yi: (M, K, d) -- class-specific offsets (class-conditional)
    class_cond_shift_scale = 0 -> no class-conditional shifts
    rank_class_cond: if >0, generate low-rank per-class offsets
    """
    delta_i = rng.normal(scale=feature_shift_scale, size=(M, d))
    if class_cond_shift_scale <= 0.0:
        delta_yi = np.zeros((M, K, d))
        return delta_i, delta_yi

    # Efficient low-rank class-conditional offsets:
    if rank_class_cond <= 0:
        delta_yi = rng.normal(
            scale=class_cond_shift_scale, size=(M, K, d)
        )
    else:
        # sample low-rank in r_c dims then project to d
        r_c = rank_class_cond
        # factors small to avoid huge shifts
        B = rng.normal(scale=0.7, size=(M, K, r_c))
        U = rng.normal(scale=0.6, size=(K, d, r_c))
        # delta_yi[m,k,:] = U[k] @ B[m,k]
        delta_yi = np.einsum("kdr,mkr->mkd", U, B)
        # scale to desired magnitude
        cur_std = np.std(delta_yi)
        if cur_std > 0:
            delta_yi = delta_yi / cur_std * class_cond_shift_scale
    return delta_i, delta_yi


def sample_feature_given_class(
    rng: np.random.Generator,
    y: int,
    client_id: int,
    mu_y: np.ndarray,
    A_y: np.ndarray,
    delta_i: np.ndarray,
    delta_yi: np.ndarray,
    sigma_noise: float,
) -> np.ndarray:
    """
    Sample one d-dimensional x for class y at client client_id.
    """
    d, r = A_y.shape
    z = rng.normal(size=(r,))
    x = mu_y + A_y @ z
    if delta_i is not None:
        x = x + delta_i[client_id]
    if delta_yi is not None:
        x = x + delta_yi[client_id, y]
    if sigma_noise > 0:
        x = x + rng.normal(scale=sigma_noise, size=(d,))
    return x


def make_classifiers(
    rng: np.random.Generator,
    d: int,
    K: int,
    concept_drift_fraction: float = 0.0,
    concept_drift_scale: float = 0.2,
    M: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a shared global classifier W_global (K, d) and per-client W_i (M, K, d)
    For clients without concept drift, W_i == W_global.
    For clients with drift, W_i = W_global + noise (Gaussian or sparse).
    """
    W_global = rng.normal(scale=1.0 / np.sqrt(d), size=(K, d))
    W_i = np.tile(W_global[None, :, :], (M, 1, 1))
    if concept_drift_fraction <= 0.0:
        return W_global, W_i

    num_drift = max(1, int(np.round(concept_drift_fraction * M)))
    drift_clients = rng.choice(M, size=num_drift, replace=False)
    for m in drift_clients:
        noise = rng.normal(
            scale=concept_drift_scale / np.sqrt(d), size=(K, d)
        )
        W_i[m] = W_global + noise
    return W_global, W_i


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sample_client_dataset(
    rng: np.random.Generator,
    client_id: int,
    n_samples: int,
    K: int,
    mu_y: np.ndarray,
    A_y: np.ndarray,
    delta_i: np.ndarray,
    delta_yi: np.ndarray,
    sigma_noise: float,
    W_global: np.ndarray,
    W_i: np.ndarray,
    label_mode: str = "shared",  # "shared" | "client" | "mixed"
    mix_prob_client_labels: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample n_samples for one client. Label modes:
      - "shared": use W_global for logits -> labels
      - "client": use W_i[client_id]
      - "mixed": with probability mix_prob_client_labels use client label,
                 else shared label (stochastic).
    Returns X: (n_samples, d), y: (n_samples,)
    Note: we sample y labels after drawing x (so P(y|x) = f(x) possibly
    client-specific).
    """
    d = mu_y.shape[1]
    X = np.zeros((n_samples, d))
    y = np.zeros(n_samples, dtype=int)
    for t in range(n_samples):
        # sample a class prior uniform (we assume same P(y) across clients)
        cls = rng.integers(0, K)
        x = sample_feature_given_class(
            rng, cls,
            client_id,
            mu_y[cls],
            A_y[cls],
            delta_i,
            delta_yi,
            sigma_noise,
        )
        # choose labeling rule
        if label_mode == "shared":
            logits = W_global @ x
        elif label_mode == "client":
            logits = W_i[client_id] @ x
        elif label_mode == "mixed":
            if rng.random() < mix_prob_client_labels:
                logits = W_i[client_id] @ x
            else:
                logits = W_global @ x
        else:
            raise ValueError("label_mode must be shared|client|mixed")
        p = softmax(logits)
        # sample label from categorical p
        sampled_label = rng.choice(K, p=p)
        X[t] = x
        y[t] = sampled_label
    return X, y


def generate_federated_dataset(
    seed: Optional[int] = 0,
    d: int = 512,
    r: int = 20,
    K: int = 10,
    M: int = 200,
    N_per_client: int = 100,
    feature_shift_scale: float = 0.1,
    class_cond_shift_scale: float = 0.0,
    rank_class_cond: int = 0,
    sigma_noise: float = 0.5,
    concept_drift_fraction: float = 0.0,
    concept_drift_scale: float = 0.2,
    label_mode: str = "shared",
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Generate dataset for M clients. Returns a dict:
      {client_id: {"X": array(n_i, d), "y": array(n_i,) } }
    All clients here get N_per_client samples (you can modify easily).
    Also returns internal generator parameters for reproducibility (not returned;
    see closure variables).
    """
    rng = make_rng(seed)
    mu_y, A_y = generate_global_class_components(
        rng, K=K, d=d, r=r, class_scale=1.2
    )
    delta_i, delta_yi = generate_client_offsets(
        rng,
        M=M,
        d=d,
        K=K,
        feature_shift_scale=feature_shift_scale,
        class_cond_shift_scale=class_cond_shift_scale,
        rank_class_cond=rank_class_cond,
    )
    W_global, W_i = make_classifiers(
        rng,
        d=d,
        K=K,
        concept_drift_fraction=concept_drift_fraction,
        concept_drift_scale=concept_drift_scale,
        M=M,
    )
    dataset = {}
    for m in range(M):
        X, y = sample_client_dataset(
            rng,
            client_id=m,
            n_samples=N_per_client,
            K=K,
            mu_y=mu_y,
            A_y=A_y,
            delta_i=delta_i,
            delta_yi=delta_yi,
            sigma_noise=sigma_noise,
            W_global=W_global,
            W_i=W_i,
            label_mode=label_mode,
        )
        dataset[m] = {"X": X, "y": y}
    # also return generator internals as attributes on the result for inspection
    dataset["_meta"] = {
        "mu_y": mu_y,
        "A_y": A_y,
        "delta_i": delta_i,
        "delta_yi": delta_yi,
        "W_global": W_global,
        "W_i": W_i,
    }
    return dataset


if __name__ == "__main__":
    seed = 42
    M = 200  # number of clients (cross-device style)
    N = 50  # samples per client (small data)
    K = 10
    d = 2
    r = 2

    ds = generate_federated_dataset(
        seed=seed,
        d=d,
        r=r,
        K=K,
        M=M,
        N_per_client=N,
        feature_shift_scale=0.15,  # per-client global shift
        class_cond_shift_scale=0.08,  # small per-class shifts
        rank_class_cond=6,
        sigma_noise=0.4,
        concept_drift_fraction=0.2,  # 20% clients use different label function
        concept_drift_scale=0.4,
        label_mode="mixed",  # some labels follow client-specific rule in sampling
    )

    # Quick diagnostics (print shapes and example):
    print("clients:", M)
    print("sample client 0 X shape:", ds[0]["X"].shape)
    print(
        "example labels unique counts (client 0):",
        np.unique(ds[0]["y"], return_counts=True)[1],
    )
    print("meta keys:", list(ds["_meta"].keys()))

    # ---- Plotting ----
    import matplotlib.pyplot as plt
    import os
    from gen_plot import plot_global_pca, plot_client_mean_scatter, plot_class_cond_means, plot_pairwise_client_distances, plot_classifier_disagreement

    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Global PCA colored by client (subset)
    fig1, ax1 = plot_global_pca(ds, per_client=30, color_by="client", method="pca")
    fig1.suptitle("Global PCA (colored by client)", fontsize=10)
    f1_path = os.path.join(out_dir, "global_pca_by_client.png")
    fig1.savefig(f1_path, dpi=150)
    print("Saved:", f1_path)

    # 2) Global PCA colored by class
    fig2, ax2 = plot_global_pca(ds, per_client=30, color_by="class", method="pca")
    fig2.suptitle("Global PCA (colored by class)", fontsize=10)
    f2_path = os.path.join(out_dir, "global_pca_by_class.png")
    fig2.savefig(f2_path, dpi=150)
    print("Saved:", f2_path)

    # 3) Per-client mean scatter (projected)
    fig3, ax3 = plot_client_mean_scatter(ds, method="pca")
    f3_path = os.path.join(out_dir, "client_mean_scatter.png")
    fig3.savefig(f3_path, dpi=150)
    print("Saved:", f3_path)

    # 4) Class-conditional means for a handful of classes
    classes_to_plot = list(range(min(4, K)))  # first few classes
    for cls in classes_to_plot:
        try:
            figc, axc = plot_class_cond_means(ds, class_id=cls, method="pca")
            pathc = os.path.join(out_dir, f"class_{cls}_cond_means.png")
            figc.savefig(pathc, dpi=150)
            print("Saved:", pathc)
        except ValueError as e:
            print(f"Skipping class {cls}: {e}")

    # 5) Pairwise distances between client mean vectors
    fig4, ax4 = plot_pairwise_client_distances(ds)
    f4_path = os.path.join(out_dir, "pairwise_client_mean_distances.png")
    fig4.savefig(f4_path, dpi=150)
    print("Saved:", f4_path)

    # 6) Classifier disagreement (W_global vs W_i)
    fig5, ax5, disagreements = plot_classifier_disagreement(ds, n_samples=5000)
    f5_path = os.path.join(out_dir, "classifier_disagreement_hist.png")
    fig5.savefig(f5_path, dpi=150)
    print("Saved:", f5_path)
    print(
        f"Classifier disagreement stats: mean={disagreements.mean():.3f}, "
        f"std={disagreements.std():.3f}, max={disagreements.max():.3f}"
    )

    # Show everything interactively (useful in notebooks / local runs)
    # plt.show()