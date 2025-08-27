import numpy as np

def alpha_from_variance_fraction(u, K, alpha_min=0.1, p=1.0, alpha_max=1e6):
    """
    Map knob u in [0,1] to symmetric Dirichlet alpha.
    u=0 -> alpha=alpha_max (default: inf, uniform)
    u=1 -> alpha=alpha_min (max variance)
    """
    if u <= 0:
        return alpha_max  # uniform
    if u >= 1:
        return alpha_min  # max variance
    
    # variance at alpha_min
    v_max = (K - 1) / (K**2 * (K * alpha_min + 1))
    # target variance
    v_target = v_max * (u**p)
    # invert variance formula
    alpha = ((K - 1) / (K**2 * v_target) - 1) / K
    return alpha

def _softmax(z):
    z = np.asarray(z)
    if z.ndim == 1:
        z = z - np.max(z)
        ex = np.exp(z)
        return ex / np.sum(ex)
    else:
        z = z - np.max(z, axis=1, keepdims=True)
        ex = np.exp(z)
        return ex / np.sum(ex, axis=1, keepdims=True)



def generate_non_iid(
    num_users=10,
    num_classes=3,
    dimension=5,
    total_samples=1000,
    covariate_shift=0.0,
    prior_shift=0.0,
    conditional_shift=0.0,
    label_shift=0.0,
    quantity_skew=0.0,
    seed=None,
):
    """
    Generate dataset with 5 controllable non-iid types and return a dict:
    {
        "XX": np.array shape (n_samples, dimension),
        "YY": np.array shape (n_samples,),
        "PP": np.array shape (n_samples,) of client indices,
        "n_classes": int,
        "n_samples": int,
        "n_partitions": int
    }
    """
    rng = np.random.default_rng(seed)

    # ---- Base global objects ----
    mu = rng.normal(0.0, 1.0, size=(num_classes, dimension))
    diag = np.array([(j + 1) ** -1.2 for j in range(dimension)])
    Sigma = np.diag(diag)
    W0 = rng.normal(0.0, 1.0, size=(dimension, num_classes))
    b0 = rng.normal(0.0, 1.0, size=(num_classes,))
    base_log_prior = np.log(np.ones(num_classes) / num_classes)

    # ---- determine per-client sample counts (quantity skew) ----
    dira = alpha_from_variance_fraction(quantity_skew, num_users)
    counts = np.random.dirichlet([dira] * num_users)
    counts = [int(x * total_samples) for x in counts]
    while total_samples - sum(counts) > 0:
        rdm_idx = rng.integers(0, num_users)
        counts[rdm_idx] += 1

    X_parts = []
    y_parts = []
    pp_parts = []

    cov_scale = covariate_shift * 2.0
    cond_scale = conditional_shift * 1.5
    prior_scale = prior_shift * 1.0
    label_scale = label_shift * 1.0

    for i in range(num_users):
        n_i = counts[i]
        s_i = (
            rng.normal(0.0, cov_scale, size=(dimension,))
            if covariate_shift > 0
            else np.zeros(dimension)
        )

        if conditional_shift > 0:
            c_i = rng.normal(0.0, cond_scale, size=(num_classes, dimension))
        else:
            c_i = np.zeros((num_classes, dimension))

        if prior_shift > 0:
            u = rng.normal(0.0, prior_scale, size=(num_classes,))
            logp = base_log_prior + u
            p_i = _softmax(logp)
        else:
            p_i = np.ones(num_classes) / num_classes

        if label_shift > 0:
            DeltaW = rng.normal(0.0, label_scale, size=W0.shape)
            Deltab = rng.normal(0.0, label_scale, size=b0.shape)
            W_i = W0 + DeltaW
            b_i = b0 + Deltab
        else:
            W_i = W0
            b_i = b0

        Xi = np.zeros((n_i, dimension))
        yi = np.zeros(n_i, dtype=int)

        for j in range(n_i):
            y_gen = rng.choice(num_classes, p=p_i)
            class_mean = mu[y_gen] + s_i + c_i[y_gen]
            x = rng.multivariate_normal(class_mean, Sigma)
            Xi[j] = x

            if label_shift > 0:
                logits = x @ W_i + b_i
                probs = _softmax(logits)
                y_obs = rng.choice(num_classes, p=probs)
            else:
                y_obs = int(y_gen)

            yi[j] = y_obs

        X_parts.append(Xi)
        y_parts.append(yi)
        pp_parts.append(np.full(n_i, i, dtype=int))

    XX = np.vstack(X_parts)
    YY = np.concatenate(y_parts)
    PP = np.concatenate(pp_parts)
    n_samples_actual = XX.shape[0]

    return {
        "XX": XX,
        "YY": YY,
        "PP": PP,
        "n_classes": num_classes,
        "n_samples": int(n_samples_actual),
        "n_partitions": num_users,
    }
    
    
def save(path, ds):
    np.savez(
        path,
        XX=ds["XX"],
        YY=ds["YY"],
        PP=ds["PP"],
        n_classes=np.array(ds["n_classes"], dtype=np.int32),
        n_samples=np.array(ds["n_samples"], dtype=np.int32),
        n_partitions=np.array(ds["n_partitions"], dtype=np.int32)
    )
    
if __name__ == '__main__':
    save("output.npz", generate_non_iid(
        num_users=10,
        num_classes=30,
        dimension=5,
        total_samples=10000,
        covariate_shift=0.0,
        prior_shift=0.0,
        conditional_shift=0.0,
        label_shift=0.0,
        quantity_skew=0.1,
        seed=12,
    ))
