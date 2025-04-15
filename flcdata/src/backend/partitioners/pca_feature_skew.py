from backend.interfaces import Module, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter

def pca_feature_skew(X, Y, distr_map, feature_skewness=1.0, seed=None):
    import numpy as np
    from sklearn.decomposition import PCA

    partitions = -np.ones(len(Y), dtype=int)

    # Process each class separately.
    for label in range(len(distr_map)):
        # Get indices for samples in this class.
        indices = np.where(Y == label)[0]
        n_samples_class = len(indices)

        partition_counts = distr_map[label]
        assert sum(partition_counts) == n_samples_class, f"! Partition counts do not match the number of samples in class {label}. Expected: {n_samples_class}, Got: {sum(partition_counts)}"

        # Extract features for the current class.
        X_class = X[indices]

        # Determine an ordering of these samples.
        if feature_skewness == 0:
            order = np.random.permutation(n_samples_class)
        else:
            assert n_samples_class > 1, "No samples in this class"
            
            pca = PCA(n_components=1, random_state=seed)
            proj = pca.fit_transform(X_class).flatten()

            noise = np.random.rand(n_samples_class)
            combined = feature_skewness * proj + (1 - feature_skewness) * noise
            order = np.argsort(combined)

        sorted_indices = indices[order]

        # Assign contiguous blocks to partitions exactly as determined by partition_counts.
        start = 0
        for part_id, count in enumerate(partition_counts):
            end = start + count
            partitions[sorted_indices[start:end]] = part_id
            start = end

        # Check if all samples have been assigned to a partition.
        assert start == n_samples_class, f"Not all samples in class {label} have been assigned to a partition. Assigned: {start}, Expected: {n_samples_class}"


    return partitions


def generate(params, X, Y, distr_map):
    return pca_feature_skew(
        X=X,
        Y=Y,
        distr_map=distr_map,
        feature_skewness=params["feature_skewness"],
        seed=params["seed"],
    )

generator = Module(
    name="pca_feature_skew",
    description="Data generators from sklearn",
    parameters={
        "feature_skewness": FloatParameter(0, 1, 1),
        "seed": IntParameter(1, 1000000000, 1),
    },
    fn=generate,
)    
