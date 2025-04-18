from backend.interfaces import Module, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter

def uniform_label_skew_map(n_classes, n_samples, n_partitions):
    import numpy as np
    n_samples_per_class = n_samples // n_classes

    mapping = np.zeros((n_classes, n_partitions), dtype=int)
    base = n_samples_per_class // n_partitions
    class_reminder = n_samples_per_class % n_partitions
    reminder = n_samples % n_classes

    for i in range(n_classes):
        for j in range(n_partitions):
            mapping[i, j] = base + (1 if j < class_reminder else 0)
        if i < reminder:
            mapping[i, 0] += 1
        


    return mapping

def generate(params, n_classes, n_samples):

    distr_map = uniform_label_skew_map(
        n_classes=n_classes,
        n_samples=n_samples,
        n_partitions=params["n_partitions"],
    )

    assert distr_map.sum(axis=1).sum() == n_samples, f"Expected {n_samples} samples, got {distr_map.sum(axis=1).sum()}"

    return distr_map

generator = Module(
    name="uniform_label_skew_map",
    description="",
    parameters={
        "n_partitions": IntParameter(1, 10000, 10),
    },
    fn=generate,
)    
