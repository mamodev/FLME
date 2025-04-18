from backend.interfaces import Module, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter

def generate(params, XX, YY, PP, distr_map):
    import numpy as np
    
    np.random.seed(params["seed"])

    n_partitions = len(distr_map[0])

    n_skewed_partitions = int(n_partitions * params["prc"])

    partitions = [i for i in range(n_partitions)]
    np.random.shuffle(partitions)
    skewed_partitions = partitions[:n_skewed_partitions]


    for p in skewed_partitions:
        # pick a random direction vector
        dims = XX.shape[1]
       

        # amplitude is the skew factor
        amplitude = params["amplitude"]

        # apply skew to the partition
        indices = np.where(PP == p)[0]

        if params["unique_dir"]:
            direction = np.random.randn(dims)
            direction = direction / np.linalg.norm(direction)
            XX[indices] = XX[indices] + (direction * amplitude)
        else:
            for c in range(len(distr_map)):
                indices = np.where((PP == p) & (YY == c))[0]
                if len(indices) > 0:
                    direction = np.random.rand(dims)
                    direction = direction / np.linalg.norm(direction)
                    XX[indices] = XX[indices] + (direction * amplitude)


    return XX, YY, PP, distr_map


generator = Module(
    name="random_p_skew",
    description="Apply random skew to partitions.",
    parameters={
        "prc": FloatParameter(0.0, 1.0, 0.5),
        "amplitude": FloatParameter(0.0, 1000, 1.0),
        "seed": IntParameter(1, 1000000000, 1),
        "unique_dir": BooleanParameter(True),
    },

    fn=generate,
)    
