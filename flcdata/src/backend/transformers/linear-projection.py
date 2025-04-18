from backend.interfaces import TransformerModule, Module, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter
import numpy as np

def generate_w(alpha, mean_w_p, nfeatures, out_dim):
    """
    Generates the W matrix based on the given parameters.

    Args:
        alpha (float): The alpha parameter.
        mean_w_p (float): The mean for the normal distribution.
        nfeatures (int): Number of input features.
        out_dim (int): Output dimension.

    Returns:
        numpy.ndarray: The generated W matrix.
    """
    if alpha == 0:
        if nfeatures == out_dim:
            W = np.eye(nfeatures)
        elif nfeatures < out_dim:
            identity_block = np.eye(nfeatures)
            zeros_padding = np.zeros((nfeatures, out_dim - nfeatures))
            W = np.hstack((identity_block, zeros_padding))
        else:
            W = np.eye(nfeatures)[:, :out_dim]
    else:
        W = np.random.normal(mean_w_p, 1, (nfeatures, out_dim))
    return W

def generate(params, XX, YY, PP, distr_map):
    np.random.seed(params["seed"])
    alpha = params["alpha"]
    beta = params["beta"]

    out_dim = params["out_dim"]

    nfeatures = len(XX[0])
    npartitions = len(distr_map[0])
    nclasses = len(distr_map)
    nsamples = len(XX)

    meanW = np.random.normal(0, alpha, npartitions)
    meanB = np.random.normal(0, beta, npartitions)
    
    NEW_XX = np.zeros((nsamples, out_dim))

    if params["iid"]:
        # global_W = np.random.normal(0, 1, (nfeatures, out_dim))
        global_W = generate_w(alpha, 0, nfeatures, out_dim)
        global_B = np.random.normal(0, 1, out_dim)

    for p in range(npartitions):
        
        if params["iid"]:
            W = global_W
            B = global_B
        else:
            W = generate_w(alpha, meanW[p], nfeatures, out_dim)
            B = np.random.normal(meanB[p], 0, out_dim)

        indices = np.where(PP == p)[0]

        NEW_XX[indices] = np.dot(XX[indices], W) + B
        # NEW_XX[indices] = XX[indices] + B // here only the bias is added
    
    return NEW_XX, YY, PP, distr_map

generator = TransformerModule(
    name="linear-projection",
    description="apply tranformation to the data using a linear model and softmax (fedprox papaer inspired)",
    parameters={
        "alpha": FloatParameter(0.0, 100.0, 0.5),
        "beta": FloatParameter(0.0, 100.0, 0.5),
        "out_dim": IntParameter(1, 100, 10),
        "iid": BooleanParameter(False),
        "seed": IntParameter(1, 1000000000, 1),
    },

    fn=generate,
)    
