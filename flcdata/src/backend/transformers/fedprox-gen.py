from backend.interfaces import TransformerModule, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter
import numpy as np

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def generate(params, XX, YY, PP, distr_map):
    np.random.seed(params["seed"])
    alpha = params["alpha"]
    beta = params["beta"]

    nfeatures = len(XX[0])
    npartitions = len(distr_map[0])
    nclasses = len(distr_map)
    nsamples = len(XX)
    IID = params["iid"]

    XX=[]
    YY=[]
    PP=PP.tolist()

    meanW = np.random.normal(0, alpha, npartitions)
    meanB = meanW

    B = np.random.normal(0, beta, npartitions)

    diagonal = np.zeros(nfeatures)
    for j in range(nfeatures):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)


    mean_x = np.zeros((npartitions, nfeatures))
    for p in range(npartitions):
        if IID:
            mean_x[p] = np.ones(nfeatures) * B[p]
        else:
            mean_x[p] = np.random.normal(B[p], 1, nfeatures)

    if IID:
        W_global = np.random.normal(0, 1, (nfeatures, nclasses))
        b_global = np.random.normal(0, 1, nclasses)

    for p in range(npartitions):
        if IID:
            W = W_global
            b = b_global
        else:
            W = np.random.normal(meanW[p], 1, (nfeatures, nclasses))
            b = np.random.normal(meanB[p], 1, nclasses)


        p_samples = [distr_map[c][p] for c in range(nclasses)]
        p_nsamples = sum(p_samples)

        xx = np.random.multivariate_normal(mean_x[p], cov_x, p_nsamples)
        yy = np.zeros(p_nsamples)

        for j in range(p_nsamples):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        XX.extend(xx.tolist())
        YY.extend(yy.tolist())
        PP.extend([p] * p_nsamples)

    XX = np.array(XX)
    YY = np.array(YY)
    PP = np.array(PP)

    return XX, YY, PP, distr_map


generator = TransformerModule(
    name="fedprox-gen",
    description="Apply random noise to features.",
    parameters={
        "alpha": FloatParameter(0.0, 1.0, 0.5),
        "beta": FloatParameter(0.0, 1000, 1.0),
        "iid": BooleanParameter(False),
        "seed": IntParameter(1, 1000000000, 1),
    },

    fn=generate,
)    
