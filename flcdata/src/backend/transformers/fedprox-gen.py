from backend.interfaces import TransformerModule, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter
import numpy as np
import math 

# def softmax(x):
#     ex = np.exp(x)
#     sum_ex = np.sum(np.exp(x))
#     return ex/sum_ex


# def gen_samples(mean_x, cov_x, W, b, n_samples):
#     xx = np.random.multivariate_normal(mean_x, cov_x, n_samples)
#     yy = np.zeros(n_samples)

#     for j in range(n_samples):
#         tmp = np.dot(xx[j], W) + b
#         yy[j] = np.argmax(softmax(tmp))

#     return xx, yy

def softmax_batch(logits: np.ndarray) -> np.ndarray:
    # logits: shape (n_samples, n_classes)
    # subtract max per row for numerical stability
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def gen_samples(mean_x: np.ndarray,
                cov_x: np.ndarray,
                W: np.ndarray,
                b: np.ndarray,
                n_samples: int):
    # draw all x's at once
    xx = np.random.multivariate_normal(mean_x, cov_x, n_samples)
    # compute logits for all samples: shape (n_samples, n_classes)
    logits = xx.dot(W) + b  # b broadcasts over rows
    # softmax across each row, then pick the max‐index per sample
    yy = np.argmax(softmax_batch(logits), axis=1)
    return xx, yy

def generate(params, XX, YY, PP, distr_map):
    np.random.seed(params["seed"])
    alpha = params["alpha"]
    beta = params["beta"]

    nfeatures = len(XX[0])
    npartitions = len(distr_map[0])
    nclasses = len(distr_map)
    nsamples = len(XX)
    IID = params["iid"]


    errored = False
    MAX_RETRIES = 10
    for iii in range(MAX_RETRIES):
        XX=[]
        YY=[]
        PP=[]
        errored = False
        try:
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


                # distr_map[c][p] = Number of samples of class c in partition p
                p_samples = sum([distr_map[c][p] for c in range(nclasses)])
                # so p_samples = number of samples in partition p

                # xx, yy = gen_samples(mean_x[p], cov_x, W, b, p_samples)
                # while there is some class with len(yy[yy == c]) !== distr_map[c][p]:
                    # generate some more samples


                if params["balanced"]:
                    # next_free_class_target = nclasses
                    # class_target = [c for c in range(nclasses)]


                    xx = np.zeros((p_samples, nfeatures))
                    yy = np.full((p_samples), -1, dtype=int)
                    writeIdx = 0
                    iterations = 0
                    while writeIdx < p_samples:
                        iterations += 1
                        xx2, yy2 = gen_samples(mean_x[p], cov_x, W, b, math.floor(p_samples * 2))
                        for c in range(nclasses):
                            l = len(yy[yy == c])
                            need_to_fulfill = distr_map[c][p] - l
                            if need_to_fulfill > 0:

                                idx =  np.where(yy2 == c)[0]
                                # while len(idx) == 0 and next_free_class_target < (nclasses * 10) - 1:
                                #     idx = np.where(yy2 == class_target[c])[0]
                                #     if len(idx) == 0:
                                #         class_target[c] = next_free_class_target
                                #         next_free_class_target += 1

                                if len(idx) == 0:
                                    raise Exception(f"No samples of class {c} found.")

                                if len(idx) > need_to_fulfill:
                                    idx = idx[:need_to_fulfill]

                                xx[writeIdx:writeIdx+len(idx)] = xx2[idx]
                                yy[writeIdx:writeIdx+len(idx)] = c
                                writeIdx += len(idx)
                        
                        if iterations > 500:
                   
                            for c in range(nclasses):
                                print(f"p: {p}, c: {c}, nsamples = {len(yy[yy == c])}/{distr_map[c][p]}", flush=True)
                            raise Exception("Too many iterations to generate balanced samples.")

           
                else:
                    xx, yy = gen_samples(mean_x[p], cov_x, W, b, p_samples)
            
                # for c in range(nclasses):
                #     print(f"p: {p}, c: {c}, nsamples = {len(yy[yy == c])}/{distr_map[c][p]}", flush=True)

                XX.extend(xx.tolist())
                YY.extend(yy.tolist())
                PP.extend([p] * p_samples)

            XX = np.array(XX, dtype=np.float32)
            YY = np.array(YY, dtype=np.int32)
            PP = np.array(PP, dtype=np.int32)

            return XX, YY, PP, distr_map
        
        except Exception as e:
            errored = True
            print(f"Error: {e}", flush=True)
            print("Retrying...", flush=True)
            continue

    raise Exception("Failed to generate balanced samples after many attempts.")

generator = TransformerModule(
    name="fedprox-gen",
    description="Apply random noise to features.",
    parameters={
        "alpha": FloatParameter(0.0, 1.0, 0.5),
        "beta": FloatParameter(0.0, 1000, 1.0),
        "iid": BooleanParameter(False),
        "balanced": BooleanParameter(True),
        "seed": IntParameter(1, 1000000000, 1),
    },

    fn=generate,
)    
