import json
import numpy as np
import random
import argparse
import os

NUM_USER = 30

def softmax(x):
    x_max = np.max(x)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex)

def generate_synthetic(alpha, beta, iid, samples_per_user):

    dimension = 60
    NUM_CLASS = 10
    
    # samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50
    samples_per_user = [samples_per_user for _ in range(NUM_USER)]

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 0, dimension)

    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1,  NUM_CLASS)

    SOFT_MAX_MARGIN = 0.3
    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)
        if iid == 1:
            W = W_global
            b = b_global

        # Balanced generation: model inversion with correct shapes and centering
        cov_term = np.dot(W.T, W)  # W.T @ W: (10,10)
        pinv_cov_term = np.linalg.pinv(cov_term)  # (10,10), stable and fast

        projector_row = np.dot(W, np.dot(pinv_cov_term, W.T))  # W @ pinv @ W.T: (60,10) @ (10,10) @ (10,60) = (60,60)
        P = np.eye(dimension) - projector_row  # Null-space projector onto ker(W.T): (60,60)

        num_per_class = samples_per_user[i] // NUM_CLASS
        remainder = samples_per_user[i] % NUM_CLASS

        for c in range(NUM_CLASS):
            for _ in range(num_per_class):
                # Generate target logits l with l[c] > max others (margin m ~ logit confidence)
                others = np.random.normal(0, 0.5, NUM_CLASS - 1)
                max_other = np.max(others)
                l = np.full(NUM_CLASS, 0.0)
                l[np.arange(NUM_CLASS) != c] = others
                l[c] = max(max_other + 0.1, SOFT_MAX_MARGIN)  # Enforce margin; m=3.0 typical

                # Particular solution x_min (min ||x|| s.t. W.T x = l - b)
                l_adjusted = l - b
                x_min = np.dot(W, np.dot(pinv_cov_term, l_adjusted))  # W @ pinv @ (l - b): (60,10) @ (10,10) @ (10,) = (60,)

                # Project mean_x[i] onto affine space: closest_mean = x_min + P @ (mean_x[i] - x_min)
                delta_to_mean = mean_x[i] - x_min
                closest_mean = x_min + np.dot(P, delta_to_mean)

                # Sample noise in null space ~ N(0, cov_x)
                z_noise = np.random.multivariate_normal(np.zeros(dimension), cov_x)

                # Final sample
                xx_j = closest_mean + np.dot(P, z_noise)

                # Verify (should always be c; tolerance for float errors)
                tmp = np.dot(xx_j, W) + b
                y_pred = np.argmax(softmax(tmp))
                assert y_pred == c, f"Inversion failed for user {i}, class {c}: got {y_pred} (logits diff: {np.max(tmp) - tmp[c]:.2f})"

                X_split[i].append(xx_j.tolist())
                y_split[i].append(int(c))

        # Remainder: unchanged normal generation
        for _ in range(remainder):
            xx_j = np.random.multivariate_normal(mean_x[i], cov_x)
            tmp = np.dot(xx_j, W) + b
            y_pred = np.argmax(softmax(tmp))
            X_split[i].append(xx_j.tolist())
            y_split[i].append(int(y_pred))

        print("{}-th users has {} exampls".format(i, len(y_split[i])))


    return X_split, y_split



def main():
    # usage: python3 gen_data.py --iid=True
    # or python3 gen_data.py --alpha=<float> --beta=<float>
    # incorrect: python3 gen_data.py --iid=True --alpha=0.5

    argparser = argparse.ArgumentParser()
    argparser.add_argument("folder", type=str, help="folder to save data")
    argparser.add_argument('--iid',  action="store_true", help='whether i.i.d or not')
    argparser.add_argument('--alpha', type=float, default=0.0, help='alpha')
    argparser.add_argument('--beta', type=float, default=0.0, help='beta')
    argparser.add_argument('--samples_per_user', type=int, default=280, help='number of samples per user, default 280 to match paper quantity')
    args = argparser.parse_args()   


    folder = args.folder
    alpha = args.alpha
    beta = args.beta
    iid = args.iid

    if iid and (alpha != 0 or beta != 0):
        print("Error: incorrect setting, when --iid=True, --alpha and --beta must be 0")    
        print(f"provided: --iid={iid}, --alpha={alpha}, --beta={beta}")
        exit(1)

    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    if os.path.exists(folder) == False:
        os.makedirs(folder)

    train_path = os.path.join(folder, "train.json")
    test_path = os.path.join(folder, "test.json")

    X, y = generate_synthetic(alpha=alpha, beta=beta, iid=int(iid), samples_per_user=args.samples_per_user) 

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    for i in range(NUM_USER):
        uname = 'f_{0:05d}'.format(i)        
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)
    

    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()

