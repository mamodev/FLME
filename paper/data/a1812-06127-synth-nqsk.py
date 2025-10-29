import json
import numpy as np
import random
import argparse
import os

NUM_USER = 30

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


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
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1,  NUM_CLASS)

    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)

        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

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

