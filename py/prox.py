import json
import numpy as np
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('GTK3Agg')

NUM_USER = 100
dimension = 3
NUM_CLASS = 3
NUM_SAMPLES = 1000

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate_synthetic(alpha, beta, iid):

    samples_per_user = np.full((NUM_USER,), NUM_SAMPLES // NUM_USER)  # Fixed 100 samples per user


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

def visualize_data(X, y, ax):


    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w', 'orange', 'purple']
    markers = ['o', 'x', 's', 'p', 'P', '*', '+', 'X', 'D', 'd']

    # Plot each user's data in 3D space
    for i in range(NUM_USER):
        x_data = np.array(X[i])
        y_data = np.array(y[i])

        classes = [np.where(y_data == i)[0] for i in range(NUM_CLASS)]

        maker = markers[i % len(markers)]
        for j in range(NUM_CLASS):
            ax.scatter(x_data[classes[j], 0], x_data[classes[j], 1], x_data[classes[j], 2], c=colors[j], marker=maker)

    # Set labels for axes
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')

def main():
    # make a 4x4 grid of subplots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("IID, alpha=0, beta=0")
    X, y = generate_synthetic(alpha=0, beta=0, iid=1)
    visualize_data(X, y, ax)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Non-IID, alpha=0, beta=0")
    X, y = generate_synthetic(alpha=0, beta=0, iid=0)
    visualize_data(X, y, ax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("IID, alpha=0, beta=1")
    X, y = generate_synthetic(alpha=0, beta=10, iid=1)
    visualize_data(X, y, ax)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Non-IID, alpha=0, beta=1")
    X, y = generate_synthetic(alpha=0, beta=10, iid=0)
    visualize_data(X, y, ax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("IID, alpha=1, beta=0")
    X, y = generate_synthetic(alpha=10, beta=0, iid=1)
    visualize_data(X, y, ax)
    fig = plt.figure()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Non-IID, alpha=1, beta=0")
    X, y = generate_synthetic(alpha=10, beta=0, iid=0)
    visualize_data(X, y, ax)



    plt.show()

if __name__ == "__main__":
    main()
