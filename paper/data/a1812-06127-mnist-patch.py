
# IMPORTANT!
# this script is a minimal patch for the original MNIST dataset generator of the paper
# arxiv 1812.06127
# Code founded in repo https://github.com/litian96/FedProx
# The patch was required becouse the original code becouse fetch_mldata was used.
# The function fetch_mldata was a utility in scikit-learn used to download datasets from the now-defunct mldata.org repository.

import os
import numpy as np
import random
import json
from types import SimpleNamespace

# --- PATCH START: removed sklearn and tqdm imports ---
# from sklearn.datasets import fetch_mldata
# from tqdm import trange
# Replaced above with torchvision + torch to fetch MNIST into .torch_datasets
import torch
from torchvision import datasets
# --- PATCH END ---

# Setup directory for train/test data
train_path = './data/train/all_data_0_niid_0_keep_10_train_9.json'
test_path = './data/test/all_data_0_niid_0_keep_10_test_9.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# --- PATCH START: replace fetch_mldata with torchvision MNIST download into
# .torch_datasets ---
# Get MNIST data via torchvision, normalize, and divide by level
# Dataset files will be stored in .torch_datasets as requested
train_ds = datasets.MNIST(root='../.torch_datasets', train=True, download=True)
test_ds = datasets.MNIST(root='../.torch_datasets', train=False, download=True)

# combine train and test to match "MNIST original" (70000 samples)
data_t = torch.cat([train_ds.data, test_ds.data], dim=0)
targets_t = torch.cat([train_ds.targets, test_ds.targets], dim=0)

# flatten and convert to numpy float32, targets to numpy
data_np = data_t.view(data_t.size(0), -1).numpy().astype(np.float32)
targets_np = targets_t.numpy()

mnist = SimpleNamespace(data=data_np, target=targets_np)
# --- PATCH END ---

mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)
mnist_data = []

# --- PATCH START: replaced trange with range to remove tqdm dependency ---
for i in range(10):
    idx = mnist.target==i
    mnist_data.append(mnist.data[idx])
# --- PATCH END ---

# print([len(v) for v in mnist_data])

###### CREATE USER DATA SPLIT #######
# Assign 10 samples to each user
X = [[] for _ in range(1000)]
y = [[] for _ in range(1000)]
idx = np.zeros(10, dtype=np.int64)
for user in range(1000):
    for j in range(2):
        l = (user+j)%10
        X[user] += mnist_data[l][idx[l]:idx[l]+5].tolist()
        y[user] += (l*np.ones(5)).tolist()
        idx[l] += 5
# print(idx)

# Assign remaining sample by power law
user = 0
props = np.random.lognormal(0, 2.0, (10,100,2))
props = np.array([[[len(v)-1000]] for v in mnist_data])*props/np.sum(
    props,(1,2), keepdims=True
)
#idx = 1000*np.ones(10, dtype=np.int64)

# --- PATCH START: replaced trange with range to remove tqdm dependency ---
for user in range(1000):
    for j in range(2):
        l = (user+j)%10
        num_samples = int(props[l,user//10,j])
        #print(num_samples)
        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples
# --- PATCH END ---

# print(idx)

# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# --- PATCH START: replaced trange with range (removed ncols argument) ---
# Setup 1000 users
for i in range(1000):
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.9*num_samples)
    test_len = num_samples - train_len
    
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)
# --- PATCH END ---

# print(train_data['num_samples'])
# print(sum(train_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)