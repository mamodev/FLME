import argparse 

import matplotlib
import matplotlib.pyplot as plt
import utils.plot_utils as plot_utils
from utils.plot_utils import MRK_PLT, CLR_PLT
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


parser = argparse.ArgumentParser(description='Synthetic dataset generator')
parser.add_argument('--plot-backend', type=str, default='gtk3agg', help='Plot backend')
args = parser.parse_args()

if args.plot_backend not in matplotlib.backends.backend_registry.list_builtin():
    print(f'Invalid backend: {args.plot_backend}')
    print(f'Valid backends: {matplotlib.backends.backend_registry.list_builtin()}')
    exit(1)

matplotlib.use(args.plot_backend)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


def generate_positive_semi_definite_cov_matrix(n):
    A = np.random.randn(n, n)
    cov_matrix = np.dot(A, A.T)
    cov_matrix += np.eye(n) * 1e-5  
    diag = np.sqrt(np.diag(cov_matrix))
    cov_matrix_normalized = cov_matrix / (diag[:, None] * diag[None, :])  # Normalize by diagonal
    return cov_matrix_normalized


for i in range(3):
    mu = np.random.uniform(-5, 5, 3)
    cov = generate_positive_semi_definite_cov_matrix(3)
    sampeles = np.random.multivariate_normal(mu, cov, 1000)
    
    ax.scatter(sampeles[:, 0], sampeles[:, 1], sampeles[:, 2], marker=MRK_PLT[i], label=f'Class {i+1}', c=CLR_PLT[i])


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()