# Paper using Synth(a, b)

FedProx: Federated Optimization in Heterogeneous Networks https://arxiv.org/abs/1812.06127
- code: https://github.com/litian96/FedProx

FedDane: A Federated Newton-Type Method https://arxiv.org/abs/2001.01920
- (Same exact dataset code as FedProx) 
- code: https://github.com/litian96/FedDANE

FedNova: https://arxiv.org/abs/2007.07481
    - code: https://github.com/JYWa/FedNova (not cited in the paper)
        The code is suspiciously hardcoded to use CIFAR10 
    Experiments in the paper:
    - Logistic Regression on Synthetic Data (1,1) 60 clients
    - DNN on CIFAR10 16 clients (non-iidness by Dichlet distribution of data)
    Notes:
    - The local learning rate η is decayed by a constant factor after finishing 50% and 75% of the communication rounds

pFedMe: Personalized Federated Learning with Moreau Envelopes 
    https://arxiv.org/abs/2006.08848 
    https://github.com/CharlieDinh/pFedMe 
    It uses exactly the same code of FedProx.

FedDyn: FEDERATED LEARNING BASED ONDYNAMIC REGULARIZATION
    - paper: https://arxiv.org/pdf/2111.04263
    - code: https://github.com/alpemreacar/FedDyn
    Dataset:
        - MNIST
        - CIFAR10
        - CIFAR100
        - EMNIST
        - Shakespeare
        - Synthetic (unknown params) (results not even shown in the main paper)
    Clients:
        - MNIST: 100, 1000 (10% participation per round)
    Heterogeneity methods:
        - Dirichlet distribution 


# Other Federated Optimization
Scaffold:
    code: https://github.com/KarhouTam/SCAFFOLD-PyTorch (not cited in the paper)
    Experiments in the paper:
    - Logistic Regression
    - 2 Layer non-convex fully connected NN
    Clients: 100 (10% participation per round) and some 1000 clients
    Comparison: FedAvg, FedProx, SDG (baseline)
    Dataset:
        - EMNIST paritioned as https://arxiv.org/pdf/1909.06335 by using a mixed iid setting and 
            dirichlet distribution  (alpha value is not mentioned)

FedSplit: 
    - paper: https://arxiv.org/pdf/2005.05238
    - code: https://github.com/111xutingting/FedSplitLog (nnot cited in the paper)
    - Dataset: CMCC, Aliyun
    - No NON-IID experiments

MOON: Model-Contrastive Federated Learning
    - paper: https://arxiv.org/pdf/2103.16257
    Dataset:
        - CIFAR10 (10 clients)
        - CIFAR100 (100 clients)
        - TinyImageNet (1000 clients)
    - Non-IIDness: Dirichlet distribution (alpha=0.5)

FedAvgM: Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification
    - paper: https://arxiv.org/abs/1909.06335
    - Dataset: Cifar10
    - Non-IIDness: 1) Shard and 2) Dirichlet distribution
         
ViT-FL: Rethinking Architecture Design for Tackling Data Heterogeneity in
Federated Learning
    - paper: https://arxiv.org/pdf/2106.06047
    - code: https://github.com/Liangqiong/ViT-FL-main
    Dataset:
        - CIFAR10
        - CelebA
    Non-IIDness: Label distribution skew (Unknown)

FedSAM: Generalized Federated Learning via Sharpness Aware Minimization
- paper: https://arxiv.org/pdf/2206.02618
Clients: 100 (20% participation per round)
Dataset:
    - CIFAR10
    - CIFAR100
    - EMNIST
Non-IIDness: dirichlet distribution (alpha=0.6)

VHL: Virtual Homogeneity Learning
- paper: https://arxiv.org/pdf/2206.02465
Clients: 100 and 10
Dataset:
    - CIFAR10
    - FMNIST
    - SVHN
    - CIFAR100
Non-IIDness: dirichlet distribution, Sharding

# Third Party Benchmarking

Benchmarking FedAvg and FedCurv for Image Classification Tasks https://arxiv.org/abs/2303.17942
= Test  Uniform, Prior Shift, Covariate Shift
Prior shift as: Quantity Skew, Dirchlet label skew, Pathological label skew
Covariate shift (Feature distribution skew), derived from PCA on data
[tecnique from](https://ieeexplore.ieee.org/abstract/document/9892284)

Framework: OpenFL (https://arxiv.org/abs/2105.06413)

Datasets: MNIST, CIFAR10, MedMNIST (OrganAMNIST)
Note:
As for data augmentation, we performed random horizontal flips and angle
rotation of 10° with a probability of 80%. 

Conclusion:
in this paper, we experimented with two federated Learning algorithms in five different non-IID
settings. In our experiments, neither of the two algorithms outperforms the other in all the
partitioning strategies. However, somewhat unexpectedly, FedAvg produced better models
in a majority of non-IID settings despite competing with an algorithm that was explicitly
developed to improve in this scenario. Interestingly, both algorithms seem to perform better
when the number of epochs per round is increased (which also has the benefit of reducing the
communication cost). This is, to the best of our knowledge, a new observation, and we aim to



