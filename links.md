agnostic federated learning (AFL)
scenario where the target distribution can be modeled as an unknown mixture of the distributions
https://arxiv.org/pdf/1902.00146
!Troppo complicato... sembra interessante 
spigato anche in https://arxiv.org/pdf/2002.10619 appendice B.2


# clustering
https://arxiv.org/pdf/2004.11791


# non iids
https://arxiv.org/pdf/2102.02079

types of non-iid
https://arxiv.org/pdf/1912.04977, page 18


# clustering in FL
IFCA (Iterative Federated Cluster Algorithm)
https://proceedings.neurips.cc/paper_files/paper/2020/file/e32cc80bf07915058ce90722ee17bb71-Paper.pdf

Referred from IFCA, 
    clustering, Hypothesis-based clustering (HYPCLUSTER), and ?model interpolation?
https://arxiv.org/abs/2002.10619 

38 - MOCHA FL algo

Model-Agnostic
Distributed Multi-Task Optimization under Privacy
Constraints
https://arxiv.org/pdf/1910.01991

Robust Federated Learning in a Heterogeneous
Environment
https://arxiv.org/pdf/1906.06629


# keywords to explore
Alternating Minimization Algorithms
Bergman dovergence (ctx of clutering) https://jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf

# too see


Aiutami a capire questo paper su Clustered Federated Learninig:

NC = number of clients
T = number of rounds, t = round
k = local optimization iteration

M(t) = Global Model at round {t}
LM(t, i) = Client {i} local model at round {t}
Data(i) = Local Dataset of client {i}
Data = concatenation of all local datasets

Each client perform:
Delta LM(t+1, i) = SDG(k, M(t), Data(i))  - M(t)

After all client completed computation new global model is computed as follow:

M(t + 1) = 
    M(t) + sum(len(Data(i) / len(Data) * Delta LM(t+1, i))

Assumption 1:
Exists a parameter configuration M* that (locally) minize the risk on all clients data geretating distributiona at the same time:

R(i, M*) <= R(i, M(T)) forall c in NC

It's easy to see tha this assumption is not always satisfied.
Concretely it is violated if either (a) clients have disagreeing
conditional distributions Pi(y|x) != Pj (y|x) or (b) the model is not expressive enough to fit all distributions at the same time

In the following we will call a set of clients and their data
generating distributions P congruent (with respect to f and l, objective function and loss function) if they satisfy Assumption 1 and incongruent if they don’t.


Every Client is associated with an empirical risk function r(i, M(t))

r(i, M(t)) = sum foreach x, y in Data(i) { l(M(t), f(x), y)}

which approximates the true risk arbitrarily well if the number
of data points on every client is sufficiently large

For demonstration purposes let us first assume equality, Then the Federated Learning objective becomes:

F(M) = sum for each i in C {
    len(Data(i)/len(Data)) * r(i, M)
}


Under standard assumptions it has been shown that the
Federated Learning optimization protocol described in equations converges to a stationary point M* of the Federated Learning objective. In this point it holds that:

0 = Gradient(F(M*))

