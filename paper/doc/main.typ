#show link: underline

= Beyond SOA of Federated Learning convergence in Heterogeneus settings

Many published papers that tackle the study of convergence of Federated Learning algorithms and strategies exists nowadays, although many of them focus on cross-silo scenarios leaving on the table various problems that arise with the scaling of number of participants and with the increasing heterogeneity of data, computational resources and device availability.

== Let's define heterogeneity
From now on I will refer to *system heterogeneity* as the variance of computational resources, device availability and network bandwidth among federated devices. (This variance could lead to the necessity of using different training hyperparameters among devices)

To define *statistical heterogeneity* and to classify various forms of it we will refer to the sourvey: #link("https://arxiv.org/pdf/1912.04977")[Advances and Open Problems in Federated Learning (page 18)].

We can separate statistical heterogeneity into 5 subclasses:
+ Feature distribution skew (covariate shift): For example, in a handwriting recognition domain, users who write the same words might still have different stroke width, slant, etc.

+ Label distribution skew (prior probability shift): For example, when clients are tied to particular geo-regions, the distribution of labels varies across clients — kangaroos are only in Australia or zoos; a person’s face is only in a few locations worldwide; for mobile device keyboards, certain emoji are used by one demographic but not others

+ Same label, different features (concept drift): For example, images of homes can vary dramatically around the world and items of clothing vary widely. Even within the U.S., images of parked cars in the winter will be snow-covered only in certain parts of the country. The same label can also look very different at different times, and at different time scales: day vs. night, seasonal effects, natural disasters, fashion and design trends, etc

+ Same features, different label (concept shift): For example, labels that reflect sentiment or next word predictors have personal and regional variation.

+ Quantity skew or unbalancedness: Different clients can hold vastly different amounts of data.

== SOA work on FL in heterogeneous settings
The original Federated Learning article by #link("https://arxiv.org/abs/1602.05629")[McMahan et al. (2016)] describe this technology as a solution for cross-device learning as a method to preserve privacy, reduce centralized computational costs and reduce network bandwidth usage. This paper deeply discusses the effect of non iid data on model convergence and shows that the approach is robust on non iid data distributions. The problem with this statement is that the paper only considers two types of data heterogeneity (Quantity skew or unbalancedness, Label distribution skew) and that it completely ignores system heterogeneity problems. Furthermore empirical experiments are executed with a limited number of clients: about 100, this obviously is not really representative of true cross-device scenarious.

A few years later, in 2018, a new paper was released: #link("")[Federated Learning with Non-IID Data] which delved deeper into the statistical challenges of Federated Learning, particularly in the cross-device context. Their paper went beyond merely noting the presence of non-IID data by rigorously demonstrating that FedAvg's "robustness" has significant limitations when confronted with extreme label distribution skew, such as scenarios where clients exclusively possess data from a single class. In these pathological non-IID settings, they revealed drastic reductions in model accuracy, sometimes by as much as 55%. A pivotal contribution was their quantification of data heterogeneity using the Earth Mover's Distance (EMD), establishing a direct, measurable link between the degree of non-IIDness and the observed performance degradation. They further theorized and mathematically proved that this accuracy loss stems from weight divergence, a phenomenon where the models trained on highly disparate local datasets diverge significantly from an ideal global optimum. Crucially, Zhao et al. didn't just diagnose the problem; they proposed a practical and effective solution: sharing a small, uniformly distributed subset of global data among all clients at the outset of training. Their experiments showcased that even a minimal shared dataset (e.g., 5% for CIFAR-10) could dramatically improve accuracy by mitigating the negative impact of non-IID data. While both papers acknowledged the broad problem of system heterogeneity, neither extensively explored its direct impact on model convergence or explicitly designed experiments to address it, maintaining a primary focus on the statistical aspects of data distribution.

In the same year


A year later two papers were released:

#link("")[On the Convergence of FedAvg on Non-IID Data (cited by 3145)]:  that focuses on the theoretical analysis of the convergence of FedAvg. This is its main contribution. It establishes for the first time a convergence rate O(1/T) for FedAvg in non-IID and partial device participation settings, under strong convexity and smoothness assumptions. It also proves the necessity of learning rate decay. This paper also considers cases where partial participation of devices is present (stragglers effect). As in the original FL paper also in this one the experiments are done with the same amount of clients. For real datasets like MNIST experiments, the study takes in account only Quantity skew or unbalancedness and Label distribution skew.  The news here is the integration with a synthetic dataset generated from two parameters (a, b):
- A higher value of a implies that local model distributions will be more diverse across clients. This leads to greater heterogeneity in terms of "same features, different labels" (concept shift) or, more generally, greater divergence of local objectives.
- A higher value of b implies that feature distributions will be more diverse across clients. This leads to greater heterogeneity in terms of "same label, different features" (concept drift) or "label distribution skew" (prior probability shift).
We will discuss the problems of this synthetic data generation in further paragraphs.

#link("")[Federated Optimization in Heterogeneous Networks (FedProx) Citato da 7345]:  This paper, a companion piece to "On the Convergence of FedAvg on Non-IID Data" by the same research group, introduces FedProx, a generalized framework of FedAvg, designed to specifically tackle both statistical and system heterogeneity. Its main contribution lies in providing robust convergence guarantees for federated learning in heterogeneous networks, a significant advancement over FedAvg's limitations. FedProx addresses statistical heterogeneity by incorporating a proximal term into the local objective, which helps stabilize training and prevent divergence in non-IID settings. Critically, it directly confronts system heterogeneity by allowing for variable local work (i.e., different numbers of local epochs or computational effort) across devices and integrates partial updates from stragglers, rather than simply dropping them. This flexibility is a key differentiator from prior work. The paper empirically demonstrates that FedProx leads to more stable and accurate convergence, particularly in highly heterogeneous environments, improving absolute test accuracy by a notable margin. While it also utilizes the same alpha and beta synthetic datasets, its theoretical framework and empirical evaluation extend to explicitly incorporate the impact of varying computational resources and device availability, acknowledging these as distinct challenges alongside data distribution. This paper executes experiments on synth data with 30 client and 10 active clients (not really representative of a cross-device scenario)

Finally, in 2020, Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization (Wang et al)   Citato da 1783 addressed a critical, often overlooked problem: the objective inconsistency inherent in heterogeneous Federated Learning. While previous works like FedProx (Li et al., 2020) attempted to mitigate issues arising from non-IID data and variable local work by adding a proximal term, Wang et al. rigorously demonstrated that naive aggregation methods, including FedAvg and even FedProx, still lead to the global model converging to a stationary point of a mismatched objective function. This fundamental bias stems from the clients performing highly variable numbers of local updates, a common scenario given diverse computational resources, device availability, and network bandwidth (system heterogeneity), which previous theoretical analyses often simplified or ignored. Wang et al. provide a foundational understanding of this "solution bias" and its impact on convergence speed. Building upon this insight, they proposed FedNova, a normalized averaging method. FedNova fundamentally resolves the objective inconsistency by correctly normalizing local model updates before aggregation, thereby ensuring convergence to the true global objective. This paper also replicates the same heterogeneous data settings of previous ones using the previously discussed synthetic dataset and  CIFAR-10 partitioned by a Dirichlet distribution. The number of clients used in simulations perfectly matches the FedProx paper ones.

== Problems with current SOA
The main problem with the current SOA is that all the papers that tackle the problem of convergence in heterogeneous settings use the same approach to generate synthetic heterogeneous dat(that is limited to introduce some quantity skew and some label distribution skew).
This approach is not really representative of real world scenarios. In fact, in real world scenarios, statistical heterogeneity is a mix of various types of heterogeneity and not only two of them. The only approach that tries to simulate various types of statistical heterogeneity seams the synthetic dataset proposed in #link("https://arxiv.org/abs/1812.06127")[Federated Optimization in Heterogeneous Networks (FedProx)] that we will discuss in the next paragraph.

Furthermore, all the papers that tackle the problem of convergence in heterogeneous settings use a limited number of clients (about 10-100, in limited cases 1000) and a limited number of active clients (about 10, or 10%). This is not really representative of real world cross-device scenarios where the number of clients can be in the order of millions and the number of active clients can be in the order of thousands. This is a problem because it is well known that the number of clients and the number of active clients have a significant impact on the convergence of Federated Learning algorithms.

== The case of FedProx Synthetic dataset
Many paper on Federated Learning in heterogeneous settings uses the synthetic dataset proposed in #link("https://arxiv.org/abs/1812.06127")[Federated Optimization in Heterogeneous Networks (FedProx)] in order to simulate various levels of statistical heterogeneity that go beyond the prior probability shift (label distribution skew, and quantity skew). (papers like FedProx, FedDane, FedNova, pFedMe, FedDyn, etc.)

// This approach is based on two parameters (a, b):
This synthetic dataset generation takes in account two parameters (a, b):
- A higher value of *a* implies that local model distributions will be more diverse across clients. This leads to greater heterogeneity in terms of "same features, different labels" (concept shift) or, more generally, greater divergence of local objectives.
- A higher value of *b* implies that feature distributions will be more diverse across clients. This leads to greater heterogeneity in terms of "same label, different features" (concept drift) or "label distribution skew" (prior probability shift).

The problem with this approach is that it is not really clear what is the impact of these two parameters on the various types of statistical heterogeneity. For example, it is not clear if a higher value of a implies a higher value of "same features, different labels" (concept shift) or a higher value of "same label, different features" (concept drift). This is a problem because it is well known that different types of statistical heterogeneity could have a different impact on the convergence of Federated Learning algorithms.


As we can see in #ref(<fig:p_distr>) the distribution of samples among clients seams to be uncontrolled, leading to an IID case being more heterogeneous than the non IID case with a=0.5 and b=0.5. Furthermore class distribution among clients is also uncontrolled as we can notice in #ref(<fig:cp_distr>).  In #ref(<fig:nsamples>) we can see also that synthetic datasets have different number of samples, leading to a further uncontrolled variable in the experiments (number of samples seams to be correlated with *a* and *b* parameters).


#figure(
  stack(
    dir: ttb,
    image("assets/p_distr_a1812-06127_synth_iid.png", height: 110pt),
    image("assets/p_distr_a1812-06127_synth_0505.png", height: 110pt),
  ),
   caption: "Distribution of samples among partitions (Clients), on top the IID case, on the bottom the non IID case with a=0.5 and b=0.5"
) #label("fig:p_distr")


#figure(
  stack(
    dir: ttb,
    image("assets/cp_distr_a1812-06127_synth_iid.png", height: 200pt),
    image("assets/cp_distr_a1812-06127_synth_0505.png", height: 200pt),
  ),
   caption: "Distribution of classes among partitions (Clients), on top the IID case, on the bottom the non IID case with a=0.5 and b=0.5"
) #label("fig:cp_distr")


#figure(
  image("assets/nsamples_a1812-06127.png", width: 40%),
  caption: "Number of samples per dataset",
) #label("fig:nsamples")




Another problem with the current SOA is that performance is measured only in terms of weighted accuracy. This is not enough. In fact, in real world scenarios, other metrics are important such as fairness, robustness and efficiency. We replicated experiments from papers like FedProx and we found that if even the weighted accuracy seams to convergence in a stable way, the accuracy of individual clients can vary significantly going from 0% to 100% at each round as we can see in #ref(<fig:fedavg_acc>). This is a problem because it means that some clients are not learning at all. This can be a problem in scenarios where fairness is important.

#figure(
  grid(
    columns: 2,
    image("assets/synthiid_part30_drp0.0_lr0.01_seed42_accuracy_vs_version_full.png", width: 250pt),
    image("assets/synthiid_part30_drp0.5_lr0.01_seed42_accuracy_vs_version_full.png", width: 250pt),
    image("assets/synth0505_part30_drp0.0_lr0.01_seed42_accuracy_vs_version_full.png", width: 250pt),
    image("assets/synth0505_part30_drp0.5_lr0.01_seed42_accuracy_vs_version_full.png", width: 250pt),
  ),

  caption: "Accuracy of FedAvg both weighted and per client."
) #label("fig:fedavg_acc")


A deeper examination of the mathematical foundations underlying this synthetic data generation technique reveals additional shortcomings. The approach models client-specific feature distributions using multivariate normal distributions with means influenced by parameters $alpha$ and $beta$, and local model parameters (weights $W$ and biases $b$) drawn from normals centered on client-specific means. Labels are then assigned via a softmax over linear transformations of features. While this aims to introduce controlled heterogeneity, the parameterization leads to datasets where varying $alpha$ and $beta$ between 0 and 1 primarily amplifies randomness rather than systematically controlling specific types of statistical skew. For instance, higher $alpha$ diversifies the means for $W$ and $b$ across clients, but this manifests as erratic shifts in decision boundaries that do not cleanly isolate concept shift ("same features, different labels") from other forms like prior probability shift. Similarly, $beta$'s influence on feature means introduces variability that blends covariate shift with uncontrolled noise, resulting in datasets that are more stochastic artifacts than structured representations of heterogeneity. This makes datasets generated with different $alpha$-$beta$ pairs inherently incomparable, as the underlying distributions lack consistency in scale, variance, or alignment, undermining any claims of parametric control over heterogeneity levels.

Compounding this issue, the IID variant of the dataset is generated by enforcing global $W$ and $b$ (with $alpha = 0$, $beta = 0$), which fundamentally differs from non-IID cases even at low $alpha$-$beta$ values. In non-IID settings, client-specific perturbations persist regardless of parameter values, creating a structural mismatch: the IID case resembles a single, unified distribution, while non-IID cases fragment into divergent subspaces. Comparing FL algorithm performance across these---such as benchmarking convergence on IID versus non-IID synthetic data---is mathematically flawed, as it conflates true algorithmic robustness with artifacts of incompatible data geometries. Visualizing such datasets via PCA, for example, often yields nonsensical projections because the high-dimensional feature space (e.g., dimension 60 here) is dominated by the diagonal covariance ma trix with decaying eigenvalues, leading to clusters that reflect arbitrary noise rather than meaningful heterogeneity patterns.

Ultimately, these mathematical inconsistencies render the synthetic dataset ill-suited for benchmarking and comparing Federated Learning algorithms. It fails to provide a reliable, controllable proxy for real-world heterogeneity, where mixtures of skew types emerge from complex, interdependent factors rather than tunable parameters that yield random, non-comparable outputs. This highlights the need for more rigorous, mathematically grounded data generation methods that ensure comparability and targeted control over heterogeneity dimensions.


// == Resource Driven Learning