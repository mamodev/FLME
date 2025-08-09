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

