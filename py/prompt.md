I have implemented a FederatedLearning Framework.

I need to test performance when data is not IID.
I need to do it in a sistematic and repetible manner.

Here is what i want to do: create a python cli to generate a synthetic dataset that
enable me to parametrize ds non iid features.

Let formally and matematically describe what i mean whit non iid

Let's define features as x and targes as y
a distribution for a device i is defined as Pi(x, y) that can be rewrited 
as Pi(y | x)Pi(x) and Pi(x | y)Pi(y)

Given this formal definition we can now classify different cases of non iid

1) Feature distribution skew (covariate shift): 
The distribution og Pi(x) may vary across clients
even if P(y | x) is shared. For example, in a handwriting recognition domain, users who write the
same words might still have different stroke width, slant, etc.

2) Label distribution skew (prior probability shift): 
The marginal distributions Pi(y) may vary across
clients, even if P(x | y) is the same. For example, when clients are tied to particular geo-regions,
the distribution of labels varies across clients — kangaroos are only in Australia or zoos; a person’s
face is only in a few locations worldwide; for mobile device keyboards, certain emoji are used by one
demographic but not others.

3) Same label, different features (concept drift): 
The conditional distributions Pi(x | y) may vary across
clients even if P(y) is shared. The same label y can have very different features x for different
clients, e.g. due to cultural differences, weather effects, standards of living, etc. For example, images
of homes can vary dramatically around the world and items of clothing vary widely. Even within the
U.S., images of parked cars in the winter will be snow-covered only in certain parts of the country. The
same label can also look very different at different times, and at different time scales: day vs. night,
seasonal effects, natural disasters, fashion and design trends, etc

4) Same features, different label (concept shift): 
The conditional distribution Pi(y | x) may vary across
clients, even if P(x) is the same. Because of personal preferences, the same feature vectors in a
training data item can have different labels. For example, labels that reflect sentiment or next word
predictors have personal and regional variation.

5) Quantity skew or unbalancedness: Different clients can hold vastly different amounts of data.


i would like to have a manner to generate a dataset that allow me to parametrize the degree of non iid in the dataset.

like this pseudo code

generate_dataset(
    n_clients=10,
    n_samples=1000,
    n_features=10,
    n_classes=10,

    feature_distribution_skew=0.1,
    label_distribution_skew=0.1,
    concept_drift=0.1,
    concept_shift=0.1,
    quantity_skew=0.1
)


there should be also some function to visualize the dataset in a 3d plot
using PCA or t-SNE to reduce feature dimensionality if needed.
this is necessary in order to visually check the correctness of the dataset generation.
if samples are too many, it should be possible to sample a subset of the dataset to visualize it.

PLESE GENERATE ME PYTHON CODE,
reason a lot, and start dividing the problem in subproblems
ask me if you have any doubt