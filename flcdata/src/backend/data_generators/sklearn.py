from backend.interfaces import Module, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter
import math

def sklearn(n_classes, n_samples, n_features=3,
            n_clusters_per_class=1,
            n_informative=3, n_redundant=0, n_repeated=0,
            class_sep=2,
            random_state=None):
    
    from sklearn.datasets import make_classification
    import numpy as np

    assert n_features >= 3, "n_features must be greater than 3"
    assert n_classes >= 2, "n_classes must be greater than 2"

    XX, YY = make_classification(
        n_samples=n_samples + max(math.floor(n_samples * 0.2), 500),
        n_features=n_features,
        
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,

        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class, 
        class_sep=class_sep,
        random_state=random_state
    )

    balanced_XX = []
    balanced_YY = []
    samples_per_class = n_samples // n_classes

    assert samples_per_class * n_classes == n_samples, "n_samples must be divisible by n_classes"



    for c in range(n_classes):
        indices = np.where(YY == c)[0]
        
        if len(indices) > samples_per_class:
            # print(f"Class {c} has {len(indices)} samples, cutting to {samples_per_class}")
            selected_indices = indices[:samples_per_class]
        elif len(indices) < samples_per_class:
            raise ValueError(f"Class {c} has {len(indices)} samples, but we need {samples_per_class}")
        else:
            # print(f"Class {c} has {len(indices)} samples, keeping all")
            selected_indices = indices
        
        balanced_XX.append(XX[selected_indices])
        balanced_YY.append(YY[selected_indices])

    XX = np.vstack(balanced_XX)
    YY = np.concatenate(balanced_YY)


    print(f"==DONE==", flush=True)
    return XX, YY


def generate(params):
    return sklearn(
        n_classes=params["n_classes"],
        n_samples=params["n_samples"],
        n_features=params["n_features"],
        n_clusters_per_class=params["n_clusters_per_class"],

        n_informative=params["n_informative"],
        n_redundant=params["n_redundant"],
        n_repeated=params["n_repeated"],


        class_sep=params["class_sep"],
        random_state=params["seed"],
    )


generator = Module(
    name="sklearn",
    description="Data generators from sklearnn",
    parameters={
        "n_samples": IntParameter(1, 10000, 2000),
        "n_classes": IntParameter(1, 100, 2),
        "n_features": IntParameter(1, 600, 100),
        "n_informative": IntParameter(1, 600, 3),
        "n_redundant": IntParameter(1, 600, 0),
        "n_repeated": IntParameter(1, 600, 0),
        
        "n_clusters_per_class": IntParameter(1, 100, 1),
        "class_sep": FloatParameter(0, 100, 2),
        "seed": IntParameter(1, 1000000000, 1),
    },
    fn=generate,
)

