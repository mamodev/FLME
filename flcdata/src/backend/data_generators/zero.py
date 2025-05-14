from backend.interfaces import Module, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter
from typing import List, Tuple
import numpy as np

def generate(params):
    n_samples = params["n_samples"]
    n_classes = params["n_classes"]
    n_features = params["n_features"]

    n_samples_per_class = n_samples // n_classes
    n_samples = n_samples_per_class * n_classes

    # xx = np.ones((n_samples, n_features))
    xx = np.random.rand(n_samples, n_features)
    yy = np.zeros((n_samples), dtype=int)

    for i in range(n_classes):
        yy[i*n_samples_per_class:(i+1)*n_samples_per_class] = i

    return xx, yy

generator = Module(
    name="zero",
    description="this create a dataset initialized with zeros features",
    parameters={
        "n_samples": IntParameter(1, 10000, 2000),
        "n_classes": IntParameter(1, 100, 10),
        "n_features": IntParameter(1, 100, 3),
        "seed": IntParameter(1, 1000000000, 1),
    },
    fn=generate,
)
