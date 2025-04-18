from backend.interfaces import (
    TransformerModule,
    IntParameter,
    FloatParameter,
    StringParameter,
    BooleanParameter,
    EnumParameter,
)
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def generate(params, XX, YY, PP, distr_map):
    np.random.seed(params["seed"])
    degree = params["degree"]

    poly = PolynomialFeatures(degree=degree)
    XX_poly = poly.fit_transform(XX)

    return XX_poly, YY, PP, distr_map


generator = TransformerModule(
    name="polynomial_features",
    description="Apply polynomial features to XX.",
    parameters={
        "degree": IntParameter(1, 5, 2),
        "seed": IntParameter(1, 1000000000, 1),
    },
    fn=generate,
)
