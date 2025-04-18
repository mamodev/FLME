from backend.interfaces import (
    TransformerModule,
    IntParameter,
    FloatParameter,
    StringParameter,
    BooleanParameter,
    EnumParameter,
)
import numpy as np
from sklearn.preprocessing import PowerTransformer


def generate(params, XX, YY, PP, distr_map):
    np.random.seed(params["seed"])
    method = params["method"]

    pt = PowerTransformer(method=method, standardize=False)  # standardize=False to avoid StandardScaler issues

    XX_power = pt.fit_transform(XX)

    return XX_power, YY, PP, distr_map


generator = TransformerModule(
    name="power_transformation",
    description="Apply power transformation (Box-Cox or Yeo-Johnson) to XX.",
    parameters={
        "method": EnumParameter(["box-cox", "yeo-johnson"], "yeo-johnson"),
        "seed": IntParameter(1, 1000000000, 1),
    },
    fn=generate,
)
