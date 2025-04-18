from backend.interfaces import (
    TransformerModule,
    IntParameter,
    FloatParameter,
    StringParameter,
    BooleanParameter,
    EnumParameter,
)
import numpy as np


def generate(params, XX, YY, PP, distr_map):

    from sklearn.preprocessing import Normalizer
    norm = params["norm"]
    normalizer = Normalizer(norm=norm)
    XX_normalized = normalizer.fit_transform(XX)
    XX = XX_normalized

    return XX, YY, PP, distr_map

generator = TransformerModule(
    name="normalize",
    description="Apply sklearn Normalizer to XX.",
    parameters={
        "norm": EnumParameter(["l1", "l2", "max"], "l2"),
    },
    
    fn=generate,
)
