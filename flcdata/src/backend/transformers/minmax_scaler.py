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
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(params["min"], params["max"]))
    XX_scaled = scaler.fit_transform(XX)
    XX = XX_scaled

    return XX, YY, PP, distr_map


generator = TransformerModule(
    name="minmax_scaler",
    description="Apply sklearn MinMaxScaler to XX.",
    parameters={
        "min": FloatParameter(-1000000000, 1000000000, 0),
        "max": FloatParameter(-1000000000, 1000000000, 1),
    },
    
    fn=generate,
)
