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
    # Apply the function to the data
    dims = params["dims"]
    assert dims > 0, "dims must be greater than 0"
    assert dims <= XX.shape[1], "dims must be less than or equal to the number of features in XX"

    axis = [i for i in range(dims)]
    np.random.shuffle(axis)

    fn = params["fn"]
    if fn == "cos":
        XX[:, axis] = np.cos(XX[:, axis])
    elif fn == "sin":
        XX[:, axis] = np.sin(XX[:, axis])
    elif fn == "tan":
        XX[:, axis] = np.tan(XX[:, axis])
    elif fn == "exp":
        XX[:, axis] = np.exp(XX[:, axis])
    elif fn == "log":
        XX[:, axis] = np.log(XX[:, axis])
    elif fn == "sqrt":
        XX[:, axis] = np.sqrt(XX[:, axis])
    elif fn == "abs":
        XX[:, axis] = np.abs(XX[:, axis])

    return XX, YY, PP, distr_map


generator = TransformerModule(
    name="fn_apply",
    description="Apply a function to the data",
    parameters={
        "fn": EnumParameter(
            [
                "cos",
                "sin",
                "tan",
                "exp",
                "log",
                "sqrt",
                "abs",
            ],
            default="cos"),

        "dims": IntParameter(1, 10000, 2000),
    },
    fn=generate,
)
