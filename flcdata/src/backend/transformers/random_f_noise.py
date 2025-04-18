from backend.interfaces import TransformerModule, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter

def generate(params, XX, YY, PP, distr_map):
    import numpy as np
    
    np.random.seed(params["seed"])


    # noise rapresent the standard deviation of the noise
    # amplitude represent the amplitude of the noise

    noise = params["noise"]
    amplitude = params["amplitude"]

    # Generate random noise
    noise = np.random.normal(0, noise, XX.shape)
    noise = noise * amplitude
    # Add noise to the features
    XX = XX + noise

    return XX, YY, PP, distr_map


generator = TransformerModule(
    name="random_f_noise",
    description="Apply random noise to features.",
    parameters={
        "noise": FloatParameter(0.0, 1.0, 0.5),
        "amplitude": FloatParameter(0.0, 1000, 1.0),
        "seed": IntParameter(1, 1000000000, 1),
    },

    fn=generate,
)    
