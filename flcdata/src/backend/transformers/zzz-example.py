from backend.interfaces import TransformerModule, IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter
import numpy as np


def generate(params, XX, YY, PP, distr_map):
    np.random.seed(params["seed"])

    # This data comes from previous steps of the pipeline that generate the data
    # XX shape is (nsamples, nfeatures) is the feature vector of the sample
    # YY shape is (nsamples,) is the label of the sample
    # PP shape is (nsamples,) is the index of the partition in wich the sample belongs
    # distr_map shape is (nclasses, npartitions) is the number of samples in each partition for each class
    
    # here put the transformation logic

    return XX, YY, PP, distr_map


generator = TransformerModule(
    name="zzz-example",
    description="Apply random noise to features.",
    parameters={
        "alpha": FloatParameter(0.0, 1.0, 0.5),# this is an exaple
        "beta": FloatParameter(0.0, 1000, 1.0),# this is an exaple
        "seed": IntParameter(1, 1000000000, 1),#is good practice to put a seed to make the transformation reproducible
        # here put the parameters that you need for the transformation
    },

    fn=generate,
)    
