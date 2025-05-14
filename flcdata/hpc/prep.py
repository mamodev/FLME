import argparse

from preprocessors.prep_dataset import preprocess_dataset
from preprocessors.prep_model   import preprocess_model
from preprocessors.prep_timeline import preprocess_timeline

preprocess_dataset("../.splits/data", ".data/")
preprocess_model("../.splits/data", ".data/")
preprocess_timeline("../.timelines/t.json", ".data/")


