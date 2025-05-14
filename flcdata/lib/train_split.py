import argparse
import numpy as np
import os
import sys

print("Raw args:", sys.argv)

parser = argparse.ArgumentParser(description="Split a dataset into training and validation sets.")
parser.add_argument("--input_file", type=str, help="Path to the input dataset file.")
parser.add_argument("--output_dir", type=str, help="Path to the output folder where the split datasets will be saved.")

parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of the dataset to include in the training split.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset.")
parser.add_argument("--ignore-if-exists", action="store_true", help="if split files already exist, do nothing.")

args = parser.parse_args()


if args.ignore_if_exists and os.path.exists(args.output_dir):
    print(f"Output directory {args.output_dir} already exists. Exiting without doing anything.")
    sys.exit(0)

np.random.seed(args.seed)

def __load_npz__(ds_folder):
    data = np.load(ds_folder, allow_pickle=True)
    
    needed_keys = ["XX", "YY", "PP", "n_classes", "n_samples", "n_partitions"]
    file_keys = list(data.keys())

    return {
        "XX": data["XX"],
        "YY": data["YY"],
        "PP": data["PP"],
        "n_classes": data["n_classes"],
        "n_samples": data["n_samples"],
        "n_partitions": data["n_partitions"],
    }



raw_dataset = __load_npz__(args.input_file)

# Shuffle the dataset
indices = np.random.permutation(len(raw_dataset["XX"]))
split_index = int(len(raw_dataset["XX"]) * args.train_split)
train_indices = indices[:split_index]
test_indices = indices[split_index:]
train_dataset = {
    "XX": raw_dataset["XX"][train_indices],
    "YY": raw_dataset["YY"][train_indices],
    "PP": raw_dataset["PP"][train_indices],
    "n_classes": raw_dataset["n_classes"],
    "n_samples": len(train_indices),
    "n_partitions": raw_dataset["n_partitions"],
}

test_dataset = {
    "XX": raw_dataset["XX"][test_indices],
    "YY": raw_dataset["YY"][test_indices],
    "PP": raw_dataset["PP"][test_indices],
    "n_classes": raw_dataset["n_classes"],
    "n_samples": len(test_indices),
    "n_partitions": raw_dataset["n_partitions"],
}

os.makedirs(args.output_dir, exist_ok=True)

train_file = os.path.join(args.output_dir, f"train.npz")
test_file = os.path.join(args.output_dir, f"test.npz")

# Save the training dataset
np.savez_compressed(train_file, **train_dataset)
# Save the testing dataset
np.savez_compressed(test_file, **test_dataset)

print(f"{__file__} SUCCESS! Training dataset saved to {train_file}")

