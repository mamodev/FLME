import argparse
import os
import shutil
import numpy as np
import seaborn

from flcdata import load_npz_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to the dataset .npz file')
    parser.add_argument('out_folder', type=str, help='Path to the output folder')
    parser.add_argument('--nuke-output', action='store_true', help='Remove output folder if it exists')
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        
    if not os.path.isdir(args.out_folder):
        print("Provided an out folder path that is not a valid Directory!")
        exit(1)
        
    if os.listdir(args.out_folder):
        if args.nuke_output:
            shutil.rmtree(args.out_folder)
            os.makedirs(args.out_folder)
        else:
            print("Output folder is not empty. Use --nuke-output to remove it.")
            exit(1)
    
    data = load_npz_file(args.dataset)
    
    n_partitions = data['n_partitions']
    n_samples = data['n_samples']
    n_classes = data['n_classes']

    print(f"Number of partitions: {n_partitions}")
    print(f"Number of samples: {n_samples}")
    print(f"Number of classes: {n_classes}")
    
    samples_per_partition = [
        data['YY'][data['PP'] == i]
        for i in range(n_partitions)
    ]

    
    partitions = [f"P-{i}" for i in range(n_partitions)]
    partition_samples_count = [len(samples) for samples in samples_per_partition]

    import matplotlib.pyplot as plt 

    # Create a plot to show for each partition a barchart of number of samples per class
    fig, ax = plt.subplots(figsize=(max(6, len(partitions) * 0.5), 6))
    ax.bar(partitions, partition_samples_count)
    ax.set_xlabel('Partition')
    ax.set_ylabel('Number of samples')
    ax.set_title('Number of samples per partition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_folder, 'partition_distribution.png'))
    plt.close()

    bar_values = np.array([
        [len(samples[samples == c]) for c in range(n_classes)]
        for samples in samples_per_partition
    ])

    x = np.arange(len(partitions))  # positions for partitions
    width = 2 / n_classes  # width of each bar

    fig, ax = plt.subplots()

    for c in range(n_classes):
        ax.bar(
            x + c * width,
            bar_values[:, c],
            width,
            label=f"Class {c}"
        )

    ax.set_xlabel("Partition")
    ax.set_ylabel("Number of samples")
    ax.set_title("Number of samples per partition/class")
    ax.set_yscale("log")
    ax.set_xticks(x + width * (n_classes - 1) / 2)
    ax.set_xticklabels(partitions, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.out_folder, "partition_distribution_classes.png")
    )
    plt.close()