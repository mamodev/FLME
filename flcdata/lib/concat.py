import numpy as np
import os

def concatenate_npz_datasets(npz_file1_path, npz_file2_path, output_npz_path):
    """
    Concatenates two .npz formatted datasets into a single new .npz file.

    Args:
        npz_file1_path (str): Path to the first input NPZ file.
        npz_file2_path (str): Path to the second input NPZ file.
        output_npz_path (str): Path where the concatenated NPZ file will be saved.
    """
    if not os.path.exists(npz_file1_path):
        print(f"Error: First NPZ file not found at '{npz_file1_path}'")
        return
    if not os.path.exists(npz_file2_path):
        print(f"Error: Second NPZ file not found at '{npz_file2_path}'")
        return

    print(f"Loading '{npz_file1_path}'...")
    data1 = np.load(npz_file1_path, allow_pickle=True)
    XX1 = data1['XX']
    YY1 = data1['YY']
    PP1 = data1['PP']
    n_classes1 = data1['n_classes'].item()
    n_samples1 = data1['n_samples'].item()
    n_partitions1 = data1['n_partitions'].item()

    print(f"Loading '{npz_file2_path}'...")
    data2 = np.load(npz_file2_path, allow_pickle=True)
    XX2 = data2['XX']
    YY2 = data2['YY']
    PP2 = data2['PP']
    n_classes2 = data2['n_classes'].item()
    n_samples2 = data2['n_samples'].item()
    n_partitions2 = data2['n_partitions'].item()

    # --- Validation and Merging Logic ---

    # Check for consistency in number of features
    if XX1.shape[1] != XX2.shape[1]:
        raise ValueError(
            f"Feature dimensions (XX.shape[1]) mismatch: "
            f"Dataset 1 has {XX1.shape[1]} features, Dataset 2 has {XX2.shape[1]} features."
        )

    # Determine the total number of classes. We take the max of the two datasets
    # because class labels are usually 0-indexed.
    n_classes_combined = max(n_classes1, n_classes2)

    # Concatenate XX and YY
    XX_combined = np.concatenate((XX1, XX2), axis=0)
    YY_combined = np.concatenate((YY1, YY2), axis=0)

    # Handle PP (Partitions)
    # The new partition indices for the second dataset will start after the
    # last partition index of the first dataset.
    # We add n_partitions1 to all PP values from the second dataset.
    PP2_reindexed = PP2 + n_partitions1
    PP_combined = np.concatenate((PP1, PP2_reindexed), axis=0)

    # Calculate combined metadata
    n_samples_combined = n_samples1 + n_samples2
    n_partitions_combined = n_partitions1 + n_partitions2
    n_features_combined = XX_combined.shape[1] # Should be same as XX1.shape[1]

    # Sanity check: ensure all combined arrays have the same first dimension
    if not (XX_combined.shape[0] == YY_combined.shape[0] == PP_combined.shape[0] == n_samples_combined):
        raise RuntimeError("Concatenation led to inconsistent sample counts across arrays.")

    # Save the combined dataset
    output_dir = os.path.dirname(output_npz_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savez(
        output_npz_path,
        XX=XX_combined,
        YY=YY_combined,
        PP=PP_combined,
        n_classes=np.array(n_classes_combined, dtype=np.int32),
        n_samples=np.array(n_samples_combined, dtype=np.int32),
        n_partitions=np.array(n_partitions_combined, dtype=np.int32)
    )

    print(f"\nSuccessfully concatenated '{os.path.basename(npz_file1_path)}' and "
          f"'{os.path.basename(npz_file2_path)}' into '{output_npz_path}'.")
    print(f"Combined Dataset Stats:")
    print(f"  Total Samples: {n_samples_combined}")
    print(f"  Total Partitions: {n_partitions_combined}")
    print(f"  Total Classes: {n_classes_combined}")
    print(f"  Features per Sample: {n_features_combined}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Concatenate two NPZ datasets into one.")
    parser.add_argument("npz_file1", type=str, help="Path to the first NPZ file.")
    parser.add_argument("npz_file2", type=str, help="Path to the second NPZ file.")
    parser.add_argument("output_npz", type=str, help="Path for the output concatenated NPZ file.")

    args = parser.parse_args()

    concatenate_npz_datasets(args.npz_file1, args.npz_file2, args.output_npz)   