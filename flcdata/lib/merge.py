import os
import numpy as np

def merge_npz_by_partition(npz_file1_path, npz_file2_path, output_npz_path):
    """
    Merges two .npz formatted datasets by combining samples from corresponding partitions.
    For example, partition P of dataset 1 is merged with partition P of dataset 2.

    Args:
        npz_file1_path (str): Path to the first input NPZ file.
        npz_file2_path (str): Path to the second input NPZ file.
        output_npz_path (str): Path where the merged NPZ file will be saved.
    """
    if not os.path.exists(npz_file1_path):
        print(f"Error: First NPZ file not found at '{npz_file1_path}'")
        return
    if not os.path.exists(npz_file2_path):
        print(f"Error: Second NPZ file not found at '{npz_file2_path}'")
        return

    # Save the merged dataset
    output_dir = os.path.dirname(output_npz_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # --- Validation ---
    if XX1.shape[1] != XX2.shape[1]:
        raise ValueError(
            f"Feature dimensions (XX.shape[1]) mismatch: "
            f"Dataset 1 has {XX1.shape[1]} features, Dataset 2 has {XX2.shape[1]} features."
        )

    if n_partitions1 != n_partitions2:
        raise ValueError(
            f"Number of partitions mismatch: "
            f"Dataset 1 has {n_partitions1} partitions, Dataset 2 has {n_partitions2} partitions. "
            f"Merging requires the same number of partitions."
        )

    # --- Merging Logic ---

    n_partitions_merged = n_partitions1 # Same number of partitions

    # Determine the total number of classes. Take the max as class labels are 0-indexed.
    n_classes_merged = max(n_classes1, n_classes2)

    # Prepare lists to collect merged data for each partition
    merged_xx_by_partition = [[] for _ in range(n_partitions_merged)]
    merged_yy_by_partition = [[] for _ in range(n_partitions_merged)]

    # Populate merged lists from Dataset 1
    print(f"Processing '{os.path.basename(npz_file1_path)}'...")
    for i in range(n_samples1):
        p_idx = PP1[i]
        merged_xx_by_partition[p_idx].append(XX1[i])
        merged_yy_by_partition[p_idx].append(YY1[i])

    # Populate merged lists from Dataset 2
    print(f"Processing '{os.path.basename(npz_file2_path)}'...")
    for i in range(n_samples2):
        p_idx = PP2[i]
        merged_xx_by_partition[p_idx].append(XX2[i])
        merged_yy_by_partition[p_idx].append(YY2[i])

    # Convert lists of samples back to numpy arrays for each partition, then concatenate
    final_XX = []
    final_YY = []
    final_PP = [] # This will still reference original partition indices

    n_samples_merged = 0
    n_features_merged = XX1.shape[1] # Already checked for consistency

    print("Concatenating merged partitions...")
    for p_idx in range(n_partitions_merged):
        if merged_xx_by_partition[p_idx]: # Check if partition has any data
            partition_xx = np.array(merged_xx_by_partition[p_idx], dtype=XX1.dtype)
            partition_yy = np.array(merged_yy_by_partition[p_idx], dtype=YY1.dtype)
            
            final_XX.append(partition_xx)
            final_YY.append(partition_yy)
            
            # Create PP entries for this merged partition
            partition_samples_count = partition_xx.shape[0]
            final_PP.append(np.full(partition_samples_count, p_idx, dtype=np.int32))
            
            n_samples_merged += partition_samples_count

    # Concatenate all partition arrays into global arrays
    XX_combined = np.concatenate(final_XX, axis=0) if final_XX else np.array([])
    YY_combined = np.concatenate(final_YY, axis=0) if final_YY else np.array([])
    PP_combined = np.concatenate(final_PP, axis=0) if final_PP else np.array([])

    # Sanity check
    if not (XX_combined.shape[0] == YY_combined.shape[0] == PP_combined.shape[0] == n_samples_merged):
        raise RuntimeError("Merging led to inconsistent sample counts across arrays.")



    np.savez(
        output_npz_path,
        XX=XX_combined,
        YY=YY_combined,
        PP=PP_combined,
        n_classes=np.array(n_classes_merged, dtype=np.int32),
        n_samples=np.array(n_samples_merged, dtype=np.int32),
        n_partitions=np.array(n_partitions_merged, dtype=np.int32)
    )

    print(f"\nSuccessfully merged '{os.path.basename(npz_file1_path)}' and "
          f"'{os.path.basename(npz_file2_path)}' by partition into '{output_npz_path}'.")
    print(f"Merged Dataset Stats:")
    print(f"  Total Samples: {n_samples_merged}")
    print(f"  Total Partitions: {n_partitions_merged}")
    print(f"  Total Classes: {n_classes_merged}")
    print(f"  Features per Sample: {n_features_merged}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge two NPZ datasets by partition.")
    parser.add_argument("npz_file1", type=str, help="Path to the first NPZ file.")
    parser.add_argument("npz_file2", type=str, help="Path to the second NPZ file.")
    parser.add_argument("output_npz", type=str, help="Path to save the merged NPZ file.")

    args = parser.parse_args()

    merge_npz_by_partition(args.npz_file1, args.npz_file2, args.output_npz)