import json
import numpy as np
import os

def transform_json_to_npz(json_file_path, output_dir):
    """
    Transforms a dataset from JSON format (Version 2) to NPZ format (Version 1).

    Args:
        json_file_path (str): Path to the input JSON file (e.g., 'data/train/mytrain.json').
        output_dir (str): Directory where the output NPZ file will be saved.
                          The NPZ file will have the same name as the JSON file
                          (without extension) plus '.npz'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_file_path, 'r') as f:
        data_json = json.load(f)

    all_xx = []
    all_yy = []
    all_pp = [] # This will map samples to their original user/partition

    n_classes = 0 # We'll determine this by finding the max label
    n_samples = 0
    n_partitions = len(data_json['users']) # Each user becomes a partition

    # Iterate through users and collect their data
    for i, user_id in enumerate(data_json['users']):
        user_data = data_json['user_data'][user_id]
        user_xx = np.array(user_data['x'], dtype=np.float32)
        user_yy = np.array(user_data['y'], dtype=np.long)

        # Append data to the global lists
        all_xx.append(user_xx)
        all_yy.append(user_yy)

        # Create PP array: assign the current user's index to all their samples
        all_pp.append(np.full(user_xx.shape[0], i, dtype=np.int32))

        # Update total samples and max class
        n_samples += user_xx.shape[0]
        if user_yy.size > 0: # Ensure there are labels to check
            n_classes = max(n_classes, np.max(user_yy) + 1)

    # Concatenate all arrays to form the final XX, YY, PP
    XX = np.concatenate(all_xx, axis=0) if all_xx else np.array([])
    YY = np.concatenate(all_yy, axis=0) if all_yy else np.array([])
    PP = np.concatenate(all_pp, axis=0) if all_pp else np.array([])

    # Determine n_features from XX (assuming all samples have the same feature dimension)
    n_features = XX.shape[1] if XX.size > 0 else 0

    # Sanity checks
    if XX.shape[0] != n_samples:
        print(f"Warning: Calculated n_samples ({n_samples}) does not match XX shape ({XX.shape[0]})")
    if YY.shape[0] != n_samples:
        print(f"Warning: Calculated n_samples ({n_samples}) does not match YY shape ({YY.shape[0]})")
    if PP.shape[0] != n_samples:
        print(f"Warning: Calculated n_samples ({n_samples}) does not match PP shape ({PP.shape[0]})")

    # Define the output file name
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_npz_path = os.path.join(output_dir, f"{base_name}.npz")

    # Save to NPZ format
    np.savez(
        output_npz_path,
        XX=XX,
        YY=YY,
        PP=PP,
        n_classes=np.array(n_classes, dtype=np.int32),
        n_samples=np.array(n_samples, dtype=np.int32),
        n_partitions=np.array(n_partitions, dtype=np.int32)
    )
    print(f"Successfully transformed '{json_file_path}' to '{output_npz_path}'")

def transform_npz_to_json(npz_file_path, output_dir):
    """
    Transforms a dataset from NPZ format (Version 1) to JSON format (Version 2).

    Args:
        npz_file_path (str): Path to the input NPZ file (e.g., 'data/train_npz/mytrain.npz').
        output_dir (str): Directory where the output JSON file will be saved.
                          The JSON file will have the same name as the NPZ file
                          (without extension) plus '.json'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data from the NPZ file
    data_npz = np.load(npz_file_path, allow_pickle=True)

    XX = data_npz['XX']
    YY = data_npz['YY']
    PP = data_npz['PP']
    n_partitions = data_npz['n_partitions'].item() # Use .item() to get scalar from array

    # Initialize the JSON data structure
    json_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Create user-specific lists for X and Y
    # We will populate these lists as we iterate through partitions
    user_xx_data = [[] for _ in range(n_partitions)]
    user_yy_data = [[] for _ in range(n_partitions)]

    # Distribute samples back to their respective users/partitions
    print(f"Distributing samples for '{os.path.basename(npz_file_path)}'...")
    # Iterating directly without tqdm
    for i in range(len(XX)): # Iterate through all samples
        partition_idx = PP[i]
        user_xx_data[partition_idx].append(XX[i].tolist()) # Convert numpy array to list for JSON
        user_yy_data[partition_idx].append(YY[i].item()) # Convert numpy int to Python int for JSON

    # Populate the final JSON data structure
    for i in range(n_partitions):
        uname = f'f_{i:05d}' # Reconstruct user name format

        json_data['users'].append(uname)
        json_data['user_data'][uname] = {
            'x': user_xx_data[i],
            'y': user_yy_data[i]
        }
        json_data['num_samples'].append(len(user_xx_data[i])) # Number of samples for this user

    # Define the output file name
    base_name = os.path.splitext(os.path.basename(npz_file_path))[0]
    output_json_path = os.path.join(output_dir, f"{base_name}.json")

    # Save to JSON format
    with open(output_json_path, 'w') as outfile:
        json.dump(json_data, outfile, indent=4) # Use indent for readability

    print(f"Successfully transformed '{npz_file_path}' to '{output_json_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transform JSON dataset to NPZ format.")
    parser.add_argument("json_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output NPZ file.")
    args = parser.parse_args()

    transform_json_to_npz(args.json_file, args.output_dir)
