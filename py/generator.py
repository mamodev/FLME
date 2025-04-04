# non_iid_generator.py

import argparse
import numpy as np
import os
import pickle
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
import warnings
import logging
import math
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def save_data(client_data, output_dir, filename="federated_dataset.pkl"):
    """Saves the partitioned client data to a pickle file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(client_data, f)
        logging.info(f"Dataset saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save dataset to {output_path}: {e}")

def check_data_shapes(client_data):
    """Logs the shapes of data for each client."""
    logging.info("--- Client Data Shapes ---")
    total_samples = 0
    for client_id, data in client_data.items():
        shape_x = data['X'].shape
        shape_y = data['y'].shape
        n_samples = shape_x[0]
        total_samples += n_samples
        logging.info(f"Client {client_id}: X shape={shape_x}, y shape={shape_y}, Samples={n_samples}")
        # Check for empty clients
        if n_samples == 0:
            logging.warning(f"Client {client_id} has 0 samples!")
    logging.info(f"Total samples across all clients: {total_samples}")
    logging.info("--------------------------")


# --- Skew Implementation Functions ---

def _generate_base_data(n_samples, n_features, n_classes, class_sep, n_informative_frac):
    """Generates the initial IID dataset."""
    n_informative = max(1, int(n_features * n_informative_frac))
    n_redundant = max(0, int(n_features * (1.0-n_informative_frac)/2)) # Example allocation
    n_repeated = 0 # Avoid repeated features for now
    n_clusters_per_class = 1 # Keep it simple initially

    # Ensure n_features >= n_informative + n_redundant + n_repeated
    n_useless = n_features - n_informative - n_redundant - n_repeated
    if n_useless < 0:
        n_redundant = max(0, n_features - n_informative - n_repeated)
        n_useless = 0
        logging.warning(f"Adjusted n_redundant to {n_redundant} due to n_features constraints.")

    logging.info(f"Generating base dataset with make_classification: "
                 f"n_samples={n_samples}, n_features={n_features}, n_classes={n_classes}, "
                 f"n_informative={n_informative}, n_redundant={n_redundant}, "
                 f"n_repeated={n_repeated}, n_clusters_per_class={n_clusters_per_class}, "
                 f"class_sep={class_sep}, n_useless={n_useless}")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=0, # No label noise at generation
        random_state=42 # for reproducibility
    )
    # Standard scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def _apply_quantity_skew(n_samples, n_clients, alpha):
    """
    Generates sample counts per client based on Dirichlet distribution.
    Smaller alpha means more skew (some clients have much more data).
    """
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    # Ensure concentration parameter is a list/array of size n_clients
    concentration = np.full(n_clients, alpha)
    proportions = np.random.dirichlet(concentration)

    # Ensure proportions sum to 1 (they should, but floating point...)
    proportions /= proportions.sum()

    # Distribute samples, ensuring total is correct
    client_sample_counts = np.zeros(n_clients, dtype=int)
    remaining_samples = n_samples
    for i in range(n_clients - 1):
        samples = int(round(proportions[i] * n_samples))
        # Ensure we don't assign more than available, cap reasonably
        samples = min(samples, remaining_samples - (n_clients - 1 - i)) # Leave at least 1 for others
        samples = max(0, samples) # Ensure non-negative
        client_sample_counts[i] = samples
        remaining_samples -= samples
    client_sample_counts[n_clients - 1] = remaining_samples # Assign rest to last client

    # Check final counts
    if client_sample_counts.sum() != n_samples:
         logging.warning(f"Sample count mismatch after quantity skew. "
                         f"Target: {n_samples}, Actual: {client_sample_counts.sum()}. "
                         f"This might happen due to rounding with small sample sizes.")
         # Simple redistribution of difference (can be improved)
         diff = n_samples - client_sample_counts.sum()
         adjust_idx = 0
         while diff != 0:
             if diff > 0:
                 client_sample_counts[adjust_idx % n_clients] += 1
                 diff -= 1
             elif client_sample_counts[adjust_idx % n_clients] > 0: # Only reduce if count > 0
                 client_sample_counts[adjust_idx % n_clients] -= 1
                 diff += 1
             adjust_idx += 1
             if adjust_idx > 2 * n_clients and diff != 0: # Safety break
                 logging.error("Could not fully correct sample count mismatch.")
                 break


    # Ensure no client has negative samples after adjustment
    client_sample_counts = np.maximum(0, client_sample_counts)

    logging.info(f"Applied Quantity Skew (alpha={alpha}). Client sample counts: {client_sample_counts.tolist()}")
    return client_sample_counts.tolist()


def _apply_label_skew(X_global, y_global, client_sample_counts, n_classes, alpha):
    """
    Partitions data based on client sample counts and Dirichlet-sampled label distributions.
    """
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")

    n_samples_total, n_features = X_global.shape
    n_clients = len(client_sample_counts)
    client_data = {i: {'X': None, 'y': None} for i in range(n_clients)}
    data_indices = np.arange(n_samples_total)

    # Get indices for each class
    class_indices = {k: data_indices[y_global == k] for k in range(n_classes)}
    available_indices_count = {k: len(class_indices[k]) for k in range(n_classes)}
    assigned_indices = set()

    # Target label distribution per client
    client_label_proportions = np.random.dirichlet(np.full(n_classes, alpha), size=n_clients)

    logging.info(f"Target label distributions (proportions) per client (alpha={alpha}):")
    for i in range(n_clients):
        logging.debug(f"  Client {i}: {client_label_proportions[i].round(3)}")

    client_indices = {i: [] for i in range(n_clients)}

    # Assign data to clients trying to match target counts and label distributions
    # This is a complex allocation problem. A common practical approach:
    # Sort data by label, then distribute shards. This creates label skew but isn't Dirichlet.
    # Let's try the Dirichlet approach directly, acknowledging potential imperfections.

    target_counts_per_class_client = np.zeros((n_clients, n_classes), dtype=int)
    for i in range(n_clients):
        if client_sample_counts[i] == 0: continue
        proportions = client_label_proportions[i]
        # Distribute N_i samples according to proportions
        counts = np.zeros(n_classes, dtype=int)
        remaining = client_sample_counts[i]
        for k in range(n_classes - 1):
            # Calculate ideal count, ensure non-negative and feasible
            c = min(remaining, int(round(proportions[k] * client_sample_counts[i])))
            c = max(0, c)
            counts[k] = c
            remaining -= c
        counts[n_classes - 1] = remaining # Assign remainder to last class
        target_counts_per_class_client[i] = counts

        # Adjust if sum doesn't match N_i due to rounding
        count_sum_diff = client_sample_counts[i] - counts.sum()
        adjust_idx = 0
        while count_sum_diff != 0:
           if count_sum_diff > 0:
               counts[adjust_idx % n_classes] += 1
               count_sum_diff -= 1
           elif counts[adjust_idx % n_classes] > 0:
               counts[adjust_idx % n_classes] -= 1
               count_sum_diff += 1
           adjust_idx += 1
           if adjust_idx > 2 * n_classes: break # Safety break
        target_counts_per_class_client[i] = np.maximum(0, counts) # Ensure non-negative

    logging.info("Target sample counts per class per client:")
    for i in range(n_clients):
         logging.debug(f"  Client {i} (Total {client_sample_counts[i]}): {target_counts_per_class_client[i].tolist()}")


    # Now, distribute actual indices
    available_class_indices_sets = {k: set(inds) for k, inds in class_indices.items()}

    for i in range(n_clients):
        if client_sample_counts[i] == 0: continue
        client_indices_i = []
        for k in range(n_classes):
            target_count = target_counts_per_class_client[i, k]
            
            # Check how many indices are actually available for class k
            available_for_k = list(available_class_indices_sets[k] - assigned_indices)
            n_available = len(available_for_k)

            # Number of indices to actually take for class k for client i
            n_take = min(target_count, n_available)

            if target_count > n_available and n_available > 0:
                logging.warning(f"Client {i}, Class {k}: Target {target_count}, but only {n_available} samples available. Taking {n_take}.")
            elif target_count > 0 and n_available == 0:
                 logging.warning(f"Client {i}, Class {k}: Target {target_count}, but NO samples available.")

            if n_take > 0:
                # Sample without replacement from available indices for class k
                chosen_indices = np.random.choice(available_for_k, size=n_take, replace=False)
                client_indices_i.extend(chosen_indices)
                assigned_indices.update(chosen_indices)
        
        # Handle potential mismatch if we couldn't get enough samples for some classes
        shortfall = client_sample_counts[i] - len(client_indices_i)
        if shortfall > 0:
            logging.warning(f"Client {i}: Could not assign target number of samples ({client_sample_counts[i]}). Assigned {len(client_indices_i)}. Shortfall={shortfall}.")
            # Try to fill shortfall with any remaining available samples, regardless of class balance
            remaining_overall = list(set(data_indices) - assigned_indices)
            n_fill = min(shortfall, len(remaining_overall))
            if n_fill > 0:
                logging.info(f"Client {i}: Filling shortfall with {n_fill} available samples.")
                fill_indices = np.random.choice(remaining_overall, size=n_fill, replace=False)
                client_indices_i.extend(fill_indices)
                assigned_indices.update(fill_indices)


        client_indices[i] = np.array(client_indices_i, dtype=int)

    # Create final client data dictionary
    final_client_data = {}
    for i in range(n_clients):
        indices = client_indices[i]
        if len(indices) > 0:
            final_client_data[i] = {
                'X': X_global[indices],
                'y': y_global[indices]
            }
        else:
             # Handle clients with 0 samples explicitly
             final_client_data[i] = {
                'X': np.empty((0, n_features)),
                'y': np.empty((0,), dtype=y_global.dtype)
            }

    logging.info(f"Applied Label Skew (alpha={alpha}). Data partitioned.")
    return final_client_data


def _apply_feature_skew(client_data, level, n_features):
    """
    Applies client-specific feature transformations (shift + scale).
    Level 0: No transformation. Level 1: Significant transformation.
    """
    if level == 0:
        logging.info("Feature Skew level is 0. No transformation applied.")
        return client_data

    logging.info(f"Applying Feature Skew (level={level})...")
    rng = np.random.RandomState(42) # Separate RNG for transformations

    for client_id, data in client_data.items():
        if data['X'].shape[0] == 0: continue # Skip empty clients

        # Simple transformation: random scaling + random shift per client
        # Scale magnitude based on level
        scale_variance = level * 0.5 # Max scale factor variation around 1
        shift_variance = level * 1.0 # Max shift magnitude

        # Generate random scale factors (per feature) and shifts (per feature)
        scale_factors = rng.uniform(1 - scale_variance, 1 + scale_variance, size=n_features)
        shifts = rng.normal(0, shift_variance, size=n_features)

        # Apply transformation: X_new = X * scale_factors + shifts
        data['X'] = data['X'] * scale_factors + shifts
        logging.debug(f"  Client {client_id}: Applied shift/scale transformation.")

    return client_data


def _apply_concept_drift(client_data, level):
    """
    Shifts the mean of features for each class differently per client.
    Level 0: No drift. Level 1: Significant mean shifts.
    """
    if level == 0:
        logging.info("Concept Drift level is 0. No drift applied.")
        return client_data

    logging.info(f"Applying Concept Drift (level={level})...")
    rng = np.random.RandomState(43) # Separate RNG

    # Calculate global means per class for reference (optional, could just drift from current)
    all_X = np.concatenate([d['X'] for d in client_data.values() if d['X'].shape[0]>0])
    all_y = np.concatenate([d['y'] for d in client_data.values() if d['y'].shape[0]>0])
    if len(all_X) == 0:
        logging.warning("No data to apply concept drift.")
        return client_data

    n_features = all_X.shape[1]
    classes = np.unique(all_y)
    global_means = {k: all_X[all_y == k].mean(axis=0) if (all_y == k).sum() > 0 else np.zeros(n_features) for k in classes}

    drift_magnitude_scale = level * 0.5 # Controls how far means drift

    for client_id, data in client_data.items():
        if data['X'].shape[0] == 0: continue
        X_i, y_i = data['X'], data['y']
        X_i_new = X_i.copy()
        client_classes = np.unique(y_i)

        # Generate client-specific drift vectors for each class it holds
        client_drifts = {k: rng.normal(0, drift_magnitude_scale, size=n_features) for k in client_classes}

        for k in client_classes:
            class_mask = (y_i == k)
            n_class_samples = class_mask.sum()
            if n_class_samples == 0: continue

            current_mean = X_i[class_mask].mean(axis=0)
            # Define target mean: could be global mean + drift, or just current + drift
            # Let's use current + drift for simplicity
            target_mean = current_mean + client_drifts[k]

            # Shift features for this class towards the target mean
            X_i_new[class_mask] = X_i[class_mask] - current_mean + target_mean
            logging.debug(f"  Client {client_id}, Class {k}: Applied concept drift.")

        data['X'] = X_i_new

    return client_data


def _apply_concept_shift(client_data, level, n_classes):
    """
    Flips labels for a fraction of data points per client.
    Level 0: No flips. Level 1: High flip probability.
    """
    if level == 0:
        logging.info("Concept Shift level is 0. No label flipping applied.")
        return client_data

    logging.info(f"Applying Concept Shift (level={level})...")
    rng = np.random.RandomState(44) # Separate RNG

    max_flip_prob = level * 0.5 # E.g., level 1 means up to 50% flip probability
    
    for client_id, data in client_data.items():
        n_samples = data['y'].shape[0]
        if n_samples == 0: continue

        y_i = data['y'].copy()
        
        # Determine which samples to potentially flip
        flip_prob = rng.uniform(0, max_flip_prob) # Client-specific base probability
        flip_mask = rng.rand(n_samples) < flip_prob
        n_to_flip = flip_mask.sum()

        if n_to_flip > 0:
             logging.debug(f"  Client {client_id}: Flipping labels for {n_to_flip} samples (prob ~{flip_prob:.3f}).")
             indices_to_flip = np.where(flip_mask)[0]

             for idx in indices_to_flip:
                 original_label = y_i[idx]
                 possible_new_labels = list(range(n_classes))
                 possible_new_labels.remove(original_label)
                 
                 if not possible_new_labels: # Only one class? Cannot flip.
                      continue

                 # Choose a new label randomly
                 new_label = rng.choice(possible_new_labels)
                 y_i[idx] = new_label
        
        data['y'] = y_i # Update client data with potentially flipped labels

    return client_data

# --- Core Generation Function ---

def generate_data(n_samples, n_features, n_classes, n_clients,
                  quantity_skew_alpha, label_skew_alpha,
                  feature_skew_level, concept_drift_level, concept_shift_level,
                  base_class_sep=1.0, base_n_informative_frac=0.8,
                  output_dir='federated_data'):
    """Orchestrates the generation of the non-IID federated dataset."""

    logging.info("Starting dataset generation...")

    # 1. Generate Base Global Data
    X_global, y_global = _generate_base_data(n_samples, n_features, n_classes,
                                             base_class_sep, base_n_informative_frac)
    logging.info(f"Generated base data: X shape={X_global.shape}, y shape={y_global.shape}")

    # 2. Apply Quantity Skew (Determine N_i for each client)
    # Use a default large alpha if skew is not desired, low alpha for high skew
    client_sample_counts = _apply_quantity_skew(n_samples, n_clients, quantity_skew_alpha)

    # 3. Apply Label Skew (Partition data according to N_i and target P_i(y))
    # Use a default large alpha if skew is not desired, low alpha for high skew
    client_data = _apply_label_skew(X_global, y_global, client_sample_counts,
                                     n_classes, label_skew_alpha)

    # --- Apply skews that modify data *within* clients ---
    # Important: These should operate on the already partitioned data

    # 4. Apply Feature Skew (Transform X_i per client)
    client_data = _apply_feature_skew(client_data, feature_skew_level, n_features)

    # 5. Apply Concept Drift (Shift class means within X_i per client)
    client_data = _apply_concept_drift(client_data, concept_drift_level)

    # 6. Apply Concept Shift (Flip labels y_i per client)
    client_data = _apply_concept_shift(client_data, concept_shift_level, n_classes)


    # Final check and save
    check_data_shapes(client_data)
    save_data(client_data, output_dir)

    logging.info("Dataset generation complete.")
    return client_data


# --- Visualization Function ---

def visualize_data(client_data, n_samples_to_plot=1000, method='pca', title="Generator.py"):
    """Visualizes client data distributions using PCA or t-SNE."""
    logging.info(f"Starting visualization using {method.upper()}...")

    all_X_list = []
    all_y_list = []
    all_client_ids_list = []
    total_samples = sum(data['X'].shape[0] for data in client_data.values())

    if total_samples == 0:
        logging.warning("No data available to visualize.")
        return

    # Determine if sampling is needed
    sampling_ratio = min(1.0, n_samples_to_plot / total_samples if total_samples > 0 else 1.0)
    if sampling_ratio < 1.0:
        logging.info(f"Sampling {n_samples_to_plot} points ({sampling_ratio*100:.2f}%) for visualization.")

    for client_id, data in client_data.items():
        n_client_samples = data['X'].shape[0]
        if n_client_samples == 0:
            continue

        # Determine number of samples to take from this client
        n_take = max(1, int(round(n_client_samples * sampling_ratio))) if sampling_ratio < 1.0 else n_client_samples
        n_take = min(n_take, n_client_samples) # Cannot take more than available

        if n_take > 0:
            indices = np.random.choice(n_client_samples, size=n_take, replace=False)
            all_X_list.append(data['X'][indices])
            all_y_list.append(data['y'][indices])
            all_client_ids_list.append(np.full(n_take, client_id))

    if not all_X_list:
        logging.warning("No samples selected for visualization after sampling.")
        return

    X_sample = np.concatenate(all_X_list)
    y_sample = np.concatenate(all_y_list)
    client_ids_sample = np.concatenate(all_client_ids_list)

    n_features = X_sample.shape[1]
    n_components = min(3, n_features) # Reduce to 3D if possible, else 2D or original

    if n_components < 2:
         logging.warning(f"Cannot visualize with less than 2 features (found {n_features}). Skipping dimensionality reduction.")
         X_reduced = X_sample
         n_components = n_features # Plot original features if 1D or 2D
    elif n_features > n_components:
        logging.info(f"Reducing dimensionality from {n_features} to {n_components} using {method.upper()}...")
        if method == 'tsne':
            # t-SNE settings might need tuning
            perplexity_val = min(30.0, max(5.0, X_sample.shape[0] / 4.0)) # Adjust perplexity based on sample size
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity_val, n_iter=300)
            X_reduced = tsne.fit_transform(X_sample)
        else: # Default to PCA
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X_sample)
        logging.info("Dimensionality reduction complete.")
    else:
        logging.info("Data has 3 or fewer features. Plotting original features.")
        X_reduced = X_sample # Already 3D or less

    # --- Plotting ---
    fig = plt.figure(figsize=(18, 8))
    fig.canvas.manager.set_window_title(title)

    n_clients = len(client_data)
    n_classes = len(np.unique(y_sample)) # Use unique labels found in sample

    # Plot 1: Color by Client ID
    if n_components >= 3:
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=client_ids_sample, cmap=plt.cm.get_cmap('turbo', n_clients), alpha=0.6)
        ax1.set_zlabel('Component 3')
    elif n_components == 2:
        ax1 = fig.add_subplot(121)
        scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=client_ids_sample, cmap=plt.cm.get_cmap('turbo', n_clients), alpha=0.6)
    else: # n_components == 1
        ax1 = fig.add_subplot(121)
        # Simple strip plot for 1D
        scatter1 = ax1.scatter(X_reduced[:, 0], np.zeros_like(X_reduced[:, 0]), c=client_ids_sample, cmap=plt.cm.get_cmap('turbo', n_clients), alpha=0.6)

    ax1.set_title(f'Data Distribution by Client ID ({method.upper()})')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    legend1 = ax1.legend(*scatter1.legend_elements(num=min(n_clients, 10)), title="Clients") # Show limited legend entries
    if legend1: ax1.add_artist(legend1)


    # Plot 2: Color by Class Label
    if n_components >= 3:
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_sample, cmap=plt.cm.get_cmap('viridis', n_classes), alpha=0.6)
        ax2.set_zlabel('Component 3')
    elif n_components == 2:
        ax2 = fig.add_subplot(122)
        scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_sample, cmap=plt.cm.get_cmap('viridis', n_classes), alpha=0.6)
    else: # n_components == 1
        ax2 = fig.add_subplot(122)
        scatter2 = ax2.scatter(X_reduced[:, 0], np.zeros_like(X_reduced[:, 0]), c=y_sample, cmap=plt.cm.get_cmap('viridis', n_classes), alpha=0.6)

    ax2.set_title(f'Data Distribution by Class Label ({method.upper()})')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    legend2 = ax2.legend(*scatter2.legend_elements(num=min(n_classes, 10)), title="Classes")
    if legend2: ax2.add_artist(legend2)


    plt.tight_layout()
    plt.show()


# --- Main Execution & CLI Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Synthetic Non-IID Federated Datasets")

    # Base Dataset Parameters
    parser.add_argument('--n_samples', type=int, default=10000, help='Total number of samples to generate globally')
    parser.add_argument('--n_features', type=int, default=20, help='Number of features')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--n_clients', type=int, default=100, help='Number of clients')
    parser.add_argument('--class_sep', type=float, default=1.0, help='Class separation factor for make_classification')
    parser.add_argument('--info_frac', type=float, default=0.8, help='Fraction of informative features for make_classification')

    # Skew Control Parameters
    parser.add_argument('--quantity_skew_alpha', type=float, default=1.0,
                        help='Dirichlet alpha for quantity skew (smaller=more skew, e.g., 0.1 extreme, 100 low)')
    parser.add_argument('--label_skew_alpha', type=float, default=1.0,
                        help='Dirichlet alpha for label distribution skew (smaller=more skew)')
    parser.add_argument('--feature_skew_level', type=float, default=0.0,
                        help='Level (0 to 1) of feature transformation skew (0=none, 1=high)')
    parser.add_argument('--concept_drift_level', type=float, default=0.0,
                        help='Level (0 to 1) of concept drift (class mean shift) (0=none, 1=high)')
    parser.add_argument('--concept_shift_level', type=float, default=0.0,
                        help='Level (0 to 1) of concept shift (label flipping) (0=none, 1=high)')

    # Output & Visualization
    parser.add_argument('--output_dir', type=str, default='federated_data', help='Directory to save the generated dataset')
    parser.add_argument('--visualize', action='store_true', help='Visualize the generated data distribution')
    parser.add_argument('--vis_samples', type=int, default=2000, help='Max number of samples to use for visualization')
    parser.add_argument('--vis_method', type=str, default='pca', choices=['pca', 'tsne'], help='Dimensionality reduction method for visualization')

    return parser.parse_args()

def main():
    args = parse_args()

    # Validate parameters
    if args.quantity_skew_alpha <= 0:
        raise ValueError("quantity_skew_alpha must be positive.")
    if args.label_skew_alpha <= 0:
        raise ValueError("label_skew_alpha must be positive.")
    if not (0 <= args.feature_skew_level <= 1):
        raise ValueError("feature_skew_level must be between 0 and 1.")
    if not (0 <= args.concept_drift_level <= 1):
        raise ValueError("concept_drift_level must be between 0 and 1.")
    if not (0 <= args.concept_shift_level <= 1):
         raise ValueError("concept_shift_level must be between 0 and 1.")
    if args.n_features < 2 and args.visualize:
        logging.warning("Visualization might be trivial with n_features < 2.")
        if args.vis_method == 'tsne':
             logging.warning("t-SNE requires n_features >= 2. Defaulting visualization method to PCA or direct plot.")
             # Note: visualization function handles <2 components automatically
             args.vis_method = 'pca' # Fallback for logic check

    client_data = generate_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_clients=args.n_clients,
        quantity_skew_alpha=args.quantity_skew_alpha,
        label_skew_alpha=args.label_skew_alpha,
        feature_skew_level=args.feature_skew_level,
        concept_drift_level=args.concept_drift_level,
        concept_shift_level=args.concept_shift_level,
        base_class_sep=args.class_sep,
        base_n_informative_frac=args.info_frac,
        output_dir=args.output_dir
    )

    if args.visualize:
        # Load data back if needed (or just pass client_data if generation was successful)
        # data_path = os.path.join(args.output_dir, "federated_dataset.pkl")
        # with open(data_path, 'rb') as f:
        #     client_data_loaded = pickle.load(f)
        title = f"Non-IID Synthetic Data (Clients={args.n_clients}, Classes={args.n_classes})"
        title += f" Quantity Skew (alpha={args.quantity_skew_alpha}), Label Skew (alpha={args.label_skew_alpha})"
        title += f" Feature Skew (level={args.feature_skew_level}), Concept Drift (level={args.concept_drift_level}), Concept Shift (level={args.concept_shift_level})"
        visualize_data(client_data, n_samples_to_plot=args.vis_samples, method=args.vis_method, title=title)

if __name__ == "__main__":
    # Example Usage:
    # Basic IID-like (high alpha, low levels)
    # python generator.py --n_samples 5000 --n_features 10 --n_classes 5 --n_clients 10 --quantity_skew_alpha 100 --label_skew_alpha 100 --visualize --vis_samples 1000

    # Strong Label Skew + Moderate Quantity Skew
    # python generator.py --n_samples 10000 --n_features 20 --n_classes 10 --n_clients 50 --quantity_skew_alpha 0.5 --label_skew_alpha 0.2 --visualize --vis_samples 2000 --vis_method tsne

    # Feature Skew + Concept Drift
    # python generator.py --n_samples 10000 --n_features 20 --n_classes 10 --n_clients 50 --quantity_skew_alpha 5.0 --label_skew_alpha 5.0 --feature_skew_level 0.8 --concept_drift_level 0.5 --visualize

    # Concept Shift
    # python generator.py --n_samples 5000 --n_features 10 --n_classes 3 --n_clients 20 --concept_shift_level 0.7 --visualize

    # python generator.py --n_samples 1000 --n_features 3 --n_classes 5 --n_clients 10 --quantity_skew_alpha 100 --label_skew_alpha 100 --visualize --vis_samples 1000 --info_frac 1
    # python generator.py --n_samples 1000 --n_features 3 --n_classes 5 --n_clients 10 --quantity_skew_alpha 100 --label_skew_alpha 1 --visualize --vis_samples 1000 --info_frac 1
    # python generator.py --n_samples 1000 --n_features 3 --n_classes 5 --n_clients 10 --quantity_skew_alpha 1 --label_skew_alpha 100 --visualize --vis_samples 1000 --info_frac 1
    
    main()