# non_iid_generator_interactive.py

import argparse
import numpy as np
import os
import pickle
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib # Required for backend setting
# Set backend *before* importing pyplot if using GUI
# matplotlib.use('TkAgg') # Moved this logic to GUI startup
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
import warnings
import logging
import math
from collections import defaultdict
import tkinter as tk
from tkinter import ttk # For themed widgets
from tkinter import messagebox
from tkinter import filedialog
import sys

# --- Configure logging (same as before) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions (same as before) ---

def save_data(client_data, output_dir, filename="federated_dataset.pkl"):
    """Saves the partitioned client data to a pickle file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(client_data, f)
        logging.info(f"Dataset saved successfully to {output_path}")
        return True # Indicate success
    except Exception as e:
        logging.error(f"Failed to save dataset to {output_path}: {e}")
        messagebox.showerror("Save Error", f"Failed to save dataset:\n{e}")
        return False # Indicate failure

def check_data_shapes(client_data):
    """Logs the shapes of data for each client."""
    logging.info("--- Client Data Shapes ---")
    total_samples = 0
    if not client_data:
        logging.warning("Client data is empty.")
        return 0
    for client_id, data in client_data.items():
        # Handle potentially missing 'X' or 'y' keys if loading malformed data
        shape_x = data.get('X', np.empty((0,0))).shape
        shape_y = data.get('y', np.empty((0,))).shape
        n_samples = shape_x[0]
        total_samples += n_samples
        logging.info(f"Client {client_id}: X shape={shape_x}, y shape={shape_y}, Samples={n_samples}")
        # Check for empty clients
        if n_samples == 0:
            logging.warning(f"Client {client_id} has 0 samples!")
    logging.info(f"Total samples across all clients: {total_samples}")
    logging.info("--------------------------")
    return total_samples


# --- Skew Implementation Functions (same as before) ---

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
        # Raise ValueError in CLI, show error in GUI
        logging.error("Dirichlet alpha for quantity skew must be positive.")
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
        # Leave at least 0 for others (allow empty clients)
        samples = min(samples, remaining_samples)
        samples = max(0, samples) # Ensure non-negative
        client_sample_counts[i] = samples
        remaining_samples -= samples
    client_sample_counts[n_clients - 1] = remaining_samples # Assign rest to last client

    # Check final counts
    actual_total = client_sample_counts.sum()
    if actual_total != n_samples:
         logging.warning(f"Sample count mismatch after quantity skew. "
                         f"Target: {n_samples}, Actual: {actual_total}. "
                         f"Difference: {n_samples - actual_total}. "
                         f"This might happen due to rounding.")
         # Simple redistribution of difference (can be improved)
         diff = n_samples - actual_total
         adjust_idx = 0
         # Only adjust if we have clients to adjust on
         active_clients = np.where(client_sample_counts > 0)[0] if diff < 0 else np.arange(n_clients)
         if len(active_clients) == 0 and diff != 0:
             logging.error("Cannot correct sample count mismatch - no suitable clients to adjust.")
         else:
             while diff != 0 and len(active_clients) > 0:
                 idx_to_adjust = active_clients[adjust_idx % len(active_clients)]
                 if diff > 0:
                     client_sample_counts[idx_to_adjust] += 1
                     diff -= 1
                 elif client_sample_counts[idx_to_adjust] > 0: # Only reduce if count > 0
                     client_sample_counts[idx_to_adjust] -= 1
                     diff += 1
                 adjust_idx += 1
                 if adjust_idx > 2 * n_samples: # Safety break for large adjustments
                     logging.error("Could not fully correct sample count mismatch after many attempts.")
                     break


    # Ensure no client has negative samples after adjustment
    client_sample_counts = np.maximum(0, client_sample_counts)

    # Final check
    if client_sample_counts.sum() != n_samples:
        logging.error(f"FINAL Sample count mismatch. Target: {n_samples}, Actual: {client_sample_counts.sum()}")

    logging.info(f"Applied Quantity Skew (alpha={alpha}). Client sample counts: {client_sample_counts.tolist()}")
    return client_sample_counts.tolist()


def _apply_label_skew(X_global, y_global, client_sample_counts, n_classes, alpha):
    """
    Partitions data based on client sample counts and Dirichlet-sampled label distributions.
    """
    if alpha <= 0:
        logging.error("Dirichlet alpha for label skew must be positive.")
        raise ValueError("Dirichlet alpha must be positive.")

    n_samples_total, n_features = X_global.shape
    n_clients = len(client_sample_counts)
    client_data = {i: {'X': None, 'y': None} for i in range(n_clients)}
    data_indices = np.arange(n_samples_total)

    # Get indices for each class
    class_indices = {k: data_indices[y_global == k] for k in range(n_classes)}
    available_indices_count = {k: len(class_indices[k]) for k in range(n_classes)}
    if sum(available_indices_count.values()) == 0 and n_samples_total > 0:
        logging.error("No samples found for any class, cannot apply label skew.")
        # Create empty client data structure
        for i in range(n_clients):
             client_data[i] = {
                'X': np.empty((0, n_features)),
                'y': np.empty((0,), dtype=y_global.dtype)
            }
        return client_data

    assigned_indices = set()

    # Target label distribution per client
    client_label_proportions = np.random.dirichlet(np.full(n_classes, alpha), size=n_clients)

    logging.info(f"Target label distributions (proportions) per client (alpha={alpha}):")
    for i in range(n_clients):
        logging.debug(f"  Client {i}: {client_label_proportions[i].round(3)}")

    client_indices = {i: [] for i in range(n_clients)}

    # Calculate target counts per class per client precisely
    target_counts_per_class_client = np.zeros((n_clients, n_classes), dtype=int)
    total_target_counts_per_class = np.zeros(n_classes, dtype=int)

    for i in range(n_clients):
        if client_sample_counts[i] == 0: continue
        proportions = client_label_proportions[i]
        counts = np.zeros(n_classes, dtype=int)
        assigned_count_i = 0
        # Use multinomial distribution for potentially better distribution of samples
        # according to proportions for a fixed total client sample count.
        try:
            counts = np.random.multinomial(client_sample_counts[i], proportions)
            assigned_count_i = counts.sum()
        except ValueError as e:
            # Fallback to rounding method if multinomial fails (e.g., sum(pvals) > 1 due to float issues)
            logging.warning(f"Multinomial sampling failed for client {i} (proportions sum: {proportions.sum()}): {e}. Falling back to rounding.")
            proportions /= proportions.sum() # Ensure sum is 1
            remaining = client_sample_counts[i]
            for k in range(n_classes - 1):
                c = min(remaining, int(round(proportions[k] * client_sample_counts[i])))
                c = max(0, c)
                counts[k] = c
                remaining -= c
            counts[n_classes - 1] = remaining # Assign remainder
            assigned_count_i = counts.sum()


        # Adjust if sum doesn't match N_i (can happen with rounding method, less likely with multinomial)
        count_sum_diff = client_sample_counts[i] - assigned_count_i
        adjust_idx = 0
        active_classes = np.arange(n_classes) # Classes to potentially adjust
        while count_sum_diff != 0 and len(active_classes) > 0:
           class_to_adjust = active_classes[adjust_idx % len(active_classes)]
           if count_sum_diff > 0:
               counts[class_to_adjust] += 1
               count_sum_diff -= 1
           elif counts[class_to_adjust] > 0: # Only reduce if count > 0
               counts[class_to_adjust] -= 1
               count_sum_diff += 1
           adjust_idx += 1
           if adjust_idx > 2 * n_classes and count_sum_diff != 0 : # Safety break
                logging.error(f"Client {i}: Could not fully correct target class counts sum. Diff: {count_sum_diff}")
                break

        target_counts_per_class_client[i] = np.maximum(0, counts) # Ensure non-negative
        total_target_counts_per_class += target_counts_per_class_client[i]

    logging.info("Target sample counts per class per client:")
    for i in range(n_clients):
         logging.debug(f"  Client {i} (Total {client_sample_counts[i]}): {target_counts_per_class_client[i].tolist()}")

    # Check if total target counts per class exceed available counts
    for k in range(n_classes):
        if total_target_counts_per_class[k] > available_indices_count[k]:
             logging.warning(f"Total target count for class {k} ({total_target_counts_per_class[k]}) "
                             f"exceeds available samples ({available_indices_count[k]}). "
                             f"Allocation might not perfectly match targets.")

    # Now, distribute actual indices iteratively
    available_class_indices_sets = {k: set(inds) for k, inds in class_indices.items()}
    client_needs = {i: target_counts_per_class_client[i].copy() for i in range(n_clients)}

    # Prioritize clients/classes with fewer options or higher demand?
    # Simple approach: Iterate through classes, then clients needing that class.
    for k in range(n_classes):
        available_for_k = list(available_class_indices_sets[k])
        np.random.shuffle(available_for_k) # Shuffle to avoid bias
        available_idx_ptr = 0

        # Sort clients by their need for class k (descending)? Helps give samples to neediest first.
        clients_needing_k = [i for i in range(n_clients) if client_needs[i][k] > 0]
        clients_needing_k.sort(key=lambda i: client_needs[i][k], reverse=True)

        for i in clients_needing_k:
            needed = client_needs[i][k]
            n_take = min(needed, len(available_for_k) - available_idx_ptr)

            if n_take > 0:
                start = available_idx_ptr
                end = available_idx_ptr + n_take
                chosen_indices = available_for_k[start:end]
                client_indices[i].extend(chosen_indices)
                client_needs[i][k] -= n_take
                assigned_indices.update(chosen_indices)
                available_idx_ptr += n_take

            if client_needs[i][k] > 0:
                 logging.warning(f"Client {i}, Class {k}: Still needs {client_needs[i][k]} samples after initial allocation pass.")


    # Handle remaining shortfalls (clients that didn't get their target total)
    unassigned_overall = list(set(data_indices) - assigned_indices)
    np.random.shuffle(unassigned_overall)
    unassigned_ptr = 0
    logging.info(f"{len(unassigned_overall)} samples remain unassigned after primary allocation.")

    for i in range(n_clients):
         current_assigned = len(client_indices[i])
         total_shortfall = client_sample_counts[i] - current_assigned
         if total_shortfall > 0:
             logging.warning(f"Client {i}: Shortfall of {total_shortfall} samples (Target: {client_sample_counts[i]}, Assigned: {current_assigned}). Trying to fill.")
             n_fill = min(total_shortfall, len(unassigned_overall) - unassigned_ptr)
             if n_fill > 0:
                 start = unassigned_ptr
                 end = unassigned_ptr + n_fill
                 fill_indices = unassigned_overall[start:end]
                 client_indices[i].extend(fill_indices)
                 assigned_indices.update(fill_indices)
                 unassigned_ptr += n_fill
                 logging.info(f"Client {i}: Filled {n_fill} samples from unassigned pool.")
             elif len(unassigned_overall) - unassigned_ptr == 0:
                 logging.error(f"Client {i}: Could not fill shortfall, no unassigned samples left.")


    # Create final client data dictionary
    final_client_data = {}
    for i in range(n_clients):
        indices = np.array(client_indices[i], dtype=int)
        if len(indices) > 0:
            # Ensure indices are valid before slicing
            valid_indices = indices[indices < n_samples_total]
            if len(valid_indices) != len(indices):
                logging.error(f"Client {i}: Invalid indices detected ({len(indices) - len(valid_indices)} out of bounds). Clipping.")
                indices = valid_indices

            if len(indices) > 0:
                final_client_data[i] = {
                    'X': X_global[indices],
                    'y': y_global[indices]
                }
            else:
                 final_client_data[i] = {
                    'X': np.empty((0, n_features)),
                    'y': np.empty((0,), dtype=y_global.dtype)
                }
        else:
             # Handle clients originally assigned 0 samples or ending up with 0
             final_client_data[i] = {
                'X': np.empty((0, n_features)),
                'y': np.empty((0,), dtype=y_global.dtype)
            }

    logging.info(f"Applied Label Skew (alpha={alpha}). Data partitioned.")
    return final_client_data


def _apply_feature_skew(client_data, level, n_features):
    """
    Applies client-specific feature transformations (shift + scale).
    Level 0: No transformation. Level > 0: Increasing transformation.
    """
    if level == 0:
        logging.info("Feature Skew level is 0. No transformation applied.")
        return client_data
    if level < 0:
        logging.warning("Feature skew level is negative, treating as 0.")
        return client_data


    logging.info(f"Applying Feature Skew (level={level})...")
    rng = np.random.RandomState(42) # Separate RNG for transformations

    for client_id, data in client_data.items():
        if data['X'].shape[0] == 0: continue # Skip empty clients

        # Simple transformation: random scaling + random shift per client
        # Scale magnitude based on level
        scale_variance = level * 0.5 # Max scale factor variation around 1 (e.g., level 1 -> [0.5, 1.5])
        shift_variance = level * 1.0 # Std Dev of shift (e.g., level 1 -> N(0, 1) shifts)

        # Generate random scale factors (per feature) and shifts (per feature)
        # Ensure scale factors are positive
        scale_factors = rng.uniform(max(0.01, 1 - scale_variance), 1 + scale_variance, size=n_features)
        shifts = rng.normal(0, shift_variance, size=n_features)

        # Apply transformation: X_new = X * scale_factors + shifts
        data['X'] = data['X'] * scale_factors + shifts
        logging.debug(f"  Client {client_id}: Applied shift/scale transformation.")

    return client_data


def _apply_concept_drift(client_data, level):
    """
    Shifts the mean of features for each class differently per client.
    Level 0: No drift. Level > 0: Increasing mean shifts.
    """
    if level == 0:
        logging.info("Concept Drift level is 0. No drift applied.")
        return client_data
    if level < 0:
        logging.warning("Concept drift level is negative, treating as 0.")
        return client_data

    logging.info(f"Applying Concept Drift (level={level})...")
    rng = np.random.RandomState(43) # Separate RNG

    # Check if there's any data
    all_X_list = [d['X'] for d in client_data.values() if d['X'].shape[0]>0]
    if not all_X_list:
        logging.warning("No data to apply concept drift.")
        return client_data

    all_X = np.concatenate(all_X_list)
    all_y = np.concatenate([d['y'] for d in client_data.values() if d['y'].shape[0]>0])
    n_features = all_X.shape[1]
    classes = np.unique(all_y)

    # Use std dev of features as a base for drift magnitude? Or fixed scale?
    # Fixed scale based on level might be simpler to control.
    drift_magnitude_scale = level * 0.5 # Std Dev of the drift vector components

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

            # Apply the drift vector directly to the samples of that class
            # X_new = X_old + drift_vector
            X_i_new[class_mask] = X_i[class_mask] + client_drifts[k]
            logging.debug(f"  Client {client_id}, Class {k}: Applied concept drift shift.")

        data['X'] = X_i_new

    return client_data


def _apply_concept_shift(client_data, level, n_classes):
    """
    Flips labels for a fraction of data points per client.
    Level 0: No flips. Level (0, 1]: Probability of flipping increases.
    """
    if level == 0:
        logging.info("Concept Shift level is 0. No label flipping applied.")
        return client_data
    if not (0 <= level <= 1):
         logging.warning("Concept shift level must be between 0 and 1. Clipping.")
         level = max(0, min(1, level))


    logging.info(f"Applying Concept Shift (level={level})...")
    rng = np.random.RandomState(44) # Separate RNG

    # Level directly determines the *probability* for a sample to be flipped
    # Max flip prob could be linked to level, or level could be the direct prob.
    # Let's use level as the probability for simplicity.
    flip_prob = level

    total_flipped = 0
    total_samples_processed = 0

    for client_id, data in client_data.items():
        n_samples = data['y'].shape[0]
        total_samples_processed += n_samples
        if n_samples == 0: continue

        y_i = data['y'].copy()

        # Determine which samples to flip
        flip_mask = rng.rand(n_samples) < flip_prob
        n_to_flip = flip_mask.sum()
        total_flipped += n_to_flip

        if n_to_flip > 0:
             logging.debug(f"  Client {client_id}: Flipping labels for {n_to_flip} samples (prob={flip_prob:.3f}).")
             indices_to_flip = np.where(flip_mask)[0]

             for idx in indices_to_flip:
                 original_label = y_i[idx]
                 possible_new_labels = list(range(n_classes))
                 if n_classes > 1:
                    possible_new_labels.remove(original_label)
                 else: # Only one class? Cannot flip.
                      continue

                 if not possible_new_labels: # Should only happen if n_classes=1
                      continue

                 # Choose a new label randomly from the alternatives
                 new_label = rng.choice(possible_new_labels)
                 y_i[idx] = new_label

        data['y'] = y_i # Update client data with potentially flipped labels

    logging.info(f"Concept Shift complete. Flipped {total_flipped} labels out of {total_samples_processed} processed ({total_flipped/total_samples_processed*100:.2f}% effective flip rate).")
    return client_data

# --- Core Generation Function (modified slightly for error handling) ---

def generate_data(n_samples, n_features, n_classes, n_clients,
                  quantity_skew_alpha, label_skew_alpha,
                  feature_skew_level, concept_drift_level, concept_shift_level,
                  base_class_sep=1.0, base_n_informative_frac=0.8,
                  output_dir='federated_data'):
    """Orchestrates the generation of the non-IID federated dataset."""

    logging.info("Starting dataset generation...")
    client_data = None # Initialize

    try:
        # --- Parameter Validation --- (Basic checks)
        if n_samples <= 0 or n_features <= 0 or n_classes <= 0 or n_clients <= 0:
             raise ValueError("Samples, features, classes, and clients must be positive integers.")
        if not (0 < base_n_informative_frac <= 1):
            raise ValueError("Informative fraction must be between 0 (exclusive) and 1 (inclusive).")
        # Alpha checks are done within their respective functions now
        # Level checks are done within their respective functions now

        # 1. Generate Base Global Data
        X_global, y_global = _generate_base_data(n_samples, n_features, n_classes,
                                                 base_class_sep, base_n_informative_frac)
        logging.info(f"Generated base data: X shape={X_global.shape}, y shape={y_global.shape}")

        # 2. Apply Quantity Skew
        client_sample_counts = _apply_quantity_skew(n_samples, n_clients, quantity_skew_alpha)

        # 3. Apply Label Skew
        client_data = _apply_label_skew(X_global, y_global, client_sample_counts,
                                         n_classes, label_skew_alpha)

        # --- Apply skews that modify data *within* clients ---
        # 4. Apply Feature Skew
        client_data = _apply_feature_skew(client_data, feature_skew_level, n_features)

        # 5. Apply Concept Drift
        client_data = _apply_concept_drift(client_data, concept_drift_level)

        # 6. Apply Concept Shift
        client_data = _apply_concept_shift(client_data, concept_shift_level, n_classes)

        # Final check and save
        total_samples = check_data_shapes(client_data)
        if total_samples > 0 : # Only save if data was generated
             saved = save_data(client_data, output_dir)
             if not saved:
                 logging.error("Failed to save the generated data.")
                 # Don't return data if save failed? Or return anyway? Return for potential visualization.
        elif total_samples == 0 and n_samples > 0:
             logging.warning("Generation resulted in 0 total samples across clients. Skipping save.")
        else: # n_samples was 0 initially
             logging.info("n_samples was 0, no data generated or saved.")


        logging.info("Dataset generation process complete.")

    except ValueError as ve:
        logging.error(f"Data Generation Error: {ve}")
        messagebox.showerror("Generation Error", f"Invalid parameter value:\n{ve}")
        return None # Indicate failure
    except Exception as e:
        logging.exception(f"An unexpected error occurred during data generation: {e}")
        messagebox.showerror("Generation Error", f"An unexpected error occurred:\n{e}")
        return None # Indicate failure

    return client_data


# --- Visualization Function (same as before, added title flexibility) ---

def visualize_data(client_data, n_samples_to_plot=1000, method='pca', title="Data Visualization"):
    """Visualizes client data distributions using PCA or t-SNE."""
    if not client_data:
        logging.warning("No client data provided for visualization.")
        messagebox.showwarning("Visualize", "No data loaded or generated to visualize.")
        return

    logging.info(f"Starting visualization using {method.upper()}...")

    all_X_list = []
    all_y_list = []
    all_client_ids_list = []
    total_samples = sum(data['X'].shape[0] for data in client_data.values() if 'X' in data and data['X'] is not None)

    if total_samples == 0:
        logging.warning("No data samples available across clients to visualize.")
        messagebox.showwarning("Visualize", "The loaded/generated data contains 0 samples.")
        return

    # Determine if sampling is needed
    n_samples_to_plot = min(n_samples_to_plot, total_samples) # Cant plot more than exist
    sampling_ratio = n_samples_to_plot / total_samples if total_samples > 0 else 1.0

    if sampling_ratio < 1.0:
        logging.info(f"Sampling {n_samples_to_plot} points ({sampling_ratio*100:.2f}%) for visualization.")

    for client_id, data in client_data.items():
        # Check if data is valid and has samples
        if 'X' not in data or 'y' not in data or data['X'] is None or data['y'] is None:
            logging.warning(f"Client {client_id} has invalid data structure. Skipping.")
            continue
        n_client_samples = data['X'].shape[0]
        if n_client_samples == 0:
            continue

        # Determine number of samples to take from this client
        # Ensure at least 1 sample if ratio > 0 and client has samples, unless n_samples_to_plot is very small
        if sampling_ratio < 1.0:
            n_take = int(round(n_client_samples * sampling_ratio))
            # Ensure we take at least one if the client is non-empty and target samples > 0
            if n_take == 0 and n_samples_to_plot > 0 and n_client_samples > 0:
                 # This can happen if a client has very few samples compared to total
                 # Let's take at least 1 if we haven't reached the total plot samples yet
                 current_plot_count = sum(len(l) for l in all_X_list)
                 if current_plot_count < n_samples_to_plot:
                     n_take = 1
                     logging.debug(f"Taking 1 sample from client {client_id} to ensure representation.")

        else:
             n_take = n_client_samples

        n_take = min(n_take, n_client_samples) # Cannot take more than available

        if n_take > 0:
            try:
                indices = np.random.choice(n_client_samples, size=n_take, replace=False)
                all_X_list.append(data['X'][indices])
                all_y_list.append(data['y'][indices])
                all_client_ids_list.append(np.full(n_take, client_id))
            except ValueError as e:
                logging.error(f"Error sampling from client {client_id}: {e}. Skipping client.")
            except Exception as e:
                 logging.exception(f"Unexpected error processing client {client_id}: {e}. Skipping client.")


    if not all_X_list:
        logging.warning("No samples selected for visualization after sampling/processing.")
        messagebox.showwarning("Visualize", "Could not gather any samples for visualization.")
        return

    try:
        X_sample = np.concatenate(all_X_list)
        y_sample = np.concatenate(all_y_list)
        client_ids_sample = np.concatenate(all_client_ids_list)
    except ValueError as e:
         logging.error(f"Error concatenating sampled data: {e}")
         messagebox.showerror("Visualize Error", f"Error preparing data for plot:\n{e}")
         return

    n_features = X_sample.shape[1]
    if n_features == 0:
        logging.warning("Sampled data has 0 features. Cannot visualize.")
        messagebox.showwarning("Visualize", "Sampled data has 0 features. Cannot visualize.")
        return

    n_components = min(3, n_features) # Reduce to 3D if possible, else 2D or original

    X_reduced = None
    if n_components < 2:
         logging.warning(f"Data has only {n_features} feature(s). Plotting directly.")
         X_reduced = X_sample
         n_components = n_features # Plot original features if 1D
    elif n_features > n_components:
        logging.info(f"Reducing dimensionality from {n_features} to {n_components} using {method.upper()}...")
        try:
            if method == 'tsne':
                # t-SNE settings might need tuning
                # Perplexity should be less than the number of samples.
                n_effective_samples = X_sample.shape[0]
                perplexity_val = min(30.0, max(5.0, n_effective_samples / 4.0 ))
                if perplexity_val >= n_effective_samples:
                    perplexity_val = max(1.0, n_effective_samples - 1.0) # Adjust if too few samples
                    logging.warning(f"Adjusted t-SNE perplexity to {perplexity_val} due to small sample size ({n_effective_samples}).")

                tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity_val, n_iter=300, init='pca', learning_rate='auto')
                X_reduced = tsne.fit_transform(X_sample)
            else: # Default to PCA
                pca = PCA(n_components=n_components, random_state=42)
                X_reduced = pca.fit_transform(X_sample)
            logging.info("Dimensionality reduction complete.")
        except Exception as e:
             logging.exception(f"Error during dimensionality reduction ({method.upper()}): {e}")
             messagebox.showerror("Visualize Error", f"Error during {method.upper()}:\n{e}")
             return # Stop visualization if reduction fails
    else:
        logging.info(f"Data has {n_features} features. Plotting original features.")
        X_reduced = X_sample # Already 3D or less


    if X_reduced is None:
        logging.error("Dimensionality reduction failed to produce output.")
        return

    # --- Plotting ---
    try:
        fig = plt.figure(figsize=(18, 8))
        # Use the provided title for the window
        if hasattr(fig.canvas.manager, 'set_window_title'):
             fig.canvas.manager.set_window_title(f"Data Visualization - {title}")
        else: # Fallback if method not available
             fig.suptitle(f"Data Visualization - {title}", fontsize=14)


        unique_clients = np.unique(client_ids_sample)
        n_clients_plot = len(unique_clients)
        unique_classes = np.unique(y_sample)
        n_classes_plot = len(unique_classes)

        # Plot 1: Color by Client ID
        cmap_clients = plt.cm.get_cmap('turbo', n_clients_plot) # Use actual number of clients in plot
        if n_components >= 3:
            ax1 = fig.add_subplot(121, projection='3d')
            scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=client_ids_sample, cmap=cmap_clients, alpha=0.6, s=10) # Smaller points
            ax1.set_zlabel('Component 3')
        elif n_components == 2:
            ax1 = fig.add_subplot(121)
            scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=client_ids_sample, cmap=cmap_clients, alpha=0.6, s=10)
        else: # n_components == 1
            ax1 = fig.add_subplot(121)
            # Simple strip plot for 1D - add jitter for visibility
            y_jitter = np.random.rand(X_reduced.shape[0]) * 0.1 - 0.05
            scatter1 = ax1.scatter(X_reduced[:, 0], y_jitter, c=client_ids_sample, cmap=cmap_clients, alpha=0.6, s=10)
            ax1.yaxis.set_visible(False) # Hide Y axis for strip plot

        ax1.set_title(f'Data Distribution by Client ID ({method.upper()})')
        ax1.set_xlabel('Component 1')
        if n_components > 1: ax1.set_ylabel('Component 2')
        # Create a legend with handles and labels for clarity, limit entries
        n_legend_clients = min(n_clients_plot, 10)
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_clients(i/n_clients_plot), markersize=5)
                          for i in np.linspace(0, n_clients_plot-1, n_legend_clients, dtype=int)]
        legend_labels = [f"Client ~{int(unique_clients[int(i)])}" for i in np.linspace(0, n_clients_plot-1, n_legend_clients, dtype=int)]
        if n_clients_plot > n_legend_clients:
            legend_labels[-1] += f"... ({n_clients_plot} total)"
        legend1 = ax1.legend(legend_handles, legend_labels, title="Clients", loc='best', fontsize='small')
        if legend1: ax1.add_artist(legend1)


        # Plot 2: Color by Class Label
        cmap_classes = plt.cm.get_cmap('viridis', n_classes_plot)
        if n_components >= 3:
            ax2 = fig.add_subplot(122, projection='3d')
            scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_sample, cmap=cmap_classes, alpha=0.6, s=10)
            ax2.set_zlabel('Component 3')
        elif n_components == 2:
            ax2 = fig.add_subplot(122)
            scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_sample, cmap=cmap_classes, alpha=0.6, s=10)
        else: # n_components == 1
            ax2 = fig.add_subplot(122)
            y_jitter = np.random.rand(X_reduced.shape[0]) * 0.1 - 0.05
            scatter2 = ax2.scatter(X_reduced[:, 0], y_jitter, c=y_sample, cmap=cmap_classes, alpha=0.6, s=10)
            ax2.yaxis.set_visible(False)

        ax2.set_title(f'Data Distribution by Class Label ({method.upper()})')
        ax2.set_xlabel('Component 1')
        if n_components > 1: ax2.set_ylabel('Component 2')
        # Legend for classes
        n_legend_classes = min(n_classes_plot, 10)
        legend_handles_cls = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_classes(i/n_classes_plot), markersize=5)
                              for i in np.linspace(0, n_classes_plot-1, n_legend_classes, dtype=int)]
        legend_labels_cls = [f"Class {int(unique_classes[int(i)])}" for i in np.linspace(0, n_classes_plot-1, n_legend_classes, dtype=int)]
        if n_classes_plot > n_legend_classes:
             legend_labels_cls[-1] += f"... ({n_classes_plot} total)"
        legend2 = ax2.legend(legend_handles_cls, legend_labels_cls, title="Classes", loc='best', fontsize='small')
        if legend2: ax2.add_artist(legend2)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()
    except Exception as e:
        logging.exception(f"Error during plotting: {e}")
        messagebox.showerror("Plotting Error", f"Failed to create plot:\n{e}")


# --- GUI Application Class ---

class GeneratorApp:
    def __init__(self, root, cli_args):
        self.root = root
        self.root.title("Non-IID Federated Data Generator")
        # self.root.geometry("550x650") # Adjust size as needed

        # Use CLI args as defaults if provided
        self.defaults = cli_args

        # --- Data storage ---
        self.client_data = None
        self.last_output_dir = self.defaults.output_dir # Remember last used dir

        # --- Style ---
        style = ttk.Style()
        style.theme_use('clam') # Or 'alt', 'default', 'classic'

        # --- Variables ---
        self.vars = {}
        self._create_gui_variables()

        # --- GUI Layout ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        row_idx = 0

        # Base Parameters Frame
        base_frame = ttk.LabelFrame(main_frame, text="Base Dataset Parameters", padding="10")
        base_frame.grid(row=row_idx, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        base_frame.columnconfigure(1, weight=1)
        row_idx += 1

        self._add_entry(base_frame, "Total Samples:", "n_samples", 0)
        self._add_entry(base_frame, "Num Features:", "n_features", 1)
        self._add_entry(base_frame, "Num Classes:", "n_classes", 2)
        self._add_entry(base_frame, "Num Clients:", "n_clients", 3)
        self._add_entry(base_frame, "Class Separation:", "class_sep", 4)
        self._add_entry(base_frame, "Informative Fraction (0-1):", "info_frac", 5)

        # Skew Parameters Frame
        skew_frame = ttk.LabelFrame(main_frame, text="Skew Parameters", padding="10")
        skew_frame.grid(row=row_idx, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        skew_frame.columnconfigure(1, weight=1)
        row_idx += 1

        self._add_entry(skew_frame, "Quantity Skew Alpha (>0):", "quantity_skew_alpha", 0)
        self._add_entry(skew_frame, "Label Skew Alpha (>0):", "label_skew_alpha", 1)
        self._add_entry(skew_frame, "Feature Skew Level (0-1):", "feature_skew_level", 2)
        self._add_entry(skew_frame, "Concept Drift Level (0-1):", "concept_drift_level", 3)
        self._add_entry(skew_frame, "Concept Shift Level (0-1):", "concept_shift_level", 4)

        # Output & Visualization Frame
        out_vis_frame = ttk.LabelFrame(main_frame, text="Output & Visualization", padding="10")
        out_vis_frame.grid(row=row_idx, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        out_vis_frame.columnconfigure(1, weight=1)
        row_idx += 1

        self._add_entry(out_vis_frame, "Output Directory:", "output_dir", 0)
        self._add_entry(out_vis_frame, "Visualize Samples:", "vis_samples", 1)
        self._add_choice(out_vis_frame, "Visualize Method:", "vis_method", ['pca', 'tsne'], 2)

        # Action Buttons Frame
        action_frame = ttk.Frame(main_frame, padding="10")
        action_frame.grid(row=row_idx, column=0, columnspan=2, pady=10)
        row_idx += 1

        self.generate_button = ttk.Button(action_frame, text="Generate & Save Data", command=self._generate_data_callback)
        self.generate_button.grid(row=0, column=0, padx=5, pady=5)

        self.visualize_button = ttk.Button(action_frame, text="Visualize Current Data", command=self._visualize_data_callback, state=tk.DISABLED) # Disabled until data is loaded/generated
        self.visualize_button.grid(row=0, column=1, padx=5, pady=5)

        self.gen_vis_button = ttk.Button(action_frame, text="Generate, Save & Visualize", command=self._generate_and_visualize_callback)
        self.gen_vis_button.grid(row=1, column=0, padx=5, pady=5)

        self.load_button = ttk.Button(action_frame, text="Load Data for Visualization...", command=self._load_data_callback)
        self.load_button.grid(row=1, column=1, padx=5, pady=5)


    def get_default_args(self):
         # Use argparse to get default values defined in the CLI parser
         parser = self._create_parser()
         defaults = parser.parse_args([]) # Parse empty list to get defaults
         return defaults

    def _create_gui_variables(self):
        """Create Tkinter variables linked to default args."""
        defaults = self.defaults
        self.vars = {
            "n_samples": tk.IntVar(value=defaults.n_samples),
            "n_features": tk.IntVar(value=defaults.n_features),
            "n_classes": tk.IntVar(value=defaults.n_classes),
            "n_clients": tk.IntVar(value=defaults.n_clients),
            "class_sep": tk.DoubleVar(value=defaults.class_sep),
            "info_frac": tk.DoubleVar(value=defaults.info_frac),
            "quantity_skew_alpha": tk.DoubleVar(value=defaults.quantity_skew_alpha),
            "label_skew_alpha": tk.DoubleVar(value=defaults.label_skew_alpha),
            "feature_skew_level": tk.DoubleVar(value=defaults.feature_skew_level),
            "concept_drift_level": tk.DoubleVar(value=defaults.concept_drift_level),
            "concept_shift_level": tk.DoubleVar(value=defaults.concept_shift_level),
            "output_dir": tk.StringVar(value=defaults.output_dir),
            "vis_samples": tk.IntVar(value=defaults.vis_samples),
            "vis_method": tk.StringVar(value=defaults.vis_method),
        }

    def _add_entry(self, parent, label_text, var_name, row):
        """Helper to add a Label and Entry pair."""
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        entry = ttk.Entry(parent, textvariable=self.vars[var_name], width=20)
        entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

    def _add_choice(self, parent, label_text, var_name, choices, row):
        """Helper to add a Label and Combobox pair."""
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        combo = ttk.Combobox(parent, textvariable=self.vars[var_name], values=choices, state='readonly', width=18)
        combo.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

    def _get_params_from_gui(self, validate=True):
        """Reads parameters from GUI, performs type conversion and validation."""
        params = {}
        errors = []
        for name, tk_var in self.vars.items():
            try:
                val = tk_var.get()
                # Type conversion based on variable type
                if isinstance(tk_var, tk.IntVar):
                    params[name] = int(val)
                elif isinstance(tk_var, tk.DoubleVar):
                    params[name] = float(val)
                else: # StringVars
                    params[name] = str(val)

                # Validation (add more checks as needed)
                if validate:
                     if name in ["n_samples", "n_features", "n_classes", "n_clients"] and params[name] <= 0:
                         errors.append(f"{name.replace('_',' ').title()} must be > 0.")
                     if name in ["quantity_skew_alpha", "label_skew_alpha"] and params[name] <= 0:
                         errors.append(f"{name.replace('_',' ').title()} must be > 0.")
                     if name in ["feature_skew_level", "concept_drift_level", "concept_shift_level"] and not (0 <= params[name] <= 1):
                         errors.append(f"{name.replace('_',' ').title()} must be between 0 and 1.")
                     if name == "info_frac" and not (0 < params[name] <= 1):
                         errors.append("Informative Fraction must be between 0 (exclusive) and 1 (inclusive).")
                     if name == "output_dir" and not params[name]:
                         errors.append("Output Directory cannot be empty.")
                     if name == "vis_samples" and params[name] <= 0:
                          errors.append("Visualize Samples must be > 0.")


            except (tk.TclError, ValueError) as e:
                errors.append(f"Invalid value for {name.replace('_',' ').title()}: {e}")

        if errors:
            messagebox.showerror("Invalid Parameters", "Please correct the following errors:\n\n" + "\n".join(errors))
            return None

        # Return as a simple namespace object, similar to argparse
        return argparse.Namespace(**params)

    def _generate_data_callback(self):
        logging.info("Generate button clicked.")
        params = self._get_params_from_gui()
        if params is None:
            return # Validation failed

        logging.info("Starting generation from GUI parameters...")
        self.root.config(cursor="watch") # Indicate busy
        self.root.update_idletasks()

        self.client_data = generate_data(
            n_samples=params.n_samples,
            n_features=params.n_features,
            n_classes=params.n_classes,
            n_clients=params.n_clients,
            quantity_skew_alpha=params.quantity_skew_alpha,
            label_skew_alpha=params.label_skew_alpha,
            feature_skew_level=params.feature_skew_level,
            concept_drift_level=params.concept_drift_level,
            concept_shift_level=params.concept_shift_level,
            base_class_sep=params.class_sep,
            base_n_informative_frac=params.info_frac,
            output_dir=params.output_dir
        )

        self.root.config(cursor="") # Back to normal cursor
        self.root.update_idletasks()

        if self.client_data is not None:
            messagebox.showinfo("Generation Complete", f"Data generation finished. Data saved to '{params.output_dir}'.")
            self.visualize_button.config(state=tk.NORMAL) # Enable visualization
            self.last_output_dir = params.output_dir # Update last used dir
        else:
            # Error message shown by generate_data already
             self.visualize_button.config(state=tk.DISABLED)


    def _visualize_data_callback(self):
        logging.info("Visualize button clicked.")
        if self.client_data is None:
            messagebox.showwarning("Visualize", "No data available. Please generate or load data first.")
            return

        # Get only visualization params (no need to re-validate generation params)
        vis_params = self._get_params_from_gui(validate=False) # Quick way, ignore validation errors
        if vis_params is None: # Should ideally not happen if GUI prevents bad values
             vis_samples = self.defaults.vis_samples
             vis_method = self.defaults.vis_method
             logging.warning("Could not get vis params from GUI, using defaults.")
        else:
             vis_samples = vis_params.vis_samples
             vis_method = vis_params.vis_method
             # Validate just these two
             if vis_samples <= 0:
                  messagebox.showerror("Invalid Parameter", "Visualize Samples must be > 0.")
                  return


        logging.info(f"Starting visualization with {vis_samples} samples using {vis_method}...")
        self.root.config(cursor="watch")
        self.root.update_idletasks()

        # Create a title reflecting the *current* state (could be loaded data)
        # Try to infer some params from the data itself if possible
        n_clients_actual = len(self.client_data)
        try:
             n_classes_actual = len(np.unique(np.concatenate([d['y'] for d in self.client_data.values() if d['y'].shape[0]>0])))
             title = f"Current Data (Clients={n_clients_actual}, Classes={n_classes_actual})"
        except: # Fallback if concat fails
             title = f"Current Data (Clients={n_clients_actual})"


        visualize_data(
             self.client_data,
             n_samples_to_plot=vis_samples,
             method=vis_method,
             title=title
        )

        self.root.config(cursor="")
        self.root.update_idletasks()
        logging.info("Visualization complete.")


    def _generate_and_visualize_callback(self):
        logging.info("Generate & Visualize button clicked.")
        # First, generate the data
        self._generate_data_callback()
        # If generation was successful (client_data is not None), visualize it
        if self.client_data is not None:
            self._visualize_data_callback()

    def _load_data_callback(self):
        logging.info("Load button clicked.")
        initial_dir = self.last_output_dir if os.path.isdir(self.last_output_dir) else os.getcwd()
        filepath = filedialog.askopenfilename(
            title="Select Federated Dataset File",
            initialdir=initial_dir,
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if not filepath:
            logging.info("File loading cancelled.")
            return

        logging.info(f"Attempting to load data from: {filepath}")
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)

            # Basic validation of loaded structure
            if not isinstance(loaded_data, dict):
                raise TypeError("Loaded file is not a dictionary (expected client data).")
            # Check if keys are integers (client IDs) and values are dicts with 'X', 'y'
            valid = True
            for k, v in loaded_data.items():
                if not isinstance(k, int) or not isinstance(v, dict) or 'X' not in v or 'y' not in v:
                    valid = False
                    break
            if not valid:
                raise TypeError("Loaded dictionary does not seem to contain valid client data structure.")

            self.client_data = loaded_data
            self.last_output_dir = os.path.dirname(filepath) # Update last dir
            logging.info("Data loaded successfully.")
            messagebox.showinfo("Load Complete", f"Successfully loaded data from\n{filepath}")
            self.visualize_button.config(state=tk.NORMAL) # Enable visualization
            # Optional: Update GUI fields based on loaded data? (Complex)

        except FileNotFoundError:
            logging.error(f"File not found: {filepath}")
            messagebox.showerror("Load Error", f"File not found:\n{filepath}")
            self.client_data = None
            self.visualize_button.config(state=tk.DISABLED)
        except pickle.UnpicklingError:
            logging.error(f"Error unpickling file: {filepath}. File might be corrupted or not a pickle file.")
            messagebox.showerror("Load Error", "Could not read the file. It might be corrupted or not a valid dataset file.")
            self.client_data = None
            self.visualize_button.config(state=tk.DISABLED)
        except TypeError as te:
            logging.error(f"Error loading data: {te}")
            messagebox.showerror("Load Error", f"Loaded data structure is invalid:\n{te}")
            self.client_data = None
            self.visualize_button.config(state=tk.DISABLED)
        except Exception as e:
            logging.exception(f"An unexpected error occurred during file loading: {e}")
            messagebox.showerror("Load Error", f"An unexpected error occurred:\n{e}")
            self.client_data = None
            self.visualize_button.config(state=tk.DISABLED)
        finally:
            self.root.config(cursor="")
            self.root.update_idletasks()


    # Need to recreate parser within the class context to access its definition
    def _create_parser(self):
         parser = argparse.ArgumentParser(description="Generate Synthetic Non-IID Federated Datasets (Interactive GUI support)")
         # Base Dataset Parameters
         parser.add_argument('--n_samples', type=int, default=10000, help='Total number of samples')
         parser.add_argument('--n_features', type=int, default=20, help='Number of features')
         parser.add_argument('--n_classes', type=int, default=10, help='Number of classes')
         parser.add_argument('--n_clients', type=int, default=100, help='Number of clients')
         parser.add_argument('--class_sep', type=float, default=1.0, help='Class separation factor')
         parser.add_argument('--info_frac', type=float, default=0.8, help='Fraction of informative features (0 to 1)')
         # Skew Control Parameters
         parser.add_argument('--quantity_skew_alpha', type=float, default=1.0, help='Dirichlet alpha for quantity skew (>0)')
         parser.add_argument('--label_skew_alpha', type=float, default=1.0, help='Dirichlet alpha for label skew (>0)')
         parser.add_argument('--feature_skew_level', type=float, default=0.0, help='Level of feature skew (0 to 1)')
         parser.add_argument('--concept_drift_level', type=float, default=0.0, help='Level of concept drift (0 to 1)')
         parser.add_argument('--concept_shift_level', type=float, default=0.0, help='Level of concept shift (label flip prob, 0 to 1)')
         # Output & Visualization
         parser.add_argument('--output_dir', type=str, default='federated_data', help='Directory to save the dataset')
         parser.add_argument('--visualize', action='store_true', help='Visualize after generation (CLI only)')
         parser.add_argument('--vis_samples', type=int, default=2000, help='Max samples for visualization')
         parser.add_argument('--vis_method', type=str, default='pca', choices=['pca', 'tsne'], help='Visualization method')
         # GUI flag
         parser.add_argument('--gui', action='store_true', help='Launch the interactive GUI instead of CLI execution')
         return parser

    def run(self):
        self.root.mainloop()


# --- Main Execution & CLI Parsing (Modified) ---

def parse_args():
    # Use the parser defined within the app class or a shared definition
    app_instance_for_parser = GeneratorApp(tk.Tk()) # Create dummy instance for parser
    parser = app_instance_for_parser._create_parser()
    app_instance_for_parser.root.destroy() # Destroy the dummy window
    return parser.parse_args()

def main():
    args = parse_args()

    if args.gui:
        # --- Run GUI ---
        # Set Matplotlib backend *before* creating main Tk window if possible
        try:
            matplotlib.use('TkAgg')
            logging.info("Set Matplotlib backend to TkAgg for GUI.")
        except ImportError:
             logging.warning("Could not set Matplotlib backend to TkAgg. Visualization might fail or open in a separate process.")

        root = tk.Tk()
        app = GeneratorApp(root, cli_args=args) # Pass CLI args to potentially override GUI defaults
        app.run()
    else:
        # --- Run CLI ---
        logging.info("Running in Command-Line Interface (CLI) mode.")
        # Validate parameters (basic checks, more detailed within functions)
        try:
            if args.quantity_skew_alpha <= 0: raise ValueError("quantity_skew_alpha must be positive.")
            if args.label_skew_alpha <= 0: raise ValueError("label_skew_alpha must be positive.")
            if not (0 <= args.feature_skew_level <= 1): raise ValueError("feature_skew_level must be between 0 and 1.")
            if not (0 <= args.concept_drift_level <= 1): raise ValueError("concept_drift_level must be between 0 and 1.")
            if not (0 <= args.concept_shift_level <= 1): raise ValueError("concept_shift_level must be between 0 and 1.")
            if not (0 < args.info_frac <= 1): raise ValueError("info_frac must be > 0 and <= 1.")
            if args.n_features < 1 or args.n_samples < 1 or args.n_classes < 1 or args.n_clients < 1: raise ValueError("samples, features, classes, clients must be >= 1")
            if args.n_features < 2 and args.visualize:
                 logging.warning("Visualization might be trivial with n_features < 2.")
                 if args.vis_method == 'tsne':
                     logging.warning("t-SNE requires n_features >= 2. Defaulting visualization method to PCA or direct plot.")
                     args.vis_method = 'pca' # Fallback

        except ValueError as e:
             logging.error(f"Parameter Error: {e}")
             sys.exit(1) # Exit if CLI params are invalid

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

        if args.visualize and client_data is not None:
            # Create title for CLI visualization
            title = f"Non-IID Syn Data (CLI) | Clients={args.n_clients}, Classes={args.n_classes}\n"
            title += f" QSkew(α={args.quantity_skew_alpha}), LSkew(α={args.label_skew_alpha})\n"
            title += f" FeatSkew(lvl={args.feature_skew_level}), Drift(lvl={args.concept_drift_level}), Shift(lvl={args.concept_shift_level})"

            visualize_data(client_data, n_samples_to_plot=args.vis_samples, method=args.vis_method, title=title)
        elif args.visualize and client_data is None:
             logging.warning("Visualization requested but data generation failed.")


if __name__ == "__main__":
    # CLI Examples (same as before):
    # Basic IID-like (high alpha, low levels)
    # python non_iid_generator_interactive.py --n_samples 5000 --n_features 10 --n_classes 5 --n_clients 10 --quantity_skew_alpha 100 --label_skew_alpha 100 --visualize --vis_samples 1000

    # Strong Label Skew + Moderate Quantity Skew
    # python non_iid_generator_interactive.py --n_samples 10000 --n_features 20 --n_classes 10 --n_clients 50 --quantity_skew_alpha 0.5 --label_skew_alpha 0.2 --visualize --vis_samples 2000 --vis_method tsne

    # Feature Skew + Concept Drift
    # python non_iid_generator_interactive.py --n_samples 10000 --n_features 20 --n_classes 10 --n_clients 50 --quantity_skew_alpha 5.0 --label_skew_alpha 5.0 --feature_skew_level 0.8 --concept_drift_level 0.5 --visualize

    # Concept Shift
    # python non_iid_generator_interactive.py --n_samples 5000 --n_features 10 --n_classes 3 --n_clients 20 --concept_shift_level 0.7 --visualize

    # --- To run the GUI ---
    # python non_iid_generator_interactive.py --gui
    # Or with some overridden defaults for the GUI:
    # python non_iid_generator_interactive.py --gui --n_clients 20 --n_classes 5

    main()