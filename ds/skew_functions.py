"""Functions to apply various types of non-IID skew to datasets."""

import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

def apply_quantity_skew(n_samples, n_clients, alpha):
    """
    Generates sample counts per client based on a Dirichlet distribution.

    A smaller alpha results in higher skew (some clients having many more samples
    than others). An alpha approaching infinity results in an IID quantity distribution.

    Args:
        n_samples (int): The total number of samples to distribute.
        n_clients (int): The number of clients.
        alpha (float): The concentration parameter for the Dirichlet distribution.
                       Must be positive.

    Returns:
        list[int]: A list containing the number of samples assigned to each client.
                   Returns None if alpha is invalid or distribution fails.

    Raises:
        ValueError: If alpha is not positive.
    """
    if alpha <= 0:
        logging.error("Dirichlet alpha for quantity skew must be positive.")
        raise ValueError("Dirichlet alpha must be positive.")
    if n_clients <= 0:
        logging.warning("Number of clients is zero or negative, returning empty distribution.")
        return []
    if n_samples <= 0:
         logging.warning("Total number of samples is zero or negative, assigning 0 samples to all clients.")
         return [0] * n_clients

    # Ensure concentration parameter is a list/array of size n_clients
    concentration = np.full(n_clients, alpha)
    proportions = np.random.dirichlet(concentration)
    proportions /= proportions.sum() # Ensure sum to 1 robustness

    # Distribute samples, ensuring total is correct
    client_sample_counts = np.zeros(n_clients, dtype=int)
    remaining_samples = n_samples
    for i in range(n_clients - 1):
        samples = int(round(proportions[i] * n_samples))
        samples = min(samples, remaining_samples) # Don't assign more than available
        samples = max(0, samples) # Ensure non-negative
        client_sample_counts[i] = samples
        remaining_samples -= samples
    client_sample_counts[n_clients - 1] = remaining_samples # Assign rest to last client

    # Correction loop for rounding errors
    actual_total = client_sample_counts.sum()
    diff = n_samples - actual_total
    if diff != 0:
        logging.warning(f"Correcting sample count mismatch ({diff}) due to rounding in quantity skew.")
        adjust_clients = np.arange(n_clients) # Indices of clients to adjust
        np.random.shuffle(adjust_clients) # Adjust randomly
        adjust_idx = 0
        max_adjust_loops = 2 * abs(diff) + n_clients # Safety break

        while diff != 0 and adjust_idx < max_adjust_loops :
            client_to_adjust = adjust_clients[adjust_idx % n_clients]
            if diff > 0:
                client_sample_counts[client_to_adjust] += 1
                diff -= 1
            elif client_sample_counts[client_to_adjust] > 0: # Only decrement if > 0
                client_sample_counts[client_to_adjust] -= 1
                diff += 1
            else:
                # Skip client if it's already 0 and we need to decrease
                pass
            adjust_idx += 1

        if diff != 0: # Check if correction failed
             logging.error(f"Could not fully correct quantity skew sample count mismatch. Remaining difference: {diff}")
             # This indicates a larger issue, maybe return None or raise error? For now, log and proceed.


    # Final check
    if client_sample_counts.sum() != n_samples:
        logging.error(f"FINAL Sample count mismatch after correction. Target: {n_samples}, Actual: {client_sample_counts.sum()}")
        # Decide on behavior: error out or proceed with imperfect counts? Proceed for now.

    logging.info(f"Applied Quantity Skew (alpha={alpha}). Client sample counts sum: {client_sample_counts.sum()}")
    logging.debug(f"Client sample counts: {client_sample_counts.tolist()}")
    return client_sample_counts.tolist()


def apply_label_skew(X_global, y_global, client_sample_counts, n_classes, alpha):
    """
    Partitions data based on client sample counts and Dirichlet-sampled label distributions.

    Args:
        X_global (np.ndarray): Global feature data (n_samples, n_features).
        y_global (np.ndarray): Global label data (n_samples,).
        client_sample_counts (list[int]): Number of samples per client.
        n_classes (int): Total number of unique classes in y_global.
        alpha (float): Dirichlet concentration parameter for label distribution skew.
                       Must be positive.

    Returns:
        dict: A dictionary where keys are client IDs and values are dictionaries
              containing the client's assigned 'X' and 'y' numpy arrays.
              Returns None if alpha is invalid or partitioning fails.

    Raises:
        ValueError: If alpha is not positive.
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
             client_data[i] = {'X': np.empty((0, n_features)), 'y': np.empty((0,), dtype=y_global.dtype)}
        return client_data
    elif n_samples_total == 0:
         logging.warning("Total samples is 0, applying label skew results in empty clients.")
         for i in range(n_clients):
             client_data[i] = {'X': np.empty((0, n_features)), 'y': np.empty((0,), dtype=y_global.dtype)}
         return client_data


    assigned_indices = set()

    # Target label distribution per client
    client_label_proportions = np.random.dirichlet(np.full(n_classes, alpha), size=n_clients)

    logging.info(f"Target label distributions (proportions) per client (alpha={alpha}):")
    # Calculate target counts per class per client precisely
    target_counts_per_class_client = np.zeros((n_clients, n_classes), dtype=int)
    total_target_counts_per_class = np.zeros(n_classes, dtype=int)

    for i in range(n_clients):
        if client_sample_counts[i] == 0:
            logging.debug(f"  Client {i}: Target samples = 0. Skipping distribution.")
            continue

        proportions = client_label_proportions[i]
        proportions /= proportions.sum() # Ensure sum = 1

        # Use multinomial distribution for assignment based on proportions
        try:
            counts = np.random.multinomial(client_sample_counts[i], proportions)
        except ValueError as e:
            # Fallback (should be rare if sum=1)
            logging.warning(f"Multinomial sampling failed for client {i} (proportions sum: {proportions.sum()}): {e}. Falling back to rounding.")
            counts = np.zeros(n_classes, dtype=int)
            remaining = client_sample_counts[i]
            for k in range(n_classes - 1):
                c = min(remaining, int(round(proportions[k] * client_sample_counts[i])))
                c = max(0, c)
                counts[k] = c
                remaining -= c
            counts[n_classes - 1] = remaining # Assign remainder

        # Adjust sum if multinomial/rounding slightly off target N_i
        count_sum_diff = client_sample_counts[i] - counts.sum()
        if count_sum_diff != 0:
            active_classes = np.arange(n_classes)
            np.random.shuffle(active_classes)
            adjust_idx = 0
            max_adjust_loops = 2 * abs(count_sum_diff) + n_classes
            while count_sum_diff != 0 and adjust_idx < max_adjust_loops:
                class_to_adjust = active_classes[adjust_idx % n_classes]
                if count_sum_diff > 0:
                    counts[class_to_adjust] += 1
                    count_sum_diff -= 1
                elif counts[class_to_adjust] > 0:
                    counts[class_to_adjust] -= 1
                    count_sum_diff += 1
                adjust_idx += 1
            if count_sum_diff != 0:
                 logging.error(f"Client {i}: Could not correct target class counts sum. Diff: {count_sum_diff}")

        target_counts_per_class_client[i] = np.maximum(0, counts) # Ensure non-negative
        total_target_counts_per_class += target_counts_per_class_client[i]
        logging.debug(f"  Client {i} (Target Total {client_sample_counts[i]}): Target counts={target_counts_per_class_client[i].tolist()}")


    # Check if total target counts exceed availability (can happen with extreme skew/few samples)
    for k in range(n_classes):
        if total_target_counts_per_class[k] > available_indices_count[k]:
             logging.warning(f"Total target count for class {k} ({total_target_counts_per_class[k]}) "
                             f"exceeds available samples ({available_indices_count[k]}). "
                             f"Allocation might not perfectly match targets.")

    # Distribute actual indices
    available_class_indices_sets = {k: set(inds) for k, inds in class_indices.items()}
    client_indices = {i: [] for i in range(n_clients)}

    # Iterate through classes and assign samples to clients needing them
    indices_assigned_this_round = 0
    for k in range(n_classes):
        available_for_k = list(available_class_indices_sets[k])
        np.random.shuffle(available_for_k)
        available_idx_ptr = 0

        clients_needing_k = [i for i in range(n_clients) if target_counts_per_class_client[i][k] > 0]
        # Optional: Sort clients by need? np.random.shuffle(clients_needing_k) is fair
        np.random.shuffle(clients_needing_k)

        for i in clients_needing_k:
            needed = target_counts_per_class_client[i][k]
            can_get = len(available_for_k) - available_idx_ptr
            n_take = min(needed, can_get)

            if n_take > 0:
                start = available_idx_ptr
                end = available_idx_ptr + n_take
                chosen_indices = available_for_k[start:end]

                client_indices[i].extend(chosen_indices)
                assigned_indices.update(chosen_indices)
                available_class_indices_sets[k].difference_update(chosen_indices) # Remove assigned
                available_idx_ptr += n_take
                target_counts_per_class_client[i][k] -= n_take # Reduce need
                indices_assigned_this_round += n_take

            if target_counts_per_class_client[i][k] > 0 and can_get <= 0:
                 logging.warning(f"Client {i}, Class {k}: Could not get {target_counts_per_class_client[i][k]} required samples as none were left for this class.")

    logging.info(f"Assigned {indices_assigned_this_round} samples based on initial label skew targets.")

    # Handle remaining client sample count shortfalls (due to class availability issues)
    unassigned_overall = list(set(data_indices) - assigned_indices)
    np.random.shuffle(unassigned_overall)
    unassigned_ptr = 0
    logging.info(f"{len(unassigned_overall)} samples remain unassigned globally after primary allocation.")

    for i in range(n_clients):
         current_assigned_count = len(client_indices[i])
         total_shortfall = client_sample_counts[i] - current_assigned_count
         if total_shortfall > 0:
             logging.warning(f"Client {i}: Shortfall of {total_shortfall} samples (Target: {client_sample_counts[i]}, Assigned: {current_assigned_count}). Trying to fill from unassigned pool.")
             n_fill = min(total_shortfall, len(unassigned_overall) - unassigned_ptr)
             if n_fill > 0:
                 start = unassigned_ptr
                 end = unassigned_ptr + n_fill
                 fill_indices = unassigned_overall[start:end]
                 client_indices[i].extend(fill_indices)
                 assigned_indices.update(fill_indices) # Keep track globally too
                 unassigned_ptr += n_fill
                 logging.info(f"Client {i}: Filled {n_fill} samples from unassigned pool.")
             else:
                 logging.error(f"Client {i}: Could not fill shortfall of {total_shortfall}, no unassigned samples left or pool exhausted.")


    # Create final client data dictionary
    final_client_data = {}
    final_total_samples = 0
    for i in range(n_clients):
        indices = np.array(client_indices[i], dtype=int)
        # Ensure indices are valid before slicing (should be, but safety check)
        valid_indices = indices[(indices >= 0) & (indices < n_samples_total)]
        if len(valid_indices) != len(indices):
            logging.error(f"Client {i}: Invalid indices detected ({len(indices) - len(valid_indices)} out of bounds). Clipping.")
            indices = valid_indices

        if len(indices) > 0:
            final_client_data[i] = {
                'X': X_global[indices],
                'y': y_global[indices]
            }
            final_total_samples += len(indices)
        else:
             # Handle clients originally assigned 0 samples or ending up with 0
             final_client_data[i] = {
                'X': np.empty((0, n_features)),
                'y': np.empty((0,), dtype=y_global.dtype)
            }
             if client_sample_counts[i] > 0:
                 logging.warning(f"Client {i} was targeted for {client_sample_counts[i]} samples but ended up with 0.")

    logging.info(f"Applied Label Skew (alpha={alpha}). Data partitioned into {len(final_client_data)} clients with {final_total_samples} total samples.")
    return final_client_data


def apply_feature_skew(client_data, level, n_features):
    """
    Applies client-specific feature transformations (shift + scale).

    The transformation is X_new = X_old * scale_factors + shifts, where scale
    and shift are randomly generated per client based on the skew level.

    Args:
        client_data (dict): The dictionary of client data. The 'X' arrays
                           will be modified in-place (or replaced).
        level (float): The level of feature skew, typically between 0 (no skew)
                       and 1 (higher skew). Values outside [0, 1] might have
                       unpredictable effects but are clipped internally to >= 0.
        n_features (int): The number of features in the dataset.

    Returns:
        dict: The modified client_data dictionary.
    """
    level = max(0, level) # Ensure non-negative level
    if level == 0:
        logging.info("Feature Skew level is 0. No transformation applied.")
        return client_data

    logging.info(f"Applying Feature Skew (level={level})...")
    rng = np.random.RandomState(42) # Seeded RNG for reproducibility

    for client_id, data in client_data.items():
        if 'X' not in data or data['X'].shape[0] == 0:
            logging.debug(f"Client {client_id}: Skipping feature skew (no data).")
            continue

        # Scale magnitude based on level
        # Define max variation around 1 for scale, and std dev for shift
        scale_max_variation = level * 0.5 # e.g., level 1 -> range [0.5, 1.5] approx
        shift_std_dev = level * 1.0 # e.g., level 1 -> shifts ~ N(0, 1)

        # Generate random scale factors (per feature) and shifts (per feature)
        # Ensure scale factors are positive and reasonably bounded
        min_scale = max(0.01, 1.0 - scale_max_variation)
        max_scale = 1.0 + scale_max_variation
        scale_factors = rng.uniform(min_scale, max_scale, size=n_features)
        shifts = rng.normal(0, shift_std_dev, size=n_features)

        # Apply transformation: X_new = X * scale_factors + shifts
        # Ensure modification happens correctly (replace the array)
        data['X'] = data['X'] * scale_factors + shifts
        logging.debug(f"  Client {client_id}: Applied shift/scale transformation.")

    return client_data


def apply_concept_drift(client_data, level):
    """
    Simulates concept drift by shifting the mean of features differently per class per client.

    Args:
        client_data (dict): The dictionary of client data. The 'X' arrays
                           will be modified in-place (or replaced).
        level (float): The level of concept drift, typically between 0 (no drift)
                       and 1 (higher drift). Values outside [0, 1] are clipped
                       internally to >= 0.

    Returns:
        dict: The modified client_data dictionary.
    """
    level = max(0, level) # Ensure non-negative level
    if level == 0:
        logging.info("Concept Drift level is 0. No drift applied.")
        return client_data

    logging.info(f"Applying Concept Drift (level={level})...")
    rng = np.random.RandomState(43) # Seeded RNG

    # Check if there's any data at all
    first_client_with_data = next((d for d in client_data.values() if d.get('X', np.empty((0,0))).shape[0] > 0), None)
    if first_client_with_data is None:
        logging.warning("No data found in any client. Cannot apply concept drift.")
        return client_data

    n_features = first_client_with_data['X'].shape[1]
    if n_features == 0:
        logging.warning("Data has 0 features. Cannot apply concept drift.")
        return client_data

    # Drift magnitude scale based on level
    drift_std_dev = level * 0.5 # Std Dev of the drift vector components per class/client

    for client_id, data in client_data.items():
        if data.get('X', np.empty((0,0))).shape[0] == 0:
            logging.debug(f"Client {client_id}: Skipping concept drift (no data).")
            continue

        X_i, y_i = data['X'], data['y']
        X_i_new = X_i.copy() # Work on a copy
        client_classes = np.unique(y_i)

        # Generate client-specific drift vectors for each class it possesses
        client_drifts = {
            k: rng.normal(0, drift_std_dev, size=n_features)
            for k in client_classes
        }

        for k in client_classes:
            class_mask = (y_i == k)
            if np.any(class_mask): # Check if class actually exists in this client
                # Apply the drift vector: X_new = X_old + drift_vector
                X_i_new[class_mask] = X_i[class_mask] + client_drifts[k]
                logging.debug(f"  Client {client_id}, Class {k}: Applied concept drift shift.")

        data['X'] = X_i_new # Replace original data with drifted data

    return client_data


def apply_concept_shift(client_data, level, n_classes):
    """
    Simulates concept shift by randomly flipping labels for a fraction of data points per client.

    Args:
        client_data (dict): The dictionary of client data. The 'y' arrays
                           will be modified in-place (or replaced).
        level (float): The probability (0 to 1) that a given sample's label
                       will be flipped to a different random class.
        n_classes (int): The total number of classes in the dataset.

    Returns:
        dict: The modified client_data dictionary.
    """
    if not (0 <= level <= 1):
         logging.warning(f"Concept shift level ({level}) out of bounds [0, 1]. Clipping.")
         level = max(0, min(1, level))

    if level == 0:
        logging.info("Concept Shift level is 0. No label flipping applied.")
        return client_data
    if n_classes <= 1:
        logging.warning("Concept Shift requires n_classes > 1. No labels flipped.")
        return client_data


    logging.info(f"Applying Concept Shift (label flipping probability={level:.3f})...")
    rng = np.random.RandomState(44) # Seeded RNG

    flip_prob = level
    total_flipped = 0
    total_samples_processed = 0

    for client_id, data in client_data.items():
        n_samples = data.get('y', np.empty((0,))).shape[0]
        total_samples_processed += n_samples
        if n_samples == 0:
            logging.debug(f"Client {client_id}: Skipping concept shift (no data).")
            continue

        y_i_original = data['y']
        y_i_new = y_i_original.copy() # Work on a copy

        # Determine which samples to flip based on probability
        flip_mask = rng.rand(n_samples) < flip_prob
        indices_to_flip = np.where(flip_mask)[0]
        n_to_flip = len(indices_to_flip)
        total_flipped += n_to_flip

        if n_to_flip > 0:
             logging.debug(f"  Client {client_id}: Flipping labels for {n_to_flip} samples.")

             all_possible_labels = list(range(n_classes))

             for idx in indices_to_flip:
                 original_label = y_i_original[idx]
                 # Generate possible new labels excluding the original one
                 possible_new_labels = [l for l in all_possible_labels if l != original_label]

                 if not possible_new_labels:
                      # This should only happen if n_classes=1, already checked above
                      logging.warning(f"Client {client_id}, Sample {idx}: No alternative labels available to flip to (original={original_label}). Skipping flip.")
                      continue

                 # Choose a new label randomly from the alternatives
                 new_label = rng.choice(possible_new_labels)
                 y_i_new[idx] = new_label

             data['y'] = y_i_new # Replace original labels with potentially flipped ones

    if total_samples_processed > 0:
        effective_flip_rate = (total_flipped / total_samples_processed) * 100
        logging.info(f"Concept Shift complete. Flipped {total_flipped} labels out of {total_samples_processed} "
                     f"processed samples ({effective_flip_rate:.2f}% effective flip rate).")
    else:
        logging.info("Concept Shift complete. No samples processed.")

    return client_data