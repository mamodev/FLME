"""Core logic for generating the synthetic federated dataset."""

import numpy as np
import logging
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Import skew functions from the dedicated module
from skew_functions import (
    apply_quantity_skew,
    apply_label_skew,
    apply_feature_skew,
    apply_concept_drift,
    apply_concept_shift
)
# Import utils
from data_utils import save_data, check_data_shapes


def generate_base_data(n_samples, n_features, n_classes, class_sep, n_informative_frac):
    """
    Generates the initial IID dataset using scikit-learn's make_classification.

    Args:
        n_samples (int): Total number of samples.
        n_features (int): Total number of features.
        n_classes (int): Number of classes.
        class_sep (float): Controls the separation between classes. Higher values
                           make classification easier.
        n_informative_frac (float): Fraction of features that are informative.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): Feature matrix (n_samples, n_features), scaled.
            - y (np.ndarray): Label vector (n_samples,).
        Returns (None, None) if parameters are invalid or generation fails.
    """
    if not (0 < n_informative_frac <= 1):
        logging.error("Informative fraction must be > 0 and <= 1.")
        return None, None
    if n_samples <= 0 or n_features <= 0 or n_classes <= 0:
         logging.error("Samples, features, and classes must be positive.")
         return None, None
    if n_classes > 2**n_features: # make_classification constraint approx
         logging.warning(f"n_classes ({n_classes}) might be too high for n_features ({n_features}).")
         # Consider adjusting or raising error? Let make_classification handle it for now.


    n_informative = max(1, int(n_features * n_informative_frac))
    # Simplified feature allocation for make_classification:
    n_redundant = max(0, n_features - n_informative)
    n_repeated = 0
    n_clusters_per_class = 1

    logging.info(f"Generating base dataset with make_classification: "
                 f"n_samples={n_samples}, n_features={n_features}, n_classes={n_classes}, "
                 f"n_informative={n_informative}, n_redundant={n_redundant}, "
                 f"class_sep={class_sep}")

    try:
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

        # Check if generated data is valid
        if X is None or y is None or X.shape[0] != n_samples or y.shape[0] != n_samples:
             logging.error("make_classification did not return expected data.")
             return None, None

        # Standard scale features
        if X.shape[0] > 1: # Avoid scaling if only one sample
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif X.shape[0] == 1:
             logging.warning("Only one sample generated, skipping standard scaling.")


        logging.info(f"Generated base data: X shape={X.shape}, y shape={y.shape}")
        return X, y

    except Exception as e:
        logging.exception(f"Error during make_classification: {e}")
        return None, None


def generate_federated_dataset(
    n_samples, n_features, n_classes, n_clients,
    quantity_skew_alpha, label_skew_alpha,
    feature_skew_level, concept_drift_level, concept_shift_level,
    base_class_sep=1.0, base_n_informative_frac=0.8,
    output_dir='federated_data', save_to_file=True):
    """
    Orchestrates the generation of the non-IID federated dataset.

    This function generates base data, applies specified skews, and optionally
    saves the resulting client data dictionary to a file.

    Args:
        n_samples (int): Total number of samples for the base dataset.
        n_features (int): Number of features.
        n_classes (int): Number of classes.
        n_clients (int): Number of clients to partition data among.
        quantity_skew_alpha (float): Dirichlet alpha for quantity skew (>0).
        label_skew_alpha (float): Dirichlet alpha for label skew (>0).
        feature_skew_level (float): Level of feature skew (0 to 1).
        concept_drift_level (float): Level of concept drift (0 to 1).
        concept_shift_level (float): Level of concept shift (label flip prob, 0 to 1).
        base_class_sep (float): Class separation for the base dataset. Defaults to 1.0.
        base_n_informative_frac (float): Fraction of informative features. Defaults to 0.8.
        output_dir (str): Directory to save the dataset if save_to_file is True.
                          Defaults to 'federated_data'.
        save_to_file (bool): Whether to save the generated data to a pickle file.
                             Defaults to True.

    Returns:
        tuple: A tuple containing:
            - client_data (dict or None): The generated federated dataset dictionary,
              or None if generation failed at any critical step.
            - success (bool): True if the generation process completed (even if saving failed),
              False if a critical error occurred during generation.
            - saved_filepath (str or None): Path to the saved file if save_to_file is True
              and saving was successful, None otherwise.
    """
    logging.info("Starting federated dataset generation...")
    client_data = None
    saved_filepath = None
    success = False # Assume failure until completion

    try:
        # --- Parameter Validation (Basic checks, more detailed in sub-functions) ---
        if n_samples <= 0 or n_features <= 0 or n_classes <= 0 or n_clients < 0:
             raise ValueError("Samples, features, classes must be positive. Clients must be non-negative.")
        # Alpha/Level checks happen within their respective apply_* functions

        # 1. Generate Base Global Data
        X_global, y_global = generate_base_data(n_samples, n_features, n_classes,
                                                 base_class_sep, base_n_informative_frac)
        if X_global is None or y_global is None:
            logging.error("Failed to generate base data.")
            return None, False, None # Critical failure

        # Handle edge case of n_clients = 0
        if n_clients == 0:
            logging.warning("n_clients is 0. Returning empty client_data.")
            client_data = {}
            success = True # Generation technically "succeeded" by producing an empty dict
            # Skip saving if requested for an empty dataset
            if save_to_file:
                 saved, saved_filepath = save_data(client_data, output_dir)
                 if not saved:
                     logging.warning("Saving failed for the empty dataset structure.") # Non-critical
            return client_data, success, saved_filepath


        # 2. Apply Quantity Skew
        # Handle n_samples=0 case inside apply_quantity_skew
        client_sample_counts = apply_quantity_skew(n_samples, n_clients, quantity_skew_alpha)
        if client_sample_counts is None:
            logging.error("Failed to apply quantity skew.")
            # Don't consider this critical? Could proceed with equal distribution maybe?
            # Let's treat it as failure for now.
            return None, False, None

        # Check if total samples assigned match expected (can be 0 if n_samples=0)
        if sum(client_sample_counts) != n_samples:
             logging.warning(f"Sum of client sample counts ({sum(client_sample_counts)}) "
                             f"does not match n_samples ({n_samples}). Proceeding with assigned counts.")


        # 3. Apply Label Skew (primary data partitioning)
        # Handle n_samples=0 case inside apply_label_skew
        client_data = apply_label_skew(X_global, y_global, client_sample_counts,
                                         n_classes, label_skew_alpha)
        if client_data is None:
             logging.error("Failed to apply label skew and partition data.")
             return None, False, None # Critical partitioning failure


        # --- Apply skews that modify data *within* clients ---
        # These are less critical; if they fail, we might still have usable partitioned data.
        # However, for consistency, let's treat them as part of the required process for now.

        # 4. Apply Feature Skew
        client_data = apply_feature_skew(client_data, feature_skew_level, n_features)
        # apply_feature_skew currently returns the modified dict or logs issues, doesn't return None on failure

        # 5. Apply Concept Drift
        client_data = apply_concept_drift(client_data, concept_drift_level)
        # apply_concept_drift also returns modified dict

        # 6. Apply Concept Shift
        client_data = apply_concept_shift(client_data, concept_shift_level, n_classes)
        # apply_concept_shift also returns modified dict

        # Mark generation as successful at this point
        success = True

        # Final check and save
        total_samples_final = check_data_shapes(client_data)
        if save_to_file:
            if total_samples_final > 0 or n_samples == 0: # Save if data exists or if n_samples was 0 initially (save empty structure)
                saved, saved_filepath = save_data(client_data, output_dir)
                if not saved:
                    logging.error("Failed to save the generated data.")
                    # Return success=True because generation finished, but no saved path
                    saved_filepath = None
            elif total_samples_final == 0 and n_samples > 0:
                logging.warning("Generation resulted in 0 total samples across clients (target > 0). Skipping save.")
                saved_filepath = None # Explicitly no path
            # else: n_samples was 0, handled above

        logging.info("Federated dataset generation process complete.")

    except ValueError as ve:
        logging.error(f"Data Generation Error: Invalid parameter - {ve}")
        return None, False, None # Parameter error is critical failure
    except Exception as e:
        logging.exception(f"An unexpected error occurred during data generation: {e}")
        return None, False, None # Unexpected error is critical failure

    return client_data, success, saved_filepath