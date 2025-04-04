"""Utility functions for handling federated dataset files."""

import os
import pickle
import logging
import numpy as np

def save_data(client_data, output_dir, filename="federated_dataset.pkl"):
    """
    Saves the partitioned client data to a pickle file.

    Args:
        client_data (dict): A dictionary where keys are client IDs and values
                           are dictionaries containing 'X' and 'y' numpy arrays.
        output_dir (str): The directory path where the file should be saved.
        filename (str): The name of the output pickle file.

    Returns:
        bool: True if saving was successful, False otherwise.
        str: The full path to the saved file if successful, None otherwise.
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Failed to create output directory {output_dir}: {e}")
            return False, None

    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(client_data, f)
        logging.info(f"Dataset saved successfully to {output_path}")
        return True, output_path
    except Exception as e:
        logging.error(f"Failed to save dataset to {output_path}: {e}")
        return False, None

def check_data_shapes(client_data):
    """
    Logs the shapes of data for each client and calculates the total samples.

    Args:
        client_data (dict): The client data dictionary.

    Returns:
        int: The total number of samples across all clients. Returns 0 if
             client_data is empty or None.
    """
    logging.info("--- Client Data Shapes ---")
    total_samples = 0
    if not client_data:
        logging.warning("Client data is empty or None.")
        return 0

    for client_id, data in client_data.items():
        # Handle potentially missing 'X' or 'y' keys
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

def load_data(filepath):
    """
    Loads client data from a pickle file and performs basic validation.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        dict or None: The loaded client data dictionary if successful and valid,
                      None otherwise.
    """
    try:
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)

        # Basic validation of loaded structure
        if not isinstance(loaded_data, dict):
            logging.error("Loaded file is not a dictionary (expected client data).")
            raise TypeError("Loaded file is not a dictionary.")

        valid = True
        if not loaded_data: # Allow empty dictionary
             logging.warning("Loaded data dictionary is empty.")
        else:
            for k, v in loaded_data.items():
                # Allow string or int keys, check value structure
                if not isinstance(v, dict) or 'X' not in v or 'y' not in v or \
                   not isinstance(v['X'], np.ndarray) or not isinstance(v['y'], np.ndarray):
                    valid = False
                    logging.error(f"Invalid structure for client key '{k}'. "
                                  f"Expected dict with 'X', 'y' ndarrays.")
                    break
        if not valid:
            raise TypeError("Loaded dictionary does not contain valid client data structure.")

        logging.info(f"Data loaded successfully from {filepath}")
        return loaded_data

    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except pickle.UnpicklingError:
        logging.error(f"Error unpickling file: {filepath}. File might be corrupted or not a pickle file.")
        return None
    except TypeError as te:
        logging.error(f"Error validating loaded data structure: {te}")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred during file loading from {filepath}: {e}")
        return None