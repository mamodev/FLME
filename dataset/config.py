"""
Configuration parameters and command-line argument handling for the data generator.
"""
import argparse
import os
import logging
import sys

log_level = logging.INFO # Or change to DEBUG for more details
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

# Define default values directly or derive them as needed
DEFAULT_N_SAMPLES = 10000
DEFAULT_N_FEATURES = 20
DEFAULT_N_CLASSES = 10
DEFAULT_N_CLIENTS = 100
DEFAULT_CLASS_SEP = 1.0
DEFAULT_INFO_FRAC = 0.8
DEFAULT_QUANTITY_SKEW_ALPHA = 1.0
DEFAULT_LABEL_SKEW_ALPHA = 1.0
DEFAULT_FEATURE_SKEW_LEVEL = 0.0
DEFAULT_CONCEPT_DRIFT_LEVEL = 0.0
DEFAULT_CONCEPT_SHIFT_LEVEL = 0.0
DEFAULT_OUTPUT_DIR = 'federated_data'
DEFAULT_VIS_SAMPLES = 2000
DEFAULT_VIS_METHOD = 'pca'

def create_parser():
    """
    Creates the argparse parser for command-line execution or default retrieval.

    Defines all configurable parameters and runtime flags.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate Synthetic Non-IID Federated Datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # Base Dataset Parameters
    parser.add_argument('--n_samples', type=int, default=DEFAULT_N_SAMPLES, help='Total number of samples')
    parser.add_argument('--n_features', type=int, default=DEFAULT_N_FEATURES, help='Number of features')
    parser.add_argument('--n_classes', type=int, default=DEFAULT_N_CLASSES, help='Number of classes')
    parser.add_argument('--n_clients', type=int, default=DEFAULT_N_CLIENTS, help='Number of clients')
    parser.add_argument('--class_sep', type=float, default=DEFAULT_CLASS_SEP, help='Base class separation factor')
    parser.add_argument('--info_frac', type=float, default=DEFAULT_INFO_FRAC, help='Fraction of informative features (0 to 1)')

    # Skew Control Parameters
    parser.add_argument('--quantity_skew_alpha', type=float, default=DEFAULT_QUANTITY_SKEW_ALPHA, help='Dirichlet alpha for quantity skew (>0)')
    parser.add_argument('--label_skew_alpha', type=float, default=DEFAULT_LABEL_SKEW_ALPHA, help='Dirichlet alpha for label skew (>0)')
    parser.add_argument('--feature_skew_level', type=float, default=DEFAULT_FEATURE_SKEW_LEVEL, help='Level of feature skew (>=0)')
    parser.add_argument('--concept_drift_level', type=float, default=DEFAULT_CONCEPT_DRIFT_LEVEL, help='Level of concept drift (>=0)')
    parser.add_argument('--concept_shift_level', type=float, default=DEFAULT_CONCEPT_SHIFT_LEVEL, help='Level of concept shift (label flip prob, 0 to 1)')

    # Output & Visualization
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory to save the dataset')
    parser.add_argument('--vis_samples', type=int, default=DEFAULT_VIS_SAMPLES, help='Max samples for visualization')
    parser.add_argument('--vis_method', type=str, default=DEFAULT_VIS_METHOD, choices=['pca', 'tsne'], help='Visualization method')

    # Runtime Flags
    parser.add_argument('--visualize', action='store_true', help='Visualize after generation (CLI mode only)')
    parser.add_argument('--gui', action='store_true', help='Launch the interactive GUI instead of CLI execution')

    return parser

def get_default_args():
    """
    Parses an empty list of arguments to retrieve the default values.

    Returns:
        argparse.Namespace: An object containing the default argument values.
    """
    parser = create_parser()
    # Parse known args from empty list to get defaults without CLI influence
    defaults, _ = parser.parse_known_args([])
    return defaults

def validate_cli_args(args):
    """
    Performs semantic validation specific to CLI arguments.

    Logs errors and exits if invalid arguments are found.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        bool: True if arguments are valid, False otherwise (and logs errors).
    """
    errors = []
    if args.n_samples <= 0: errors.append("--n_samples must be > 0")
    if args.n_features <= 0: errors.append("--n_features must be > 0")
    if args.n_classes <= 0: errors.append("--n_classes must be > 0")
    if args.n_clients < 0: errors.append("--n_clients must be >= 0") # Allow 0
    if args.quantity_skew_alpha <= 0: errors.append("--quantity_skew_alpha must be > 0")
    if args.label_skew_alpha <= 0: errors.append("--label_skew_alpha must be > 0")
    if not (0 < args.info_frac <= 1): errors.append("--info_frac must be > 0 and <= 1")
    if args.feature_skew_level < 0: errors.append("--feature_skew_level must be >= 0")
    if args.concept_drift_level < 0: errors.append("--concept_drift_level must be >= 0")
    if not (0 <= args.concept_shift_level <= 1): errors.append("--concept_shift_level must be between 0 and 1")
    if args.vis_samples <= 0: errors.append("--vis_samples must be > 0")
    if not args.output_dir: errors.append("--output_dir cannot be empty")

    # Check writability of output directory *parent* if dir doesn't exist
    output_dir = args.output_dir
    try:
        if not os.path.isdir(output_dir):
            parent_dir = os.path.dirname(os.path.abspath(output_dir))
            # Check parent existence and writability
            if not os.path.isdir(parent_dir):
                # Try creating parent directory if it doesn't exist.
                # This might be too aggressive, depends on desired behavior.
                # Let's just report the error if parent doesn't exist.
                 errors.append(f"Parent directory of specified output directory does not exist: {parent_dir}")
            elif not os.access(parent_dir, os.W_OK):
                errors.append(f"Cannot write to parent directory '{parent_dir}' to create output directory '{os.path.basename(output_dir)}'")
        elif not os.access(output_dir, os.W_OK):
             errors.append(f"Cannot write to existing output directory: {output_dir}")
    except Exception as e:
         errors.append(f"Error checking output directory '{output_dir}': {e}")


    if errors:
        logging.error("Invalid command-line arguments:")
        for error in errors:
            logging.error(f"  - {error}")
        return False # Indicate failure
    return True # Indicate success

# --- Make defaults accessible directly (optional, but used by GUI) ---
# Note: This runs get_default_args() once when the module is imported.
DEFAULT_ARGS = get_default_args()


def get_config():
    """
    Parses command-line arguments and returns the configuration object.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = create_parser()
    
    if not validate_cli_args(parser.parse_args()):
        logging.error("Invalid command-line arguments. Exiting.")
        sys.exit(1)

    return parser.parse_args()