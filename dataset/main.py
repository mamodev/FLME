# CLI Examples remain the same, just run `python main.py ...`
# e.g.:
# python main.py --n_samples 5000 --n_clients 10 --quantity_skew_alpha 100 --label_skew_alpha 100 --visualize
# python main.py --gui
# python main.py --gui --n_clients 20 --n_classes 5

# Standard library imports
import sys
import logging
import plot_utils

# Local module imports
import config
from data_generator import generate_federated_dataset
from visualization import visualize_data
from data_utils import check_data_shapes # For CLI checks

def cli(args):
    client_data, success, saved_path = generate_federated_dataset(
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
        output_dir=args.output_dir,
        save_to_file=True # Always save in CLI mode
    )

    if not success:
            logging.error("CLI Data generation failed.")
            sys.exit(1) # Exit with error status if generation failed

    if success and saved_path:
            logging.info(f"CLI Data generation successful. Dataset saved to: {saved_path}")
    elif success and not saved_path:
            logging.warning("CLI Data generation finished, but data saving failed or was skipped (e.g. 0 samples). Check logs.")
            # Continue to visualization if requested, even if save failed


    # Perform visualization if requested and data exists
    if args.visualize:
        if client_data is not None and check_data_shapes(client_data) > 0:
            logging.info("Proceeding with visualization...")

            # Create title for CLI visualization
            title = f"CLI Gen | N={args.n_samples}, F={args.n_features}, C={args.n_classes}, K={args.n_clients}\n"
            title += f"QSkew(α={args.quantity_skew_alpha}), LSkew(α={args.label_skew_alpha})\n"
            title += f"Feat(lvl={args.feature_skew_level}), Drift(lvl={args.concept_drift_level}), Shift(lvl={args.concept_shift_level})"

            # Set a non-interactive backend if possible, unless TkAgg was needed and available
            try:
                    # Keep default or TkAgg if set earlier
                    pass
            except Exception:
                    logging.warning("Could not query/set matplotlib backend for CLI visualization.")


            viz_success = visualize_data(
                client_data,
                n_samples_to_plot=args.vis_samples,
                method=args.vis_method,
                title=title
            )
            if not viz_success:
                logging.error("Visualization failed during CLI execution.")
            else:
                logging.info("Visualization displayed (or saved, depending on backend).")

        elif client_data is None:
                logging.warning("Visualization requested (--visualize) but data generation failed or yielded no data.")
        else: # client_data exists but is empty
                logging.warning("Visualization requested (--visualize) but the generated dataset is empty.")

    logging.info("CLI execution finished.")

if __name__ == "__main__":
    args = config.get_config()

    if args.gui:
        logging.info("Launching GUI mode...")
        plot_utils.init_env(args)
    else:
        logging.info("Running in Command-Line Interface (CLI) mode.")
        cli(args)
