# visualization.py

import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors # Import colors for explicit mapping

# Keep PREVIEW_FIGURE_NUM constant if needed elsewhere, but plotting logic changes
PREVIEW_FIGURE_NUM = 99 # May not be strictly needed if we always pass fig/axes

def visualize_data(client_data=None, X_sample=None, y_sample=None, client_ids_sample=None,
                   n_samples_to_plot=1000, method='pca', title="Data Visualization",
                   # New arguments for embedding:
                   fig=None, axes_list=None):
    """
    Visualizes client data distributions using PCA or t-SNE.

    Can accept either a client_data dictionary (will sample from it) or
    pre-sampled X, y, and client_ids arrays.

    If fig and axes_list are provided, plots onto the existing axes instead
    of creating a new figure window.

    Args:
        client_data (dict, optional): Federated dataset dictionary. Defaults to None.
        X_sample (np.ndarray, optional): Pre-sampled feature data. Defaults to None.
        y_sample (np.ndarray, optional): Pre-sampled label data. Defaults to None.
        client_ids_sample (np.ndarray, optional): Pre-sampled client IDs. Defaults to None.
        n_samples_to_plot (int): Max points to sample if using client_data. Defaults to 1000.
        method (str): Dimensionality reduction ('pca' or 'tsne'). Defaults to 'pca'.
        title (str): Base title for the plot window/figure. Defaults to "Data Visualization".
        fig (matplotlib.figure.Figure, optional): Existing figure to plot onto. Defaults to None.
        axes_list (list[matplotlib.axes.Axes], optional): Existing axes to plot onto.
                                                          Should contain 1 or 2 Axes objects.
                                                          Defaults to None.

    Returns:
        bool: True if visualization was successful, False otherwise.

    Raises:
        ValueError: If required data/args are missing or inconsistent.
    """
    create_new_figure = (fig is None or axes_list is None)

    if create_new_figure:
        logging.info("Creating new figure window for visualization.")
    else:
        logging.info("Plotting onto existing figure and axes.")
        if not isinstance(axes_list, list) or not all(isinstance(ax, plt.Axes) for ax in axes_list):
             raise ValueError("If 'fig' is provided, 'axes_list' must be a list of valid Axes objects.")
        # Clear existing axes before plotting new data
        for ax in axes_list:
            ax.clear()

    # --- Data Acquisition/Sampling (Identical to previous version) ---
    if client_data is not None:
        # (Copy the sampling logic from the previous version here)
        # ... it should result in X_sample, y_sample, client_ids_sample ...
        logging.info(f"Starting visualization from client_data using {method.upper()}...")
        all_X_list, all_y_list, all_client_ids_list = [], [], []
        total_samples_available = 0
        client_ids_with_data = []
        for client_id, data in client_data.items():
            if isinstance(data, dict) and 'X' in data and 'y' in data and \
               isinstance(data['X'], np.ndarray) and isinstance(data['y'], np.ndarray):
                n_client_samples = data['X'].shape[0]
                if n_client_samples > 0:
                     total_samples_available += n_client_samples
                     client_ids_with_data.append(client_id)
                elif data['X'].ndim != 2 or data['y'].ndim != 1:
                     logging.warning(f"Client {client_id}: Data arrays have unexpected dimensions. Skipping.")
            else:
                logging.warning(f"Client {client_id} has invalid/incomplete data. Skipping.")
        if total_samples_available == 0: return False
        n_samples_to_plot = min(n_samples_to_plot, total_samples_available)
        sampling_ratio = n_samples_to_plot / total_samples_available if total_samples_available > 0 else 1.0
        if sampling_ratio < 1.0: logging.info(f"Sampling {n_samples_to_plot} points ({sampling_ratio*100:.2f}%) from client_data.")
        sampled_count = 0
        for client_id in client_ids_with_data:
            data = client_data[client_id]; n_client_samples = data['X'].shape[0]
            if sampling_ratio < 1.0:
                n_take = max(1, int(round(n_client_samples * sampling_ratio)))
                n_take = min(n_take, n_samples_to_plot - sampled_count); n_take = max(0, n_take)
            else: n_take = n_client_samples
            n_take = min(n_take, n_client_samples)
            if n_take > 0:
                try:
                    indices = np.random.choice(n_client_samples, size=n_take, replace=False)
                    all_X_list.append(data['X'][indices]); all_y_list.append(data['y'][indices])
                    all_client_ids_list.append(np.full(n_take, client_id)); sampled_count += n_take
                except Exception as e: logging.exception(f"Error processing client {client_id}: {e}. Skipping.")
        if not all_X_list: return False
        try:
            X_sample = np.concatenate(all_X_list); y_sample = np.concatenate(all_y_list)
            client_ids_sample = np.concatenate(all_client_ids_list)
        except ValueError as e: logging.error(f"Error concatenating sampled data: {e}."); return False
    elif X_sample is not None and y_sample is not None and client_ids_sample is not None:
        logging.info(f"Starting visualization from pre-sampled data using {method.upper()}...")
        if X_sample.shape[0] == 0: logging.warning("Provided pre-sampled data is empty."); return False
        if X_sample.shape[0] > n_samples_to_plot:
             logging.info(f"Pre-sampled data ({X_sample.shape[0]}) exceeds limit ({n_samples_to_plot}). Subsampling.")
             indices = np.random.choice(X_sample.shape[0], n_samples_to_plot, replace=False)
             X_sample = X_sample[indices]; y_sample = y_sample[indices]; client_ids_sample = client_ids_sample[indices]
    else:
        raise ValueError("Must provide 'client_data' or all of 'X_sample', 'y_sample', 'client_ids_sample'")
    # --- End Data Acquisition ---

    n_features = X_sample.shape[1]
    if n_features == 0: logging.warning("Sampled data has 0 features."); return False

    # --- Dimensionality Reduction (Identical) ---
    # ... (Copy the dimensionality reduction logic here) ...
    n_components = min(3, n_features)
    X_reduced = None
    if n_components < 2:
         logging.info(f"Data has {n_features} feature(s). Plotting directly.")
         X_reduced = X_sample; n_components = n_features
    elif n_features > n_components:
        logging.info(f"Reducing dimensionality to {n_components} using {method.upper()}...")
        try:
            if method == 'tsne':
                n_eff = X_sample.shape[0]; perp = min(30.0, max(5.0, n_eff / 4.0 ))
                if perp >= n_eff: perp = max(1.0, float(n_eff - 1)); logging.warning(f"Adjusted t-SNE perplexity to {perp:.1f}")
                tsne = TSNE(n_components=n_components, random_state=42, perplexity=perp, n_iter=300, init='pca', learning_rate='auto')
                X_reduced = tsne.fit_transform(X_sample)
            else: pca = PCA(n_components=n_components, random_state=42); X_reduced = pca.fit_transform(X_sample)
            logging.info("Dimensionality reduction complete.")
        except Exception as e: logging.exception(f"Error during DR ({method.upper()}): {e}"); return False
    else: logging.info(f"Data has {n_features} features. Plotting original."); X_reduced = X_sample
    if X_reduced is None or X_reduced.shape[0] != X_sample.shape[0]: logging.error("DR failed."); return False
    # --- End Dimensionality Reduction ---

    # --- Plotting ---
    try:
        if create_new_figure:
             # Create a new figure and axes if none were passed
             fig, axes_list = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': '3d'} if n_components >= 3 else {})
             if n_components < 3: # Adjust projection for 2D/1D
                 plt.close(fig) # Close the 3D one
                 fig = plt.figure(figsize=(16, 7))
                 axes_list = [fig.add_subplot(121), fig.add_subplot(122)]
                 if n_components == 1:
                      # For 1D, still use 2 axes, but configure them for strip plot
                      axes_list = [fig.add_subplot(121), fig.add_subplot(122, sharex=axes_list[0])] # Share X for 1D
                 elif n_components == 2:
                       axes_list = [fig.add_subplot(121), fig.add_subplot(122, sharex=axes_list[0], sharey=axes_list[0])] # Share XY for 2D

             # Try setting window title for standalone figure
             try: fig.canvas.manager.set_window_title(f"Data Visualization - {title}")
             except AttributeError: fig.suptitle(f"Data Visualization - {title}", fontsize=14)
        elif len(axes_list) != 2:
             logging.error("Expected exactly 2 axes in axes_list for plotting.")
             # If embedding, we expect the GUI to provide the correct axes.
             # If axes number changed dynamically, GUI needs recreation or axes update.
             # For now, just try using the first two if available.
             if len(axes_list) < 2: return False # Cannot proceed

        # We need two axes: one for client plot, one for class plot
        ax1 = axes_list[0]
        ax2 = axes_list[1]

        unique_clients = np.unique(client_ids_sample)
        n_clients_plot = len(unique_clients)
        unique_classes = np.unique(y_sample)
        n_classes_plot = len(unique_classes)

        # --- Plot 1: Color by Client ID on ax1 ---
        cmap_clients = plt.get_cmap('turbo', max(1, n_clients_plot))
        client_norm = matplotlib.colors.Normalize(vmin=np.min(unique_clients), vmax=np.max(unique_clients))
        client_mapper = plt.cm.ScalarMappable(norm=client_norm, cmap=cmap_clients)

        if n_components >= 3:
            # Important: Check if ax1 is already 3D. If not, we can't just scatter 3D.
            # The GUI should provide axes with the correct projection. Assume they are correct.
            scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], zs=X_reduced[:, 2],
                                   c=client_mapper.to_rgba(client_ids_sample), alpha=0.6, s=10)
            ax1.set_zlabel('Component 3')
        elif n_components == 2:
            scatter1 = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1],
                                   c=client_mapper.to_rgba(client_ids_sample), alpha=0.6, s=10)
        else: # n_components == 1
            y_jitter = np.random.uniform(-0.1, 0.1, size=X_reduced.shape[0])
            scatter1 = ax1.scatter(X_reduced[:, 0], y_jitter,
                                   c=client_mapper.to_rgba(client_ids_sample), alpha=0.6, s=10)
            ax1.yaxis.set_visible(False)

        ax1.set_title(f'Distribution by Client ID ({method.upper()})')
        ax1.set_xlabel('Component 1')
        if n_components > 1: ax1.set_ylabel('Component 2')

        # Add colorbar to ax1 (handles legend implicitly)
        # Remove previous colorbars first if they exist
        for cbar in getattr(ax1, '_colorbar_list', []): cbar.remove()
        ax1._colorbar_list = []
        if n_clients_plot > 0: # Check if there are clients to plot
             cbar1 = fig.colorbar(client_mapper, ax=ax1, label='Client ID', shrink=0.8)
             ax1._colorbar_list.append(cbar1) # Store reference
             ticks_clients = np.linspace(np.min(unique_clients), np.max(unique_clients), num=min(5, n_clients_plot))
             if unique_clients.size > 0 and np.issubdtype(unique_clients.dtype, np.integer):
                 ticks_clients = np.unique(ticks_clients.astype(int))
             cbar1.set_ticks(ticks_clients)
        # --- End Plot 1 ---


        # --- Plot 2: Color by Class Label on ax2 ---
        cmap_classes = plt.get_cmap('viridis', max(1, n_classes_plot))
        class_norm = matplotlib.colors.Normalize(vmin=np.min(unique_classes), vmax=np.max(unique_classes))
        class_mapper = plt.cm.ScalarMappable(norm=class_norm, cmap=cmap_classes)

        if n_components >= 3:
            scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                                   c=class_mapper.to_rgba(y_sample), alpha=0.6, s=10)
            ax2.set_zlabel('Component 3')
        elif n_components == 2:
            scatter2 = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1],
                                   c=class_mapper.to_rgba(y_sample), alpha=0.6, s=10)
        else: # n_components == 1
            y_jitter = np.random.uniform(-0.1, 0.1, size=X_reduced.shape[0])
            scatter2 = ax2.scatter(X_reduced[:, 0], y_jitter,
                                   c=class_mapper.to_rgba(y_sample), alpha=0.6, s=10)
            ax2.yaxis.set_visible(False)

        ax2.set_title(f'Distribution by Class Label ({method.upper()})')
        ax2.set_xlabel('Component 1')
        if n_components > 1: ax2.set_ylabel('Component 2')

        # Add colorbar to ax2
        for cbar in getattr(ax2, '_colorbar_list', []): cbar.remove()
        ax2._colorbar_list = []
        if n_classes_plot > 0: # Check if there are classes to plot
            cbar2 = fig.colorbar(class_mapper, ax=ax2, label='Class Label', shrink=0.8)
            ax2._colorbar_list.append(cbar2)
            ticks_classes = np.linspace(np.min(unique_classes), np.max(unique_classes), num=min(n_classes_plot, 10))
            if unique_classes.size > 0 and np.issubdtype(unique_classes.dtype, np.integer):
                ticks_classes = np.unique(ticks_classes.astype(int))
            cbar2.set_ticks(ticks_classes)
        # --- End Plot 2 ---


        # Adjust layout Tight layout might need careful use with embedded plots.
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # May cause issues, try without first

        if create_new_figure:
            plt.show() # Only call show for standalone figures
        # If embedded, the GUI's canvas.draw() will handle updating the display

        return True

    except Exception as e:
        logging.exception(f"Error during plotting: {e}")
        return False