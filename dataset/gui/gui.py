# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import os
import argparse
import numpy as np
import hashlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import functionalities
from config import DEFAULT_ARGS
from data_generator import generate_federated_dataset, generate_base_data
from skew_functions import apply_quantity_skew
from visualization import visualize_data # Modified visualize_data
from data_utils import load_data

# RNGs (same as before)
preview_rng_params = np.random.RandomState(100)
preview_rng_transforms = np.random.RandomState(101)
preview_rng_sampling = np.random.RandomState(102)

from gui.tkmain import TkMain
from gui.state import _args_to_tk_vars

class GeneratorApp:
    """
    GUI application with embedded plot and collapsible parameter panel.
    """
    def __init__(self, root, initial_args):
        self.root = root
        self.defaults = initial_args
        self.client_data = None 
        self.last_output_dir = initial_args.output_dir

        # Preview Cache
        self.preview_X_global = None
        self.preview_y_global = None
        self.preview_base_params_hash = None

        # window setup
        self.vars = _args_to_tk_vars(initial_args)
        self.root.title("Dataset Generator")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        style = ttk.Style()
        try: style.theme_use('clam')
        except tk.TclError: logging.warning("Clam theme not available.")

        # TkMain(self.root, self.vars)
  

    def _get_params_from_gui(self, validate=True):
        # (Same as before)
        params = {}; errors = []; names = {
            "n_samples": "Samples", "n_features": "Features", "n_classes": "Classes", "n_clients": "Clients",
            "class_sep": "Class Sep", "info_frac": "Info Frac", "quantity_skew_alpha": "Qty Skew α",
            "label_skew_alpha": "Lbl Skew α", "feature_skew_level": "Feat Skew Lvl", "concept_drift_level": "Drift Lvl",
            "concept_shift_level": "Shift Lvl", "output_dir": "Output Dir", "vis_samples": "Vis Samples", "vis_method": "Vis Method" }
        for name, tk_var in self.vars.items():
            d_name = names.get(name, name)
            try:
                val = tk_var.get()
                if isinstance(tk_var, tk.IntVar): params[name] = int(val)
                elif isinstance(tk_var, tk.DoubleVar): params[name] = float(val)
                else: params[name] = str(val).strip()
                if validate:
                     if name in ["n_samples","n_features","n_classes"] and params[name] <= 0: errors.append(f"{d_name}>0")
                     if name=="n_clients" and params[name]<0: errors.append(f"{d_name}>=0")
                     if name in ["quantity_skew_alpha","label_skew_alpha"] and params[name]<=0: errors.append(f"{d_name}>0")
                     if name in ["feature_skew_level","concept_drift_level"] and params[name]<0: errors.append(f"{d_name}>=0")
                     if name=="concept_shift_level" and not(0<=params[name]<=1): errors.append(f"{d_name} [0,1]")
                     if name=="info_frac" and not(0<params[name]<=1): errors.append(f"{d_name} (0,1]")
                     if name=="output_dir" and not params[name]: errors.append(f"{d_name} not empty")
                     if name=="vis_samples" and params[name]<=0: errors.append(f"{d_name}>0")
            except (tk.TclError, ValueError) as e: errors.append(f"Invalid {d_name}: {e}")
        if errors and validate: messagebox.showerror("Invalid Params", "Errors:\n" + "\n".join(errors)); return None
        elif errors: logging.warning(f"Param errors (validation skipped): {errors}")
        return argparse.Namespace(**params)

    # --- Cache and Plotting ---
    def _invalidate_preview_cache(self, *args):
        # (Same as before)
        if self.preview_base_params_hash is not None:
            logging.info("Base parameters changed, invalidating preview cache.")
            self.preview_X_global = None
            self.preview_y_global = None
            self.preview_base_params_hash = None
            # Optionally clear the plot when base params change?
            # self._clear_plot()

    def _generate_preview_data(self, params):
        # (Same simulation logic as before)
        # ...
        logging.info("Generating data for preview...")
        base_param_keys = ["n_samples", "n_features", "n_classes", "class_sep", "info_frac"]
        current_base_params = {k: getattr(params, k) for k in base_param_keys}
        current_hash = hashlib.md5(str(sorted(current_base_params.items())).encode()).hexdigest()

        # 1. Get Base Data
        if self.preview_base_params_hash != current_hash or self.preview_X_global is None:
            logging.info("Generating new base data for preview...")
            self.root.config(cursor="watch"); self.root.update_idletasks()
            X_global, y_global = generate_base_data(params.n_samples, params.n_features, params.n_classes, params.class_sep, params.info_frac)
            self.root.config(cursor="")
            if X_global is None: messagebox.showerror("Preview Error", "Failed base data gen."); return None, None, None
            self.preview_X_global = X_global; self.preview_y_global = y_global
            self.preview_base_params_hash = current_hash; logging.info("Cached new base data.")
        else:
            logging.info("Using cached base data."); X_global = self.preview_X_global; y_global = self.preview_y_global
        if X_global.shape[0] == 0: return np.empty((0, params.n_features)), np.empty((0,)), np.empty((0,))

        # 2. Simulate Qty/Label Skew for Sampling
        n_vis = params.vis_samples; n_clients = params.n_clients; n_classes = params.n_classes; n_features = params.n_features
        if n_clients <= 0:
            logging.warning("Preview: n_clients=0, showing base sample."); idx = preview_rng_sampling.choice(X_global.shape[0], min(n_vis, X_global.shape[0]), replace=False)
            return X_global[idx], y_global[idx], np.zeros(len(idx), dtype=int)
        try: target_client_counts = apply_quantity_skew(n_vis, n_clients, params.quantity_skew_alpha)
        except ValueError as e: messagebox.showerror("Preview Error", f"Invalid Qty Skew α: {e}"); return None, None, None
        try:
            label_alpha = params.label_skew_alpha;
            if label_alpha <= 0: raise ValueError("Lbl Skew α > 0")
            client_label_props = preview_rng_params.dirichlet(np.full(n_classes, label_alpha), size=n_clients)
        except ValueError as e: messagebox.showerror("Preview Error", f"Invalid Lbl Skew α: {e}"); return None, None, None

        target_counts_vis = np.zeros((n_clients, n_classes), dtype=int); total_assigned_vis = 0
        for i in range(n_clients):
            if target_client_counts[i] > 0:
                props = client_label_props[i] / client_label_props[i].sum()
                try: counts_i = preview_rng_params.multinomial(target_client_counts[i], props)
                except ValueError: counts_i = (props * target_client_counts[i]).round().astype(int)
                diff = target_client_counts[i] - counts_i.sum()
                if diff != 0: adj_idx = preview_rng_params.choice(n_classes, abs(diff), p=props); np.add.at(counts_i, adj_idx, np.sign(diff)); counts_i = np.maximum(0, counts_i)
                current_sum_i = counts_i.sum() # Recalculate sum after adjustment
                if current_sum_i != target_client_counts[i]:
                    diff = target_client_counts[i] - current_sum_i; adj_idx_fallback = preview_rng_params.choice(n_classes, abs(diff)); np.add.at(counts_i, adj_idx_fallback, np.sign(diff)); counts_i = np.maximum(0, counts_i)
                target_counts_vis[i] = counts_i; total_assigned_vis += counts_i.sum()
        diff_total_vis = n_vis - total_assigned_vis
        if diff_total_vis != 0 and n_clients > 0: logging.debug(f"Adjusting total preview count by {diff_total_vis}"); target_counts_vis[-1, preview_rng_params.choice(n_classes)] += diff_total_vis; target_counts_vis[-1] = np.maximum(0, target_counts_vis[-1])

        # 3. Sample Points
        X_list, y_list, c_list = [], [], []; available_idx = {k: list(np.where(y_global == k)[0]) for k in range(n_classes)}
        for k in available_idx: preview_rng_sampling.shuffle(available_idx[k])
        idx_ptrs = {k: 0 for k in range(n_classes)}
        for cid in range(n_clients):
            for k in range(n_classes):
                needed = target_counts_vis[cid, k];
                if needed == 0: continue
                avail_k = available_idx[k]; ptr = idx_ptrs[k]; num_avail = len(avail_k) - ptr
                take = min(needed, num_avail)
                if take > 0: end = ptr + take; chosen = avail_k[ptr:end]; X_list.append(X_global[chosen]); y_list.append(y_global[chosen]); c_list.append(np.full(take, cid)); idx_ptrs[k] = end
                if take < needed: logging.warning(f"Preview Sample: Client {cid}, Class {k} needed {needed}, got {take}.")
        if not X_list: return np.empty((0, n_features)), np.empty((0,)), np.empty((0,))
        X_preview = np.concatenate(X_list); y_preview_orig = np.concatenate(y_list); cid_preview = np.concatenate(c_list)

        # 4. Simulate Skews on Sample
        y_preview_final = y_preview_orig.copy()
        # Feat Skew
        level = params.feature_skew_level
        if level > 0:
            logging.debug("Simulating Feat Skew"); scale_var = level*0.5; shift_std = level*1.0; min_s = max(0.01, 1.0-scale_var); max_s = 1.0+scale_var
            c_tf = {}; unique_c = np.unique(cid_preview)
            for c in unique_c: c_tf[c] = {'s': preview_rng_transforms.uniform(min_s, max_s, size=n_features), 't': preview_rng_transforms.normal(0, shift_std, size=n_features)}
            for c in unique_c: mask = (cid_preview == c); X_preview[mask] = X_preview[mask] * c_tf[c]['s'] + c_tf[c]['t']
        # Drift
        level = params.concept_drift_level
        if level > 0:
            logging.debug("Simulating Drift"); drift_std = level*0.5; cc_drifts = {}
            unique_c = np.unique(cid_preview); unique_k = np.unique(y_preview_orig)
            for c in unique_c: cc_drifts[c] = {k: preview_rng_transforms.normal(0, drift_std, size=n_features) for k in unique_k}
            for c in unique_c:
                for k in unique_k: mask = (cid_preview==c)&(y_preview_orig==k); X_preview[mask] = X_preview[mask] + cc_drifts[c][k]
        # Shift
        level = params.concept_shift_level
        if level > 0 and n_classes > 1:
            logging.debug("Simulating Shift"); flip_prob = level; flip_mask = preview_rng_transforms.rand(X_preview.shape[0]) < flip_prob
            idx_flip = np.where(flip_mask)[0]; all_lbls = list(range(n_classes))
            for idx in idx_flip:
                orig = y_preview_final[idx]; new_lbls = [l for l in all_lbls if l != orig]
                if new_lbls: y_preview_final[idx] = preview_rng_transforms.choice(new_lbls)

        logging.info("Preview data simulation complete.")
        return X_preview, y_preview_final, cid_preview

    def _clear_plot(self):
        """Clears the embedded plot axes."""
        if hasattr(self, 'axes'):
            for ax in self.axes:
                ax.clear()
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
                if hasattr(ax, 'set_zlabel'): ax.set_zlabel("")
                 # Remove colorbars associated with this axis
                if hasattr(ax, '_colorbar_list'):
                     for cbar in ax._colorbar_list:
                         try: cbar.remove()
                         except Exception as e: logging.debug(f"Error removing colorbar: {e}")
                     ax._colorbar_list = []

            # Add placeholder text
            self.axes[0].text(0.5, 0.5, 'Plot Area\n(Update Preview or Visualize Generated)',
                              horizontalalignment='center', verticalalignment='center',
                              transform=self.axes[0].transAxes, fontsize=12, color='grey')
            self.axes[1].set_visible(False) # Hide second axis initially

            if hasattr(self, 'canvas'):
                self.canvas.draw_idle()

    # --- Action Callbacks (Modified for embedded plot) ---
    def _update_preview_callback(self):
        """Callback for the 'Update Preview' button."""
        logging.info("Update Preview button clicked.")
        params = self._get_params_from_gui(validate=True)
        if params is None: return
        if params.vis_samples <= 0: messagebox.showerror("Invalid", "Vis Samples > 0"); return
        if params.vis_method not in ['pca', 'tsne']: messagebox.showerror("Invalid", "Vis Method"); return

        self.root.config(cursor="watch"); self.root.update_idletasks()
        X_prev, y_prev, cids_prev = self._generate_preview_data(params)
        self.root.config(cursor="")

        if X_prev is not None:
            logging.info(f"Visualizing preview sample ({X_prev.shape[0]} points) in embedded plot...")
            title = f"Preview | Qα={params.quantity_skew_alpha:.1f}, Lα={params.label_skew_alpha:.1f}, " \
                    f"Feat={params.feature_skew_level:.1f}, Drift={params.concept_drift_level:.1f}, Shift={params.concept_shift_level:.1f}"

            # Ensure the second axis is visible for the plot
            self.axes[1].set_visible(True)

            viz_success = visualize_data(
                X_sample=X_prev, y_sample=y_prev, client_ids_sample=cids_prev,
                n_samples_to_plot=params.vis_samples, method=params.vis_method, title=title,
                fig=self.figure, axes_list=self.axes # Pass figure and axes
            )
            if viz_success:
                self.canvas.draw_idle() # Redraw the canvas
            else:
                messagebox.showwarning("Preview", "Could not display preview visualization.")
                self._clear_plot() # Clear plot on failure
        else:
            logging.error("Preview data generation failed, cannot visualize.")
            self._clear_plot() # Clear plot on failure


    def _visualize_generated_callback(self):
        """Callback for visualizing the fully generated/loaded data in the embedded plot."""
        logging.info("Visualize Generated button clicked.")
        if self.client_data is None:
            messagebox.showwarning("Visualize", "No generated data available."); return

        params = self._get_params_from_gui(validate=False)
        vis_samples = params.vis_samples if params else DEFAULT_ARGS.vis_samples
        vis_method = params.vis_method if params else DEFAULT_ARGS.vis_method
        if vis_samples <= 0: messagebox.showerror("Invalid", "Vis Samples > 0"); return

        logging.info(f"Visualizing generated data ({vis_samples} samples, {vis_method}) in embedded plot...")
        self.root.config(cursor="watch"); self.root.update_idletasks()

        title_str = "Generated/Loaded Data"
        try: # Attempt to add details to title
            n_cli = len(self.client_data); total_s = sum(d['X'].shape[0] for d in self.client_data.values() if 'X' in d)
            all_y = [d['y'] for d in self.client_data.values() if 'y' in d and d['y'].size > 0]
            n_cls = len(np.unique(np.concatenate(all_y))) if all_y else '?'
            title_str += f" (K={n_cli}, N={total_s}, Cls={n_cls})"
        except Exception: pass

        # Ensure the second axis is visible for the plot
        self.axes[1].set_visible(True)

        viz_success = visualize_data(
             client_data=self.client_data, # Pass the generated data
             n_samples_to_plot=vis_samples, method=vis_method, title=title_str,
             fig=self.figure, axes_list=self.axes # Pass figure and axes
        )
        self.root.config(cursor="")

        if viz_success:
            self.canvas.draw_idle() # Redraw the canvas
        else:
             messagebox.showwarning("Visualize Failed", "Could not display generated data visualization.")
             self._clear_plot()


    def _generate_data_callback(self):
        # (Generates data as before, enables visualize button)
        logging.info("Generate button clicked.")
        params = self._get_params_from_gui(validate=True)
        if params is None: return

        logging.info("Starting generation from GUI parameters...")
        self.root.config(cursor="watch"); self.root.update_idletasks()
        client_data_result, success, saved_path = generate_federated_dataset(
            n_samples=params.n_samples, n_features=params.n_features, n_classes=params.n_classes,
            n_clients=params.n_clients, quantity_skew_alpha=params.quantity_skew_alpha,
            label_skew_alpha=params.label_skew_alpha, feature_skew_level=params.feature_skew_level,
            concept_drift_level=params.concept_drift_level, concept_shift_level=params.concept_shift_level,
            base_class_sep=params.class_sep, base_n_informative_frac=params.info_frac,
            output_dir=params.output_dir, save_to_file=True
        )
        self.root.config(cursor="")

        if success:
            self.client_data = client_data_result
            if saved_path: messagebox.showinfo("Complete", f"Data generated.\nSaved to: {saved_path}"); self.last_output_dir = os.path.dirname(saved_path)
            else: messagebox.showwarning("Complete", "Data generated, but save failed.")
            if self.client_data and any(d.get('X', np.empty((0,0))).shape[0] > 0 for d in self.client_data.values()):
                self.visualize_button.config(state=tk.NORMAL)
            else: self.visualize_button.config(state=tk.DISABLED)
        else: messagebox.showerror("Failed", "Data generation failed."); self.client_data = None; self.visualize_button.config(state=tk.DISABLED)

    def _generate_and_visualize_generated_callback(self):
        # (Generate then visualize generated in embedded plot)
        logging.info("Generate & Visualize Generated button clicked.")
        self._generate_data_callback()
        if self.client_data is not None:
            self._visualize_generated_callback()

    def _load_data_callback(self):
        # (Loads data, enables visualize button, clears plot)
        logging.info("Load button clicked.")
        initial_dir = self.last_output_dir if os.path.isdir(self.last_output_dir) else os.getcwd()
        filepath = filedialog.askopenfilename(title="Select Dataset File (.pkl)", initialdir=initial_dir, filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not filepath: return
        logging.info(f"Loading from: {filepath}")
        self.root.config(cursor="watch"); self.root.update_idletasks()
        loaded_data = load_data(filepath)
        self.root.config(cursor="")
        if loaded_data is not None:
            self.client_data = loaded_data; self.last_output_dir = os.path.dirname(filepath)
            messagebox.showinfo("Load Complete", f"Loaded data from\n{filepath}")
            self._invalidate_preview_cache() # Invalidate preview
            self._clear_plot() # Clear plot area
            if self.client_data and any(d.get('X', np.empty((0,0))).shape[0] > 0 for d in self.client_data.values()):
                self.visualize_button.config(state=tk.NORMAL)
            else:
                self.visualize_button.config(state=tk.DISABLED)
                if not self.client_data: messagebox.showinfo("Load Info", f"Loaded empty dataset structure.")
                else: messagebox.showwarning("Load Info", f"Loaded data contains no samples.")
        else: messagebox.showerror("Load Error", f"Failed to load/validate data from\n{filepath}"); self.client_data = None; self.visualize_button.config(state=tk.DISABLED)

    def run(self):
        self.root.mainloop()