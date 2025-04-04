
import tkinter as tk

from components import Form
from layout import run_gui

import numpy as np

# import PCA and t-SNE for visualization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def random_sample(n_samples, ds):
    """Select a random sample from the dataset of size n_samples."""
    if n_samples > len(ds):
        raise ValueError("Sample size exceeds dataset size.")
    
    indices = np.random.choice(len(ds), n_samples, replace=False)
    return ds[indices]
    

class Dataset:
    def visualize(fig, ax, vis_method, XX, yy):
        
        """Visualize the dataset using PCA or t-SNE."""
        

        if XX.shape[1] > 3:
            if vis_method == "pca":
                pca = PCA(n_components=3)
                XX = pca.fit_transform(XX)
            elif vis_method == "tsne":
                tsne = TSNE(n_components=3)
                XX = tsne.fit_transform(XX)

        ax.scatter(XX[:, 0], XX[:, 1], XX[:, 2], c=yy, cmap='viridis', s=5, alpha=0.5)
        ax.set_title("Data Visualization")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        ax.grid(True)

    def generate(vars):
        XX = np.random.randn(vars["n_samples"].get(), vars["n_features"].get())
        yy = np.random.randint(0, vars["n_classes"].get(), vars["n_samples"].get())
        return (XX, yy)

    def get_form(ctx):
        vars = {
            "n_samples": tk.IntVar(value=10000), "n_features": tk.IntVar(value=20),
            "n_classes": tk.IntVar(value=10), "n_clients": tk.IntVar(value=100),
            "class_sep": tk.DoubleVar(value=1.0), "info_frac": tk.DoubleVar(value=0.8),
            "quantity_skew_alpha": tk.DoubleVar(value=1.0),
            "label_skew_alpha": tk.DoubleVar(value=1.0),
            "feature_skew_level": tk.DoubleVar(value=0.0),
            "concept_drift_level": tk.DoubleVar(value=0.0),
            "concept_shift_level": tk.DoubleVar(value=0.0),
            "output_dir": tk.StringVar(value="output"),
            "vis_samples": tk.IntVar(value=1000), "vis_method": tk.StringVar(value="pca"),
        }

        return Form(ctx, vars) \
            .group("Base Parameters") \
                .input("Total Samples:", "n_samples", int) \
                .input("Num Features:", "n_features", int) \
                .input("Num Classes:", "n_classes", int) \
                .input("Num Clients:", "n_clients", int) \
            .group("Skew Parameters") \
                .input("Qty Skew α (>0):", "quantity_skew_alpha", float) \
                .input("Lbl Skew α (>0):", "label_skew_alpha", float) \
                .input("Feat Skew Lvl:", "feature_skew_level", float) \
                .input("Drift Lvl:", "concept_drift_level", float) \
                .input("Shift Lvl (0-1):", "concept_shift_level", float) \
            .group("Output & Visualization") \
                .input("Output Dir:", "output_dir", str) \
                .input("Vis Samples:", "vis_samples", int) \
                .select("Vis Method:", "vis_method", ['pca', 'tsne']) \
            .done()
        
if __name__ == "__main__":
    run_gui({
        "dataset1": Dataset,
    })