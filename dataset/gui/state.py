
import tkinter as tk

def _args_to_tk_vars(args):
    return {
        "n_samples": tk.IntVar(value=args.n_samples), "n_features": tk.IntVar(value=args.n_features),
        "n_classes": tk.IntVar(value=args.n_classes), "n_clients": tk.IntVar(value=args.n_clients),
        "class_sep": tk.DoubleVar(value=args.class_sep), "info_frac": tk.DoubleVar(value=args.info_frac),
        "quantity_skew_alpha": tk.DoubleVar(value=args.quantity_skew_alpha),
        "label_skew_alpha": tk.DoubleVar(value=args.label_skew_alpha),
        "feature_skew_level": tk.DoubleVar(value=args.feature_skew_level),
        "concept_drift_level": tk.DoubleVar(value=args.concept_drift_level),
        "concept_shift_level": tk.DoubleVar(value=args.concept_shift_level),
        "output_dir": tk.StringVar(value=args.output_dir), "vis_samples": tk.IntVar(value=args.vis_samples),
        "vis_method": tk.StringVar(value=args.vis_method),
    }