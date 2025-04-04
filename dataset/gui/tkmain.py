import tkinter as tk
from tkinter import ttk

from gui.validators import validate_int, validate_float
from gui.components import Form
from gui.state import _args_to_tk_vars

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def plot_cos(ax):
    import numpy as np
    x = np.linspace(0, 10, 100)
    y = np.cos(x)
    ax.plot(x, y, color='r', label="Cos Wave")
    ax.legend()

def run_gui(args):
    root = tk.Tk()

    root.title("Synthetic Data Generator")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    vars = _args_to_tk_vars(args)

    TkMain(root, vars)

    root.update_idletasks()
    root.mainloop()

def TkMain(root, vars):
    form = Form(root, vars)
    form.frame.grid(row=0, column=0, sticky="nsew", pady=(40, 0))

    notebook = ttk.Notebook(root)
    notebook.grid(row=0, column=1, sticky="nsew")

    button = ttk.Button(root, text="Menu", )
    button.place(relx=0, rely=1, anchor="sw", x=5, y=-5, width=245, height=30)

    def toggle_menu():
        if form.frame.winfo_viewable():
            form.frame.grid_forget()
            button.place_forget()
            button.place(relx=0, rely=1, anchor="sw", x=5, y=-5, width=50, height=30)
        else:
            form.frame.grid(row=0, column=0, sticky="nsew", pady=(40, 0))
            button.place_forget()
            button.place(relx=0, rely=1, anchor="sw", x=5, y=-5, width=245, height=30)

    def visualize_data():
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Visualization")

        fig, ax = plt.subplots(figsize=(5, 3))
        plot_cos(ax)

        # Embed Matplotlib figure in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    button.config(command=toggle_menu)

    form.group("Base Parameters") \
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
        .group("Action Buttons") \
            .button("Update Preview", visualize_data) \
            .button("Visualize Generated", lambda: print("Visualize Generated")) \
            .button("Generate & Save", lambda: print("Generate & Save")) \
            .button("Load Data...", lambda: print("Load Data...")) \
            .button("Gen, Save & Visualize", lambda: print(form.winfo_width())) \
        .done()
    

        # main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    # main.grid(row=0, column=0, sticky="nsew")

    # add padding top
    
    # add absolute positioned Button over the notebook

    # main.sashpos(0, form.winfo_width())

    # plot_frame.columnconfigure(0, weight=1)
    # plot_frame.rowconfigure(0, weight=0) # Toolbar row
    # plot_frame.rowconfigure(1, weight=1) # Canvas row

    # # Create Matplotlib Figure and Axes (start with 2D axes)
    # self.figure = Figure(figsize=(8, 6), dpi=100) # Adjust size as needed
    # # Create two subplots initially, assuming 2D plot is common
    # self.axes = [self.figure.add_subplot(121), self.figure.add_subplot(122)]

    # self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
    # self.canvas_widget = self.canvas.get_tk_widget()
    # self.canvas_widget.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    # # Add Navigation Toolbar
    # self.toolbar_frame = ttk.Frame(self.plot_frame)
    # self.toolbar_frame.grid(row=0, column=0, sticky="ew", padx=5)
    # self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
    # self.toolbar.update() # Initialize toolbar state

    # # Initial plot state (e.g., empty axes or instructions)
    # self._clear_plot()

    # # Adjust initial pane size (optional)




    #    base_param_names = ["n_samples", "n_features", "n_classes", "class_sep", "info_frac"]
#     for name in base_param_names + ["n_clients"]:
#         if name in self.vars: self.vars[name].trace_add("write", self._invalidate_preview_cache)

 # row_idx = 0
    # # Action Buttons Frame
    # action_frame = ttk.Frame(parent_frame, padding="10")
    # action_frame.grid(row=row_idx, column=0, pady=10)
    # row_idx += 1

    # self.update_preview_button = ttk.Button(action_frame, text="Update Preview", command=self._update_preview_callback)
    # self.update_preview_button.grid(row=0, column=0, padx=5, pady=2)

    # self.visualize_button = ttk.Button(action_frame, text="Visualize Generated", command=self._visualize_generated_callback, state=tk.DISABLED)
    # self.visualize_button.grid(row=0, column=1, padx=5, pady=2)

    # self.generate_button = ttk.Button(action_frame, text="Generate & Save", command=self._generate_data_callback)
    # self.generate_button.grid(row=1, column=0, padx=5, pady=2)

    # self.load_button = ttk.Button(action_frame, text="Load Data...", command=self._load_data_callback)
    # self.load_button.grid(row=1, column=1, padx=5, pady=2)

    # self.gen_vis_button = ttk.Button(action_frame, text="Gen, Save & Visualize", command=self._generate_and_visualize_generated_callback)
    # self.gen_vis_button.grid(row=2, column=0, columnspan=2, padx=5, pady=2)
    