import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tkinter import messagebox

from components import Form, CustomNotebook

from typing import Dict
import numpy as np

class Dataset:  
    def get_form(parent):
        return

    def generate(vars):
        return



def run_gui(datasets: Dict[str, Dataset]):
    root = tk.Tk()

    root.title("Synthetic Data Generator")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    datasetsOptions = list(datasets.keys())
    assert len(datasetsOptions) > 0, "No datasets available"

    vars = {
        "dataset": tk.StringVar(value=datasetsOptions[0]),
    }


    sidebar = ttk.Frame(root, width=250)
    sidebar.grid(row=0, column=0, sticky="nsew")

    form = Form(sidebar, vars).group("General") \
        .select("Dataset:", "dataset", datasetsOptions) \
        .done()
    
    form.frame.grid(row=0, column=0, sticky="nsew")


    notebook = tk.Frame(root)
    notebook.grid(row=0, column=1, sticky="nsew")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    canvas = FigureCanvasTkAgg(fig, master=notebook)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

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

    button.config(command=toggle_menu)

    prev_frame = None
    prev_form = None
    def on_dataset_change(*args):
        nonlocal prev_frame, prev_form

        selected_dataset = vars["dataset"].get()

        if not selected_dataset in datasets:
            messagebox.showerror("Error", "Invalid dataset selected.")
            return

        if prev_frame is not None:
            prev_frame.grid_forget()
            prev_frame = None

        dataset_class = datasets[selected_dataset]
        form = dataset_class.get_form(sidebar)

        ds = None
        def generate_data():
            nonlocal ds
            ds = dataset_class.generate(form.vars)

        def visualize():
            generate_data()

            if ds is None:
                return   

            dataset_class.visualize(fig, ax, form.vars["vis_method"].get(), ds[0], ds[1])               
            
        form.group("Actions") \
            .button("Generate", generate_data) \
            .button("Visualize", visualize).done()
        
        prev_frame = form.frame
        prev_form = form

        form.frame.grid(row=1, column=0, sticky="nsew")

    vars["dataset"].trace_add("write", on_dataset_change)
    
    def on_closing():
        try:
            plt.close(fig)
        except Exception as e:
            print(f"   Error closing Matplotlib figure: {e}") # Log error but continue

        root.destroy()
        print("   Tkinter root destroyed.")

    root.protocol("WM_DELETE_WINDOW", on_closing) # Use the custom cleanup function
    root.update_idletasks()
    on_dataset_change()
    root.mainloop()
   
    # def visualize_data():
    #     frame = ttk.Frame(notebook)
    #     notebook.add(frame, text="Visualization")

    #     fig, ax = plt.subplots(figsize=(5, 3))

    #     # Embed Matplotlib figure in Tkinter
    #     canvas = FigureCanvasTkAgg(fig, master=frame)
    #     canvas.draw()
    #     canvas.get_tk_widget().pack(fill="both", expand=True)

    # form.group("Base Parameters") \
    #         .input("Total Samples:", "n_samples", int) \
    #         .input("Num Features:", "n_features", int) \
    #         .input("Num Classes:", "n_classes", int) \
    #         .input("Num Clients:", "n_clients", int) \
    #     .group("Skew Parameters") \
    #         .input("Qty Skew α (>0):", "quantity_skew_alpha", float) \
    #         .input("Lbl Skew α (>0):", "label_skew_alpha", float) \
    #         .input("Feat Skew Lvl:", "feature_skew_level", float) \
    #         .input("Drift Lvl:", "concept_drift_level", float) \
    #         .input("Shift Lvl (0-1):", "concept_shift_level", float) \
    #     .group("Output & Visualization") \
    #         .input("Output Dir:", "output_dir", str) \
    #         .input("Vis Samples:", "vis_samples", int) \
    #         .select("Vis Method:", "vis_method", ['pca', 'tsne']) \
    #     .group("Action Buttons") \
    #         .button("Update Preview", visualize_data) \
    #         .button("Visualize Generated", lambda: print("Visualize Generated")) \
    #         .button("Generate & Save", lambda: print("Generate & Save")) \
    #         .button("Load Data...", lambda: print("Load Data...")) \
    #         .button("Gen, Save & Visualize", lambda: print(form.winfo_width())) \
    #     .done()