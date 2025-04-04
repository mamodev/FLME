import sys
import matplotlib
import logging
import tkinter as tk


from gui.tkmain import run_gui

def use_backend(backend: str): 
    try:
        matplotlib.use('TkAgg')
        logging.info("Set Matplotlib backend to TkAgg for GUI.")
    except Exception as e:
            logging.warning(f"Could not set Matplotlib backend to TkAgg ({e}). "
                            "Trying default backend. Visualization might use a separate window or fail.")

def init_env(args):
    use_backend('TkAgg')
    try:
        run_gui(args)
    except tk.TclError as e:
        logging.error(f"Failed to initialize Tkinter GUI. Do you have a display environment set up? Error: {e}")
        print("Error: Could not initialize the graphical user interface.", file=sys.stderr)
        print("Try running without the --gui flag for the command-line version.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.exception(f"An unexpected error occurred while running the GUI: {e}")
        sys.exit(1)