
import tkinter as tk
from tkinter import ttk

from gui.validators import validate_int, validate_float


class Form:
    def __init__(self, parent, vars, **kwargs):
        self.vars = vars
        self.frame = ttk.Frame(parent, **kwargs)
        self.rows = 0

    def group(self, title):
        self.rows += 1
        return FormGroup(self, self.vars, title=title, row=self.rows - 1)
    
    def winfo_width(self):
        return self.frame.winfo_width()
    
    def winfo_height(self):
        return self.frame.winfo_height()

class FormGroup:
    def __init__(self, parent, vars, title="vars", row=0):
        self.vars = vars
        self._components = []
        self.parent = parent
        
        self.root = ttk.LabelFrame(parent.frame, text=title, padding="10")
        self.root.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        self.root.columnconfigure(1, weight=1)

    def input(self, label, var_name, input_type=str):
        self._components.append((TkInput, (self.root, label, self.vars[var_name], len(self._components) + 1, input_type)))
        return self

    def select(self, label, var_name, choices):
        self._components.append((TkSelect, (self.root, label, self.vars[var_name], choices, len(self._components) + 1)))
        return self
    
    def button(self, label, command):
        self._components.append((TkButton, (self.root, label, command, len(self._components) + 1)))
        return self

    def group(self, title="Group") -> 'FormGroup':
        return self.done().group(title)

    def done(self):
        for component, args in self._components:
            component(*args)

        return self.parent
    
def TkButton(parent, label_text, command, row):
    button = ttk.Button(parent, text=label_text, command=command)
    button.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=2)
    return button

def TkInput(parent, label_text, var, row, input_type=str):
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
    vcmd = None
    if input_type is int: vcmd = (parent.register(validate_int), '%P')
    elif input_type is float: vcmd = (parent.register(validate_float), '%P')
    entry = ttk.Entry(parent, textvariable=var, width=18, validate='key', validatecommand=vcmd)
    entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

def TkSelect(parent, label_text, var, choices, row):
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
    combo = ttk.Combobox(parent, textvariable=var, values=choices, state='readonly', width=16) # Slightly narrower
    combo.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)