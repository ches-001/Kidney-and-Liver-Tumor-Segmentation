import warnings
warnings.filterwarnings("ignore")

import os
import tkinter as tk
from tkinter import filedialog
    
def shorten_string(input_string: str, max_length: int)->str:
    if len(input_string) <= max_length:
        return input_string

    prefix_length = (max_length - 3) // 2
    suffix_length = max_length - 3 - prefix_length

    shortened_string = f"{input_string[:prefix_length]}...{input_string[-suffix_length:]}"
    return shortened_string


def on_ask_ct_file():
    ct_file = filedialog.askopenfilename()
    if ct_file:
        ct_str.set(ct_file)
        ct_field_label.config(text=shorten_string(ct_str.get(), 30))


def on_ask_target_file():
    target_file = filedialog.askopenfilename()
    if target_file:
        target_str.set(target_file)
        target_field_label.config(text=shorten_string(target_str.get(), 30))


def on_click():
    submit_button.config(state=tk.DISABLED)
    ask_ct_button.config(state=tk.DISABLED)
    ask_target_button.config(state=tk.DISABLED)

    cmd_str = f'''
        start /B start cmd /k python evaluate.py \
        "{ct_str.get()}" \
        {selected_model_type_option.get()} \
        {selected_slice_dim_option.get()} \
        --slice_idx={slice_idx_str.get()} \
        --segment_path="{target_str.get()}"
    '''
    os.system(cmd_str)


if __name__ == "__main__":
    window = tk.Tk()
    window.title("Tumor Volume and Slice Segmentation GUI")
    window.geometry("410x500")

    # Create labels for the entry fields
    ct_str = tk.StringVar()
    ct_field_label = tk.Label(window)
    ask_ct_button = tk.Button(window, text="Select CT volume Path", command=on_ask_ct_file)

    target_str = tk.StringVar()
    target_field_label = tk.Label(window)
    ask_target_button = tk.Button(window, text="Select Output Path", command=on_ask_target_file)

    model_type_label = tk.Label(window, text="Select Model Type:")
    model_type_options = ["kidney", "liver"]
    selected_model_type_option = tk.StringVar(window)
    selected_model_type_option.set(model_type_options[0]) 
    model_type_option_menu = tk.OptionMenu(window, selected_model_type_option, *model_type_options, command=None)
    
    slice_dim_label = tk.Label(window, text="Select Slice Dimension / Axis:")
    slice_dim_options = ["first", "last"]
    selected_slice_dim_option = tk.StringVar(window)
    selected_slice_dim_option.set(slice_dim_options[0]) 
    slice_dim_option_menu = tk.OptionMenu(window, selected_slice_dim_option, *slice_dim_options, command=None)

    slice_idx_str = tk.StringVar(window)
    slice_idx_label = tk.Label(window, text="Slice Index (used for visualising only 2d slices):")
    slice_idx_entry = tk.Entry(window, textvariable=slice_idx_str)


    submit_button = tk.Button(window, text="Enter", command=on_click, borderwidth=1)

    ncols = 2; nrows = 10
    window.columnconfigure(ncols, weight=1)
    window.rowconfigure(nrows, weight=1)

    # Place the widgets on the window using the grid layout manager
    ask_ct_button.grid(row=0, column=0, columnspan=ncols, padx=5, pady=5, sticky="ew")
    ct_field_label.grid(row=1, column=0, columnspan=ncols, padx=5, pady=5, sticky="ew")

    ask_target_button.grid(row=2, column=0, columnspan=ncols, padx=5, pady=5, sticky="ew")
    target_field_label.grid(row=3, column=0, columnspan=ncols, padx=5, pady=5, sticky="ew")

    model_type_label.grid(row=4, column=0, padx=10, pady=0, sticky="ew")
    model_type_option_menu.grid(row=4, column=1, padx=2, pady=5, sticky="ew")

    slice_dim_label.grid(row=5, column=0, padx=10, pady=0, sticky="ew")
    slice_dim_option_menu.grid(row=5, column=1, padx=2, pady=5, sticky="ew")

    slice_idx_label.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
    slice_idx_entry.grid(row=6, column=1, padx=2, pady=5, sticky="ew")

    submit_button.grid(row=8, column=0, columnspan=ncols, padx=10, pady=20, sticky="ew")

    window.mainloop()