"""Four-perspective tree explorer with all Bright Spot tiers merged.

The original visualize_evaluation_tree_perspectives.py remains unchanged.
This variant normalizes Prime/High/Normal/Low Bright labels to one Spot label.
"""

from __future__ import print_function

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import visualize_evaluation_tree_perspectives as tree_views
import visualize_spot_tree as workbook_reader


def merge_bright_labels(records):
    merged_records = []
    for record in records:
        normalized = dict(record)
        normalized["spot"] = workbook_reader.canonical_spot(record["spot"]) or record["spot"]
        merged_records.append(normalized)
    return merged_records


def main():
    dialog_root = tk.Tk()
    dialog_root.withdraw()
    dialog_root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select an image evaluation Excel workbook",
        filetypes=[("Excel workbooks", "*.xlsx"), ("All files", "*.*")],
    )
    dialog_root.destroy()
    if not path:
        return

    try:
        records, _trained_categories = workbook_reader.load_evaluation_data(path)
    except Exception as exc:
        error_root = tk.Tk()
        error_root.withdraw()
        messagebox.showerror("Could not read evaluation workbook", str(exc), parent=error_root)
        error_root.destroy()
        return

    app_root = tk.Tk()
    # Give this variant a distinct output prefix so it cannot overwrite the
    # artifacts created by the original four-perspective program.
    output_alias = os.path.splitext(path)[0] + "_bright_merged.xlsx"
    tree_views.TreePerspectiveApp(app_root, output_alias, merge_bright_labels(records))
    app_root.title("Evaluation Tree Perspectives — Bright Spot Merged")
    app_root.mainloop()


if __name__ == "__main__":
    main()
