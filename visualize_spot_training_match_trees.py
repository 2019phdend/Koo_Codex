"""Show missing-case trees in Spot-first and original perspectives.

The existing find_missing_tree_combinations.py is reused for workbook reading,
Spot grouping, rendering, and Excel reporting. This is a separate entry point
and writes artifacts under distinct filenames.
"""

from __future__ import print_function

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

import find_missing_tree_combinations as base
import visualize_spot_tree as workbook_reader


TREE_VIEWS = (
    ("Spot → Training Status → Prediction Matching", ("training", "match")),
    ("Spot → Prediction Matching → Training Status", ("match", "training")),
)
SPOT_GROUPS = base.SPOT_GROUPS


def build_trees(records):
    results = []
    for perspective, status_order in TREE_VIEWS:
        for group in SPOT_GROUPS:
            cases = base.spot_group_cases(records, group)
            is_bright_details = group == "Bright (Details)"
            if is_bright_details:
                dimensions = ("spot",) + status_order
                columns = ["Spot Group", "Bright Tier"] + [
                    "Training Status" if key == "training" else "Prediction Matching"
                    for key in status_order
                ]
            else:
                dimensions = status_order
                columns = ["Spot Group"] + [
                    "Training Status" if key == "training" else "Prediction Matching"
                    for key in status_order
                ]

            root_label = group
            paths = [
                ((root_label,) + tuple(case[key] for key in dimensions), case["count"])
                for case in cases
            ]
            title = "{} / {}".format(perspective, group)
            empty_count = sum(count == 0 for _path, count in paths)
            image = base.draw_tree(paths, columns, title, empty_count)
            results.append({
                "perspective": perspective,
                "group": group,
                "root": root_label,
                "columns": columns,
                "paths": paths,
                "image": image,
            })
    return results


class SpotFirstTreeApp:
    def __init__(self, root, source_path, records):
        self.root = root
        self.source_path = source_path
        self.records = records
        self.images = {}
        root.title("Spot and Status/Match Missing-Case Trees")
        root.geometry("1220x850")

        heading = ttk.Frame(root, padding=12)
        heading.pack(fill="x")
        ttk.Label(
            heading,
            text="Every Spot group includes all possible Training/Match combinations; "
                 "zero-sample leaves are marked in red.",
        ).pack(anchor="w")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        footer = ttk.Frame(root, padding=(12, 0, 12, 10))
        footer.pack(fill="x")
        self.status = ttk.Label(footer, text="Preparing trees…")
        self.status.pack(side="left")
        root.after_idle(self.safe_refresh)

    def safe_refresh(self):
        try:
            self.refresh()
        except Exception as exc:
            self.status.configure(text="Tree generation failed: {}".format(exc))
            messagebox.showerror("Could not display Spot-first trees", str(exc), parent=self.root)

    def add_tree_tab(self, notebook, graph):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=graph["group"])
        canvas = tk.Canvas(tab, background="#F6F8FB", highlightthickness=0)
        ybar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        xbar = ttk.Scrollbar(tab, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        photo = ImageTk.PhotoImage(graph["image"], master=canvas)
        self.images[id(graph)] = photo
        canvas.create_image(0, 0, anchor="nw", image=photo, tags=("tree-image",))
        canvas.configure(scrollregion=(0, 0, graph["image"].width, graph["image"].height))
        canvas.bind("<Map>", lambda _event, widget=canvas: self.root.after_idle(widget.update_idletasks))

    def refresh(self):
        for child in self.notebook.winfo_children():
            child.destroy()
        self.images = {}
        spot_first_graphs = build_trees(self.records)
        restored_graphs = [
            graph for graph in base.build_all_graphs(self.records)
            if graph["perspective"] != "Spot"
        ]
        graphs = spot_first_graphs + restored_graphs
        grouped = {}
        for graph in graphs:
            grouped.setdefault(graph["perspective"], []).append(graph)

        spot_first_perspectives = {name for name, _status_order in TREE_VIEWS}
        perspective_order = [name for name, _order in TREE_VIEWS] + [
            name for name in base.PERSPECTIVES if name != "Spot"
        ]
        for perspective in perspective_order:
            outer = ttk.Frame(self.notebook)
            self.notebook.add(outer, text=perspective)
            inner = ttk.Notebook(outer)
            inner.pack(fill="both", expand=True)
            for graph in grouped[perspective]:
                self.add_tree_tab(inner, graph)

        prefix = os.path.splitext(self.source_path)[0]
        for graph in graphs:
            key = re.sub(r"[^A-Za-z0-9_-]+", "_", "{}_{}".format(
                graph["perspective"], graph["group"]
            )).strip("_").lower()
            output_prefix = (
                "spot_first_" if graph["perspective"] in spot_first_perspectives else "restored_"
            )
            graph["image"].save(prefix + "_" + output_prefix + key + ".png")

        report_path = prefix + "_spot_training_match_missing_combinations.xlsx"
        base.save_report(report_path, graphs)
        empty_count = sum(sum(count == 0 for _path, count in graph["paths"]) for graph in graphs)
        self.status.configure(text="{} trees | {} zero-sample leaves | {}".format(
            len(graphs), empty_count, os.path.basename(report_path)
        ))


def main():
    # Reuse one Tk root for the file picker and app window on macOS.
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select an image evaluation Excel workbook",
        filetypes=[("Excel workbooks", "*.xlsx"), ("All files", "*.*")],
        parent=root,
    )
    if not path:
        root.destroy()
        return
    try:
        records, _trained_categories = workbook_reader.load_evaluation_data(path)
    except Exception as exc:
        messagebox.showerror("Could not read evaluation workbook", str(exc), parent=root)
        root.destroy()
        return

    root.attributes("-topmost", False)
    root.deiconify()
    try:
        SpotFirstTreeApp(root, path, records)
    except Exception as exc:
        messagebox.showerror("Could not create tree window", str(exc), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
