"""Interactive four-perspective tree explorer for image evaluation workbooks.

This is an additional utility. It reuses the workbook reader from
visualize_spot_tree.py and does not modify that program.
"""

from __future__ import print_function

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from PIL import Image, ImageDraw, ImageFont, ImageTk

import visualize_spot_tree as source_reader


PERSPECTIVES = (
    ("Spot", ("spot", "training", "match", "category")),
    ("Test Category", ("category", "spot", "training", "match")),
    ("Training Status", ("training", "spot", "match", "category")),
    ("Prediction Match", ("match", "spot", "training", "category")),
)
DIMENSION_NAMES = {
    "spot": "Spot",
    "training": "Training Status",
    "match": "Prediction Match",
    "category": "Test Category",
}
DIMENSION_FILTER_ORDER = ("spot", "training", "match", "category")
DIMENSION_COLORS = ("#E8F6EF", "#EAF0FA", "#FFF4E5", "#F0ECFA")
IMAGE_WIDTH = 1120
NODE_WIDTH = 240
ROW_GAP = 42


def get_fonts():
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(path):
            try:
                return (
                    ImageFont.truetype(path, 14),
                    ImageFont.truetype(path, 12),
                    ImageFont.truetype(path, 20),
                )
            except OSError:
                pass
    font = ImageFont.load_default()
    return font, font, font


def filtered_records(records, selected_values):
    """An empty selection for any dimension means include every value."""
    result = []
    for record in records:
        if all(
            not selected_values.get(dimension)
            or record[dimension] in selected_values[dimension]
            for dimension in DIMENSION_FILTER_ORDER
        ):
            result.append(record)
    return result


def collect_paths(records, dimensions):
    counts = {}
    for record in records:
        path = tuple(record[dimension] for dimension in dimensions)
        counts[path] = counts.get(path, 0) + 1
    return counts


def draw_perspective(records, perspective, dimensions):
    paths = collect_paths(records, dimensions)
    leaves = sorted(paths.items(), key=lambda item: tuple(v.casefold() for v in item[0]))
    height = max(410, 125 + ROW_GAP * max(1, len(leaves)))
    image = Image.new("RGB", (IMAGE_WIDTH, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    font, small_font, title_font = get_fonts()
    draw.text((22, 16), "{} PERSPECTIVE  |  {} samples".format(perspective.upper(), len(records)),
              fill="#1F3E5B", font=title_font)

    margin = 20
    column_gap = (IMAGE_WIDTH - 2 * margin - 4 * NODE_WIDTH) / 3.0
    xs = [int(margin + index * (NODE_WIDTH + column_gap)) for index in range(4)]
    for index, dimension in enumerate(dimensions):
        draw.text((xs[index], 80), DIMENSION_NAMES[dimension].upper(),
                  fill="#526174", font=small_font)

    if not leaves:
        draw.rounded_rectangle((22, 170, 280, 220), radius=10,
                               fill="#FFFFFF", outline="#CBD3DD", width=1)
        draw.text((36, 188), "No samples match these filters", fill="#526174", font=font)
        return image, paths

    centers = {}
    totals = {}
    for index, (path, count) in enumerate(leaves):
        y = 112 + index * ROW_GAP
        for level in range(4):
            prefix = path[:level + 1]
            centers.setdefault((level, prefix), []).append(y)
            totals[(level, prefix)] = totals.get((level, prefix), 0) + count
    centers = {key: sum(values) / float(len(values)) for key, values in centers.items()}

    for level, path in sorted(centers, key=lambda item: (item[0], item[1])):
        if level == 0:
            continue
        parent_center = centers[(level - 1, path[:-1])]
        child_center = centers[(level, path)]
        x1 = xs[level - 1] + NODE_WIDTH
        x2 = xs[level]
        elbow = x1 + max(10, (x2 - x1) // 2)
        draw.line([(x1, parent_center), (elbow, parent_center),
                   (elbow, child_center), (x2, child_center)],
                  fill="#A9B4C2", width=2)

    for (level, prefix), center_y in centers.items():
        x1, x2 = xs[level], xs[level] + NODE_WIDTH
        y1, y2 = center_y - 17, center_y + 17
        draw.rounded_rectangle(
            (x1, y1, x2, y2), radius=8,
            fill=DIMENSION_COLORS[level], outline="#A9B4C2", width=1,
        )
        label = prefix[-1]
        max_text_width = NODE_WIDTH - 70
        while draw.textbbox((0, 0), label, font=small_font)[2] > max_text_width and len(label) > 6:
            label = label[:-4] + "..."
        draw.text((x1 + 8, y1 + 10), "{} ({})".format(label, totals[(level, prefix)]),
                  fill="#263E5B", font=small_font)
    return image, paths


def save_perspective_excel(path, visible_graphs, filtered, selected_values):
    workbook = Workbook()
    summary = workbook.active
    summary.title = "Filters"
    summary.append(["Dimension", "Selected values (blank = all)"])
    for dimension in DIMENSION_FILTER_ORDER:
        values = sorted(selected_values.get(dimension, set()), key=str.casefold)
        summary.append([DIMENSION_NAMES[dimension], ", ".join(values) if values else "ALL"])
    summary.append(["Filtered sample count", len(filtered)])

    for perspective, dimensions, image, paths in visible_graphs:
        sheet_name = perspective[:31]
        sheet = workbook.create_sheet(sheet_name)
        sheet.append([DIMENSION_NAMES[dimension] for dimension in dimensions] + ["Sample count"])
        for values, count in sorted(paths.items(), key=lambda item: tuple(v.casefold() for v in item[0])):
            sheet.append(list(values) + [count])

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(name="Arial", bold=True, color="FFFFFF")
            cell.fill = PatternFill(fill_type="solid", fgColor="315C7D")
        for column in sheet.columns:
            lengths = [len(str(cell.value or "")) for cell in column]
            sheet.column_dimensions[column[0].column_letter].width = min(max(max(lengths + [12]) + 2, 14), 40)
    workbook.save(path)


class TreePerspectiveApp:
    def __init__(self, root, source_path, records):
        self.root = root
        self.source_path = source_path
        self.records = records
        self.values = {
            dimension: sorted({record[dimension] for record in records}, key=str.casefold)
            for dimension in DIMENSION_FILTER_ORDER
        }
        self.graph_vars = {}
        self.filter_boxes = {}
        self.images = {}
        self.notebook = None

        root.title("Evaluation Tree Perspectives")
        root.geometry("1240x900")

        header = ttk.Frame(root, padding=(12, 10, 12, 4))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Select one or more values per dimension. Leave a list unselected to include all values.",
        ).pack(anchor="w")

        filters = ttk.Frame(root, padding=(12, 6))
        filters.pack(fill="x")
        filter_titles = {
            "spot": "Spot",
            "training": "Training Status",
            "match": "Prediction Match",
            "category": "Test Category",
        }
        for column, dimension in enumerate(DIMENSION_FILTER_ORDER):
            frame = ttk.LabelFrame(filters, text=filter_titles[dimension], padding=6)
            frame.grid(row=0, column=column, sticky="nsew", padx=4)
            # Tk's MULTIPLE mode toggles each row on every ordinary click,
            # while retaining other selected rows for multi-value filters.
            box = tk.Listbox(frame, selectmode="multiple", height=5, exportselection=False)
            scroll = ttk.Scrollbar(frame, orient="vertical", command=box.yview)
            box.configure(yscrollcommand=scroll.set)
            for value in self.values[dimension]:
                box.insert("end", value)
            box.grid(row=0, column=0, sticky="nsew")
            scroll.grid(row=0, column=1, sticky="ns")
            buttons = ttk.Frame(frame)
            buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
            ttk.Button(
                buttons, text="Select All",
                command=lambda d=dimension, b=box: self.select_all_filter(d, b),
            ).pack(side="left", expand=True, fill="x", padx=(0, 3))
            ttk.Button(
                buttons, text="Clear",
                command=lambda d=dimension, b=box: self.clear_filter(d, b),
            ).pack(side="left", expand=True, fill="x", padx=(3, 0))
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            self.filter_boxes[dimension] = box
            filters.columnconfigure(column, weight=1)

        view_frame = ttk.LabelFrame(root, text="Tree perspectives to show", padding=8)
        view_frame.pack(fill="x", padx=16, pady=(0, 8))
        for index, (perspective, _dimensions) in enumerate(PERSPECTIVES):
            variable = tk.BooleanVar(value=True)
            self.graph_vars[perspective] = variable
            ttk.Checkbutton(view_frame, text=perspective, variable=variable,
                            command=self.refresh).grid(row=0, column=index, sticky="w", padx=8)

        actions = ttk.Frame(root, padding=(12, 0, 12, 8))
        actions.pack(fill="x")
        ttk.Button(actions, text="Update trees", command=self.refresh).pack(side="left")
        self.status = ttk.Label(actions, text="")
        self.status.pack(side="left", padx=12)

        self.views = ttk.Frame(root)
        self.views.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.refresh()

    def selected_filter_values(self):
        selected = {}
        for dimension, box in self.filter_boxes.items():
            indexes = box.curselection()
            selected[dimension] = {box.get(index) for index in indexes}
        return selected

    def select_all_filter(self, dimension, box):
        box.selection_set(0, tk.END)
        self.refresh()

    def clear_filter(self, dimension, box):
        box.selection_clear(0, tk.END)
        self.refresh()

    def refresh(self):
        selected_values = self.selected_filter_values()
        filtered = filtered_records(self.records, selected_values)
        selected_perspectives = [
            (name, dimensions) for name, dimensions in PERSPECTIVES
            if self.graph_vars[name].get()
        ]

        for child in self.views.winfo_children():
            child.destroy()
        self.images = {}
        if not selected_perspectives:
            self.notebook = None
            ttk.Label(self.views, text="Select at least one tree perspective to display.").pack(pady=24)
            self.status.configure(text="{} filtered samples".format(len(filtered)))
            return

        self.notebook = ttk.Notebook(self.views)
        self.notebook.pack(fill="both", expand=True)
        visible_graphs = []
        for perspective, dimensions in selected_perspectives:
            image, paths = draw_perspective(filtered, perspective, dimensions)
            visible_graphs.append((perspective, dimensions, image, paths))
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=perspective)
            canvas = tk.Canvas(tab, background="#F6F8FB", highlightthickness=0)
            ybar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
            xbar = ttk.Scrollbar(tab, orient="horizontal", command=canvas.xview)
            canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
            canvas.grid(row=0, column=0, sticky="nsew")
            ybar.grid(row=0, column=1, sticky="ns")
            xbar.grid(row=1, column=0, sticky="ew")
            tab.rowconfigure(0, weight=1)
            tab.columnconfigure(0, weight=1)
            photo = ImageTk.PhotoImage(image)
            self.images[perspective] = photo
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.configure(scrollregion=(0, 0, image.width, image.height))

            output_base = os.path.splitext(self.source_path)[0] + "_tree_" + perspective.lower().replace(" ", "_")
            image.save(output_base + ".png")

        excel_path = os.path.splitext(self.source_path)[0] + "_tree_perspectives.xlsx"
        save_perspective_excel(excel_path, visible_graphs, filtered, selected_values)
        self.status.configure(text="{} samples | {} tree(s) | {}".format(
            len(filtered), len(selected_perspectives), os.path.basename(excel_path)
        ))


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
        records, _trained_categories = source_reader.load_evaluation_data(path)
    except Exception as exc:
        error_root = tk.Tk()
        error_root.withdraw()
        messagebox.showerror("Could not read evaluation workbook", str(exc), parent=error_root)
        error_root.destroy()
        return

    root = tk.Tk()
    TreePerspectiveApp(root, path, records)
    root.mainloop()


if __name__ == "__main__":
    main()
