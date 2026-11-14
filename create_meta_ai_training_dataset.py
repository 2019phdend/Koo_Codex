"""Create seven X1-X4 to Y-label datasets and a selectable class dashboard.

The source is an evaluation Excel workbook containing X1-X4 and three Y
components. Each output dataset contains only the selected target component(s)
so labels from other targets do not leak into that training sheet.
"""

from __future__ import print_function

import os
import re
import tkinter as tk
from collections import Counter
from tkinter import filedialog, messagebox, ttk

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from PIL import Image, ImageDraw, ImageFont, ImageTk


FEATURES = ("X1", "X2", "X3", "X4")
TARGETS = ("Y1", "Y2", "Y3")
TARGET_MODES = (
    ("Y1", ("Y1",)),
    ("Y2", ("Y2",)),
    ("Y3", ("Y3",)),
    ("Y1_Y2", ("Y1", "Y2")),
    ("Y2_Y3", ("Y2", "Y3")),
    ("Y3_Y1", ("Y3", "Y1")),
    ("Y1_Y2_Y3", ("Y1", "Y2", "Y3")),
)
OUTPUT_COLUMNS = ("Sample ID", "Image path", "Test category")
CHART_WIDTH = 500
CHART_HEIGHT = 330
MAX_VISIBLE_CLASSES = 9


def norm(value):
    return "" if value is None else str(value).strip()


def header_key(value):
    return re.sub(r"[^a-z0-9]", "", norm(value).casefold())


def find_column(headers, candidates):
    keys = [header_key(value) for value in headers]
    for candidate in candidates:
        key = header_key(candidate)
        if key in keys:
            return keys.index(key)
    return None


def locate_data_sheet(workbook):
    """Find the sheet and header row containing four X and three Y columns."""
    y_aliases = {
        "Y1": ("Y1", "Y1 Trained status", "Y Trained status", "Trained status"),
        "Y2": ("Y2", "Y2 Match status", "Y Match status", "Match status"),
        "Y3": ("Y3", "Y3 Spot", "Y Spot", "Spot"),
    }
    for sheet in workbook.worksheets:
        for header_row in range(1, min(sheet.max_row, 30) + 1):
            headers = [cell.value for cell in sheet[header_row]]
            feature_columns = {
                name: find_column(headers, (name, name + " score", name + " confidence"))
                for name in FEATURES
            }
            y_columns = {
                name: find_column(headers, y_aliases[name]) for name in TARGETS
            }
            if all(value is not None for value in feature_columns.values()) and all(
                value is not None for value in y_columns.values()
            ):
                image_col = find_column(headers, ("Image path", "Image Path", "Relative path", "File name"))
                category_col = find_column(headers, ("Test category", "Folder label", "Category"))
                return sheet, header_row, feature_columns, y_columns, image_col, category_col, headers
    raise ValueError(
        "Could not find a sheet with X1-X4 and three target columns. "
        "Accepted target headers are Y1/Y2/Y3 or Y Trained status/Y Match status/Y Spot."
    )


def parse_feature(cell):
    """Return a numeric feature, expressing percentage-formatted data as 0..100."""
    value = cell.value
    if value is None or norm(value) == "":
        raise ValueError("missing feature")
    is_percent_string = isinstance(value, str) and "%" in value
    text = norm(value).replace(",", "").replace("%", "")
    number = float(text)
    if is_percent_string:
        return number
    if isinstance(value, (int, float)) and "%" in (cell.number_format or ""):
        return number * 100.0
    return number


def read_source(path):
    workbook = load_workbook(path, read_only=True, data_only=True)
    located = locate_data_sheet(workbook)
    sheet, header_row, feature_cols, target_cols, image_col, category_col, headers = located
    records = []
    skipped = []
    y_source_headers = {
        target: norm(headers[column]) for target, column in target_cols.items()
    }

    for row_number, cells in enumerate(
        sheet.iter_rows(min_row=header_row + 1), start=header_row + 1
    ):
        if not any(cell.value is not None for cell in cells):
            continue
        try:
            features = {name: parse_feature(cells[column]) for name, column in feature_cols.items()}
        except (TypeError, ValueError, OverflowError):
            skipped.append({"Source row": row_number, "Reason": "Missing or non-numeric X1-X4"})
            continue

        labels = {name: norm(cells[column].value) for name, column in target_cols.items()}
        if any(not labels[name] for name in TARGETS):
            missing = ", ".join(name for name in TARGETS if not labels[name])
            skipped.append({"Source row": row_number, "Reason": "Missing target value: " + missing})
            continue

        records.append({
            "source_row": row_number,
            "image_path": norm(cells[image_col].value) if image_col is not None else "",
            "test_category": norm(cells[category_col].value) if category_col is not None else "",
            "features": features,
            "labels": labels,
        })
    workbook.close()
    if not records:
        raise ValueError("No rows with complete X1-X4 and Y1-Y3 values were found.")
    return records, skipped, y_source_headers, sheet.title, header_row


def target_class(labels, components):
    return " | ".join(labels[name] for name in components)


def summarize_modes(records):
    results = {}
    for mode, components in TARGET_MODES:
        counts = Counter(target_class(record["labels"], components) for record in records)
        results[mode] = {
            "components": components,
            "counts": counts,
            "samples": len(records),
        }
    return results


def make_output_rows(records, components):
    rows = []
    for record in records:
        row = {
            "Sample ID": record["source_row"],
            "Image path": record["image_path"],
            "Test category": record["test_category"],
        }
        row.update(record["features"])
        for index, target in enumerate(components, start=1):
            row["Y{}".format(index)] = record["labels"][target]
        row["Target Class"] = target_class(record["labels"], components)
        rows.append(row)
    return rows


def style_sheet(sheet):
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(name="Arial", bold=True, color="FFFFFF")
        cell.fill = PatternFill(fill_type="solid", fgColor="315C7D")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.row_dimensions[1].height = 30
    for column in sheet.columns:
        values = [len(str(cell.value or "")) for cell in column[:150]]
        letter = column[0].column_letter
        sheet.column_dimensions[letter].width = min(max(max(values + [12]) + 2, 12), 42)
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = "0.00"


def write_training_workbook(path, records, skipped, y_source_headers, source_sheet, header_row):
    workbook = Workbook()
    overview = workbook.active
    overview.title = "Dataset Summary"
    overview.append(["Meta AI training dataset", "Details"])
    overview.append(["Source sheet", source_sheet])
    overview.append(["Header row", header_row])
    overview.append(["Valid samples", len(records)])
    overview.append(["Skipped source rows", len(skipped)])
    overview.append(["Input features", ", ".join(FEATURES)])
    overview.append(["Y1 source column", y_source_headers["Y1"]])
    overview.append(["Y2 source column", y_source_headers["Y2"]])
    overview.append(["Y3 source column", y_source_headers["Y3"]])
    overview.append([])
    overview.append(["Dataset mode", "Target components", "Samples", "Classes"])

    distribution = workbook.create_sheet("Class Distribution")
    distribution.append(["Dataset mode", "Target class", "Y1", "Y2", "Y3", "Samples", "Share (%)"])

    for mode, components in TARGET_MODES:
        rows = make_output_rows(records, components)
        sheet = workbook.create_sheet(mode)
        component_columns = ["Y{}".format(index) for index in range(1, len(components) + 1)]
        headers = list(OUTPUT_COLUMNS) + list(FEATURES) + component_columns + ["Target Class"]
        sheet.append(headers)
        for row in rows:
            sheet.append([row.get(header, "") for header in headers])
        style_sheet(sheet)

        counts = Counter(row["Target Class"] for row in rows)
        overview.append([mode, " + ".join(components), len(rows), len(counts)])
        for class_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold())):
            parts = class_name.split(" | ")
            y_values = parts + [""] * (3 - len(parts))
            distribution.append([
                mode, class_name, y_values[0], y_values[1], y_values[2],
                count, count / float(len(rows)) * 100.0,
            ])

    skipped_sheet = workbook.create_sheet("Skipped Rows")
    skipped_sheet.append(["Source row", "Reason"])
    for item in skipped:
        skipped_sheet.append([item["Source row"], item["Reason"]])

    for sheet in (overview, distribution, skipped_sheet):
        style_sheet(sheet)
    distribution.column_dimensions["B"].width = 42
    distribution.column_dimensions["G"].width = 15
    overview.column_dimensions["A"].width = 26
    overview.column_dimensions["B"].width = 34
    skipped_sheet.column_dimensions["B"].width = 42
    workbook.save(path)


def load_fonts():
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, 14), ImageFont.truetype(path, 12), ImageFont.truetype(path, 19)
            except OSError:
                pass
    font = ImageFont.load_default()
    return font, font, font


def draw_class_chart(mode, summary):
    image = Image.new("RGB", (CHART_WIDTH, CHART_HEIGHT), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    font, small_font, title_font = load_fonts()
    counts = summary[mode]["counts"]
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))
    visible = ranked[:MAX_VISIBLE_CLASSES]
    remainder = sum(count for _label, count in ranked[MAX_VISIBLE_CLASSES:])
    if remainder:
        visible.append(("Other classes ({})".format(len(ranked) - MAX_VISIBLE_CLASSES), remainder))

    draw.text((18, 14), "{} CLASS DISTRIBUTION".format(mode.replace("_", " + ")),
              fill="#1F3E5B", font=title_font)
    draw.text((18, 43), "{} samples  |  {} classes".format(summary[mode]["samples"], len(counts)),
              fill="#526174", font=small_font)
    label_x, bar_x = 20, 195
    bar_width, top_y, row_height = 230, 78, 24
    max_count = max([count for _label, count in visible] or [1])
    for index, (label, count) in enumerate(visible):
        y = top_y + index * row_height
        display_label = label
        while draw.textbbox((0, 0), display_label, font=small_font)[2] > 165 and len(display_label) > 8:
            display_label = display_label[:-4] + "..."
        draw.text((label_x, y + 3), display_label, fill="#263E5B", font=small_font)
        draw.rounded_rectangle((bar_x, y, bar_x + bar_width, y + 18), radius=5, fill="#E3E8EF")
        bar_len = int(bar_width * count / max_count)
        draw.rounded_rectangle((bar_x, y, bar_x + bar_len, y + 18), radius=5, fill="#4C8A72")
        draw.text((bar_x + bar_width + 8, y + 2), str(count), fill="#263E5B", font=small_font)
    return image


class DatasetDashboard:
    def __init__(self, root, output_path, summary):
        self.root = root
        self.output_path = output_path
        self.summary = summary
        self.images = {}
        root.title("Meta AI Dataset Class Dashboard")
        root.geometry("1320x900")

        toolbar = ttk.Frame(root, padding=10)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Select graph(s) to display. Clear selection hides all charts.").pack(anchor="w")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        controls = ttk.LabelFrame(body, text="Target label combinations", padding=8)
        controls.pack(side="left", fill="y", padx=(0, 10))
        self.listbox = tk.Listbox(controls, selectmode="multiple", height=12, width=22, exportselection=False)
        for mode, _components in TARGET_MODES:
            self.listbox.insert("end", mode.replace("_", " + "))
        self.listbox.pack(fill="y", expand=True)
        self.listbox.selection_set(0, tk.END)
        ttk.Button(controls, text="Select All", command=self.select_all).pack(fill="x", pady=(8, 3))
        ttk.Button(controls, text="Clear", command=self.clear).pack(fill="x", pady=3)
        ttk.Button(controls, text="Show Selected", command=self.refresh).pack(fill="x", pady=(3, 0))

        chart_area = ttk.Frame(body)
        chart_area.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(chart_area, background="#E9EEF3", highlightthickness=0)
        scrollbar = ttk.Scrollbar(chart_area, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.window_id, width=event.width))

        footer = ttk.Frame(root, padding=(10, 4, 10, 10))
        footer.pack(fill="x")
        self.status = ttk.Label(footer, text="Dataset: {}".format(os.path.basename(output_path)))
        self.status.pack(anchor="w")
        root.after_idle(self.safe_refresh)

    def safe_refresh(self):
        try:
            self.refresh()
        except Exception as exc:
            self.status.configure(text="Dashboard error: {}".format(exc))
            messagebox.showerror("Could not display dashboard", str(exc), parent=self.root)

    def selected_modes(self):
        return [TARGET_MODES[index][0] for index in self.listbox.curselection()]

    def select_all(self):
        self.listbox.selection_set(0, tk.END)
        self.refresh()

    def clear(self):
        self.listbox.selection_clear(0, tk.END)
        self.refresh()

    def refresh(self):
        for child in self.inner.winfo_children():
            child.destroy()
        self.images = {}
        modes = self.selected_modes()
        if not modes:
            ttk.Label(self.inner, text="No graphs selected.", padding=20).grid(row=0, column=0, sticky="w")
            self.status.configure(text="No charts displayed. Dataset: {}".format(os.path.basename(self.output_path)))
            return

        chart_dir = os.path.splitext(self.output_path)[0] + "_charts"
        os.makedirs(chart_dir, exist_ok=True)
        for index, mode in enumerate(modes):
            image = draw_class_chart(mode, self.summary)
            image.save(os.path.join(chart_dir, mode.lower() + "_class_distribution.png"))
            photo = ImageTk.PhotoImage(image, master=self.canvas)
            self.images[mode] = photo
            card = ttk.LabelFrame(self.inner, text=mode.replace("_", " + "), padding=4)
            card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=5, pady=5)
            ttk.Label(card, image=photo).pack()
        for column in range(2):
            self.inner.columnconfigure(column, weight=1)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.status.configure(text="Showing {} of 7 graph(s) | Dataset: {}".format(
            len(modes), os.path.basename(self.output_path)
        ))


def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    source_path = filedialog.askopenfilename(
        title="Select an evaluation Excel workbook with X1-X4 and Y1-Y3",
        filetypes=[("Excel workbooks", "*.xlsx"), ("All files", "*.*")],
        parent=root,
    )
    if not source_path:
        root.destroy()
        return
    try:
        records, skipped, y_source_headers, source_sheet, header_row = read_source(source_path)
    except Exception as exc:
        messagebox.showerror("Could not read source workbook", str(exc), parent=root)
        root.destroy()
        return

    default_output = os.path.splitext(os.path.basename(source_path))[0] + "_meta_ai_training_dataset.xlsx"
    output_path = filedialog.asksaveasfilename(
        title="Save Meta AI training dataset workbook",
        initialdir=os.path.dirname(source_path),
        initialfile=default_output,
        defaultextension=".xlsx",
        filetypes=[("Excel workbooks", "*.xlsx")],
        parent=root,
    )
    if not output_path:
        root.destroy()
        return
    try:
        write_training_workbook(
            output_path, records, skipped, y_source_headers, source_sheet, header_row
        )
        summary = summarize_modes(records)
    except Exception as exc:
        messagebox.showerror("Could not create training datasets", str(exc), parent=root)
        root.destroy()
        return

    root.attributes("-topmost", False)
    root.deiconify()
    try:
        DatasetDashboard(root, output_path, summary)
    except Exception as exc:
        messagebox.showerror("Could not open class dashboard", str(exc), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
