"""Horizontal percentage-bar dashboard for three evaluation perspectives.

Case A and Case B can each be selected or skipped independently. Bright Spot
tiers are merged; each subgroup's combination percentages use that subgroup's
sample count as the denominator.
"""

from __future__ import print_function

import colorsys
import datetime
import os
import tkinter as tk
from collections import Counter
from itertools import product
from tkinter import filedialog, messagebox, ttk

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from PIL import Image, ImageDraw, ImageFont, ImageTk

import visualize_spot_tree as workbook_reader


PERSPECTIVES = (
    ("Spot", "spot", ("Bright Spot", "Blind Spot", "Gray Spot"), ("training", "match")),
    ("Training Status", "training", ("Trained", "Untrained"), ("spot", "match")),
    ("Prediction Match", "match", ("Matched", "Unmatched"), ("spot", "training")),
)
DIMENSION_NAMES = {
    "spot": "Spot",
    "training": "Training Status",
    "match": "Prediction Match",
}
DIMENSION_VALUES = {
    "spot": ("Bright Spot", "Blind Spot", "Gray Spot"),
    "training": ("Trained", "Untrained"),
    "match": ("Matched", "Unmatched"),
}
WIDTH = 1280


def load_fonts():
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(path):
            try:
                return (ImageFont.truetype(path, 15), ImageFont.truetype(path, 12),
                        ImageFont.truetype(path, 22))
            except OSError:
                pass
    font = ImageFont.load_default()
    return font, font, font


def unique_path(path):
    """Never overwrite an existing generated chart or workbook."""
    stem, extension = os.path.splitext(path)
    candidate, suffix = path, 1
    while os.path.exists(candidate):
        candidate = "{}_{:02d}{}".format(stem, suffix, extension)
        suffix += 1
    return candidate


def read_case(path, case_name):
    records, _trained_categories = workbook_reader.load_evaluation_data(path)
    cleaned = []
    for record in records:
        normalized = dict(record)
        normalized["spot"] = workbook_reader.canonical_spot(record["spot"]) or record["spot"]
        normalized["case"] = case_name
        cleaned.append(normalized)
    return cleaned


def series_for(dimensions):
    return list(product(*(DIMENSION_VALUES[dimension] for dimension in dimensions)))


def percentage_table(records, case_name, pivot_key, category, series_dimensions):
    selected = [record for record in records
                if record.get("case") == case_name and record[pivot_key] == category]
    total = len(selected)
    series_values = series_for(series_dimensions)
    counts = Counter(tuple(record[dimension] for dimension in series_dimensions) for record in selected)
    percentages = {values: counts[values] * 100.0 / total if total else 0.0 for values in series_values}
    return total, counts, percentages, series_values


def palette(series_values):
    colors = {}
    for index, values in enumerate(series_values):
        hue = (index * 0.61803398875) % 1.0
        rgb = colorsys.hls_to_rgb(hue, 0.46, 0.72)
        colors[values] = "#{:02X}{:02X}{:02X}".format(*(int(channel * 255) for channel in rgb))
    return colors


def category_display(pivot_key, category):
    if pivot_key == "training":
        return category + " Category"
    return category


def draw_horizontal_bars(records, perspective, case_names, case_label, layout="side_by_side"):
    title, pivot_key, categories, series_dimensions = perspective
    series_values = series_for(series_dimensions)
    colors = palette(series_values)
    font, small, title_font = load_fonts()
    legend_columns = 3
    legend_rows = int((len(series_values) + legend_columns - 1) / legend_columns)
    panels_top = 112 + legend_rows * 24
    bars_per_panel = len(series_values)
    panel_height = 42 + bars_per_panel * 24 + 22
    if layout == "single_column":
        panel_items = [(category, case_name) for category in categories for case_name in case_names]
        height = panels_top + len(panel_items) * panel_height + 42
    else:
        panel_items = []
        height = panels_top + len(categories) * panel_height + 74
    image = Image.new("RGB", (WIDTH, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    total_records = len(records)
    layout_text = "CASE A/B — SINGLE COLUMN" if layout == "single_column" else "CASE A/B — SIDE BY SIDE"
    draw.text((24, 15), "{} PERSPECTIVE — {} — {}".format(title.upper(), case_label.upper(), layout_text),
              fill="#1F3E5B", font=title_font)
    draw.text((24, 46), "{} samples across {} case(s) | percentages are normalized within each case and category (N shown)".format(
        total_records, len(case_names)), fill="#526174", font=small)

    draw.text((24, 75), "Legend — each color is a combination of {}".format(
        " × ".join(DIMENSION_NAMES[name] for name in series_dimensions)), fill="#1F3E5B", font=font)
    for index, values in enumerate(series_values):
        column, row = index % legend_columns, index // legend_columns
        x, y = 24 + column * 410, 99 + row * 24
        draw.rectangle((x, y + 4, x + 15, y + 16), fill=colors[values])
        label = " | ".join(values)
        draw.text((x + 22, y), label, fill="#263E5B", font=small)

    if not case_names:
        draw.text((25, panels_top + 10), "No valid case data", fill="#A12620", font=font)
        return image
    margin_x, panel_gap = 24, 22
    if layout == "single_column":
        panel_width = WIDTH - 2 * margin_x
        arranged_panels = [
            (category, case_name, margin_x, panels_top + index * panel_height)
            for index, (category, case_name) in enumerate(panel_items)
        ]
    else:
        panel_width = (WIDTH - 2 * margin_x - panel_gap * (len(case_names) - 1)) / float(len(case_names))
        for case_index, case_name in enumerate(case_names):
            case_x = margin_x + case_index * (panel_width + panel_gap)
            draw.rounded_rectangle((case_x, panels_top - 3, case_x + panel_width, panels_top + 25),
                                   radius=6, fill="#E8EDF3")
            draw.text((case_x + 10, panels_top + 3), case_name, fill="#1F3E5B", font=font)
        arranged_panels = []
        for category_index, category in enumerate(categories):
            row_y = panels_top + 32 + category_index * panel_height
            for case_index, case_name in enumerate(case_names):
                case_x = margin_x + case_index * (panel_width + panel_gap)
                arranged_panels.append((category, case_name, case_x, row_y))

    for category, case_name, panel_x, row_y in arranged_panels:
        total, counts, percentages, _values = percentage_table(
            records, case_name, pivot_key, category, series_dimensions)
        draw.rounded_rectangle((panel_x, row_y, panel_x + panel_width, row_y + panel_height - 5),
                               radius=7, fill="#FFFFFF", outline="#D6DDE6", width=1)
        heading = "{}  |  N={}".format(category_display(pivot_key, category), total)
        if layout == "single_column":
            heading = "{} — {}  |  N={}".format(category_display(pivot_key, category), case_name, total)
        draw.text((panel_x + 10, row_y + 5), heading, fill="#263E5B", font=font)
        label_x = panel_x + 12
        bar_x = panel_x + 180
        bar_width = panel_width - 260
        for series_index, values in enumerate(series_values):
            y = row_y + 33 + series_index * 24
            label = " | ".join(values)
            while draw.textbbox((0, 0), label, font=small)[2] > 160 and len(label) > 8:
                label = label[:-4] + "..."
            draw.text((label_x, y + 3), label, fill="#405166", font=small)
            draw.rounded_rectangle((bar_x, y + 2, bar_x + bar_width, y + 17),
                                   radius=4, fill="#E8EDF3")
            pct = percentages[values]
            if pct > 0:
                draw.rounded_rectangle((bar_x, y + 2, bar_x + bar_width * pct / 100.0, y + 17),
                                       radius=4, fill=colors[values])
            draw.text((bar_x + bar_width + 7, y + 2), "{:.1f}%".format(pct),
                      fill="#263E5B", font=small)

    footer_y = panels_top + len(panel_items) * panel_height + 10 if layout == "single_column" else panels_top + len(categories) * panel_height + 42
    draw.text((24, footer_y), "Within each Case × {} panel, all combination percentages sum to 100% when N > 0.".format(title),
              fill="#526174", font=small)
    return image


def save_counts_workbook(path, records, selected_cases):
    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    summary.append(["Perspective bar chart counts", "Value"])
    summary.append(["Cases included", " + ".join("{}: {}".format(name, os.path.basename(source)) for name, source in selected_cases)])
    rule = "Case A and Case B are reported separately" if len(selected_cases) == 2 else "Only the selected standalone case is reported"
    summary.append(["Comparison rule", rule])
    summary.append(["Percentage denominator", "Samples within the same Case and selected category"])
    summary.append(["Bright Spot rule", "All Bright Spot tiers are merged into Bright Spot"])
    summary.append(["Total samples", len(records)])
    source_sheet = workbook.create_sheet("Source Files")
    source_sheet.append(["Case", "Source workbook"])
    for case_name, filename in selected_cases:
        source_sheet.append([case_name, filename])

    for title, pivot_key, categories, series_dimensions in PERSPECTIVES:
        sheet = workbook.create_sheet(title[:31])
        sheet.append(["Case", title, "Combination", "Sample count", "Category N", "Percentage"])
        series_values = series_for(series_dimensions)
        for case_name, _filename in selected_cases:
            for category in categories:
                total, counts, percentages, _values = percentage_table(
                    records, case_name, pivot_key, category, series_dimensions)
                for series in series_values:
                    sheet.append([case_name, category, " × ".join(series), counts[series],
                                  total, percentages[series]])
        for row in sheet.iter_rows(min_row=2, min_col=6, max_col=6):
            row[0].number_format = "0.0%"
            row[0].value = row[0].value / 100.0
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(name="Arial", bold=True, color="FFFFFF")
            cell.fill = PatternFill(fill_type="solid", fgColor="315C7D")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for column in sheet.columns:
            letter = column[0].column_letter
            longest = max([len(str(cell.value or "")) for cell in column[:100]] + [12])
            sheet.column_dimensions[letter].width = min(max(longest + 2, 14), 42)
        sheet.row_dimensions[1].height = 32
    workbook.save(path)


class DeviationBarDashboard:
    def __init__(self, root, records, case_names, case_label, output_prefix, excel_path):
        self.root = root
        self.records = records
        self.case_names = case_names
        self.case_label = case_label
        self.output_prefix = output_prefix
        self.excel_path = excel_path
        self.graph_paths = {
            (title, layout): unique_path(output_prefix + "_" + title.lower().replace(" ", "_")
                                         + "_" + layout + ".png")
            for title, _key, _categories, _others in PERSPECTIVES
            for layout in ("side_by_side", "single_column")
        }
        self.photos = {}
        root.title("Evaluation Deviation Bar Perspectives — Bright Spot Merged")
        root.geometry("1360x900")

        header = ttk.Frame(root, padding=(12, 10, 12, 5))
        header.pack(fill="x")
        ttk.Label(header, text="Choose a perspective. Each category shows a separate Case panel with horizontal percentages for combinations of the other two dimensions.").pack(anchor="w")
        self.perspective_var = tk.StringVar(value="Spot")
        selectors = ttk.Frame(root, padding=(12, 2, 12, 8))
        selectors.pack(fill="x")
        for title, _key, _categories, _others in PERSPECTIVES:
            ttk.Radiobutton(selectors, text=title, variable=self.perspective_var,
                            value=title, command=self.refresh).pack(side="left", padx=(0, 18))
        self.view_tabs = ttk.Notebook(root)
        self.view_tabs.pack(fill="both", expand=True, padx=10, pady=5)
        self.canvases = {}
        for layout, tab_label in (("side_by_side", "Case A | Case B side by side"),
                                  ("single_column", "Case A/B single column")):
            tab = ttk.Frame(self.view_tabs)
            self.view_tabs.add(tab, text=tab_label)
            graph_frame = ttk.Frame(tab)
            graph_frame.pack(fill="both", expand=True)
            canvas = tk.Canvas(graph_frame, background="#EEF2F6", highlightthickness=0)
            ybar = ttk.Scrollbar(graph_frame, orient="vertical", command=canvas.yview)
            xbar = ttk.Scrollbar(graph_frame, orient="horizontal", command=canvas.xview)
            canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
            canvas.grid(row=0, column=0, sticky="nsew")
            ybar.grid(row=0, column=1, sticky="ns")
            xbar.grid(row=1, column=0, sticky="ew")
            graph_frame.rowconfigure(0, weight=1)
            graph_frame.columnconfigure(0, weight=1)
            self.canvases[layout] = canvas
        self.status = ttk.Label(root, text="", padding=(12, 5))
        self.status.pack(fill="x")
        self.refresh()

    def refresh(self):
        perspective = next(item for item in PERSPECTIVES if item[0] == self.perspective_var.get())
        title, _pivot_key, _categories, _series_dimensions = perspective
        saved_names = []
        for layout in ("side_by_side", "single_column"):
            image = draw_horizontal_bars(self.records, perspective, self.case_names,
                                         self.case_label, layout)
            photo = ImageTk.PhotoImage(image, master=self.canvases[layout])
            self.photos[layout] = photo
            canvas = self.canvases[layout]
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.configure(scrollregion=(0, 0, image.width, image.height))
            graph_path = self.graph_paths[(title, layout)]
            image.save(graph_path)
            saved_names.append(os.path.basename(graph_path))
        self.status.configure(text="{} samples | {} | chart: {} | counts workbook: {}".format(
            len(self.records), self.case_label, ", ".join(saved_names),
            os.path.basename(self.excel_path)))


def select_workbooks(root):
    case_a = filedialog.askopenfilename(
        title="Select Case A workbook (Cancel to skip Case A)",
        filetypes=[("Excel workbooks", "*.xlsx"), ("All files", "*.*")], parent=root)
    case_b = filedialog.askopenfilename(
        title="Select Case B workbook (Cancel to skip Case B)",
        filetypes=[("Excel workbooks", "*.xlsx"), ("All files", "*.*")], parent=root)
    selected = []
    if case_a:
        selected.append(("Case A", case_a))
    if case_b and (not case_a or os.path.normcase(os.path.abspath(case_b)) != os.path.normcase(os.path.abspath(case_a))):
        selected.append(("Case B", case_b))
    return selected


def main():
    dialog = tk.Tk()
    dialog.withdraw()
    dialog.attributes("-topmost", True)
    selected_cases = select_workbooks(dialog)
    if not selected_cases:
        messagebox.showinfo("No workbooks selected", "Select at least one workbook: Case A or Case B.", parent=dialog)
        dialog.destroy()
        return
    try:
        records = []
        valid_cases = []
        for case_name, path in selected_cases:
            records.extend(read_case(path, case_name))
            valid_cases.append((case_name, path))
    except Exception as error:
        messagebox.showerror("Could not read evaluation workbook", str(error), parent=dialog)
        dialog.destroy()
        return
    dialog.destroy()

    case_label = "Case A & Case B comparison" if len(valid_cases) == 2 else "{} standalone".format(valid_cases[0][0])
    output_prefix = os.path.splitext(valid_cases[0][1])[0] + "_bright_merged_deviation_bar"
    date_stamp = datetime.date.today().strftime("%Y%m%d")
    excel_candidate = unique_path(output_prefix + "_counts_{}.xlsx".format(date_stamp))
    try:
        save_counts_workbook(excel_candidate, records, [(name, os.path.basename(path)) for name, path in valid_cases])
    except Exception as error:
        error_root = tk.Tk()
        error_root.withdraw()
        messagebox.showerror("Could not save count workbook", str(error), parent=error_root)
        error_root.destroy()
        return

    root = tk.Tk()
    root.title("Evaluation Deviation Bar Perspectives — Bright Spot Merged")
    DeviationBarDashboard(root, records, [name for name, _path in valid_cases],
                          case_label, output_prefix, excel_candidate)
    root.mainloop()


if __name__ == "__main__":
    main()
