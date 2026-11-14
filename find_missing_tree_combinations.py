"""Find empty Spot × Training Status × Prediction Match combinations."""

from __future__ import print_function

import os
import re
import tkinter as tk
from collections import Counter
from tkinter import filedialog, messagebox, ttk

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from PIL import Image, ImageDraw, ImageFont, ImageTk

import visualize_spot_tree as workbook_reader


TRAINING_VALUES = ("Untrained", "Trained")
MATCH_VALUES = ("Unmatched", "Matched")
BRIGHT_TIERS = (
    ("Bright Spot Prime", "Bright Prime"),
    ("Bright Spot High", "Bright High"),
    ("Bright Spot", "Bright Normal"),
    ("Bright Spot Low 1", "Bright Low 1"),
    ("Bright Spot Low 2", "Bright Low 2"),
)
SPOT_GROUPS = (
    "Bright (Integrated)",
    "Bright (Details)",
    "Gray",
    "Blind",
)
PERSPECTIVES = ("Spot", "Training Status", "Prediction Matching")
IMAGE_WIDTH = 1120
NODE_WIDTH = 225
ROW_GAP = 48


def load_fonts():
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(path):
            try:
                return (ImageFont.truetype(path, 14), ImageFont.truetype(path, 12),
                        ImageFont.truetype(path, 20))
            except OSError:
                pass
    default = ImageFont.load_default()
    return default, default, default


def normalize_spot(value, detailed):
    if detailed:
        for original, display in BRIGHT_TIERS:
            if value == original:
                return display
        canonical = workbook_reader.canonical_spot(value)
        if canonical == "Gray Spot":
            return "Gray"
        if canonical == "Blind Spot":
            return "Blind"
        return value

    canonical = workbook_reader.canonical_spot(value)
    if canonical == "Bright Spot":
        return "Bright (Integrated)"
    if canonical == "Gray Spot":
        return "Gray"
    if canonical == "Blind Spot":
        return "Blind"
    return value


def build_cases(records, detailed=False):
    """Build the full known category product, including combinations with count 0."""
    normalized = []
    counts = Counter()
    for record in records:
        item = dict(record)
        item["spot_group"] = normalize_spot(record["spot"], detailed)
        normalized.append(item)
        counts[(item["spot_group"], item["training"], item["match"])] += 1

    if detailed:
        expected_spots = tuple(display for _source, display in BRIGHT_TIERS) + ("Gray", "Blind")
    else:
        expected_spots = ("Bright (Integrated)", "Gray", "Blind")
    extras = {item["spot_group"] for item in normalized}.difference(expected_spots)
    spot_values = expected_spots + tuple(sorted(extras, key=str.casefold))

    cases = []
    for spot in spot_values:
        for training in TRAINING_VALUES:
            for match in MATCH_VALUES:
                cases.append({
                    "spot": spot,
                    "training": training,
                    "match": match,
                    "count": counts.get((spot, training, match), 0),
                })
    return normalized, cases


def spot_group_cases(records, group):
    """Return all status/match combinations for a selected Spot group tab."""
    if group == "Bright (Integrated)":
        _normalized, all_cases = build_cases(records, detailed=False)
        spots = {"Bright (Integrated)"}
    elif group == "Bright (Details)":
        _normalized, all_cases = build_cases(records, detailed=True)
        spots = {display for _source, display in BRIGHT_TIERS}
    elif group == "Gray":
        _normalized, all_cases = build_cases(records, detailed=False)
        spots = {"Gray"}
    else:
        _normalized, all_cases = build_cases(records, detailed=False)
        spots = {"Blind"}
    return [case for case in all_cases if case["spot"] in spots]


def rows_for_training(records, training, detailed=False):
    _normalized, cases = build_cases(records, detailed)
    return [case for case in cases if case["training"] == training]


def rows_for_match(records, match, detailed=False):
    _normalized, cases = build_cases(records, detailed)
    return [case for case in cases if case["match"] == match]


def make_paths(cases, dimensions, root_label):
    return [
        ((root_label,) + tuple(case[dimension] for dimension in dimensions), case["count"])
        for case in cases
    ]


def draw_tree(paths, column_labels, title, empty_count):
    depth = len(column_labels)
    height = max(360, 112 + ROW_GAP * max(1, len(paths)))
    image = Image.new("RGB", (IMAGE_WIDTH, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    font, small_font, title_font = load_fonts()
    draw.text((20, 15), "{}  |  {} EMPTY CASE(S)".format(title.upper(), empty_count),
              fill="#1F3E5B", font=title_font)

    margin = 18
    gap = (IMAGE_WIDTH - 2 * margin - depth * NODE_WIDTH) / float(max(1, depth - 1))
    xs = [int(margin + index * (NODE_WIDTH + gap)) for index in range(depth)]
    for index, label in enumerate(column_labels):
        draw.text((xs[index], 72), label.upper(), fill="#526174", font=small_font)

    y_positions = {path: 102 + index * ROW_GAP for index, (path, _count) in enumerate(paths)}
    centers, totals = {}, {}
    for path, count in paths:
        y = y_positions[path]
        for level in range(depth):
            prefix = path[:level + 1]
            centers.setdefault((level, prefix), []).append(y)
            totals[(level, prefix)] = totals.get((level, prefix), 0) + count
    centers = {key: sum(values) / float(len(values)) for key, values in centers.items()}

    for level, prefix in centers:
        if level == 0:
            continue
        parent_y = centers[(level - 1, prefix[:-1])]
        child_y = centers[(level, prefix)]
        x1, x2 = xs[level - 1] + NODE_WIDTH, xs[level]
        elbow = x1 + (x2 - x1) / 2.0
        draw.line([(x1, parent_y), (elbow, parent_y), (elbow, child_y), (x2, child_y)],
                  fill="#A9B4C2", width=2)

    for (level, prefix), center_y in centers.items():
        count = totals[(level, prefix)]
        x1, x2 = xs[level], xs[level] + NODE_WIDTH
        y1, y2 = center_y - 17, center_y + 17
        no_samples = level == depth - 1 and count == 0
        fill = "#FCE8E6" if no_samples else ("#E8F6EF" if level == 0 else "#FFFFFF")
        outline = "#C23B32" if no_samples else "#A9B4C2"
        color = "#A12620" if no_samples else "#263E5B"
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=fill,
                               outline=outline, width=2 if no_samples else 1)
        label = prefix[-1]
        suffix = " — NO SAMPLES" if no_samples else ""
        while draw.textbbox((0, 0), label + suffix, font=small_font)[2] > NODE_WIDTH - 20 and len(label) > 6:
            label = label[:-4] + "..."
        draw.text((x1 + 8, y1 + 10), "{} ({}){}".format(label, count, suffix),
                  fill=color, font=small_font)
    return image


def create_graph_result(records, perspective, group, root, cases, dimensions, root_label):
    paths = make_paths(cases, dimensions, root_label)
    columns = [root] + [
        {"spot": "Spot Group", "training": "Training Status", "match": "Prediction Matching"}[key]
        for key in dimensions
    ]
    title = "{} / {}".format(perspective, group or root_label)
    image = draw_tree(paths, columns, title, sum(count == 0 for _path, count in paths))
    return {
        "perspective": perspective,
        "group": group or root_label,
        "root": root_label,
        "columns": columns,
        "paths": paths,
        "image": image,
    }


def build_all_graphs(records, detailed_for_status_match=False):
    results = []

    # Spot perspective: four requested group tabs. Bright Details retains all
    # five Bright tiers as separate branches in the same tree.
    for group in SPOT_GROUPS:
        if group == "Bright (Details)":
            cases = spot_group_cases(records, group)
            results.append(create_graph_result(
                records, "Spot", group, "Bright (Details)", cases,
                ("spot", "training", "match"), "Bright (Details)",
            ))
        else:
            cases = spot_group_cases(records, group)
            root_label = group
            results.append(create_graph_result(
                records, "Spot", group, "Spot Group", cases,
                ("training", "match"), root_label,
            ))

    # Each status is a distinct root and its own sub-tab.
    for training in TRAINING_VALUES:
        cases = rows_for_training(records, training, detailed_for_status_match)
        results.append(create_graph_result(
            records, "Training Status", training, "Training Status", cases,
            ("spot", "match"), training,
        ))

    # Each prediction outcome is a distinct root and its own sub-tab.
    for match in MATCH_VALUES:
        cases = rows_for_match(records, match, detailed_for_status_match)
        results.append(create_graph_result(
            records, "Prediction Matching", match, "Prediction Matching", cases,
            ("spot", "training"), match,
        ))
    return results


def save_report(path, results):
    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    summary.append(["Perspective", "Root / Spot Group", "Combinations", "Empty combinations"])
    for result in results:
        empty = sum(count == 0 for _path, count in result["paths"])
        summary.append([result["perspective"], result["group"], len(result["paths"]), empty])

    used_names = set(workbook.sheetnames)
    for index, result in enumerate(results, start=1):
        base_name = "{} {}".format(result["perspective"], result["group"])
        sheet_name = re.sub(r"[\\/*?:\[\]]", " ", base_name).strip()[:31] or "Cases {}".format(index)
        while sheet_name in used_names:
            suffix = " {}".format(index)
            sheet_name = sheet_name[:31 - len(suffix)] + suffix
        used_names.add(sheet_name)
        sheet = workbook.create_sheet(sheet_name)
        sheet.append(result["columns"] + ["Sample Count", "Result"])
        for path_values, count in result["paths"]:
            sheet.append(list(path_values) + [count, "NO SAMPLES" if count == 0 else "Present"])

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(name="Arial", bold=True, color="FFFFFF")
            cell.fill = PatternFill(fill_type="solid", fgColor="315C7D")
        for column in sheet.columns:
            widths = [len(str(cell.value or "")) for cell in column]
            sheet.column_dimensions[column[0].column_letter].width = min(max(max(widths + [12]) + 2, 14), 40)
    workbook.save(path)


class MissingCasesApp:
    def __init__(self, root, source_path, records):
        self.root = root
        self.source_path = source_path
        self.records = records
        self.images = {}
        root.title("Missing Spot Combination Finder")
        root.geometry("1200x850")

        header = ttk.Frame(root, padding=12)
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Spot groups: Bright (Integrated), Bright (Details), Gray, Blind. "
                 "Training and Prediction Matching are split into root tabs.",
        ).pack(anchor="w")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        footer = ttk.Frame(root, padding=(12, 0, 12, 10))
        footer.pack(fill="x")
        self.status = ttk.Label(footer, text="")
        self.status.pack(side="left")
        # Let Tk map the nested notebooks before creating their image items.
        # Aqua can otherwise finish initialization while the canvas is still
        # unmapped, leaving the status/footer visible but the diagrams blank.
        root.after_idle(self.safe_refresh)

    def safe_refresh(self):
        try:
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Could not display missing-case trees", str(exc), parent=self.root)
            self.status.configure(text="Tree display failed: {}".format(exc))

    def add_graph_tab(self, notebook, graph):
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
        canvas.update_idletasks()
        canvas.bind(
            "<Map>",
            lambda _event, widget=canvas: self.root.after_idle(widget.update_idletasks),
        )

    def refresh(self):
        for child in self.notebook.winfo_children():
            child.destroy()
        self.images = {}
        graphs = build_all_graphs(self.records)
        groups = {}
        for graph in graphs:
            groups.setdefault(graph["perspective"], []).append(graph)

        for perspective in PERSPECTIVES:
            outer_tab = ttk.Frame(self.notebook)
            self.notebook.add(outer_tab, text=perspective)
            inner = ttk.Notebook(outer_tab)
            inner.pack(fill="both", expand=True)
            for graph in groups[perspective]:
                self.add_graph_tab(inner, graph)

        prefix = os.path.splitext(self.source_path)[0]
        for graph in graphs:
            key = re.sub(r"[^A-Za-z0-9_-]+", "_", "{}_{}".format(
                graph["perspective"], graph["group"]
            )).strip("_").lower()
            graph["image"].save(prefix + "_missing_" + key + ".png")
        report_path = prefix + "_missing_combinations.xlsx"
        save_report(report_path, graphs)
        combinations = sum(len(graph["paths"]) for graph in graphs)
        empty = sum(sum(count == 0 for _path, count in graph["paths"]) for graph in graphs)
        self.status.configure(text="{} graph cases | {} empty | {}".format(
            combinations, empty, os.path.basename(report_path)
        ))


def main():
    # Reuse the same Tk interpreter for the file dialog and the app window.
    # Destroying one Tk root and creating another can leave Aqua/Tcl in a
    # deleted-table state on macOS, preventing the tree window from appearing.
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
        MissingCasesApp(root, path, records)
    except Exception as exc:
        messagebox.showerror("Could not build missing-case trees", str(exc), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
