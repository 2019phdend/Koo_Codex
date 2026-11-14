"""Create Meta AI datasets for five X-feature sets and seven Y-target sets.

The source Excel reader and target definitions are shared with the established
generator. This file keeps all feature selection explicit when creating sheets
and graphs, so a chart cannot silently fall back to X1-X3 or lose its records.
"""

from __future__ import print_function

import colorsys
import datetime
import math
import os
import re
import tkinter as tk
from collections import Counter
from tkinter import filedialog, messagebox, ttk

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from PIL import Image, ImageDraw, ImageTk

import create_meta_ai_training_dataset as base


ALL_FEATURES = tuple(base.FEATURES)
FEATURE_SETS = (
    ("X1_X2", ("X1", "X2")),
    ("X2_X3", ("X2", "X3")),
    ("X3_X1", ("X3", "X1")),
    ("X1_X2_X3", ("X1", "X2", "X3")),
    ("X1_X2_X4", ("X1", "X2", "X4")),
    ("X1_X3_X4", ("X1", "X3", "X4")),
    ("X2_X3_X4", ("X2", "X3", "X4")),
    ("X1_X2_X3_X4", ALL_FEATURES),
)


def unique_excel_path(path):
    """Return an unused Excel filename rather than overwriting an existing one."""
    stem, extension = os.path.splitext(path)
    candidate = path
    suffix = 1
    while os.path.exists(candidate):
        candidate = "{}_{:02d}{}".format(stem, suffix, extension)
        suffix += 1
    return candidate

GUIDES = {
    "Class distribution": (
        "Bar length and number show samples per target class.",
        "Large differences between bars indicate class imbalance; inspect this before splitting training and validation data.",
    ),
    "Class relationships": (
        "Each node is a target value or value combination. The number in parentheses is its sample count.",
        "Small branches have fewer examples and may be harder to learn reliably.",
    ),
    "Class feature profile": (
        "Rows are target classes; columns are selected X features. Cells show the class mean; color runs from low to high mean in this graph.",
        "Different averages can be useful, but averages alone do not show within-class spread or prove that classes are separable.",
    ),
    "Feature-class association": (
        "η² ranges from 0 to 1 and estimates how much of each feature's variance is between target classes.",
        "Higher η² suggests class-related variation in that feature. It is descriptive, not model accuracy or an overlap score.",
    ),
    "Pairwise feature overlap": (
        "Each dot is a sample; color indicates target class. Each panel plots a pair of selected X features.",
        "Distinct color clouds suggest separation for that pair. Mixed colors suggest overlap in that 2D view.",
    ),
    "PCA class separation": (
        "All selected X features are standardized and projected onto PC1 and PC2. Each dot is a sample, colored by target class.",
        "Separated color groups suggest separation in this projection. PCA compresses dimensions, so also inspect pairwise plots and cluster boundaries.",
    ),
    "Cluster boundaries": (
        "Dots are samples; colored outlines are convex hulls around each class after PCA projection of the selected X features.",
        "Separated hulls suggest distinct clusters in this 2D projection. Overlapping hulls flag possible overlap; hulls are descriptive and sensitive to outliers.",
    ),
    "Y distribution by X": (
        "Each bar label gives the numeric X interval used to group samples; n is the number of samples in that interval. Stacked colors show target-class percentages.",
        "Compare the class mix from low to high X values. This reverses the usual class-by-feature view and is descriptive; it does not imply causation.",
    ),
}


def feature_set_name(features):
    return "_".join(features)


def integrate_bright_spot(value):
    text = str(value or "").strip()
    folded = text.casefold()
    if folded.startswith("bright spot"):
        return "Bright Spot"
    if folded.startswith("gray spot") or folded == "gray":
        return "Gray Spot"
    if folded.startswith("blind spot") or folded == "blind":
        return "Blind Spot"
    return text


def label_for(record, targets):
    return base.target_class(record["labels"], targets)


def read_union(paths):
    """Read one or two source workbooks and preserve their row provenance."""
    records, skipped, sources = [], [], []
    common_mapping = None
    for path in paths:
        file_records, file_skipped, y_headers, sheet_name, header_row = base.read_source(path)
        for record in file_records:
            record["labels"]["Y3"] = integrate_bright_spot(record["labels"]["Y3"])
        mapping = tuple(base.header_key(y_headers[target]) for target in base.TARGETS)
        if common_mapping is not None and mapping != common_mapping:
            raise ValueError(
                "The selected workbooks map Y1/Y2/Y3 to different source columns.\n\n"
                "Select workbooks with the same Y-column meanings and order."
            )
        common_mapping = mapping
        filename = os.path.basename(path)
        for record in file_records:
            record["source_file"] = filename
        records.extend(file_records)
        skipped.extend({
            "Source file": filename,
            "Source row": item["Source row"],
            "Reason": item["Reason"],
        } for item in file_skipped)
        sources.append({
            "Source file": filename,
            "Full path": os.path.abspath(path),
            "Source sheet": sheet_name,
            "Header row": header_row,
            "Valid samples": len(file_records),
            "Skipped rows": len(file_skipped),
            "Y columns": [y_headers[target] for target in base.TARGETS],
        })
    if not records:
        raise ValueError("No valid records were found in the selected workbook(s).")
    return records, skipped, sources


def style_sheet(sheet):
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(name="Arial", bold=True, color="FFFFFF")
        cell.fill = PatternFill(fill_type="solid", fgColor="315C7D")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.row_dimensions[1].height = 30
    for column in sheet.columns:
        sample_lengths = [len(str(cell.value or "")) for cell in column[:150]]
        letter = column[0].column_letter
        sheet.column_dimensions[letter].width = min(max(max(sample_lengths + [12]) + 2, 12), 42)
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = "0.00"


def write_workbook(path, records, skipped, sources):
    """Write all 5 X sets × 7 established Y sets as separate training sheets."""
    workbook = Workbook()
    summary = workbook.active
    summary.title = "Dataset Summary"
    summary.append(["Meta AI feature-combination datasets", "Details"])
    summary.append(["Valid samples", len(records)])
    summary.append(["X feature sets", len(FEATURE_SETS)])
    summary.append(["Y target sets", len(base.TARGET_MODES)])
    summary.append(["Dataset sheets", len(FEATURE_SETS) * len(base.TARGET_MODES)])
    summary.append(["X combinations", " | ".join(feature_set_name(features) for _name, features in FEATURE_SETS)])
    summary.append(["Y combinations", " | ".join(mode for mode, _targets in base.TARGET_MODES)])
    summary.append([])
    summary.append(["X set", "Y set", "Input features", "Targets", "Samples", "Classes", "Sheet"])

    distribution = workbook.create_sheet("Class Distribution")
    distribution.append(["X set", "Y set", "Target Class", "Samples", "Share (%)"])

    for x_name, features in FEATURE_SETS:
        for y_name, targets in base.TARGET_MODES:
            sheet_name = "{}_{}".format(x_name.replace("_", ""), y_name.replace("_", ""))
            sheet = workbook.create_sheet(sheet_name[:31])
            y_headers = ["Y{}".format(index) for index in range(1, len(targets) + 1)]
            headers = ["Sample ID", "Source file", "Source row", "Image path", "Test category"]
            headers += list(features) + y_headers + ["Target Class"]
            sheet.append(headers)
            counts = Counter(label_for(record, targets) for record in records)
            for record in records:
                sample_id = "{}::{}".format(record["source_file"], record["source_row"])
                sheet.append([
                    sample_id, record["source_file"], record["source_row"],
                    record.get("image_path", ""), record.get("test_category", ""),
                ] + [record["features"][feature] for feature in features]
                    + [record["labels"][target] for target in targets]
                    + [label_for(record, targets)])
            style_sheet(sheet)
            summary.append([
                x_name, y_name, ", ".join(features), ", ".join(targets),
                len(records), len(counts), sheet.title,
            ])
            for class_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold())):
                distribution.append([
                    x_name, y_name, class_name, count,
                    count / float(len(records)) * 100.0,
                ])

    source_sheet = workbook.create_sheet("Source Files")
    source_sheet.append(["Source file", "Full path", "Source sheet", "Header row", "Valid samples", "Skipped rows", "Y1 header", "Y2 header", "Y3 header"])
    for item in sources:
        source_sheet.append([
            item["Source file"], item["Full path"], item["Source sheet"],
            item["Header row"], item["Valid samples"], item["Skipped rows"],
        ] + item["Y columns"])
    skipped_sheet = workbook.create_sheet("Skipped Rows")
    skipped_sheet.append(["Source file", "Source row", "Reason"])
    for item in skipped:
        skipped_sheet.append([item["Source file"], item["Source row"], item["Reason"]])
    for sheet in (summary, distribution, source_sheet, skipped_sheet):
        style_sheet(sheet)
    summary.column_dimensions["A"].width = 24
    summary.column_dimensions["B"].width = 48
    distribution.column_dimensions["C"].width = 50
    source_sheet.column_dimensions["B"].width = 55
    skipped_sheet.column_dimensions["C"].width = 45
    workbook.save(path)


def class_counts(records, targets):
    return Counter(label_for(record, targets) for record in records)


def class_colors(records, targets):
    labels = sorted(class_counts(records, targets), key=lambda name: name.casefold())
    result = {}
    for index, label in enumerate(labels):
        hue = (index * 0.61803398875) % 1.0
        rgb = colorsys.hls_to_rgb(hue, 0.46, 0.78)
        result[label] = "#{:02X}{:02X}{:02X}".format(*(int(channel * 255) for channel in rgb))
    return result


def draw_distribution(x_name, y_name, records, targets):
    counts = sorted(class_counts(records, targets).items(), key=lambda item: (-item[1], item[0].casefold()))
    font, small, title = base.load_fonts()
    height = max(340, 105 + len(counts) * 25)
    image = Image.new("RGB", (760, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    draw.text((18, 14), "{} | {} CLASS DISTRIBUTION".format(x_name, y_name), fill="#1F3E5B", font=title)
    draw.text((18, 45), "{} samples | {} classes | bar length = sample count".format(len(records), len(counts)), fill="#526174", font=small)
    max_count = max([count for _label, count in counts] or [1])
    for index, (label, count) in enumerate(counts):
        y = 78 + index * 25
        name = label
        while draw.textbbox((0, 0), name, font=small)[2] > 225 and len(name) > 8:
            name = name[:-4] + "..."
        draw.text((18, y + 3), name, fill="#263E5B", font=small)
        draw.rounded_rectangle((255, y, 615, y + 18), radius=5, fill="#E3E8EF")
        draw.rounded_rectangle((255, y, 255 + int(360 * count / max_count), y + 18), radius=5, fill="#4C8A72")
        draw.text((625, y + 2), str(count), fill="#263E5B", font=small)
    return image


def draw_tree(x_name, y_name, records, targets):
    counts = Counter(tuple(record["labels"][target] for target in targets) for record in records)
    ordered = sorted(counts.items(), key=lambda item: tuple(value.casefold() for value in item[0]))
    node_w, gap, row_h = 168, 30, 42
    depth = len(targets) + 1
    width = 28 + depth * node_w + (depth - 1) * gap + 20
    height = max(340, 105 + len(ordered) * row_h)
    image = Image.new("RGB", (width, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    font, small, title = base.load_fonts()
    draw.text((18, 14), "{} | {} LABEL TREE".format(x_name, y_name), fill="#1F3E5B", font=title)
    draw.text((18, 44), "Node label (count) | branches follow target order: {}".format(" → ".join(targets)), fill="#526174", font=small)
    xs = [18 + level * (node_w + gap) for level in range(depth)]
    labels = ["All samples"] + list(targets)
    for level, text in enumerate(labels):
        draw.text((xs[level], 75), text.upper(), fill="#526174", font=small)
    if not ordered:
        return image
    leaf_y = {path: 102 + index * row_h for index, (path, _count) in enumerate(ordered)}
    centers, totals = {}, {}
    for path, count in ordered:
        values = ("All samples",) + path
        y = leaf_y[path]
        for level in range(depth):
            prefix = values[:level + 1]
            centers.setdefault((level, prefix), []).append(y)
            totals[(level, prefix)] = totals.get((level, prefix), 0) + count
    centers = {key: sum(value) / float(len(value)) for key, value in centers.items()}
    for level, prefix in centers:
        if level:
            py = centers[(level - 1, prefix[:-1])]
            cy = centers[(level, prefix)]
            x1, x2 = xs[level - 1] + node_w, xs[level]
            mid = (x1 + x2) / 2.0
            draw.line([(x1, py), (mid, py), (mid, cy), (x2, cy)], fill="#A9B4C2", width=2)
    fills = ("#E8F6EF", "#EAF0FA", "#FFF4E5", "#F0ECFA")
    for (level, prefix), cy in centers.items():
        x1, x2 = xs[level], xs[level] + node_w
        y1, y2 = cy - 16, cy + 16
        draw.rounded_rectangle((x1, y1, x2, y2), radius=7, fill=fills[level % len(fills)], outline="#A9B4C2")
        text = "{} ({})".format(prefix[-1], totals[(level, prefix)])
        while draw.textbbox((0, 0), text, font=small)[2] > node_w - 12 and len(text) > 8:
            text = text[:-4] + "..."
        draw.text((x1 + 6, y1 + 8), text, fill="#263E5B", font=small)
    return image


def draw_profile(x_name, y_name, records, targets, features):
    labels = class_counts(records, targets)
    rows = []
    for label in labels:
        group = [record for record in records if label_for(record, targets) == label]
        means = {feature: sum(float(item["features"][feature]) for item in group) / len(group) for feature in features}
        rows.append((label, len(group), means))
    rows.sort(key=lambda item: (-item[1], item[0].casefold()))
    all_means = [value for _label, _count, means in rows for value in means.values()]
    low, high = min(all_means or [0.0]), max(all_means or [1.0])
    width = max(760, 330 + len(features) * 125)
    height = max(360, 110 + len(rows) * 32 + 55)
    image = Image.new("RGB", (width, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    _font, small, title = base.load_fonts()
    draw.text((18, 14), "{} | {} CLASS FEATURE PROFILE".format(x_name, y_name), fill="#1F3E5B", font=title)
    draw.text((18, 45), "Class-wise feature mean | cell color is scaled to this graph's minimum and maximum", fill="#526174", font=small)
    label_x, count_x, start_x, cell_w, top, row_h = 18, 280, 350, 120, 100, 32
    draw.text((label_x, 75), "TARGET CLASS", fill="#526174", font=small)
    draw.text((count_x, 75), "N", fill="#526174", font=small)
    for i, feature in enumerate(features):
        draw.text((start_x + i * cell_w + 28, 75), feature, fill="#526174", font=small)
    for row_i, (label, count, means) in enumerate(rows):
        y = top + row_i * row_h
        text = label
        while draw.textbbox((0, 0), text, font=small)[2] > 250 and len(text) > 8:
            text = text[:-4] + "..."
        draw.text((label_x, y + 7), text, fill="#263E5B", font=small)
        draw.text((count_x, y + 7), str(count), fill="#263E5B", font=small)
        for i, feature in enumerate(features):
            value = means[feature]
            ratio = (value - low) / (high - low) if high > low else 0.5
            color = "#{:02X}{:02X}{:02X}".format(int(231 - ratio * 191), int(240 - ratio * 117), int(247 - ratio * 106))
            x = start_x + i * cell_w
            draw.rounded_rectangle((x, y + 2, x + cell_w - 8, y + 27), radius=4, fill=color)
            draw.text((x + 20, y + 7), "{:.2f}".format(value), fill="#102A35", font=small)
    y = top + len(rows) * row_h + 15
    draw.text((18, y), "LOW {:.2f}   ━━━━━━━━━━━━━━━━━   HIGH {:.2f}".format(low, high), fill="#526174", font=small)
    return image


def eta_squared(records, targets, features):
    labels = [label_for(record, targets) for record in records]
    result = {}
    for feature in features:
        values = [float(record["features"][feature]) for record in records]
        overall = sum(values) / float(len(values)) if values else 0.0
        total_ss = sum((value - overall) ** 2 for value in values)
        between = 0.0
        for group_label in set(labels):
            group = [float(record["features"][feature]) for record, label in zip(records, labels) if label == group_label]
            if group:
                mean = sum(group) / float(len(group))
                between += len(group) * (mean - overall) ** 2
        result[feature] = between / total_ss if total_ss else 0.0
    return result


def draw_association(x_name, y_name, records, targets, features):
    scores = eta_squared(records, targets, features)
    width, height = 760, 130 + 58 * len(features)
    image = Image.new("RGB", (width, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    _font, small, title = base.load_fonts()
    draw.text((18, 14), "{} | {} FEATURE-CLASS ASSOCIATION".format(x_name, y_name), fill="#1F3E5B", font=title)
    draw.text((18, 45), "η² = between-class variance / total variance; descriptive score (0 to 1)", fill="#526174", font=small)
    for index, feature in enumerate(features):
        y = 92 + index * 58
        score = scores[feature]
        draw.text((25, y + 7), feature, fill="#263E5B", font=small)
        draw.rounded_rectangle((150, y, 650, y + 28), radius=6, fill="#E3E8EF")
        if score:
            draw.rounded_rectangle((150, y, 150 + int(500 * score), y + 28), radius=6, fill="#4D82A8")
        draw.text((665, y + 7), "{:.3f}".format(score), fill="#263E5B", font=small)
    return image


def scatter_pairs(features):
    return [(features[left], features[right]) for left in range(len(features)) for right in range(left + 1, len(features))]


def draw_pairwise(x_name, y_name, records, targets, features):
    pairs = scatter_pairs(features)
    colors = class_colors(records, targets)
    _font, small, title = base.load_fonts()
    cols, plot_h, margin = 2, 205, 24
    legend_rows = int(math.ceil(len(colors) / 4.0))
    top = 78 + legend_rows * 19
    rows = int(math.ceil(len(pairs) / float(cols)))
    image = Image.new("RGB", (1040, max(350, top + rows * plot_h + 24)), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    draw.text((18, 14), "{} | {} PAIRWISE FEATURE OVERLAP".format(x_name, y_name), fill="#1F3E5B", font=title)
    draw.text((18, 44), "Each dot is a sample; color identifies target class. Distinct clouds suggest less overlap for that feature pair.", fill="#526174", font=small)
    for index, (label, color) in enumerate(colors.items()):
        x = 20 + (index % 4) * 250
        y = 70 + (index // 4) * 19
        draw.ellipse((x, y + 3, x + 11, y + 14), fill=color)
        shown = label
        while draw.textbbox((0, 0), shown, font=small)[2] > 220 and len(shown) > 8:
            shown = shown[:-4] + "..."
        draw.text((x + 16, y), shown, fill="#263E5B", font=small)
    for index, (xfeature, yfeature) in enumerate(pairs):
        col, row = index % cols, index // cols
        left, ytop = margin + col * 510, top + row * plot_h
        rect = (left + 35, ytop + 22, left + 490, ytop + 170)
        draw.text((rect[0], ytop), "{} vs {}".format(xfeature, yfeature), fill="#315C7D", font=small)
        draw.rectangle(rect, outline="#8493A3")
        xv = [float(item["features"][xfeature]) for item in records]
        yv = [float(item["features"][yfeature]) for item in records]
        xmin, xmax = min(xv), max(xv)
        ymin, ymax = min(yv), max(yv)
        if xmin == xmax: xmax = xmin + 1.0
        if ymin == ymax: ymax = ymin + 1.0
        for record in records:
            px = rect[0] + (float(record["features"][xfeature]) - xmin) / (xmax - xmin) * (rect[2] - rect[0])
            py = rect[3] - (float(record["features"][yfeature]) - ymin) / (ymax - ymin) * (rect[3] - rect[1])
            color = colors[label_for(record, targets)]
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
        draw.text((rect[0] + 190, rect[3] + 3), xfeature, fill="#526174", font=small)
        draw.text((left, ytop + 80), yfeature, fill="#526174", font=small)
    return image


def jacobi_eigen(matrix):
    n = len(matrix)
    a = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max(1, 20 * n * n)):
        p, q, largest = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > largest:
                    p, q, largest = i, j, abs(a[i][j])
        if largest < 1e-10:
            break
        theta = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(theta), math.sin(theta)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c*c*app - 2*s*c*apq + s*s*aqq
        a[q][q] = s*s*app + 2*s*c*apq + c*c*aqq
        a[p][q] = a[q][p] = 0.0
        for k in range(n):
            if k != p and k != q:
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c*akp - s*akq
                a[k][q] = a[q][k] = s*akp + c*akq
            vkp, vkq = vectors[k][p], vectors[k][q]
            vectors[k][p] = c*vkp - s*vkq
            vectors[k][q] = s*vkp + c*vkq
    pairs = [(a[i][i], [vectors[row][i] for row in range(n)]) for i in range(n)]
    return sorted(pairs, key=lambda pair: pair[0], reverse=True)


def pca_points(records, features):
    if not records:
        return [], (0.0, 0.0)
    rows = [[float(record["features"][feature]) for feature in features] for record in records]
    n, d = len(rows), len(features)
    means = [sum(row[j] for row in rows) / n for j in range(d)]
    stds = []
    for j in range(d):
        variance = sum((row[j] - means[j]) ** 2 for row in rows) / float(max(1, n - 1))
        stds.append(math.sqrt(variance) if variance > 0 else 1.0)
    normalized = [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in rows]
    covariance = [[sum(row[i] * row[j] for row in normalized) / float(max(1, n - 1)) for j in range(d)] for i in range(d)]
    eig = jacobi_eigen(covariance)
    axes = [pair[1] for pair in eig[:2]]
    total = sum(max(0.0, value) for value, _vector in eig)
    explained = tuple(max(0.0, eig[i][0]) / total if total else 0.0 for i in range(min(2, d)))
    if len(explained) == 1:
        explained += (0.0,)
    points = [tuple(sum(row[j] * axes[axis][j] for j in range(d)) for axis in range(2)) for row in normalized]
    return points, explained


def pixel_points(points, rect):
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax: xmax = xmin + 1.0
    if ymin == ymax: ymax = ymin + 1.0
    xpad, ypad = (xmax - xmin) * 0.06, (ymax - ymin) * 0.06
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad
    return [
        (rect[0] + (x - xmin) / (xmax - xmin) * (rect[2] - rect[0]),
         rect[3] - (y - ymin) / (ymax - ymin) * (rect[3] - rect[1]))
        for x, y in points
    ]


def draw_projection(x_name, y_name, records, targets, features, hulls=False):
    points, explained = pca_points(records, features)
    colors = class_colors(records, targets)
    font, small, title = base.load_fonts()
    legend_rows = int(math.ceil(len(colors) / 4.0))
    image = Image.new("RGB", (1000, 540 + legend_rows * 20), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    label = "CLUSTER BOUNDARIES" if hulls else "PCA CLASS SEPARATION"
    draw.text((18, 14), "{} | {} {}".format(x_name, y_name, label), fill="#1F3E5B", font=title)
    draw.text((18, 44), "Standardized {} → PC1 {:.1f}% / PC2 {:.1f}% variance".format(
        ", ".join(features), explained[0] * 100, explained[1] * 100), fill="#526174", font=small)
    if hulls:
        draw.text((18, 64), "Outline = convex hull of each class; hulls may be influenced by outliers.", fill="#526174", font=small)
    plot = (75, 105, 950, 430)
    draw.rectangle(plot, outline="#8493A3")
    for tick in range(1, 5):
        gx = plot[0] + (plot[2] - plot[0]) * tick / 5.0
        gy = plot[1] + (plot[3] - plot[1]) * tick / 5.0
        draw.line([(gx, plot[1]), (gx, plot[3])], fill="#E0E5EB")
        draw.line([(plot[0], gy), (plot[2], gy)], fill="#E0E5EB")
    if points:
        pix = pixel_points(points, plot)
        grouped = {}
        for point, record in zip(pix, records):
            grouped.setdefault(label_for(record, targets), []).append(point)
        for class_name, members in grouped.items():
            color = colors[class_name]
            if hulls:
                hull = convex_hull(members)
                if len(hull) > 2:
                    draw.polygon(hull, outline=color, width=3)
                elif len(hull) == 2:
                    draw.line(hull, fill=color, width=3)
            stride = max(1, int(math.ceil(len(members) / 1600.0)))
            for px, py in members[::stride]:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
    draw.text((410, 435), "PC1", fill="#526174", font=small)
    draw.text((20, 250), "PC2", fill="#526174", font=small)
    for index, (class_name, color) in enumerate(colors.items()):
        x, y = 20 + (index % 4) * 245, 470 + (index // 4) * 20
        draw.rectangle((x, y + 2, x + 12, y + 14), fill=color)
        shown = class_name
        while draw.textbbox((0, 0), shown, font=small)[2] > 215 and len(shown) > 8:
            shown = shown[:-4] + "..."
        draw.text((x + 18, y), shown, fill="#263E5B", font=small)
    return image


def convex_hull(points):
    """Monotonic-chain convex hull for projected points."""
    unique = sorted(set((int(round(x)), int(round(y))) for x, y in points))
    if len(unique) <= 1:
        return unique
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def build_graphs(x_name, features, y_name, targets, records):
    return {
        "Class distribution": draw_distribution(x_name, y_name, records, targets),
        "Class relationships": draw_tree(x_name, y_name, records, targets),
        "Class feature profile": draw_profile(x_name, y_name, records, targets, features),
        "Feature-class association": draw_association(x_name, y_name, records, targets, features),
        "Pairwise feature overlap": draw_pairwise(x_name, y_name, records, targets, features),
        "PCA class separation": draw_projection(x_name, y_name, records, targets, features, False),
        "Cluster boundaries": draw_projection(x_name, y_name, records, targets, features, True),
        "Y distribution by X": draw_y_distribution_by_x(x_name, y_name, records, targets, features),
    }


def draw_y_distribution_by_x(x_name, y_name, records, targets, features):
    """Show the Y-class distribution within value ranges of each numeric X."""
    _font, small, title = base.load_fonts()
    labels = [label_for(record, targets) for record in records]
    totals = Counter(labels)
    shown_classes = [label for label, _count in totals.most_common()]
    colors = {
        label: "#{:02X}{:02X}{:02X}".format(*(
            int(channel * 255) for channel in colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.68, 0.78)
        ))
        for index, label in enumerate(shown_classes)
    }
    panel_h, top, width = 158, 104, 1480
    chart_bottom = top + panel_h * len(features)
    legend_right = len(shown_classes) > 4
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    legend_text_width = max((measure_draw.textbbox((0, 0), label, font=small)[2] for label in shown_classes), default=0)
    width = max(width, 950 + legend_text_width + 40) if legend_right else width
    height = max(chart_bottom + 32, top + 45 + 22 * len(shown_classes)) if legend_right else chart_bottom + 65
    image = Image.new("RGB", (width, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    draw.text((18, 14), "{} | {} | Y CLASS MIX ACROSS X RANGES".format(x_name, y_name), fill="#1F3E5B", font=title)
    draw.text((18, 45), "Range label = X-value interval | n = samples in interval | bar colors = Y-class percentage", fill="#526174", font=small)
    for feature_index, feature in enumerate(features):
        y0 = top + feature_index * panel_h
        values = sorted(float(record["features"][feature]) for record in records)
        draw.text((20, y0), feature, fill="#315C7D", font=title)
        if not values:
            continue
        cuts = [values[min(len(values) - 1, int(len(values) * q / 4.0))] for q in (1, 2, 3)]
        bins = [[] for _ in range(4)]
        for record, label in zip(records, labels):
            value = float(record["features"][feature])
            index = 0 if value <= cuts[0] else 1 if value <= cuts[1] else 2 if value <= cuts[2] else 3
            bins[index].append(label if label in colors else "Other classes")
        left, bar_top, bar_width, bar_height, gap = 280, y0 + 27, 600, 22, 7
        for index, group in enumerate(bins):
            y = bar_top + index * (bar_height + gap)
            lower = min(values) if index == 0 else cuts[index - 1]
            upper = max(values) if index == 3 else cuts[index]
            interval = "{:.2f} ≤ X ≤ {:.2f}".format(lower, upper) if index == 0 else "{:.2f} < X ≤ {:.2f}".format(lower, upper)
            draw.text((20, y + 3), "{} (n={})".format(interval, len(group)), fill="#526174", font=small)
            if group:
                counts = Counter(group)
                x = left
                for label in shown_classes:
                    amount = counts[label]
                    segment = int(bar_width * amount / len(group))
                    if amount and segment == 0:
                        segment = 1
                    if amount:
                        draw.rectangle((x, y, min(left + bar_width, x + segment), y + bar_height), fill=colors[label])
                        x += segment
    if legend_right:
        legend_x, legend_y = 930, top
        draw.text((legend_x, legend_y), "Y CLASS LEGEND", fill="#315C7D", font=title)
        for index, label in enumerate(shown_classes):
            y = legend_y + 30 + index * 22
            draw.rectangle((legend_x, y + 2, legend_x + 13, y + 15), fill=colors[label])
            draw.text((legend_x + 22, y), label, fill="#526174", font=small)
    else:
        legend_y, legend_x = chart_bottom + 12, 280
        for label in shown_classes:
            item_width = 20 + draw.textbbox((0, 0), label, font=small)[2] + 20
            if legend_x + item_width > width - 20:
                legend_y += 24
                legend_x = 280
            draw.rectangle((legend_x, legend_y + 2, legend_x + 13, legend_y + 15), fill=colors[label])
            draw.text((legend_x + 20, legend_y), label, fill="#526174", font=small)
            legend_x += item_width
    return image


def save_powerpoint(path, selected_cases, records, charts_dir):
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.util import Inches, Pt
    except ImportError:
        raise RuntimeError("PowerPoint export needs python-pptx. Install it in PyCharm: pip install python-pptx")
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333), Inches(7.5)
    cover = presentation.slides.add_slide(presentation.slide_layouts[0])
    cover.shapes.title.text = "Meta AI Feature and Class Separation Report"
    cover.placeholders[1].text = "Cases: {}\nSamples: {}\nCharts are descriptive diagnostics and do not establish model accuracy.".format(
        len(selected_cases), len(records))
    for x_name, features, y_name, targets in selected_cases:
        charts = build_graphs(x_name, features, y_name, targets, records)
        for title, image in charts.items():
            image_path = os.path.join(charts_dir, "{}_{}_{}.png".format(x_name, y_name, re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")))
            image.save(image_path)
            slide = presentation.slides.add_slide(presentation.slide_layouts[6])
            box = slide.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.5))
            p = box.text_frame.paragraphs[0]
            p.text = "{} + {} | {}".format(x_name, y_name, title)
            p.font.size, p.font.bold = Pt(22), True
            p.font.color.rgb = RGBColor(31, 62, 91)
            image_width, image_height = image.size
            scale = min(8.7 / image_width, 6.0 / image_height)
            slide.shapes.add_picture(image_path, Inches(0.35), Inches(0.95), width=Inches(image_width * scale), height=Inches(image_height * scale))
            panel = slide.shapes.add_textbox(Inches(9.15), Inches(1.0), Inches(3.8), Inches(5.7))
            frame = panel.text_frame
            frame.word_wrap = True
            frame.text = "Legend\n{}\n\nHow to read\n{}\n\nCase\n{} features: {}\n{} target classes across {} samples.".format(
                GUIDES[title][0], GUIDES[title][1], ", ".join(features),
                len(class_counts(records, targets)), len(records))
            for i, paragraph in enumerate(frame.paragraphs):
                paragraph.font.size = Pt(13 if i else 15)
                if i == 0:
                    paragraph.font.bold = True
    presentation.save(path)


class Dashboard:
    def __init__(self, root, output_path, records):
        self.root, self.output_path, self.records = root, output_path, records
        self.images = []
        root.title("Meta AI Feature Combination Dashboard")
        root.geometry("1420x940")
        container = ttk.Frame(root, padding=10)
        container.pack(fill="both", expand=True)
        controls = ttk.Frame(container)
        controls.pack(side="left", fill="y", padx=(0, 10))
        xbox = ttk.LabelFrame(controls, text="X feature sets", padding=6)
        xbox.pack(fill="x")
        self.x_list = tk.Listbox(xbox, selectmode="multiple", height=7, width=26, exportselection=False)
        for name, features in FEATURE_SETS:
            self.x_list.insert("end", "{} ({})".format(name, len(features)))
        self.x_list.selection_set(len(FEATURE_SETS) - 1)
        self.x_list.pack(fill="x")
        ybox = ttk.LabelFrame(controls, text="Y target sets", padding=6)
        ybox.pack(fill="x", pady=(8, 0))
        self.y_list = tk.Listbox(ybox, selectmode="multiple", height=9, width=26, exportselection=False)
        for mode, _targets in base.TARGET_MODES:
            self.y_list.insert("end", mode)
        self.y_list.selection_set(0, tk.END)
        self.y_list.pack(fill="x")
        ttk.Button(controls, text="Show selected graphs", command=self.refresh).pack(fill="x", pady=(8, 3))
        ttk.Button(controls, text="Export PowerPoint", command=self.export_ppt).pack(fill="x", pady=3)
        ttk.Label(controls, text="Select 3 or 4 X features.\n\nDefault: all 4 X features and all existing Y cases.", wraplength=210).pack(anchor="w", pady=(8, 0))
        self.view = ttk.Frame(container)
        self.view.pack(side="left", fill="both", expand=True)
        self.status = ttk.Label(root, text="Dataset workbook: {}".format(os.path.basename(output_path)), padding=8)
        self.status.pack(fill="x")
        root.after_idle(self.safe_refresh)

    def selected_cases(self):
        return [
            (FEATURE_SETS[i][0], FEATURE_SETS[i][1], base.TARGET_MODES[j][0], base.TARGET_MODES[j][1])
            for i in self.x_list.curselection()
            for j in self.y_list.curselection()
        ]

    def safe_refresh(self):
        try:
            self.refresh()
        except Exception as error:
            self.status.configure(text="Graph generation error: {}".format(error))
            messagebox.showerror("Graph generation failed", str(error), parent=self.root)

    def add_graph_tab(self, notebook, title, image, guide, interpretation):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)
        graph_area = ttk.Frame(tab)
        graph_area.pack(fill="both", expand=True)
        canvas = tk.Canvas(graph_area, background="#EEF2F6", highlightthickness=0)
        vbar = ttk.Scrollbar(graph_area, orient="vertical", command=canvas.yview)
        hbar = ttk.Scrollbar(graph_area, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        graph_area.rowconfigure(0, weight=1)
        graph_area.columnconfigure(0, weight=1)
        photo = ImageTk.PhotoImage(image, master=canvas)
        self.images.append(photo)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.configure(scrollregion=(0, 0, image.width, image.height))
        ttk.Label(tab, text="Legend: {}\nInterpretation: {}".format(guide, interpretation), wraplength=1050,
                  justify="left", padding=8).pack(fill="x", side="bottom")

    def refresh(self):
        for child in self.view.winfo_children():
            child.destroy()
        self.images = []
        cases = self.selected_cases()
        if not cases:
            ttk.Label(self.view, text="Select at least one X set and one Y set.", padding=20).pack(anchor="nw")
            self.status.configure(text="No analysis cases selected.")
            return
        charts_dir = os.path.splitext(self.output_path)[0] + "_charts"
        os.makedirs(charts_dir, exist_ok=True)
        outer = ttk.Notebook(self.view)
        outer.pack(fill="both", expand=True)
        cases_by_x = {}
        for x_name, features, y_name, targets in cases:
            cases_by_x.setdefault(x_name, {"features": features, "y_cases": []})
            cases_by_x[x_name]["y_cases"].append((y_name, targets))

        # Split the formerly long single row of X/Y case tabs into levels:
        # X set → Y set → graph. This keeps all tab labels visible.
        for x_name, x_data in cases_by_x.items():
            x_tab = ttk.Frame(outer)
            outer.add(x_tab, text=x_name)
            y_notebook = ttk.Notebook(x_tab)
            y_notebook.pack(fill="both", expand=True)
            for y_name, targets in x_data["y_cases"]:
                y_tab = ttk.Frame(y_notebook)
                y_notebook.add(y_tab, text=y_name)
                graph_tabs = ttk.Notebook(y_tab)
                graph_tabs.pack(fill="both", expand=True)
                # Pass this function's records explicitly through every graph builder.
                charts = build_graphs(x_name, x_data["features"], y_name, targets, self.records)
                for title, image in charts.items():
                    safe_title = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
                    image.save(os.path.join(charts_dir, "{}_{}_{}.png".format(x_name, y_name, safe_title)))
                    guide, interpretation = GUIDES[title]
                    self.add_graph_tab(graph_tabs, title, image, guide, interpretation)
        self.status.configure(text="Showing {} X/Y case(s), with eight graph types per case. Charts saved in {}".format(
            len(cases), os.path.basename(charts_dir)))

    def export_ppt(self):
        cases = self.selected_cases()
        if not cases:
            messagebox.showinfo("Select cases", "Select at least one X set and one Y set first.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            title="Save PowerPoint analysis report",
            initialdir=os.path.dirname(self.output_path),
            initialfile=os.path.splitext(os.path.basename(self.output_path))[0] + "_feature_cluster_report.pptx",
            defaultextension=".pptx", filetypes=[("PowerPoint", "*.pptx")], parent=self.root)
        if not path:
            return
        charts_dir = os.path.splitext(self.output_path)[0] + "_charts"
        os.makedirs(charts_dir, exist_ok=True)
        try:
            save_powerpoint(path, cases, self.records, charts_dir)
        except Exception as error:
            messagebox.showerror("PowerPoint export failed", str(error), parent=self.root)
            return
        messagebox.showinfo("Report saved", "Saved PowerPoint report:\n{}".format(path), parent=self.root)
        self.status.configure(text="PowerPoint report saved: {}".format(os.path.basename(path)))


def choose_sources(root):
    first = filedialog.askopenfilename(title="Select first evaluation workbook", filetypes=[("Excel workbooks", "*.xlsx")], parent=root)
    if not first:
        return []
    second = filedialog.askopenfilename(title="Optional second workbook (Cancel to use only first)", filetypes=[("Excel workbooks", "*.xlsx")], parent=root)
    paths = [first]
    if second and os.path.abspath(second) != os.path.abspath(first):
        paths.append(second)
    return paths


def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = choose_sources(root)
    if not paths:
        root.destroy()
        return
    try:
        records, skipped, sources = read_union(paths)
    except Exception as error:
        messagebox.showerror("Could not read workbooks", str(error), parent=root)
        root.destroy()
        return
    output_dir = os.path.dirname(paths[0])
    default_name = os.path.splitext(os.path.basename(paths[0]))[0] + "_meta_ai_feature_sets_{}.xlsx".format(
        datetime.date.today().strftime("%Y%m%d"))
    default_path = unique_excel_path(os.path.join(output_dir, default_name))
    output_path = filedialog.asksaveasfilename(
        title="Save feature-combination training workbook",
        initialdir=output_dir,
        initialfile=os.path.basename(default_path),
        defaultextension=".xlsx", filetypes=[("Excel workbooks", "*.xlsx")],
        confirmoverwrite=False, parent=root)
    if not output_path:
        root.destroy()
        return
    output_path = unique_excel_path(output_path)
    try:
        write_workbook(output_path, records, skipped, sources)
    except Exception as error:
        messagebox.showerror("Could not save workbook", str(error), parent=root)
        root.destroy()
        return
    messagebox.showinfo("Workbook saved", "Saved Excel workbook:\n{}".format(output_path), parent=root)
    root.attributes("-topmost", False)
    root.deiconify()
    Dashboard(root, output_path, records)
    root.mainloop()


if __name__ == "__main__":
    main()
