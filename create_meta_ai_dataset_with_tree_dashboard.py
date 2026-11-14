"""Build Meta AI datasets from one or two evaluation workbooks.

Creates the same seven X1-X4 to Y-component datasets as
create_meta_ai_training_dataset.py, plus a selectable dashboard containing
both class-count bar charts and label-relationship trees. The existing
generator is not modified.
"""

from __future__ import print_function

import os
import re
import datetime
import tkinter as tk
import colorsys
import math
from collections import Counter
from tkinter import filedialog, messagebox, ttk

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from PIL import Image, ImageDraw, ImageFont, ImageTk

import create_meta_ai_training_dataset as base


GRAPH_GUIDES = {
    "Class distribution": (
        "Legend: each bar is the number of samples in a target class. Longer bars mean more samples.",
        "Interpretation: compare the largest and smallest classes. A large imbalance can bias training and makes stratified train/validation splits important.",
    ),
    "Class relationships": (
        "Legend: each node shows a label value and its sample count; each branch adds the next target component.",
        "Interpretation: inspect whether combinations have enough examples. Very small or empty branches may be difficult for a Meta AI model to learn reliably.",
    ),
    "Class feature profile": (
        "Legend: each cell is the mean of one X feature for one target class. The color scale runs from the lowest to highest mean in this view.",
        "Interpretation: different class means suggest useful feature differences. This chart does not show spread within a class, so it cannot establish low overlap by itself.",
    ),
    "Feature-class association": (
        "Legend: η² ranges from 0 to 1 and estimates the fraction of a feature's variance associated with differences between target classes.",
        "Interpretation: larger η² suggests stronger class-level separation for that feature. It is descriptive and does not measure predictive accuracy or within-class overlap.",
    ),
    "Pairwise feature overlap": (
        "Legend: each dot is a sample; color identifies its target class. Each panel plots one pair of X features.",
        "Interpretation: distinct color clouds suggest separation in that feature pair; mixed colors suggest overlap. This is a two-feature-at-a-time view.",
    ),
    "PCA class separation": (
        "Legend: each dot is a sample; color identifies its target class and the outlined circle marks the class centroid. PC1/PC2 show variance captured.",
        "Interpretation: separated color groups in this 2D projection are encouraging. Overlap may reflect real similarity, and separation in the remaining dimensions is not shown.",
    ),
    "Cluster boundaries": (
        "Legend: each dot is a sample; color identifies its target class. The matching outline is that class's convex hull after projecting standardized X1-X4 onto PC1 and PC2.",
        "Interpretation: well-separated outlines suggest distinct clusters in this 2D projection; intersecting outlines indicate possible overlap. Convex hulls can expand around outliers and are a visual aid, not a numeric overlap score.",
    ),
}


TREE_WIDTH = 820
NODE_WIDTH = 170
TREE_ROW_GAP = 42
PROFILE_WIDTH = 820
ASSOCIATION_WIDTH = 760


def unique_excel_path(path):
    """Return an unused Excel filename rather than overwriting an existing one."""
    stem, extension = os.path.splitext(path)
    candidate = path
    suffix = 1
    while os.path.exists(candidate):
        candidate = "{}_{:02d}{}".format(stem, suffix, extension)
        suffix += 1
    return candidate


def read_and_union_sources(paths):
    all_records = []
    all_skipped = []
    source_details = []
    used_display_names = set()

    for path in paths:
        try:
            # read_source locates the semantic Y1/Y2/Y3 columns by their
            # accepted header aliases. The original header wording can differ
            # between workbooks without changing the canonical label mapping.
            records, skipped, y_headers, sheet_name, header_row = base.read_source(path)
        except Exception as exc:
            raise ValueError("Could not read workbook '{}': {}".format(os.path.basename(path), exc))

        file_name = os.path.basename(path)
        if file_name in used_display_names:
            stem, extension = os.path.splitext(file_name)
            suffix = 2
            while "{} ({}){}".format(stem, suffix, extension) in used_display_names:
                suffix += 1
            file_name = "{} ({}){}".format(stem, suffix, extension)
        used_display_names.add(file_name)
        for record in records:
            record["source_file"] = file_name
        all_records.extend(records)
        for item in skipped:
            all_skipped.append({
                "Source file": file_name,
                "Source row": item["Source row"],
                "Reason": item["Reason"],
            })
        source_details.append({
            "Source file": file_name,
            "Full path": os.path.abspath(path),
            "Source sheet": sheet_name,
            "Header row": header_row,
            "Valid samples": len(records),
            "Skipped rows": len(skipped),
            "Y columns": [y_headers[target] for target in base.TARGETS],
        })
    return all_records, all_skipped, source_details


def class_paths(records, components):
    counts = Counter(
        tuple(record["labels"][component] for component in components)
        for record in records
    )
    return sorted(counts.items(), key=lambda item: tuple(value.casefold() for value in item[0]))


def draw_relationship_tree(mode, components, records):
    paths = class_paths(records, components)
    depth = len(components) + 1  # Include the dataset root.
    height = max(370, 130 + TREE_ROW_GAP * max(1, len(paths)))
    image = Image.new("RGB", (TREE_WIDTH, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    font, small_font, title_font = base.load_fonts()

    draw.text((18, 16), "{} LABEL RELATIONSHIP TREE".format(mode.replace("_", " + ")),
              fill="#1F3E5B", font=title_font)
    draw.text((18, 48), "{} samples  |  {} observed classes  |  counts shown at every node".format(
        len(records), len(paths)
    ), fill="#526174", font=small_font)

    labels = ["Dataset root"] + list(components)
    margin = 16
    gap = (TREE_WIDTH - 2 * margin - depth * NODE_WIDTH) / float(max(1, depth - 1))
    xs = [int(margin + index * (NODE_WIDTH + gap)) for index in range(depth)]
    for level, label in enumerate(labels):
        draw.text((xs[level], 83), label.upper(), fill="#526174", font=small_font)

    if not paths:
        draw.text((20, 140), "No classes found", fill="#A12620", font=font)
        return image

    leaf_paths = [(tuple(["All samples"] + list(values)), count) for values, count in paths]
    y_positions = {path: 110 + index * TREE_ROW_GAP for index, (path, _count) in enumerate(leaf_paths)}
    centers, totals = {}, {}
    for path, count in leaf_paths:
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
        x1 = xs[level - 1] + NODE_WIDTH
        x2 = xs[level]
        elbow = x1 + (x2 - x1) / 2.0
        draw.line([(x1, parent_y), (elbow, parent_y), (elbow, child_y), (x2, child_y)],
                  fill="#A9B4C2", width=2)

    colors = ("#E8F6EF", "#EAF0FA", "#FFF4E5", "#F0ECFA")
    for (level, prefix), center_y in centers.items():
        x1, x2 = xs[level], xs[level] + NODE_WIDTH
        y1, y2 = center_y - 17, center_y + 17
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8,
                               fill=colors[level % len(colors)], outline="#A9B4C2", width=1)
        label = prefix[-1]
        while draw.textbbox((0, 0), label, font=small_font)[2] > NODE_WIDTH - 54 and len(label) > 7:
            label = label[:-4] + "..."
        draw.text((x1 + 8, y1 + 10), "{} ({})".format(label, totals[(level, prefix)]),
                  fill="#263E5B", font=small_font)
    return image


def color_between(low, high, ratio):
    ratio = max(0.0, min(1.0, ratio))
    low_rgb = tuple(int(low[index:index + 2], 16) for index in (1, 3, 5))
    high_rgb = tuple(int(high[index:index + 2], 16) for index in (1, 3, 5))
    rgb = tuple(round(low_rgb[index] + (high_rgb[index] - low_rgb[index]) * ratio)
                for index in range(3))
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def draw_class_feature_profile(mode, components, records):
    """Heatmap of mean X1-X4 values for each Target Class."""
    grouped = {}
    for record in records:
        label = base.target_class(record["labels"], components)
        summary = grouped.setdefault(label, {"count": 0, "sums": dict.fromkeys(base.FEATURES, 0.0)})
        summary["count"] += 1
        for feature in base.FEATURES:
            summary["sums"][feature] += record["features"][feature]

    rows = []
    for label, data in grouped.items():
        means = {feature: data["sums"][feature] / data["count"] for feature in base.FEATURES}
        rows.append((label, data["count"], means))
    rows.sort(key=lambda item: (-item[1], item[0].casefold()))
    all_means = [value for _label, _count, means in rows for value in means.values()]
    low = min(all_means or [0.0])
    high = max(all_means or [1.0])

    _, small_font, title_font = base.load_fonts()
    row_height = 32
    top = 105
    height = max(380, top + row_height * max(1, len(rows)) + 65)
    image = Image.new("RGB", (PROFILE_WIDTH, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    draw.text((18, 16), "{} CLASS FEATURE PROFILE".format(mode.replace("_", " + ")),
              fill="#1F3E5B", font=title_font)
    draw.text((18, 48), "Mean X1-X4 by Target Class; cell color is relative to all means in this view",
              fill="#526174", font=small_font)

    class_x, count_x, feature_x = 18, 275, 350
    feature_width = 112
    draw.text((class_x, 82), "TARGET CLASS", fill="#526174", font=small_font)
    draw.text((count_x, 82), "N", fill="#526174", font=small_font)
    for index, feature in enumerate(base.FEATURES):
        draw.text((feature_x + index * feature_width + 28, 82), feature,
                  fill="#526174", font=small_font)

    for row_index, (label, count, means) in enumerate(rows):
        y = top + row_index * row_height
        display_label = label
        while draw.textbbox((0, 0), display_label, font=small_font)[2] > 245 and len(display_label) > 8:
            display_label = display_label[:-4] + "..."
        draw.text((class_x, y + 8), display_label, fill="#263E5B", font=small_font)
        draw.text((count_x, y + 8), str(count), fill="#263E5B", font=small_font)
        for index, feature in enumerate(base.FEATURES):
            value = means[feature]
            ratio = (value - low) / (high - low) if high > low else 0.5
            x = feature_x + index * feature_width
            draw.rounded_rectangle((x, y + 2, x + feature_width - 8, y + 28),
                                   radius=4, fill=color_between("#E7F0F7", "#287B8E", ratio))
            text_color = "#FFFFFF" if ratio > 0.68 else "#102A35"
            draw.text((x + 20, y + 8), "{:.1f}".format(value), fill=text_color, font=small_font)

    legend_y = top + row_height * len(rows) + 20
    draw.text((18, legend_y), "LOW {:.2f}".format(low), fill="#526174", font=small_font)
    draw.rounded_rectangle((95, legend_y, 340, legend_y + 16), radius=5,
                           fill="#287B8E", outline="#A9B4C2")
    draw.text((350, legend_y), "HIGH {:.2f}".format(high), fill="#526174", font=small_font)
    return image


def feature_class_associations(components, records):
    """Compute eta-squared for each numeric feature against Target Class."""
    groups = {}
    totals = {feature: [] for feature in base.FEATURES}
    for record in records:
        label = base.target_class(record["labels"], components)
        group = groups.setdefault(label, {feature: [] for feature in base.FEATURES})
        for feature in base.FEATURES:
            value = float(record["features"][feature])
            group[feature].append(value)
            totals[feature].append(value)

    results = {}
    for feature in base.FEATURES:
        values = totals[feature]
        overall = sum(values) / float(len(values)) if values else 0.0
        total_ss = sum((value - overall) ** 2 for value in values)
        between_ss = 0.0
        for group in groups.values():
            group_values = group[feature]
            group_mean = sum(group_values) / float(len(group_values))
            between_ss += len(group_values) * (group_mean - overall) ** 2
        results[feature] = between_ss / total_ss if total_ss > 0 else 0.0
    return results


def draw_feature_class_association(mode, components, records):
    scores = feature_class_associations(components, records)
    _, small_font, title_font = base.load_fonts()
    image = Image.new("RGB", (ASSOCIATION_WIDTH, 380), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), "{} FEATURE-CLASS ASSOCIATION".format(mode.replace("_", " + ")),
              fill="#1F3E5B", font=title_font)
    draw.text((20, 50), "η²: fraction of feature variance explained by Target Class (descriptive measure)",
              fill="#526174", font=small_font)
    left, bar_x, bar_width = 28, 180, 470
    row_top, row_gap = 112, 54
    for index, feature in enumerate(base.FEATURES):
        y = row_top + index * row_gap
        score = scores[feature]
        draw.text((left, y + 8), feature, fill="#263E5B", font=small_font)
        draw.rounded_rectangle((bar_x, y, bar_x + bar_width, y + 28), radius=7, fill="#E3E8EF")
        fill_width = int(bar_width * score)
        if fill_width:
            draw.rounded_rectangle((bar_x, y, bar_x + fill_width, y + 28), radius=7, fill="#4D82A8")
        draw.text((bar_x + bar_width + 12, y + 8), "{:.3f}".format(score),
                  fill="#263E5B", font=small_font)
    draw.text((28, 342), "Higher η² means class groups have more distinct means for that feature.",
              fill="#526174", font=small_font)
    return image


def class_color_map(records, components, limit=None):
    counts = Counter(base.target_class(record["labels"], components) for record in records)
    ordered = sorted(counts, key=lambda label: (-counts[label], label.casefold()))
    selected = ordered if limit is None else ordered[:limit]
    colors = {}
    for index, label in enumerate(selected):
        # Evenly spaced hues keep every class visible instead of hiding rare
        # classes inside a gray "Other" group.
        hue = (index * 0.61803398875) % 1.0
        red, green, blue = colorsys.hls_to_rgb(hue, 0.46, 0.78)
        colors[label] = "#{:02X}{:02X}{:02X}".format(round(red * 255), round(green * 255), round(blue * 255))
    return counts, colors, set(selected)


def draw_pairwise_feature_overlap(mode, components, records):
    """Six pairwise scatters reveal where class samples overlap in feature space."""
    counts, colors, selected = class_color_map(records, components)
    font, small_font, title_font = base.load_fonts()
    width = 1020
    legend_y = 88
    legend_items = [(label, colors[label]) for label in colors]
    legend_cols = 4
    legend_rows = (len(legend_items) + legend_cols - 1) // legend_cols
    plot_top = 142 + 20 * legend_rows
    height = max(690, plot_top + 530)
    image = Image.new("RGB", (width, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    draw.text((20, 14), "{} PAIRWISE FEATURE OVERLAP".format(mode.replace("_", " + ")),
              fill="#1F3E5B", font=title_font)
    draw.text((20, 43), "Each dot is one sample; similarly colored dots that mix together indicate pairwise overlap.",
              fill="#526174", font=small_font)
    draw.text((20, 63), "All {} classes are shown in distinct colors; color mixing indicates overlap.".format(
        len(counts)), fill="#526174", font=small_font)
    for index, (label, color) in enumerate(legend_items):
        col, row = index % legend_cols, index // legend_cols
        x, y = 22 + col * 245, legend_y + row * 20
        draw.rectangle((x, y + 2, x + 12, y + 14), fill=color)
        display = label
        while draw.textbbox((0, 0), display, font=small_font)[2] > 205 and len(display) > 8:
            display = display[:-4] + "..."
        draw.text((x + 18, y), display, fill="#263E5B", font=small_font)

    pairs = [("X1", "X2"), ("X1", "X3"), ("X1", "X4"),
             ("X2", "X3"), ("X2", "X4"), ("X3", "X4")]
    all_values = {feature: [float(record["features"][feature]) for record in records]
                  for feature in base.FEATURES}
    bounds = {}
    for feature, values in all_values.items():
        low, high = min(values or [0.0]), max(values or [1.0])
        if high <= low:
            high = low + 1.0
        pad = (high - low) * 0.04
        bounds[feature] = (low - pad, high + pad)

    positions = {}
    for index, (x_feature, y_feature) in enumerate(pairs):
        col, row = index % 2, index // 2
        left = 28 + col * 500
        top = plot_top + row * 177
        rect = (left + 35, top + 18, left + 470, top + 153)
        positions[(x_feature, y_feature)] = rect
        draw.text((left + 35, top), "{} vs {}".format(x_feature, y_feature),
                  fill="#315C7D", font=small_font)
        draw.rectangle(rect, outline="#8493A3", width=1)
        for tick in range(1, 4):
            gx = rect[0] + (rect[2] - rect[0]) * tick / 4.0
            gy = rect[1] + (rect[3] - rect[1]) * tick / 4.0
            draw.line([(gx, rect[1]), (gx, rect[3])], fill="#E0E5EB", width=1)
            draw.line([(rect[0], gy), (rect[2], gy)], fill="#E0E5EB", width=1)
        draw.text((rect[0] + (rect[2] - rect[0]) / 2 - 8, rect[3] + 3), x_feature,
                  fill="#526174", font=small_font)
        draw.text((left + 3, top + 70), y_feature, fill="#526174", font=small_font)

    per_class = {}
    for record in records:
        label = base.target_class(record["labels"], components)
        per_class.setdefault(label, []).append(record)
    stride_by_class = {
        label: max(1, int(math.ceil(len(items) / 700.0)))
        for label, items in per_class.items()
    }
    for x_feature, y_feature in pairs:
        rect = positions[(x_feature, y_feature)]
        xlow, xhigh = bounds[x_feature]
        ylow, yhigh = bounds[y_feature]
        for label, items in per_class.items():
            color = colors[label]
            stride = stride_by_class[label]
            for record in items[::stride]:
                xvalue = float(record["features"][x_feature])
                yvalue = float(record["features"][y_feature])
                px = rect[0] + (xvalue - xlow) / (xhigh - xlow) * (rect[2] - rect[0])
                py = rect[3] - (yvalue - ylow) / (yhigh - ylow) * (rect[3] - rect[1])
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    return image


def jacobi_eigenvectors(matrix, max_iterations=100):
    """Eigen-decompose a small symmetric matrix using Jacobi rotations."""
    size = len(matrix)
    a = [row[:] for row in matrix]
    vectors = [[1.0 if row == col else 0.0 for col in range(size)] for row in range(size)]
    for _iteration in range(max_iterations):
        p, q, maximum = 0, 1, 0.0
        for row in range(size):
            for col in range(row + 1, size):
                value = abs(a[row][col])
                if value > maximum:
                    p, q, maximum = row, col, value
        if maximum < 1e-10:
            break
        angle = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0
        for index in range(size):
            if index in (p, q):
                continue
            aip, aiq = a[index][p], a[index][q]
            a[index][p] = a[p][index] = c * aip - s * aiq
            a[index][q] = a[q][index] = s * aip + c * aiq
        for index in range(size):
            vip, viq = vectors[index][p], vectors[index][q]
            vectors[index][p] = c * vip - s * viq
            vectors[index][q] = s * vip + c * viq
    eigenpairs = [(a[index][index], [vectors[row][index] for row in range(size)])
                  for index in range(size)]
    return sorted(eigenpairs, key=lambda item: item[0], reverse=True)


def pca_projection(records):
    rows = [[float(record["features"][feature]) for feature in base.FEATURES]
            for record in records]
    if not rows:
        return [], (0.0, 0.0)
    means = [sum(row[col] for row in rows) / float(len(rows)) for col in range(4)]
    stds = []
    for col in range(4):
        variance = sum((row[col] - means[col]) ** 2 for row in rows) / float(max(1, len(rows) - 1))
        stds.append(math.sqrt(variance) if variance > 0 else 1.0)
    standardized = [[(row[col] - means[col]) / stds[col] for col in range(4)] for row in rows]
    covariance = [[
        sum(row[a] * row[b] for row in standardized) / float(max(1, len(rows) - 1))
        for b in range(4)
    ] for a in range(4)]
    eigenpairs = jacobi_eigenvectors(covariance)
    components = [pair[1] for pair in eigenpairs[:2]]
    eigenvalues = [max(0.0, pair[0]) for pair in eigenpairs]
    total_variance = sum(eigenvalues)
    explained = tuple(value / total_variance if total_variance > 0 else 0.0
                      for value in eigenvalues[:2])
    projected = [tuple(sum(row[index] * components[axis][index] for index in range(4))
                       for axis in range(2)) for row in standardized]
    return projected, explained


def draw_pca_class_separation(mode, components, records):
    projected, explained = pca_projection(records)
    counts, colors, selected = class_color_map(records, components)
    legend_rows = (len(colors) + 3) // 4
    width, height = 900, max(570, 520 + 24 * legend_rows)
    image = Image.new("RGB", (width, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    font, small_font, title_font = base.load_fonts()
    draw.text((20, 16), "{} PCA CLASS SEPARATION".format(mode.replace("_", " + ")),
              fill="#1F3E5B", font=title_font)
    draw.text((20, 47), "Standardized X1-X4 projected to two principal components; inspect color mixing for overlap.",
              fill="#526174", font=small_font)
    counts_text = "{} samples, {} classes; every class has its own color".format(
        len(records), len(counts)
    )
    draw.text((20, 68), counts_text, fill="#526174", font=small_font)

    plot = (75, 115, 850, 455)
    draw.rectangle(plot, outline="#8493A3", width=1)
    for tick in range(1, 5):
        gx = plot[0] + (plot[2] - plot[0]) * tick / 5.0
        gy = plot[1] + (plot[3] - plot[1]) * tick / 5.0
        draw.line([(gx, plot[1]), (gx, plot[3])], fill="#E0E5EB", width=1)
        draw.line([(plot[0], gy), (plot[2], gy)], fill="#E0E5EB", width=1)
    draw.text((plot[0] + 300, plot[3] + 5), "PC1 ({:.1f}% variance)".format(explained[0] * 100),
              fill="#526174", font=small_font)
    draw.text((15, plot[1] + 150), "PC2\n{:.1f}%".format(explained[1] * 100),
              fill="#526174", font=small_font)
    if not projected:
        draw.text((100, 160), "No samples to project", fill="#526174", font=font)
        return image

    xs = [point[0] for point in projected]
    ys = [point[1] for point in projected]
    xlow, xhigh = min(xs), max(xs)
    ylow, yhigh = min(ys), max(ys)
    if xhigh == xlow:
        xhigh = xlow + 1.0
    if yhigh == ylow:
        yhigh = ylow + 1.0
    xpad, ypad = (xhigh - xlow) * 0.06, (yhigh - ylow) * 0.06
    xlow, xhigh = xlow - xpad, xhigh + xpad
    ylow, yhigh = ylow - ypad, yhigh + ypad

    class_points = {}
    for index, record in enumerate(records):
        label = base.target_class(record["labels"], components)
        class_points.setdefault(label, []).append(projected[index])
    for label, points in class_points.items():
        color = colors[label]
        stride = max(1, int(math.ceil(len(points) / 1000.0)))
        for px_value, py_value in points[::stride]:
            px = plot[0] + (px_value - xlow) / (xhigh - xlow) * (plot[2] - plot[0])
            py = plot[3] - (py_value - ylow) / (yhigh - ylow) * (plot[3] - plot[1])
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        if points:
            center_x = sum(point[0] for point in points) / float(len(points))
            center_y = sum(point[1] for point in points) / float(len(points))
            cx = plot[0] + (center_x - xlow) / (xhigh - xlow) * (plot[2] - plot[0])
            cy = plot[3] - (center_y - ylow) / (yhigh - ylow) * (plot[3] - plot[1])
            draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline="#152536", width=2)

    legend_y = 490
    for index, (label, color) in enumerate(colors.items()):
        x = 24 + (index % 4) * 218
        y = legend_y + (index // 4) * 24
        draw.rectangle((x, y + 3, x + 12, y + 15), fill=color)
        display = label
        while draw.textbbox((0, 0), display, font=small_font)[2] > 185 and len(display) > 8:
            display = display[:-4] + "..."
        draw.text((x + 18, y), display, fill="#263E5B", font=small_font)
    return image


def convex_hull(points):
    """Return the monotonic-chain convex hull of 2D points."""
    points = sorted(set((int(round(x)), int(round(y))) for x, y in points))
    if len(points) <= 1:
        return points

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def draw_cluster_boundaries(mode, components, records):
    """Show per-class convex-hull outlines over the existing 4-feature PCA."""
    projected, explained = pca_projection(records)
    counts, colors, _selected = class_color_map(records, components)
    legend_rows = (len(colors) + 3) // 4
    width, height = 1000, max(590, 560 + 24 * legend_rows)
    image = Image.new("RGB", (width, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    font, small_font, title_font = base.load_fonts()
    draw.text((20, 14), "{} CLUSTER BOUNDARIES".format(mode.replace("_", " + ")),
              fill="#1F3E5B", font=title_font)
    draw.text((20, 45), "Standardized X1-X4 projected to PC1 and PC2; outline = per-class convex hull",
              fill="#526174", font=small_font)
    draw.text((20, 66), "{} samples | {} classes | hulls are sensitive to outliers and show only a 2D projection".format(
        len(records), len(counts)), fill="#526174", font=small_font)

    plot = (76, 105, 950, 470)
    draw.rectangle(plot, outline="#8493A3", width=1)
    for tick in range(1, 5):
        gx = plot[0] + (plot[2] - plot[0]) * tick / 5.0
        gy = plot[1] + (plot[3] - plot[1]) * tick / 5.0
        draw.line([(gx, plot[1]), (gx, plot[3])], fill="#E0E5EB", width=1)
        draw.line([(plot[0], gy), (plot[2], gy)], fill="#E0E5EB", width=1)
    draw.text((plot[0] + 360, plot[3] + 5), "PC1 ({:.1f}% variance)".format(explained[0] * 100),
              fill="#526174", font=small_font)
    draw.text((18, plot[1] + 165), "PC2\n{:.1f}%".format(explained[1] * 100),
              fill="#526174", font=small_font)

    if not projected:
        draw.text((110, 180), "No samples to project", fill="#526174", font=font)
        return image

    xs, ys = [point[0] for point in projected], [point[1] for point in projected]
    xlow, xhigh = min(xs), max(xs)
    ylow, yhigh = min(ys), max(ys)
    if xhigh == xlow:
        xhigh = xlow + 1.0
    if yhigh == ylow:
        yhigh = ylow + 1.0
    xpad, ypad = (xhigh - xlow) * 0.06, (yhigh - ylow) * 0.06
    xlow, xhigh = xlow - xpad, xhigh + xpad
    ylow, yhigh = ylow - ypad, yhigh + ypad

    pixel_rows = []
    grouped = {}
    for record, (xvalue, yvalue) in zip(records, projected):
        px = plot[0] + (xvalue - xlow) / (xhigh - xlow) * (plot[2] - plot[0])
        py = plot[3] - (yvalue - ylow) / (yhigh - ylow) * (plot[3] - plot[1])
        label = base.target_class(record["labels"], components)
        grouped.setdefault(label, []).append((px, py))

    # Draw hull outlines first so sample dots remain visible on top.
    for label, points in grouped.items():
        hull = convex_hull(points)
        color = colors[label]
        if len(hull) >= 3:
            draw.polygon(hull, outline=color, width=3)
        elif len(hull) == 2:
            draw.line(hull, fill=color, width=3)
        for px, py in points:
            pixel_rows.append((label, px, py))
    for label, px, py in pixel_rows:
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=colors[label])

    legend_y = 505
    for index, (label, color) in enumerate(colors.items()):
        x = 24 + (index % 4) * 240
        y = legend_y + (index // 4) * 24
        draw.rectangle((x, y + 4, x + 18, y + 13), outline=color, width=3)
        display = label
        while draw.textbbox((0, 0), display, font=small_font)[2] > 205 and len(display) > 8:
            display = display[:-4] + "..."
        draw.text((x + 25, y), display, fill="#263E5B", font=small_font)
    return image


def create_mode_charts(mode, records, summary):
    components = dict(base.TARGET_MODES)[mode]
    return {
        "Class distribution": base.draw_class_chart(mode, summary),
        "Class relationships": draw_relationship_tree(mode, components, records),
        "Class feature profile": draw_class_feature_profile(mode, components, records),
        "Feature-class association": draw_feature_class_association(mode, components, records),
        "Pairwise feature overlap": draw_pairwise_feature_overlap(mode, components, records),
        "PCA class separation": draw_pca_class_separation(mode, components, records),
        "Cluster boundaries": draw_cluster_boundaries(mode, components, records),
    }


def mode_interpretation(title, mode, records):
    """Add a concise, data-aware note to a report slide without claiming accuracy."""
    components = dict(base.TARGET_MODES)[mode]
    labels = [base.target_class(record["labels"], components) for record in records]
    counts = Counter(labels)
    if title == "Class distribution":
        if not counts:
            return "No valid samples were available for this target mode."
        largest = max(counts.values())
        smallest = min(counts.values())
        ratio = float(largest) / smallest if smallest else float("inf")
        return "{} classes; largest has {} samples and smallest has {} ({}× ratio). Review class balance before choosing a split or sampling strategy.".format(
            len(counts), largest, smallest, "{:.1f}".format(ratio) if ratio != float("inf") else "∞"
        )
    if title == "Class relationships":
        small = sum(1 for count in counts.values() if count < 10)
        return "{} observed target classes. {} class(es) have fewer than 10 samples, which may make their learned decision patterns unstable.".format(
            len(counts), small
        )
    if title == "Feature-class association":
        scores = feature_class_associations(components, records)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return "Strongest association: {} (η²={:.3f}); weakest: {} (η²={:.3f}). Higher η² indicates more between-class variance for that feature, not guaranteed separability.".format(
            ranked[0][0], ranked[0][1], ranked[-1][0], ranked[-1][1]
        )
    if title == "PCA class separation":
        _points, explained = pca_projection(records)
        return "PC1 and PC2 capture {:.1f}% and {:.1f}% of variance ( {:.1f}% combined ). This is only a 2D projection; check pairwise plots and other features before judging overlap.".format(
            explained[0] * 100.0, explained[1] * 100.0,
            (explained[0] + explained[1]) * 100.0
        )
    if title == "Cluster boundaries":
        return "Each colored outline encloses the projected samples for one class. Non-overlapping hulls suggest separation in this two-dimensional view; overlaps may indicate boundary ambiguity. Inspect pairwise plots too, because PCA compresses four features into two."
    if title == "Pairwise feature overlap":
        return "Use all six feature pairs to identify which feature combinations show visibly distinct color clouds and which mix. A pair can overlap even when the full four-feature patterns are more separable."
    if title == "Class feature profile":
        return "Compare each class row across X1-X4. Different means can help distinguish classes, but similar averages or distinct averages alone do not reveal the within-class spread; use the overlap charts as well."
    return "Descriptive view of the selected target labels."


def save_powerpoint_report(path, modes, records, summary, charts_dir):
    """Save each selected target case as seven graph slides with guides and notes."""
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
        from pptx.util import Inches, Pt
    except ImportError:
        raise RuntimeError(
            "PowerPoint export requires python-pptx. Install it in the PyCharm interpreter with: pip install python-pptx"
        )

    presentation = Presentation()
    presentation.slide_width = Inches(13.333)
    presentation.slide_height = Inches(7.5)
    blank_layout = presentation.slide_layouts[6]
    title_layout = presentation.slide_layouts[0]
    cover = presentation.slides.add_slide(title_layout)
    cover.shapes.title.text = "Meta AI Dataset Class Separation Report"
    cover.placeholders[1].text = "Selected target cases: {}\nSamples analyzed: {}\nInput features: X1, X2, X3, X4\n\nCharts are descriptive diagnostics; they do not by themselves establish model accuracy.".format(
        ", ".join(mode.replace("_", " + ") for mode in modes), len(records)
    )

    chart_order = list(GRAPH_GUIDES)
    for mode in modes:
        charts = create_mode_charts(mode, records, summary)
        mode_dir = charts_dir
        os.makedirs(mode_dir, exist_ok=True)
        for title in chart_order:
            chart = charts[title]
            image_path = os.path.join(mode_dir, "{}_{}.png".format(
                mode.lower(), re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
            ))
            chart.save(image_path)
            slide = presentation.slides.add_slide(blank_layout)
            heading = slide.shapes.add_textbox(Inches(0.45), Inches(0.18), Inches(12.4), Inches(0.55))
            paragraph = heading.text_frame.paragraphs[0]
            paragraph.text = "{}  |  {}".format(mode.replace("_", " + "), title)
            paragraph.font.size = Pt(24)
            paragraph.font.bold = True
            paragraph.font.color.rgb = RGBColor(31, 62, 91)

            picture = Image.open(image_path)
            image_width, image_height = picture.size
            picture.close()
            max_width, max_height = 8.55, 6.15
            scale = min(max_width / image_width, max_height / image_height)
            render_width, render_height = image_width * scale, image_height * scale
            slide.shapes.add_picture(
                image_path, Inches(0.35), Inches(0.95),
                width=Inches(render_width), height=Inches(render_height)
            )
            panel = slide.shapes.add_textbox(Inches(9.1), Inches(1.0), Inches(3.8), Inches(5.8))
            text_frame = panel.text_frame
            text_frame.word_wrap = True
            text_frame.clear()
            for label, text in (("Legend", GRAPH_GUIDES[title][0]),
                                ("How to read it", GRAPH_GUIDES[title][1]),
                                ("This case", mode_interpretation(title, mode, records))):
                para = text_frame.paragraphs[0] if not text_frame.paragraphs[0].text else text_frame.add_paragraph()
                para.text = label
                para.font.bold = True
                para.font.size = Pt(16)
                para.font.color.rgb = RGBColor(49, 92, 125)
                body = text_frame.add_paragraph()
                body.text = text
                body.font.size = Pt(12)
                body.space_after = Pt(12)
    presentation.save(path)


def write_combined_workbook(path, records, skipped, source_details):
    workbook = Workbook()
    overview = workbook.active
    overview.title = "Dataset Summary"
    overview.append(["Meta AI training dataset", "Details"])
    overview.append(["Source workbooks", len(source_details)])
    overview.append(["Union method", "Rows concatenated from selected workbook(s)"])
    overview.append(["Valid samples", len(records)])
    overview.append(["Skipped source rows", len(skipped)])
    overview.append(["Input features", ", ".join(base.FEATURES)])
    overview.append(["Target mapping", "Y1, Y2, Y3 follow the source workbook headers listed on Source Files"])
    overview.append([])
    overview.append(["Dataset mode", "Target components", "Samples", "Classes"])

    distribution = workbook.create_sheet("Class Distribution")
    distribution.append(["Dataset mode", "Target Class", "Samples", "Share (%)"])
    for mode, components in base.TARGET_MODES:
        counts = Counter(
            base.target_class(record["labels"], components) for record in records
        )
        overview.append([mode, " + ".join(components), len(records), len(counts)])
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold())):
            distribution.append([mode, label, count, count / float(len(records)) * 100.0 if records else 0])

        sheet = workbook.create_sheet(mode)
        component_fields = ["Y{}".format(index) for index in range(1, len(components) + 1)]
        headers = ["Sample ID", "Source file", "Source row", "Image path", "Test category"]
        headers += list(base.FEATURES) + component_fields + ["Target Class"]
        sheet.append(headers)
        for record in records:
            sample_id = "{}::{}".format(record["source_file"], record["source_row"])
            parts = [record["labels"][component] for component in components]
            row = [sample_id, record["source_file"], record["source_row"],
                   record["image_path"], record["test_category"]]
            row += [record["features"][feature] for feature in base.FEATURES]
            row += parts + [base.target_class(record["labels"], components)]
            sheet.append(row)
        base.style_sheet(sheet)

    sources = workbook.create_sheet("Source Files")
    sources.append(["Source file", "Full path", "Source sheet", "Header row", "Valid samples",
                    "Skipped rows", "Y1 source header", "Y2 source header", "Y3 source header"])
    for detail in source_details:
        sources.append([
            detail["Source file"], detail["Full path"], detail["Source sheet"],
            detail["Header row"], detail["Valid samples"], detail["Skipped rows"],
        ] + detail["Y columns"])

    skipped_sheet = workbook.create_sheet("Skipped Rows")
    skipped_sheet.append(["Source file", "Source row", "Reason"])
    for item in skipped:
        skipped_sheet.append([item["Source file"], item["Source row"], item["Reason"]])

    for sheet in (overview, distribution, sources, skipped_sheet):
        base.style_sheet(sheet)
    for cell in overview[9]:
        cell.font = Font(name="Arial", bold=True, color="FFFFFF")
        cell.fill = PatternFill(fill_type="solid", fgColor="315C7D")
    overview.freeze_panes = "A10"
    overview.auto_filter.ref = "A9:D{}".format(overview.max_row)
    sources.column_dimensions["B"].width = 55
    distribution.column_dimensions["B"].width = 46
    skipped_sheet.column_dimensions["C"].width = 42
    workbook.save(path)


class DatasetTreeDashboard:
    def __init__(self, root, output_path, records):
        self.root = root
        self.output_path = output_path
        self.records = records
        self.summary = base.summarize_modes(records)
        self.images = {}
        root.title("Meta AI Dataset: Class and Tree Dashboard")
        root.geometry("1320x900")

        body = ttk.Frame(root, padding=10)
        body.pack(fill="both", expand=True)
        controls = ttk.LabelFrame(body, text="Dataset targets", padding=8)
        controls.pack(side="left", fill="y", padx=(0, 10))
        ttk.Label(controls, text="Select one or more target modes:", wraplength=210).pack(anchor="w", pady=(0, 6))
        self.listbox = tk.Listbox(controls, selectmode="multiple", height=12, width=24, exportselection=False)
        for mode, _components in base.TARGET_MODES:
            self.listbox.insert("end", mode.replace("_", " + "))
        self.listbox.selection_set(0, tk.END)
        self.listbox.pack(fill="y", expand=True)
        ttk.Button(controls, text="Select All", command=self.select_all).pack(fill="x", pady=(8, 3))
        ttk.Button(controls, text="Clear", command=self.clear).pack(fill="x", pady=3)
        ttk.Button(controls, text="Show Selected", command=self.refresh).pack(fill="x", pady=(3, 0))
        ttk.Button(controls, text="Export PowerPoint", command=self.export_powerpoint).pack(fill="x", pady=(10, 0))

        self.view_area = ttk.Frame(body)
        self.view_area.pack(side="left", fill="both", expand=True)
        footer = ttk.Frame(root, padding=(10, 2, 10, 8))
        footer.pack(fill="x")
        self.status = ttk.Label(footer, text="Dataset workbook: {}".format(os.path.basename(output_path)))
        self.status.pack(anchor="w")
        root.after_idle(self.safe_refresh)

    def safe_refresh(self):
        try:
            self.refresh()
        except Exception as exc:
            self.status.configure(text="Dashboard error: {}".format(exc))
            messagebox.showerror("Could not display charts", str(exc), parent=self.root)

    def selected_modes(self):
        return [base.TARGET_MODES[index][0] for index in self.listbox.curselection()]

    def select_all(self):
        self.listbox.selection_set(0, tk.END)
        self.refresh()

    def clear(self):
        self.listbox.selection_clear(0, tk.END)
        self.refresh()

    def add_image_tab(self, notebook, title, image, key):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)
        chart_area = ttk.Frame(tab)
        chart_area.pack(fill="both", expand=True)
        canvas = tk.Canvas(chart_area, background="#EEF2F6", highlightthickness=0)
        ybar = ttk.Scrollbar(chart_area, orient="vertical", command=canvas.yview)
        xbar = ttk.Scrollbar(chart_area, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        chart_area.rowconfigure(0, weight=1)
        chart_area.columnconfigure(0, weight=1)
        photo = ImageTk.PhotoImage(image, master=canvas)
        self.images[key] = photo
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.configure(scrollregion=(0, 0, image.width, image.height))
        canvas.bind("<Map>", lambda _event, widget=canvas: self.root.after_idle(widget.update_idletasks))
        guide, interpretation = GRAPH_GUIDES[title]
        ttk.Label(tab, text=guide + "\n" + interpretation, wraplength=1000,
                  justify="left", padding=(8, 6)).pack(fill="x", side="bottom")

    def export_powerpoint(self):
        modes = self.selected_modes()
        if not modes:
            messagebox.showinfo("Select target cases", "Select one or more target cases before exporting.", parent=self.root)
            return
        default_name = os.path.splitext(os.path.basename(self.output_path))[0] + "_class_separation_report.pptx"
        path = filedialog.asksaveasfilename(
            title="Save PowerPoint class analysis report",
            initialdir=os.path.dirname(self.output_path),
            initialfile=default_name,
            defaultextension=".pptx",
            filetypes=[("PowerPoint presentation", "*.pptx")],
            parent=self.root,
        )
        if not path:
            return
        charts_dir = os.path.splitext(self.output_path)[0] + "_charts"
        try:
            save_powerpoint_report(path, modes, self.records, self.summary, charts_dir)
        except Exception as exc:
            messagebox.showerror("Could not export PowerPoint", str(exc), parent=self.root)
            return
        self.status.configure(text="PowerPoint report saved: {}".format(os.path.basename(path)))
        messagebox.showinfo("PowerPoint report saved", "Saved report:\n{}".format(path), parent=self.root)

    def refresh(self):
        for child in self.view_area.winfo_children():
            child.destroy()
        self.images = {}
        modes = self.selected_modes()
        if not modes:
            ttk.Label(self.view_area, text="Select at least one target mode.", padding=20).pack(anchor="nw")
            self.status.configure(text="No graphs selected. Dataset saved: {}".format(os.path.basename(self.output_path)))
            return

        charts_dir = os.path.splitext(self.output_path)[0] + "_charts"
        os.makedirs(charts_dir, exist_ok=True)
        outer = ttk.Notebook(self.view_area)
        outer.pack(fill="both", expand=True)

        for mode in modes:
            mode_tab = ttk.Frame(outer)
            outer.add(mode_tab, text=mode.replace("_", " + "))
            inner = ttk.Notebook(mode_tab)
            inner.pack(fill="both", expand=True)

            charts = create_mode_charts(mode, self.records, self.summary)
            file_suffixes = {
                "Class distribution": "bar",
                "Class relationships": "tree",
                "Class feature profile": "feature_profile",
                "Feature-class association": "feature_class_association",
                "Pairwise feature overlap": "pairwise_overlap",
                "PCA class separation": "pca_separation",
                "Cluster boundaries": "cluster_boundaries",
            }
            for title, chart in charts.items():
                chart.save(os.path.join(charts_dir, mode.lower() + "_" + file_suffixes[title] + ".png"))
                self.add_image_tab(inner, title, chart, mode + "_" + file_suffixes[title])

        self.status.configure(text="Showing {} target mode(s); each has seven analysis views | Dataset: {}".format(
            len(modes), os.path.basename(self.output_path)
        ))


def select_input_files(root):
    first = filedialog.askopenfilename(
        title="Select the first evaluation Excel workbook",
        filetypes=[("Excel workbooks", "*.xlsx"), ("All files", "*.*")],
        parent=root,
    )
    if not first:
        return []
    second = filedialog.askopenfilename(
        title="Optional: select a second workbook (Cancel to analyze only the first)",
        filetypes=[("Excel workbooks", "*.xlsx"), ("All files", "*.*")],
        parent=root,
    )
    paths = [first]
    if second and os.path.normcase(os.path.abspath(second)) != os.path.normcase(os.path.abspath(first)):
        paths.append(second)
    return paths


def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = select_input_files(root)
    if not paths:
        root.destroy()
        return

    try:
        records, skipped, source_details = read_and_union_sources(paths)
    except Exception as exc:
        messagebox.showerror("Could not combine source workbooks", str(exc), parent=root)
        root.destroy()
        return

    date_tag = datetime.date.today().strftime("%Y%m%d")
    output_dir = os.path.dirname(paths[0])
    default_name = os.path.splitext(os.path.basename(paths[0]))[0] + "_meta_ai_training_dataset_{}.xlsx".format(date_tag)
    default_output = unique_excel_path(os.path.join(output_dir, default_name))
    output_path = filedialog.asksaveasfilename(
        title="Save combined Meta AI training dataset",
        initialdir=output_dir,
        initialfile=os.path.basename(default_output),
        defaultextension=".xlsx",
        filetypes=[("Excel workbooks", "*.xlsx")],
        confirmoverwrite=False,
        parent=root,
    )
    if not output_path:
        root.destroy()
        return
    output_path = unique_excel_path(output_path)

    try:
        write_combined_workbook(output_path, records, skipped, source_details)
    except Exception as exc:
        messagebox.showerror("Could not save training datasets", str(exc), parent=root)
        root.destroy()
        return

    messagebox.showinfo("Workbook saved", "Saved Excel workbook:\n{}".format(output_path), parent=root)

    root.attributes("-topmost", False)
    root.deiconify()
    try:
        DatasetTreeDashboard(root, output_path, records)
    except Exception as exc:
        messagebox.showerror("Could not open dataset dashboard", str(exc), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
