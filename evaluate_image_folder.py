"""Evaluate every image in a selected folder with the four trained classifiers."""

from __future__ import print_function

import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

import compare_trained_image_models as comparison


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
BATCH_SIZE = 8
SPOT_NAMES = (
    "Bright Spot Prime",
    "Bright Spot High",
    "Bright Spot",
    "Bright Spot Low 1",
    "Bright Spot Low 2",
    "Gray Spot",
    "Blind Spot",
)


def choose_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select a folder containing images")
    root.destroy()
    if not folder:
        raise SystemExit("Folder selection was cancelled.")
    return os.path.abspath(folder)


def collect_images(folder):
    paths = []
    for current_dir, subdirs, filenames in os.walk(folder):
        subdirs[:] = sorted(
            name for name in subdirs
            if name not in ("_evaluation_graphs", "_evaluation_results")
        )
        for filename in sorted(filenames):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(current_dir, filename))
    return paths


def save_summary_graph(image_path, image, class_names, summary, output_dir, sample_number):
    """Save a compact PNG chart without importing Matplotlib/OpenMP backends."""
    os.makedirs(output_dir, exist_ok=True)
    width, height = 1500, 900
    canvas = Image.new("RGB", (width, height), "#F3F5F8")
    draw = ImageDraw.Draw(canvas)
    try:
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        regular = ImageFont.truetype(font_path, 22)
        small = ImageFont.truetype(font_path, 18)
        title = ImageFont.truetype(font_path, 34)
        big = ImageFont.truetype(font_path, 42)
    except (IOError, OSError):
        regular = ImageFont.load_default()
        small = ImageFont.load_default()
        title = ImageFont.load_default()
        big = ImageFont.load_default()

    spot = summary["spot"]
    bright = spot.startswith("Bright Spot")
    gray = spot == "Gray Spot"
    spot_color = "#16845B" if bright else ("#64748B" if gray else "#B45309")
    spot_bg = "#E7F5EF" if bright else ("#EEF1F5" if gray else "#FFF4E5")
    draw.rectangle((0, 0, width, 112), fill=spot_bg)
    draw.text((40, 25), spot.upper(), fill=spot_color, font=title)
    draw.text((40, 72), "Sample {}  |  {}".format(
        sample_number, os.path.relpath(image_path, os.path.dirname(output_dir))
    ), fill="#182230", font=small)

    preview = image.copy()
    preview.thumbnail((360, 300))
    canvas.paste(preview, (40, 155))
    draw.text((40, 475), "Consensus: {}".format(
        " / ".join(summary["leading_labels"])
    ), fill="#182230", font=big)
    draw.text((40, 535), "Agreement: {}/4".format(summary["max_votes"]),
              fill="#315C9B", font=regular)
    draw.text((40, 580), "Mean confidence: {:.1f}%".format(
        summary["mean_confidence"] * 100.0
    ), fill="#182230", font=regular)
    draw.text((40, 620), "Minimum: {:.1f}%   SD: {:.1f}%".format(
        summary["minimum_confidence"] * 100.0,
        summary["confidence_std_dev"] * 100.0,
    ), fill="#64748B", font=regular)

    left, bar_width = 570, 600
    draw.text((left, 145), "PREDICTION VOTES", fill="#182230", font=title)
    for index, label in enumerate(class_names):
        count = sum(1 for item in summary["results"] if item["prediction"] == label)
        y = 205 + index * 54
        draw.text((left, y), label, fill="#182230", font=regular)
        draw.rectangle((left + 190, y + 3, left + 190 + bar_width, y + 34), fill="#E3E8EF")
        if count:
            fill = "#16845B" if label in summary["leading_labels"] else "#4C78A8"
            draw.rectangle((left + 190, y + 3,
                            left + 190 + bar_width * count / 4.0, y + 34), fill=fill)
        draw.text((left + 810, y), "{}/4".format(count), fill="#182230", font=small)

    draw.text((left, 520), "MODEL CONFIDENCE", fill="#182230", font=title)
    for index, result in enumerate(summary["results"]):
        y = 580 + index * 52
        draw.text((left, y), result["name"], fill="#64748B", font=small)
        draw.rectangle((left + 250, y + 2, left + 250 + bar_width, y + 30), fill="#E3E8EF")
        fill = "#16845B" if result["prediction"] in summary["leading_labels"] else "#D97706"
        draw.rectangle((left + 250, y + 2,
                        left + 250 + bar_width * result["confidence"], y + 30), fill=fill)
        draw.text((left + 860, y), "{:.1f}%".format(result["confidence"] * 100.0),
                  fill="#182230", font=small)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    safe_image_name = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in image_name
    )
    output_path = os.path.join(
        output_dir,
        "sample_{:04d}_{}_summary.png".format(sample_number, safe_image_name),
    )
    canvas.save(output_path)
    return output_path


def create_prediction_records(folder, image_paths, checkpoint_dir, device):
    checkpoints = comparison.find_checkpoints(checkpoint_dir)
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(comparison.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(comparison.IMAGENET_MEAN, comparison.IMAGENET_STD),
    ])

    records = []
    for path in image_paths:
        record = {
            "path": path,
            "relative_path": os.path.relpath(path, folder),
            "folder_label": os.path.basename(os.path.dirname(path)),
            "models": {},
            "error": "",
        }
        records.append(record)

    class_names = None
    model_details = {}
    for model_index, (display_name, model_name, _) in enumerate(comparison.MODEL_SPECS, start=1):
        checkpoint_path = checkpoints[model_name]
        print("\n[{}/{}] Loading {} checkpoint: {}".format(
            model_index, len(comparison.MODEL_SPECS), display_name,
            os.path.basename(checkpoint_path),
        ), flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_classes = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
        if not saved_classes:
            raise ValueError("Checkpoint has no class_names: {}".format(checkpoint_path))
        if class_names is None:
            class_names = list(saved_classes)
        elif list(saved_classes) != class_names:
            raise ValueError(
                "Model label mappings differ. {} has {} but expected {}.".format(
                    os.path.basename(checkpoint_path), saved_classes, class_names
                )
            )

        model, checkpoint = comparison.load_model(
            model_name, checkpoint, device, class_names, checkpoint_path
        )
        print("Loaded {} on {}. Evaluating {} images in batches of {}...".format(
            display_name, device, len(records), BATCH_SIZE
        ), flush=True)
        accuracy, epoch = comparison.checkpoint_accuracy_epoch(checkpoint_path)
        if "_train_acc_" in os.path.basename(checkpoint_path):
            # The filename describes training accuracy; report this checkpoint's
            # validation accuracy from its metadata instead.
            accuracy = float(checkpoint.get("validation_accuracy", -1.0)) * 100.0
        if accuracy < 0:
            accuracy = float(checkpoint.get("validation_accuracy", 0.0)) * 100.0
            epoch = int(checkpoint.get("epoch", 0))
        model_details[model_name] = {
            "display_name": display_name,
            "checkpoint": os.path.basename(checkpoint_path),
            "validation_accuracy": accuracy,
            "epoch": epoch,
        }

        pending = [record for record in records if not record["error"]]
        completed = 0
        with torch.no_grad():
            for start in range(0, len(pending), BATCH_SIZE):
                batch_records = []
                batch_images = []
                for record in pending[start:start + BATCH_SIZE]:
                    try:
                        with Image.open(record["path"]) as source_image:
                            image = source_image.convert("RGB")
                        batch_images.append(preprocessing(image))
                        batch_records.append(record)
                    except Exception as error:
                        record["error"] = "{}: {}".format(type(error).__name__, error)

                if batch_images:
                    batch_tensor = torch.stack(batch_images).to(device)
                    batch_probabilities = torch.softmax(model(batch_tensor), dim=1).cpu().tolist()
                    for record, probabilities in zip(batch_records, batch_probabilities):
                        predicted_index = max(
                            range(len(probabilities)), key=probabilities.__getitem__
                        )
                        record["models"][model_name] = {
                            "prediction": class_names[predicted_index],
                            "confidence": float(probabilities[predicted_index]),
                            "probabilities": [float(value) for value in probabilities],
                        }
                    batch_output = " | ".join(
                        "{}: {} {:.1f}%".format(
                            os.path.basename(record["path"]),
                            record["models"][model_name]["prediction"],
                            record["models"][model_name]["confidence"] * 100.0,
                        )
                        for record in batch_records
                    )
                    print("  [{} batch {}-{}] {}".format(
                        display_name, start + 1,
                        min(start + len(batch_records), len(pending)), batch_output,
                    ), flush=True)
                completed += len(batch_records)
                if completed and (completed % (BATCH_SIZE * 10) == 0 or start + BATCH_SIZE >= len(pending)):
                    print("  {}: {}/{} images processed".format(
                        display_name, min(start + BATCH_SIZE, len(pending)), len(pending)
                    ), flush=True)
        del model
        del checkpoint
        print("Finished {}.".format(display_name), flush=True)

    if class_names is None:
        raise RuntimeError("Could not read model labels from checkpoints.")
    for record in records:
        relative_directory = os.path.relpath(os.path.dirname(record["path"]), folder)
        path_labels = [os.path.basename(folder)]
        if relative_directory != ".":
            path_labels.extend(relative_directory.split(os.sep))
        record["folder_label"] = next(
            (label for label in path_labels if label in class_names), ""
        )
        if record["folder_label"]:
            record["source_category"] = record["folder_label"]
        elif relative_directory == ".":
            record["source_category"] = "Uncategorized"
        else:
            directory_parts = relative_directory.split(os.sep)
            generic_parts = {
                "data", "dataset", "image", "images", "test", "tests",
                "train", "training", "val", "valid", "validation",
            }
            category_parts = [
                part for part in directory_parts
                if part.lower() not in generic_parts
            ]
            record["source_category"] = (
                category_parts[0] if category_parts else directory_parts[-1]
            )
    return records, class_names, model_details


def summarize_record(record, class_names):
    results = []
    for display_name, model_name, _ in comparison.MODEL_SPECS:
        model_result = record["models"][model_name]
        results.append({
            "name": display_name,
            "model": model_name,
            "prediction": model_result["prediction"],
            "confidence": model_result["confidence"],
            "probabilities": model_result["probabilities"],
        })

    counts = {}
    for result in results:
        counts[result["prediction"]] = counts.get(result["prediction"], 0) + 1
    max_votes = max(counts.values())
    leading_labels = [label for label in class_names if counts.get(label, 0) == max_votes]
    leading_results = [item for item in results if item["prediction"] in leading_labels]
    confidences = [item["confidence"] for item in leading_results]
    mean_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    sd_conf = (sum((value - mean_conf) ** 2 for value in confidences) / len(confidences)) ** 0.5
    majority = max_votes > len(results) / 2.0

    if majority and min_conf >= 0.80 and max_votes == len(results) and mean_conf >= 0.99:
        spot = "Bright Spot Prime"
    elif majority and min_conf >= 0.80 and max_votes == len(results) and mean_conf >= 0.95:
        spot = "Bright Spot High"
    elif majority and mean_conf >= 0.95 and min_conf >= 0.80:
        spot = "Bright Spot"
    elif majority and mean_conf >= 0.90 and min_conf >= 0.80 and sd_conf < 0.075:
        spot = "Bright Spot Low 1"
    elif (max_votes == len(results) and mean_conf >= 0.80
          and min_conf >= 0.60 and sd_conf < 0.075):
        spot = "Bright Spot Low 2"
    elif majority and mean_conf >= 0.90 and 0.70 <= min_conf < 0.80:
        spot = "Gray Spot"
    else:
        spot = "Blind Spot"

    return {
        "results": results,
        "leading_labels": leading_labels,
        "max_votes": max_votes,
        "agreement_rate": max_votes / float(len(results)),
        "mean_confidence": mean_conf,
        "minimum_confidence": min_conf,
        "confidence_std_dev": sd_conf,
        "spot": spot,
    }


def save_excel(folder, records, class_names, model_details):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise RuntimeError(
            "openpyxl is required to write the Excel report. Install it with: "
            "python -m pip install openpyxl"
        )

    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "Summary"
    summary_sheet.append(["Folder image evaluation", os.path.basename(folder)])
    summary_sheet.append(["Images found", len(records)])
    summary_sheet.append(["Successful evaluations", sum(1 for item in records if not item["error"])])
    summary_sheet.append(["Errors", sum(1 for item in records if item["error"])])
    labeled_records = [
        item for item in records
        if not item["error"] and item["folder_label"] in class_names
    ]
    summary_sheet.append(["Images with folder labels", len(labeled_records)])
    summary_sheet.append([])
    summary_sheet.append(["Spot classification", "Count"])
    spot_counts = {}
    for record in records:
        spot = record.get("summary", {}).get("spot", "Error")
        spot_counts[spot] = spot_counts.get(spot, 0) + 1
    for spot in ("Bright Spot Prime", "Bright Spot High", "Bright Spot",
                 "Bright Spot Low 1", "Bright Spot Low 2", "Gray Spot",
                 "Blind Spot", "Error"):
        summary_sheet.append([spot, spot_counts.get(spot, 0)])
    summary_sheet.append([])
    model_header_row = summary_sheet.max_row + 1
    summary_sheet.append(["Model", "Correct", "Labeled images", "Folder-label accuracy (%)"])
    for _, model_name, _ in comparison.MODEL_SPECS:
        details = model_details[model_name]
        correct = sum(
            1 for item in labeled_records
            if item["models"][model_name]["prediction"] == item["folder_label"]
        )
        count = len(labeled_records)
        accuracy = correct / float(count) * 100.0 if count else None
        summary_sheet.append([details["display_name"], correct, count, accuracy])

    bright_spots = (
        "Bright Spot Prime", "Bright Spot High", "Bright Spot",
        "Bright Spot Low 1", "Bright Spot Low 2",
    )
    bright_total = sum(spot_counts.get(name, 0) for name in bright_spots)
    gray_total = spot_counts.get("Gray Spot", 0)
    blind_total = spot_counts.get("Blind Spot", 0)
    categorized_total = bright_total + gray_total + blind_total
    consistency_header_row = summary_sheet.max_row + 2
    summary_sheet.append([])
    summary_sheet.append(["Spot totals consistency check", "Count"])
    summary_sheet.append(["Bright Spot total (all Bright variants)", bright_total])
    summary_sheet.append(["Gray Spot total", gray_total])
    summary_sheet.append(["Blind Spot total", blind_total])
    summary_sheet.append(["Bright + Gray + Blind total", categorized_total])
    summary_sheet.append(["Test samples", len(records)])
    summary_sheet.append(["Difference (Spot total - test samples)", categorized_total - len(records)])
    summary_sheet.append(["Counts match test samples", "MATCH" if categorized_total == len(records) else "MISMATCH"])
    summary_sheet.append(["Evaluation errors included in test samples", spot_counts.get("Error", 0)])

    predictions = workbook.create_sheet("Predictions")
    headers = [
        "File name", "Relative path", "Folder label", "Status", "Spot",
        "Consensus label", "Leading votes", "Agreement (%)",
        "Leading mean confidence (%)", "Leading min confidence (%)",
        "Leading confidence SD (%)",
    ]
    for _, model_name, _ in comparison.MODEL_SPECS:
        details = model_details[model_name]
        headers.extend([
            "{} prediction".format(details["display_name"]),
            "{} top confidence (%)".format(details["display_name"]),
            "{} correct vs folder label".format(details["display_name"]),
            "{} checkpoint".format(details["display_name"]),
            "{} checkpoint val acc (%)".format(details["display_name"]),
            "{} checkpoint epoch".format(details["display_name"]),
        ])
        headers.extend("{} P({}) (%)".format(details["display_name"], label) for label in class_names)
    headers.append("Error")
    predictions.append(headers)

    bright_fill = PatternFill("solid", fgColor="E2F0D9")
    gray_fill = PatternFill("solid", fgColor="E7E6E6")
    blind_fill = PatternFill("solid", fgColor="FCE4D6")
    error_fill = PatternFill("solid", fgColor="F4CCCC")
    spot_fill = {
        "Bright Spot Prime": bright_fill,
        "Bright Spot High": bright_fill,
        "Bright Spot": bright_fill,
        "Bright Spot Low 1": bright_fill,
        "Bright Spot Low 2": bright_fill,
        "Gray Spot": gray_fill,
        "Blind Spot": blind_fill,
        "Error": error_fill,
    }

    for record in records:
        summary = record.get("summary")
        if record["error"] or summary is None:
            row = [os.path.basename(record["path"]), record["relative_path"],
                   record["folder_label"], "ERROR", "Error", "", "", "", "", "", ""]
            for _, model_name, _ in comparison.MODEL_SPECS:
                details = model_details[model_name]
                row.extend(["", "", "", details["checkpoint"],
                            details["validation_accuracy"], details["epoch"]])
                row.extend([""] * len(class_names))
            row.append(record["error"])
            predictions.append(row)
            row_number = predictions.max_row
            predictions.cell(row_number, 4).fill = error_fill
            continue

        leading_label = summary["leading_labels"][0] if len(summary["leading_labels"]) == 1 else " / ".join(summary["leading_labels"])
        true_label = record["folder_label"]
        row = [
            os.path.basename(record["path"]), record["relative_path"], true_label,
            "OK", summary["spot"], leading_label, summary["max_votes"],
            summary["agreement_rate"] * 100.0,
            summary["mean_confidence"] * 100.0,
            summary["minimum_confidence"] * 100.0,
            summary["confidence_std_dev"] * 100.0,
        ]
        for _, model_name, _ in comparison.MODEL_SPECS:
            output = record["models"][model_name]
            details = model_details[model_name]
            row.extend([
                output["prediction"], output["confidence"] * 100.0,
                ("Yes" if output["prediction"] == true_label else "No") if true_label else "N/A",
                details["checkpoint"], details["validation_accuracy"], details["epoch"],
            ])
            row.extend(value * 100.0 for value in output["probabilities"])
        row.append("")
        predictions.append(row)
        row_number = predictions.max_row
        predictions.cell(row_number, 5).fill = spot_fill.get(summary["spot"], blind_fill)

    # Category-level performance for images whose folder identifies a class.
    category_sheet = workbook.create_sheet("Category Analysis")
    category_headers = [
        "Category", "Images", "Consensus correct", "Consensus accuracy (%)",
        "Avg consensus confidence (%)",
    ]
    for _, model_name, _ in comparison.MODEL_SPECS:
        display_name = model_details[model_name]["display_name"]
        category_headers.extend([
            "{} correct".format(display_name),
            "{} accuracy (%)".format(display_name),
            "{} avg confidence (%)".format(display_name),
        ])
    category_headers.extend([
        "Bright Spot Prime", "Bright Spot High", "Bright Spot",
        "Bright Spot Low 1", "Bright Spot Low 2", "Gray Spot",
        "Blind Spot",
    ])
    category_sheet.append(category_headers)
    labeled_successful = [
        item for item in records
        if not item["error"] and item["folder_label"] in class_names
        and item.get("summary") is not None
    ]
    category_spots = [
        "Bright Spot Prime", "Bright Spot High", "Bright Spot",
        "Bright Spot Low 1", "Bright Spot Low 2", "Gray Spot", "Blind Spot",
    ]
    for category in class_names:
        items = [item for item in labeled_successful if item["folder_label"] == category]
        consensus_correct = sum(
            1 for item in items
            if item["summary"]["leading_labels"] == [category]
        )
        row = [
            category, len(items), consensus_correct,
            consensus_correct / float(len(items)) * 100.0 if items else None,
            sum(item["summary"]["mean_confidence"] for item in items)
            / float(len(items)) * 100.0 if items else None,
        ]
        for _, model_name, _ in comparison.MODEL_SPECS:
            correct = sum(
                1 for item in items
                if item["models"][model_name]["prediction"] == category
            )
            row.extend([
                correct,
                correct / float(len(items)) * 100.0 if items else None,
                sum(item["models"][model_name]["confidence"] for item in items)
                / float(len(items)) * 100.0 if items else None,
            ])
        row.extend(sum(1 for item in items if item["summary"]["spot"] == spot)
                   for spot in category_spots)
        category_sheet.append(row)

    # Images from categories absent from the trained label mapping are still
    # summarized by source category, confidence, predicted label and Spot.
    untrained_sheet = workbook.create_sheet("Untrained Category Analysis")
    untrained_headers = [
        "Category", "Images", "Successful evaluations", "Errors",
        "Avg agreement (%)", "Avg consensus confidence (%)",
    ] + category_spots
    for _, model_name, _ in comparison.MODEL_SPECS:
        display_name = model_details[model_name]["display_name"]
        untrained_headers.append("{} avg top confidence (%)".format(display_name))
        untrained_headers.extend(
            "{} predicted {} count".format(display_name, label)
            for label in class_names
        )
    untrained_sheet.append(untrained_headers)
    untrained_categories = sorted({
        item.get("source_category", "Uncategorized")
        for item in records
        if item.get("source_category", "Uncategorized") not in class_names
    }, key=str.lower)
    for category in untrained_categories:
        category_items = [
            item for item in records
            if item.get("source_category", "Uncategorized") == category
        ]
        successful_items = [
            item for item in category_items
            if not item["error"] and item.get("summary") is not None
        ]
        row = [
            category, len(category_items), len(successful_items),
            len(category_items) - len(successful_items),
            sum(item["summary"]["agreement_rate"] for item in successful_items)
            / float(len(successful_items)) * 100.0 if successful_items else None,
            sum(item["summary"]["mean_confidence"] for item in successful_items)
            / float(len(successful_items)) * 100.0 if successful_items else None,
        ]
        row.extend(
            sum(1 for item in successful_items
                if item["summary"]["spot"] == spot)
            for spot in category_spots
        )
        for _, model_name, _ in comparison.MODEL_SPECS:
            row.append(
                sum(item["models"][model_name]["confidence"] for item in successful_items)
                / float(len(successful_items)) * 100.0 if successful_items else None
            )
            row.extend(
                sum(1 for item in successful_items
                    if item["models"][model_name]["prediction"] == label)
                for label in class_names
            )
        untrained_sheet.append(row)

    # Confusion matrices show where each source category was predicted to land.
    confusion_sheet = workbook.create_sheet("Category Confusion")
    confusion_headers = ["Model", "Actual category"] + [
        "Predicted: {}".format(label) for label in class_names
    ]
    confusion_sheet.append(confusion_headers)
    confusion_counts = comparison.category_confusion_counts(
        labeled_successful, class_names
    )
    for _, model_name, _ in comparison.MODEL_SPECS:
        for actual in class_names:
            confusion_sheet.append([
                model_details[model_name]["display_name"], actual,
            ] + [confusion_counts[model_name][actual][predicted]
                 for predicted in class_names])

    # A compact matrix groups results by Spot first, then by source folder.
    folder_groups = {}
    for record in records:
        summary = record.get("summary")
        spot = summary["spot"] if summary else "Error"
        folder_label = record["folder_label"] or "Unlabeled"
        group = folder_groups.setdefault((spot, folder_label), {
            "count": 0, "mean_confidences": [], "minimum_confidences": [],
        })
        group["count"] += 1
        if summary:
            group["mean_confidences"].append(summary["mean_confidence"] * 100.0)
            group["minimum_confidences"].append(summary["minimum_confidence"] * 100.0)

    grouped_sheet = workbook.create_sheet("Spot-Folder Summary")
    grouped_headers = [
        "Spot", "Source folder label", "Images", "Avg leading confidence (%)",
        "Avg minimum confidence (%)",
    ]
    grouped_sheet.append(grouped_headers)
    spot_order = ["Bright Spot Prime", "Bright Spot High", "Bright Spot",
                  "Bright Spot Low 1", "Bright Spot Low 2", "Gray Spot",
                  "Blind Spot", "Error"]
    spot_rank = {name: index for index, name in enumerate(spot_order)}
    for (spot, folder_label), group in sorted(
        folder_groups.items(),
        key=lambda item: (spot_rank.get(item[0][0], len(spot_rank)), item[0][1].lower()),
    ):
        means = group["mean_confidences"]
        minimums = group["minimum_confidences"]
        grouped_sheet.append([
            spot, folder_label, group["count"],
            sum(means) / len(means) if means else None,
            sum(minimums) / len(minimums) if minimums else None,
        ])

    # Each Spot also gets a detail sheet, sorted by original folder then file.
    spot_sheet_names = [
        ("Bright Spot Prime", "Bright Prime"),
        ("Bright Spot High", "Bright High"),
        ("Bright Spot", "Bright Spot"),
        ("Bright Spot Low 1", "Bright Low 1"),
        ("Bright Spot Low 2", "Bright Low 2"),
        ("Gray Spot", "Gray Spot"),
        ("Blind Spot", "Blind Spot"),
        ("Error", "Errors"),
    ]
    grouped_detail_headers = [
        "Source folder label", "Relative path", "Consensus label", "Votes",
        "Agreement (%)", "Leading mean confidence (%)",
        "Leading minimum confidence (%)", "Leading confidence SD (%)",
    ]
    for _, model_name, _ in comparison.MODEL_SPECS:
        name = model_details[model_name]["display_name"]
        grouped_detail_headers.extend(["{} prediction".format(name), "{} confidence (%)".format(name)])
    grouped_detail_headers.append("Error")
    for spot, sheet_name in spot_sheet_names:
        sheet = workbook.create_sheet(sheet_name)
        sheet.append(grouped_detail_headers)
        matching = [
            record for record in records
            if record.get("summary", {}).get("spot", "Error") == spot
        ]
        matching.sort(key=lambda record: (
            (record["folder_label"] or "Unlabeled").lower(),
            record["relative_path"].lower(),
        ))
        for record in matching:
            summary = record.get("summary")
            if summary:
                consensus_label = (
                    summary["leading_labels"][0]
                    if len(summary["leading_labels"]) == 1
                    else " / ".join(summary["leading_labels"])
                )
                row = [
                    record["folder_label"] or "Unlabeled", record["relative_path"],
                    consensus_label, summary["max_votes"],
                    summary["agreement_rate"] * 100.0,
                    summary["mean_confidence"] * 100.0,
                    summary["minimum_confidence"] * 100.0,
                    summary["confidence_std_dev"] * 100.0,
                ]
                for _, model_name, _ in comparison.MODEL_SPECS:
                    result = record["models"][model_name]
                    row.extend([result["prediction"], result["confidence"] * 100.0])
                row.append("")
            else:
                row = [record["folder_label"] or "Unlabeled", record["relative_path"]]
                row.extend([""] * 6)
                for _ in comparison.MODEL_SPECS:
                    row.extend(["", ""])
                row.append(record["error"])
            sheet.append(row)

    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in predictions[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    predictions.row_dimensions[1].height = 42
    predictions.freeze_panes = "A2"
    predictions.auto_filter.ref = predictions.dimensions
    for col_index, header in enumerate(headers, start=1):
        if header in ("Relative path", "Error") or "checkpoint" in header.lower():
            width = 34
        elif "P(" in header:
            width = 16
        else:
            width = min(max(len(header) + 2, 14), 28)
        predictions.column_dimensions[get_column_letter(col_index)].width = width
        if "(%)" in header:
            for row_number in range(2, predictions.max_row + 1):
                predictions.cell(row_number, col_index).number_format = '0.00"%"'

    for sheet in [category_sheet, untrained_sheet, confusion_sheet, grouped_sheet] + [workbook[name] for _, name in spot_sheet_names]:
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.row_dimensions[1].height = 38
        for column_cells in sheet.columns:
            column_index = column_cells[0].column
            header = str(column_cells[0].value or "")
            if "path" in header.lower():
                width = 38
            elif "(%)" in header:
                width = 24
            else:
                width = min(max(len(header) + 2, 14), 30)
            sheet.column_dimensions[get_column_letter(column_index)].width = width
            if "(%)" in header:
                for row_number in range(2, sheet.max_row + 1):
                    sheet.cell(row_number, column_index).number_format = '0.00"%"'

    for cell in summary_sheet[1]:
        cell.font = Font(bold=True, size=14, color="1F4E78")
    summary_header_fill = PatternFill("solid", fgColor="D9EAF7")
    for row_number in (7, model_header_row, consistency_header_row):
        for cell in summary_sheet[row_number]:
            cell.font = Font(bold=True, color="1F1F1F")
            cell.fill = summary_header_fill
    for row_number in range(model_header_row + 1, model_header_row + len(comparison.MODEL_SPECS) + 1):
        summary_sheet.cell(row_number, 4).number_format = '0.00"%"'
    summary_sheet.column_dimensions["A"].width = 34
    summary_sheet.column_dimensions["B"].width = 28
    summary_sheet.freeze_panes = "A6"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        folder, "image_folder_evaluation_{}.xlsx".format(timestamp)
    )
    workbook.save(output_path)
    return output_path


def main():
    torch.set_num_threads(1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "Trained Images")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError("Trained Images folder not found: {}".format(checkpoint_dir))

    folder = choose_folder()
    image_paths = collect_images(folder)
    if not image_paths:
        raise ValueError("No supported image files found in: {}".format(folder))
    device = comparison.choose_device()
    print("Image folder:", folder)
    print("Images found:", len(image_paths))
    print("Device:", device)

    records, class_names, model_details = create_prediction_records(
        folder, image_paths, checkpoint_dir, device
    )
    for record in records:
        if record["error"]:
            continue
        record["summary"] = summarize_record(record, class_names)

    successful = [record for record in records if not record["error"]]
    for index, record in enumerate(successful, start=1):
        summary = record["summary"]
        consensus = " / ".join(summary["leading_labels"])
        model_results = " | ".join(
            "{}: {} ({:.1f}%)".format(
                item["name"], item["prediction"], item["confidence"] * 100.0
            )
            for item in summary["results"]
        )
        print("[{}/{}] {} | {} | {}/4 agree | {} | {}".format(
            index, len(successful), summary["spot"], consensus,
            summary["max_votes"], record["relative_path"], model_results,
        ), flush=True)
    for record in records:
        if record["error"]:
            print("[ERROR] {} | {}".format(record["relative_path"], record["error"]), flush=True)

    result_images_dir = os.path.join(folder, "_evaluation_results")
    # Create every classification directory so all defined Spot categories are
    # present in the output, including categories with zero matching samples.
    for spot_name in SPOT_NAMES:
        os.makedirs(os.path.join(result_images_dir, spot_name, "_graphs"), exist_ok=True)
    category_graph_root = os.path.join(result_images_dir, "By Category")
    source_categories = sorted({
        record["source_category"] for record in records
    }, key=str.lower)
    for category in source_categories:
        group_name = "Trained Categories" if category in class_names else "Untrained Categories"
        for spot_name in SPOT_NAMES:
            os.makedirs(os.path.join(
                category_graph_root, group_name, category, spot_name, "_graphs"
            ), exist_ok=True)
    print("Spot-organized graph output:", result_images_dir, flush=True)
    print("Category-organized graph output:", category_graph_root, flush=True)

    # Save every chart directly to disk. This evaluator does not open graph popups.
    graph_count = 0
    category_graph_count = 0
    graph_counts_by_spot = {}
    graph_failures = []
    for sample_number, record in enumerate(successful, start=1):
        try:
            with Image.open(record["path"]) as source_image:
                image = source_image.convert("RGB")
            spot = record["summary"]["spot"]
            source_subfolder = os.path.dirname(record["relative_path"])
            graph_dir = os.path.join(
                result_images_dir, spot, "_graphs", source_subfolder
            )
            graph_path = save_summary_graph(
                record["path"], image, class_names, record["summary"],
                graph_dir, sample_number,
            )
            record["graph_path"] = graph_path
            graph_count += 1
            graph_counts_by_spot[spot] = graph_counts_by_spot.get(spot, 0) + 1
            category = record["source_category"]
            category_group = (
                "Trained Categories" if category in class_names
                else "Untrained Categories"
            )
            source_parts = source_subfolder.split(os.sep) if source_subfolder else []
            if category in source_parts:
                category_index = source_parts.index(category)
                remaining_parts = source_parts[category_index + 1:]
                category_subfolder = os.path.join(*remaining_parts) if remaining_parts else ""
            else:
                category_subfolder = source_subfolder
            category_graph_dir = os.path.join(
                category_graph_root, category_group, category, spot,
                "_graphs", category_subfolder,
            )
            category_graph_path = save_summary_graph(
                record["path"], image, class_names, record["summary"],
                category_graph_dir, sample_number,
            )
            record["category_graph_path"] = category_graph_path
            category_graph_count += 1
            print("Saved graph [{}/{}] [{}]: {}".format(
                sample_number, len(successful), spot, graph_path
            ), flush=True)
        except Exception as error:
            graph_failures.append((record["relative_path"], error))
            print("[GRAPH ERROR] {} | {}: {}".format(
                record["relative_path"], type(error).__name__, error
            ), flush=True)

    print("Saved {} evaluation graph(s) by Spot under: {}".format(
        graph_count, result_images_dir
    ), flush=True)
    print("Saved {} category-organized graph(s) under: {}".format(
        category_graph_count, category_graph_root
    ), flush=True)
    for spot_name in SPOT_NAMES:
        print("  {}: {} graph(s)".format(
            spot_name, graph_counts_by_spot.get(spot_name, 0)
        ), flush=True)
    if graph_failures:
        print("Graph save failures: {}".format(len(graph_failures)), flush=True)

    excel_path = save_excel(folder, records, class_names, model_details)
    print("Excel report saved:", excel_path, flush=True)

    print("Evaluation complete: {} successful / {} total".format(
        len(successful), len(records)
    ))


if __name__ == "__main__":
    main()
