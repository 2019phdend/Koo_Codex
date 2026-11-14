"""Compact agreement/confidence summary for the four trained image models."""

from __future__ import print_function

import math
import os
import tkinter as tk
from tkinter import ttk

import torch
from PIL import Image, ImageTk
from torchvision import transforms

import compare_trained_image_models as comparison


def run_inference(image_path, checkpoint_paths, device):
    image = Image.open(image_path).convert("RGB")
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(comparison.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(comparison.IMAGENET_MEAN, comparison.IMAGENET_STD),
    ])
    input_tensor = preprocessing(image).unsqueeze(0).to(device)

    class_names = None
    results = []
    with torch.no_grad():
        for display_name, model_name, _ in comparison.MODEL_SPECS:
            checkpoint_path = checkpoint_paths[model_name]
            checkpoint = torch.load(checkpoint_path, map_location=device)
            saved_classes = checkpoint.get("class_names")
            if not saved_classes:
                raise ValueError("Checkpoint has no class_names: {}".format(checkpoint_path))
            if class_names is None:
                class_names = list(saved_classes)

            model, checkpoint = comparison.load_model(
                model_name, checkpoint, device, class_names, checkpoint_path
            )
            probabilities = torch.softmax(model(input_tensor), dim=1)[0].cpu().tolist()
            predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            results.append({
                "name": display_name,
                "prediction": class_names[predicted_index],
                "confidence": float(probabilities[predicted_index]),
                "probabilities": probabilities,
            })
            del model
    return image, class_names, results


def draw_summary(image_path, image, class_names, results):
    vote_counts = {}
    for result in results:
        vote_counts[result["prediction"]] = vote_counts.get(result["prediction"], 0) + 1
    most_votes = max(vote_counts.values())
    winning_classes = [name for name in class_names if vote_counts.get(name, 0) == most_votes]
    if len(winning_classes) == 1:
        consensus_text = winning_classes[0]
    else:
        consensus_text = "Tie: {}".format(" / ".join(winning_classes))

    agreeing = [result for result in results if result["prediction"] in winning_classes]
    confidences = [result["confidence"] for result in agreeing]
    mean_confidence = sum(confidences) / len(confidences)
    std_confidence = math.sqrt(
        sum((value - mean_confidence) ** 2 for value in confidences) / len(confidences)
    )
    min_confidence = min(confidences)
    agreement_rate = most_votes / float(len(results))
    majority_pass = most_votes > len(results) / 2.0
    unanimous = most_votes == len(results)
    mean_confidence_pass = mean_confidence >= 0.90
    min_confidence_pass = min_confidence >= 0.80
    bright_quality_rule = None
    if min_confidence_pass and unanimous and mean_confidence >= 0.99:
        bright_quality_rule = "Unanimous · mean ≥99% · minimum ≥80%"
        spot_name = "BRIGHT SPOT PRIME"
    elif min_confidence_pass and unanimous and mean_confidence >= 0.95:
        bright_quality_rule = "Unanimous · mean ≥95% · minimum ≥80%"
        spot_name = "BRIGHT SPOT HIGH"
    elif min_confidence_pass and majority_pass and mean_confidence >= 0.95:
        bright_quality_rule = "Majority · mean ≥95% · minimum ≥80%"
        spot_name = "BRIGHT SPOT"
    elif (min_confidence_pass and majority_pass and mean_confidence >= 0.90
          and std_confidence < 0.075):
        bright_quality_rule = "Majority · mean ≥90% · minimum ≥80% · SD <7.5%"
        spot_name = "BRIGHT SPOT LOW 1"
    elif (unanimous and mean_confidence >= 0.80 and min_confidence >= 0.60
          and std_confidence < 0.075):
        bright_quality_rule = "Unanimous · mean ≥80% · minimum ≥60% · SD <7.5%"
        spot_name = "BRIGHT SPOT LOW 2"
    else:
        spot_name = None

    is_bright_spot = bright_quality_rule is not None
    is_gray_spot = (
        not is_bright_spot and majority_pass and mean_confidence_pass
        and 0.70 <= min_confidence < 0.80
    )
    if is_gray_spot:
        spot_name = "GRAY SPOT"
    elif not is_bright_spot:
        spot_name = "BLIND SPOT"

    blind_reasons = []
    if not majority_pass:
        blind_reasons.append("Only {}/4 models agree; at least 3 are required.".format(most_votes))
    if mean_confidence < 0.90 and not is_gray_spot:
        blind_reasons.append("Agreement mean is {:.1f}%; minimum Bright Spot threshold is 90%.".format(
            mean_confidence * 100.0
        ))
    if not min_confidence_pass and not is_gray_spot:
        if min_confidence < 0.70:
            blind_reasons.append(
                "Lowest agreement confidence is {:.1f}%; below the 70% Gray Spot floor.".format(
                    min_confidence * 100.0
                )
            )
        else:
            blind_reasons.append(
                "Minimum confidence is {:.1f}% (Gray range), but Gray also requires majority and mean thresholds.".format(
                    min_confidence * 100.0
                )
            )
    low_1_candidate = (
        majority_pass and mean_confidence >= 0.90 and min_confidence >= 0.80
    )
    low_2_candidate = (
        unanimous and mean_confidence >= 0.80 and min_confidence >= 0.60
    )
    if (not is_bright_spot and not is_gray_spot
            and (low_1_candidate or low_2_candidate) and std_confidence >= 0.075):
        blind_reasons.append(
            "Confidence SD is {:.1f}%; Bright Spot Low requires SD below 7.5%.".format(
                std_confidence * 100.0
            )
        )

    root = tk.Tk()
    root.title("Image Model Summary")
    root.geometry("1320x860")
    root.minsize(1000, 720)
    background = "#F3F5F8"
    ink = "#182230"
    muted = "#64748B"
    accent = "#315C9B"
    success = "#16845B"
    root.configure(background=background)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("TFrame", background=background)
    style.configure("Card.TFrame", background="white")
    style.configure("TLabel", background=background, foreground=ink, font=("Arial", 10))
    style.configure("Title.TLabel", font=("Arial", 22, "bold"), foreground=ink)
    style.configure("Subtitle.TLabel", font=("Arial", 10), foreground=muted)
    style.configure("CardTitle.TLabel", background="white", font=("Arial", 11, "bold"), foreground=ink)
    style.configure("KpiLabel.TLabel", background="white", font=("Arial", 9), foreground=muted)
    style.configure("KpiValue.TLabel", background="white", font=("Arial", 19, "bold"), foreground=ink)
    style.configure("ModelName.TLabel", background="white", font=("Arial", 9), foreground=muted)
    style.configure("ModelPrediction.TLabel", background="white", font=("Arial", 12, "bold"), foreground=ink)

    page = ttk.Frame(root, padding=20)
    page.pack(fill="both", expand=True)
    ttk.Label(page, text="MODEL COMPARISON", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        page, text="Four-model consensus and confidence at a glance", style="Subtitle.TLabel"
    ).pack(anchor="w", pady=(3, 14))

    spot_color = "#16845B" if is_bright_spot else ("#64748B" if is_gray_spot else "#B45309")
    spot_bg = "#E7F5EF" if is_bright_spot else ("#EEF1F5" if is_gray_spot else "#FFF4E5")
    spot_banner = tk.Frame(page, background=spot_bg, padx=14, pady=10)
    spot_banner.pack(fill="x", pady=(0, 12))
    tk.Label(
        spot_banner, text=spot_name, background=spot_bg, foreground=spot_color,
        font=("Arial", 22, "bold"),
    ).pack(side="left", padx=(0, 16))
    if is_bright_spot:
        rule_text = "Criteria: {}\nResult: {} models choose {} · mean {:.1f}% · minimum {:.1f}%".format(
            bright_quality_rule, most_votes, consensus_text,
            mean_confidence * 100.0, min_confidence * 100.0
        )
    elif is_gray_spot:
        rule_text = (
            "Consensus and mean confidence pass. Minimum confidence is in the gray range.\n"
            "Minimum: {:.1f}% (gray range: 70% to <80%)"
        ).format(min_confidence * 100.0)
    else:
        rule_text = "WHY THIS IS A BLIND SPOT\n• " + "\n• ".join(blind_reasons)
    tk.Label(
        spot_banner,
        text=("WHY THIS IS A GRAY SPOT\n" + rule_text) if is_gray_spot else rule_text,
        background=spot_bg, foreground=ink, justify="left",
        font=("Arial", 11 if not is_bright_spot else 12),
    ).pack(side="left")

    true_label = None
    current_dir = os.path.dirname(os.path.abspath(image_path))
    while True:
        current_name = os.path.basename(current_dir)
        if current_name in class_names:
            true_label = current_name
            break
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    hero = ttk.Frame(page, style="Card.TFrame", padding=14)
    hero.pack(fill="x", pady=(0, 12))
    thumbnail = image.copy()
    thumbnail.thumbnail((170, 140))
    photo = ImageTk.PhotoImage(thumbnail)
    image_label = ttk.Label(hero, image=photo, background="white")
    image_label.image = photo
    image_label.grid(row=0, column=0, rowspan=2, padx=(0, 16), sticky="w")
    ttk.Label(
        hero, text=os.path.basename(image_path), style="CardTitle.TLabel", wraplength=250
    ).grid(row=0, column=1, sticky="w", padx=(0, 14))
    if true_label:
        ttk.Label(hero, text="Folder label: {}".format(true_label),
                  style="KpiLabel.TLabel").grid(row=1, column=1, sticky="w")
    else:
        ttk.Label(hero, text="Selected sample", style="KpiLabel.TLabel").grid(
            row=1, column=1, sticky="w"
        )

    def add_metric(column, label, value, value_color=ink):
        metric = ttk.Frame(hero, style="Card.TFrame", padding=(14, 8))
        metric.grid(row=0, column=column, rowspan=2, sticky="nsew", padx=5)
        ttk.Label(metric, text=label.upper(), style="KpiLabel.TLabel").pack(anchor="w")
        ttk.Label(metric, text=value, style="KpiValue.TLabel", foreground=value_color).pack(
            anchor="w", pady=(4, 0)
        )
        hero.columnconfigure(column, weight=1)

    add_metric(2, "Consensus", consensus_text, success if len(winning_classes) == 1 else accent)
    add_metric(3, "Agreement", "{}/{} · {:.0f}%".format(
        most_votes, len(results), agreement_rate * 100.0
    ), accent)
    add_metric(4, "Confidence · mean", "{:.1f}%".format(mean_confidence * 100.0),
               success if mean_confidence_pass else ink)
    add_metric(5, "Confidence · minimum", "{:.1f}%".format(min_confidence * 100.0),
               success if min_confidence_pass else ink)
    add_metric(6, "Confidence · SD", "±{:.1f}%".format(std_confidence * 100.0), muted)

    charts = ttk.Frame(page)
    charts.pack(fill="both", expand=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"]

    if is_bright_spot:
        # Bright cases prioritize the final, stable consensus and keep detail compact.
        result_card = ttk.Frame(charts, style="Card.TFrame", padding=18)
        result_card.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Label(result_card, text="FINAL CONSENSUS", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            result_card, text=consensus_text, font=("Arial", 30, "bold"),
            foreground=success, background="white",
        ).pack(anchor="w", pady=(8, 2))
        ttk.Label(
            result_card,
            text="{}/{} models agree  ·  Mean confidence {:.1f}%  ·  Minimum {:.1f}%  ·  SD {:.1f}%".format(
                most_votes, len(results), mean_confidence * 100.0,
                min_confidence * 100.0, std_confidence * 100.0,
            ),
            style="KpiLabel.TLabel",
        ).pack(anchor="w", pady=(0, 14))
        ttk.Label(result_card, text="MODEL CHECK", style="CardTitle.TLabel").pack(anchor="w")
        for result in results:
            ttk.Label(
                result_card,
                text="✓  {}  →  {}  ({:.1f}%)".format(
                    result["name"], result["prediction"], result["confidence"] * 100.0
                ),
                background="white", foreground=success, font=("Arial", 11),
            ).pack(anchor="w", pady=3)
    else:
        # Gray and blind cases expose confidence details and model-level evidence.
        diagnostics = ttk.Frame(charts)
        diagnostics.pack(fill="both", expand=True)
        diagnostics.columnconfigure(0, weight=1)
        diagnostics.columnconfigure(1, weight=1)
        diagnostics.rowconfigure(0, weight=1)

        vote_card = ttk.Frame(diagnostics, style="Card.TFrame", padding=14)
        vote_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=6)
        ttk.Label(vote_card, text="WHY THE MODELS DISAGREE", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(vote_card, text="Vote count by predicted label (4 models)", style="KpiLabel.TLabel").pack(
            anchor="w", pady=(2, 10)
        )
        vote_canvas = tk.Canvas(vote_card, height=220, background="white", highlightthickness=0)
        vote_canvas.pack(fill="both", expand=True)
        vote_canvas.update_idletasks()
        vote_width = max(vote_canvas.winfo_width(), 430)
        vote_bar_left = 112
        vote_bar_width = max(vote_width - vote_bar_left - 58, 170)
        for index, label in enumerate(class_names):
            count = vote_counts.get(label, 0)
            y = 12 + index * 38
            is_winner = label in winning_classes
            vote_canvas.create_text(
                vote_bar_left - 10, y + 11, text=label, anchor="e",
                fill=ink if is_winner else muted,
                font=("Arial", 10, "bold" if is_winner else "normal"),
            )
            vote_canvas.create_rectangle(
                vote_bar_left, y, vote_bar_left + vote_bar_width, y + 22,
                fill="#EDF0F4", outline="",
            )
            if count:
                vote_canvas.create_rectangle(
                    vote_bar_left, y,
                    vote_bar_left + vote_bar_width * count / len(results), y + 22,
                    fill=success if is_winner else colors[index % len(colors)], outline="",
                )
            vote_canvas.create_text(
                vote_bar_left + vote_bar_width + 8, y + 11,
                text="{}/{}".format(count, len(results)), anchor="w",
                fill=ink, font=("Arial", 9, "bold"),
            )

        confidence_card = ttk.Frame(diagnostics, style="Card.TFrame", padding=14)
        confidence_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=6)
        ttk.Label(confidence_card, text="CONFIDENCE BY MODEL", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            confidence_card, text="Agreement group is green; dissenting models are orange", style="KpiLabel.TLabel"
        ).pack(anchor="w", pady=(2, 10))
        confidence_canvas = tk.Canvas(confidence_card, height=220, background="white", highlightthickness=0)
        confidence_canvas.pack(fill="both", expand=True)
        confidence_canvas.update_idletasks()
        confidence_width = max(confidence_canvas.winfo_width(), 430)
        conf_label_width = 138
        conf_bar_width = max(confidence_width - conf_label_width - 65, 170)
        for index, result in enumerate(results):
            y = 10 + index * 37
            is_agreeing = result["prediction"] in winning_classes
            confidence_canvas.create_text(
                conf_label_width - 9, y + 10, text=result["name"], anchor="e",
                fill=muted, font=("Arial", 9),
            )
            confidence_canvas.create_rectangle(
                conf_label_width, y, conf_label_width + conf_bar_width, y + 20,
                fill="#EDF0F4", outline="",
            )
            confidence_canvas.create_rectangle(
                conf_label_width, y,
                conf_label_width + conf_bar_width * result["confidence"], y + 20,
                fill=success if is_agreeing else "#D97706", outline="",
            )
            confidence_canvas.create_text(
                conf_label_width + conf_bar_width + 8, y + 10,
                text="{:.1f}%".format(result["confidence"] * 100.0), anchor="w",
                fill=ink, font=("Arial", 9, "bold"),
            )
        # Reference lines show individual confidence thresholds, without treating
        # them as extra decision rules beyond the mean/minimum checks.
        x80 = conf_label_width + conf_bar_width * 0.80
        x90 = conf_label_width + conf_bar_width * 0.90
        x70 = conf_label_width + conf_bar_width * 0.70
        confidence_canvas.create_line(x70, 0, x70, 165, fill="#64748B", dash=(3, 3))
        confidence_canvas.create_line(x80, 0, x80, 165, fill="#D97706", dash=(3, 3))
        confidence_canvas.create_line(x90, 0, x90, 165, fill=accent, dash=(3, 3))
        confidence_canvas.create_text(x70, 155, text="70%", anchor="n", fill="#64748B",
                                      font=("Arial", 8, "bold"))
        confidence_canvas.create_text(x80, 169, text="80% min", anchor="n", fill="#D97706",
                                      font=("Arial", 8, "bold"))
        confidence_canvas.create_text(x90, 183, text="90% mean", anchor="n", fill=accent,
                                      font=("Arial", 8, "bold"))

    if not is_bright_spot:
        results_card = ttk.Frame(page, style="Card.TFrame", padding=12)
        results_card.pack(fill="x", pady=(12, 0))
        ttk.Label(results_card, text="MODEL-BY-MODEL EVIDENCE", style="CardTitle.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        model_row = ttk.Frame(results_card, style="Card.TFrame")
        model_row.pack(fill="x")
        for index, result in enumerate(results):
            is_agreeing = result["prediction"] in winning_classes
            status = "AGREES" if is_agreeing else "DISSENTS"
            status_color = success if is_agreeing else "#B45309"
            model_card = ttk.Frame(model_row, style="Card.TFrame", padding=(12, 6))
            model_card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 8, 0))
            model_row.columnconfigure(index, weight=1)
            ttk.Label(model_card, text=result["name"], style="ModelName.TLabel").pack(anchor="w")
            ttk.Label(model_card, text=result["prediction"], style="ModelPrediction.TLabel").pack(
                anchor="w", pady=2
            )
            ttk.Label(
                model_card, text="{:.1f}% confidence · {}".format(
                    result["confidence"] * 100.0, status
                ), background="white", foreground=status_color, font=("Arial", 9, "bold"),
            ).pack(anchor="w")

    print("{} | agreement {}/{} | consensus mean confidence {:.2f}% | minimum {:.2f}% | SD {:.2f}%".format(
        spot_name, most_votes, len(results), mean_confidence * 100.0,
        min_confidence * 100.0, std_confidence * 100.0,
    ))

    root.mainloop()


def main():
    torch.set_num_threads(1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "Trained Images")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError("Trained Images folder not found: {}".format(checkpoint_dir))
    image_path = comparison.choose_image()
    checkpoint_paths = comparison.find_checkpoints(checkpoint_dir)
    device = comparison.choose_device()
    print("Selected image:", image_path)
    print("Device:", device)
    image, class_names, results = run_inference(image_path, checkpoint_paths, device)
    for result in results:
        print("{}: {} ({:.2f}%)".format(
            result["name"], result["prediction"], result["confidence"] * 100.0
        ))
    draw_summary(image_path, image, class_names, results)


if __name__ == "__main__":
    main()
