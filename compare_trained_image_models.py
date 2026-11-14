"""Compare four trained image classifiers on one image chosen in a popup."""

from __future__ import print_function

import os
import glob
import re
import tkinter as tk
from tkinter import filedialog, ttk

import torch
import torch.nn as nn
from PIL import Image
from PIL import ImageTk
from torchvision import models, transforms


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_SPECS = [
    ("ResNet-18", "resnet18", "best_resnet18_val_acc_*_epoch_*.pth"),
    ("EfficientNet-B0", "efficientnet_b0", "best_efficientnet_b0_val_acc_*_epoch_*.pth"),
    ("Vision Transformer B/16", "vit_b_16", "best_vit_b_16_val_acc_*_epoch_*.pth"),
    ("Swin Transformer-T", "swin_t", "best_swin_t_val_acc_*_epoch_*.pth"),
]


def category_confusion_counts(records, class_names):
    """Count model predictions by actual folder-label category."""
    counts = {
        model_name: {
            actual: {predicted: 0 for predicted in class_names}
            for actual in class_names
        }
        for _, model_name, _ in MODEL_SPECS
    }
    for record in records:
        actual = record.get("folder_label")
        if record.get("error") or actual not in class_names:
            continue
        for _, model_name, _ in MODEL_SPECS:
            predicted = record["models"][model_name]["prediction"]
            if predicted in counts[model_name][actual]:
                counts[model_name][actual][predicted] += 1
    return counts


def choose_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Choose an image to classify",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        raise SystemExit("No image was selected.")
    return path


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_accuracy_epoch(path):
    match = re.search(
        r"_(?:val|train)_acc_([0-9.]+)_epoch_(\d+)(?:_dataset_.+)?\.pth$",
        os.path.basename(path),
    )
    if not match:
        return (-1.0, -1)
    return float(match.group(1)), int(match.group(2))


def find_checkpoints(checkpoint_dir):
    found = {}
    for display_name, model_name, pattern in MODEL_SPECS:
        candidates = glob.glob(os.path.join(checkpoint_dir, pattern))
        if not candidates:
            train_pattern = pattern.replace("_val_acc_", "_train_acc_")
            candidates = glob.glob(os.path.join(checkpoint_dir, train_pattern))
            if candidates:
                print(
                    "No validation checkpoint for {}; using training-accuracy checkpoint(s).".format(
                        display_name
                    )
                )
        if not candidates:
            raise FileNotFoundError(
                "{} checkpoint not found. Expected pattern: {}".format(display_name, pattern)
            )
        # If multiple validation checkpoints exist, select the highest validation accuracy.
        found[model_name] = max(candidates, key=checkpoint_accuracy_epoch)
    return found


def build_model(model_name, number_of_classes):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, number_of_classes)
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, number_of_classes)
    elif model_name == "swin_t":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, number_of_classes)
    else:
        raise ValueError("Unsupported model: {}".format(model_name))
    return model


def load_weights_compatibly(model, state_dict, model_name):
    """Translate the known torchvision ViT MLP key rename, then load strictly."""
    if model_name != "vit_b_16":
        model.load_state_dict(state_dict, strict=True)
        return

    target_keys = set(model.state_dict().keys())
    translated = {}
    for key, value in state_dict.items():
        candidates = [key]
        if ".mlp.0." in key:
            candidates.append(key.replace(".mlp.0.", ".mlp.linear_1."))
        if ".mlp.3." in key:
            candidates.append(key.replace(".mlp.3.", ".mlp.linear_2."))
        if ".mlp.linear_1." in key:
            candidates.append(key.replace(".mlp.linear_1.", ".mlp.0."))
        if ".mlp.linear_2." in key:
            candidates.append(key.replace(".mlp.linear_2.", ".mlp.3."))
        target_key = next((item for item in candidates if item in target_keys), None)
        if target_key is None:
            raise RuntimeError("Unsupported ViT checkpoint key: {}".format(key))
        translated[target_key] = value
    model.load_state_dict(translated, strict=True)


def load_model(model_name, checkpoint, device, expected_classes, checkpoint_path):
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: {}".format(checkpoint_path))
    saved_classes = checkpoint.get("class_names")
    if not saved_classes:
        raise ValueError("Checkpoint has no class_names: {}".format(checkpoint_path))
    if list(saved_classes) != list(expected_classes):
        raise ValueError(
            "Label mapping mismatch in {}. Expected {}, got {}.".format(
                os.path.basename(checkpoint_path), expected_classes, saved_classes
            )
        )

    model = build_model(model_name, len(expected_classes))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    load_weights_compatibly(model, state_dict, model_name)
    model.to(device)
    model.eval()
    return model, checkpoint


def main():
    torch.set_num_threads(1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "Trained Images")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError("Trained Images folder not found: {}".format(checkpoint_dir))

    image_path = choose_image()
    device = choose_device()
    checkpoints = find_checkpoints(checkpoint_dir)
    print("Selected image:", image_path)
    print("Checkpoint folder:", checkpoint_dir)
    print("Device:", device)

    image = Image.open(image_path).convert("RGB")
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    input_tensor = preprocessing(image).unsqueeze(0).to(device)

    predictions = []
    class_names = None
    with torch.no_grad():
        for display_name, model_name, _ in MODEL_SPECS:
            checkpoint_path = checkpoints[model_name]
            checkpoint_info = torch.load(checkpoint_path, map_location=device)
            model_classes = checkpoint_info.get("class_names")
            if class_names is None:
                class_names = list(model_classes)
            model, checkpoint = load_model(
                model_name, checkpoint_info, device, class_names, checkpoint_path
            )
            probabilities = torch.softmax(model(input_tensor), dim=1)[0].cpu().tolist()
            predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            accuracy, epoch = checkpoint_accuracy_epoch(checkpoint_path)
            if "_train_acc_" in os.path.basename(checkpoint_path):
                accuracy = float(checkpoint.get("validation_accuracy", -1.0)) * 100.0
            if accuracy < 0:
                accuracy = float(checkpoint.get("validation_accuracy", 0.0)) * 100.0
                epoch = int(checkpoint.get("epoch", 0))
            predictions.append({
                "name": display_name,
                "model": model_name,
                "path": checkpoint_path,
                "probabilities": probabilities,
                "predicted": class_names[predicted_index],
                "confidence": probabilities[predicted_index],
                "validation_accuracy": accuracy,
                "epoch": epoch,
            })
            del model

    # If the selected image is inside a label-named folder, show its folder label.
    parent_name = os.path.basename(os.path.dirname(image_path))
    actual_label = parent_name if parent_name in class_names else None

    for result in predictions:
        print("{}: {} ({:.2f}%) | validation checkpoint: {}".format(
            result["name"], result["predicted"], result["confidence"] * 100.0,
            os.path.basename(result["path"]),
        ))

    # Draw the comparison with Tkinter Canvas. Keeping Matplotlib/NumPy out of
    # this process avoids the common libiomp5/libomp collision on macOS.
    root = tk.Tk()
    root.title("Image Model Comparison")
    root.geometry("1400x900")
    root.minsize(1000, 700)

    header = ttk.Frame(root, padding=12)
    header.pack(fill="x")
    preview = image.copy()
    preview.thumbnail((240, 190))
    image_photo = ImageTk.PhotoImage(preview)
    image_widget = ttk.Label(header, image=image_photo)
    image_widget.image = image_photo
    image_widget.pack(side="left", padx=(0, 18))

    header_text = "Selected image: {}\n".format(os.path.basename(image_path))
    if actual_label:
        header_text += "Folder label: {}\n".format(actual_label)
    header_text += "Input preprocessing: Resize(256) -> CenterCrop(224) -> ImageNet normalization"
    ttk.Label(header, text=header_text, justify="left", wraplength=1000).pack(side="left", anchor="center")

    cards = ttk.Frame(root, padding=(12, 0, 12, 12))
    cards.pack(fill="both", expand=True)
    cards.columnconfigure(0, weight=1)
    cards.columnconfigure(1, weight=1)
    cards.rowconfigure(0, weight=1)
    cards.rowconfigure(1, weight=1)
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"]

    for index, result in enumerate(predictions):
        row, column = divmod(index, 2)
        card = ttk.LabelFrame(cards, text=result["name"], padding=12)
        card.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
        correct_text = ""
        if actual_label:
            correct_text = " | {}".format(
                "Correct" if result["predicted"] == actual_label else "Incorrect"
            )
        ttk.Label(
            card,
            text="Prediction: {} ({:.1f}%){}\nCheckpoint validation accuracy: {:.2f}% | epoch {}".format(
                result["predicted"], result["confidence"] * 100.0, correct_text,
                result["validation_accuracy"], result["epoch"],
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        chart = tk.Canvas(card, height=170, background="white", highlightthickness=0)
        chart.pack(fill="both", expand=True)
        chart.update_idletasks()
        chart_width = max(chart.winfo_width(), 420)
        label_width = 100
        value_width = 55
        bar_left = label_width
        bar_right = chart_width - value_width
        bar_width = max(bar_right - bar_left, 120)
        row_height = 29
        for class_index, (label, probability) in enumerate(
            zip(class_names, result["probabilities"])
        ):
            y = 12 + class_index * row_height
            percent = probability * 100.0
            chart.create_text(label_width - 8, y + 9, text=label, anchor="e")
            chart.create_rectangle(
                bar_left, y, bar_left + bar_width, y + 18,
                fill="#E8E8E8", outline="",
            )
            chart.create_rectangle(
                bar_left, y, bar_left + bar_width * probability, y + 18,
                fill=palette[class_index % len(palette)], outline="",
            )
            chart.create_text(
                bar_right + 8, y + 9, text="{:.1f}%".format(percent), anchor="w"
            )

    root.mainloop()


if __name__ == "__main__":
    main()
