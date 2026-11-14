"""Transfer-learn an ImageNet-pretrained Vision Transformer (ViT-B/16).

Expected layout:
    dataset_root/
        class_a/ image files (nested folders are supported)
        class_b/ image files

Python 3.7 compatible macOS environment:
    python -m pip install torch==1.13.1 torchvision==0.14.1 pillow

The pretrained ViT-B/16 weights are downloaded on first run. The script performs
an 80/20 stratified split, trains the classification head, and saves the best
validation checkpoint. Images and folder labels are paired through ImageFolder.
"""

from __future__ import print_function

import os
import random
import sys
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


IMAGE_SIZE = 224
BATCH_SIZE = 8  # ViT-B/16 is larger than ResNet-18; reduce if memory is limited.
EPOCHS = 30
VALIDATION_RATIO = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 0
LEARNING_RATE_CLASSIFIER = 0.001
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Tee(object):
    """Write console output to both the terminal and the run log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def configure_ssl_certificates():
    """Point urllib/PyTorch downloads at certifi's trusted CA bundle when available."""
    try:
        import certifi
    except ImportError:
        print("certifi is not installed. On python.org macOS Python, run:")
        print('  open "/Applications/Python 3.10/Install Certificates.command"')
        print("or install certifi in this interpreter: python -m pip install certifi")
        return

    ca_bundle = certifi.where()
    configured_bundle = os.environ.get("SSL_CERT_FILE")
    if not configured_bundle or not os.path.isfile(configured_bundle):
        os.environ["SSL_CERT_FILE"] = ca_bundle
    print("TLS certificate bundle:", os.environ["SSL_CERT_FILE"])


def find_resume_checkpoint(dataset_tag):
    """Find the stable ViT checkpoint for the selected dataset only."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    expected_name = "best_vit_image_model_dataset_{}.pth".format(dataset_tag)
    direct_path = os.path.join(script_dir, expected_name)
    if os.path.isfile(direct_path):
        return direct_path

    for current_dir, subdirs, filenames in os.walk(script_dir):
        subdirs[:] = sorted(name for name in subdirs if name != ".git")
        if expected_name in filenames:
            return os.path.join(current_dir, expected_name)
    return None


def choose_dataset_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="이미지 데이터셋 최상위 폴더 선택")
    root.destroy()
    if not folder:
        raise SystemExit("폴더 선택이 취소되었습니다.")
    return folder


def dataset_checkpoint_tag(dataset_root):
    folder_name = os.path.basename(os.path.normpath(dataset_root))
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in folder_name
    ) or "dataset"


class ImageFolderWithPath(datasets.ImageFolder):
    """ImageFolder that also returns the source path for terminal logging."""

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPath, self).__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def split_indices_by_class(samples, validation_ratio, seed):
    rng = random.Random(seed)
    by_class = {}
    for index, (_, label) in enumerate(samples):
        by_class.setdefault(label, []).append(index)

    train_indices = []
    validation_indices = []
    for label in sorted(by_class):
        class_indices = by_class[label]
        rng.shuffle(class_indices)
        if len(class_indices) < 2:
            raise ValueError(
                "각 label에는 최소 2개 이미지가 필요합니다. "
                "label index {}에는 {}개가 있습니다.".format(label, len(class_indices))
            )
        validation_count = max(1, int(round(len(class_indices) * validation_ratio)))
        validation_count = min(validation_count, len(class_indices) - 1)
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def build_model(number_of_classes, device, pretrained=True):
    # pretrained=True is compatible with torchvision 0.14 / Python 3.7.
    model = models.vit_b_16(pretrained=pretrained)

    # Freeze the pretrained transformer and train only its new classification head.
    for parameter in model.parameters():
        parameter.requires_grad = False
    input_features = model.heads.head.in_features
    model.heads.head = nn.Linear(input_features, number_of_classes)
    for parameter in model.heads.parameters():
        parameter.requires_grad = True

    return model.to(device)


def load_compatible_vit_state_dict(model, state_dict):
    """Load ViT weights across torchvision versions with renamed MLP keys.

    Older torchvision versions stored the encoder MLP as ``mlp.0`` and
    ``mlp.3``; newer versions call those layers ``mlp.linear_1`` and
    ``mlp.linear_2``. The tensors are equivalent, so translate only these
    known aliases and still require a complete strict load.
    """
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

        matching_key = next((candidate for candidate in candidates if candidate in target_keys), None)
        if matching_key is None:
            raise RuntimeError(
                "Checkpoint contains an unsupported ViT parameter key: {}".format(key)
            )
        translated[matching_key] = value

    # strict=True catches missing weights and prevents partially initialized models.
    model.load_state_dict(translated, strict=True)


def run_epoch(model, loader, criterion, device, optimizer=None,
              class_names=None, print_samples=False):
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch_number, (images, labels, paths) in enumerate(loader, start=1):
            if is_training and print_samples:
                print("  Training batch {}/{}".format(batch_number, len(loader)))
                for label_index, path in zip(labels.tolist(), paths):
                    print("    label: {} | sample: {}".format(
                        class_names[label_index], os.path.basename(path)
                    ))

            images = images.to(device)
            labels = labels.to(device)
            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += batch_size

    return total_loss / total_items, total_correct / total_items


def run_training():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(1)

    dataset_root = os.path.abspath(choose_dataset_folder())
    dataset_tag = dataset_checkpoint_tag(dataset_root)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_source = ImageFolderWithPath(dataset_root, transform=train_transform)
    validation_source = ImageFolderWithPath(dataset_root, transform=validation_transform)
    if train_source.classes != validation_source.classes:
        raise RuntimeError("Train/validation label mappings do not match.")
    if not train_source.samples:
        raise ValueError("선택한 폴더에서 이미지 파일을 찾지 못했습니다.")

    train_indices, validation_indices = split_indices_by_class(
        train_source.samples, VALIDATION_RATIO, RANDOM_SEED
    )
    train_dataset = Subset(train_source, train_indices)
    validation_dataset = Subset(validation_source, validation_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    resume_path = find_resume_checkpoint(dataset_tag)
    checkpoint = None
    if resume_path:
        print("Found ViT checkpoint:", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        saved_classes = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
        if saved_classes and saved_classes != train_source.classes:
            raise ValueError(
                "Checkpoint label classes {} do not match selected dataset {}.".format(
                    saved_classes, train_source.classes
                )
            )
    else:
        print("No ViT checkpoint for dataset '{}'; starting from ImageNet pretrained weights.".format(
            dataset_tag
        ))
        configure_ssl_certificates()

    model = build_model(
        len(train_source.classes), device, pretrained=(checkpoint is None)
    )
    if checkpoint is not None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        load_compatible_vit_state_dict(model, state_dict)
        print("Loaded saved ViT weights.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.heads.parameters(), lr=LEARNING_RATE_CLASSIFIER, weight_decay=0.01
    )

    class_to_idx = train_source.class_to_idx
    print("Dataset:", dataset_root)
    print("Model: pretrained ViT-B/16 (frozen backbone, trainable classifier head)")
    print("Device:", device)
    print("Label mapping:", class_to_idx)
    print("Train images: {} | Validation images: {}".format(
        len(train_dataset), len(validation_dataset)
    ))
    print("Input: preprocessed color image [batch, 3, 224, 224] + folder label")

    resume_epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
    best_validation_accuracy = (
        checkpoint.get("validation_accuracy", -1.0) if isinstance(checkpoint, dict) else -1.0
    )
    best_validation_run_accuracy = -1.0
    best_training_accuracy = -1.0
    best_validation_checkpoint = resume_path
    best_validation_run_checkpoint = None
    best_training_checkpoint = None
    final_epoch = resume_epoch + EPOCHS

    if checkpoint is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Restored optimizer state from checkpoint.")
    if checkpoint is not None:
        print("Continuing after epoch {} for {} additional epochs.".format(resume_epoch, EPOCHS))

    for epoch in range(resume_epoch + 1, final_epoch + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            class_names=train_source.classes,
            print_samples=(epoch <= resume_epoch + 3),
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )
        print(
            "Epoch {}/{} | train loss {:.4f}, accuracy {:.2%} | "
            "validation loss {:.4f}, accuracy {:.2%}".format(
                epoch, final_epoch, train_loss, train_accuracy,
                validation_loss, validation_accuracy,
            )
        )

        if train_accuracy > best_training_accuracy:
            best_training_accuracy = train_accuracy
            train_accuracy_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "best_vit_b_16_train_acc_{:.2f}_epoch_{}_dataset_{}.pth".format(
                    train_accuracy * 100.0, epoch, dataset_tag
                ),
            )
            if best_training_checkpoint and os.path.exists(best_training_checkpoint):
                os.remove(best_training_checkpoint)
            torch.save({
                "model_name": "vit_b_16",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": train_source.classes,
                "class_to_idx": class_to_idx,
                "image_size": IMAGE_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "epoch": epoch,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            }, train_accuracy_path)
            print("  New best training accuracy: {:.2%}; saved model: {}".format(
                train_accuracy, train_accuracy_path
            ))
            best_training_checkpoint = train_accuracy_path

        # Keep one best validation checkpoint for this run, independently from
        # the global best checkpoint used to resume training.
        if validation_accuracy > best_validation_run_accuracy:
            best_validation_run_accuracy = validation_accuracy
            validation_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "best_vit_b_16_val_acc_{:.2f}_epoch_{}_dataset_{}.pth".format(
                    validation_accuracy * 100.0, epoch, dataset_tag
                ),
            )
            if best_validation_run_checkpoint and os.path.exists(best_validation_run_checkpoint):
                os.remove(best_validation_run_checkpoint)
            torch.save({
                "model_name": "vit_b_16",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": train_source.classes,
                "class_to_idx": class_to_idx,
                "image_size": IMAGE_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "epoch": epoch,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            }, validation_path)
            best_validation_run_checkpoint = validation_path
            best_validation_checkpoint = validation_path
            print("  New best validation accuracy this run: {:.2%}; saved model: {}".format(
                validation_accuracy, validation_path
            ))

        # The stable resume checkpoint is updated only when validation
        # accuracy beats the best value already stored in the resume file.
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            resume_save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "best_vit_image_model_dataset_{}.pth".format(dataset_tag),
            )
            torch.save({
                "model_name": "vit_b_16",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": train_source.classes,
                "class_to_idx": class_to_idx,
                "image_size": IMAGE_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "epoch": epoch,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "dataset_size": len(train_source.samples),
                "validation_ratio": VALIDATION_RATIO,
                "random_seed": RANDOM_SEED,
            }, resume_save_path)
            print("  Saved resumable validation checkpoint:", resume_save_path)

    print("Training complete. Best training accuracy: {:.2%}".format(best_training_accuracy))
    print("Best validation accuracy: {:.2%}".format(best_validation_accuracy))
    print("Best training-accuracy checkpoint:", best_training_checkpoint)
    print("Best validation checkpoint:", best_validation_checkpoint)
    print("Resume checkpoint:", os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "best_vit_image_model_dataset_{}.pth".format(dataset_tag),
    ))


def main():
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "training_vit_b_16_{}.log".format(run_stamp),
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        with open(log_path, "w") as log_file:
            sys.stdout = Tee(original_stdout, log_file)
            sys.stderr = Tee(original_stderr, log_file)
            print("Training log:", log_path)
            run_training()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == "__main__":
    main()
