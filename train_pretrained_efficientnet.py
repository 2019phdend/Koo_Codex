"""Fine-tune an ImageNet-pretrained EfficientNet-B0 on a folder-labeled dataset.

Expected layout:
    dataset_root/
        class_a/ image files (nested folders are supported)
        class_b/ image files

Python 3.7 compatible environment (macOS):
    python -m pip install torch==1.13.1 torchvision==0.14.1 pillow

The pretrained EfficientNet-B0 weights are downloaded by torchvision on first run.
The script uses the folder name as the class label, creates a stratified
train/validation split, fine-tunes the model, and saves the best validation model.
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
BATCH_SIZE = 16
EPOCHS = 30
VALIDATION_RATIO = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 0  # Tkinter/macOS/PyCharm friendly.
LEARNING_RATE_BACKBONE = 0.0001
LEARNING_RATE_CLASSIFIER = 0.001

# ImageNet normalization used by the pretrained EfficientNet-B0 weights.
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
    """Make a reproducible split while keeping each class in both partitions."""
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


def build_model(number_of_classes, device):
    # pretrained=True works with torchvision 0.14 (Python 3.7 compatible).
    model = models.efficientnet_b0(pretrained=True)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, number_of_classes)
    model = model.to(device)
    return model


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
    # Prevent CPU oversubscription; this does not bypass OpenMP runtime conflicts.
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

    # Separate dataset instances allow train augmentation and clean validation images.
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

    configure_ssl_certificates()
    model = build_model(len(train_source.classes), device)
    criterion = nn.CrossEntropyLoss()
    # Fine-tune all pretrained layers gently; learn the new classification head faster.
    optimizer = optim.Adam([
        {"params": [p for name, p in model.named_parameters() if not name.startswith("classifier.")],
         "lr": LEARNING_RATE_BACKBONE},
        {"params": model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER},
    ])

    class_to_idx = train_source.class_to_idx
    print("Dataset:", dataset_root)
    print("Device:", device)
    print("Label mapping:", class_to_idx)
    print("Train images: {} | Validation images: {}".format(
        len(train_dataset), len(validation_dataset)
    ))
    print("Model: pretrained EfficientNet-B0 (fine-tuning all layers)")
    print("Input to model: preprocessed color image [batch, 3, 224, 224] + folder label")

    best_validation_accuracy = -1.0
    best_training_accuracy = -1.0
    best_validation_checkpoint = None
    best_training_checkpoint = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            class_names=train_source.classes,
            print_samples=(epoch <= 3),
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )
        print(
            "Epoch {}/{} | train loss {:.4f}, accuracy {:.2%} | "
            "validation loss {:.4f}, accuracy {:.2%}".format(
                epoch, EPOCHS, train_loss, train_accuracy,
                validation_loss, validation_accuracy,
            )
        )

        if train_accuracy > best_training_accuracy:
            best_training_accuracy = train_accuracy
            train_accuracy_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "best_efficientnet_b0_train_acc_{:.2f}_epoch_{}_dataset_{}.pth".format(
                    train_accuracy * 100.0, epoch, dataset_tag
                ),
            )
            if best_training_checkpoint and os.path.exists(best_training_checkpoint):
                os.remove(best_training_checkpoint)
            torch.save({
                "model_name": "efficientnet_b0",
                "model_state_dict": model.state_dict(),
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

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            validation_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "best_efficientnet_b0_val_acc_{:.2f}_epoch_{}_dataset_{}.pth".format(
                    validation_accuracy * 100.0, epoch, dataset_tag
                ),
            )
            if best_validation_checkpoint and os.path.exists(best_validation_checkpoint):
                os.remove(best_validation_checkpoint)
            torch.save({
                "model_name": "efficientnet_b0",
                "model_state_dict": model.state_dict(),
                "class_names": train_source.classes,
                "class_to_idx": class_to_idx,
                "image_size": IMAGE_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "epoch": epoch,
                "validation_accuracy": validation_accuracy,
            }, validation_path)
            best_validation_checkpoint = validation_path
            print("  Saved best validation model:", validation_path)

    print("Training complete. Best training accuracy: {:.2%}".format(best_training_accuracy))
    print("Best validation accuracy: {:.2%}".format(best_validation_accuracy))
    print("Best training-accuracy checkpoint:", best_training_checkpoint)
    print("Best validation checkpoint:", best_validation_checkpoint)


def main():
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "training_efficientnet_b0_{}.log".format(run_stamp),
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
