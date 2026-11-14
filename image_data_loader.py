"""Folder-based image data loader with a first-batch preprocessing preview.

Expected directory layout:
    dataset_root/
        class_a/ image files (nested folders are also searched)
        class_b/ image files

Install dependencies:
    pip install numpy matplotlib pillow
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
_FOLDER_DIALOG_ROOT = None  # Keep Tk alive while Matplotlib displays its window.


def choose_dataset_folder() -> Path:
    """Open a native folder picker and return the selected dataset directory."""
    import tkinter as tk
    from tkinter import filedialog

    global _FOLDER_DIALOG_ROOT
    root = tk.Tk()
    _FOLDER_DIALOG_ROOT = root
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="이미지 데이터셋 최상위 폴더 선택")
    # Do not destroy this Tk root here: on macOS the Matplotlib Tk backend
    # may share Tcl state with the folder dialog and crash if it is deleted.
    if not folder:
        raise SystemExit("폴더 선택이 취소되었습니다.")
    return Path(folder)


class FolderImageDataset:
    """Read images under class-named directories and return image, label, path."""

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        if not self.root_dir.is_dir():
            raise ValueError(f"데이터셋 폴더를 찾을 수 없습니다: {self.root_dir}")

        class_dirs = sorted(p for p in self.root_dir.iterdir() if p.is_dir())
        if not class_dirs:
            raise ValueError("선택한 폴더 바로 아래에 라벨 폴더가 없습니다.")

        self.classes = [p.name for p in class_dirs]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))

        if not self.samples:
            raise ValueError("라벨 폴더 안에서 지원되는 이미지 파일을 찾지 못했습니다.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path, label = self.samples[index]
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}") from exc

        original = image.copy()
        processed = self.transform(image) if self.transform else image_to_array(image)
        return {
            "image": processed,
            "label": label,
            "label_name": self.classes[label],
            "path": str(path),
            "original": original,
        }


def collate_samples(items: list[dict]) -> dict:
    """Stack image arrays while keeping original PIL images as a Python list."""
    return {
        "image": np.stack([item["image"] for item in items]),
        "label": np.asarray([item["label"] for item in items], dtype=np.int64),
        "label_name": [item["label_name"] for item in items],
        "path": [item["path"] for item in items],
        "original": [item["original"] for item in items],
    }


def image_to_array(image: Image.Image) -> np.ndarray:
    """Convert an RGB PIL image to a float32 HWC array in [0, 1]."""
    return np.asarray(image, dtype=np.float32) / 255.0


# ImageNet RGB channel statistics, commonly used to initialize pretrained CNNs.
CHANNEL_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
CHANNEL_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def make_preprocessing(image_size: int = 224, augment: bool = True):
    """Crop, optionally augment, convert to float, then ImageNet-normalize."""
    def preprocess(image: Image.Image) -> np.ndarray:
        # ImageOps.fit preserves aspect ratio and center-crops to the target size.
        image = ImageOps.fit(image, (image_size, image_size), method=Image.BILINEAR)

        if augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.5:
                image = image.rotate(random.uniform(-10.0, 10.0), resample=Image.BILINEAR)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))

        pixels = image_to_array(image)
        return (pixels - CHANNEL_MEAN) / CHANNEL_STD

    return preprocess


def for_display(normalized_image: np.ndarray) -> np.ndarray:
    """Undo channel normalization for plotting."""
    return (normalized_image * CHANNEL_STD + CHANNEL_MEAN).clip(0, 1)


def preview_first_batch(batch: dict, preview_seconds: float = 5.0) -> None:
    """Show the first batch, then automatically close the figure."""
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"):
        if candidate in installed_fonts:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False

    count = len(batch["label"])
    figure, axes = plt.subplots(count, 2, figsize=(10, max(4, count * 2.8)), squeeze=False)
    figure.suptitle("첫 번째 batch: 원본 / 전처리 이미지 (창을 닫으면 계속합니다)")

    for i in range(count):
        original = batch["original"][i]
        processed = for_display(batch["image"][i])
        label_name = batch["label_name"][i]
        sample_name = Path(batch["path"][i]).name

        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"원본 | label: {label_name} | {sample_name}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(processed)
        axes[i, 1].set_title(f"전처리 ({processed.shape[1]}×{processed.shape[0]}) | label: {label_name}")
        axes[i, 1].axis("off")

    figure.tight_layout()
    plt.show(block=False)
    plt.pause(preview_seconds)
    plt.close(figure)

    # The plot is closed now, so it is safe to release the hidden Tk root.
    global _FOLDER_DIALOG_ROOT
    if _FOLDER_DIALOG_ROOT is not None:
        _FOLDER_DIALOG_ROOT.destroy()
        _FOLDER_DIALOG_ROOT = None


def main() -> None:
    dataset_root = choose_dataset_folder()
    dataset = FolderImageDataset(
        dataset_root,
        transform=make_preprocessing(image_size=224, augment=True),
    )
    batch_size = 8
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    print(f"데이터셋: {dataset.root_dir}")
    print(f"이미지 수: {len(dataset):,}")
    print(f"라벨: {dataset.class_to_idx}")

    total_batches = len(batches)
    for batch_index, batch_indices in enumerate(batches, start=1):
        # 첫 batch 그래프가 남아 있으면 두 번째 batch를 읽기 전에 확실히 닫습니다.
        if batch_index == 2:
            plt.close("all")

        batch = collate_samples([dataset[i] for i in batch_indices])
        if batch_index == 1:
            print("첫 batch image array shape:", tuple(batch["image"].shape))
            preview_first_batch(batch)
        else:
            print(f"\nBatch {batch_index}/{total_batches}")
            for label_name, path in zip(batch["label_name"], batch["path"]):
                print(f"  label: {label_name}	 sample: {Path(path).name}")

        images = batch["image"]  # [B, 224, 224, 3], float32, ImageNet-normalized
        labels = batch["label"]  # NumPy int64 class indices
        # PyTorch 학습 시: torch.from_numpy(images).permute(0, 3, 1, 2)로 변환하세요.
        print(f"배치 처리 완료: {batch_index}/{total_batches} ({len(labels)}개 이미지)")

    print("데이터 전체 1회 순회(epoch)가 끝났습니다.")


if __name__ == "__main__":
    main()
