"""Compare resized RGB pixels with handcrafted image descriptors.

Dependencies:
    pip install numpy matplotlib pillow

The extracted descriptors are classic, human-designed features (not CNN embeddings):
RGB color histograms, cell-based gradient orientation histograms (HOG-style), and
simple brightness/contrast/edge statistics. Results are saved beside this script.
"""

from __future__ import annotations

from pathlib import Path
import random
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
PIXEL_SIZE = 32
HOG_SIZE = 64
CELL_SIZE = 8
ORIENTATION_BINS = 9
COLOR_BINS = 16
BATCH_SIZE = 8
EPOCHS = 1


def choose_dataset_folder() -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="이미지 데이터셋 최상위 폴더 선택")
    root.destroy()
    if not folder:
        raise SystemExit("폴더 선택이 취소되었습니다.")
    return Path(folder)


def discover_samples(root: Path):
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError("선택한 폴더 바로 아래에서 라벨 폴더를 찾지 못했습니다.")
    class_names = [path.name for path in class_dirs]
    samples = []
    for label, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    if not samples:
        raise ValueError("라벨 폴더에서 이미지 파일을 찾지 못했습니다.")
    return class_names, samples


def hog_like_features(image: Image.Image):
    """Compute color-aware HOG: use the strongest RGB-channel gradient per pixel."""
    rgb = np.asarray(
        image.convert("RGB").resize((HOG_SIZE, HOG_SIZE), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    gradients = []
    for channel in range(3):
        gy, gx = np.gradient(rgb[:, :, channel])
        magnitude = np.sqrt(gx * gx + gy * gy)
        angle = (np.arctan2(gy, gx) * (180.0 / np.pi)) % 180.0
        gradients.append((magnitude, angle))
    channel_strength = np.stack([item[0] for item in gradients], axis=2)
    strongest_channel = np.argmax(channel_strength, axis=2)
    magnitude = np.max(channel_strength, axis=2)
    angle = np.choose(strongest_channel, [item[1] for item in gradients])

    cells_per_axis = HOG_SIZE // CELL_SIZE
    cell_histograms = np.zeros((cells_per_axis, cells_per_axis, ORIENTATION_BINS), dtype=np.float32)
    bin_width = 180.0 / ORIENTATION_BINS
    bin_indices = np.floor(angle / bin_width).astype(np.int32) % ORIENTATION_BINS

    for row in range(cells_per_axis):
        for col in range(cells_per_axis):
            y0, y1 = row * CELL_SIZE, (row + 1) * CELL_SIZE
            x0, x1 = col * CELL_SIZE, (col + 1) * CELL_SIZE
            bins = bin_indices[y0:y1, x0:x1].ravel()
            weights = magnitude[y0:y1, x0:x1].ravel()
            cell_histograms[row, col] = np.bincount(
                bins, weights=weights, minlength=ORIENTATION_BINS
            )[:ORIENTATION_BINS]

    # Normalize each cell so descriptors are less sensitive to overall brightness.
    norms = np.linalg.norm(cell_histograms, axis=2, keepdims=True) + 1e-6
    cell_histograms = cell_histograms / norms
    return cell_histograms.ravel(), rgb, cell_histograms


def extract_features(image: Image.Image):
    """Return baseline pixels, engineered feature vector, and display data."""
    rgb = image.convert("RGB")

    # Baseline: resized RGB pixel values flattened into one vector.
    pixel_image = ImageOps.fit(rgb, (PIXEL_SIZE, PIXEL_SIZE), method=Image.BILINEAR)
    pixel_vector = np.asarray(pixel_image, dtype=np.float32).ravel() / 255.0

    # Color distribution: per-channel histograms, each normalized to sum to one.
    rgb_array = np.asarray(rgb, dtype=np.uint8)
    color_parts = []
    for channel in range(3):
        hist, _ = np.histogram(rgb_array[:, :, channel], bins=COLOR_BINS, range=(0, 256))
        hist = hist.astype(np.float32)
        color_parts.append(hist / max(float(hist.sum()), 1.0))
    color_features = np.concatenate(color_parts)

    hog_vector, processed_rgb, cell_histograms = hog_like_features(rgb)

    gray_image = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
    gray_pixels = gray_image.ravel()
    gx = np.gradient(gray_image, axis=1)
    gy = np.gradient(gray_image, axis=0)
    gradient_magnitude = np.sqrt(gx * gx + gy * gy)
    stats = np.asarray([
        gray_pixels.mean(),
        gray_pixels.std(),
        float(np.mean(gradient_magnitude > 0.1)),  # edge/texture density estimate
        gradient_magnitude.mean(),
    ], dtype=np.float32)

    handcrafted_vector = np.concatenate([color_features, hog_vector, stats]).astype(np.float32)
    display = {
        "pixel_image": pixel_image,
        "processed_rgb": processed_rgb,
        "cell_histograms": cell_histograms,
        "color_features": color_features.reshape(3, COLOR_BINS),
        "stats": stats,
        "model_features": handcrafted_vector,
    }
    return pixel_vector, handcrafted_vector, display


def show_example(image: Image.Image, label: str, filename: str, display: dict, seconds: float = 5.0) -> None:
    """Show English-labeled source, processed image, and model-input features."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("First-batch sample | Label: {} | {}".format(label, filename))

    axes[0, 0].imshow(image.convert("RGB"))
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(display["processed_rgb"].clip(0, 1))
    axes[0, 1].set_title("Final color image used for feature extraction (64×64 RGB)")
    axes[0, 1].axis("off")

    bin_centers = np.arange(COLOR_BINS) * (256.0 / COLOR_BINS)
    for channel, color in enumerate(("red", "green", "blue")):
        axes[0, 2].plot(bin_centers, display["color_features"][channel], color=color, label=color)
    axes[0, 2].set_title(
        "RGB color histograms\n"
        "X: dark → bright (0 → 255); Y: fraction of pixels\n"
        "A peak means many pixels have that channel intensity",
        fontsize=10,
    )
    axes[0, 2].set_xlabel("Channel intensity")
    axes[0, 2].set_ylabel("Pixel fraction")
    axes[0, 2].legend()

    hog_map = display["cell_histograms"].max(axis=2)
    axes[1, 0].imshow(hog_map, cmap="magma")
    axes[1, 0].set_title(
        "Color-aware HOG gradient strength (8×8 cells)\n"
        "Brighter cells = stronger local edge / texture patterns\n"
        "Uses the strongest R/G/B edge; direction is in the feature vector",
        fontsize=10,
    )
    axes[1, 0].axis("off")

    axes[1, 1].plot(display["model_features"], linewidth=0.7)
    axes[1, 1].set_title("Actual model input: 628-D feature vector")
    axes[1, 1].set_xlabel("Feature index")
    axes[1, 1].set_ylabel("Normalized feature value")
    axes[1, 1].axvline(3 * COLOR_BINS - 0.5, color="red", linestyle="--", linewidth=1)
    axes[1, 1].axvline(3 * COLOR_BINS + (HOG_SIZE // CELL_SIZE) ** 2 * ORIENTATION_BINS - 0.5,
                       color="red", linestyle="--", linewidth=1)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02, 0.95,
        "Feature vector parts\n\n"
        "• RGB histograms: 48 values\n"
        "• HOG-style gradients: 576 values\n"
        "• Brightness / contrast / edges: 4 values\n\n"
        "The model receives this vector, not the raw image.",
        va="top", ha="left", fontsize=11,
    )
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(seconds)
    plt.close(fig)


def main() -> None:
    dataset_root = choose_dataset_folder().expanduser().resolve()
    class_names, samples = discover_samples(dataset_root)
    pixel_vectors = []
    feature_vectors = []
    labels = []
    paths = []
    example = None
    total_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE

    print("데이터셋: {}".format(dataset_root))
    print("이미지 수: {} | batch size: {} | epochs: {}".format(
        len(samples), BATCH_SIZE, EPOCHS
    ))
    print("라벨 인덱스: {}".format({name: i for i, name in enumerate(class_names)}))

    # Feature extraction is part of the input pipeline and runs as each batch is loaded.
    for epoch in range(1, EPOCHS + 1):
        indices = list(range(len(samples)))
        random.shuffle(indices)
        epoch_pixel_vectors = []
        epoch_feature_vectors = []
        epoch_labels = []
        epoch_paths = []

        for batch_number, start in enumerate(range(0, len(indices), BATCH_SIZE), start=1):
            # Ensure every graph from batch 1 is closed before batch 2 starts.
            if batch_number == 2:
                plt.close("all")
            batch_indices = indices[start:start + BATCH_SIZE]
            batch_pixels = []
            batch_features = []
            batch_labels = []
            batch_paths = []
            batch_displays = []

            for sample_index in batch_indices:
                path, label = samples[sample_index]
                try:
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                        pixel_vector, feature_vector, display = extract_features(image)
                        batch_pixels.append(pixel_vector)
                        batch_features.append(feature_vector)
                        batch_labels.append(label)
                        batch_paths.append(str(path))
                        batch_displays.append((image.copy(), class_names[label], path.name, display))
                except (OSError, ValueError) as exc:
                    print("이미지 건너뜀: {} ({})".format(path, exc))

            if not batch_features:
                continue

            # Keep standard per-batch arrays, ready to pass to a model/training loop.
            pixel_batch = np.stack(batch_pixels)
            feature_batch = np.stack(batch_features)
            label_batch = np.asarray(batch_labels, dtype=np.int64)
            epoch_pixel_vectors.extend(pixel_batch)
            epoch_feature_vectors.extend(feature_batch)
            epoch_labels.extend(label_batch)
            epoch_paths.extend(batch_paths)

            print("Epoch {}/{} | Batch {}/{} | {}개 샘플 특성 추출 완료".format(
                epoch, EPOCHS, batch_number, total_batches, len(label_batch)
            ))
            if batch_number == 1:
                print("첫 batch feature shape: {}".format(feature_batch.shape))
                for sample_number, example in enumerate(batch_displays, start=1):
                    print("첫 batch 그래프 {}/{}: label={} | sample={}".format(
                        sample_number, len(batch_displays), example[1], example[2]
                    ))
                    show_example(*example)  # Each sample graph auto-closes before the next.
            else:
                for label, path in zip(batch_labels, batch_paths):
                    print("  label: {}\t sample: {}".format(class_names[label], Path(path).name))

        pixel_vectors.extend(epoch_pixel_vectors)
        feature_vectors.extend(epoch_feature_vectors)
        labels.extend(epoch_labels)
        paths.extend(epoch_paths)
        print("Epoch {}/{} 완료: {}개 이미지 처리".format(epoch, EPOCHS, len(epoch_labels)))

    if not feature_vectors:
        raise RuntimeError("특성을 추출한 이미지가 없습니다.")

    pixel_matrix = np.stack(pixel_vectors)
    feature_matrix = np.stack(feature_vectors)
    output_path = Path(__file__).resolve().with_name("image_feature_comparison.npz")
    np.savez_compressed(
        output_path,
        pixel_vectors=pixel_matrix,
        feature_vectors=feature_matrix,
        labels=np.asarray(labels, dtype=np.int64),
        class_names=np.asarray(class_names),
        paths=np.asarray(paths),
    )

    print("\n기본 픽셀 벡터 크기: {} ({}×{}×3)".format(pixel_matrix.shape[1], PIXEL_SIZE, PIXEL_SIZE))
    print("추출 특성 벡터 크기: {} (색상 {} + HOG {} + 통계 4)".format(
        feature_matrix.shape[1], 3 * COLOR_BINS, (HOG_SIZE // CELL_SIZE) ** 2 * ORIENTATION_BINS
    ))
    print("저장 위치: {}".format(output_path))


if __name__ == "__main__":
    main()
