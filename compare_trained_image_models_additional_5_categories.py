"""Checkpoint discovery for the additional-five-category model set.

Model architecture and inference helpers are shared with the original
comparison script, while checkpoint discovery is isolated here so the
original evaluator keeps its existing behavior.
"""

from __future__ import print_function

import os
import re

from compare_trained_image_models import *  # noqa: F401,F403
import compare_trained_image_models as _base


def category_confusion_counts(records, class_names):
    """Count additional-model predictions by actual folder-label category."""
    return _base.category_confusion_counts(records, class_names)


def find_checkpoints(checkpoint_dir):
    """Find checkpoints without glob, since [] in this folder name are special to glob."""
    try:
        filenames = os.listdir(checkpoint_dir)
    except OSError as exc:
        raise FileNotFoundError(
            "Cannot read checkpoint folder: {} ({})".format(checkpoint_dir, exc)
        )

    found = {}
    for display_name, model_name, pattern in _base.MODEL_SPECS:
        validation_candidates = []
        training_candidates = []
        prefix = "best_{}_".format(model_name)
        for filename in filenames:
            normalized = filename.strip()
            if not normalized.startswith(prefix) or not normalized.lower().endswith(".pth"):
                continue
            path = os.path.join(checkpoint_dir, filename)
            if re.search(r"_val_acc_[0-9.]+_epoch_\d+", normalized):
                validation_candidates.append(path)
            elif re.search(r"_train_acc_[0-9.]+_epoch_\d+", normalized):
                training_candidates.append(path)

        candidates = validation_candidates or training_candidates
        if not validation_candidates and training_candidates:
            print(
                "No validation checkpoint for {}; using training-accuracy checkpoint(s).".format(
                    display_name
                )
            )
        if not candidates:
            available = sorted(
                filename for filename in filenames
                if filename.lower().endswith(".pth")
            )
            raise FileNotFoundError(
                "{} checkpoint not found in {!r}. Expected pattern: {}. "
                ".pth files found: {}".format(
                    display_name, checkpoint_dir, pattern, available
                )
            )
        found[model_name] = max(candidates, key=_base.checkpoint_accuracy_epoch)
    return found
