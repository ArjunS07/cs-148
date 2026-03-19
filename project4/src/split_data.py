"""
Create and save a reproducible stratified 85:15 train/val split.

Usage:
    cd /path/to/project4
    uv run python src/split_data.py
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import ProvidedDigitDataset

DATA_DIR = "data/dataset"
EMBEDDINGS_DIR = "embeddings"
VAL_FRACTION = 0.15
SEED = 42


def main():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    ds = ProvidedDigitDataset(DATA_DIR)
    print(f"Dataset: {len(ds)} images")

    # Group indices by label
    by_label: dict[int, list[int]] = {}
    for i, label in enumerate(ds.labels):
        by_label.setdefault(label, []).append(i)

    rng = random.Random(SEED)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in sorted(by_label):
        idxs = by_label[label][:]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * VAL_FRACTION))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_labels = [ds.labels[i] for i in train_idx]
    val_labels = [ds.labels[i] for i in val_idx]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    print("Train distribution:", Counter(train_labels))
    print("Val   distribution:", Counter(val_labels))

    # Verify stratification
    for label in sorted(by_label):
        n_total = len(by_label[label])
        n_val_c = sum(1 for lbl in val_labels if lbl == label)
        print(f"  Class {label}: {n_total} total, {n_val_c} val ({n_val_c/n_total:.1%})")

    torch.save(
        {"train_indices": train_idx, "val_indices": val_idx},
        os.path.join(EMBEDDINGS_DIR, "split_indices.pt"),
    )

    # Index → image path mapping for tracing misclassifications
    image_paths = [os.path.join(DATA_DIR, f) for f in ds.files]
    with open(os.path.join(EMBEDDINGS_DIR, "image_paths.json"), "w") as fh:
        json.dump(image_paths, fh, indent=2)

    print(f"\nSaved split_indices.pt and image_paths.json to {EMBEDDINGS_DIR}/")


if __name__ == "__main__":
    main()
