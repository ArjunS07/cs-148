#!/usr/bin/env python3
"""Save example images showing each augmentation applied to real training images.

Creates augmentation_examples/ with numbered side-by-side comparisons.
"""

import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import make_grid, save_image

sns.set_theme(
    style="whitegrid",
    palette="deep",
    font="serif",
    rc={
        "text.usetex": True,
        "font.family": "serif",
    },
)

from dataset import (
    ProvidedDigitDataset,
    SyntheticMNISTDataset,
    get_train_transform,
    get_val_transform,
    mixup_batch,
    stratified_split,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_SIZE = 128
N_EXAMPLES = 8  # images per row
OUT_DIR = "augmentation_examples"


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def load_sample_pil_images(data_dir: str, n: int = N_EXAMPLES) -> list[tuple[Image.Image, int]]:
    """Load n random PIL images with labels, one per class then random fill."""
    train_files, _ = stratified_split(data_dir)

    # Group by label
    by_label: dict[int, list[str]] = {}
    for f in train_files:
        label = int(f.split("_")[-1].replace(".jpg", "").replace("label", ""))
        by_label.setdefault(label, []).append(f)

    # Pick one per class, then fill remaining slots randomly
    random.seed(42)
    selected = []
    for digit in range(min(n, 10)):
        f = random.choice(by_label[digit])
        selected.append(f)
    while len(selected) < n:
        f = random.choice(train_files)
        selected.append(f)

    results = []
    for f in selected[:n]:
        path = os.path.join(data_dir, f)
        img = Image.open(path).convert("RGB")
        label = int(f.split("_")[-1].replace(".jpg", "").replace("label", ""))
        results.append((img, label))
    return results


def save_grid(tensors: list[torch.Tensor], path: str, labels: list[int] | None = None):
    """Save a grid of tensors as a PNG. Denormalizes if needed."""
    # Detect if normalized (values can be negative)
    stack = torch.stack(tensors)
    if stack.min() < -0.1:
        stack = denormalize(stack)
    grid = make_grid(stack, nrow=N_EXAMPLES, padding=4, pad_value=1.0)
    save_image(grid, path)


def apply_single_augmentation(pil_images: list[Image.Image], transform) -> list[torch.Tensor]:
    """Apply a transform to each PIL image and return tensors."""
    results = []
    for img in pil_images:
        out = transform(img)
        results.append(out)
    return results


def main(out_dir=None, data_dir=None):
    if out_dir is None:
        out_dir = OUT_DIR
    if data_dir is None:
        data_dir = "data/dataset"
    os.makedirs(out_dir, exist_ok=True)

    samples = load_sample_pil_images(data_dir, N_EXAMPLES)
    pil_images = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    print(f"Saving augmentation examples to {out_dir}/")
    print(f"Using {N_EXAMPLES} sample images, labels: {labels}")

    to_tensor = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

    # --- Original (just resized, no augmentation) ---
    originals = apply_single_augmentation(pil_images, to_tensor)
    save_grid(originals, f"{out_dir}/01_original.png", labels)
    print("  01_original.png — raw images resized to 128x128")

    # --- Individual augmentations ---
    augmentations = {
        "02_color_jitter": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
        ]),
        "03_random_resized_crop": T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            T.ToTensor(),
        ]),
        "04_rotation": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomRotation(20),
            T.ToTensor(),
        ]),
        "05_perspective": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomPerspective(distortion_scale=0.2, p=1.0),
            T.ToTensor(),
        ]),
        "06_affine": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
        ]),
        "07_gaussian_blur": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToTensor(),
        ]),
        "08_gaussian_noise": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
        ]),
        "09_random_erasing": T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.RandomErasing(p=1.0, scale=(0.05, 0.2)),
        ]),
    }

    for name, transform in augmentations.items():
        tensors = apply_single_augmentation(pil_images, transform)
        save_grid(tensors, f"{out_dir}/{name}.png", labels)
        desc = name.split("_", 1)[1].replace("_", " ")
        print(f"  {name}.png — {desc}")

    # --- Full pipeline (what training actually sees) ---
    full_tf = get_train_transform(IMG_SIZE)
    tensors = apply_single_augmentation(pil_images, full_tf)
    save_grid(tensors, f"{out_dir}/10_full_pipeline.png", labels)
    print("  10_full_pipeline.png — full training augmentation")

    # --- MixUp ---
    batch = torch.stack(originals)
    label_tensor = torch.tensor(labels)
    mixed, soft_labels = mixup_batch(batch, label_tensor, alpha=1.0)
    save_grid(list(mixed), f"{out_dir}/11_mixup.png")
    print("  11_mixup.png — MixUp (alpha-blended pairs)")

    # --- Synthetic MNIST ---
    syn_ds = SyntheticMNISTDataset(
        data_dir, transform=to_tensor, num_samples=N_EXAMPLES, img_size=IMG_SIZE, seed=42
    )
    syn_tensors = []
    syn_labels = []
    for i in range(N_EXAMPLES):
        img, label, _source = syn_ds[i]
        syn_tensors.append(img)
        syn_labels.append(label)
    save_grid(syn_tensors, f"{out_dir}/12_synthetic_mnist.png", syn_labels)
    print(f"  12_synthetic_mnist.png — synthetic MNIST on real backgrounds (labels: {syn_labels})")

    print(f"\nDone! Open {out_dir}/ to inspect all examples.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    main(out_dir=args.out_dir, data_dir=args.data_dir)