#!/usr/bin/env python3
"""Save example images showing each augmentation applied to real training images."""

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

sns.set_theme(style="whitegrid", palette="deep")

from dataset import (
    ProvidedDigitDataset,
    SyntheticMNISTDataset,
    get_train_transform,
    get_val_transform,
    stratified_split,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 128
N_EXAMPLES = 8
OUT_DIR = "augmentation_examples"


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def load_sample_pil_images(data_dir: str, n: int = N_EXAMPLES) -> list[tuple[Image.Image, int]]:
    train_files, _ = stratified_split(data_dir)
    by_label: dict[int, list[str]] = {}
    for f in train_files:
        label = int(f.split("_")[-1].replace(".jpg", "").replace("label", ""))
        by_label.setdefault(label, []).append(f)

    random.seed(42)
    selected = []
    for digit in range(min(n, 10)):
        f = random.choice(by_label[digit])
        selected.append(f)
    while len(selected) < n:
        selected.append(random.choice(train_files))

    results = []
    for f in selected[:n]:
        path = os.path.join(data_dir, f)
        img = Image.open(path).convert("RGB")
        label = int(f.split("_")[-1].replace(".jpg", "").replace("label", ""))
        results.append((img, label))
    return results


def save_grid(tensors: list[torch.Tensor], path: str):
    stack = torch.stack(tensors)
    if stack.min() < -0.1:
        stack = denormalize(stack)
    grid = make_grid(stack, nrow=N_EXAMPLES, padding=4, pad_value=1.0)
    save_image(grid, path)


def apply_transform(pil_images: list[Image.Image], transform) -> list[torch.Tensor]:
    return [transform(img) for img in pil_images]


def main(out_dir=None, data_dir=None):
    if out_dir is None:
        out_dir = OUT_DIR
    if data_dir is None:
        data_dir = "../data/dataset"
    os.makedirs(out_dir, exist_ok=True)

    samples = load_sample_pil_images(data_dir, N_EXAMPLES)
    pil_images = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    print(f"Saving augmentation examples to {out_dir}/")
    print(f"Using {N_EXAMPLES} sample images, labels: {labels}")

    to_tensor = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

    # 01 — Original
    originals = apply_transform(pil_images, to_tensor)
    save_grid(originals, f"{out_dir}/01_original.png")
    print("  01_original.png — raw images resized to 128x 128")

    # 02 — RandAugment
    tensors = apply_transform(pil_images, T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
    ]))
    save_grid(tensors, f"{out_dir}/02_rand_augment.png")
    print("  02_rand_augment.png — RandAugment (num_ops=2, magnitude=9)")

    # 03 — ElasticTransform
    tensors = apply_transform(pil_images, T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ElasticTransform(alpha=50.0, sigma=5.0),
        T.ToTensor(),
    ]))
    save_grid(tensors, f"{out_dir}/03_elastic_transform.png")
    print("  03_elastic_transform.png — ElasticTransform (alpha=50, sigma=5)")

    # 04 — RandomResizedCrop
    tensors = apply_transform(pil_images, T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        T.ToTensor(),
    ]))
    save_grid(tensors, f"{out_dir}/04_random_resized_crop.png")
    print("  04_random_resized_crop.png — RandomResizedCrop (scale 0.7–1.0)")

    # 05 — GaussianBlur
    tensors = apply_transform(pil_images, T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToTensor(),
    ]))
    save_grid(tensors, f"{out_dir}/05_gaussian_blur.png")
    print("  05_gaussian_blur.png — GaussianBlur (sigma 0.1–2.0)")

    # 06 — Gaussian Noise
    tensors = apply_transform(pil_images, T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
    ]))
    save_grid(tensors, f"{out_dir}/06_gaussian_noise.png")
    print("  06_gaussian_noise.png — Additive Gaussian noise (σ=0.05)")

    # 07 — RandomErasing (CutOut)
    tensors = apply_transform(pil_images, T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.RandomErasing(p=1.0, scale=(0.10, 0.33)),
    ]))
    save_grid(tensors, f"{out_dir}/07_random_erasing.png")
    print("  07_random_erasing.png — RandomErasing / CutOut (scale 0.10–0.33)")

    # 08 — Full pipeline
    full_tf = get_train_transform(IMG_SIZE)
    tensors = apply_transform(pil_images, full_tf)
    save_grid(tensors, f"{out_dir}/08_full_pipeline.png")
    print("  08_full_pipeline.png — full training augmentation pipeline")

    # 09 — Synthetic MNIST
    syn_ds = SyntheticMNISTDataset(
        data_dir, transform=to_tensor, num_samples=N_EXAMPLES, img_size=IMG_SIZE, seed=42
    )
    syn_tensors, syn_labels = [], []
    for i in range(N_EXAMPLES):
        img, label, _ = syn_ds[i]
        syn_tensors.append(img)
        syn_labels.append(label)
    save_grid(syn_tensors, f"{out_dir}/09_synthetic_mnist.png")
    print(f"  09_synthetic_mnist.png — synthetic MNIST on real backgrounds (labels: {syn_labels})")

    print(f"\nDone! Open {out_dir}/ to inspect all examples.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    main(out_dir=args.out_dir, data_dir=args.data_dir)
