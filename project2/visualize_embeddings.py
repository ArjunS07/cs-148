#!/usr/bin/env python3
"""Visualize dataset embeddings in 2D using PCA on model features.

Generates:
  1. original_classes.png — Original data colored by class
  2. sources_overlay.png — Original + augmented + synthetic colored by source
  3. per_class/ — Per-class scatterplots showing all three sources

Uses the penultimate layer (before final FC) of the trained model as the
feature extractor for more meaningful projections than raw pixels.

Usage:
    python visualize_embeddings.py checkpoints/run2/best_model.pt
    python visualize_embeddings.py checkpoints/run2/best_model.pt --model convnext --method tsne
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from model import build_model, ResNet18, ConvNeXtFemto
from dataset import (
    ProvidedDigitDataset,
    SyntheticMNISTDataset,
    get_train_transform,
    get_val_transform,
    stratified_split,
)

sns.set_theme(
    style="whitegrid",
    palette="deep",
    font="serif",
    rc={
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    },
)

NUM_CLASSES = 10
PALETTE = sns.color_palette("deep", NUM_CLASSES)
SOURCE_PALETTE = {"original": sns.color_palette("deep")[0],
                  "augmented": sns.color_palette("deep")[1],
                  "synthetic": sns.color_palette("deep")[2]}
SOURCE_MARKERS = {"original": "o", "augmented": "^", "synthetic": "s"}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """Wraps a model to return penultimate features instead of logits."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._hook_output = None

        # Register hook on the layer before the final FC
        if isinstance(model, ResNet18):
            # Hook after avgpool, before fc
            model.avgpool.register_forward_hook(self._hook)
        elif isinstance(model, ConvNeXtFemto):
            # Hook after adaptive avg pool in head (first element)
            model.head[0].register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._hook_output = output

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        self.model(images)
        features = self._hook_output
        return features.view(features.size(0), -1).cpu()


def extract_features(extractor: FeatureExtractor, loader: DataLoader,
                     device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from a dataloader."""
    all_features = []
    all_labels = []
    for images, labels, _sources in loader:
        feats = extractor.extract(images.to(device))
        all_features.append(feats.numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# 2D projection
# ---------------------------------------------------------------------------

def project_2d(features: np.ndarray, method: str = "pca",
               perplexity: int = 30) -> np.ndarray:
    """Project features to 2D."""
    if method == "pca":
        proj = PCA(n_components=2)
        return proj.fit_transform(features)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    init="pca", learning_rate="auto")
        return proj.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Plot 1: Original data colored by class
# ---------------------------------------------------------------------------

def plot_original_classes(coords: np.ndarray, labels: np.ndarray,
                          out_dir: str, method: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(NUM_CLASSES):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=[PALETTE[c]], label=str(c), s=15, alpha=0.6)
    ax.legend(title="Digit", markerscale=2)
    ax.set_title(f"Original Data by Class ({method.upper()})")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = os.path.join(out_dir, "original_classes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Plot 2: All sources overlaid, colored by source
# ---------------------------------------------------------------------------

def plot_sources_overlay(all_coords: dict[str, np.ndarray],
                         out_dir: str, method: str):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot in reverse order so original is on top
    for source in ["synthetic", "augmented", "original"]:
        if source not in all_coords:
            continue
        coords = all_coords[source]
        ax.scatter(coords[:, 0], coords[:, 1],
                   color=[SOURCE_PALETTE[source]], label=f"{source} (n={len(coords)})",
                   s=12, alpha=0.5, marker=SOURCE_MARKERS[source])

    ax.legend(markerscale=2)
    ax.set_title(f"Data Sources Overlay ({method.upper()})")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = os.path.join(out_dir, "sources_overlay.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Plot 3: Per-class scatterplots
# ---------------------------------------------------------------------------

def plot_per_class(all_coords: dict[str, np.ndarray],
                   all_labels: dict[str, np.ndarray],
                   out_dir: str, method: str):
    class_dir = os.path.join(out_dir, "per_class")
    os.makedirs(class_dir, exist_ok=True)

    # Compute global axis limits
    all_pts = np.concatenate(list(all_coords.values()))
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05

    for c in range(NUM_CLASSES):
        fig, ax = plt.subplots(figsize=(8, 6))

        for source in ["synthetic", "augmented", "original"]:
            if source not in all_coords:
                continue
            mask = all_labels[source] == c
            coords = all_coords[source][mask]
            if len(coords) == 0:
                continue
            ax.scatter(coords[:, 0], coords[:, 1],
                       color=[SOURCE_PALETTE[source]], label=f"{source} (n={len(coords)})",
                       s=15, alpha=0.5, marker=SOURCE_MARKERS[source])

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.legend(markerscale=2)
        ax.set_title(f"Class {c} — All Sources ({method.upper()})")
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        path = os.path.join(class_dir, f"class_{c}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)

    print(f"  {class_dir}/class_0.png ... class_9.png")

    # Also make a combined 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    for c in range(NUM_CLASSES):
        ax = axes[c // 5][c % 5]
        for source in ["synthetic", "augmented", "original"]:
            if source not in all_coords:
                continue
            mask = all_labels[source] == c
            coords = all_coords[source][mask]
            if len(coords) == 0:
                continue
            ax.scatter(coords[:, 0], coords[:, 1],
                       color=[SOURCE_PALETTE[source]], s=8, alpha=0.4,
                       marker=SOURCE_MARKERS[source], label=f"{source}")
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_title(f"Class {c}", fontsize=11)
        ax.grid(True, alpha=0.2)
        if c == 0:
            ax.legend(fontsize=7, markerscale=1.5)

    fig.suptitle(f"Per-Class Embeddings ({method.upper()})", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "per_class_grid.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings in 2D")
    parser.add_argument("checkpoint", type=str, help="Path to best_model.pt")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "convnext"])
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--method", type=str, default="pca",
                        choices=["pca", "tsne"])
    parser.add_argument("--synthetic-n", type=int, default=3000,
                        help="Number of synthetic samples to include")
    parser.add_argument("--max-augmented", type=int, default=3000,
                        help="Number of augmented samples to include")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.checkpoint), "embeddings")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(args.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded {args.model} from {args.checkpoint}")

    extractor = FeatureExtractor(model)

    # Prepare datasets
    train_files, _ = stratified_split(args.data_dir)
    val_tf = get_val_transform(args.img_size)
    train_tf = get_train_transform(args.img_size)

    # 1. Original (clean, no augmentation)
    print("Extracting features: original data...")
    orig_ds = ProvidedDigitDataset(args.data_dir, transform=val_tf, file_list=train_files)
    orig_loader = DataLoader(orig_ds, batch_size=64, shuffle=False, num_workers=0)
    orig_feats, orig_labels = extract_features(extractor, orig_loader, device)
    print(f"  {len(orig_feats)} samples")

    # 2. Augmented (same real images with training augmentation)
    print("Extracting features: augmented data...")
    n_aug = min(args.max_augmented, len(train_files))
    aug_ds = ProvidedDigitDataset(args.data_dir, transform=train_tf,
                              file_list=train_files[:n_aug])
    aug_loader = DataLoader(aug_ds, batch_size=64, shuffle=False, num_workers=0)
    aug_feats, aug_labels = extract_features(extractor, aug_loader, device)
    print(f"  {len(aug_feats)} samples")

    # 3. Synthetic
    print("Extracting features: synthetic data...")
    syn_ds = SyntheticMNISTDataset(args.data_dir, transform=val_tf,
                                   num_samples=args.synthetic_n,
                                   img_size=args.img_size, seed=42)
    syn_loader = DataLoader(syn_ds, batch_size=64, shuffle=False, num_workers=0)
    syn_feats, syn_labels = extract_features(extractor, syn_loader, device)
    print(f"  {len(syn_feats)} samples")

    # Project all features together for consistent axes
    print(f"Projecting to 2D ({args.method})...")
    combined_feats = np.concatenate([orig_feats, aug_feats, syn_feats])
    combined_2d = project_2d(combined_feats, method=args.method)

    n_orig = len(orig_feats)
    n_aug = len(aug_feats)
    orig_2d = combined_2d[:n_orig]
    aug_2d = combined_2d[n_orig:n_orig + n_aug]
    syn_2d = combined_2d[n_orig + n_aug:]

    # Generate plots
    print(f"Saving to {args.out_dir}/")

    plot_original_classes(orig_2d, orig_labels, args.out_dir, args.method)

    all_coords = {"original": orig_2d, "augmented": aug_2d, "synthetic": syn_2d}
    plot_sources_overlay(all_coords, args.out_dir, args.method)

    all_labels = {"original": orig_labels, "augmented": aug_labels,
                  "synthetic": syn_labels}
    plot_per_class(all_coords, all_labels, args.out_dir, args.method)

    print("Done.")


if __name__ == "__main__":
    main()
