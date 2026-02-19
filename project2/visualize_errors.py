#!/usr/bin/env python3
"""Visualize misclassified examples from the validation set.

Generates grids of the most confidently wrong predictions, grouped by
(true_class, predicted_class) pairs.

Usage:
    python visualize_errors.py checkpoints/run2/best_model.pt
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from model import build_model
from dataset import ProvidedDigitDataset, get_val_transform, stratified_split

sns.set_theme(
    style="whitegrid",
    palette="deep",
    font="serif",
    rc={
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    },
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
NUM_CLASSES = 10


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def collect_predictions(model, loader, device):
    """Run model on all val samples, return per-sample info."""
    model.eval()
    results = []

    with torch.no_grad():
        for images, labels, _sources in loader:
            logits = model(images.to(device)).cpu()
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            confidences = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

            for i in range(images.size(0)):
                results.append({
                    "image": images[i],  # still normalized
                    "true": labels[i].item(),
                    "pred": preds[i].item(),
                    "confidence": confidences[i].item(),
                    "probs": probs[i],
                })

    return results


def plot_misclassified_grid(results, out_dir, max_per_image: int = 8):
    """Save a grid of the most confidently wrong predictions."""
    errors = [r for r in results if r["true"] != r["pred"]]
    # Sort by confidence (most confidently wrong first)
    errors.sort(key=lambda r: r["confidence"], reverse=True)

    n = min(len(errors), max_per_image * 4)  # up to 4 rows
    if n == 0:
        print("  No misclassified examples!")
        return

    cols = min(n, max_per_image)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        e = errors[i]
        img = denormalize(e["image"]).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(
            f"true={e['true']} pred={e['pred']}\nconf={e['confidence']:.2f}",
            fontsize=9,
            color="red",
        )

    fig.suptitle(f"Most Confidently Wrong ({len(errors)} total errors)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "misclassified_top.png"), dpi=150)
    plt.close(fig)
    print(f"  misclassified_top.png ({n} examples)")


def plot_errors_by_class(results, out_dir, max_per_class: int = 6):
    """For each true class, show its misclassified examples."""
    errors_by_true = {c: [] for c in range(NUM_CLASSES)}
    for r in results:
        if r["true"] != r["pred"]:
            errors_by_true[r["true"]].append(r)

    # Sort each class by confidence
    for c in range(NUM_CLASSES):
        errors_by_true[c].sort(key=lambda r: r["confidence"], reverse=True)

    fig, axes = plt.subplots(NUM_CLASSES, max_per_class,
                             figsize=(2.5 * max_per_class, 2.5 * NUM_CLASSES))

    for c in range(NUM_CLASSES):
        errs = errors_by_true[c][:max_per_class]
        for j in range(max_per_class):
            ax = axes[c][j]
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(f"True={c}", fontsize=10, rotation=0, labelpad=40,
                              va="center")
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if j < len(errs):
                e = errs[j]
                img = denormalize(e["image"]).permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(f"→{e['pred']} ({e['confidence']:.2f})", fontsize=8,
                             color="red")

    fig.suptitle("Misclassified Examples by True Class", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "misclassified_by_class.png"), dpi=150)
    plt.close(fig)
    print(f"  misclassified_by_class.png")


def plot_error_pair_grid(results, out_dir, max_examples: int = 4):
    """For each (true, pred) confusion pair with errors, show examples."""
    pairs: dict[tuple[int, int], list] = {}
    for r in results:
        if r["true"] != r["pred"]:
            key = (r["true"], r["pred"])
            pairs.setdefault(key, []).append(r)

    # Sort pairs by frequency
    sorted_pairs = sorted(pairs.items(), key=lambda kv: len(kv[1]), reverse=True)
    top_pairs = sorted_pairs[:15]  # show top 15 confusion pairs

    if not top_pairs:
        return

    fig, axes = plt.subplots(len(top_pairs), max_examples,
                             figsize=(2.5 * max_examples, 2.2 * len(top_pairs)))
    if len(top_pairs) == 1:
        axes = [axes]

    for i, ((true_c, pred_c), errs) in enumerate(top_pairs):
        errs.sort(key=lambda r: r["confidence"], reverse=True)
        for j in range(max_examples):
            ax = axes[i][j] if max_examples > 1 else axes[i]
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(f"{true_c}→{pred_c}\n(n={len(errs)})",
                              fontsize=9, rotation=0, labelpad=40, va="center")
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if j < len(errs):
                e = errs[j]
                img = denormalize(e["image"]).permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(f"conf={e['confidence']:.2f}", fontsize=8)

    fig.suptitle("Top Confusion Pairs (true→pred)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_pairs.png"), dpi=150)
    plt.close(fig)
    print(f"  confusion_pairs.png ({len(top_pairs)} pairs)")


def save_misclassified_images(results, out_dir):
    """Save each misclassified example as an individual image, organized by true class."""
    errors = [r for r in results if r["true"] != r["pred"]]
    if not errors:
        print("  No misclassified examples to save.")
        return

    base = os.path.join(out_dir, "misclassified")
    for c in range(NUM_CLASSES):
        os.makedirs(os.path.join(base, str(c)), exist_ok=True)

    for i, e in enumerate(errors):
        img = denormalize(e["image"]).permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(figsize=(3, 3.5))
        ax.imshow(img)
        ax.set_title(
            f"True: {e['true']}  Pred: {e['pred']}  Conf: {e['confidence']:.2f}",
            fontsize=10,
        )
        ax.axis("off")
        fig.tight_layout()
        fname = f"pred{e['pred']}_conf{e['confidence']:.2f}_{i:04d}.png"
        fig.savefig(os.path.join(base, str(e["true"]), fname), dpi=100)
        plt.close(fig)

    print(f"  misclassified/ — {len(errors)} images in {NUM_CLASSES} class folders")


def main():
    parser = argparse.ArgumentParser(description="Visualize misclassified examples")
    parser.add_argument("checkpoint", type=str, help="Path to best_model.pt")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18"])
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint)")
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.checkpoint)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(args.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded {args.model} from {args.checkpoint} "
          f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    # Load val data
    _, val_files = stratified_split(args.data_dir)
    val_ds = ProvidedDigitDataset(args.data_dir, transform=get_val_transform(args.img_size),
                              file_list=val_files)
    from torch.utils.data import DataLoader
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # Collect predictions
    print("Running inference on val set...")
    results = collect_predictions(model, loader, device)
    n_errors = sum(1 for r in results if r["true"] != r["pred"])
    print(f"  {len(results)} samples, {n_errors} errors "
          f"({n_errors/len(results)*100:.1f}%)")

    # Generate plots
    print(f"Saving to {args.out_dir}/")
    plot_misclassified_grid(results, args.out_dir)
    plot_errors_by_class(results, args.out_dir)
    plot_error_pair_grid(results, args.out_dir)

    val_acc = ckpt.get("val_acc", 0)
    if val_acc > 0.92:
        save_misclassified_images(results, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
