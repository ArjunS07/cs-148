#!/usr/bin/env python3
"""Generate plots from training_log.json.

Usage:
    python plot_metrics.py checkpoints/run1/training_log.json
    python plot_metrics.py checkpoints/run1/training_log.json --out-dir plots/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

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
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
)

NUM_CLASSES = 10
PALETTE = sns.color_palette("deep", NUM_CLASSES)


def load_history(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# -----------------------------------------------------------------------
# 1. Loss and accuracy curves
# -----------------------------------------------------------------------

def plot_loss_acc(history: list[dict], out_dir: str):
    epochs = [e["epoch"] for e in history]
    train_loss = [e["train"]["loss"] for e in history]
    val_loss = [e["val"]["loss"] for e in history]
    train_acc = [e["train"]["acc"] for e in history]
    val_acc = [e["val"]["acc"] for e in history]
    lr = [e["lr"] for e in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train", linewidth=1.5)
    ax.plot(epochs, val_loss, label="Val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, train_acc, label="Train", linewidth=1.5)
    ax.plot(epochs, val_acc, label="Val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.set_ylim(0, 1)

    ax = axes[2]
    ax.plot(epochs, lr, color=PALETTE[2], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_acc_lr.png"), dpi=150)
    plt.close(fig)
    print(f"  loss_acc_lr.png")


# -----------------------------------------------------------------------
# 2. Per-class loss history
# -----------------------------------------------------------------------

def plot_per_class_loss(history: list[dict], out_dir: str):
    epochs = [e["epoch"] for e in history]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for split_name, ax in zip(["train", "val"], axes):
        for c in range(NUM_CLASSES):
            key = str(c)
            losses = [e[split_name]["per_class_loss"].get(key, 0) for e in history]
            ax.plot(epochs, losses, label=f"{c}", color=PALETTE[c], linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{split_name.capitalize()} Loss by Class")
        ax.legend(title="Digit", ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_loss.png"), dpi=150)
    plt.close(fig)
    print(f"  per_class_loss.png")


# -----------------------------------------------------------------------
# 3. Per-class accuracy history
# -----------------------------------------------------------------------

def plot_per_class_acc(history: list[dict], out_dir: str):
    epochs = [e["epoch"] for e in history]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for split_name, ax in zip(["train", "val"], axes):
        for c in range(NUM_CLASSES):
            key = str(c)
            accs = [e[split_name]["per_class_acc"].get(key, 0) for e in history]
            ax.plot(epochs, accs, label=f"{c}", color=PALETTE[c], linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{split_name.capitalize()} Accuracy by Class")
        ax.legend(title="Digit", ncol=2, fontsize=8)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_acc.png"), dpi=150)
    plt.close(fig)
    print(f"  per_class_acc.png")


# -----------------------------------------------------------------------
# 4. Per-source (real vs synthetic) loss/acc history
# -----------------------------------------------------------------------

def plot_per_source(history: list[dict], out_dir: str):
    has_source = any("per_source" in e["train"] for e in history)
    if not has_source:
        print("  (skipping per_source plot --- no synthetic data)")
        return

    epochs = [e["epoch"] for e in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for src_name, color in [("real", PALETTE[0]), ("synthetic", PALETTE[1])]:
        losses = []
        accs = []
        valid_epochs = []
        for e in history:
            ps = e["train"].get("per_source", {})
            if src_name in ps:
                losses.append(ps[src_name]["loss"])
                accs.append(ps[src_name]["acc"])
                valid_epochs.append(e["epoch"])

        if valid_epochs:
            axes[0].plot(valid_epochs, losses, label=src_name.capitalize(),
                         color=color, linewidth=1.5)
            axes[1].plot(valid_epochs, accs, label=src_name.capitalize(),
                         color=color, linewidth=1.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train Loss by Source")
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Train Accuracy by Source")
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_source.png"), dpi=150)
    plt.close(fig)
    print(f"  per_source.png")


# -----------------------------------------------------------------------
# 5. Confusion matrix (latest epoch)
# -----------------------------------------------------------------------

def plot_confusion_matrix(history: list[dict], out_dir: str, split: str = "val"):
    """Plot confusion matrix from the last epoch using seaborn heatmap."""
    last = history[-1]
    cm_dict = last[split]["confusion_matrix"]

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true_c in range(NUM_CLASSES):
        for pred_c in range(NUM_CLASSES):
            cm[true_c, pred_c] = cm_dict.get(str(true_c), {}).get(str(pred_c), 0)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES),
                cbar_kws={"shrink": 0.8})
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"{split.capitalize()} Confusion Matrix (counts, epoch {last['epoch']})")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                vmin=0, vmax=1,
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES),
                cbar_kws={"shrink": 0.8})
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"{split.capitalize()} Confusion Matrix (row-normalized)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"confusion_matrix_{split}.png"), dpi=150)
    plt.close(fig)
    print(f"  confusion_matrix_{split}.png")


# -----------------------------------------------------------------------
# 6. Overfitting gap (train - val)
# -----------------------------------------------------------------------

def plot_overfit_gap(history: list[dict], out_dir: str):
    epochs = [e["epoch"] for e in history]
    loss_gap = [e["train"]["loss"] - e["val"]["loss"] for e in history]
    acc_gap = [e["train"]["acc"] - e["val"]["acc"] for e in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(epochs, loss_gap, color=PALETTE[3], linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Train Loss $-$ Val Loss")
    ax.set_title("Overfitting Gap (Loss)")

    ax = axes[1]
    ax.plot(epochs, acc_gap, color=PALETTE[3], linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Train Acc $-$ Val Acc")
    ax.set_title("Overfitting Gap (Accuracy)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overfit_gap.png"), dpi=150)
    plt.close(fig)
    print(f"  overfit_gap.png")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def plot_all(log_path: str, out_dir: str | None = None):
    if out_dir is None:
        out_dir = os.path.dirname(log_path)
    os.makedirs(out_dir, exist_ok=True)

    history = load_history(log_path)
    if not history:
        print("Empty training log, nothing to plot.")
        return

    print(f"Generating plots from {log_path} ({len(history)} epochs)...")
    plot_loss_acc(history, out_dir)
    plot_per_class_loss(history, out_dir)
    plot_per_class_acc(history, out_dir)
    plot_per_source(history, out_dir)
    plot_confusion_matrix(history, out_dir, split="val")
    plot_confusion_matrix(history, out_dir, split="train")
    plot_overfit_gap(history, out_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str, help="Path to training_log.json")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    plot_all(args.log_path, args.out_dir)


if __name__ == "__main__":
    main()
