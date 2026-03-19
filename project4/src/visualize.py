"""
Generate all plots and figures for the writeup.

Reads saved results from checkpoints/ and produces publication-quality figures.

Usage:
    cd /path/to/project4
    uv run python src/visualize.py
"""

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CLIP_CKPT_DIR = "checkpoints/clip_downstream"
DINO_CKPT_DIR = "checkpoints/dino_downstream"
ZS_DIR = "checkpoints/clip_zero_shot"
CROSS_DIR = "checkpoints/cross_model"
FIG_DIR = "checkpoints/figures"


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves() -> None:
    """Plot train/val loss and accuracy for best CLIP and DINO configs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for col, (label, ckpt_dir) in enumerate([("CLIP Downstream", CLIP_CKPT_DIR),
                                              ("DINO Downstream", DINO_CKPT_DIR)]):
        curves_path = os.path.join(ckpt_dir, "training_curves.json")
        if not os.path.exists(curves_path):
            log.warning(f"Missing {curves_path}, skipping.")
            continue

        curves = load_json(curves_path)
        cfg = load_json(os.path.join(ckpt_dir, "best_config.json"))
        epochs = list(range(len(curves["train_loss"])))
        cfg_str = f"lr={cfg['lr']} drop={cfg['dropout']} mixup={cfg['use_mixup']}"
        if cfg.get("dino_input") and cfg["dino_input"] != "N/A":
            cfg_str += f" input={cfg['dino_input']}"

        # Loss
        ax = axes[0, col]
        ax.plot(epochs, curves["train_loss"], label="Train loss", linewidth=1.5)
        ax.plot(epochs, curves["val_loss"], label="Val loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{label} — Loss\n{cfg_str}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[1, col]
        ax.plot(epochs, curves["train_acc"], label="Train acc", linewidth=1.5)
        ax.plot(epochs, curves["val_acc"], label="Val acc", linewidth=1.5)
        best_val = max(curves["val_acc"])
        ax.axhline(best_val, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"Best val={best_val:.3f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{label} — Accuracy\n{cfg_str}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "training_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


# ---------------------------------------------------------------------------
# Hyperparameter sweep summary
# ---------------------------------------------------------------------------

def plot_sweep_sensitivity() -> None:
    """Bar charts showing val_acc sensitivity to each hyperparameter."""
    for model_name, ckpt_dir in [("CLIP", CLIP_CKPT_DIR), ("DINO", DINO_CKPT_DIR)]:
        csv_path = os.path.join(ckpt_dir, "sweep_results.csv")
        if not os.path.exists(csv_path):
            log.warning(f"Missing {csv_path}, skipping.")
            continue

        df = pd.read_csv(csv_path)

        hp_cols = ["lr", "dropout", "use_mixup"]
        if model_name == "DINO":
            hp_cols.append("dino_input")

        n_hp = len(hp_cols)
        fig, axes = plt.subplots(1, n_hp, figsize=(4 * n_hp, 4))
        if n_hp == 1:
            axes = [axes]

        for ax, hp in zip(axes, hp_cols):
            grouped = df.groupby(hp)["best_val_acc"].mean().reset_index()
            labels = [str(v) for v in grouped[hp]]
            vals = grouped["best_val_acc"].values
            bars = ax.bar(labels, vals, color="steelblue", edgecolor="white", width=0.6)
            ax.set_ylim(max(0, vals.min() - 0.05), min(1.0, vals.max() + 0.05))
            ax.set_xlabel(hp)
            ax.set_ylabel("Mean best val acc")
            ax.set_title(f"{model_name} — effect of {hp}", fontsize=9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"{model_name.lower()}_sweep_sensitivity.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved {out}")


def plot_sweep_table() -> None:
    """Top-10 configs per model as a table figure."""
    for model_name, ckpt_dir in [("CLIP", CLIP_CKPT_DIR), ("DINO", DINO_CKPT_DIR)]:
        csv_path = os.path.join(ckpt_dir, "sweep_results.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path).sort_values("best_val_acc", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        ax.set_title(f"{model_name} Sweep — Top 10 Configs", fontsize=10)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"{model_name.lower()}_sweep_table.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved {out}")


# ---------------------------------------------------------------------------
# Per-class accuracy comparison
# ---------------------------------------------------------------------------

def plot_per_class_accuracy() -> None:
    """Bar chart comparing per-class accuracy across all evaluated models."""
    model_results = {}

    for name, path in [
        ("CLIP zero-shot", os.path.join(ZS_DIR, "results.json")),
        ("CLIP downstream", os.path.join(CLIP_CKPT_DIR, "results.json")),
        ("DINO downstream", os.path.join(DINO_CKPT_DIR, "results.json")),
    ]:
        if os.path.exists(path):
            data = load_json(path)
            model_results[name] = {int(k): v for k, v in data["per_class_accuracy"].items()}

    if not model_results:
        log.warning("No results found for per-class plot.")
        return

    classes = list(range(10))
    n_models = len(model_results)
    x = np.arange(len(classes))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, accs) in enumerate(model_results.items()):
        vals = [accs.get(c, 0.0) for c in classes]
        ax.bar(x + i * width - (n_models - 1) * width / 2, vals, width,
               label=name, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Digit {c}" for c in classes], rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "per_class_accuracy_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out}")


# ---------------------------------------------------------------------------
# Cross-model comparison grids
# ---------------------------------------------------------------------------

def _load_and_plot_grid(json_path: str, out_path: str, title: str, n_cols: int = 5) -> None:
    if not os.path.exists(json_path):
        return
    items = load_json(json_path)[:10]
    if not items:
        return

    import math
    n_rows = math.ceil(len(items) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.2))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for ax in axes.flat:
        ax.axis("off")
    for i, item in enumerate(items):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        try:
            ax.imshow(Image.open(item["image_path"]))
        except Exception:
            pass
        lines = [f"True: {item.get('true_label', '?')}"]
        for key, label in [("clip_pred", "CLIP"), ("dino_pred", "DINO"),
                            ("zs_pred", "ZS"), ("clip_conf", None)]:
            if key in item:
                lines.append(f"{label or key}: {item[key]}")
        ax.set_title("\n".join(lines), fontsize=6)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}")


def plot_cross_model_grids() -> None:
    grids = [
        (os.path.join(CROSS_DIR, "clip_right_dino_wrong.json"),
         os.path.join(FIG_DIR, "clip_right_dino_wrong.png"),
         "CLIP Correct, DINO Wrong"),
        (os.path.join(CROSS_DIR, "dino_right_clip_wrong.json"),
         os.path.join(FIG_DIR, "dino_right_clip_wrong.png"),
         "DINO Correct, CLIP Wrong"),
        (os.path.join(CROSS_DIR, "zs_right_clip_downstream_wrong.json"),
         os.path.join(FIG_DIR, "zs_right_clip_downstream_wrong.png"),
         "Zero-Shot Correct, CLIP Downstream Wrong"),
        (os.path.join(CROSS_DIR, "clip_downstream_right_zs_wrong.json"),
         os.path.join(FIG_DIR, "clip_downstream_right_zs_wrong.png"),
         "CLIP Downstream Correct, Zero-Shot Wrong"),
    ]
    for json_path, out_path, title in grids:
        _load_and_plot_grid(json_path, out_path, title)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("logs/visualize.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    log.info("Generating training curves...")
    plot_training_curves()

    log.info("Generating sweep sensitivity plots...")
    plot_sweep_sensitivity()
    plot_sweep_table()

    log.info("Generating per-class accuracy comparison...")
    plot_per_class_accuracy()

    log.info("Generating cross-model comparison grids...")
    plot_cross_model_grids()

    log.info(f"All figures saved to {FIG_DIR}/")


log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
