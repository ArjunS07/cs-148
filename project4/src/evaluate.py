"""
Evaluate best downstream CLIP and DINO classifiers and generate all artifacts.

Produces per-model confusion matrices, misclassified lists, cross-model comparisons,
and zero-shot vs downstream comparisons.

Usage:
    cd /path/to/project4
    uv run python src/evaluate.py
"""

import json
import logging
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

EMBEDDINGS_DIR = "embeddings"
CLIP_CKPT_DIR = "checkpoints/clip_downstream"
DINO_CKPT_DIR = "checkpoints/dino_downstream"
ZS_DIR = "checkpoints/clip_zero_shot"
CROSS_DIR = "checkpoints/cross_model"

H1_FOR_DIM = {512: 256, 768: 384, 1536: 512}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Rebuild MLP (must match train_downstream.py)
# ---------------------------------------------------------------------------

def build_mlp(in_dim: int, dropout: float) -> nn.Sequential:
    h1 = H1_FOR_DIM[in_dim]
    return nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.BatchNorm1d(h1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h1, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 10),
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    embed_val: dict,
    device: torch.device,
) -> tuple[list[int], list[int], list[float], list[int]]:
    """Returns (preds, true_labels, confidences, image_indices)."""
    model.eval()
    ds = TensorDataset(embed_val["embeddings"], embed_val["labels"], embed_val["image_indices"])
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    all_preds, all_labels, all_confs, all_img_idx = [], [], [], []
    for embs, labels, img_idx in loader:
        embs = embs.to(device)
        logits = model(embs)
        probs = F.softmax(logits, dim=-1).cpu()
        preds = probs.argmax(dim=-1)
        confs = probs.max(dim=-1).values
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_confs.extend(confs.tolist())
        all_img_idx.extend(img_idx.tolist())

    return all_preds, all_labels, all_confs, all_img_idx


def compute_and_save_results(
    preds: list[int],
    labels: list[int],
    confs: list[float],
    img_indices: list[int],
    image_paths: list[str],
    out_dir: str,
    title: str,
) -> None:
    """Save confusion matrix, per-class accuracy, and misclassified list."""
    correct = sum(p == t for p, t in zip(preds, labels))
    total = len(labels)
    overall_acc = correct / total
    log.info(f"{title} val accuracy: {overall_acc:.4f}  ({correct}/{total})")

    per_class_acc: dict[int, float] = {}
    for c in range(10):
        idxs = [i for i, lbl in enumerate(labels) if lbl == c]
        if idxs:
            per_class_acc[c] = sum(preds[i] == c for i in idxs) / len(idxs)
        else:
            per_class_acc[c] = 0.0
        log.info(f"  Class {c}: {per_class_acc[c]:.4f}")

    confusion = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        confusion[t][p] += 1

    misclassified = []
    all_predictions = []
    for i in range(total):
        entry = {
            "image_path": image_paths[img_indices[i]],
            "true_label": labels[i],
            "pred_label": preds[i],
            "confidence": round(confs[i], 4),
        }
        all_predictions.append(entry)
        if labels[i] != preds[i]:
            misclassified.append(entry)

    misclassified.sort(key=lambda x: x["confidence"], reverse=True)

    results = {
        "overall_accuracy": round(overall_acc, 4),
        "per_class_accuracy": {str(c): round(v, 4) for c, v in per_class_acc.items()},
        "total_correct": correct,
        "total_samples": total,
        "num_misclassified": len(misclassified),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(out_dir, "misclassified.json"), "w") as fh:
        json.dump(misclassified, fh, indent=2)
    with open(os.path.join(out_dir, "all_predictions.json"), "w") as fh:
        json.dump(all_predictions, fh, indent=2)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=list(range(10)), yticklabels=list(range(10)))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}  (acc = {overall_acc:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    log.info(f"Saved artifacts → {out_dir}/")


def save_image_grid(items: list[dict], out_path: str, n_cols: int = 5) -> None:
    if not items:
        return
    n = len(items)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.2))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for ax in axes.flat:
        ax.axis("off")
    for i, item in enumerate(items):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        try:
            ax.imshow(Image.open(item["image_path"]))
            clip_str = f"C:{item.get('clip_pred', '?')}"
            dino_str = f"D:{item.get('dino_pred', '?')}"
            ax.set_title(f"True:{item['true_label']}\n{clip_str} {dino_str}", fontsize=7)
        except Exception:
            pass
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(CROSS_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("logs/evaluate.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    with open(f"{EMBEDDINGS_DIR}/image_paths.json") as fh:
        image_paths: list[str] = json.load(fh)

    device = get_device()
    log.info(f"Device: {device}")

    # --- CLIP downstream ---
    log.info("Evaluating CLIP downstream model...")
    with open(os.path.join(CLIP_CKPT_DIR, "best_config.json")) as fh:
        clip_cfg = json.load(fh)

    clip_model = build_mlp(clip_cfg["in_dim"], clip_cfg["dropout"]).to(device)
    clip_model.load_state_dict(
        torch.load(os.path.join(CLIP_CKPT_DIR, "best_model.pt"), weights_only=True)
    )
    clip_val = torch.load(f"{EMBEDDINGS_DIR}/clip_val_embeddings.pt", weights_only=True)
    clip_preds, clip_labels, clip_confs, clip_img_idx = evaluate_model(clip_model, clip_val, device)
    compute_and_save_results(
        clip_preds, clip_labels, clip_confs, clip_img_idx,
        image_paths, CLIP_CKPT_DIR, "CLIP Downstream",
    )

    # --- DINO downstream ---
    log.info("Evaluating DINO downstream model...")
    with open(os.path.join(DINO_CKPT_DIR, "best_config.json")) as fh:
        dino_cfg = json.load(fh)

    dino_input = dino_cfg.get("dino_input", "cls")
    dino_val_path = f"{EMBEDDINGS_DIR}/dino_val_cls.pt" if dino_input == "cls" \
        else f"{EMBEDDINGS_DIR}/dino_val_concat.pt"
    dino_val = torch.load(dino_val_path, weights_only=True)

    dino_model = build_mlp(dino_cfg["in_dim"], dino_cfg["dropout"]).to(device)
    dino_model.load_state_dict(
        torch.load(os.path.join(DINO_CKPT_DIR, "best_model.pt"), weights_only=True)
    )
    dino_preds, dino_labels, dino_confs, dino_img_idx = evaluate_model(dino_model, dino_val, device)
    compute_and_save_results(
        dino_preds, dino_labels, dino_confs, dino_img_idx,
        image_paths, DINO_CKPT_DIR, "DINO Downstream",
    )

    # --- Cross-model comparison ---
    log.info("Computing cross-model comparison...")

    clip_correct = {img_idx: (p == t) for p, t, img_idx in zip(clip_preds, clip_labels, clip_img_idx)}
    dino_correct = {img_idx: (p == t) for p, t, img_idx in zip(dino_preds, dino_labels, dino_img_idx)}

    # Build lookup: img_idx → full prediction info
    clip_info = {
        clip_img_idx[i]: {
            "image_path": image_paths[clip_img_idx[i]],
            "true_label": clip_labels[i],
            "clip_pred": clip_preds[i],
            "clip_conf": round(clip_confs[i], 4),
        }
        for i in range(len(clip_preds))
    }
    dino_info = {
        dino_img_idx[i]: {"dino_pred": dino_preds[i], "dino_conf": round(dino_confs[i], 4)}
        for i in range(len(dino_preds))
    }

    # Common image indices present in both val sets
    common_idx = set(clip_correct.keys()) & set(dino_correct.keys())

    clip_right_dino_wrong = []
    dino_right_clip_wrong = []
    for idx in sorted(common_idx):
        c_right = clip_correct[idx]
        d_right = dino_correct[idx]
        entry = {**clip_info[idx], **dino_info.get(idx, {})}
        if c_right and not d_right:
            clip_right_dino_wrong.append(entry)
        elif d_right and not c_right:
            dino_right_clip_wrong.append(entry)

    with open(os.path.join(CROSS_DIR, "clip_right_dino_wrong.json"), "w") as fh:
        json.dump(clip_right_dino_wrong, fh, indent=2)
    with open(os.path.join(CROSS_DIR, "dino_right_clip_wrong.json"), "w") as fh:
        json.dump(dino_right_clip_wrong, fh, indent=2)

    log.info(f"CLIP right / DINO wrong: {len(clip_right_dino_wrong)}")
    log.info(f"DINO right / CLIP wrong: {len(dino_right_clip_wrong)}")

    # Save example image grids (up to 10 examples each direction)
    examples_dir = os.path.join(CROSS_DIR, "comparison_examples")
    os.makedirs(examples_dir, exist_ok=True)

    save_image_grid(
        clip_right_dino_wrong[:10],
        os.path.join(examples_dir, "clip_right_dino_wrong.png"),
    )
    save_image_grid(
        dino_right_clip_wrong[:10],
        os.path.join(examples_dir, "dino_right_clip_wrong.png"),
    )

    # --- Zero-shot vs downstream CLIP comparison ---
    zs_pred_path = os.path.join(ZS_DIR, "all_predictions.json")
    if os.path.exists(zs_pred_path):
        log.info("Computing zero-shot vs downstream CLIP comparison...")
        with open(zs_pred_path) as fh:
            zs_preds_list: list[dict] = json.load(fh)

        # Only val images (use clip val image_indices)
        val_img_idx_set = set(clip_img_idx)
        # Build path → zs prediction
        zs_by_path = {x["image_path"]: x for x in zs_preds_list}

        # Build path → clip downstream prediction
        clip_by_path = {
            image_paths[clip_img_idx[i]]: {
                "true_label": clip_labels[i],
                "clip_pred": clip_preds[i],
                "clip_conf": round(clip_confs[i], 4),
            }
            for i in range(len(clip_preds))
        }

        zs_right_clip_wrong = []
        clip_right_zs_wrong = []
        for path, clip_entry in clip_by_path.items():
            if path not in zs_by_path:
                continue
            zs_entry = zs_by_path[path]
            t = clip_entry["true_label"]
            zs_correct = zs_entry["pred_label"] == t
            clip_correct_here = clip_entry["clip_pred"] == t
            entry = {
                "image_path": path,
                "true_label": t,
                "zs_pred": zs_entry["pred_label"],
                "zs_conf": zs_entry["confidence"],
                **clip_entry,
            }
            if zs_correct and not clip_correct_here:
                zs_right_clip_wrong.append(entry)
            elif clip_correct_here and not zs_correct:
                clip_right_zs_wrong.append(entry)

        with open(os.path.join(CROSS_DIR, "zs_right_clip_downstream_wrong.json"), "w") as fh:
            json.dump(zs_right_clip_wrong, fh, indent=2)
        with open(os.path.join(CROSS_DIR, "clip_downstream_right_zs_wrong.json"), "w") as fh:
            json.dump(clip_right_zs_wrong, fh, indent=2)

        log.info(f"Zero-shot right / CLIP-downstream wrong: {len(zs_right_clip_wrong)}")
        log.info(f"CLIP-downstream right / Zero-shot wrong: {len(clip_right_zs_wrong)}")

    log.info("evaluate.py complete.")


log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
