"""
Run zero-shot CLIP evaluation and save all artifacts.

Evaluates on full dataset (train clean + val) using cached text and image embeddings.
Compares three text-prompt variants: numeral ("0"), word ("zero"), combined.

Usage:
    cd /path/to/project4
    uv run python src/zero_shot_eval.py
"""

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image

EMBEDDINGS_DIR = "embeddings"
OUT_DIR = "checkpoints/clip_zero_shot"
LOG_PATH = "logs/zero_shot_eval.log"


def zero_shot_predict(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (preds [N], probs [N, 10])."""
    img_feats = F.normalize(img_embeds, dim=-1)
    text_feats = F.normalize(text_embeds, dim=-1)
    logits = logit_scale * (img_feats @ text_feats.T)
    probs = F.softmax(logits, dim=-1)
    return logits.argmax(dim=-1), probs


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def save_worst_per_class_grid(worst_per_class: dict, out_path: str) -> None:
    """5 rows × 6 cols grid: each row holds 2 classes, each class gets 3 columns."""
    n_rows, n_cols = 5, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.2))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for ax in axes.flat:
        ax.axis("off")

    for cls in range(10):
        row = cls // 2
        col_start = (cls % 2) * 3
        for rank, item in enumerate(worst_per_class.get(cls, [])):
            ax = axes[row, col_start + rank]
            try:
                ax.imshow(Image.open(item["image_path"]))
            except Exception:
                pass
            ax.set_title(
                f"cls {cls}  #{rank+1}\npred:{item['pred_label']}  conf:{item['confidence']:.2f}",
                fontsize=7,
            )

    fig.suptitle("CLIP Zero-Shot: 3 Worst Misclassifications per Class", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved worst-per-class grid → {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Load image embeddings
    clip_train = torch.load(f"{EMBEDDINGS_DIR}/clip_train_embeddings.pt", weights_only=True)
    clip_val = torch.load(f"{EMBEDDINGS_DIR}/clip_val_embeddings.pt", weights_only=True)

    with open(f"{EMBEDDINGS_DIR}/image_paths.json") as fh:
        image_paths: list[str] = json.load(fh)

    # Clean train + all val
    clean_mask = clip_train["view_type"] == 0
    all_embeds = torch.cat([clip_train["embeddings"][clean_mask], clip_val["embeddings"]], dim=0)
    all_labels = torch.cat([clip_train["labels"][clean_mask], clip_val["labels"]], dim=0)
    all_img_idx = torch.cat([clip_train["image_indices"][clean_mask], clip_val["image_indices"]], dim=0)
    log.info(f"Evaluating on {len(all_embeds)} images (train clean + val)")

    # Load CLIP for logit_scale
    from transformers import CLIPModel
    log.info("Loading CLIP for logit_scale...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    logit_scale = clip_model.logit_scale.exp().item()
    log.info(f"logit_scale = {logit_scale:.4f}")

    # --- Three-way variant comparison ---
    variants: dict[str, str] = {
        "numeral": f"{EMBEDDINGS_DIR}/clip_text_embeddings.pt",
        "word":    f"{EMBEDDINGS_DIR}/clip_text_embeddings_words.pt",
        "combined": f"{EMBEDDINGS_DIR}/clip_text_embeddings_combined.pt",
    }

    comparison: dict[str, dict] = {}
    best_variant = "numeral"
    best_acc = 0.0

    log.info("\n--- Prompt variant comparison ---")
    for variant_name, emb_path in variants.items():
        if not os.path.exists(emb_path):
            log.info(f"  {variant_name}: embedding file not found, skipping")
            continue
        text_embeds = torch.load(emb_path, weights_only=True)
        preds, _ = zero_shot_predict(all_embeds, text_embeds, logit_scale)
        acc = accuracy(preds, all_labels)
        per_class = {
            c: (preds[all_labels == c] == c).float().mean().item()
            for c in range(10)
            if (all_labels == c).any()
        }
        comparison[variant_name] = {
            "overall_accuracy": round(acc, 4),
            "per_class_accuracy": {str(c): round(v, 4) for c, v in per_class.items()},
        }
        log.info(f"  {variant_name:8s}: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_variant = variant_name

    log.info(f"Best variant: {best_variant} ({best_acc:.4f})")
    comparison["best_variant"] = best_variant

    with open(os.path.join(OUT_DIR, "variant_comparison.json"), "w") as fh:
        json.dump(comparison, fh, indent=2)

    # Variant comparison bar chart
    names = [v for v in variants if v in comparison]
    accs = [comparison[v]["overall_accuracy"] for v in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, accs, color=["steelblue", "seagreen", "darkorange"], width=0.5)
    ax.set_ylim(0, max(accs) + 0.1)
    ax.set_ylabel("Overall accuracy")
    ax.set_title("Zero-Shot CLIP: Prompt Variant Comparison")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "variant_comparison.png"), dpi=150)
    plt.close(fig)
    log.info("Saved variant_comparison.png")

    # --- Full evaluation using the best variant ---
    results_path = os.path.join(OUT_DIR, "results.json")

    best_text_path = variants[best_variant]
    text_embeds = torch.load(best_text_path, weights_only=True)
    preds, probs = zero_shot_predict(all_embeds, text_embeds, logit_scale)

    correct = (preds == all_labels).sum().item()
    total = len(all_labels)
    overall_acc = correct / total
    log.info(f"\nFull evaluation ({best_variant} prompts): {overall_acc:.4f}  ({correct}/{total})")

    per_class_acc: dict[int, float] = {}
    for c in range(10):
        mask = all_labels == c
        n = mask.sum().item()
        per_class_acc[c] = (preds[mask] == c).sum().item() / n if n > 0 else 0.0
        log.info(f"  Class {c}: {per_class_acc[c]:.4f}")

    confusion = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_labels.tolist(), preds.tolist()):
        confusion[t][p] += 1

    all_preds_list = []
    for i in range(total):
        all_preds_list.append({
            "image_path": image_paths[all_img_idx[i].item()],
            "true_label": all_labels[i].item(),
            "pred_label": preds[i].item(),
            "confidence": round(probs[i, preds[i].item()].item(), 4),
        })

    misclassified = [x for x in all_preds_list if x["true_label"] != x["pred_label"]]
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)
    log.info(f"Misclassified: {len(misclassified)}/{total}")

    worst_per_class: dict[int, list] = {c: [] for c in range(10)}
    for m in misclassified:
        c = m["true_label"]
        if len(worst_per_class[c]) < 3:
            worst_per_class[c].append(m)

    results = {
        "overall_accuracy": round(overall_acc, 4),
        "prompt_variant": best_variant,
        "per_class_accuracy": {str(c): round(v, 4) for c, v in per_class_acc.items()},
        "total_correct": correct,
        "total_samples": total,
        "num_misclassified": len(misclassified),
    }
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(OUT_DIR, "all_predictions.json"), "w") as fh:
        json.dump(all_preds_list, fh, indent=2)
    with open(os.path.join(OUT_DIR, "misclassified.json"), "w") as fh:
        json.dump(misclassified, fh, indent=2)
    with open(os.path.join(OUT_DIR, "worst_per_class.json"), "w") as fh:
        json.dump({str(c): v for c, v in worst_per_class.items()}, fh, indent=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=list(range(10)), yticklabels=list(range(10)))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"CLIP Zero-Shot Confusion Matrix  (acc = {overall_acc:.3f}, {best_variant} prompts)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)



    save_worst_per_class_grid(worst_per_class, os.path.join(OUT_DIR, "worst_per_class.png"))

    log.info(f"All artifacts saved to {OUT_DIR}/")


log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()

