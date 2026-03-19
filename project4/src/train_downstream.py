"""
Run hyperparameter sweep for CLIP and DINO downstream MLP classifiers.

Trains on cached embeddings with random view sampling (1 of N per image per epoch).
Saves best model, config, training curves, and sweep results CSV.

Usage:
    cd /path/to/project4
    uv run python src/train_downstream.py
"""

import csv
import itertools
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

EMBEDDINGS_DIR = "embeddings"
CLIP_OUT_DIR = "checkpoints/clip_downstream"
DINO_OUT_DIR = "checkpoints/dino_downstream"
LOG_PATH_CLIP = "logs/clip_sweep.log"
LOG_PATH_DINO = "logs/dino_sweep.log"

SEED = 42
MAX_EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 64
LABEL_SMOOTHING = 0.0
P_MIX = 0.2
MIXUP_ALPHA = 0.2
WARMUP_EPOCHS = 5
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4

LR_VALUES = [1e-3, 3e-4, 1e-4]
DROPOUT_VALUES = [0.2, 0.3]
MIXUP_VALUES = [True, False]
DINO_INPUT_VALUES = ["cls", "concat"]

H1_FOR_DIM = {512: 256, 768: 384, 1536: 512}


# ---------------------------------------------------------------------------
# Model
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
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    def __init__(self, embed_dict: dict, train_mode: bool = True):
        self.embeddings = embed_dict["embeddings"]
        self.labels = embed_dict["labels"]
        self.train_mode = train_mode

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.embeddings[idx], self.labels[idx].item()


# ---------------------------------------------------------------------------
# Scheduler (same as project 2)
# ---------------------------------------------------------------------------

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_config(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    use_mixup: bool,
) -> tuple[float, int, dict]:
    """Train one hyperparameter config. Returns (best_val_acc, best_epoch, curves)."""
    curves = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state: dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(MAX_EPOCHS):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for embs, labels in train_loader:
            embs = embs.to(device)
            labels_dev = labels.to(device) if isinstance(labels, torch.Tensor) \
                else torch.tensor(labels, device=device)
            labels_cpu = labels_dev.cpu()

            if use_mixup and np.random.random() < P_MIX:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                idx = torch.randperm(embs.size(0), device=device)
                mixed = lam * embs + (1 - lam) * embs[idx]
                labels_oh = F.one_hot(labels_dev, 10).float()
                mixed_labels = lam * labels_oh + (1 - lam) * labels_oh[idx]
                logits = model(mixed)
                log_probs = F.log_softmax(logits, dim=-1)
                loss = -(mixed_labels * log_probs).sum(dim=-1).mean()
            else:
                logits = model(embs)
                loss = F.cross_entropy(logits, labels_dev, label_smoothing=LABEL_SMOOTHING)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1).cpu()
                correct += (preds == labels_cpu).sum().item()
                total += labels_cpu.size(0)
                total_loss += loss.item() * labels_cpu.size(0)

        scheduler.step()

        # --- Val ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for embs, labels in val_loader:
                embs = embs.to(device)
                labels_dev = labels.to(device) if isinstance(labels, torch.Tensor) \
                    else torch.tensor(labels, device=device)
                labels_cpu = labels_dev.cpu()
                logits = model(embs)
                loss = F.cross_entropy(logits, labels_dev)
                preds = logits.argmax(dim=-1).cpu()
                val_correct += (preds == labels_cpu).sum().item()
                val_total += labels_cpu.size(0)
                val_loss += loss.item() * labels_cpu.size(0)

        train_acc = correct / total
        val_acc = val_correct / val_total
        curves["train_loss"].append(round(total_loss / total, 5))
        curves["train_acc"].append(round(train_acc, 5))
        curves["val_loss"].append(round(val_loss / val_total, 5))
        curves["val_acc"].append(round(val_acc, 5))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return best_val_acc, best_epoch, curves


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    train_dicts: dict[str, dict],
    val_dicts: dict[str, dict],
    model_type: str,
    configs: list[tuple],
    out_dir: str,
    device: torch.device,
) -> None:
    """
    Run the hyperparameter sweep.

    train_dicts / val_dicts: keyed by "clip", "dino_cls", "dino_concat".
    configs: list of (lr, dropout, use_mixup) for CLIP or
                     (lr, dropout, use_mixup, dino_input) for DINO.
    """
    os.makedirs(out_dir, exist_ok=True)
    curves_dir = os.path.join(out_dir, "all_curves")
    os.makedirs(curves_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sweep_results.csv")

    # Skip if sweep already complete
    if os.path.exists(csv_path) and os.path.exists(os.path.join(out_dir, "best_model.pt")):
        log.info(f"{model_type} sweep already complete, skipping.")
        return

    all_rows: list[dict] = []
    best_val_acc = 0.0
    best_row: dict = {}
    best_state: dict = {}
    best_curves: dict = {}

    for cfg_idx, config in enumerate(configs):
        lr, dropout, use_mixup = config[0], config[1], config[2]
        dino_input = config[3] if model_type == "dino" else None
        weight_decay = WEIGHT_DECAY

        cfg_str = f"lr={lr} dropout={dropout} mixup={use_mixup}"
        if dino_input:
            cfg_str += f" input={dino_input}"
        log.info(f"[{cfg_idx + 1}/{len(configs)}] {model_type} | {cfg_str}")

        # Select embeddings
        if model_type == "clip":
            train_dict = train_dicts["clip"]
            val_dict = val_dicts["clip"]
            in_dim = 512
        elif dino_input == "cls":
            train_dict = train_dicts["dino_cls"]
            val_dict = val_dicts["dino_cls"]
            in_dim = 768
        else:
            train_dict = train_dicts["dino_concat"]
            val_dict = val_dicts["dino_concat"]
            in_dim = 1536

        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        model = build_mlp(in_dim, dropout).to(device)
        train_ds = EmbeddingDataset(train_dict, train_mode=True)
        val_ds = EmbeddingDataset(val_dict, train_mode=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS, MIN_LR)

        val_acc, best_ep, curves = train_one_config(
            model, train_loader, val_loader, optimizer, scheduler, device, use_mixup
        )

        row = {
            "cfg_idx": cfg_idx,
            "lr": lr,
            "dropout": dropout,
            "use_mixup": use_mixup,
            "dino_input": dino_input or "N/A",
            "in_dim": in_dim,
            "best_val_acc": round(val_acc, 4),
            "best_epoch": best_ep,
        }
        all_rows.append(row)
        log.info(f"  → val_acc={val_acc:.4f} at epoch {best_ep}")

        # Save this config's curves keyed by index and hyperparams
        curves_path = os.path.join(curves_dir, f"cfg{cfg_idx:03d}.json")
        with open(curves_path, "w") as fh:
            json.dump({"config": row, "curves": curves}, fh)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_row = row
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_curves = curves

    # Save sweep CSV
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # Save best model artifacts
    torch.save(best_state, os.path.join(out_dir, "best_model.pt"))
    with open(os.path.join(out_dir, "best_config.json"), "w") as fh:
        json.dump(best_row, fh, indent=2)
    with open(os.path.join(out_dir, "training_curves.json"), "w") as fh:
        json.dump(best_curves, fh, indent=2)

    log.info(f"{model_type} best val_acc={best_val_acc:.4f}: {best_row}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(CLIP_OUT_DIR, exist_ok=True)
    os.makedirs(DINO_OUT_DIR, exist_ok=True)

    for log_path in [LOG_PATH_CLIP, LOG_PATH_DINO]:
        open(log_path, "a").close()  # ensure file exists

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH_CLIP),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    device = get_device()
    log.info(f"Device: {device}")

    # Load all embedding dicts upfront
    log.info("Loading cached embeddings...")
    train_dicts = {
        "clip":       torch.load(f"{EMBEDDINGS_DIR}/clip_train_embeddings.pt", weights_only=True),
        "dino_cls":   torch.load(f"{EMBEDDINGS_DIR}/dino_train_cls.pt", weights_only=True),
        "dino_concat": torch.load(f"{EMBEDDINGS_DIR}/dino_train_concat.pt", weights_only=True),
    }
    val_dicts = {
        "clip":       torch.load(f"{EMBEDDINGS_DIR}/clip_val_embeddings.pt", weights_only=True),
        "dino_cls":   torch.load(f"{EMBEDDINGS_DIR}/dino_val_cls.pt", weights_only=True),
        "dino_concat": torch.load(f"{EMBEDDINGS_DIR}/dino_val_concat.pt", weights_only=True),
    }
    log.info("Embeddings loaded.")

    # CLIP sweep: 3 LR × 2 dropout × 2 mixup = 12 configs
    clip_configs = list(itertools.product(LR_VALUES, DROPOUT_VALUES, MIXUP_VALUES))
    log.info(f"CLIP sweep: {len(clip_configs)} configs")
    run_sweep(train_dicts, val_dicts, "clip", clip_configs, CLIP_OUT_DIR, device)

    # Re-configure logging for DINO
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH_DINO),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    # DINO sweep: 3 LR × 2 dropout × 2 mixup × 2 input = 24 configs
    dino_configs = list(itertools.product(LR_VALUES, DROPOUT_VALUES, 
                                          _VALUES, DINO_INPUT_VALUES))
    log.info(f"DINO sweep: {len(dino_configs)} configs")
    run_sweep(train_dicts, val_dicts, "dino", dino_configs, DINO_OUT_DIR, device)

    log.info("All sweeps complete.")


log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
