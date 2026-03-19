"""
Downstream scaling experiments for CLIP and DINOv2 embeddings.

Reads pre-extracted embeddings from project4/embeddings/, subsamples by --real-n,
trains a two-hidden-layer MLP, and saves training_log.json + best_model.pt.

Usage (from any directory):
    uv run --project /path/to/bonus-project python train_scaling.py \\
        --model clip --real-n 200 --h1 256 --h2 128 --save-dir /tmp/clip_test
"""

import json
import math
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
BONUS_DIR  = os.path.dirname(SRC_DIR)
EMBED_DIR  = os.path.abspath(os.path.join(BONUS_DIR, "../project4/embeddings"))

MODEL_FILES = {
    "clip":        ("clip_train_embeddings.pt",  "clip_val_embeddings.pt",  512),
    "dino_cls":    ("dino_train_cls.pt",          "dino_val_cls.pt",         768),
    "dino_concat": ("dino_train_concat.pt",        "dino_val_concat.pt",     1536),
}

# ---------------------------------------------------------------------------
# Hyperparameter constants (not swept here)
# ---------------------------------------------------------------------------

MAX_EPOCHS   = 200
PATIENCE     = 30
BATCH_SIZE   = 64
WARMUP_EPOCHS = 5
MIN_LR       = 1e-6
WEIGHT_DECAY = 1e-4
P_MIX        = 0.2
MIXUP_ALPHA  = 0.2


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def build_mlp(in_dim: int, h1: int, h2: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.BatchNorm1d(h1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h1, h2),
        nn.BatchNorm1d(h2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h2, 10),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.embeddings[idx], self.labels[idx].item()


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

def stratified_subsample(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    real_n: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if real_n <= 0 or real_n >= len(labels):
        return embeddings, labels
    rng = np.random.default_rng(seed)
    per_cls = max(1, real_n // 10)
    indices: list[int] = []
    for c in range(10):
        cls_idx = (labels == c).nonzero(as_tuple=True)[0].numpy()
        n = min(len(cls_idx), per_cls)
        chosen = rng.choice(cls_idx, size=n, replace=False)
        indices.extend(chosen.tolist())
    indices = indices[:real_n]
    idx_t = torch.tensor(indices, dtype=torch.long)
    return embeddings[idx_t], labels[idx_t]


# ---------------------------------------------------------------------------
# LR scheduler
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
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mixup: bool,
) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for embs, labels in loader:
        embs = embs.to(device)
        labels_dev = torch.tensor(labels, device=device) if not isinstance(labels, torch.Tensor) \
            else labels.to(device)
        labels_cpu = labels_dev.cpu()

        if use_mixup and np.random.random() < P_MIX:
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            idx = torch.randperm(embs.size(0), device=device)
            mixed = lam * embs + (1.0 - lam) * embs[idx]
            oh = F.one_hot(labels_dev, 10).float()
            soft = lam * oh + (1.0 - lam) * oh[idx]
            logits = model(mixed)
            loss = -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        else:
            logits = model(embs)
            loss = F.cross_entropy(logits, labels_dev)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1).cpu()
            correct += (preds == labels_cpu).sum().item()
            total += labels_cpu.size(0)
            total_loss += loss.item() * labels_cpu.size(0)

    return {
        "loss": round(total_loss / max(1, total), 5),
        "acc":  round(correct / max(1, total), 5),
    }


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for embs, labels in loader:
        embs = embs.to(device)
        labels_dev = torch.tensor(labels, device=device) if not isinstance(labels, torch.Tensor) \
            else labels.to(device)
        labels_cpu = labels_dev.cpu()
        logits = model(embs).cpu()
        loss = F.cross_entropy(logits, labels_cpu)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels_cpu).sum().item()
        total += labels_cpu.size(0)
        total_loss += loss.item() * labels_cpu.size(0)

    return {
        "loss": round(total_loss / max(1, total), 5),
        "acc":  round(correct / max(1, total), 5),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = get_device()
    print(f"Device: {device}", flush=True)
    print(f"Embedding dir: {EMBED_DIR}", flush=True)

    # Load embeddings
    train_file, val_file, in_dim = MODEL_FILES[args.model]
    print(f"Loading {args.model} embeddings...", flush=True)
    train_dict = torch.load(os.path.join(EMBED_DIR, train_file), weights_only=True)
    val_dict   = torch.load(os.path.join(EMBED_DIR, val_file),   weights_only=True)

    train_embs, train_lbls = train_dict["embeddings"], train_dict["labels"]
    val_embs,   val_lbls   = val_dict["embeddings"],   val_dict["labels"]

    # Stratified subsample
    train_embs, train_lbls = stratified_subsample(train_embs, train_lbls, args.real_n, args.seed)
    print(f"Train: {len(train_lbls)} samples | Val: {len(val_lbls)} samples", flush=True)

    # Build model
    model = build_mlp(in_dim, args.h1, args.h2, args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"MLP: in={in_dim} h1={args.h1} h2={args.h2} → {n_params:,} params", flush=True)

    # Datasets / loaders
    train_ds = EmbeddingDataset(train_embs, train_lbls)
    val_ds   = EmbeddingDataset(val_embs,   val_lbls)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=(len(train_ds) > BATCH_SIZE))
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

    # Optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS, MIN_LR)

    os.makedirs(args.save_dir, exist_ok=True)

    history: list[dict] = []
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.mixup)
        val_metrics   = validate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        entry = {
            "epoch": epoch,
            "lr": round(lr, 7),
            "train": train_metrics,
            "val":   val_metrics,
        }
        history.append(entry)

        print(
            f"Epoch {epoch:3d}/{MAX_EPOCHS}  "
            f"train {train_metrics['loss']:.4f}/{train_metrics['acc']:.4f}  "
            f"val {val_metrics['loss']:.4f}/{val_metrics['acc']:.4f}  "
            f"lr={lr:.2e}",
            flush=True,
        )

        va = val_metrics["acc"]
        if va > best_val_acc:
            best_val_acc = va
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": va,
                    "args": vars(args),
                },
                os.path.join(args.save_dir, "best_model.pt"),
            )
            print(f"  ** New best val_acc={va:.4f} **", flush=True)
        else:
            patience_counter += 1

        with open(os.path.join(args.save_dir, "training_log.json"), "w") as f:
            json.dump(history, f, indent=2)

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}", flush=True)
            break

    print(f"\nDone. Best val_acc={best_val_acc:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Downstream MLP scaling experiments")
    parser.add_argument("--model", required=True, choices=list(MODEL_FILES.keys()))
    parser.add_argument("--real-n", type=int, default=0,
                        help="Subsample training embeddings (0 = all)")
    parser.add_argument("--h1", type=int, default=256,
                        help="First hidden layer dimension")
    parser.add_argument("--h2", type=int, default=128,
                        help="Second hidden layer dimension")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mixup", type=lambda x: x.lower() == "true", default=True)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
