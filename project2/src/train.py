import os
import json
import math
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import build_model, count_parameters
from dataset import (
    ProvidedDigitDataset,
    SyntheticMNISTDataset,
    CombinedDataset,
    get_train_transform,
    get_val_transform,
    mixup_batch,
    stratified_split,
)

NUM_CLASSES = 10
SOURCE_REAL = 0
SOURCE_SYNTHETIC = 1


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Cosine annealing with linear warmup
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
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------

class EpochMetrics:
    """Accumulates batch metrics within an epoch using vectorized ops."""

    def __init__(self):
        self.total_loss = 0.0
        self.correct = 0
        self.total = 0
        # Fixed-size tensors for per-class and confusion tracking
        self.class_loss = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.class_correct = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        self.class_total = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        # Per-source (0=real, 1=synthetic)
        self.source_loss = torch.zeros(2, dtype=torch.float64)
        self.source_correct = torch.zeros(2, dtype=torch.int64)
        self.source_total = torch.zeros(2, dtype=torch.int64)
        # Confusion matrix [true, pred]
        self.confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    def update(self, loss_per_sample: torch.Tensor, preds: torch.Tensor,
               labels: torch.Tensor, sources: torch.Tensor | None = None):
        """Update with a batch — fully vectorized, no Python per-sample loop.

        labels and sources should already be on CPU to avoid MPS corruption.
        """
        loss_cpu = loss_per_sample.detach().float().cpu()
        preds_cpu = preds.detach().cpu().long()
        labels_cpu = labels.detach().long()

        batch_size = labels_cpu.size(0)
        self.total_loss += loss_cpu.sum().item()
        correct_mask = preds_cpu == labels_cpu
        self.correct += correct_mask.sum().item()
        self.total += batch_size

        # Per-class loss, accuracy, counts
        for c in range(NUM_CLASSES):
            mask = labels_cpu == c
            n = mask.sum().item()
            if n > 0:
                self.class_total[c] += n
                self.class_loss[c] += loss_cpu[mask].sum().item()
                self.class_correct[c] += correct_mask[mask].sum().item()

        # Confusion matrix — vectorised via bincount
        indices = (labels_cpu * NUM_CLASSES + preds_cpu).long()
        counts = torch.bincount(indices, minlength=NUM_CLASSES * NUM_CLASSES)
        self.confusion += counts[:NUM_CLASSES * NUM_CLASSES].reshape(NUM_CLASSES, NUM_CLASSES)

        # Per-source
        if sources is not None:
            sources_cpu = sources.detach().long()
            for s in range(2):
                mask = sources_cpu == s
                n = mask.sum().item()
                if n > 0:
                    self.source_total[s] += n
                    self.source_loss[s] += loss_cpu[mask].sum().item()
                    self.source_correct[s] += correct_mask[mask].sum().item()

    def summarise(self) -> dict:
        """Return a JSON-serialisable summary dict."""
        result = {
            "loss": round(self.total_loss / max(1, self.total), 5),
            "acc": round(self.correct / max(1, self.total), 5),
            "total": self.total,
        }

        # Per-class
        per_class_loss = {}
        per_class_acc = {}
        for c in range(NUM_CLASSES):
            n = self.class_total[c].item()
            if n > 0:
                per_class_loss[c] = round(self.class_loss[c].item() / n, 5)
                per_class_acc[c] = round(self.class_correct[c].item() / n, 5)
            else:
                per_class_loss[c] = 0.0
                per_class_acc[c] = 0.0
        result["per_class_loss"] = per_class_loss
        result["per_class_acc"] = per_class_acc

        # Per-source
        source_names = {SOURCE_REAL: "real", SOURCE_SYNTHETIC: "synthetic"}
        per_source = {}
        for src, name in source_names.items():
            n = self.source_total[src].item()
            if n > 0:
                per_source[name] = {
                    "loss": round(self.source_loss[src].item() / n, 5),
                    "acc": round(self.source_correct[src].item() / n, 5),
                    "n": n,
                }
        if per_source:
            result["per_source"] = per_source

        # Confusion matrix
        cm = {}
        for true_c in range(NUM_CLASSES):
            row = {}
            for pred_c in range(NUM_CLASSES):
                row[pred_c] = self.confusion[true_c, pred_c].item()
            cm[true_c] = row
        result["confusion_matrix"] = cm

        return result


# ---------------------------------------------------------------------------
# Training & validation steps
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    mix_prob: float = 0.5,
) -> dict:
    model.train()
    metrics = EpochMetrics()
    num_batches = len(loader)

    for batch_idx, (images, labels, sources) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels_cpu = labels.clone()
        sources_cpu = sources.clone()
        labels = labels.to(device, non_blocking=True)

        use_soft_labels = False
        if np.random.random() < mix_prob:
            images, soft_labels = mixup_batch(images, labels, alpha=mixup_alpha)
            use_soft_labels = True

        optimizer.zero_grad()
        logits = model(images)

        if use_soft_labels:
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(soft_labels * log_probs).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        loss.backward()
        optimizer.step()

        # Per-sample CE for metrics (no label smoothing, no mixup — clean signal)
        with torch.no_grad():
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            preds = logits.argmax(dim=1)
        metrics.update(per_sample_loss, preds, labels_cpu, sources_cpu)

        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx+1}/{num_batches}  "
                  f"loss={metrics.total_loss / metrics.total:.4f}  "
                  f"acc={metrics.correct / metrics.total:.4f}", flush=True)

    return metrics.summarise()


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run validation. All metric computation happens on CPU to avoid MPS
    tensor corruption issues with integer tensors."""
    model.eval()
    metrics = EpochMetrics()

    for images, labels, _sources in loader:
        # Only images go to device; labels stay on CPU throughout
        images = images.to(device, non_blocking=True)
        logits = model(images)

        # Bring logits to CPU, compute everything there
        logits_cpu = logits.cpu()
        per_sample_loss = F.cross_entropy(logits_cpu, labels, reduction="none")
        preds = logits_cpu.argmax(dim=1)
        metrics.update(per_sample_loss, preds, labels, sources=None)

    return metrics.summarise()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = get_device()
    print(f"Device: {device}", flush=True)
    # Override mixup/label-smoothing when augmentations are disabled
    if args.no_augment:
        args.mixup_alpha = 0.0
        args.mix_prob = 0.0
        args.label_smoothing = 0.0

    print(f"Config: lr={args.lr}, wd={args.weight_decay}, batch={args.batch_size}, "
          f"epochs={args.epochs}, drop_path={args.drop_path_rate}, "
          f"mixup={args.mixup_alpha}@{args.mix_prob}, ls={args.label_smoothing}, "
          f"synthetic_n={args.synthetic_n}, no_augment={args.no_augment}",
          flush=True)

    # Data split
    train_files, val_files = stratified_split(
        args.data_dir, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"../data: {len(train_files)} train, {len(val_files)} val", flush=True)

    # Model
    model_kwargs = {}
    if args.model == "resnet18":
        model_kwargs["drop_rate"] = args.drop_rate
        model_kwargs["base_width"] = args.base_width
    model = build_model(args.model, **model_kwargs)
    n_params = count_parameters(model)
    print(f"Model: {args.model}, {n_params:,} params ({n_params/1e6:.2f}M)", flush=True)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs, min_lr=1e-6,
    )

    # State
    history: list[dict] = []
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # Save augmentation examples for this run
    try:
        from visualize_augmentations import main as viz_aug_main
        aug_dir = os.path.join(args.save_dir, "augmentation_examples")
        print("Saving augmentation examples...", flush=True)
        viz_aug_main(out_dir=aug_dir, data_dir=args.data_dir)
    except Exception as e:
        print(f"Warning: could not save augmentation examples: {e}", flush=True)

    # Build datasets and loaders
    img_size = args.img_size
    train_tf = get_val_transform(img_size) if args.no_augment else get_train_transform(img_size)
    val_tf = get_val_transform(img_size)

    real_train = ProvidedDigitDataset(args.data_dir, transform=train_tf, file_list=train_files)
    if args.synthetic_n > 0:
        synthetic = SyntheticMNISTDataset(
            args.data_dir, transform=train_tf,
            num_samples=args.synthetic_n,
            img_size=img_size, seed=args.seed,
        )
        train_dataset = CombinedDataset(real_train, synthetic)
        print(f"Train: {len(real_train)} real + {len(synthetic)} synthetic "
              f"= {len(train_dataset)}", flush=True)
    else:
        train_dataset = real_train
        print(f"Train: {len(real_train)} real", flush=True)

    val_dataset = ProvidedDigitDataset(args.data_dir, transform=val_tf, file_list=val_files)
    print(f"Val: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Batches: {len(train_loader)} train, {len(val_loader)} val", flush=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            mix_prob=args.mix_prob,
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        # Build log entry
        entry = {
            "epoch": epoch,
            "lr": round(lr, 7),
            "img_size": img_size,
            "time_s": round(elapsed, 1),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(entry)

        # Print summary
        tl = train_metrics["loss"]
        ta = train_metrics["acc"]
        vl = val_metrics["loss"]
        va = val_metrics["acc"]
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train {tl:.4f}/{ta:.4f}  val {vl:.4f}/{va:.4f}  "
              f"lr={lr:.2e}  {elapsed:.0f}s", flush=True)

        # Per-class detail every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            vacc = val_metrics["per_class_acc"]
            vloss = val_metrics["per_class_loss"]
            print("  Val per-class acc : " +
                  " ".join(f"{c}:{vacc[c]:.2f}" for c in range(NUM_CLASSES)), flush=True)
            print("  Val per-class loss: " +
                  " ".join(f"{c}:{vloss[c]:.2f}" for c in range(NUM_CLASSES)), flush=True)

        # Per-source detail if available
        if "per_source" in train_metrics and epoch % 10 == 0:
            for src_name, src_data in train_metrics["per_source"].items():
                print(f"  Train {src_name:9s}: loss={src_data['loss']:.4f} "
                      f"acc={src_data['acc']:.4f} (n={src_data['n']})", flush=True)

        # Save best model
        if va > best_val_acc or (va == best_val_acc and vl < best_val_loss):
            best_val_acc = va
            best_val_loss = vl
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_acc": va, "val_loss": vl},
                os.path.join(args.save_dir, "best_model.pt"),
            )
            print(f"  ** New best val_acc={va:.4f} **", flush=True)
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % 20 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(),
                 "val_acc": va},
                os.path.join(args.save_dir, "latest_checkpoint.pt"),
            )

        # Write history after every epoch (so it's always up to date on disk)
        with open(os.path.join(args.save_dir, "training_log.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}", flush=True)

    # Generate plots
    try:
        from plot_metrics import plot_all
        plot_all(os.path.join(args.save_dir, "training_log.json"), args.save_dir)
    except Exception as e:
        print(f"Warning: could not generate plots: {e}", flush=True)

    # Visualize misclassified examples using the best checkpoint
    try:
        from visualize_errors import collect_predictions, plot_misclassified_grid, \
            plot_errors_by_class, plot_error_pair_grid, save_misclassified_images
        from torch.utils.data import DataLoader as DL

        best_path = os.path.join(args.save_dir, "best_model.pt")
        if os.path.exists(best_path):
            best_model = build_model(args.model)
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
            best_model.load_state_dict(best_ckpt["model_state_dict"])
            best_model = best_model.to(device)
            best_model.eval()

            err_loader = DL(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
            results = collect_predictions(best_model, err_loader, device)
            n_err = sum(1 for r in results if r["true"] != r["pred"])
            print(f"Misclassified: {n_err}/{len(results)} "
                  f"({n_err/len(results)*100:.1f}%)", flush=True)

            plot_misclassified_grid(results, args.save_dir)
            plot_errors_by_class(results, args.save_dir)
            plot_error_pair_grid(results, args.save_dir)

            val_acc = best_ckpt.get("val_acc", 0)
            if val_acc > 0.92:
                save_misclassified_images(results, args.save_dir)
    except Exception as e:
        print(f"Warning: could not generate error visualizations: {e}", flush=True)

    # Generate interactive dashboards into site/ subfolder
    try:
        best_path = os.path.join(args.save_dir, "best_model.pt")
        if os.path.exists(best_path):
            import subprocess, sys
            site_dir = os.path.join(args.save_dir, "site")
            os.makedirs(site_dir, exist_ok=True)
            print("Generating embedding dashboard...", flush=True)
            subprocess.run([sys.executable, "generate_dashboard.py", best_path,
                            "--model", args.model, "--data-dir", args.data_dir,
                            "--img-size", str(args.img_size),
                            "--out", os.path.join(site_dir, "dashboard.html")],
                           check=True)
            print("Generating validation PCA dashboard...", flush=True)
            subprocess.run([sys.executable, "visualize_val_pca.py", best_path,
                            "--model", args.model, "--data-dir", args.data_dir,
                            "--img-size", str(args.img_size),
                            "--out-dir", site_dir],
                           check=True)
    except Exception as e:
        print(f"Warning: could not generate dashboards: {e}", flush=True)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 on corrupted MNIST data")
    parser.add_argument("--data-dir", type=str, default="../data/dataset")
    parser.add_argument("--save-dir", type=str, default="../checkpoints")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--no-augment", action="store_true", default=False,
                        help="Disable all training augmentations (use val transform for train)")
    parser.add_argument("--synthetic-n", type=int, default=5000,
                        help="Number of synthetic MNIST samples (0 to disable)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18"],
                        help="Model architecture")
    parser.add_argument("--drop-rate", type=float, default=0.0,
                        help="Dropout before final FC (resnet18 only)")
    parser.add_argument("--base-width", type=int, default=64,
                        help="Base channel width (resnet18 only)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
