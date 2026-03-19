import os
import json
import math
import time
import argparse
import importlib.util
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
    RepeatedAugSampler,
    get_train_transform,
    get_val_transform,
    stratified_split,
)

NUM_CLASSES = 10
SOURCE_REAL = 0
SOURCE_SYNTHETIC = 1


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
# Teacher model loading
# ---------------------------------------------------------------------------

def load_teacher(teacher_path: str, device: torch.device) -> nn.Module:
    """Load the pretrained ResNet18 teacher from project2.

    Uses importlib so we can load project2/src/model.py without adding it to
    sys.path (which would conflict with project3's own model.py).
    """
    p2_model_py = os.path.join(os.path.dirname(__file__), "../../project2/src/model.py")
    p2_model_py = os.path.abspath(p2_model_py)

    if not os.path.exists(p2_model_py):
        raise FileNotFoundError(
            f"Cannot find project2 model at {p2_model_py}. "
            "Set --teacher-path to the checkpoint and ensure project2/src/model.py exists."
        )

    spec = importlib.util.spec_from_file_location("project2_model", p2_model_py)
    p2_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2_mod)

    teacher = p2_mod.build_model("resnet18")
    ckpt = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    n = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher loaded: ResNet18, {n:,} params, val_acc={ckpt.get('val_acc', '?')}", flush=True)
    return teacher


# ---------------------------------------------------------------------------
# LR scheduler — cosine annealing with linear warmup
# ---------------------------------------------------------------------------

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-5):
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
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class EpochMetrics:
    """Accumulates per-batch metrics within an epoch."""

    def __init__(self):
        self.total_loss = 0.0
        self.correct = 0
        self.total = 0
        self.class_loss = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.class_correct = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        self.class_total = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        self.source_loss = torch.zeros(2, dtype=torch.float64)
        self.source_correct = torch.zeros(2, dtype=torch.int64)
        self.source_total = torch.zeros(2, dtype=torch.int64)
        self.confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    def update(
        self,
        loss_per_sample: torch.Tensor,
        preds: torch.Tensor,
        labels: torch.Tensor,
        sources: torch.Tensor | None = None,
    ):
        loss_cpu = loss_per_sample.detach().float().cpu()
        preds_cpu = preds.detach().cpu().long()
        labels_cpu = labels.detach().long()

        batch_size = labels_cpu.size(0)
        self.total_loss += loss_cpu.sum().item()
        correct_mask = preds_cpu == labels_cpu
        self.correct += correct_mask.sum().item()
        self.total += batch_size

        for c in range(NUM_CLASSES):
            mask = labels_cpu == c
            n = mask.sum().item()
            if n > 0:
                self.class_total[c] += n
                self.class_loss[c] += loss_cpu[mask].sum().item()
                self.class_correct[c] += correct_mask[mask].sum().item()

        indices = (labels_cpu * NUM_CLASSES + preds_cpu).long()
        counts = torch.bincount(indices, minlength=NUM_CLASSES * NUM_CLASSES)
        self.confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES)

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
        result = {
            "loss": round(self.total_loss / max(1, self.total), 5),
            "acc": round(self.correct / max(1, self.total), 5),
            "total": self.total,
        }

        per_class_loss, per_class_acc = {}, {}
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

        cm = {}
        for true_c in range(NUM_CLASSES):
            cm[true_c] = {
                pred_c: self.confusion[true_c, pred_c].item()
                for pred_c in range(NUM_CLASSES)
            }
        result["confusion_matrix"] = cm
        return result


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def compute_distillation_loss(
    cls_logits: torch.Tensor,
    dist_logits: torch.Tensor | None,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    distillation: str,
    tau: float,
    lambda_dist: float,
) -> torch.Tensor:
    if distillation == "none":
        return F.cross_entropy(cls_logits, labels)

    if distillation == "soft":
        ce = F.cross_entropy(cls_logits, labels)
        kl = F.kl_div(
            F.log_softmax(cls_logits / tau, dim=1),
            F.softmax(teacher_logits / tau, dim=1),
            reduction="batchmean",
        ) * (tau ** 2)
        return (1.0 - lambda_dist) * ce + lambda_dist * kl

    if distillation == "hard":
        teacher_preds = teacher_logits.argmax(dim=1)
        ce_gt = F.cross_entropy(cls_logits, labels)
        ce_teacher = F.cross_entropy(cls_logits, teacher_preds)
        return 0.5 * ce_gt + 0.5 * ce_teacher

    if distillation == "hard-dist":
        assert dist_logits is not None, "hard-dist requires use_dist_token=True"
        teacher_preds = teacher_logits.argmax(dim=1)
        ce_gt = F.cross_entropy(cls_logits, labels)
        ce_teacher = F.cross_entropy(dist_logits, teacher_preds)
        return 0.5 * ce_gt + 0.5 * ce_teacher

    raise ValueError(f"Unknown distillation mode: {distillation!r}")


# ---------------------------------------------------------------------------
# Train / val steps
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher: nn.Module | None,
    distillation: str,
    tau: float,
    lambda_dist: float,
) -> dict:
    model.train()
    metrics = EpochMetrics()
    num_batches = len(loader)

    for batch_idx, (images, labels, sources) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels_cpu = labels.clone()
        sources_cpu = sources.clone()
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward (training mode — returns both heads)
        cls_logits, dist_logits = model.forward_train(images)

        # Teacher logits (not needed for "none" mode)
        teacher_logits = None
        if teacher is not None and distillation != "none":
            with torch.no_grad():
                teacher_logits = teacher(images)

        loss = compute_distillation_loss(
            cls_logits, dist_logits, labels,
            teacher_logits, distillation, tau, lambda_dist,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Metrics: always use cls_logits vs ground truth (clean signal)
        with torch.no_grad():
            per_sample_loss = F.cross_entropy(cls_logits, labels, reduction="none")
            preds = cls_logits.argmax(dim=1)
        metrics.update(per_sample_loss, preds, labels_cpu, sources_cpu)

        if (batch_idx + 1) % 50 == 0:
            print(
                f"    batch {batch_idx+1}/{num_batches}  "
                f"loss={metrics.total_loss / metrics.total:.4f}  "
                f"acc={metrics.correct / metrics.total:.4f}",
                flush=True,
            )

    return metrics.summarise()


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    metrics = EpochMetrics()

    for images, labels, _sources in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)   # forward() always returns single tensor

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

    use_dist_token = (args.distillation == "hard-dist")

    print(
        f"Config: lr={args.lr}, wd={args.weight_decay}, batch={args.batch_size}, "
        f"epochs={args.epochs}, warmup={args.warmup_epochs}, "
        f"drop_path={args.drop_path_rate}, repeat_aug={args.repeat_aug}, "
        f"synthetic_n={args.synthetic_n}, distillation={args.distillation}, "
        f"use_spt={args.use_spt}, use_lsa={args.use_lsa}",
        flush=True,
    )

    # Data split
    train_files, val_files = stratified_split(
        args.data_dir, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"Data split: {len(train_files)} train, {len(val_files)} val", flush=True)

    # Stratified subsample real training files if requested
    if args.real_n > 0 and args.real_n < len(train_files):
        rng = np.random.default_rng(args.seed)
        by_class: dict[int, list] = {}
        for f in train_files:
            lbl = int(os.path.splitext(os.path.basename(f))[0].split("_")[-1].replace("label", ""))
            by_class.setdefault(lbl, []).append(f)
        per_cls = max(1, args.real_n // 10)
        sampled: list[str] = []
        for lbl in sorted(by_class):
            n = min(len(by_class[lbl]), per_cls)
            sampled.extend(rng.choice(by_class[lbl], size=n, replace=False).tolist())
        train_files = sampled[:args.real_n]
        print(f"Subsampled to {len(train_files)} real training files", flush=True)

    # Model
    model = build_model(
        img_size=args.img_size,
        depth=args.depth,
        dim=args.dim,
        heads=args.heads,
        drop_path_rate=args.drop_path_rate,
        use_spt=args.use_spt,
        use_lsa=args.use_lsa,
        use_dist_token=use_dist_token,
    )
    n_params = count_parameters(model)
    print(f"ViT: {n_params:,} params ({n_params/1e6:.2f}M)", flush=True)
    model = model.to(device)
    model = torch.compile(model)
    # Keep a reference to the underlying uncompiled module for state-dict I/O.
    # torch.compile wraps the model and prefixes all keys with "_orig_mod.",
    # which breaks load_state_dict on plain VisionTransformer instances.
    _raw_model = getattr(model, "_orig_mod", model)

    # Resume from checkpoint if requested
    resume_ckpt = getattr(args, "resume_checkpoint", None)
    _resume_data = None
    if resume_ckpt and os.path.exists(resume_ckpt):
        _resume_data = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        _raw_model.load_state_dict(_resume_data["model_state_dict"])
        print(f"Resumed model weights from {resume_ckpt}", flush=True)

    # Teacher
    teacher = None
    if args.distillation != "none":
        teacher = load_teacher(args.teacher_path, device)

    # Optimizer + scheduler
    # ViT training requires skipping weight decay on 1-D params (biases, LayerNorm
    # weights/biases) and token/positional embeddings; applying wd to these decays
    # away the spatial signal and prevents learning.
    no_decay_names = {"bias", "cls_token", "dist_token", "pos_embed"}
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs, min_lr=args.min_lr,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Save augmentation examples for this run
    try:
        from visualize_augmentations import main as viz_aug_main
        aug_dir = os.path.join(args.save_dir, "augmentation_examples")
        print("Saving augmentation examples...", flush=True)
        viz_aug_main(out_dir=aug_dir, data_dir=args.data_dir)
    except Exception as e:
        print(f"Warning: could not save augmentation examples: {e}", flush=True)

    # Build datasets
    img_size = args.img_size
    train_tf = get_train_transform(img_size)
    val_tf = get_val_transform(img_size)

    real_train = ProvidedDigitDataset(args.data_dir, transform=train_tf, file_list=train_files)
    if args.synthetic_n > 0:
        synthetic = SyntheticMNISTDataset(
            args.data_dir, transform=train_tf,
            num_samples=args.synthetic_n,
            img_size=img_size, seed=args.seed,
        )
        train_dataset = CombinedDataset(real_train, synthetic)
        print(
            f"Train: {len(real_train)} real + {len(synthetic)} synthetic "
            f"= {len(train_dataset)} unique", flush=True,
        )
    else:
        train_dataset = real_train
        print(f"Train: {len(real_train)} real", flush=True)

    val_dataset = ProvidedDigitDataset(args.data_dir, transform=val_tf, file_list=val_files)
    print(f"Val: {len(val_dataset)}", flush=True)

    # Repeated augmentation sampler
    train_sampler = RepeatedAugSampler(train_dataset, num_repeats=args.repeat_aug, seed=args.seed)
    print(
        f"Effective train samples per epoch: {len(train_sampler)} "
        f"({len(train_dataset)} unique x  {args.repeat_aug} repeats)",
        flush=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Batches per epoch: {len(train_loader)} train, {len(val_loader)} val", flush=True)

    # Restore optimizer/scheduler state if the checkpoint contains them
    if _resume_data is not None:
        if "optimizer_state_dict" in _resume_data:
            optimizer.load_state_dict(_resume_data["optimizer_state_dict"])
            print("Restored optimizer state.", flush=True)
        if "scheduler_state_dict" in _resume_data:
            scheduler.load_state_dict(_resume_data["scheduler_state_dict"])
            print("Restored scheduler state.", flush=True)

    history: list[dict] = []
    best_val_acc = _resume_data["val_acc"] if _resume_data and "val_acc" in _resume_data else 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0
    if _resume_data and "epoch" in _resume_data and "optimizer_state_dict" in _resume_data:
        start_epoch = _resume_data["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            teacher=teacher,
            distillation=args.distillation,
            tau=args.tau,
            lambda_dist=args.lambda_dist,
        )
        val_metrics = validate(model, val_loader, device)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        entry = {
            "epoch": epoch,
            "lr": round(lr, 7),
            "img_size": img_size,
            "time_s": round(elapsed, 1),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(entry)

        tl = train_metrics["loss"]
        ta = train_metrics["acc"]
        vl = val_metrics["loss"]
        va = val_metrics["acc"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train {tl:.4f}/{ta:.4f}  val {vl:.4f}/{va:.4f}  "
            f"lr={lr:.2e}  {elapsed:.0f}s",
            flush=True,
        )

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            vacc = val_metrics["per_class_acc"]
            vloss = val_metrics["per_class_loss"]
            print("  Val per-class acc : " +
                  " ".join(f"{c}:{vacc[c]:.2f}" for c in range(NUM_CLASSES)), flush=True)
            print("  Val per-class loss: " +
                  " ".join(f"{c}:{vloss[c]:.2f}" for c in range(NUM_CLASSES)), flush=True)

        if "per_source" in train_metrics and epoch % 10 == 0:
            for src_name, src_data in train_metrics["per_source"].items():
                print(
                    f"  Train {src_name:9s}: loss={src_data['loss']:.4f} "
                    f"acc={src_data['acc']:.4f} (n={src_data['n']})",
                    flush=True,
                )

        # Save best model
        if va > best_val_acc or (va == best_val_acc and vl < best_val_loss):
            best_val_acc = va
            best_val_loss = vl
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": _raw_model.state_dict(),
                 "val_acc": va, "val_loss": vl,
                 "args": vars(args)},
                os.path.join(args.save_dir, "best_model.pt"),
            )
            print(f"  ** New best val_acc={va:.4f} **", flush=True)
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": _raw_model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(),
                 "val_acc": va},
                os.path.join(args.save_dir, "latest_checkpoint.pt"),
            )

        with open(os.path.join(args.save_dir, "training_log.json"), "w") as f:
            json.dump(history, f, indent=2)

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    # Save final checkpoint so resume always has a complete state
    torch.save(
        {"epoch": epoch, "model_state_dict": _raw_model.state_dict(),
         "optimizer_state_dict": optimizer.state_dict(),
         "scheduler_state_dict": scheduler.state_dict(),
         "val_acc": best_val_acc},
        os.path.join(args.save_dir, "latest_checkpoint.pt"),
    )

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}", flush=True)

    # Post-training visualizations
    try:
        from plot_metrics import plot_all
        plot_all(os.path.join(args.save_dir, "training_log.json"), args.save_dir)
    except Exception as e:
        print(f"Warning: could not generate plots: {e}", flush=True)

    try:
        from visualize_errors import collect_predictions, plot_misclassified_grid, \
            plot_errors_by_class, plot_error_pair_grid, save_misclassified_images

        best_path = os.path.join(args.save_dir, "best_model.pt")
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_model = build_model(
                img_size=args.img_size,
                depth=args.depth,
                dim=args.dim,
                heads=args.heads,
                drop_path_rate=args.drop_path_rate,
                use_spt=args.use_spt,
                use_lsa=args.use_lsa,
                use_dist_token=use_dist_token,
            )
            best_model.load_state_dict(best_ckpt["model_state_dict"])
            best_model = best_model.to(device)
            best_model.eval()

            from torch.utils.data import DataLoader as DL
            err_loader = DL(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
            results = collect_predictions(best_model, err_loader, device)
            n_err = sum(1 for r in results if r["true"] != r["pred"])
            print(
                f"Misclassified: {n_err}/{len(results)} ({n_err/len(results)*100:.1f}%)",
                flush=True,
            )
            plot_misclassified_grid(results, args.save_dir)
            plot_errors_by_class(results, args.save_dir)
            plot_error_pair_grid(results, args.save_dir)
            val_acc = best_ckpt.get("val_acc", 0)
            if val_acc > 0.90:
                save_misclassified_images(results, args.save_dir)
    except Exception as e:
        print(f"Warning: could not generate error visualizations: {e}", flush=True)

    try:
        best_path = os.path.join(args.save_dir, "best_model.pt")
        if os.path.exists(best_path):
            import subprocess, sys
            site_dir = os.path.join(args.save_dir, "site")
            os.makedirs(site_dir, exist_ok=True)
            print("Generating embedding dashboard...", flush=True)
            subprocess.run(
                [sys.executable, "generate_dashboard.py", best_path,
                 "--data-dir", args.data_dir, "--img-size", str(args.img_size),
                 "--out", os.path.join(site_dir, "dashboard.html"),
                 "--use-spt", str(args.use_spt), "--use-lsa", str(args.use_lsa),
                 "--use-dist-token", str(use_dist_token)],
                check=True,
            )
            print("Generating validation PCA dashboard...", flush=True)
            subprocess.run(
                [sys.executable, "visualize_val_pca.py", best_path,
                 "--data-dir", args.data_dir, "--img-size", str(args.img_size),
                 "--out-dir", site_dir,
                 "--use-spt", str(args.use_spt), "--use-lsa", str(args.use_lsa),
                 "--use-dist-token", str(use_dist_token)],
                check=True,
            )
    except Exception as e:
        print(f"Warning: could not generate dashboards: {e}", flush=True)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train DeiT-tiny on adversarial MNIST")

    # Data
    parser.add_argument("--data-dir", type=str, default="../data/dataset")
    parser.add_argument("--save-dir", type=str, default="../checkpoints/run")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from. Use latest_checkpoint.pt "
                             "to restore optimizer/scheduler state, or best_model.pt for "
                             "weights only.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--synthetic-n", type=int, default=3000)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--repeat-aug", type=int, default=3,
                        help="Number of augmentation draws per image per epoch")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Model extensions
    parser.add_argument("--use-spt", action="store_true", default=False,
                        help="Enable Shifted Patch Tokenization")
    parser.add_argument("--use-lsa", action="store_true", default=False,
                        help="Enable Locality Self-Attention")
    parser.add_argument("--depth", type=int, default=12,
                        help="Number of transformer blocks")
    parser.add_argument("--dim", type=int, default=192,
                        help="Token embedding dimension")
    parser.add_argument("--heads", type=int, default=3,
                        help="Number of attention heads")
    parser.add_argument("--real-n", type=int, default=0,
                        help="Subsample real training images (0 = use all)")

    # Distillation
    parser.add_argument("--distillation", type=str, default="none",
                        choices=["none", "soft", "hard", "hard-dist"],
                        help="Distillation mode. 'hard-dist' adds a distillation token.")
    parser.add_argument("--teacher-path", type=str,
                        default="../../project2/checkpoints/run10_final_9k/best_model.pt",
                        help="Path to pretrained ResNet18 checkpoint")
    parser.add_argument("--tau", type=float, default=4.0,
                        help="Temperature for soft distillation")
    parser.add_argument("--lambda-dist", type=float, default=0.5,
                        help="Lambda for soft distillation loss weighting")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
