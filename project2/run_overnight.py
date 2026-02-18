#!/usr/bin/env python3
"""Overnight training script.

Runs two training configs with ConvNeXt-Tiny (27.8M params), picks the best,
and exports a TorchScript pipeline.

Usage:
    python run_overnight.py
    python run_overnight.py --export-only checkpoints/run1/best_model.pt
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_training(name: str, extra_args: list[str] | None = None) -> str:
    """Run train.py with given args. Returns path to best_model.pt."""
    save_dir = f"checkpoints/{name}"
    cmd = [
        sys.executable, "train.py",
        "--data-dir", "data/dataset",
        "--save-dir", save_dir,
    ]
    if extra_args:
        cmd.extend(extra_args)

    log(f"Starting run '{name}': {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=False)

    elapsed = time.time() - t0
    log(f"Run '{name}' finished in {elapsed/60:.1f}min (exit code {result.returncode})")

    best_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_path):
        import torch
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        log(f"  Best val_acc={ckpt['val_acc']:.4f} at epoch {ckpt['epoch']}")
    else:
        log(f"  WARNING: {best_path} not found")

    return best_path


def export_pipeline(checkpoint_path: str, model_name: str = "resnet18",
                    output_path: str = "pipeline-cnn.pt"):
    """Export the best model as a TorchScript pipeline."""
    log(f"Exporting {checkpoint_path} -> {output_path}")

    from model import build_model
    from pipeline import DigitClassifierPipeline
    import torch

    model = build_model(model_name)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    pipe = DigitClassifierPipeline(model=model)
    pipe.save_pipeline_local(output_path)
    log(f"Exported to {output_path} (val_acc={ckpt['val_acc']:.4f}, epoch {ckpt['epoch']})")

    # Quick sanity check
    loaded = torch.jit.load(output_path)
    x = torch.randn(1, 3, 256, 256)
    pred = loaded(x)
    log(f"Sanity check: random image -> class {pred.item()}")

    return output_path


def pick_best_checkpoint(paths: list[str]) -> str:
    """Given a list of checkpoint paths, return the one with highest val_acc."""
    import torch
    best_path = None
    best_acc = -1.0
    for p in paths:
        if not os.path.exists(p):
            continue
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        acc = ckpt["val_acc"]
        log(f"  {p}: val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_path = p
    log(f"  -> Best: {best_path} (val_acc={best_acc:.4f})")
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-only", type=str, default=None,
                        help="Skip training, just export this checkpoint")
    args = parser.parse_args()

    start = time.time()
    log("=" * 60)
    log("OVERNIGHT TRAINING PIPELINE")
    log("=" * 60)

    if args.export_only:
        export_pipeline(args.export_only)
        return

    checkpoints = []

    # ---- Run 1: ResNet-18, 200 epochs, real only ----
    # Same config as the 93.76% run but with more epochs for the
    # cosine schedule to work with
    p1 = run_training("run3", [
        "--model", "resnet18",
        "--epochs", "200",
        "--batch-size", "64",
        "--lr", "1e-3",
        "--weight-decay", "1e-4",
        "--warmup-epochs", "5",
        "--label-smoothing", "0.05",
        "--drop-path-rate", "0.0",
        "--drop-rate", "0.2",
        "--mixup-alpha", "0.2",
        "--mix-prob", "0.3",
        "--img-size", "128",
        "--synthetic-n", "0",
        "--patience", "40",
    ])
    checkpoints.append(p1)

    # ---- Run 2: ResNet-18, 200 epochs, with 3000 synthetic ----
    p2 = run_training("run4", [
        "--model", "resnet18",
        "--epochs", "200",
        "--batch-size", "64",
        "--lr", "1e-3",
        "--weight-decay", "1e-4",
        "--warmup-epochs", "5",
        "--label-smoothing", "0.05",
        "--drop-path-rate", "0.0",
        "--drop-rate", "0.2",
        "--mixup-alpha", "0.2",
        "--mix-prob", "0.3",
        "--img-size", "128",
        "--synthetic-n", "3000",
        "--patience", "40",
    ])
    checkpoints.append(p2)

    # Also consider the previous best
    prev_best = "checkpoints/run2/best_model.pt"
    if os.path.exists(prev_best):
        checkpoints.append(prev_best)

    # ---- Pick best and export ----
    log("\nComparing checkpoints:")
    best = pick_best_checkpoint(checkpoints)
    if best:
        export_pipeline(best, model_name="resnet18")
    else:
        log("ERROR: No valid checkpoints found!")

    total = time.time() - start
    log(f"\nTotal time: {total/3600:.1f}hr")
    log("Done. Check checkpoints/*/training_log.json for details.")


if __name__ == "__main__":
    main()
