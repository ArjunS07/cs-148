#!/usr/bin/env python3
"""Ablation: augmentations x synthetic data.

Hyperparameters match run4 exactly (except patience=20).

Run 5: Real only,  no augmentations
Run 6: Real only,  with augmentations
Run 7: Real + syn, no augmentations
Run 8: Real + syn, with augmentations

Usage:
    nohup python run_augment_comparison.py > augment_comparison.log 2>&1 &
    tail -f augment_comparison.log
"""

import os
import subprocess
import sys
import time
from datetime import datetime


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_training(name: str, extra_args: list[str]) -> str:
    save_dir = f"../checkpoints/{name}"
    cmd = [
        sys.executable, "train.py",
        "--data-dir", "../data/dataset",
        "--save-dir", save_dir,
    ] + extra_args

    log(f"Starting '{name}': {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    log(f"'{name}' finished in {elapsed/60:.1f}min")

    best_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_path):
        import torch
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        log(f"  Best val_acc={ckpt['val_acc']:.4f} at epoch {ckpt['epoch']}")
    else:
        log(f"  WARNING: {best_path} not found")

    return best_path


# Matches run4 hyperparameters exactly (except patience=20)
BASE_ARGS = [
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
    "--patience", "50",
]


def main():
    start = time.time()
    
    log("\n--- Run 9: Real + 6000 synthetic, with augmentations ---")
    run_training("run9_final_6k", BASE_ARGS + [
        "--synthetic-n", "6000",
    ])

    log("\n--- Run 10: Real + 9000 synthetic, with augmentations ---")
    run_training("run10_final_9k", BASE_ARGS + [
        "--synthetic-n", "9000",
    ])

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    import torch
    for name in ["run9_final_6k", "run10_final_9k"]:
        p = f"../checkpoints/{name}/best_model.pt"
        if os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            log(f"  {name:20s}  val_acc={ckpt['val_acc']:.4f}  epoch={ckpt['epoch']}")
        else:
            log(f"  {name:20s}  FAILED")

    total = time.time() - start
    log(f"\nTotal time: {total/3600:.1f}hr")
    log("Done.")


if __name__ == "__main__":
    main()
