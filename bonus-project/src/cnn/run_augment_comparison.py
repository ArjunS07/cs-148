#!/usr/bin/env python3
"""Ablation: augmentations and synthetic data.
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
    "--patience", "20",
]


def main():
    start = time.time()
    log("\n--- Run 5: Real only, no augmentations ---")
    run_training("run5_real_noaug", BASE_ARGS + [
        "--synthetic-n", "0", "--no-augment",
    ])

    log("\n--- Run 6: Real only, with augmentations ---")
    run_training("run6_real_aug", BASE_ARGS + [
        "--synthetic-n", "0",
    ])

    log("\n--- Run 7: Real + synthetic, no augmentations ---")
    run_training("run7_syn_noaug", BASE_ARGS + [
        "--synthetic-n", "3000", "--no-augment",
    ])

    log("\n--- Run 8: Real + synthetic, with augmentations ---")
    run_training("run8_syn_aug", BASE_ARGS + [
        "--synthetic-n", "3000",
    ])

    import torch
    for name in ["run5_real_noaug", "run6_real_aug", "run7_syn_noaug", "run8_syn_aug"]:
        p = f"../checkpoints/{name}/best_model.pt"
        if os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            log(f"  {name:20s}  val_acc={ckpt['val_acc']:.4f}  epoch={ckpt['epoch']}")

    total = time.time() - start
    log(f"\nTotal time: {total/3600:.1f}hr")
    log("Done.")


if __name__ == "__main__":
    main()
