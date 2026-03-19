#!/usr/bin/env python3
"""Grid search over learning rate x batch size.

LRs:          1e-3, 3e-4, 1e-4
Batch sizes:  32, 64, 128
= 9 runs, 50 epochs each, 1000 synthetic samples.

All other hyperparameters match run4.

Usage:
    cd src && caffeinate -s nohup python run_grid_search.py > ../logs/grid_search.log 2>&1 &
    tail -f ../logs/grid_search.log
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import product


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


# Fixed hyperparameters (match run4 otherwise)
BASE_ARGS = [
    "--model", "resnet18",
    "--epochs", "50",
    "--weight-decay", "1e-4",
    "--warmup-epochs", "5",
    "--label-smoothing", "0.05",
    "--drop-path-rate", "0.0",
    "--drop-rate", "0.2",
    "--mixup-alpha", "0.2",
    "--mix-prob", "0.3",
    "--img-size", "128",
    "--synthetic-n", "1000",
    "--patience", "15",
]

LRS = ["1e-3", "3e-4", "1e-4"]
BATCH_SIZES = [32, 64, 128]


def main():
    start = time.time()
    log("=" * 60)
    log("GRID SEARCH: learning rate x batch size")
    log(f"  LRs:         {LRS}")
    log(f"  Batch sizes: {BATCH_SIZES}")
    log(f"  Runs:        {len(LRS) * len(BATCH_SIZES)}")
    log("=" * 60)

    results = []

    for lr, bs in product(LRS, BATCH_SIZES):
        name = f"gs_lr{lr}_bs{bs}"
        log(f"\n--- {name} ---")
        run_training(name, BASE_ARGS + [
            "--lr", lr,
            "--batch-size", str(bs),
        ])

        p = f"../checkpoints/{name}/best_model.pt"
        if os.path.exists(p):
            import torch
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            results.append((ckpt["val_acc"], lr, bs, ckpt["epoch"], name))

    # Summary table
    log("\n" + "=" * 60)
    log("SUMMARY (sorted by val_acc)")
    log("=" * 60)
    log(f"  {'val_acc':>8}  {'lr':>8}  {'bs':>5}  {'epoch':>6}  name")
    log(f"  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*6}  ----")
    for val_acc, lr, bs, epoch, name in sorted(results, reverse=True):
        log(f"  {val_acc:>8.4f}  {lr:>8}  {bs:>5}  {epoch:>6}  {name}")

    total = time.time() - start
    log(f"\nTotal time: {total/3600:.1f}hr")
    log("Done.")


if __name__ == "__main__":
    main()
