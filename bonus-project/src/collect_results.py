"""collect_results.py — parse sweep checkpoints into a CSV.

Usage:
    python collect_results.py --sweep samples
    python collect_results.py --sweep params
"""

import argparse
import csv
import importlib.util
import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BONUS_DIR = os.path.dirname(SCRIPT_DIR)
CNN_MODEL_PY         = os.path.join(SCRIPT_DIR, "cnn", "model.py")
VIT_MODEL_PY         = os.path.join(SCRIPT_DIR, "vit", "model.py")
DOWNSTREAM_TRAIN_PY  = os.path.join(SCRIPT_DIR, "downstream", "train_scaling.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_cnn_params(ckpt_args: dict) -> int:
    mod = _load_module("cnn_model", CNN_MODEL_PY)
    model = mod.build_model(
        "resnet18",
        base_width=ckpt_args.get("base_width", 64),
        drop_rate=ckpt_args.get("drop_rate", 0.0),
    )
    return mod.count_parameters(model)


def count_vit_params(ckpt_args: dict) -> int:
    mod = _load_module("vit_model", VIT_MODEL_PY)
    model = mod.build_model(
        img_size=ckpt_args.get("img_size", 128),
        depth=ckpt_args.get("depth", 12),
        dim=ckpt_args.get("dim", 192),
        heads=ckpt_args.get("heads", 3),
        drop_path_rate=0.0,
        use_spt=ckpt_args.get("use_spt", False),
        use_lsa=ckpt_args.get("use_lsa", False),
        use_dist_token=False,
    )
    return mod.count_parameters(model)


def count_downstream_params(ckpt_args: dict) -> int:
    mod = _load_module("downstream_train", DOWNSTREAM_TRAIN_PY)
    in_dim_map = {"clip": 512, "dino_cls": 768, "dino_concat": 1536}
    in_dim = in_dim_map.get(ckpt_args.get("model", "clip"), 512)
    mlp = mod.build_mlp(
        in_dim,
        ckpt_args.get("h1", 256),
        ckpt_args.get("h2", 128),
        ckpt_args.get("dropout", 0.3),
    )
    return mod.count_parameters(mlp)


def parse_checkpoint_dir(ckpt_dir: str) -> dict | None:
    """Return metrics dict for one checkpoint directory, or None on error."""
    tag = os.path.basename(ckpt_dir)
    if tag.startswith("cnn"):
        model_type = "cnn"
    elif tag.startswith("vit"):
        model_type = "vit"
    elif tag.startswith("clip"):
        model_type = "clip"
    elif tag.startswith("dino"):
        model_type = "dino"
    else:
        print(f"  [SKIP] unrecognized tag prefix: {tag}", file=sys.stderr)
        return None

    log_path = os.path.join(ckpt_dir, "training_log.json")
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    if not os.path.exists(log_path):
        print(f"  [SKIP] no training_log.json in {ckpt_dir}", file=sys.stderr)
        return None
    if not os.path.exists(best_path):
        print(f"  [SKIP] no best_model.pt in {ckpt_dir}", file=sys.stderr)
        return None

    with open(log_path) as f:
        history = json.load(f)

    if not history:
        print(f"  [SKIP] empty history in {ckpt_dir}", file=sys.stderr)
        return None

    best_val_acc = max(ep["val"]["acc"] for ep in history)
    best_val_error = round(1.0 - best_val_acc, 6)

    final = history[-1]
    final_val_error = round(1.0 - final["val"]["acc"], 6)
    final_train_error = round(1.0 - final["train"]["acc"], 6)

    # Load args from checkpoint
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    ckpt_args: dict = ckpt.get("args", {})

    real_n = ckpt_args.get("real_n", 0)

    try:
        if model_type == "cnn":
            n_params = count_cnn_params(ckpt_args)
        elif model_type == "vit":
            n_params = count_vit_params(ckpt_args)
        else:
            n_params = count_downstream_params(ckpt_args)
    except Exception as e:
        print(f"  [WARN] could not count params for {tag}: {e}", file=sys.stderr)
        n_params = -1

    return {
        "model": model_type,
        "tag": tag,
        "real_n": real_n,
        "n_params": n_params,
        "best_val_error": best_val_error,
        "final_val_error": final_val_error,
        "final_train_error": final_train_error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, choices=["samples", "params"],
                        help="Which sweep to collect (matches checkpoints/{sweep}/ dir)")
    args = parser.parse_args()

    sweep_dir = os.path.join(BONUS_DIR, "checkpoints", args.sweep)
    results_dir = os.path.join(BONUS_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isdir(sweep_dir):
        print(f"Sweep directory not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for name in sorted(os.listdir(sweep_dir)):
        ckpt_dir = os.path.join(sweep_dir, name)
        if not os.path.isdir(ckpt_dir):
            continue
        print(f"Processing {name}...")
        row = parse_checkpoint_dir(ckpt_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No results found!", file=sys.stderr)
        sys.exit(1)

    out_path = os.path.join(results_dir, f"{args.sweep}_results.csv")
    fieldnames = ["model", "tag", "real_n", "n_params",
                  "best_val_error", "final_val_error", "final_train_error"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")
    for r in rows:
        print(f"  {r['tag']:30s}  params={r['n_params']:>10,}  best_val_err={r['best_val_error']:.4f}")


if __name__ == "__main__":
    main()
