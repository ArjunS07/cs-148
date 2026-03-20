"""benchmark_inference.py — Plot 3: Error vs. Wall-Clock Inference Latency.

Benchmarks CNN, ViT, CLIP, DINO on a fixed 64-sample batch (K=10 timed passes
after 2 warm-up passes), saves timing CSV, and produces a log-log scatter plot.

Usage (from bonus-project/src):
    uv run --project .. python benchmark_inference.py
"""

import csv
import importlib.util
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BONUS_DIR  = os.path.dirname(SCRIPT_DIR)
CS148_DIR  = os.path.dirname(BONUS_DIR)

P2_SRC     = os.path.join(CS148_DIR, "project2", "src")
P3_SRC     = os.path.join(CS148_DIR, "project3", "src")
P4_DIR     = os.path.join(CS148_DIR, "project4")

DATA_DIR   = os.path.join(BONUS_DIR, "data", "dataset")
EMBED_DIR  = os.path.join(P4_DIR, "embeddings")

CNN_CKPT   = os.path.join(CS148_DIR, "project2", "checkpoints", "run10_final_9k", "best_model.pt")
VIT_CKPT   = os.path.join(CS148_DIR, "project3", "checkpoints", "final-results", "best_model.pt")
CLIP_CKPT  = os.path.join(P4_DIR, "checkpoints", "clip_downstream", "best_model.pt")
DINO_CKPT  = os.path.join(P4_DIR, "checkpoints", "dino_downstream", "best_model.pt")

RESULTS_DIR = os.path.join(BONUS_DIR, "results")
PLOTS_DIR   = os.path.join(BONUS_DIR, "plots")

BATCH_SIZE = 64
K = 10

# ---------------------------------------------------------------------------
# Style (matches plot_scaling.py)
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", palette="deep", font="serif")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

_PAL    = sns.color_palette("deep")
COLORS  = {"cnn": _PAL[0], "vit": _PAL[1], "clip": _PAL[2], "dino": _PAL[3]}
MARKERS = {"cnn": "o",     "vit": "s",     "clip": "^",     "dino": "D"}
LABELS  = {
    "cnn":  "CNN (ResNet-18)",
    "vit":  r"ViT (DeiT-tiny)",
    "clip": "CLIP (downstream)",
    "dino": "DINO (downstream)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_results_json(ckpt_path: str) -> float | None:
    """Read overall_accuracy from results.json next to checkpoint."""
    results_path = os.path.join(os.path.dirname(ckpt_path), "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            d = json.load(f)
        return d.get("overall_accuracy")
    return None

# ---------------------------------------------------------------------------
# Batch builders
# ---------------------------------------------------------------------------

def build_image_batch(device: torch.device) -> torch.Tensor:
    """Load val split, take first 64 images → (64, 3, 128, 128) tensor."""
    p2_ds = load_module("p2_dataset", os.path.join(P2_SRC, "dataset.py"))
    _, val_files = p2_ds.stratified_split(DATA_DIR, val_fraction=0.15, seed=42)
    transform = p2_ds.get_val_transform(128)
    ds = p2_ds.ProvidedDigitDataset(DATA_DIR, transform=transform, file_list=val_files[:BATCH_SIZE])
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    return imgs.to(device)


def build_clip_batch(device: torch.device) -> torch.Tensor:
    d = torch.load(os.path.join(EMBED_DIR, "clip_val_embeddings.pt"), weights_only=True)
    return d["embeddings"][:BATCH_SIZE].float().to(device)


def build_dino_batch(device: torch.device) -> torch.Tensor:
    d = torch.load(os.path.join(EMBED_DIR, "dino_val_concat.pt"), weights_only=True)
    return d["embeddings"][:BATCH_SIZE].float().to(device)

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_cnn(device: torch.device):
    p2_mod = load_module("p2_model", os.path.join(P2_SRC, "model.py"))
    ckpt   = torch.load(CNN_CKPT, map_location=device, weights_only=True)
    val_acc = ckpt.get("val_acc")
    model  = p2_mod.build_model("resnet18", base_width=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, val_acc, n_params


def load_vit(device: torch.device):
    p3_mod = load_module("p3_model", os.path.join(P3_SRC, "model.py"))
    ckpt   = torch.load(VIT_CKPT, map_location=device, weights_only=False)
    val_acc = 0.729  # accepted result for this checkpoint
    args   = ckpt.get("args", {})
    model  = p3_mod.build_model(
        depth=args.get("depth", 12),
        dim=args.get("dim", 192),
        heads=args.get("heads", 3),
        img_size=args.get("img_size", 128),
        use_spt=args.get("use_spt", True),
        use_lsa=args.get("use_lsa", True),
        use_dist_token=args.get("use_dist_token", False),
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [ViT] missing keys (non-fatal): {missing}", flush=True)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, val_acc, n_params


def load_clip(device: torch.device):
    ds_mod  = load_module("ds_train", os.path.join(SCRIPT_DIR, "downstream", "train_scaling.py"))
    state   = torch.load(CLIP_CKPT, map_location=device, weights_only=True)
    # Infer h1, h2 from weight shapes (state is a raw state dict)
    h1 = state["0.weight"].shape[0]
    h2 = state["4.weight"].shape[0]
    model = ds_mod.build_mlp(in_dim=512, h1=h1, h2=h2, dropout=0.3)
    model.load_state_dict(state)
    model.to(device).eval()
    val_acc = _read_results_json(CLIP_CKPT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, val_acc, n_params


def load_dino(device: torch.device):
    ds_mod  = load_module("ds_train2", os.path.join(SCRIPT_DIR, "downstream", "train_scaling.py"))
    state   = torch.load(DINO_CKPT, map_location=device, weights_only=True)
    h1 = state["0.weight"].shape[0]
    h2 = state["4.weight"].shape[0]
    model = ds_mod.build_mlp(in_dim=1536, h1=h1, h2=h2, dropout=0.2)
    model.load_state_dict(state)
    model.to(device).eval()
    val_acc = _read_results_json(DINO_CKPT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, val_acc, n_params

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_model(model: torch.nn.Module, batch: torch.Tensor, device: torch.device) -> float:
    """Return ms per image after K timed passes (2 warm-up passes discarded)."""
    # Warm-up
    for _ in range(2):
        with torch.no_grad():
            model(batch)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(K):
        with torch.no_grad():
            model(batch)
        _sync(device)
    elapsed = time.perf_counter() - t0

    return (elapsed / (K * BATCH_SIZE)) * 1000  # ms per image

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_wallclock(rows: list[dict], out_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for r in rows:
        key = r["model"]
        x   = r["ms_per_image"]
        y   = np.log10(r["val_error"])
        ax.scatter(x, y,
                   color=COLORS[key], marker=MARKERS[key],
                   s=90, zorder=4, label=LABELS[key])
        ax.annotate(
            LABELS[key],
            (x, y),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color=COLORS[key],
        )

    ax.set_xlabel(r"Inference time per image, ms")
    ax.set_ylabel(r"Val.\ error rate $(\log_{10})$")
    ax.legend(framealpha=0.9)
    sns.despine(ax=ax, left=False, bottom=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip benchmarking; regenerate plot from saved inference_timing.csv")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "inference_timing.csv")

    if args.plot_only:
        rows = []
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "model":        r["model"],
                    "val_error":    float(r["val_error"]),
                    "ms_per_image": float(r["ms_per_image"]),
                    "n_params":     int(r["n_params"]),
                })
        plot_wallclock(rows, os.path.join(PLOTS_DIR, "plot3_wallclock.pdf"))
        return

    device = get_device()
    print(f"Device: {device}", flush=True)

    # --- Build batches ---
    print("Loading image batch for CNN/ViT ...", flush=True)
    img_batch = build_image_batch(device)
    print(f"  image batch: {tuple(img_batch.shape)}", flush=True)

    print("Loading CLIP embedding batch ...", flush=True)
    clip_batch = build_clip_batch(device)
    print(f"  CLIP batch: {tuple(clip_batch.shape)}", flush=True)

    print("Loading DINO embedding batch ...", flush=True)
    dino_batch = build_dino_batch(device)
    print(f"  DINO batch: {tuple(dino_batch.shape)}", flush=True)

    # --- Load models and benchmark ---
    configs = [
        ("cnn",  load_cnn,  img_batch),
        ("vit",  load_vit,  img_batch),
        ("clip", load_clip, clip_batch),
        ("dino", load_dino, dino_batch),
    ]

    rows = []
    for key, loader, batch in configs:
        print(f"\n[{key.upper()}] Loading model ...", flush=True)
        model, val_acc, n_params = loader(device)
        print(f"  n_params={n_params:,}  val_acc={val_acc}", flush=True)

        print(f"  Timing ({K} passes, batch={BATCH_SIZE}) ...", flush=True)
        ms = time_model(model, batch, device)
        val_error = 1.0 - val_acc if val_acc is not None else float("nan")
        print(f"  {ms:.4f} ms/image  val_error={val_error:.4f}", flush=True)

        rows.append({
            "model":        key,
            "val_error":    val_error,
            "ms_per_image": ms,
            "n_params":     n_params,
        })

    # --- Save CSV ---
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "val_error", "ms_per_image", "n_params"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # --- Plot ---
    plot_wallclock(rows, os.path.join(PLOTS_DIR, "plot3_wallclock.pdf"))


if __name__ == "__main__":
    main()
