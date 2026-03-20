"""plot_scaling.py — produce scaling law log-log plots.

Plot 1: log error vs log sample count  (samples_results.csv)
Plot 2: log error vs log param count   (params_results.csv)

Usage:
    python plot_scaling.py
"""

import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Style
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BONUS_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(BONUS_DIR, "results")
PLOTS_DIR = os.path.join(BONUS_DIR, "plots")

# Approximate full training set size after 85/15 split of ~10k images
FULL_TRAIN_N = 8750

# Pull colours from seaborn "deep" palette
_PAL = sns.color_palette("deep")
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

def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Plot 1: error vs sample count
# ---------------------------------------------------------------------------

def plot_samples(rows: list[dict], out_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for model in ("cnn", "vit", "clip", "dino"):
        subset = [r for r in rows if r["model"] == model]
        if not subset:
            continue

        ns = [int(r["real_n"]) if int(r["real_n"]) > 0 else FULL_TRAIN_N
              for r in subset]
        errs = [float(r["best_val_error"]) for r in subset]

        pairs = sorted(
            [(n, e) for n, e in zip(ns, errs) if n > 0 and e > 0]
        )
        if not pairs:
            continue

        xs = np.log10([p[0] for p in pairs])
        ys = np.log10([p[1] for p in pairs])

        ax.plot(xs, ys,
                marker=MARKERS[model], color=COLORS[model],
                label=LABELS[model],
                linewidth=1.8, markersize=6, zorder=3)

        # Power-law fit
        if len(xs) >= 2:
            coeffs = np.polyfit(xs, ys, 1)
            x_fit = np.linspace(xs.min(), xs.max(), 200)
            slope_label = f"slope $= {coeffs[0]:.2f}$"
            ax.plot(x_fit, np.polyval(coeffs, x_fit),
                    color=COLORS[model], linestyle="--", linewidth=1.2,
                    alpha=0.65, label=f"{LABELS[model]} fit ({slope_label})",
                    zorder=2)

    # Nicely formatted x-ticks showing actual sample counts
    tick_ns = [200, 400, 800, 1600, 3200, 6400, FULL_TRAIN_N]
    ax.set_xticks(np.log10(tick_ns))
    ax.set_xticklabels([f"${n:,}$" for n in tick_ns], rotation=30, ha="right")

    ax.set_xlabel(r"Training samples $N$")
    ax.set_ylabel(r"Val.\ error rate $(\log_{10})$")
    ax.legend(framealpha=0.9)
    sns.despine(ax=ax, left=False, bottom=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: error vs param count
# ---------------------------------------------------------------------------

def plot_params(rows: list[dict], out_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for model in ("cnn", "vit", "clip", "dino"):
        subset = [r for r in rows if r["model"] == model]
        if not subset:
            continue

        params = [int(r["n_params"]) for r in subset]
        errs   = [float(r["best_val_error"]) for r in subset]
        tags   = [r["tag"] for r in subset]

        triples = sorted(
            [(p, e, t) for p, e, t in zip(params, errs, tags) if p > 0 and e > 0]
        )
        if not triples:
            continue

        xs = np.log10([p[0] for p in triples])
        ys = np.log10([p[1] for p in triples])
        tag_labels = [p[2] for p in triples]

        ax.plot(xs, ys,
                marker=MARKERS[model], color=COLORS[model],
                label=LABELS[model],
                linewidth=1.8, markersize=7, zorder=3)

        # Annotate each point
        for x, y, t in zip(xs, ys, tag_labels):
            short = (t.replace("cnn_", "").replace("vit_", "")
                      .replace("clip_", "").replace("dino_", ""))
            ax.annotate(short, (x, y),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=COLORS[model])

        # Power-law fit
        if len(xs) >= 2:
            coeffs = np.polyfit(xs, ys, 1)
            x_fit = np.linspace(xs.min(), xs.max(), 200)
            slope_label = f"slope $= {coeffs[0]:.2f}$"
            ax.plot(x_fit, np.polyval(coeffs, x_fit),
                    color=COLORS[model], linestyle="--", linewidth=1.2,
                    alpha=0.65, label=f"{LABELS[model]} fit ({slope_label})",
                    zorder=2)

    ax.set_xlabel(r"Parameter count $(\log_{10})$  [MLP head only for CLIP/DINO]")
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
    os.makedirs(PLOTS_DIR, exist_ok=True)
    missing = []

    samples_csv = os.path.join(RESULTS_DIR, "samples_results.csv")
    if os.path.exists(samples_csv):
        plot_samples(read_csv(samples_csv),
                     os.path.join(PLOTS_DIR, "plot1_samples.pdf"))
    else:
        print(f"[WARN] {samples_csv} not found — skipping Plot 1", file=sys.stderr)
        missing.append("samples_results.csv")

    params_csv = os.path.join(RESULTS_DIR, "params_results.csv")
    if os.path.exists(params_csv):
        plot_params(read_csv(params_csv),
                    os.path.join(PLOTS_DIR, "plot2_params.pdf"))
    else:
        print(f"[WARN] {params_csv} not found — skipping Plot 2", file=sys.stderr)
        missing.append("params_results.csv")

    if missing:
        print("Missing:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
