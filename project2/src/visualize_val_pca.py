#!/usr/bin/env python3
"""Visualize validation predictions in 2D PCA, colored by correct/incorrect.

Generates:
  - 10 static PNGs (one per digit class)
  - An interactive HTML dashboard with 10 tabs (hover shows image)

Usage:
    python visualize_val_pca.py checkpoints/run3/best_model.pt
    python visualize_val_pca.py checkpoints/run3/best_model.pt --method tsne
"""

import argparse
import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from model import build_model, ResNet18
from dataset import ProvidedDigitDataset, get_val_transform, stratified_split

sns.set_theme(
    style="whitegrid",
    palette="deep",
    font="serif",
    rc={
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    },
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
NUM_CLASSES = 10
THUMB_SIZE = 56


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self._features = None
        if isinstance(model, ResNet18):
            model.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._features = out

    @torch.no_grad()
    def extract(self, images, device):
        self.model(images.to(device))
        return self._features.view(self._features.size(0), -1).cpu().numpy()


def tensor_to_thumb_b64(tensor):
    img = tensor.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img).resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=65)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@torch.no_grad()
def collect_val_data(model, extractor, loader, device):
    """Extract features, predictions, confidence, and thumbnails for all val samples."""
    model.eval()
    all_feats, all_labels, all_preds, all_confs, all_thumbs = [], [], [], [], []

    for images, labels, _sources in loader:
        feats = extractor.extract(images, device)
        logits = model(images.to(device)).cpu()
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        confs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

        all_feats.append(feats)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_confs.extend(confs.tolist())
        for i in range(images.size(0)):
            all_thumbs.append(tensor_to_thumb_b64(images[i]))

    return {
        "features": np.concatenate(all_feats),
        "labels": all_labels,
        "preds": all_preds,
        "confs": all_confs,
        "thumbs": all_thumbs,
    }


# ---------------------------------------------------------------------------
# Static PNGs
# ---------------------------------------------------------------------------

CORRECT_COLOR = sns.color_palette("deep")[2]   # green
INCORRECT_COLOR = sns.color_palette("deep")[3]  # red


def plot_per_class_pngs(coords, labels, preds, confs, out_dir, method):
    """Generate one PNG per digit class showing correct vs incorrect."""
    labels = np.array(labels)
    preds = np.array(preds)
    correct = labels == preds

    # Global axis limits
    pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.05
    pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.05
    xlim = (coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ylim = (coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

    os.makedirs(out_dir, exist_ok=True)

    for c in range(NUM_CLASSES):
        mask = labels == c
        n_total = mask.sum()
        n_correct = (mask & correct).sum()
        n_wrong = (mask & ~correct).sum()

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot correct first, then incorrect on top
        cm = mask & correct
        if cm.any():
            ax.scatter(coords[cm, 0], coords[cm, 1],
                       color=[CORRECT_COLOR], s=15, alpha=0.5,
                       label=f"Correct ({n_correct})")
        wm = mask & ~correct
        if wm.any():
            ax.scatter(coords[wm, 0], coords[wm, 1],
                       color=[INCORRECT_COLOR], s=25, alpha=0.8,
                       marker="x", linewidths=1.5,
                       label=f"Incorrect ({n_wrong})")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(markerscale=1.5)
        ax.set_title(f"Class {c} — Validation Predictions ({method.upper()})")
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"val_pca_class_{c}.png"), dpi=150)
        plt.close(fig)

    print(f"  {out_dir}/val_pca_class_0.png ... val_pca_class_9.png")


# ---------------------------------------------------------------------------
# Interactive HTML dashboard
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Validation PCA Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'IBM Plex Mono', monospace; background: #fafafa; color: #222; }
  nav { background: #2c3e50; padding: 10px 24px; display: flex; gap: 8px; flex-wrap: wrap; }
  nav button {
    background: #34495e; color: #ecf0f1; border: none; padding: 8px 16px;
    border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 13px;
  }
  nav button:hover { background: #4a6a8a; }
  nav button.active { background: #2980b9; }
  .container { display: flex; padding: 16px; gap: 16px; }
  #plot { flex: 1; height: 80vh; }
  #sidebar {
    width: 200px; display: flex; flex-direction: column;
    align-items: center; gap: 8px; padding-top: 20px;
  }
  #hover-img {
    width: 160px; height: 160px; object-fit: contain;
    border: 2px solid #bbb; border-radius: 4px; background: #eee;
    display: none;
  }
  #hover-label {
    font-size: 13px; text-align: center; min-height: 60px;
    white-space: pre-line;
  }
  h2 { padding: 16px 24px 0; font-size: 18px; }
</style>
</head>
<body>

<nav id="nav"></nav>
<h2 id="title"></h2>
<div class="container">
  <div id="plot"></div>
  <div id="sidebar">
    <img id="hover-img">
    <div id="hover-label"></div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;

const COLOR_CORRECT = '#55a868';
const COLOR_INCORRECT = '#c44e52';

const pages = [];
for (let c = 0; c < 10; c++) {
  pages.push({id: 'class_' + c, title: 'Class ' + c});
}

let currentPage = 'class_0';

function buildNav() {
  const nav = document.getElementById('nav');
  pages.forEach(p => {
    const btn = document.createElement('button');
    btn.textContent = p.title;
    btn.dataset.page = p.id;
    btn.onclick = () => showPage(p.id);
    nav.appendChild(btn);
  });
}

function showPage(pageId) {
  currentPage = pageId;
  document.querySelectorAll('nav button').forEach(b => {
    b.classList.toggle('active', b.dataset.page === pageId);
  });
  const cls = parseInt(pageId.split('_')[1]);
  document.getElementById('title').textContent = 'Class ' + cls + ' — Validation Predictions';
  document.getElementById('hover-img').style.display = 'none';
  document.getElementById('hover-label').textContent = '';
  plotClass(cls);
}

function attachHover(plotDiv, thumbs, labels, preds, confs) {
  plotDiv.on('plotly_hover', function(ev) {
    const pt = ev.points[0];
    const idx = pt.customdata;
    const img = document.getElementById('hover-img');
    img.src = 'data:image/jpeg;base64,' + thumbs[idx];
    img.style.display = 'block';
    const ok = labels[idx] === preds[idx];
    document.getElementById('hover-label').textContent =
      'True: ' + labels[idx] + '\nPred: ' + preds[idx] +
      '\nConf: ' + confs[idx].toFixed(3) +
      '\n' + (ok ? 'CORRECT' : 'INCORRECT');
  });
  plotDiv.on('plotly_unhover', function() {
    document.getElementById('hover-img').style.display = 'none';
    document.getElementById('hover-label').textContent = '';
  });
}

function plotClass(cls) {
  const d = DATA;
  const correctIdx = [], incorrectIdx = [];
  for (let i = 0; i < d.labels.length; i++) {
    if (d.labels[i] !== cls) continue;
    if (d.labels[i] === d.preds[i]) correctIdx.push(i);
    else incorrectIdx.push(i);
  }

  const traces = [];
  if (correctIdx.length > 0) {
    traces.push({
      x: correctIdx.map(i => d.x[i]),
      y: correctIdx.map(i => d.y[i]),
      customdata: correctIdx,
      mode: 'markers', type: 'scattergl',
      marker: {size: 5, color: COLOR_CORRECT, opacity: 0.5},
      name: 'Correct (' + correctIdx.length + ')',
    });
  }
  if (incorrectIdx.length > 0) {
    traces.push({
      x: incorrectIdx.map(i => d.x[i]),
      y: incorrectIdx.map(i => d.y[i]),
      customdata: incorrectIdx,
      mode: 'markers', type: 'scattergl',
      marker: {size: 8, color: COLOR_INCORRECT, opacity: 0.85, symbol: 'x'},
      name: 'Incorrect (' + incorrectIdx.length + ')',
    });
  }

  const layout = {
    xaxis: {title: d.method.toUpperCase() + ' 1', range: [d.xlim[0], d.xlim[1]]},
    yaxis: {title: d.method.toUpperCase() + ' 2', range: [d.ylim[0], d.ylim[1]]},
    margin: {t: 10},
    hovermode: 'closest',
  };
  const div = document.getElementById('plot');
  Plotly.react(div, traces, layout, {responsive: true});
  attachHover(div, d.thumbs, d.labels, d.preds, d.confs);
}

buildNav();
showPage('class_0');
</script>
</body>
</html>"""


def generate_dashboard(coords, data, method, out_path):
    """Build and write the interactive HTML dashboard."""
    pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.05
    pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.05

    payload = {
        "method": method,
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "labels": data["labels"],
        "preds": data["preds"],
        "confs": data["confs"],
        "thumbs": data["thumbs"],
        "xlim": [float(coords[:, 0].min() - pad_x),
                 float(coords[:, 0].max() + pad_x)],
        "ylim": [float(coords[:, 1].min() - pad_y),
                 float(coords[:, 1].max() + pad_y)],
    }

    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))
    with open(out_path, "w") as f:
        f.write(html)

    size_mb = len(html) / (1024 * 1024)
    print(f"  {out_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize val predictions in 2D PCA (correct vs incorrect)")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18"])
    parser.add_argument("--data-dir", type=str, default="../data/dataset")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--method", type=str, default="pca",
                        choices=["pca", "tsne"])
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.checkpoint), "val_pca")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(args.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded {args.model} from {args.checkpoint} "
          f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    extractor = FeatureExtractor(model)

    # Load val data
    _, val_files = stratified_split(args.data_dir)
    val_ds = ProvidedDigitDataset(args.data_dir, transform=get_val_transform(args.img_size),
                              file_list=val_files)
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    print("Extracting features and predictions...")
    data = collect_val_data(model, extractor, loader, device)
    n = len(data["labels"])
    n_err = sum(1 for l, p in zip(data["labels"], data["preds"]) if l != p)
    print(f"  {n} samples, {n_err} errors ({n_err/n*100:.1f}%)")

    # Project to 2D
    print(f"Computing {args.method.upper()}...")
    if args.method == "pca":
        coords = PCA(n_components=2).fit_transform(data["features"])
    else:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, perplexity=30, random_state=42,
                      init="pca", learning_rate="auto").fit_transform(data["features"])

    # Static PNGs
    print("Generating static PNGs...")
    plot_per_class_pngs(coords, data["labels"], data["preds"], data["confs"],
                        args.out_dir, args.method)

    # Interactive dashboard
    print("Generating dashboard...")
    dashboard_path = os.path.join(args.out_dir, "val_dashboard.html")
    generate_dashboard(coords, data, args.method, dashboard_path)

    print(f"Done! Open: file://{os.path.abspath(dashboard_path)}")


if __name__ == "__main__":
    main()
