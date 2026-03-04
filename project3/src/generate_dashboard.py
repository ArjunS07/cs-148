#!/usr/bin/env python3
"""Generate an interactive HTML embedding dashboard for the ViT.

Features CLS-token PCA/t-SNE colored by class and source, with image hover.

Usage:
    python generate_dashboard.py checkpoints/run1/best_model.pt
    python generate_dashboard.py checkpoints/run1/best_model.pt --method tsne
"""

import argparse
import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from model import build_model, VisionTransformer
from dataset import (
    ProvidedDigitDataset,
    SyntheticMNISTDataset,
    get_train_transform,
    get_val_transform,
    stratified_split,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
THUMB_SIZE = 56


# ---------------------------------------------------------------------------
# Feature extraction — CLS token after final LayerNorm
# ---------------------------------------------------------------------------

class FeatureExtractor:
    def __init__(self, model: VisionTransformer):
        self.model = model
        self._features = None
        model.norm.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._features = out[:, 0]   # CLS token: (B, dim)

    @torch.no_grad()
    def extract(self, images, device) -> np.ndarray:
        self.model(images.to(device))
        return self._features.cpu().numpy()


def tensor_to_thumb_b64(tensor):
    img = tensor.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img).resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=65)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def collect_data(extractor, loader, device, source_name):
    all_feats, all_labels, all_thumbs = [], [], []
    for images, labels, _sources in loader:
        feats = extractor.extract(images, device)
        all_feats.append(feats)
        all_labels.extend(labels.tolist())
        for i in range(images.size(0)):
            all_thumbs.append(tensor_to_thumb_b64(images[i]))
    return {
        "features": np.concatenate(all_feats),
        "labels": all_labels,
        "thumbs": all_thumbs,
        "source": source_name,
    }


# ---------------------------------------------------------------------------
# HTML template (same as project2)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PCA Embedding Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'IBM Plex Mono', monospace; background: #fafafa; color: #222; }
  nav { background: #2c3e50; padding: 10px 24px; display: flex; gap: 8px; flex-wrap: wrap; }
  nav button { background: #34495e; color: #ecf0f1; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 13px; }
  nav button:hover { background: #4a6a8a; }
  nav button.active { background: #2980b9; }
  .container { display: flex; padding: 16px; gap: 16px; }
  #plot { flex: 1; height: 80vh; }
  #sidebar { width: 180px; display: flex; flex-direction: column; align-items: center; gap: 8px; padding-top: 20px; }
  #hover-img { width: 160px; height: 160px; object-fit: contain; border: 2px solid #bbb; border-radius: 4px; background: #eee; display: none; }
  #hover-label { font-size: 14px; text-align: center; min-height: 40px; }
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
const COLORS = ['#4c72b0','#dd8452','#55a868','#c44e52','#8172b3','#937860','#da8bc3','#8c8c8c','#ccb974','#64b5cd'];
const SOURCE_COLORS = {original:'#4c72b0', augmented:'#dd8452', synthetic:'#55a868'};
const pages = [
  {id: 'classes', title: 'Original Data by Class'},
  {id: 'sources', title: 'All Sources Overlay'},
];
for (let c = 0; c < 10; c++) pages.push({id: 'class_' + c, title: 'Class ' + c + ' by Source'});
let currentPage = 'classes';
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
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b.dataset.page === pageId));
  const page = pages.find(p => p.id === pageId);
  document.getElementById('title').textContent = page.title;
  document.getElementById('hover-img').style.display = 'none';
  document.getElementById('hover-label').textContent = '';
  if (pageId === 'classes') plotClasses();
  else if (pageId === 'sources') plotSources();
  else plotSingleClass(parseInt(pageId.split('_')[1]));
}
function attachHover(plotDiv, allThumbs, allLabels, allSources) {
  plotDiv.on('plotly_hover', function(ev) {
    const pt = ev.points[0];
    const idx = pt.customdata;
    const img = document.getElementById('hover-img');
    img.src = 'data:image/jpeg;base64,' + allThumbs[idx];
    img.style.display = 'block';
    const src = allSources ? allSources[idx] : 'original';
    document.getElementById('hover-label').textContent = 'Label: ' + allLabels[idx] + '\nSource: ' + src;
  });
  plotDiv.on('plotly_unhover', function() {
    document.getElementById('hover-img').style.display = 'none';
    document.getElementById('hover-label').textContent = '';
  });
}
function plotClasses() {
  const d = DATA.original;
  const traces = [];
  for (let c = 0; c < 10; c++) {
    const idx = [];
    d.labels.forEach((l, i) => { if (l === c) idx.push(i); });
    traces.push({ x: idx.map(i => d.x[i]), y: idx.map(i => d.y[i]), customdata: idx, mode: 'markers', type: 'scattergl', marker: {size: 5, color: COLORS[c], opacity: 0.65}, name: '' + c });
  }
  const div = document.getElementById('plot');
  Plotly.react(div, traces, {xaxis: {title: DATA.method.toUpperCase() + ' 1'}, yaxis: {title: DATA.method.toUpperCase() + ' 2'}, legend: {title: {text: 'Digit'}}, margin: {t: 10}, hovermode: 'closest'}, {responsive: true});
  attachHover(div, d.thumbs, d.labels, null);
}
function plotSources() {
  const traces = [];
  let globalThumbs = [], globalLabels = [], globalSources = [], offset = 0;
  ['original','augmented','synthetic'].forEach(src => {
    const d = DATA[src];
    if (!d) return;
    const n = d.x.length;
    const idxArr = Array.from({length: n}, (_, i) => i + offset);
    globalThumbs.push(...d.thumbs); globalLabels.push(...d.labels); globalSources.push(...Array(n).fill(src));
    traces.push({ x: d.x, y: d.y, customdata: idxArr, mode: 'markers', type: 'scattergl', marker: {size: 5, color: SOURCE_COLORS[src], opacity: 0.5, symbol: src === 'augmented' ? 'triangle-up' : src === 'synthetic' ? 'square' : 'circle'}, name: src + ' (' + n + ')' });
    offset += n;
  });
  const div = document.getElementById('plot');
  Plotly.react(div, traces, {xaxis: {title: DATA.method.toUpperCase() + ' 1'}, yaxis: {title: DATA.method.toUpperCase() + ' 2'}, margin: {t: 10}, hovermode: 'closest'}, {responsive: true});
  attachHover(div, globalThumbs, globalLabels, globalSources);
}
function plotSingleClass(cls) {
  const traces = [];
  let globalThumbs = [], globalLabels = [], globalSources = [], offset = 0;
  ['original','augmented','synthetic'].forEach(src => {
    const d = DATA[src];
    if (!d) return;
    const idx = [];
    d.labels.forEach((l, i) => { if (l === cls) idx.push(i); });
    if (idx.length === 0) return;
    const idxArr = Array.from({length: idx.length}, (_, i) => i + offset);
    idx.forEach(i => { globalThumbs.push(d.thumbs[i]); globalLabels.push(d.labels[i]); globalSources.push(src); });
    traces.push({ x: idx.map(i => d.x[i]), y: idx.map(i => d.y[i]), customdata: idxArr, mode: 'markers', type: 'scattergl', marker: {size: 6, color: SOURCE_COLORS[src], opacity: 0.6, symbol: src === 'augmented' ? 'triangle-up' : src === 'synthetic' ? 'square' : 'circle'}, name: src + ' (' + idx.length + ')' });
    offset += idx.length;
  });
  const div = document.getElementById('plot');
  Plotly.react(div, traces, {xaxis: {title: DATA.method.toUpperCase() + ' 1'}, yaxis: {title: DATA.method.toUpperCase() + ' 2'}, margin: {t: 10}, hovermode: 'closest'}, {responsive: true});
  attachHover(div, globalThumbs, globalLabels, globalSources);
}
buildNav();
showPage('classes');
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate interactive PCA dashboard (ViT)")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--data-dir", type=str, default="../data/dataset")
    parser.add_argument("--out", type=str, default="dashboard.html")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--synthetic-n", type=int, default=2000)
    parser.add_argument("--augmented-n", type=int, default=2000)
    parser.add_argument("--use-spt", action="store_true", default=False)
    parser.add_argument("--use-lsa", action="store_true", default=False)
    parser.add_argument("--use-dist-token", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    use_spt = saved_args.get("use_spt", args.use_spt)
    use_lsa = saved_args.get("use_lsa", args.use_lsa)
    use_dist_token = (saved_args.get("distillation", "none") == "hard-dist") or args.use_dist_token

    model = build_model(use_spt=use_spt, use_lsa=use_lsa, use_dist_token=use_dist_token)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded ViT from {args.checkpoint}")

    extractor = FeatureExtractor(model)
    train_files, _ = stratified_split(args.data_dir)
    val_tf = get_val_transform(args.img_size)
    train_tf = get_train_transform(args.img_size)

    print("Extracting: original...")
    orig_ds = ProvidedDigitDataset(args.data_dir, transform=val_tf, file_list=train_files)
    orig = collect_data(extractor, DataLoader(orig_ds, batch_size=64, num_workers=0), device, "original")
    print(f"  {len(orig['labels'])} samples")

    print("Extracting: augmented...")
    aug_files = train_files[:args.augmented_n]
    aug_ds = ProvidedDigitDataset(args.data_dir, transform=train_tf, file_list=aug_files)
    aug = collect_data(extractor, DataLoader(aug_ds, batch_size=64, num_workers=0), device, "augmented")
    print(f"  {len(aug['labels'])} samples")

    print("Extracting: synthetic...")
    syn_ds = SyntheticMNISTDataset(args.data_dir, transform=val_tf,
                                   num_samples=args.synthetic_n, img_size=args.img_size, seed=42)
    syn = collect_data(extractor, DataLoader(syn_ds, batch_size=64, num_workers=0), device, "synthetic")
    print(f"  {len(syn['labels'])} samples")

    print(f"Computing {args.method.upper()}...")
    all_feats = np.concatenate([orig["features"], aug["features"], syn["features"]])
    if args.method == "pca":
        proj = PCA(n_components=2).fit_transform(all_feats)
    else:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=30, random_state=42,
                    init="pca", learning_rate="auto").fit_transform(all_feats)

    n_orig = len(orig["labels"])
    n_aug = len(aug["labels"])

    def make_entry(coords, labels, thumbs):
        return {"x": coords[:, 0].tolist(), "y": coords[:, 1].tolist(),
                "labels": labels, "thumbs": thumbs}

    data = {
        "method": args.method,
        "original": make_entry(proj[:n_orig], orig["labels"], orig["thumbs"]),
        "augmented": make_entry(proj[n_orig:n_orig + n_aug], aug["labels"], aug["thumbs"]),
        "synthetic": make_entry(proj[n_orig + n_aug:], syn["labels"], syn["thumbs"]),
    }

    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    with open(args.out, "w") as f:
        f.write(html)

    print(f"Wrote {args.out} ({len(html) / 1024 / 1024:.1f} MB)")
    print(f"Open in browser: file://{os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
