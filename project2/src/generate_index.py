"""Auto-generate index.html for a training run's site/ directory."""

import os


def generate_site_index(site_dir: str, run_name: str, val_acc: float, epoch: int):
    """Write site_dir/index.html with links to dashboard and val_dashboard."""
    val_pct = f"{val_acc * 100:.2f}%"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{run_name} — CS 148 Project 2</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'IBM Plex Mono', monospace;
    background: #fafafa; color: #222;
    display: flex; justify-content: center;
    padding: 48px 32px;
  }}
  .card {{ max-width: 600px; width: 100%; }}
  h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 8px; }}
  .meta {{ font-size: 12px; color: #777; margin-bottom: 24px; }}
  a {{
    display: block; padding: 16px 20px; margin-bottom: 10px;
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    text-decoration: none; color: #222; transition: border-color 0.15s;
  }}
  a:hover {{ border-color: #2980b9; }}
  a strong {{ font-size: 14px; display: block; margin-bottom: 4px; color: #2980b9; }}
  a span {{ font-size: 12px; color: #555; }}
</style>
</head>
<body>
<div class="card">
  <h1>{run_name}</h1>
  <div class="meta">Best epoch {epoch}, val_acc = {val_pct}</div>
  <a href="dashboard.html">
    <strong>Embedding Dashboard</strong>
    <span>Interactive PCA of training data colored by class and source, with image hover.</span>
  </a>
  <a href="val_dashboard.html">
    <strong>Validation Error Dashboard</strong>
    <span>Per-class PCA of validation predictions colored by correct vs incorrect, with image hover.</span>
  </a>
</div>
</body>
</html>"""

    os.makedirs(site_dir, exist_ok=True)
    path = os.path.join(site_dir, "index.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"Wrote {path}")
