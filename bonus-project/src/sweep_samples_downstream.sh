#!/usr/bin/env bash
# Plot 1: sweep --real-n for CLIP and DINO downstream MLPs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BONUS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_BASE="$BONUS_DIR/checkpoints/samples"
TRAIN_PY="$SCRIPT_DIR/downstream/train_scaling.py"

REAL_NS=(200 400 800 1600 3200 6400 0)

mkdir -p "$CKPT_BASE"

# ---- CLIP (best config: lr=1e-3, dropout=0.3, mixup=True, h1=256, h2=128) ----
for N in "${REAL_NS[@]}"; do
    TAG="clip_realn${N}"
    SAVE="$CKPT_BASE/$TAG"
    echo "=== CLIP real_n=$N -> $SAVE ==="
    uv run --project "$BONUS_DIR" python "$TRAIN_PY" \
        --model clip \
        --real-n "$N" \
        --h1 256 \
        --h2 128 \
        --lr 1e-3 \
        --dropout 0.3 \
        --mixup True \
        --save-dir "$SAVE"
done

# ---- DINO (best config: lr=1e-3, dropout=0.2, mixup=True, concat, h1=512, h2=256) ----
for N in "${REAL_NS[@]}"; do
    TAG="dino_realn${N}"
    SAVE="$CKPT_BASE/$TAG"
    echo "=== DINO real_n=$N -> $SAVE ==="
    uv run --project "$BONUS_DIR" python "$TRAIN_PY" \
        --model dino_concat \
        --real-n "$N" \
        --h1 512 \
        --h2 256 \
        --lr 1e-3 \
        --dropout 0.2 \
        --mixup True \
        --save-dir "$SAVE"
done

echo "=== sweep_samples_downstream.sh complete ==="
