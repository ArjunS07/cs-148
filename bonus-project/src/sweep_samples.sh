#!/usr/bin/env bash
# Plot 1: sweep --real-n across CNN and ViT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BONUS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$BONUS_DIR/data/dataset"
CKPT_BASE="$BONUS_DIR/checkpoints/samples"
TEACHER="$(cd "$BONUS_DIR/../project2/checkpoints/run10_final_9k" && pwd)/best_model.pt"

REAL_NS=(200 400 800 1600 3200 6400 0)

mkdir -p "$CKPT_BASE"

# ---- CNN ----
cd "$SCRIPT_DIR/cnn"
for N in "${REAL_NS[@]}"; do
    TAG="cnn_realn${N}"
    SAVE="$CKPT_BASE/$TAG"
    echo "=== CNN real_n=$N -> $SAVE ==="
    uv run --project "$BONUS_DIR" python train.py \
        --data-dir "$DATA_DIR" \
        --save-dir "$SAVE" \
        --epochs 30 \
        --warmup-epochs 5 \
        --batch-size 64 \
        --synthetic-n 0 \
        --real-n "$N" \
        --patience 0 \
        --num-workers 0
done

# ---- ViT ----
cd "$SCRIPT_DIR/vit"
for N in "${REAL_NS[@]}"; do
    TAG="vit_realn${N}"
    SAVE="$CKPT_BASE/$TAG"
    echo "=== ViT real_n=$N -> $SAVE ==="
    uv run --project "$BONUS_DIR" python train.py \
        --data-dir "$DATA_DIR" \
        --save-dir "$SAVE" \
        --teacher-path "$TEACHER" \
        --epochs 20 \
        --warmup-epochs 3 \
        --batch-size 64 \
        --synthetic-n 0 \
        --real-n "$N" \
        --repeat-aug 1 \
        --distillation none \
        --patience 0 \
        --num-workers 0
done

echo "=== sweep_samples.sh complete ==="
