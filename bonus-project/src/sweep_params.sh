#!/usr/bin/env bash
# Plot 2: sweep model size (--base-width for CNN, --depth/--dim for ViT)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BONUS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$BONUS_DIR/data/dataset"
CKPT_BASE="$BONUS_DIR/checkpoints/params"
TEACHER="$(cd "$BONUS_DIR/../project2/checkpoints/run10_final_9k" && pwd)/best_model.pt"

mkdir -p "$CKPT_BASE"

# ---- CNN: sweep --base-width ----
cd "$SCRIPT_DIR/cnn"
declare -A CNN_WIDTHS=([cnn_w8]=8 [cnn_w16]=16 [cnn_w24]=24 [cnn_w32]=32 [cnn_w48]=48 [cnn_w64]=64)
for TAG in cnn_w8 cnn_w16 cnn_w24 cnn_w32 cnn_w48 cnn_w64; do
    W="${CNN_WIDTHS[$TAG]}"
    SAVE="$CKPT_BASE/$TAG"
    echo "=== CNN base_width=$W -> $SAVE ==="
    uv run --project "$BONUS_DIR" python train.py \
        --data-dir "$DATA_DIR" \
        --save-dir "$SAVE" \
        --epochs 30 \
        --warmup-epochs 5 \
        --batch-size 64 \
        --synthetic-n 0 \
        --real-n 0 \
        --base-width "$W" \
        --patience 0 \
        --num-workers 0
done

# ---- ViT: sweep --depth + --dim ----
cd "$SCRIPT_DIR/vit"
# Format: TAG depth dim
VIT_CONFIGS=(
    "vit_d2_dim96  2  96"
    "vit_d4_dim96  4  96"
    "vit_d6_dim96  6  96"
    "vit_d4_dim192 4  192"
    "vit_d6_dim192 6  192"
    "vit_d8_dim192 8  192"
    "vit_d12_dim192 12 192"
)
for CONFIG in "${VIT_CONFIGS[@]}"; do
    read -r TAG DEPTH DIM <<< "$CONFIG"
    SAVE="$CKPT_BASE/$TAG"
    echo "=== ViT depth=$DEPTH dim=$DIM -> $SAVE ==="
    uv run --project "$BONUS_DIR" python train.py \
        --data-dir "$DATA_DIR" \
        --save-dir "$SAVE" \
        --teacher-path "$TEACHER" \
        --epochs 20 \
        --warmup-epochs 3 \
        --batch-size 64 \
        --synthetic-n 0 \
        --real-n 0 \
        --depth "$DEPTH" \
        --dim "$DIM" \
        --heads 3 \
        --repeat-aug 1 \
        --distillation none \
        --patience 0 \
        --num-workers 0
done

echo "=== sweep_params.sh complete ==="
