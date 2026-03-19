#!/usr/bin/env bash
# Plot 2: sweep MLP head width for CLIP and DINO downstream MLPs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BONUS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_BASE="$BONUS_DIR/checkpoints/params"
TRAIN_PY="$SCRIPT_DIR/downstream/train_scaling.py"

WIDTHS=(8 16 32 64 128 256 512)

mkdir -p "$CKPT_BASE"

# ---- CLIP: h2 = h1 // 2 ----
for H1 in "${WIDTHS[@]}"; do
    H2=$((H1 / 2))
    TAG="clip_h${H1}"
    SAVE="$CKPT_BASE/$TAG"
    echo "=== CLIP h1=$H1 h2=$H2 -> $SAVE ==="
    uv run --project "$BONUS_DIR" python "$TRAIN_PY" \
        --model clip \
        --real-n 0 \
        --h1 "$H1" \
        --h2 "$H2" \
        --lr 1e-3 \
        --dropout 0.3 \
        --mixup True \
        --save-dir "$SAVE"
done

# ---- DINO: h2 = h1 // 2 ----
for H1 in "${WIDTHS[@]}"; do
    H2=$((H1 / 2))
    TAG="dino_h${H1}"
    SAVE="$CKPT_BASE/$TAG"
    echo "=== DINO h1=$H1 h2=$H2 -> $SAVE ==="
    uv run --project "$BONUS_DIR" python "$TRAIN_PY" \
        --model dino_concat \
        --real-n 0 \
        --h1 "$H1" \
        --h2 "$H2" \
        --lr 1e-3 \
        --dropout 0.2 \
        --mixup True \
        --save-dir "$SAVE"
done

echo "=== sweep_params_downstream.sh complete ==="
