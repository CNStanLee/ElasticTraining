#!/usr/bin/env bash
# Gradual step-down compression: 1000 → 800 → 600 eBOPs
# Each stage: full training (Phase1=5000 + Phase2=10000 epochs)
# Stage N+1 starts from the best checkpoint of Stage N

PYTHON=/home/changhong/anaconda3/envs/py312tf/bin/python
SCRIPT=compare_oneshot_methods_train.py
SQ1000_CKPT="results/sq_1000/epoch=12001-val_acc=0.748-ebops=999-val_loss=0.744.keras"

cd "$(dirname "$0")"

# ── Stage 1 : 1000 → 800 ──────────────────────────────────────────────────
echo "============================================================"
echo "  STAGE 1 : 1000 → 800 eBOPs"
echo "  Source  : $SQ1000_CKPT"
echo "============================================================"

$PYTHON $SCRIPT \
    --prune_method spectral_quant \
    --target_ebops 800 \
    --checkpoint "$SQ1000_CKPT" \
    --output_dir results/gradual_1000to800 \
    --spectral_revival \
    2>&1 | tee results/gradual_1000to800_train.log

# Find best checkpoint from stage 1 (highest val_acc)
BEST_800=$(ls results/gradual_1000to800/*.keras 2>/dev/null \
    | awk -F'val_acc=' '{print $2, $0}' \
    | sort -rn | head -1 | awk '{print $2}')

if [[ -z "$BEST_800" ]]; then
    echo "ERROR: No checkpoint found in results/gradual_1000to800/" >&2
    exit 1
fi

echo ""
echo "  Stage 1 best checkpoint: $BEST_800"
echo ""

# ── Stage 2 : 800 → 600 ──────────────────────────────────────────────────
echo "============================================================"
echo "  STAGE 2 : 800 → 600 eBOPs"
echo "  Source  : $BEST_800"
echo "============================================================"

$PYTHON $SCRIPT \
    --prune_method spectral_quant \
    --target_ebops 600 \
    --checkpoint "$BEST_800" \
    --output_dir results/gradual_800to600 \
    --spectral_revival \
    2>&1 | tee results/gradual_800to600_train.log

BEST_600=$(ls results/gradual_800to600/*.keras 2>/dev/null \
    | awk -F'val_acc=' '{print $2, $0}' \
    | sort -rn | head -1 | awk '{print $2}')

echo ""
echo "============================================================"
echo "  DONE"
echo "  800 eBOPs best : $BEST_800"
echo "  600 eBOPs best : $BEST_600"
echo "============================================================"
