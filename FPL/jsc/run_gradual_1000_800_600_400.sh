#!/usr/bin/env bash
# Gradual step-down compression: 1000 → 800 → 600 → 400 eBOPs
#
# Strategy:
#   - Each stage starts from the best checkpoint of the previous stage
#   - --spectral_revival + --swap_kill: revives dead connections while
#     simultaneously killing the weakest alive ones, keeping eBOPs budget stable
#   - Lower-budget stages get more epochs and more aggressive revival params
#
# Stage 1 (1000→800): may already be done; skipped if checkpoint exists
# Stage 2 (800→600):  swap_kill, default revival params
# Stage 3 (600→400):  swap_kill, 2× revival rate, extended Phase 1

PYTHON=/home/changhong/anaconda3/envs/py312tf/bin/python
SCRIPT=compare_oneshot_methods_train.py
SQ1000_CKPT="results/sq_1000/epoch=12001-val_acc=0.748-ebops=999-val_loss=0.744.keras"

# ── Helper: pick best checkpoint by val_acc, constrained by max eBOPs ────
# Usage: best_ckpt <dir> [max_ebops]
best_ckpt() {
    local dir="$1"
    local max_ebops="${2:-999999}"
    ls "$dir"/*.keras 2>/dev/null \
        | awk -F'ebops=' -v max="$max_ebops" '
            {
                # extract numeric eBOPs from filename
                split($2, a, "-")
                ebops = a[1]+0
                if (ebops <= max) print $0
            }' \
        | awk -F'val_acc=' '{print $2, $0}' \
        | sort -rn | head -1 | awk '{print $2}'
}

cd "$(dirname "$0")"

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 : 1000 → 800 eBOPs
# ══════════════════════════════════════════════════════════════════════════
CKPT_800=$(best_ckpt results/gradual_1000to800 959)  # < 1.6×600 → near_budget_preserve for Stage 2
if [[ -z "$CKPT_800" ]]; then
    # Fallback: any checkpoint
    CKPT_800=$(best_ckpt results/gradual_1000to800)
fi
if [[ -n "$CKPT_800" ]]; then
    echo "  [STAGE 1] Skipping — best checkpoint found: $CKPT_800"
else
    echo "============================================================"
    echo "  STAGE 1 : 1000 → 800 eBOPs"
    echo "  Source  : $SQ1000_CKPT"
    echo "============================================================"
    mkdir -p results/gradual_1000to800
    $PYTHON $SCRIPT \
        --prune_method spectral_quant \
        --target_ebops 800 \
        --checkpoint "$SQ1000_CKPT" \
        --output_dir results/gradual_1000to800 \
        --spectral_revival \
        --swap_kill \
        2>&1 | tee results/gradual_1000to800_train.log
    CKPT_800=$(best_ckpt results/gradual_1000to800)
fi

if [[ -z "$CKPT_800" ]]; then
    echo "ERROR: No checkpoint in results/gradual_1000to800/" >&2; exit 1
fi
echo ""
echo "  Stage 1 done → $CKPT_800"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 : 800 → 600 eBOPs
# ══════════════════════════════════════════════════════════════════════════
CKPT_600=$(best_ckpt results/gradual_800to600 639)  # for skip check
if [[ -z "$CKPT_600" ]]; then
    CKPT_600=$(best_ckpt results/gradual_800to600)  # fallback: any eBOPs
fi
if [[ -n "$CKPT_600" ]]; then
    echo "  [STAGE 2] Skipping — best checkpoint found: $CKPT_600"
else
    echo "============================================================"
    echo "  STAGE 2 : 800 → 600 eBOPs"
    echo "  Source  : $CKPT_800"
    echo "============================================================"
    mkdir -p results/gradual_800to600
    $PYTHON $SCRIPT \
        --prune_method spectral_quant \
        --target_ebops 600 \
        --checkpoint "$CKPT_800" \
        --output_dir results/gradual_800to600 \
        --spectral_revival \
        --swap_kill \
        --revival_interval 200 \
        --revival_max_per_layer 12 \
        2>&1 | tee results/gradual_800to600_train.log
    CKPT_600=$(best_ckpt results/gradual_800to600 639)
    if [[ -z "$CKPT_600" ]]; then
        CKPT_600=$(best_ckpt results/gradual_800to600)
    fi
fi

if [[ -z "$CKPT_600" ]]; then
    echo "ERROR: No checkpoint in results/gradual_800to600/" >&2; exit 1
fi
echo ""
echo "  Stage 2 done → $CKPT_600"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 : ~550 → 400 eBOPs
# Key fixes vs failed attempts:
#   - revival_b_val=1.2: cheap revivals, no eBOPs explosion
#   - phase1_beta_max=4e-4: 2× default (NOT 10×); keeps ~5-6% connections
#     alive (vs 2.3% with 2e-3). More capacity → better accuracy.
#   - NO swap_kill: prevents topology thrashing after compression is done.
#     Natural beta pressure kills weakest connections organically.
#   - revival_interval=400: less churn, more stable topology learning
# ══════════════════════════════════════════════════════════════════════════
# Use lowest-eBOPs Stage 2 checkpoint (≤560) to minimise compression gap
CKPT_600_S3=$(best_ckpt results/gradual_800to600 560)
if [[ -z "$CKPT_600_S3" ]]; then
    CKPT_600_S3="$CKPT_600"   # fallback to any Stage 2 best
fi
echo "  Stage 3 source checkpoint: $CKPT_600_S3"

CKPT_400_EXISTING=$(best_ckpt results/gradual_600to400 440)
if [[ -n "$CKPT_400_EXISTING" ]]; then
    echo "  [STAGE 3] Skipping — best checkpoint found: $CKPT_400_EXISTING"
else
    echo "============================================================"
    echo "  STAGE 3 : ~550 → 400 eBOPs"
    echo "  Source  : $CKPT_600_S3"
    echo "============================================================"
    mkdir -p results/gradual_600to400
    $PYTHON $SCRIPT \
        --prune_method spectral_quant \
        --target_ebops 400 \
        --checkpoint "$CKPT_600_S3" \
        --output_dir results/gradual_600to400 \
        --spectral_revival \
        --no-swap_kill \
        --revival_interval 400 \
        --revival_max_per_layer 8 \
        --revival_b_val 1.2 \
        --phase1_beta_max 4e-4 \
        --phase1_epochs 12000 \
        2>&1 | tee results/gradual_600to400_train.log
fi

CKPT_400=$(best_ckpt results/gradual_600to400)
echo ""
echo "============================================================"
echo "  DONE"
echo "  800 eBOPs best : $CKPT_800"
echo "  600 eBOPs best : $CKPT_600"
echo "  400 eBOPs best : $CKPT_400"
echo "============================================================"
