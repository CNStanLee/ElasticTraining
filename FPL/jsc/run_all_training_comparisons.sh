#!/bin/bash
# run_all_training_comparisons.sh
# ================================
# 批量运行所有一次性剪枝方法的全量训练对比实验。
#
# 在 5 个 eBOPs 预算 × 7 种方法 = 35 个实验中依次运行。
# 每个实验的输出保存在 results/oneshot_train_{target}_{method}/ 下。
#
# 用法:
#   cd FPL/jsc
#   bash run_all_training_comparisons.sh             # 全量运行
#   bash run_all_training_comparisons.sh --dry-run   # 仅打印命令不执行
#
# 可选: 限制方法或目标
#   METHODS="snip grasp" TARGETS="1000 2500" bash run_all_training_comparisons.sh

set -euo pipefail

# ── 配置 ─────────────────────────────────────────────────────────────────────
TARGETS="${TARGETS:-400 1000 1500 2500 6800}"
METHODS="${METHODS:-uniform sensitivity snip grasp synflow spectral_quant snows}"

PHASE1_EPOCHS="${PHASE1_EPOCHS:-5000}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-10000}"
CHECKPOINT="${CHECKPOINT:-results/baseline/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras}"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ── 计数 ─────────────────────────────────────────────────────────────────────
total=$(( $(echo $TARGETS | wc -w) * $(echo $METHODS | wc -w) ))
count=0

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Batch Training Comparison"
echo "  Targets: $TARGETS"
echo "  Methods: $METHODS"
echo "  Total experiments: $total"
echo "  Phase1: ${PHASE1_EPOCHS} epochs  Phase2: ${PHASE2_EPOCHS} epochs"
echo "═══════════════════════════════════════════════════════════════════════"

LOGFILE="results/batch_training_comparison.log"
mkdir -p results
echo "Batch run started at $(date)" > "$LOGFILE"

for target in $TARGETS; do
    for method in $METHODS; do
        count=$((count + 1))
        output_dir="results/oneshot_train_${target}_${method}"

        echo ""
        echo "─────────────────────────────────────────────────────────────────"
        echo "  [$count/$total] $method @ $target eBOPs"
        echo "  Output: $output_dir"
        echo "─────────────────────────────────────────────────────────────────"

        CMD="python compare_oneshot_methods_train.py \
  --prune_method $method \
  --target_ebops $target \
  --checkpoint $CHECKPOINT \
  --phase1_epochs $PHASE1_EPOCHS \
  --phase2_epochs $PHASE2_EPOCHS \
  --output_dir $output_dir"

        if $DRY_RUN; then
            echo "  [DRY-RUN] $CMD"
        else
            echo "  Starting at $(date)" | tee -a "$LOGFILE"
            if eval "$CMD" 2>&1 | tee -a "$LOGFILE"; then
                echo "  ✓ Completed: $method @ $target  ($(date))" | tee -a "$LOGFILE"
            else
                echo "  ✗ FAILED: $method @ $target  ($(date))" | tee -a "$LOGFILE"
                echo "  Continuing to next experiment..."
            fi
        fi
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  All $total experiments $($DRY_RUN && echo 'listed' || echo 'completed')"
echo "  Log: $LOGFILE"
echo "═══════════════════════════════════════════════════════════════════════"
