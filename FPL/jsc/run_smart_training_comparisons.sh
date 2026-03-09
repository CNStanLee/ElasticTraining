#!/bin/bash
# run_smart_training_comparisons.sh
# ===================================
# 基于快速拓扑分析结果，仅运行**可训练**的 method×budget 组合。
# 跳过 400 eBOPs (全部不可训练) 和低预算下不可训练的方法。
#
# 实验清单 (19 个，按预算从低到高):
#
#   1000 eBOPs (2):  synflow, spectral_quant
#   1500 eBOPs (3):  synflow, spectral_quant, uniform
#   2500 eBOPs (7):  synflow, spectral_quant, uniform, sensitivity,
#                    snows, grasp, snip (snip IBR=0.5 但 PRI 可达, 作为对照)
#   6800 eBOPs (7):  全部方法
#
# 用法:
#   cd FPL/jsc
#   nohup bash run_smart_training_comparisons.sh > results/batch_run.log 2>&1 &
#   bash run_smart_training_comparisons.sh --dry-run

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

PHASE1_EPOCHS="${PHASE1_EPOCHS:-5000}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-10000}"
CHECKPOINT="${CHECKPOINT:-results/baseline/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras}"

# ── 定义实验列表 (target method) ─────────────────────────────────────────────
# 基于快速拓扑分析: 仅包含 IBR>=1.0 的组合 + snip@2500(对照)
declare -a EXPERIMENTS=(
    # target=1000: 仅 2 个可训练
    "1000 synflow"
    "1000 spectral_quant"
    # target=1500: 3 个可训练
    "1500 synflow"
    "1500 spectral_quant"
    "1500 uniform"
    # target=2500: 6 个可训练 + 1 个对照
    "2500 spectral_quant"
    "2500 sensitivity"
    "2500 snows"
    "2500 uniform"
    "2500 grasp"
    "2500 synflow"
    "2500 snip"
    # target=6800: 全部 7 个
    "6800 snip"
    "6800 grasp"
    "6800 synflow"
    "6800 spectral_quant"
    "6800 sensitivity"
    "6800 snows"
    "6800 uniform"
)

total=${#EXPERIMENTS[@]}
count=0
passed=0
failed=0

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Smart Training Comparison (trainable combinations only)"
echo "  Total experiments: $total"
echo "  Phase1: ${PHASE1_EPOCHS} epochs  Phase2: ${PHASE2_EPOCHS} epochs"
echo "  Start: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"

LOGFILE="results/smart_batch_training.log"
mkdir -p results
echo "Batch run started at $(date)" > "$LOGFILE"
echo "Total experiments: $total" >> "$LOGFILE"

start_time=$(date +%s)

for exp in "${EXPERIMENTS[@]}"; do
    read -r target method <<< "$exp"
    count=$((count + 1))
    output_dir="results/oneshot_train_${target}_${method}"

    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    echo "  [$count/$total] $method @ $target eBOPs"
    echo "  Output: $output_dir"
    echo "─────────────────────────────────────────────────────────────────"

    # 跳过已完成的实验
    if [[ -f "$output_dir/experiment_meta.json" ]]; then
        echo "  ⏭  SKIP (already completed)"
        echo "  ⏭  SKIP: $method @ $target (already completed)" >> "$LOGFILE"
        passed=$((passed + 1))
        continue
    fi

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
        exp_start=$(date +%s)
        echo "  Starting at $(date)" | tee -a "$LOGFILE"
        if eval "$CMD" 2>&1 | tee -a "$output_dir.log"; then
            exp_end=$(date +%s)
            elapsed=$(( exp_end - exp_start ))
            echo "  ✓ Completed: $method @ $target  (${elapsed}s, $(date))" | tee -a "$LOGFILE"
            passed=$((passed + 1))
        else
            exp_end=$(date +%s)
            elapsed=$(( exp_end - exp_start ))
            echo "  ✗ FAILED: $method @ $target  (${elapsed}s, $(date))" | tee -a "$LOGFILE"
            failed=$((failed + 1))
            echo "  Continuing to next experiment..."
        fi
    fi
done

end_time=$(date +%s)
total_elapsed=$(( end_time - start_time ))

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  All $total experiments $($DRY_RUN && echo 'listed' || echo 'finished')"
echo "  Passed: $passed  Failed: $failed  Skipped: $((total - passed - failed))"
echo "  Total time: ${total_elapsed}s ($(( total_elapsed / 3600 ))h $(( (total_elapsed % 3600) / 60 ))m)"
echo "  Log: $LOGFILE"
echo "═══════════════════════════════════════════════════════════════════════"
echo "Finished at $(date). Passed=$passed Failed=$failed Total_time=${total_elapsed}s" >> "$LOGFILE"
