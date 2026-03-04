#!/usr/bin/env bash
# =============================================================================
# 消融实验脚本  —  target_ebops = 400
# =============================================================================
#
# 消融设计（留一法 + 全关基线）
# ──────────────────────────────────────────────────────────────────────────────
#  编号  实验名称              禁用模块（相比 full 去掉一个）
#   A0   full                  无（全部启用，参考上界）
#   A1   wo_topo_warmup        --no_topo_warmup
#   A2   wo_spectral_reg       --no_spectral_reg
#   A3   wo_rewire             --no_rewiring
#   A4   wo_adaptive_lr        --no_adaptive_lr
#   A5   wo_progressive        --no_progressive
#   A6   wo_beta_curriculum    --no_beta_curriculum
#   A7   baseline              全部禁用（参考下界）
# ──────────────────────────────────────────────────────────────────────────────
#
# 使用方式
#   bash run_ablation_400.sh          # 顺序运行全部 8 个实验
#   bash run_ablation_400.sh A1 A3    # 仅运行指定实验编号
#
# 输出目录结构
#   results/ablation_400/
#     full/              training_trace.h5, *.keras, train.log
#     wo_topo_warmup/
#     wo_spectral_reg/
#     wo_rewire/
#     wo_adaptive_lr/
#     wo_progressive/
#     wo_beta_curriculum/
#     baseline/
#     summary.tsv        # 汇总各实验最佳 val_acc 与最终 ebops
# =============================================================================

set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/changhong/anaconda3/envs/py312tf/bin/python"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_run_low_budget.py"
BASE_OUT="${SCRIPT_DIR}/results/ablation_400"

# ── 共享超参（固定，不参与消融）─────────────────────────────────────────────
TARGET=400
WARMUP=2000
P1_EP=6000
P2_EP=12000
CKPT="results/baseline/epoch=2236-val_acc=0.770-ebops=23589-val_loss=0.641.keras"

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')] $*${NC}"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $*${NC}"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $*${NC}"; }

# ── 实验定义  {id : "name|extra_flags"} ──────────────────────────────────────
declare -A EXP_NAME
declare -A EXP_FLAGS

EXP_NAME[A0]="full"
EXP_FLAGS[A0]=""

EXP_NAME[A1]="wo_topo_warmup"
EXP_FLAGS[A1]="--no_topo_warmup"

EXP_NAME[A2]="wo_spectral_reg"
EXP_FLAGS[A2]="--no_spectral_reg"

EXP_NAME[A3]="wo_rewire"
EXP_FLAGS[A3]="--no_rewiring"

EXP_NAME[A4]="wo_adaptive_lr"
EXP_FLAGS[A4]="--no_adaptive_lr"

EXP_NAME[A5]="wo_progressive"
EXP_FLAGS[A5]="--no_progressive"

EXP_NAME[A6]="wo_beta_curriculum"
EXP_FLAGS[A6]="--no_beta_curriculum"

EXP_NAME[A7]="baseline"
EXP_FLAGS[A7]="--no_topo_warmup --no_spectral_reg --no_rewiring --no_adaptive_lr --no_progressive --no_beta_curriculum"

# ── 确定本次运行哪些实验 ──────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    RUN_IDS=(A0 A1 A2 A3 A4 A5 A6 A7)
else
    RUN_IDS=("$@")
fi

mkdir -p "${BASE_OUT}"
SUMMARY="${BASE_OUT}/summary.tsv"
echo -e "exp_id\tname\tbest_val_acc_in_budget\tebops_of_best\tn_ckpts_in_budget\tstatus" > "${SUMMARY}"

# ── 从 Pareto 检查点文件名提取预算达标的最高 val_acc ─────────────────────────
# 文件名格式: epoch=XXX-val_acc=0.XXX-ebops=YYY-val_loss=0.XXX.keras
# 仅统计 ebops <= TARGET * BUDGET_RATIO 的检查点
BUDGET_RATIO="1.20"   # 允许超出目标 20%（容忍短暂波动）

parse_best_in_budget() {
    local out_dir="$1"
    local budget_limit
    budget_limit=$(echo "${TARGET} * ${BUDGET_RATIO}" | bc | cut -d. -f1)

    local best_acc="N/A"
    local best_ebops="N/A"
    local n_ckpts=0

    # 遍历目录下所有 .keras 文件（Pareto 快照）
    while IFS= read -r ckpt; do
        fname=$(basename "${ckpt}")
        # 提取 val_acc 和 ebops 字段
        acc=$(echo "${fname}"  | grep -oP 'val_acc=\K[0-9.]+' || true)
        ops=$(echo "${fname}"  | grep -oP 'ebops=\K[0-9]+'    || true)
        [[ -z "${acc}" || -z "${ops}" ]] && continue

        # 过滤：仅保留 ebops 在预算内的检查点
        if [ "${ops}" -le "${budget_limit}" ] 2>/dev/null; then
            n_ckpts=$((n_ckpts + 1))
            # 比较浮点数（用 awk）
            if [ "${best_acc}" = "N/A" ] || \
               awk "BEGIN{exit !(${acc} > ${best_acc})}"; then
                best_acc="${acc}"
                best_ebops="${ops}"
            fi
        fi
    done < <(find "${out_dir}" -maxdepth 1 -name "*.keras" 2>/dev/null | sort)

    echo "${best_acc}|${best_ebops}|${n_ckpts}"
}

# ── 主循环 ────────────────────────────────────────────────────────────────────
TOTAL=${#RUN_IDS[@]}
IDX=0

for EID in "${RUN_IDS[@]}"; do
    IDX=$((IDX + 1))
    NAME="${EXP_NAME[$EID]}"
    FLAGS="${EXP_FLAGS[$EID]}"
    OUT_DIR="${BASE_OUT}/${NAME}"
    LOG="${OUT_DIR}/train.log"

    mkdir -p "${OUT_DIR}"

    log "═════════════════════════════════════════════════════════════════"
    log "[${IDX}/${TOTAL}]  ${EID}  ─  ${NAME}"
    log "  flags    : ${FLAGS:-<none>}"
    log "  output   : ${OUT_DIR}"
    log "═════════════════════════════════════════════════════════════════"

    # 构建命令（output_folder 由脚本内部生成，需要用 --checkpoint 和手动覆盖
    # 我们通过修改环境变量传递输出目录，然后在命令行 patch 结果目录）
    CMD="${PYTHON} ${TRAIN_SCRIPT} \
        --target_ebops ${TARGET} \
        --warmup_ebops ${WARMUP} \
        --phase1_epochs ${P1_EP} \
        --phase2_epochs ${P2_EP} \
        --checkpoint ${CKPT} \
        ${FLAGS}"

    # 训练脚本根据 target_ebops 自动创建 results/low_budget_400/，
    # 我们在脚本运行后将其移动到消融专用目录
    SCRIPT_OUT_DIR="${SCRIPT_DIR}/results/low_budget_${TARGET}"

    # 若上一次实验残留了同名目录，先备份
    if [ -d "${SCRIPT_OUT_DIR}" ]; then
        mv "${SCRIPT_OUT_DIR}" "${SCRIPT_OUT_DIR}_bak_$(date +%s)" 2>/dev/null || true
    fi

    STATUS="success"
    if ! (cd "${SCRIPT_DIR}" && eval ${CMD} 2>&1 | tee "${LOG}"); then
        STATUS="failed"
        fail "${EID} (${NAME}) 训练失败，查看日志: ${LOG}"
    fi

    # 将输出目录移动到消融专用位置
    if [ -d "${SCRIPT_OUT_DIR}" ]; then
        # 合并（防止文件冲突）
        cp -rn "${SCRIPT_OUT_DIR}/." "${OUT_DIR}/" 2>/dev/null || true
        rm -rf "${SCRIPT_OUT_DIR}"
    fi

    # 解析指标写入汇总
    if [ "${STATUS}" = "success" ] && [ -d "${OUT_DIR}" ]; then
        METRICS=$(parse_best_in_budget "${OUT_DIR}")
        BEST_ACC=$(echo "${METRICS}"  | cut -d'|' -f1)
        BEST_OPS=$(echo "${METRICS}"  | cut -d'|' -f2)
        N_CKPTS=$(echo  "${METRICS}"  | cut -d'|' -f3)
    else
        BEST_ACC="N/A"; BEST_OPS="N/A"; N_CKPTS="0"
    fi

    echo -e "${EID}\t${NAME}\t${BEST_ACC}\t${BEST_OPS}\t${N_CKPTS}\t${STATUS}" >> "${SUMMARY}"
    ok "${EID} (${NAME}) 完成   best_val_acc(in_budget)=${BEST_ACC}  ebops=${BEST_OPS}  n_ckpts=${N_CKPTS}"
done

# ── 打印汇总表 ────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  消融实验汇总  (target_ebops = ${TARGET}, budget_limit = TARGET×${BUDGET_RATIO})"
echo "  指标：预算达标检查点中最高 val_acc"
echo "══════════════════════════════════════════════════════════"
column -t -s $'\t' "${SUMMARY}"
echo ""
echo "  详细日志目录: ${BASE_OUT}"
echo "══════════════════════════════════════════════════════════"
