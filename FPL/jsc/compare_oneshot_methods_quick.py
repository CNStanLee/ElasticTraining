#!/usr/bin/env python3
"""
compare_oneshot_methods_quick.py
================================
在 [400, 1000, 1500, 2500, 6800] eBOPs 预算下，快速对比先进的一次性剪枝方法。

对比方法:
  1. magnitude (uniform)   — 按幅度均匀缩放位宽
  2. sensitivity           — 按层敏感度分配预算
  3. SNIP                  — 单次连接敏感度 (Lee et al., ICLR 2019)
  4. GraSP                 — 梯度信号保持 (Wang et al., ICLR 2020)
  5. SynFlow               — 数据无关迭代剪枝 (Tanaka et al., NeurIPS 2020)
  6. spectral_quant        — 谱/拓扑友好剪枝 (本项目)
  7. SNOWS                 — 二阶近似表征重建 (本项目)

评估指标 (来自 compare_spectral_vs_natural_topology_v2):
  - 结构秩 r_l^s (structural rank)
  - IBR (信息瓶颈秩比)  — 可训练性判据
  - QIT_l (量化信息吞吐量)
  - PRI (Pareto 可达指数)  — 可达前沿判据
  - 剪枝后精度 (无训练)
  - 有效容量 C_eff, 有效路径 P_eff
  - 谱间隙, Ramanujan 判定, 条件数
  - 各层度统计, 活跃连接数

使用方式:
  cd FPL/jsc
  python compare_oneshot_methods_quick.py

输出:
  results/oneshot_methods_comparison/
    ├── quick_comparison_summary.log
    ├── quick_metrics_table.csv
    ├── quick_comparison_chart.png
    └── topology_target{ebops}_{method}_*.png
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 保证相对路径基于脚本目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
import csv
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import keras
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.data import get_data
import model.model as _   # noqa: F401  # 注册自定义层

# ── 复用已有基础设施 ─────────────────────────────────────────────────────────
from compare_spectral_vs_natural_topology_v2 import (
    analyze_model,
    ModelMetrics,
    LayerMetrics,
    K_CLASSES,
    _print_metrics,
)
from run_one_shot_prune_only import (
    compute_model_ebops,
    bisect_ebops_to_target,
    spectral_quant_prune_to_ebops,
    saliency_prune_to_ebops,
    snows_prune_to_ebops,
    build_sample_input,
    _dense_prunable_layers,
)
from utils.ramanujan_budget_utils import (
    HighBitPruner,
    SensitivityAwarePruner,
)

np.random.seed(42)
tf.random.set_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_CKPT = "results/baseline/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras"
INPUT_H5 = "data/dataset.h5"

TARGET_EBOPS_LIST = [400, 1000, 1500, 2500, 6800]

# 待对比方法 (按先进程度排序)
METHODS = [
    'uniform',           # 1. 基本幅度剪枝 (均匀缩放位宽)
    'sensitivity',       # 2. 按层敏感度分配
    'snip',              # 3. SNIP (连接敏感度)
    'grasp',             # 4. GraSP (梯度信号保持)
    'synflow',           # 5. SynFlow (数据无关)
    'spectral_quant',    # 6. 谱/拓扑友好剪枝
    'snows',             # 7. SNOWS (二阶表征重建)
]

# 方法显示名称
METHOD_LABELS = {
    'uniform':        'Magnitude (Uniform)',
    'sensitivity':    'Sensitivity',
    'snip':           'SNIP',
    'grasp':          'GraSP',
    'synflow':        'SynFlow',
    'spectral_quant': 'Spectral-Quant',
    'snows':          'SNOWS',
}

# ══════════════════════════════════════════════════════════════════════════════
# 命令行参数
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Quick comparison of one-shot pruning methods')
parser.add_argument('--checkpoint', type=str, default=BASELINE_CKPT,
                    help='Baseline checkpoint path')
parser.add_argument('--targets', type=float, nargs='+', default=TARGET_EBOPS_LIST,
                    help='List of target eBOPs budgets')
parser.add_argument('--methods', type=str, nargs='+', default=METHODS,
                    choices=METHODS + ['all'],
                    help='Methods to compare')
parser.add_argument('--output_dir', type=str, default='results/oneshot_methods_comparison',
                    help='Output directory')
parser.add_argument('--input_h5', type=str, default=INPUT_H5)
parser.add_argument('--sample_size', type=int, default=512)
parser.add_argument('--no_plot', action='store_true', help='Skip topology plots')
parser.add_argument('--no_calibrate', action='store_true',
                    help='Skip bisection calibration after pruning')
args, _ = parser.parse_known_args()

if 'all' in args.methods:
    args.methods = METHODS

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 日志设置
# ══════════════════════════════════════════════════════════════════════════════

log_path = OUTPUT_DIR / "quick_comparison_summary.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 剪枝执行器
# ══════════════════════════════════════════════════════════════════════════════

def prune_model(method: str, model, target_ebops: float,
                sample_input, teacher_model=None) -> dict:
    """对模型应用指定的一次性剪枝方法，返回剪枝信息。"""
    current_ebops = compute_model_ebops(model, sample_input)

    info = {'method': method, 'target_ebops': target_ebops}

    if method == 'uniform':
        pruner = HighBitPruner(target_ebops=target_ebops, pruned_threshold=0.1)
        pruner.prune_to_ebops(model, current_ebops=current_ebops, verbose=True)

    elif method == 'sensitivity':
        pruner = SensitivityAwarePruner(
            target_ebops=target_ebops, pruned_threshold=0.1, b_k_min=0.3)
        pruner.prune_to_ebops(model, current_ebops=current_ebops, verbose=True)

    elif method in ('snip', 'grasp', 'synflow'):
        _, report = saliency_prune_to_ebops(
            model, target_ebops=target_ebops,
            sample_input=sample_input, method=method,
            input_h5=args.input_h5, sample_size=args.sample_size,
            b_floor=0.35, verbose=True,
        )
        info['saliency_report'] = report

    elif method == 'spectral_quant':
        _, used_structured = spectral_quant_prune_to_ebops(
            model, target_ebops=target_ebops,
            sample_input=sample_input,
            min_degree=2, b_floor=0.25, b_ceiling=6.0,
            verbose=True,
        )
        info['used_structured'] = used_structured

    elif method == 'snows':
        if teacher_model is None:
            raise ValueError("SNOWS requires a teacher model")
        _, snows_report = snows_prune_to_ebops(
            model, teacher_model=teacher_model,
            target_ebops=target_ebops,
            sample_input=sample_input,
            init_method='sensitivity',
            b_floor=0.30, k_step=2, newton_steps=2,
            verbose=True,
        )
        info['snows_report'] = snows_report

    else:
        raise ValueError(f"Unknown method: {method}")

    # 二分校准
    if not args.no_calibrate:
        preserve = (method == 'spectral_quant' and info.get('used_structured', False))
        calibrated = bisect_ebops_to_target(
            model, target_ebops=target_ebops,
            sample_input=sample_input,
            tolerance=0.05, max_iter=30,
            allow_connection_kill=not preserve,
        )
        info['calibrated_ebops'] = calibrated

    info['final_ebops'] = compute_model_ebops(model, sample_input)
    return info


# ══════════════════════════════════════════════════════════════════════════════
# 汇总表输出
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(all_metrics: list[ModelMetrics]):
    """打印多方法多预算汇总对比表。"""
    tau_reach = (K_CLASSES - 1) / 2.0

    logger.info(f"\n{'═'*160}")
    logger.info("                                  一次性剪枝方法 快速对比汇总表")
    logger.info(f"{'═'*160}")
    logger.info(f"  判据: IBR = min(r_l^s)/2 ≥ 1.0 → 可训练;  "
                f"PRI = min(QIT_l)/{tau_reach:.1f} ≥ 1.0 → 可达前沿;  K={K_CLASSES}")
    logger.info(f"{'─'*160}")

    header = (
        f"  {'target':>7}  {'method':<22}  {'eBOPs':>7}  {'acc':>7}  "
        f"{'min_r':>5}  {'IBR':>6}  {'min_QIT':>8}  {'PRI':>6}  "
        f"{'C_eff':>7}  {'P_eff':>10}  "
        f"{'层结构秩':^20}  {'train?':<14}  {'reach?':<12}"
    )
    logger.info(header)
    logger.info(f"{'─'*160}")

    targets_seen = sorted(set(m.target_ebops for m in all_metrics))
    for t in targets_seen:
        group = [m for m in all_metrics if m.target_ebops == t]
        # 按 IBR 降序排，同 IBR 按 acc 降序
        group.sort(key=lambda m: (-m.IBR, -m.val_acc))
        for m in group:
            layer_ranks = [f"{lm.structural_rank}" for lm in m.layers]
            ranks_str = "[" + ",".join(layer_ranks) + "]"
            row = (
                f"  {m.target_ebops:>7}  {METHOD_LABELS.get(m.method, m.method):<22}  "
                f"{m.measured_ebops:>7.0f}  {m.val_acc:>7.4f}  "
                f"{m.min_structural_rank:>5}  {m.IBR:>6.3f}  "
                f"{m.min_QIT:>8.3f}  {m.PRI:>6.3f}  "
                f"{m.C_eff:>7.1f}  {m.P_eff:>10.2e}  "
                f"{ranks_str:<20}  {m.trainable_verdict:<14}  {m.reachable_verdict:<12}"
            )
            logger.info(row)
        logger.info(f"{'─'*160}")


def print_ranking_table(all_metrics: list[ModelMetrics]):
    """打印排名汇总表 — 横向比较方法。"""
    targets_seen = sorted(set(m.target_ebops for m in all_metrics))
    methods_seen = list(dict.fromkeys(m.method for m in all_metrics))

    logger.info(f"\n{'═'*140}")
    logger.info("                              方法排名汇总 (按 IBR, 高 = 好)")
    logger.info(f"{'═'*140}")

    # IBR 排名表
    header = f"  {'method':<22}"
    for t in targets_seen:
        header += f"  {t:>7}"
    logger.info(header)
    logger.info(f"{'─'*140}")

    for method in methods_seen:
        row = f"  {METHOD_LABELS.get(method, method):<22}"
        for t in targets_seen:
            m = next((x for x in all_metrics if x.method == method and x.target_ebops == t), None)
            if m:
                verdict_mark = "✓" if m.IBR >= 1.0 else "✗"
                row += f"  {m.IBR:>5.2f}{verdict_mark:>2}"
            else:
                row += f"  {'N/A':>7}"
        logger.info(row)

    logger.info(f"\n{'═'*140}")
    logger.info("                              方法排名汇总 (按 PRI, 高 = 好)")
    logger.info(f"{'═'*140}")

    header = f"  {'method':<22}"
    for t in targets_seen:
        header += f"  {t:>7}"
    logger.info(header)
    logger.info(f"{'─'*140}")

    for method in methods_seen:
        row = f"  {METHOD_LABELS.get(method, method):<22}"
        for t in targets_seen:
            m = next((x for x in all_metrics if x.method == method and x.target_ebops == t), None)
            if m:
                verdict_mark = "✓" if m.PRI >= 1.0 else "✗"
                row += f"  {m.PRI:>5.2f}{verdict_mark:>2}"
            else:
                row += f"  {'N/A':>7}"
        logger.info(row)

    logger.info(f"\n{'═'*140}")
    logger.info("                              方法排名汇总 (按 剪枝后精度, 高 = 好)")
    logger.info(f"{'═'*140}")

    header = f"  {'method':<22}"
    for t in targets_seen:
        header += f"  {t:>7}"
    logger.info(header)
    logger.info(f"{'─'*140}")

    for method in methods_seen:
        row = f"  {METHOD_LABELS.get(method, method):<22}"
        for t in targets_seen:
            m = next((x for x in all_metrics if x.method == method and x.target_ebops == t), None)
            if m:
                row += f"  {m.val_acc:>7.4f}"
            else:
                row += f"  {'N/A':>7}"
        logger.info(row)


def print_detailed_per_target(all_metrics: list[ModelMetrics]):
    """逐预算逐方法打印关键指标。"""
    targets_seen = sorted(set(m.target_ebops for m in all_metrics))

    for t in targets_seen:
        group = [m for m in all_metrics if m.target_ebops == t]
        group.sort(key=lambda m: (-m.IBR, -m.PRI))

        logger.info(f"\n{'═'*100}")
        logger.info(f"  TARGET eBOPs = {t}   ({len(group)} 方法)")
        logger.info(f"{'═'*100}")

        best_ibr = max(m.IBR for m in group)
        best_pri = max(m.PRI for m in group)
        best_acc = max(m.val_acc for m in group)

        for m in group:
            ibr_flag = " ★" if m.IBR == best_ibr else ""
            pri_flag = " ★" if m.PRI == best_pri else ""
            acc_flag = " ★" if m.val_acc == best_acc else ""

            logger.info(f"\n  ── {METHOD_LABELS.get(m.method, m.method)} ──")
            logger.info(f"    eBOPs:         {m.measured_ebops:.0f}")
            logger.info(f"    剪枝后精度:    {m.val_acc:.4f}{acc_flag}")
            logger.info(f"    IBR:           {m.IBR:.3f}  → {m.trainable_verdict}{ibr_flag}")
            logger.info(f"    PRI:           {m.PRI:.3f}  → {m.reachable_verdict}{pri_flag}")
            logger.info(f"    min结构秩:     {m.min_structural_rank}  (瓶颈: {m.bottleneck_layer_ibr})")
            logger.info(f"    min_QIT:       {m.min_QIT:.3f}  (瓶颈: {m.bottleneck_layer_pri})")
            logger.info(f"    C_eff:         {m.C_eff:.1f}")
            logger.info(f"    P_eff:         {m.P_eff:.2e}")

            for lm in m.layers:
                logger.info(
                    f"      {lm.name:15s} ({lm.shape[0]:>3}→{lm.shape[1]:>3})  "
                    f"active={lm.n_active:>5}/{lm.n_total}  "
                    f"r_s={lm.structural_rank:>3}  "
                    f"QIT={lm.QIT:>6.2f}  "
                    f"deg_col={lm.col_degree_mean:>5.1f}  "
                    f"σ_min={lm.sigma_min:.3e}  κ={lm.condition_number:>8.1f}  "
                    f"Ram={'✓' if lm.is_ramanujan else '✗'}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# CSV 保存
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(all_metrics: list[ModelMetrics], csv_path: Path):
    """保存全量指标到 CSV。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    model_fields = [
        'target_ebops', 'method', 'method_label', 'measured_ebops',
        'val_acc', 'val_loss',
        'min_structural_rank', 'IBR', 'bottleneck_layer_ibr',
        'min_QIT', 'PRI', 'bottleneck_layer_pri',
        'trainable_verdict', 'reachable_verdict',
        'C_eff', 'P_eff', 'rank_product', 'cheeger_product',
    ]
    layer_fields = [
        'n_active', 'sparsity', 'structural_rank', 'QIT',
        'n_active_outputs', 'n_active_inputs',
        'col_degree_mean', 'col_degree_std',
        'spectral_gap', 'lambda_2_adj',
        'ramanujan_bound', 'is_ramanujan', 'cheeger_lower',
        'sigma_min', 'sigma_max', 'condition_number',
        'effective_rank', 'numerical_rank',
        'mean_bk_active', 'std_bk_active', 'n_dead_quant',
    ]

    max_layers = max(len(m.layers) for m in all_metrics) if all_metrics else 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = model_fields[:]
        for i in range(max_layers):
            header.extend([f"L{i}_{field}" for field in layer_fields])
        w.writerow(header)

        for m in all_metrics:
            row = [
                m.target_ebops, m.method, METHOD_LABELS.get(m.method, m.method),
                f'{m.measured_ebops:.2f}',
                f'{m.val_acc:.6f}', f'{m.val_loss:.6f}',
                m.min_structural_rank, f'{m.IBR:.4f}', m.bottleneck_layer_ibr,
                f'{m.min_QIT:.4f}', f'{m.PRI:.4f}', m.bottleneck_layer_pri,
                m.trainable_verdict, m.reachable_verdict,
                f'{m.C_eff:.2f}', f'{m.P_eff:.6e}',
                f'{m.rank_product:.8f}', f'{m.cheeger_product:.8e}',
            ]
            for lm in m.layers:
                row.extend([
                    lm.n_active, f'{lm.sparsity:.6f}',
                    lm.structural_rank, f'{lm.QIT:.4f}',
                    lm.n_active_outputs, lm.n_active_inputs,
                    f'{lm.col_degree_mean:.4f}', f'{lm.col_degree_std:.4f}',
                    f'{lm.spectral_gap:.6f}', f'{lm.lambda_2_adj:.6f}',
                    f'{lm.ramanujan_bound:.4f}', int(lm.is_ramanujan),
                    f'{lm.cheeger_lower:.6f}',
                    f'{lm.sigma_min:.6e}', f'{lm.sigma_max:.6e}',
                    f'{lm.condition_number:.4f}',
                    f'{lm.effective_rank:.4f}', lm.numerical_rank,
                    f'{lm.mean_bk_active:.6f}', f'{lm.std_bk_active:.6f}',
                    lm.n_dead_quant,
                ])
            for _ in range(max_layers - len(m.layers)):
                row.extend([''] * len(layer_fields))
            w.writerow(row)

    logger.info(f"  CSV 已保存: {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_chart(all_metrics: list[ModelMetrics], save_path: Path):
    """绘制多面板对比图: IBR / PRI / Accuracy 随 eBOPs 变化。"""
    targets_seen = sorted(set(m.target_ebops for m in all_metrics))
    methods_seen = list(dict.fromkeys(m.method for m in all_metrics))

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_seen)))
    color_map = {m: c for m, c in zip(methods_seen, colors)}
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h']
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(methods_seen)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('One-Shot Pruning Methods: Quick Topology Comparison', fontsize=14, fontweight='bold')

    # Panel 1: IBR
    ax = axes[0, 0]
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, label='Trainable threshold')
    for method in methods_seen:
        xs, ys = [], []
        for t in targets_seen:
            m = next((x for x in all_metrics if x.method == method and x.target_ebops == t), None)
            if m:
                xs.append(t)
                ys.append(m.IBR)
        if xs:
            ax.plot(xs, ys, marker=marker_map[method], color=color_map[method],
                    label=METHOD_LABELS.get(method, method), linewidth=1.5, markersize=7)
    ax.set_xlabel('Target eBOPs')
    ax.set_ylabel('IBR (≥1.0 = Trainable)')
    ax.set_title('Information Bottleneck Ratio (IBR)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Panel 2: PRI
    ax = axes[0, 1]
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, label='Reachable threshold')
    for method in methods_seen:
        xs, ys = [], []
        for t in targets_seen:
            m = next((x for x in all_metrics if x.method == method and x.target_ebops == t), None)
            if m:
                xs.append(t)
                ys.append(m.PRI)
        if xs:
            ax.plot(xs, ys, marker=marker_map[method], color=color_map[method],
                    label=METHOD_LABELS.get(method, method), linewidth=1.5, markersize=7)
    ax.set_xlabel('Target eBOPs')
    ax.set_ylabel('PRI (≥1.0 = Reachable)')
    ax.set_title('Pareto Reachability Index (PRI)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Panel 3: Post-prune Accuracy  (无训练)
    ax = axes[1, 0]
    for method in methods_seen:
        xs, ys = [], []
        for t in targets_seen:
            m = next((x for x in all_metrics if x.method == method and x.target_ebops == t), None)
            if m:
                xs.append(t)
                ys.append(m.val_acc)
        if xs:
            ax.plot(xs, ys, marker=marker_map[method], color=color_map[method],
                    label=METHOD_LABELS.get(method, method), linewidth=1.5, markersize=7)
    ax.set_xlabel('Target eBOPs')
    ax.set_ylabel('Val Accuracy (post-prune, no training)')
    ax.set_title('Post-Prune Accuracy')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Panel 4: Min Structural Rank
    ax = axes[1, 1]
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.6, label='Trainable min rank=2')
    for method in methods_seen:
        xs, ys = [], []
        for t in targets_seen:
            m = next((x for x in all_metrics if x.method == method and x.target_ebops == t), None)
            if m:
                xs.append(t)
                ys.append(m.min_structural_rank)
        if xs:
            ax.plot(xs, ys, marker=marker_map[method], color=color_map[method],
                    label=METHOD_LABELS.get(method, method), linewidth=1.5, markersize=7)
    ax.set_xlabel('Target eBOPs')
    ax.set_ylabel('min_l(structural_rank)')
    ax.set_title('Minimum Structural Rank')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  对比图已保存: {save_path}")


def plot_heatmap(all_metrics: list[ModelMetrics], save_path: Path):
    """绘制方法×预算 热力图 (IBR / PRI)。"""
    targets_seen = sorted(set(m.target_ebops for m in all_metrics))
    methods_seen = list(dict.fromkeys(m.method for m in all_metrics))

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(methods_seen) * 0.8)))
    fig.suptitle('Method × Budget Heatmap', fontsize=13, fontweight='bold')

    for ax_idx, (metric_name, title) in enumerate([('IBR', 'IBR (≥1.0=Trainable)'),
                                                     ('PRI', 'PRI (≥1.0=Reachable)')]):
        ax = axes[ax_idx]
        matrix = np.full((len(methods_seen), len(targets_seen)), np.nan)
        for mi, method in enumerate(methods_seen):
            for ti, t in enumerate(targets_seen):
                m = next((x for x in all_metrics if x.method == method and x.target_ebops == t), None)
                if m:
                    matrix[mi, ti] = getattr(m, metric_name)

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=2.5)
        ax.set_xticks(range(len(targets_seen)))
        ax.set_xticklabels([str(int(t)) for t in targets_seen], fontsize=9)
        ax.set_yticks(range(len(methods_seen)))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods_seen], fontsize=9)
        ax.set_xlabel('Target eBOPs')
        ax.set_title(title)

        # 标注数值
        for mi in range(len(methods_seen)):
            for ti in range(len(targets_seen)):
                val = matrix[mi, ti]
                if np.isfinite(val):
                    color = 'white' if val < 0.5 or val > 2.0 else 'black'
                    mark = '✓' if val >= 1.0 else '✗'
                    ax.text(ti, mi, f'{val:.2f}\n{mark}', ha='center', va='center',
                            fontsize=8, color=color, fontweight='bold')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  热力图已保存: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("═" * 80)
    logger.info("  一次性剪枝方法快速对比实验 (Quick Topology Comparison)")
    logger.info("═" * 80)

    logger.info(f"\n  基线模型: {args.checkpoint}")
    logger.info(f"  目标预算: {args.targets}")
    logger.info(f"  对比方法: {args.methods}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")

    tau_reach = (K_CLASSES - 1) / 2.0
    logger.info(f"\n  理论判据:")
    logger.info(f"    IBR = min_l(r_l^s) / 2 ≥ 1.0 → 可训练  (τ_train=2)")
    logger.info(f"    PRI = min_l(QIT_l) / {tau_reach:.1f} ≥ 1.0 → 可达 Pareto 前沿  (τ_reach={tau_reach:.1f})")

    # ── 加载数据 ─────────────────────────────────────────────────────────
    logger.info("\n加载数据...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(args.input_h5, src='openml')
    sample_input = tf.constant(X_val[:args.sample_size], dtype=tf.float32)
    x_eval = tf.constant(X_val[:4096], dtype=tf.float32)
    y_eval = tf.constant(y_val[:4096], dtype=tf.int32)

    # ── 加载 teacher 模型 (SNOWS 需要) ───────────────────────────────────
    teacher_model = None
    if 'snows' in args.methods:
        teacher_model = keras.models.load_model(args.checkpoint, compile=False)

    # ── 逐预算逐方法执行 ─────────────────────────────────────────────────
    all_metrics: list[ModelMetrics] = []
    total_experiments = len(args.targets) * len(args.methods)
    exp_count = 0

    for target_ebops in sorted(args.targets):
        logger.info(f"\n{'='*80}")
        logger.info(f"  TARGET eBOPs = {int(target_ebops)}")
        logger.info(f"{'='*80}")

        for method in args.methods:
            exp_count += 1
            logger.info(f"\n  [{exp_count}/{total_experiments}] "
                        f"{METHOD_LABELS.get(method, method)} @ {int(target_ebops)} eBOPs")
            logger.info(f"  {'─'*60}")

            try:
                # 每次从头加载模型，确保独立性
                model = keras.models.load_model(args.checkpoint, compile=False)

                # 执行剪枝
                prune_info = prune_model(
                    method, model, float(target_ebops),
                    sample_input, teacher_model=teacher_model,
                )

                # 分析拓扑指标
                metrics = analyze_model(
                    model, sample_input, method, int(target_ebops),
                    source_path=args.checkpoint,
                    x_val=x_eval, y_val=y_eval,
                )
                all_metrics.append(metrics)

                # 简要结果
                logger.info(
                    f"    → eBOPs={metrics.measured_ebops:.0f}  "
                    f"acc={metrics.val_acc:.4f}  "
                    f"IBR={metrics.IBR:.3f} ({metrics.trainable_verdict})  "
                    f"PRI={metrics.PRI:.3f} ({metrics.reachable_verdict})  "
                    f"ranks={[lm.structural_rank for lm in metrics.layers]}"
                )

            except Exception as exc:
                logger.error(f"    ✗ 失败: {exc}")
                import traceback
                traceback.print_exc()

            finally:
                # 释放内存
                try:
                    del model
                except NameError:
                    pass
                keras.backend.clear_session()

    if not all_metrics:
        logger.error("没有成功的实验结果!")
        return

    # ── 汇总输出 ─────────────────────────────────────────────────────────
    logger.info(f"\n\n{'█'*80}")
    logger.info("                         结 果 汇 总")
    logger.info(f"{'█'*80}")

    print_summary_table(all_metrics)
    print_ranking_table(all_metrics)
    print_detailed_per_target(all_metrics)

    # ── 保存 CSV ─────────────────────────────────────────────────────────
    save_csv(all_metrics, OUTPUT_DIR / "quick_metrics_table.csv")

    # ── 绘图 ─────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_comparison_chart(all_metrics, OUTPUT_DIR / "quick_comparison_chart.png")
        plot_heatmap(all_metrics, OUTPUT_DIR / "quick_comparison_heatmap.png")

    # ── 建议 ─────────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*80}")
    logger.info("                         结 论 与 建 议")
    logger.info(f"{'═'*80}")

    targets_seen = sorted(set(m.target_ebops for m in all_metrics))
    for t in targets_seen:
        group = [m for m in all_metrics if m.target_ebops == t]

        # 找最佳方法
        trainable = [m for m in group if m.IBR >= 1.0]
        reachable = [m for m in group if m.PRI >= 1.0]

        logger.info(f"\n  target={int(t)} eBOPs:")
        if reachable:
            best = max(reachable, key=lambda m: (m.PRI, m.val_acc))
            logger.info(f"    ★ 最佳可达方法: {METHOD_LABELS.get(best.method, best.method)}  "
                        f"(IBR={best.IBR:.2f}, PRI={best.PRI:.2f}, acc={best.val_acc:.4f})")
        elif trainable:
            best = max(trainable, key=lambda m: (m.IBR, m.val_acc))
            logger.info(f"    ▲ 可训练但不确定可达: {METHOD_LABELS.get(best.method, best.method)}  "
                        f"(IBR={best.IBR:.2f}, PRI={best.PRI:.2f})")
            logger.info(f"    → 建议膨胀预算或调整方法")
        else:
            logger.info(f"    ✗ 所有方法均不可训练 (IBR < 1.0)")
            best_ibr = max(group, key=lambda m: m.IBR)
            logger.info(f"    → 最高 IBR: {METHOD_LABELS.get(best_ibr.method, best_ibr.method)} "
                        f"(IBR={best_ibr.IBR:.2f})")
            logger.info(f"    → 建议增加 eBOPs 预算")

        n_trainable = len(trainable)
        n_total = len(group)
        logger.info(f"    可训练方法: {n_trainable}/{n_total}  "
                    f"可达方法: {len(reachable)}/{n_total}")

    logger.info(f"\n  所有结果已保存到: {OUTPUT_DIR}")
    logger.info(f"  满意后运行全量训练对比: python compare_oneshot_methods_train.py")
    logger.info("完成。")


if __name__ == '__main__':
    main()
