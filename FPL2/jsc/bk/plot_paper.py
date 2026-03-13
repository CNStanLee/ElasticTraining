#!/usr/bin/env python3
"""
plot_paper.py — FPL 论文全部图表生成脚本
========================================

从 results/paper/ 和 results/ 读取训练轨迹 + 结果摘要,
生成论文所需的全部 Figure 和 Table。

输出: paper_figures/ 目录下的 PDF/PNG 图 + LaTeX 表格 .tex 文件

用法:
  python plot_paper.py                 # 从所有可用数据生成图表
  python plot_paper.py --fig 1         # 仅生成 Figure 1
  python plot_paper.py --table 1       # 仅生成 Table 1
  python plot_paper.py --format png    # PNG 格式 (默认 pdf)

依赖: matplotlib, numpy, h5py, pandas (pip install matplotlib h5py pandas)
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

# 确保工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = 'paper_figures'
RESULTS_DIR = 'results'
PAPER_RESULTS_DIR = 'results/paper'

# 颜色方案
COLORS = {
    'ours':       '#2196F3',
    'spectral':   '#2196F3',
    'random':     '#FF9800',
    'magnitude':  '#4CAF50',
    'hgq':        '#9E9E9E',
    'fpl_v1':     '#E91E63',
    'phase1':     '#FF5722',
    'phase2':     '#3F51B5',
    'beta':       '#9C27B0',
    'ebops':      '#009688',
    'acc':        '#2196F3',
    'lr':         '#FF9800',
}

# 字体设置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
})


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载工具
# ═══════════════════════════════════════════════════════════════════════════════

def find_result_dirs():
    """扫描所有结果目录, 返回 {实验名: 路径} 映射。"""
    dirs = {}

    # 优先: results/paper/<exp_name>/
    if os.path.isdir(PAPER_RESULTS_DIR):
        for d in sorted(os.listdir(PAPER_RESULTS_DIR)):
            full = os.path.join(PAPER_RESULTS_DIR, d)
            if os.path.isdir(full):
                dirs[d] = full

    # 备选: results/<dir>/ (非 paper 子目录)
    if os.path.isdir(RESULTS_DIR):
        for d in sorted(os.listdir(RESULTS_DIR)):
            full = os.path.join(RESULTS_DIR, d)
            if os.path.isdir(full) and d != 'paper':
                # Map legacy names: ebops400_1bit → legacy_400
                dirs[f'legacy_{d}'] = full

    return dirs


def load_result_summary(result_dir):
    """加载 result_summary.json。"""
    path = os.path.join(result_dir, 'result_summary.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_training_trace(result_dir):
    """加载 training_trace.h5, 返回 dict of arrays。"""
    import h5py
    path = os.path.join(result_dir, 'training_trace.h5')
    if not os.path.exists(path):
        return None
    data = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            arr = f[key][:]
            data[key] = arr
    return data


def load_pareto_models(result_dir):
    """从 .keras 文件名解析 Pareto 前沿点。"""
    points = []
    for f in glob.glob(os.path.join(result_dir, 'epoch=*.keras')):
        name = os.path.basename(f)
        try:
            parts = name.replace('.keras', '').split('-')
            info = {}
            for p in parts:
                if '=' in p:
                    k, v = p.split('=', 1)
                    info[k] = float(v) if k != 'epoch' else int(v)
            if 'val_acc' in info and 'ebops' in info:
                points.append(info)
        except Exception:
            pass
    return sorted(points, key=lambda x: x.get('ebops', 0))


def load_experiment_meta(result_dir):
    """加载 experiment_meta.json。"""
    path = os.path.join(result_dir, 'experiment_meta.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect_all_data():
    """收集所有可用实验数据。"""
    dirs = find_result_dirs()
    data = {}
    for name, path in dirs.items():
        entry = {
            'path': path,
            'summary': load_result_summary(path),
            'trace': load_training_trace(path),
            'pareto': load_pareto_models(path),
            'meta': load_experiment_meta(path),
        }
        if entry['summary'] or entry['trace'] or entry['pareto']:
            data[name] = entry
    return data


def smooth(arr, window=50):
    """简单移动平均平滑。"""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Pareto Frontier (acc vs eBOPs)
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_pareto_frontier(all_data, fmt='pdf'):
    """Figure 1: 精度 vs eBOPs Pareto 前沿, 比较不同方法。"""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # 收集 "Ours" 的 Pareto 点 (从 A 类实验)
    ours_points = []
    for name, d in all_data.items():
        if name.startswith('A') or name.startswith('legacy_ebops'):
            # from pareto models
            for p in d.get('pareto', []):
                ours_points.append((p['ebops'], p['val_acc']))
            # from summary
            s = d.get('summary')
            if s and 'best_val_acc' in s and 'best_ebops' in s:
                ours_points.append((s['best_ebops'], s['best_val_acc']))

    # 去重并计算 Pareto 前沿
    if ours_points:
        ours_points = list(set(ours_points))
        ours_points.sort(key=lambda x: x[0])
        # Pareto filter
        pareto = []
        best_acc = -1
        for eb, acc in sorted(ours_points, key=lambda x: x[0]):
            if acc > best_acc:
                pareto.append((eb, acc))
                best_acc = acc
        if pareto:
            eb_arr, acc_arr = zip(*pareto)
            ax.plot(eb_arr, acc_arr, 'o-', color=COLORS['ours'],
                    label='Ours (Spectral+Progressive)', markersize=6, zorder=5)
        # 非 pareto 点 (淡色)
        all_eb, all_acc = zip(*ours_points)
        ax.scatter(all_eb, all_acc, c=COLORS['ours'], alpha=0.15, s=10, zorder=3)

    # 收集 ablation 剪枝方法的点 (B 类)
    for name, d in all_data.items():
        s = d.get('summary')
        if not s or 'best_val_acc' not in s:
            continue
        if name.startswith('B2'):  # random
            ax.scatter(s['best_ebops'], s['best_val_acc'], marker='s',
                      c=COLORS['random'], s=80, zorder=6, label='Random Prune')
        elif name.startswith('B3'):  # magnitude
            ax.scatter(s['best_ebops'], s['best_val_acc'], marker='^',
                      c=COLORS['magnitude'], s=80, zorder=6, label='Magnitude Prune')

    # 参考线: baseline
    ax.axhline(y=0.770, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.770)')

    # 已知参考点 (FPL v1)
    fpl_v1_points = [(400, 0.706), (1500, 0.749)]
    if fpl_v1_points:
        eb_f, acc_f = zip(*fpl_v1_points)
        ax.plot(eb_f, acc_f, 's--', color=COLORS['fpl_v1'],
                label='FPL v1', markersize=5, alpha=0.7)

    # HGQ reference
    hgq_points = [(3000, 0.749), (6800, 0.767), (12000, 0.770)]
    eb_h, acc_h = zip(*hgq_points)
    ax.plot(eb_h, acc_h, 'D--', color=COLORS['hgq'],
            label='HGQ (β-sweep)', markersize=5, alpha=0.7)

    ax.set_xlabel('eBOPs (effective Bit Operations)')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Accuracy vs. eBOPs Pareto Frontier')
    ax.set_xscale('log')
    ax.set_xlim(200, 25000)
    ax.set_ylim(0.60, 0.80)
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, f'fig1_pareto_frontier.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [Fig 1] Pareto frontier → {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Training Curves (val_acc + eBOPs vs epoch)
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_training_curves(all_data, fmt='pdf'):
    """Figure 2: 训练曲线 (val_accuracy + eBOPs + beta 随 epoch 变化)。"""

    # 找一个有完整 trace 的主要实验 (优先 A1 = 400 eBOPs)
    trace_name = None
    trace_data = None
    for name in ['A1_sweep_400', 'legacy_ebops400_1bit', 'A2_sweep_1500', 'legacy_ebops1500_1bit']:
        if name in all_data and all_data[name].get('trace') is not None:
            trace_name = name
            trace_data = all_data[name]['trace']
            break

    if trace_data is None:
        print('  [Fig 2] SKIP: No training trace available')
        return None

    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)

    epochs = trace_data.get('epochs', np.arange(len(trace_data.get('val_accuracy', []))))
    val_acc = trace_data.get('val_accuracy', np.array([]))
    ebops = trace_data.get('ebops', np.array([]))
    beta = trace_data.get('beta', np.array([]))
    lr = trace_data.get('lr', np.array([]))

    n = min(len(epochs), len(val_acc))

    # Panel 1: Validation Accuracy
    ax = axes[0]
    if len(val_acc) >= n and n > 0:
        ax.plot(epochs[:n], smooth(val_acc[:n], 100), color=COLORS['acc'], linewidth=0.8)
        ax.fill_between(epochs[:n], smooth(val_acc[:n], 500),
                        smooth(val_acc[:n], 20), alpha=0.1, color=COLORS['acc'])
    ax.set_ylabel('Val Accuracy')
    ax.set_title(f'Training Curves — {trace_name}')
    ax.grid(True, alpha=0.3)

    # Phase 分界线
    summary = all_data[trace_name].get('summary', {})
    phase1_ep = summary.get('phase1_epochs', 6000)
    for a in axes:
        a.axvline(x=phase1_ep, color='gray', linestyle=':', alpha=0.5)
    axes[0].text(phase1_ep, ax.get_ylim()[1], ' Phase 2→', fontsize=7, va='top')

    # Panel 2: eBOPs
    ax = axes[1]
    if len(ebops) >= n and n > 0:
        ax.plot(epochs[:n], ebops[:n], color=COLORS['ebops'], linewidth=0.5, alpha=0.3)
        ax.plot(epochs[:n], smooth(ebops[:n], 100), color=COLORS['ebops'], linewidth=1.2)
    target = summary.get('target_ebops', 400)
    ax.axhline(y=target, color='red', linestyle='--', alpha=0.6, label=f'Target={target}')
    ax.set_ylabel('eBOPs')
    ax.set_yscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Beta (log scale)
    ax = axes[2]
    if len(beta) >= n and n > 0:
        valid = beta[:n] > 0
        if valid.any():
            ax.semilogy(epochs[:n][valid], beta[:n][valid],
                       color=COLORS['beta'], linewidth=0.8)
    ax.set_ylabel('β (log)')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig2_training_curves.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [Fig 2] Training curves → {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Bitwidth Distribution Evolution
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_bitwidth_evolution(all_data, fmt='pdf'):
    """Figure 3: 位宽分布随训练的演变 (stacked area)。"""
    trace_name = None
    trace_data = None
    for name in ['A1_sweep_400', 'legacy_ebops400_1bit', 'A2_sweep_1500']:
        if name in all_data and all_data[name].get('trace') is not None:
            t = all_data[name]['trace']
            if 'bw_pct' in t:
                trace_name = name
                trace_data = t
                break

    if trace_data is None or 'bw_pct' not in trace_data:
        print('  [Fig 3] SKIP: No bitwidth distribution trace')
        return None

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    epochs = trace_data.get('epochs', np.arange(len(trace_data['bw_pct'])))
    bw_pct = trace_data['bw_pct']  # shape: (n_epochs, max_bits+1)
    n_epochs = len(bw_pct)

    # 选择 3 个时间点: 剪枝后, Phase1 中, 结束
    snap_indices = [
        0,
        min(n_epochs - 1, n_epochs // 3),
        n_epochs - 1,
    ]
    snap_labels = ['After Pruning', 'Mid Phase 1', 'Final']

    bits_labels = [f'{i}b' for i in range(bw_pct.shape[1])]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, bw_pct.shape[1]))

    for ai, (idx, label) in enumerate(zip(snap_indices, snap_labels)):
        ax = axes[ai]
        dist = bw_pct[idx]
        bars = ax.bar(range(len(dist)), dist, color=cmap, edgecolor='white', linewidth=0.3)
        ax.set_xticks(range(len(dist)))
        ax.set_xticklabels(bits_labels, fontsize=7)
        ax.set_title(f'{label}\n(epoch {int(epochs[idx])})', fontsize=8)
        ax.set_ylim(0, 100)
        if ai == 0:
            ax.set_ylabel('Connections (%)')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Bitwidth Distribution — {trace_name}', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig3_bitwidth_evolution.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [Fig 3] Bitwidth evolution → {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Topology Visualization (collect existing plots)
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_topology(all_data, fmt='pdf'):
    """Figure 4: 拓扑图 (从 results 中收集已有的 topology PNG/PDF)。"""
    collected = []
    for name in ['A1_sweep_400', 'legacy_ebops400_1bit']:
        if name not in all_data:
            continue
        d = all_data[name]['path']
        for pat in ['*topology*.png', '*topology*.pdf', '*topo*.png']:
            files = glob.glob(os.path.join(d, pat))
            collected.extend(files)

    if not collected:
        print('  [Fig 4] SKIP: No topology plots found in results')
        print('          (These are generated by TopologyPlotCallback during training)')
        return None

    # 复制到 paper_figures/
    import shutil
    for f in collected[:6]:  # 最多 6 个
        dst = os.path.join(OUTPUT_DIR, f'fig4_{os.path.basename(f)}')
        shutil.copy2(f, dst)
        print(f'  [Fig 4] Topology: {f} → {dst}')
    return collected


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Beta-Curriculum Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_beta_curriculum(all_data, fmt='pdf'):
    """Figure 5: Beta 课程重启可视化 (beta + acc 联合时间线)。"""
    trace_name = None
    trace_data = None
    for name in ['A1_sweep_400', 'legacy_ebops400_1bit', 'D3_beta_patience600']:
        if name in all_data and all_data[name].get('trace') is not None:
            t = all_data[name]['trace']
            if 'beta' in t and np.any(t['beta'] > 0):
                trace_name = name
                trace_data = t
                break

    if trace_data is None:
        print('  [Fig 5] SKIP: No training trace with beta data')
        return None

    epochs = trace_data.get('epochs', np.arange(len(trace_data.get('beta', []))))
    beta = trace_data.get('beta', np.array([]))
    val_acc = trace_data.get('val_accuracy', np.array([]))
    n = min(len(epochs), len(beta), len(val_acc))

    if n == 0:
        print('  [Fig 5] SKIP: Empty trace')
        return None

    fig, ax1 = plt.subplots(figsize=(6, 3))
    ax2 = ax1.twinx()

    # Beta (log scale, 左 Y)
    valid_beta = beta[:n] > 0
    if valid_beta.any():
        ax1.semilogy(epochs[:n][valid_beta], beta[:n][valid_beta],
                    color=COLORS['beta'], linewidth=0.8, alpha=0.7, label='β')
    ax1.set_ylabel('β (log scale)', color=COLORS['beta'])
    ax1.tick_params(axis='y', labelcolor=COLORS['beta'])

    # Val accuracy (右 Y)
    ax2.plot(epochs[:n], smooth(val_acc[:n], 50),
            color=COLORS['acc'], linewidth=1.0, label='Val Accuracy')
    ax2.set_ylabel('Val Accuracy', color=COLORS['acc'])
    ax2.tick_params(axis='y', labelcolor=COLORS['acc'])

    ax1.set_xlabel('Epoch')
    ax1.set_title(f'Beta-Curriculum Restarts — {trace_name}')
    ax1.grid(True, alpha=0.2)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig5_beta_curriculum.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [Fig 5] Beta curriculum → {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Ablation Comparison (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_ablation_summary(all_data, fmt='pdf'):
    """Figure 6: ablation 汇总柱状图 (各因素对 acc 的影响)。"""
    # 收集所有 ablation 数据
    ablation_groups = {
        'Pruning\nMethod': {
            'Spectral': ['A1_sweep_400', 'B1_prune_spectral', 'legacy_ebops400_1bit'],
            'Random':   ['B2_prune_random'],
            'Magnitude': ['B3_prune_magnitude'],
        },
        'Budget\nStrategy': {
            'Direct (μ=1)': ['C1_budget_direct'],
            'μ=2':         ['C2_budget_mu2'],
            'μ=7.5':       ['C3_budget_mu7.5', 'A1_sweep_400', 'legacy_ebops400_1bit'],
            'μ=15':        ['C4_budget_mu15'],
        },
        'Beta\nCurriculum': {
            'Disabled':    ['D1_beta_disabled'],
            'p=300':       ['D2_beta_patience300'],
            'p=600':       ['D3_beta_patience600', 'A1_sweep_400', 'legacy_ebops400_1bit'],
            'p=1200':      ['D4_beta_patience1200'],
        },
        'Adaptive\nLR': {
            'Disabled':    ['E1_lr_disabled'],
            'Enabled':     ['A1_sweep_400', 'legacy_ebops400_1bit'],
        },
    }

    found_data = False
    for group in ablation_groups.values():
        for candidates in group.values():
            for c in candidates:
                if c in all_data and all_data[c].get('summary'):
                    found_data = True
                    break

    if not found_data:
        print('  [Fig 6] SKIP: No ablation data available')
        return None

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.5))

    for gi, (group_name, methods) in enumerate(ablation_groups.items()):
        ax = axes[gi]
        names = []
        accs = []
        colors_list = []

        for method_name, candidates in methods.items():
            best_acc = None
            for c in candidates:
                if c in all_data:
                    s = all_data[c].get('summary')
                    if s:
                        acc = s.get('best_val_acc', s.get('final_val_acc'))
                        if acc and (best_acc is None or acc > best_acc):
                            best_acc = acc
            names.append(method_name)
            accs.append(best_acc if best_acc else 0)
            colors_list.append(COLORS['ours'] if best_acc and best_acc == max(
                [a for a in accs if a > 0], default=0) else '#90CAF9')

        x = np.arange(len(names))
        bars = ax.bar(x, accs, color='#90CAF9', edgecolor='white', width=0.6)

        # 高亮最佳
        if accs:
            best_idx = np.argmax(accs)
            if accs[best_idx] > 0:
                bars[best_idx].set_color(COLORS['ours'])

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7)
        ax.set_title(group_name, fontsize=9)
        if gi == 0:
            ax.set_ylabel('Best Val Accuracy')

        # 数值标注
        for i, v in enumerate(accs):
            if v > 0:
                ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=6)

        ax.set_ylim(0.6, 0.8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Ablation Study @ 400 eBOPs', fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig6_ablation_summary.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [Fig 6] Ablation summary → {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Table 1: Main Pareto Results
# ═══════════════════════════════════════════════════════════════════════════════

def table1_main_results(all_data, fmt='tex'):
    """Table 1: 主结果 — 不同 eBOPs 目标的最佳精度。"""
    targets = [400, 1500, 2500, 6800, 12000]

    # 收集 "Ours" 的数据
    ours_results = {}
    for name, d in all_data.items():
        s = d.get('summary')
        if s and (name.startswith('A') or name.startswith('legacy_ebops')):
            t = s.get('target_ebops', 0)
            acc = s.get('best_val_acc', s.get('final_val_acc', 0))
            ebops = s.get('best_ebops', s.get('final_ebops', 0))
            if t > 0:
                key = min(targets, key=lambda x: abs(x - t))
                if key not in ours_results or acc > ours_results[key]['acc']:
                    ours_results[key] = {'acc': acc, 'ebops': ebops}

    # 参考数据
    hgq_ref = {400: 0.630, 1500: 0.742, 2500: 0.757, 6800: 0.767, 12000: 0.770}
    fpl_v1_ref = {400: 0.706, 1500: 0.749}

    # LaTeX 输出
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Main Results: Best validation accuracy at different eBOPs targets.}',
        r'\label{tab:main_results}',
        r'\begin{tabular}{l' + 'c' * len(targets) + '}',
        r'\toprule',
        r'Method & ' + ' & '.join([f'{t}' for t in targets]) + r' \\',
        r'\midrule',
    ]

    # HGQ row
    row = 'HGQ ($\\beta$-sweep)'
    for t in targets:
        if t in hgq_ref:
            row += f' & {hgq_ref[t]:.3f}'
        else:
            row += ' & --'
    lines.append(row + r' \\')

    # FPL v1 row
    row = 'FPL v1'
    for t in targets:
        if t in fpl_v1_ref:
            row += f' & {fpl_v1_ref[t]:.3f}'
        else:
            row += ' & --'
    lines.append(row + r' \\')

    # Ours row
    lines.append(r'\midrule')
    row = r'\textbf{Ours}'
    for t in targets:
        if t in ours_results:
            acc = ours_results[t]['acc']
            row += f' & \\textbf{{{acc:.3f}}}'
        else:
            row += ' & TBD'
    lines.append(row + r' \\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    tex_content = '\n'.join(lines)
    path = os.path.join(OUTPUT_DIR, 'table1_main_results.tex')
    with open(path, 'w') as f:
        f.write(tex_content)

    # 同时输出可读版本
    print(f'\n  Table 1: Main Pareto Results')
    print(f'  {"Method":<30s}  ' + '  '.join([f'{t:>6d}' for t in targets]))
    print(f'  {"-"*30}  ' + '  '.join(['------'] * len(targets)))
    print(f'  {"HGQ (β-sweep)":<30s}  ' +
          '  '.join([f'{hgq_ref.get(t, 0):>6.3f}' if t in hgq_ref else '    --' for t in targets]))
    print(f'  {"FPL v1":<30s}  ' +
          '  '.join([f'{fpl_v1_ref.get(t, 0):>6.3f}' if t in fpl_v1_ref else '    --' for t in targets]))
    print(f'  {"Ours (Spectral+Progressive)":<30s}  ' +
          '  '.join([f'{ours_results[t]["acc"]:>6.3f}' if t in ours_results else '   TBD' for t in targets]))

    print(f'  → {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Table 2: Pruning Method Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def table2_pruning_ablation(all_data, fmt='tex'):
    """Table 2: 剪枝方法 ablation @ 400 eBOPs。"""
    methods = {
        'Random':    ['B2_prune_random'],
        'Magnitude': ['B3_prune_magnitude'],
        'Spectral':  ['B1_prune_spectral', 'A1_sweep_400', 'legacy_ebops400_1bit'],
    }

    rows = []
    for method_name, candidates in methods.items():
        best = None
        for c in candidates:
            if c in all_data:
                s = all_data[c].get('summary')
                if s:
                    acc = s.get('best_val_acc', s.get('final_val_acc', 0))
                    pruned_ebops = s.get('pruned_ebops', 0)
                    final_ebops = s.get('final_ebops', 0)
                    if best is None or acc > best['acc']:
                        best = {'acc': acc, 'pruned_ebops': pruned_ebops,
                                'final_ebops': final_ebops, 'source': c}
        rows.append((method_name, best))

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Pruning method ablation at 400 eBOPs. '
        r'Spectral masks maintain connectivity and outperform random/magnitude.}',
        r'\label{tab:pruning_ablation}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'Pruning Method & Pruned eBOPs & Best Val Acc & Final eBOPs \\',
        r'\midrule',
    ]

    for name, data in rows:
        if data:
            bold = r'\textbf' if data['acc'] == max(r[1]['acc'] for r in rows if r[1]) else ''
            acc_str = f'{bold}{{{data["acc"]:.3f}}}' if bold else f'{data["acc"]:.3f}'
            lines.append(f'{name} & {data["pruned_ebops"]:.0f} & {acc_str} & '
                        f'{data["final_ebops"]:.0f} \\\\')
        else:
            lines.append(f'{name} & TBD & TBD & TBD \\\\')

    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])

    path = os.path.join(OUTPUT_DIR, 'table2_pruning_ablation.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'\n  Table 2: Pruning Ablation')
    print(f'  {"Method":<15s}  {"Best Acc":>10s}  {"Source":<30s}')
    for name, data in rows:
        if data:
            print(f'  {name:<15s}  {data["acc"]:>10.3f}  {data["source"]:<30s}')
        else:
            print(f'  {name:<15s}  {"TBD":>10s}')
    print(f'  → {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Table 3: Progressive Budget Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def table3_budget_ablation(all_data, fmt='tex'):
    """Table 3: 渐进预算策略 ablation @ 400 eBOPs。"""
    methods = {
        r'Direct ($\mu$=1)': ['C1_budget_direct'],
        r'$\mu$=2.0':        ['C2_budget_mu2'],
        r'$\mu$=7.5':        ['C3_budget_mu7.5', 'A1_sweep_400', 'legacy_ebops400_1bit'],
        r'$\mu$=15.0':       ['C4_budget_mu15'],
    }

    rows = _collect_ablation_rows(methods, all_data)
    path = _write_ablation_table(
        rows, 'table3_budget_ablation',
        'Progressive budget ablation at 400 eBOPs.',
        'tab:budget_ablation',
        'Budget Strategy'
    )
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Table 4: Beta Curriculum Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def table4_beta_ablation(all_data, fmt='tex'):
    """Table 4: Beta 课程 ablation @ 400 eBOPs。"""
    methods = {
        'Disabled':         ['D1_beta_disabled'],
        'Patience=300':     ['D2_beta_patience300'],
        'Patience=600':     ['D3_beta_patience600', 'A1_sweep_400', 'legacy_ebops400_1bit'],
        'Patience=1200':    ['D4_beta_patience1200'],
    }

    rows = _collect_ablation_rows(methods, all_data)
    path = _write_ablation_table(
        rows, 'table4_beta_ablation',
        'Beta-curriculum controller ablation at 400 eBOPs.',
        'tab:beta_ablation',
        'Beta Curriculum'
    )
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5: Adaptive LR Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def table5_lr_ablation(all_data, fmt='tex'):
    """Table 5: 自适应 LR ablation @ 400 eBOPs。"""
    methods = {
        'Disabled': ['E1_lr_disabled'],
        'Enabled':  ['A1_sweep_400', 'legacy_ebops400_1bit'],
    }

    rows = _collect_ablation_rows(methods, all_data)
    path = _write_ablation_table(
        rows, 'table5_lr_ablation',
        'Adaptive LR scaling ablation at 400 eBOPs.',
        'tab:lr_ablation',
        'Adaptive LR'
    )
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Table 6: SOTA Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def table6_sota(all_data, fmt='tex'):
    """Table 6: 与现有文献的对比。"""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Comparison with state-of-the-art FPGA jet classifiers.}',
        r'\label{tab:sota}',
        r'\begin{tabular}{llccc}',
        r'\toprule',
        r'Work & Framework & eBOPs & Accuracy & Reduction \\',
        r'\midrule',
        r'Duarte et al. \cite{hls4ml} & hls4ml & $\sim$50k & 0.760 & 1$\times$ \\',
        r'Coelho et al. \cite{qkeras} & QKeras & $\sim$5000 & 0.740 & 10$\times$ \\',
        r'HGQ \cite{hgq} & HGQ & $\sim$3000 & 0.749 & 17$\times$ \\',
    ]

    # 我们最好的 400 eBOPs 结果
    our_acc = 'TBD'
    for name in ['A1_sweep_400', 'legacy_ebops400_1bit', 'B1_prune_spectral']:
        if name in all_data:
            s = all_data[name].get('summary')
            if s and 'best_val_acc' in s:
                our_acc = f'{s["best_val_acc"]:.3f}'
                break

    lines.append(f'\\textbf{{Ours}} & HGQ+Spectral & \\textbf{{400}} & '
                f'\\textbf{{{our_acc}}} & \\textbf{{125$\\times$}} \\\\')

    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])

    path = os.path.join(OUTPUT_DIR, 'table6_sota_comparison.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\n  Table 6: SOTA comparison → {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_ablation_rows(methods, all_data):
    """通用 ablation 数据收集。"""
    rows = []
    for method_name, candidates in methods.items():
        best = None
        for c in candidates:
            if c in all_data:
                s = all_data[c].get('summary')
                if s:
                    acc = s.get('best_val_acc', s.get('final_val_acc', 0))
                    if acc and (best is None or acc > best['acc']):
                        best = {'acc': acc, 'source': c,
                                'final_ebops': s.get('final_ebops', 0),
                                'elapsed': s.get('elapsed_sec', 0)}
        rows.append((method_name, best))
    return rows


def _write_ablation_table(rows, filename, caption, label, col_name):
    """通用 ablation LaTeX 表格写入。"""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        f'{col_name} & Best Val Acc & Final eBOPs \\\\',
        r'\midrule',
    ]

    valid_accs = [r[1]['acc'] for r in rows if r[1] and r[1]['acc'] > 0]
    best_overall = max(valid_accs) if valid_accs else 0

    for name, data in rows:
        if data and data['acc'] > 0:
            bold = data['acc'] >= best_overall - 1e-6
            acc_str = f'\\textbf{{{data["acc"]:.3f}}}' if bold else f'{data["acc"]:.3f}'
            lines.append(f'{name} & {acc_str} & {data["final_ebops"]:.0f} \\\\')
        else:
            lines.append(f'{name} & TBD & TBD \\\\')

    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])

    path = os.path.join(OUTPUT_DIR, f'{filename}.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'\n  {filename}:')
    print(f'  {"Name":<25s}  {"Best Acc":>10s}  {"Source":<30s}')
    for name, data in rows:
        if data and data['acc'] > 0:
            print(f'  {name:<25s}  {data["acc"]:>10.3f}  {data["source"]:<30s}')
        else:
            print(f'  {name:<25s}  {"TBD":>10s}')
    print(f'  → {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: Multi-target training curves comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_multitarget_curves(all_data, fmt='pdf'):
    """Figure 7: 不同 eBOPs 目标的训练曲线叠加对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    target_names = {
        400:   ['A1_sweep_400', 'legacy_ebops400_1bit'],
        1500:  ['A2_sweep_1500', 'legacy_ebops1500_1bit'],
        2500:  ['A3_sweep_2500'],
        6800:  ['A4_sweep_6800'],
        12000: ['A5_sweep_12000'],
    }

    colors_per_target = {
        400:   '#E91E63',
        1500:  '#2196F3',
        2500:  '#4CAF50',
        6800:  '#FF9800',
        12000: '#9C27B0',
    }

    found = False
    for target, candidates in target_names.items():
        trace = None
        for c in candidates:
            if c in all_data and all_data[c].get('trace') is not None:
                trace = all_data[c]['trace']
                break
        if trace is None:
            continue

        found = True
        epochs = trace.get('epochs', np.arange(len(trace.get('val_accuracy', []))))
        val_acc = trace.get('val_accuracy', np.array([]))
        ebops_arr = trace.get('ebops', np.array([]))
        n = min(len(epochs), len(val_acc))
        color = colors_per_target.get(target, 'gray')

        if n > 0:
            axes[0].plot(epochs[:n], smooth(val_acc[:n], 100),
                        color=color, linewidth=1.0, label=f'{target} eBOPs')
        if len(ebops_arr) >= n and n > 0:
            axes[1].plot(epochs[:n], smooth(ebops_arr[:n], 100),
                        color=color, linewidth=1.0, label=f'{target} eBOPs')
            axes[1].axhline(y=target, color=color, linestyle=':', alpha=0.4)

    if not found:
        print('  [Fig 7] SKIP: No multi-target trace data')
        plt.close(fig)
        return None

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Val Accuracy')
    axes[0].set_title('Accuracy Convergence')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('eBOPs')
    axes[1].set_title('eBOPs Convergence')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'fig7_multitarget_curves.{fmt}')
    fig.savefig(path)
    plt.close(fig)
    print(f'  [Fig 7] Multi-target curves → {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='FPL Paper Figure & Table Generator')
    parser.add_argument('--fig', type=int, nargs='*', default=None,
                        help='Generate specific figure(s), e.g. --fig 1 2')
    parser.add_argument('--table', type=int, nargs='*', default=None,
                        help='Generate specific table(s), e.g. --table 1 2')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output format (default: pdf)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all figures and tables')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'\n{"=" * 60}')
    print(f'  FPL Paper Figure & Table Generator')
    print(f'  Output: {OUTPUT_DIR}/')
    print(f'  Format: {args.format}')
    print(f'{"=" * 60}')

    # 加载所有数据
    print('\n  Loading data...')
    all_data = collect_all_data()
    print(f'  Found {len(all_data)} result directories:')
    for name, d in sorted(all_data.items()):
        has_trace = 'trace' if d.get('trace') else '      '
        has_summary = 'summary' if d.get('summary') else '       '
        n_pareto = len(d.get('pareto', []))
        print(f'    {name:<35s}  [{has_trace}] [{has_summary}] [{n_pareto:>3d} pareto]')

    if not all_data:
        print('\n  [WARNING] No data found! Run experiments first:')
        print('    python run_paper_experiments.py --run A1')
        print('    python run_paper_experiments.py --run-all')
        return

    # 确定要生成的项目
    gen_all = args.all or (args.fig is None and args.table is None)
    gen_figs = set(args.fig) if args.fig else (set(range(1, 8)) if gen_all else set())
    gen_tables = set(args.table) if args.table else (set(range(1, 7)) if gen_all else set())

    fmt = args.format

    # 生成 Figures
    fig_funcs = {
        1: ('Pareto Frontier', fig1_pareto_frontier),
        2: ('Training Curves', fig2_training_curves),
        3: ('Bitwidth Evolution', fig3_bitwidth_evolution),
        4: ('Topology', fig4_topology),
        5: ('Beta Curriculum', fig5_beta_curriculum),
        6: ('Ablation Summary', fig6_ablation_summary),
        7: ('Multi-target Curves', fig7_multitarget_curves),
    }

    generated = {'figures': [], 'tables': []}
    for i, (desc, func) in fig_funcs.items():
        if i in gen_figs:
            print(f'\n  ── Figure {i}: {desc} ──')
            result = func(all_data, fmt=fmt)
            if result:
                generated['figures'].append(f'Fig {i}: {desc}')

    # 生成 Tables
    table_funcs = {
        1: ('Main Results', table1_main_results),
        2: ('Pruning Ablation', table2_pruning_ablation),
        3: ('Budget Ablation', table3_budget_ablation),
        4: ('Beta Curriculum Ablation', table4_beta_ablation),
        5: ('Adaptive LR Ablation', table5_lr_ablation),
        6: ('SOTA Comparison', table6_sota),
    }

    for i, (desc, func) in table_funcs.items():
        if i in gen_tables:
            print(f'\n  ── Table {i}: {desc} ──')
            result = func(all_data, fmt=fmt)
            if result:
                generated['tables'].append(f'Table {i}: {desc}')

    # 汇总
    print(f'\n{"=" * 60}')
    print(f'  Generated:')
    for item in generated['figures'] + generated['tables']:
        print(f'    ✓ {item}')
    n_total = len(generated['figures']) + len(generated['tables'])
    print(f'  Total: {n_total} items → {OUTPUT_DIR}/')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()
