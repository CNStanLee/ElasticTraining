#!/usr/bin/env python3
"""绘制最小拉马努金图（所有活跃连接 1-bit）并标注 eBOPs。

不做训练，不做校准——纯粹展示最小 Ramanujan 拓扑在 1-bit 量化下的结构和代价。
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data.data import get_data
import model.model as model_module  # noqa: F401
from model.model import get_model_hgq
from hgq.layers import QLayerBase
from utils.tf_device import get_tf_device
from utils.ramanujan_budget_utils import (
    _flatten_layers,
    _get_kq_var,
    apply_ramanujan_bw_init,
    compute_bw_aware_degree,
)
from utils.topology_graph_plot_utils import TopologyGraphPlotter, LayerGraphData


def compute_model_ebops(mdl, sample_input) -> float:
    from keras import ops
    bn_layers, old_momentum = [], []
    for layer in _flatten_layers(mdl):
        if isinstance(layer, keras.layers.BatchNormalization):
            bn_layers.append(layer)
            old_momentum.append(float(layer.momentum))
            layer.momentum = 1.0
    try:
        mdl(sample_input, training=True)
    finally:
        for layer, m in zip(bn_layers, old_momentum):
            layer.momentum = m
    total = 0
    for layer in mdl._flatten_layers():
        if isinstance(layer, QLayerBase) and layer.enable_ebops and layer._ebops is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def main():
    device = get_tf_device()
    output_dir = Path('results/min_ramanujan_1bit')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 数据（只要 sample 计算 eBOPs）────────────────────────────────────
    (X_train, _), _, _ = get_data('data/dataset.h5', src='openml')
    _sample = tf.constant(X_train[:2048], dtype=tf.float32)

    # ── 2. 模型 ─────────────────────────────────────────────────────────────
    model = get_model_hgq(init_bw_k=3, init_bw_a=3)

    # ── 3. 最小 Ramanujan 度 ─────────────────────────────────────────────────
    # 用一个极大的 target 来获取按谱条件计算的最小度（multiplier=1.5）
    per_layer_degree, _ = compute_bw_aware_degree(
        model,
        target_ebops=1e9,          # 不限制 eBOPs，纯粹按拓扑条件决定度
        b_a_init=3.0,
        b_k_min=0.5,
        b_k_max=8.0,
        multiplier=1.5,
        min_degree=4,
        budget_weight='capacity',
        verbose=True,
    )

    # ── 4. 全 1-bit 初始化 ──────────────────────────────────────────────────
    BK_1BIT = 1.0   # 1-bit fractional → 实际 bits = i + b = 1 + 1 = 2? 不，HGQ 中 b=1 就是 1-bit

    apply_ramanujan_bw_init(
        model,
        per_layer_degree=per_layer_degree,
        per_layer_bk={name: BK_1BIT for name in per_layer_degree},
        seed=42,
        pruned_frac_bits=0.0,
        pruned_int_bits=0.0,
        active_int_bits=0.0,      # 0 整数位 → 纯小数，最小化 eBOPs
        also_zero_kernel=True,
        verbose=True,
    )

    # ── 5. 计算实际 eBOPs ────────────────────────────────────────────────────
    actual_ebops = compute_model_ebops(model, _sample)
    print(f'\n  Minimum Ramanujan 1-bit eBOPs = {actual_ebops:.0f}')

    # 每层 eBOPs
    layer_ebops = {}
    for layer in model._flatten_layers():
        if isinstance(layer, QLayerBase) and layer.enable_ebops and layer._ebops is not None:
            from keras import ops
            layer_ebops[layer.name] = int(ops.convert_to_numpy(layer._ebops))
    for name, eb in layer_ebops.items():
        print(f'    {name:20s}  eBOPs = {eb}')

    # 拓扑统计
    print('\n  ┌─ Min Ramanujan Topology (1-bit) ────────────────────────────┐')
    for layer in _flatten_layers(model):
        mask = getattr(layer, 'ramanujan_mask', None)
        if mask is None:
            continue
        m = mask.numpy()
        active = int(m.sum())
        total = int(m.size)
        sparsity = 1.0 - m.mean()
        if m.ndim == 2:
            row_deg = m.sum(axis=1)
            col_deg = m.sum(axis=0)
            print(f'  │ {layer.name:20s}  {active:4d}/{total:5d}  sparse={sparsity:.1%}  '
                  f'row=[{row_deg.min():.0f},{row_deg.max():.0f}]  '
                  f'col=[{col_deg.min():.0f},{col_deg.max():.0f}]  '
                  f'd={int(col_deg[0])}')
    print('  └─────────────────────────────────────────────────────────────┘')

    # ── 6. 保存模型 + 绘图 ──────────────────────────────────────────────────
    model_path = output_dir / 'min_ramanujan_1bit.keras'
    model.save(str(model_path))

    plotter = TopologyGraphPlotter(symmetric_topology_plot=False, mirror_edges=False)
    layers_data = plotter.extract_layer_graph_data(model)

    # --- Matrix plot with eBOPs ---
    matrix_path = output_dir / 'min_ramanujan_1bit_topology_matrix.png'
    plotter.plot_weighted_topology_matrices(layers_data, matrix_path)

    # --- Circle graph with eBOPs annotation ---
    circle_path = output_dir / 'min_ramanujan_1bit_circle_graph.png'
    _plot_circle_with_ebops(plotter, layers_data, layer_ebops, actual_ebops,
                            per_layer_degree, circle_path)

    print(f'\n  Matrix plot: {matrix_path}')
    print(f'  Circle plot: {circle_path}')
    print(f'  Model saved: {model_path}')


def _plot_circle_with_ebops(plotter, layers, layer_ebops, total_ebops,
                            per_layer_degree, out_png):
    """绘制带 eBOPs 标注的 circle graph。"""
    layer_sizes = [layers[0].weighted_matrix.shape[0]] + [x.weighted_matrix.shape[1] for x in layers]
    x_pos = np.arange(len(layer_sizes), dtype=np.float32)
    y_pos = [plotter._symmetric_y_positions(n) for n in layer_sizes]

    values = [x.weighted_matrix[x.weighted_matrix > 0] for x in layers]
    values = [v for v in values if v.size > 0]
    vmax = float(np.percentile(np.concatenate(values), 99.0)) if values else 1.0
    vmax = max(vmax, 1e-8)

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=vmax)

    total_edges = 0
    for li, item in enumerate(layers):
        mat = item.weighted_matrix
        ys = y_pos[li]
        yt = y_pos[li + 1]
        src, dst = np.where(mat > 1e-8)
        total_edges += int(src.size)

        for i, j in zip(src, dst):
            score = float(mat[i, j])
            intensity = min(score / vmax, 1.0)
            linewidth = 0.5 + 1.6 * intensity
            alpha = 0.35 + 0.45 * intensity
            color = cmap(norm(score))
            ax.plot(
                [x_pos[li], x_pos[li + 1]],
                [ys[i], yt[j]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

    # 节点
    for li, n in enumerate(layer_sizes):
        y = y_pos[li]
        ax.scatter(
            np.full(n, x_pos[li]),
            y,
            s=28 if n <= 32 else 18,
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    # 标签 + 每层 eBOPs + degree
    labels = ["input"] + [x.name for x in layers]
    for li, name in enumerate(labels):
        top_text = f"{name}\n(n={layer_sizes[li]})"
        if li > 0:
            lname = layers[li - 1].name
            eb = layer_ebops.get(lname, 0)
            d = per_layer_degree.get(lname, '?')
            top_text += f"\nd={d}, eb={eb}"
        ax.text(x_pos[li], 1.08, top_text, ha="center", va="bottom", fontsize=9)

    # colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Edge score (bit × |weight|)")

    ax.set_xlim(-0.4, x_pos[-1] + 0.4)
    ax.set_ylim(-1.08, 1.28)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_title(
        f"Minimum Ramanujan Graph — 1-bit quantization\n"
        f"Total eBOPs = {total_ebops:.0f}  |  Active edges = {total_edges}  |  "
        f"All connections b_k=1.0 (1-bit)",
        fontsize=12, fontweight='bold',
    )
    ax.set_frame_on(False)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
