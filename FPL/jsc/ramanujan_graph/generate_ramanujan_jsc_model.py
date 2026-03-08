#!/usr/bin/env python3
"""
生成基于 Ramanujan 拓扑的 JSC 初始化模型：
1) 使用最小可训练度构造近似 d-regular 拓扑
2) 按连接重要性分配 4bit~0bit（重要连接分配更高位宽）
3) 输出模型、拓扑连接矩阵图与 EBOPs 报告
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import sys
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

THIS_DIR = Path(__file__).resolve().parent
JSC_ROOT = THIS_DIR.parent
if str(JSC_ROOT) not in sys.path:
    sys.path.insert(0, str(JSC_ROOT))

from data.data import get_data
from hgq.layers import QLayerBase
from model.model import get_model_hgq
from utils.ramanujan_budget_utils import _flatten_layers, _get_kq_var


def _assign_var_with_fallback(var: tf.Variable, arr: np.ndarray):
    """尽量按原形状赋值；若形状不兼容则退化为均值赋值。"""
    var_np = np.array(var.numpy(), dtype=np.float32)
    arr = np.array(arr, dtype=np.float32)
    if var_np.shape == arr.shape:
        var.assign(arr.astype(np.float32))
        return
    try:
        b = np.broadcast_to(arr, var_np.shape).astype(np.float32)
        var.assign(b)
        return
    except Exception:
        pass
    var.assign(np.full(var_np.shape, float(np.mean(arr)), dtype=np.float32))


def _forward_update_ebops_no_bn_drift(model: keras.Model, sample_input: tf.Tensor):
    bn_layers = []
    old_momentum = []
    for layer in _flatten_layers(model):
        if isinstance(layer, keras.layers.BatchNormalization):
            bn_layers.append(layer)
            old_momentum.append(float(layer.momentum))
            layer.momentum = 1.0
    try:
        model(sample_input, training=True)
    finally:
        for layer, m in zip(bn_layers, old_momentum):
            layer.momentum = m


def compute_model_ebops(model: keras.Model, sample_input: tf.Tensor):
    """返回 (total_ebops, per_layer_ebops)。"""
    from keras import ops

    _forward_update_ebops_no_bn_drift(model, sample_input)
    total = 0.0
    per_layer = {}
    for layer in _flatten_layers(model):
        if isinstance(layer, QLayerBase) and getattr(layer, "enable_ebops", False) and layer._ebops is not None:
            e = float(int(ops.convert_to_numpy(layer._ebops)))
            per_layer[layer.name] = e
            total += e
    return float(total), per_layer


def build_min_degree_ramanujan_mask(in_dim: int, out_dim: int, degree: int, rng: np.random.Generator):
    """构造列 d-regular 的近似 Ramanujan 拓扑掩膜（1=保留，0=剪掉）。"""
    d = int(np.clip(degree, 1, in_dim))
    mask = np.zeros((in_dim, out_dim), dtype=np.float32)

    # 每个输出节点固定最小可训练度 d。
    for o in range(out_dim):
        idx = rng.choice(in_dim, size=d, replace=False)
        mask[idx, o] = 1.0

    # 当连边总数足够时，尽量避免输入节点完全断开（提升可训练性）。
    if d * out_dim >= in_dim:
        row_deg = np.sum(mask, axis=1)
        for i in np.where(row_deg == 0)[0]:
            o = int(rng.integers(0, out_dim))
            active_rows = np.where(mask[:, o] > 0.5)[0]
            if active_rows.size > 0:
                repl = int(rng.choice(active_rows))
                mask[repl, o] = 0.0
            mask[i, o] = 1.0
    return mask


def assign_importance_aware_bits(
    kernel: np.ndarray,
    mask: np.ndarray,
    bit_low: int = 1,
    bit_high: int = 4,
):
    """按连接重要性（|w|）将活跃边分配到 [bit_low, bit_high]，剪掉边为 0。"""
    bits = np.zeros_like(kernel, dtype=np.float32)
    active_idx = np.where(mask > 0.5)
    if active_idx[0].size == 0:
        return bits

    bit_low = int(max(1, bit_low))
    bit_high = int(max(bit_low, bit_high))
    span = bit_high - bit_low + 1

    imp = np.abs(kernel[active_idx]).astype(np.float32)
    # 排序分位，重要边分配更高位宽
    order = np.argsort(imp)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    pct = ranks / max(order.size - 1, 1)
    assigned = bit_low + np.floor(pct * span).astype(np.int32)
    assigned = np.clip(assigned, bit_low, bit_high).astype(np.float32)
    bits[active_idx] = assigned
    return bits


def apply_ramanujan_importance_init(
    model: keras.Model,
    min_degree: int,
    bit_low: int,
    bit_high: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    layer_summaries = []
    bit_mats = {}

    for layer in _flatten_layers(model):
        if not hasattr(layer, "kernel") or getattr(layer, "kq", None) is None:
            continue
        kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
        if kernel.ndim != 2:
            continue

        in_dim, out_dim = kernel.shape
        d = int(np.clip(min_degree, 1, in_dim))
        mask = build_min_degree_ramanujan_mask(in_dim, out_dim, d, rng)
        bits = assign_importance_aware_bits(kernel, mask, bit_low=bit_low, bit_high=bit_high)

        # 浮点 kernel 上同步稀疏
        layer.kernel.assign((kernel * mask).astype(np.float32))

        # kq: b/f, i, k
        kq = layer.kq
        b_var = _get_kq_var(kq, "b")
        if b_var is None:
            b_var = _get_kq_var(kq, "f")
        i_var = _get_kq_var(kq, "i")
        k_var = _get_kq_var(kq, "k")

        if b_var is not None:
            _assign_var_with_fallback(b_var, bits)
        if i_var is not None:
            i_arr = np.where(mask > 0.5, 1.0, -16.0).astype(np.float32)
            _assign_var_with_fallback(i_var, i_arr)
        if k_var is not None:
            _assign_var_with_fallback(k_var, (mask > 0.5).astype(np.float32))

        # bq: 每个输出神经元按其入边最大 bit 设置
        bq = getattr(layer, "bq", None)
        if bq is not None:
            out_bits = np.max(bits, axis=0).astype(np.float32)
            bb_var = _get_kq_var(bq, "b")
            if bb_var is None:
                bb_var = _get_kq_var(bq, "f")
            bi_var = _get_kq_var(bq, "i")
            bk_var = _get_kq_var(bq, "k")
            if bb_var is not None:
                _assign_var_with_fallback(bb_var, out_bits)
            if bi_var is not None:
                _assign_var_with_fallback(bi_var, np.where(out_bits > 0, 1.0, -16.0))
            if bk_var is not None:
                _assign_var_with_fallback(bk_var, (out_bits > 0).astype(np.float32))

        hist = {str(i): int(np.sum(np.round(bits) == i)) for i in range(0, bit_high + 1)}
        layer_summaries.append({
            "layer": layer.name,
            "shape": [int(in_dim), int(out_dim)],
            "degree": int(d),
            "active_edges": int(np.sum(mask)),
            "total_edges": int(mask.size),
            "sparsity": float(1.0 - np.mean(mask)),
            "bit_hist": hist,
        })
        bit_mats[layer.name] = bits
    return layer_summaries, bit_mats


def build_sample_input(model: keras.Model, input_h5: str, sample_size: int):
    h5_path = Path(input_h5)
    if h5_path.exists():
        (_, _), (x_val, _), _ = get_data(str(h5_path), src="openml")
        n = min(sample_size, x_val.shape[0])
        return tf.constant(x_val[:n], dtype=tf.float32), f"dataset:{h5_path}"

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    dyn = [sample_size]
    for d in input_shape[1:]:
        dyn.append(1 if d is None else int(d))
    x = np.random.randn(*dyn).astype(np.float32)
    return tf.constant(x), f"synthetic:{tuple(dyn)}"


def _to_symmetric_matrix(m: np.ndarray):
    """将矩阵做镜像扩展，得到上下/左右均对称的可视化矩阵。"""
    m_lr = np.concatenate([np.fliplr(m), m], axis=1)
    m_sym = np.concatenate([np.flipud(m_lr), m_lr], axis=0)
    return m_sym


def plot_topology_bit_matrices(bit_mats: dict, out_png: Path, bit_high: int, symmetric: bool = True):
    if not bit_mats:
        return
    names = list(bit_mats.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    axes = axes[0]
    im = None
    for i, name in enumerate(names):
        ax = axes[i]
        m = _to_symmetric_matrix(bit_mats[name]) if symmetric else bit_mats[name]
        im = ax.imshow(m, aspect="auto", vmin=0, vmax=bit_high, cmap="viridis")
        ax.set_title(f"{name}{' (sym)' if symmetric else ''}")
        ax.set_xlabel("out")
        ax.set_ylabel("in")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
    cbar.set_label(f"Bit width (0~{bit_high})")
    fig.suptitle("Ramanujan Topology Bit Matrices (importance-aware, symmetric view)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _symmetric_y_positions(n: int, span: float = 0.9):
    if n <= 1:
        return np.array([0.0], dtype=np.float32)
    return np.linspace(-span, span, n, dtype=np.float32)


def plot_circle_graph(layer_names, bit_mats, out_png: Path, bit_high: int, mirror_edges: bool = True):
    if not bit_mats:
        return
    layer_sizes = [bit_mats[0].shape[0]] + [m.shape[1] for m in bit_mats]
    x_pos = np.arange(len(layer_sizes), dtype=np.float32)
    y_pos = [_symmetric_y_positions(n) for n in layer_sizes]

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=float(bit_high))

    total_edges = 0
    for li, mat in enumerate(bit_mats):
        ys = y_pos[li]
        yt = y_pos[li + 1]
        src, dst = np.where(mat > 1e-6)
        total_edges += int(src.size)
        for i, j in zip(src, dst):
            bit = float(mat[i, j])
            ax.plot(
                [x_pos[li], x_pos[li + 1]],
                [ys[i], yt[j]],
                color=cmap(norm(bit)),
                linewidth=0.9,
                alpha=0.55,
            )
            if mirror_edges and (abs(float(ys[i])) > 1e-9 or abs(float(yt[j])) > 1e-9):
                ax.plot(
                    [x_pos[li], x_pos[li + 1]],
                    [-ys[i], -yt[j]],
                    color=cmap(norm(bit)),
                    linewidth=0.9,
                    alpha=0.55,
                )

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

    labels = ["input"] + layer_names
    for li, name in enumerate(labels):
        ax.text(x_pos[li], 1.05, f"{name}\n(n={layer_sizes[li]})", ha="center", va="bottom", fontsize=10)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"Edge bit width (0~{bit_high})")

    ax.set_xlim(-0.4, x_pos[-1] + 0.4)
    ax.set_ylim(-1.08, 1.18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_title(
        f"Ramanujan Circle Connection Graph (active edges={total_edges})"
        f"{' (symmetric mirror view)' if mirror_edges else ''}"
    )
    ax.set_frame_on(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Ramanujan-initialized JSC model (4bit~0bit)")
    parser.add_argument("--output_dir", type=str, default="ramanujan_graph/output")
    parser.add_argument("--model_name", type=str, default="jsc_ramanujan_importance_init.keras")
    parser.add_argument("--plot_name", type=str, default="topology_bit_matrices.png", help="Topology matrix plot filename")
    parser.add_argument("--circle_plot_name", type=str, default="circle_connection_graph.png", help="Circle connection graph filename")
    parser.add_argument("--report_name", type=str, default="report.json")
    parser.add_argument("--input_h5", type=str, default="data/dataset.h5")
    parser.add_argument("--sample_size", type=int, default=512)
    parser.add_argument("--min_degree", type=int, default=2, help="Minimum trainable degree per output node")
    parser.add_argument("--bit_low", type=int, default=0, help="Minimum bit in range (0 means pruned)")
    parser.add_argument("--bit_high", type=int, default=4, help="Maximum quantizer bits for important edges")
    parser.add_argument("--symmetric_topology_plot", action="store_true", default=True, help="Draw symmetric topology matrices")
    parser.add_argument("--no_symmetric_topology_plot", action="store_true", help="Disable symmetric topology matrix view")
    parser.add_argument("--mirror_edges", action="store_true", default=True, help="Draw mirrored edges in circle graph")
    parser.add_argument("--no_mirror_edges", action="store_true", help="Disable mirrored edges in circle graph")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.no_symmetric_topology_plot:
        args.symmetric_topology_plot = False
    if args.no_mirror_edges:
        args.mirror_edges = False

    if args.bit_high < 1:
        raise ValueError("--bit_high must be >= 1")
    if args.bit_low < 0:
        raise ValueError("--bit_low must be >= 0")
    if args.bit_low > args.bit_high:
        raise ValueError("--bit_low must be <= --bit_high")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model_hgq(init_bw_k=args.bit_high, init_bw_a=3)
    sample_input, sample_src = build_sample_input(model, args.input_h5, args.sample_size)

    layer_summaries, bit_mats = apply_ramanujan_importance_init(
        model=model,
        min_degree=args.min_degree,
        bit_low=max(1, args.bit_low),
        bit_high=args.bit_high,
        seed=args.seed,
    )
    total_ebops, per_layer_ebops = compute_model_ebops(model, sample_input)

    model_path = out_dir / args.model_name
    plot_path = out_dir / args.plot_name
    circle_plot_path = out_dir / args.circle_plot_name
    report_path = out_dir / args.report_name

    model.save(model_path)
    plot_topology_bit_matrices(
        bit_mats,
        plot_path,
        bit_high=args.bit_high,
        symmetric=args.symmetric_topology_plot,
    )
    plot_circle_graph(
        list(bit_mats.keys()),
        list(bit_mats.values()),
        circle_plot_path,
        bit_high=args.bit_high,
        mirror_edges=args.mirror_edges,
    )

    model_layers = []
    for l in model.layers:
        out_shape = getattr(l, "output_shape", None)
        if out_shape is None and hasattr(l, "output"):
            try:
                out_shape = tuple(int(d) if d is not None else None for d in l.output.shape)
            except Exception:
                out_shape = None
        model_layers.append({
            "name": l.name,
            "class": l.__class__.__name__,
            "output_shape": out_shape,
        })

    report = {
        "model_path": str(model_path),
        "plot_path": str(plot_path),
        "circle_plot_path": str(circle_plot_path),
        "sample_input": sample_src,
        "min_degree": int(args.min_degree),
        "bit_range": [int(args.bit_low), int(args.bit_high)],
        "symmetric_topology_plot": bool(args.symmetric_topology_plot),
        "mirror_edges": bool(args.mirror_edges),
        "total_ebops": float(total_ebops),
        "per_layer_ebops": {k: float(v) for k, v in per_layer_ebops.items()},
        "model_layers": model_layers,
        "layers": layer_summaries,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Ramanujan importance-aware JSC model generated.")
    print(f"Model  : {model_path}")
    print(f"Plot   : {plot_path}")
    print(f"Circle : {circle_plot_path}")
    print(f"Report : {report_path}")
    print(f"EBOPs  : {total_ebops:.1f}")
    print("-" * 72)
    for l in layer_summaries:
        h = l["bit_hist"]
        hist_str = ",".join(str(h.get(str(i), 0)) for i in range(0, args.bit_high + 1))
        print(
            f"{l['layer']:8s} degree={l['degree']:2d} "
            f"active={l['active_edges']:4d}/{l['total_edges']:4d} "
            f"sparsity={l['sparsity']:.3f} "
            f"bits0..{args.bit_high}=[{hist_str}]"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
