#!/usr/bin/env python3
"""
run_one_shot_prune_only.py
==========================
仅执行一次性剪枝，不进入任何后续训练阶段。

流程：
1) 加载 checkpoint
2) 执行 one-shot pruning（uniform / sensitivity / SNIP / GraSP / SynFlow / ...）
3) 通过一次前向传播实测 EBOPs
4) 保存权重与元信息
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 保证相对路径基于脚本目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import random
import argparse
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from data.data import get_data
from hgq.layers import QLayerBase
# 注册自定义层，确保 load_model 可反序列化 QEinsumDenseBatchnorm
import model.model  # noqa: F401
from utils.ramanujan_budget_utils import (
    HighBitPruner,
    SensitivityAwarePruner,
    compute_bw_aware_degree,
    _flatten_layers,
    _get_kq_var,
)

np.random.seed(42)
random.seed(42)


def _forward_update_ebops_no_bn_drift(model, sample_input):
    """刷新 layer._ebops，同时避免 BatchNorm moving stats 被更新。"""
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


def compute_model_ebops(model, sample_input) -> float:
    """通过一次前向传播实测模型当前 EBOPs。"""
    from keras import ops

    _forward_update_ebops_no_bn_drift(model, sample_input)
    total = 0
    for layer in model._flatten_layers():
        if isinstance(layer, QLayerBase) and layer.enable_ebops and layer._ebops is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def print_bk_stats(model, label=''):
    all_b = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None:
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            all_b.extend(b_var.numpy().ravel().tolist())
    if all_b:
        arr = np.array(all_b)
        print(
            f"  [bk_stats {label}]  "
            f"mean={arr.mean():.3f}  std={arr.std():.3f}  "
            f"min={arr.min():.3f}  max={arr.max():.3f}  "
            f"p5={np.percentile(arr,5):.3f}  p95={np.percentile(arr,95):.3f}  "
            f"n_dead(<=0.1)={int((arr<=0.1).sum())}/{len(arr)}"
        )


def _build_topk_mask_with_connectivity(weight_2d: np.ndarray, degree: int) -> np.ndarray:
    """按输出列 top-k 构图，并修复输入孤点，保证基础连通性。"""
    in_dim, out_dim = weight_2d.shape
    d = int(np.clip(degree, 1, in_dim))
    abs_w = np.abs(weight_2d)
    mask = np.zeros_like(weight_2d, dtype=np.float32)

    # 每个输出保留 top-d 连接
    for o in range(out_dim):
        idx = np.argpartition(abs_w[:, o], -d)[-d:]
        mask[idx, o] = 1.0

    # 修复输入孤点：若某输入无连接，找其最强输出并替换该输出当前最弱边
    row_deg = np.sum(mask, axis=1)
    for i in np.where(row_deg == 0)[0]:
        o = int(np.argmax(abs_w[i, :]))
        active_rows = np.where(mask[:, o] > 0.5)[0]
        if active_rows.size == 0:
            mask[i, o] = 1.0
            continue
        weakest = active_rows[np.argmin(abs_w[active_rows, o])]
        mask[weakest, o] = 0.0
        mask[i, o] = 1.0

    return mask


def _forward_reachable(masks: list[np.ndarray]) -> list[np.ndarray]:
    if not masks:
        return []
    reachable = [np.ones(masks[0].shape[0], dtype=bool)]
    for mask in masks:
        nxt = np.any(mask > 0.5, axis=0) & ((reachable[-1].astype(np.float32) @ (mask > 0.5).astype(np.float32)) > 0)
        reachable.append(nxt.astype(bool))
    return reachable


def _backward_reachable(masks: list[np.ndarray]) -> list[np.ndarray]:
    if not masks:
        return []
    live = [None] * (len(masks) + 1)
    live[-1] = np.ones(masks[-1].shape[1], dtype=bool)
    for idx in range(len(masks) - 1, -1, -1):
        mask = masks[idx] > 0.5
        prev = np.any(mask, axis=1) & (((mask.astype(np.float32)) @ live[idx + 1].astype(np.float32)) > 0)
        live[idx] = prev.astype(bool)
    return live


def _prune_to_effective_paths(masks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, float]]:
    masks = [np.array(m, dtype=np.float32) for m in masks]
    if not masks:
        return masks, {"effective_paths": 0.0, "reachable_outputs": 0.0}

    changed = True
    while changed:
        changed = False
        fwd = _forward_reachable(masks)
        bwd = _backward_reachable(masks)
        new_masks = []
        for idx, mask in enumerate(masks):
            keep = (mask > 0.5) & fwd[idx][:, None] & bwd[idx + 1][None, :]
            new_mask = keep.astype(np.float32)
            if not np.array_equal(new_mask, mask):
                changed = True
            new_masks.append(new_mask)
        masks = new_masks

    path_counts = np.ones(masks[0].shape[0], dtype=np.float64)
    for mask in masks:
        path_counts = np.clip(path_counts @ (mask > 0.5).astype(np.float64), 0.0, 1e18)
    effective_paths = float(np.sum(path_counts))
    reachable_outputs = float(np.sum(path_counts > 0))
    return masks, {
        "effective_paths": effective_paths,
        "reachable_outputs": reachable_outputs,
    }


def _best_output_paths(kernels: list[np.ndarray]) -> list[np.ndarray]:
    if not kernels:
        return []
    path_masks = [np.zeros_like(k, dtype=np.float32) for k in kernels]
    eps = 1e-12
    n_layers = len(kernels)

    # Dynamic programming over nodes: best path score from any input to each node.
    prev_score = np.zeros(kernels[0].shape[0], dtype=np.float64)
    parents = []
    for kernel in kernels:
        score_mat = np.log(np.abs(kernel).astype(np.float64) + eps)
        total = prev_score[:, None] + score_mat
        parent = np.argmax(total, axis=0).astype(np.int32)
        cur_score = total[parent, np.arange(total.shape[1])]
        parents.append(parent)
        prev_score = cur_score

    out_dim = kernels[-1].shape[1]
    for out_idx in range(out_dim):
        node = int(out_idx)
        for layer_idx in range(n_layers - 1, -1, -1):
            prev_node = int(parents[layer_idx][node])
            path_masks[layer_idx][prev_node, node] = 1.0
            node = prev_node
    return path_masks


def _compose_path_scores(kernels: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
    bool_masks = [(m > 0.5).astype(np.float64) for m in masks]
    if not bool_masks:
        return []
    fwd = [np.ones(bool_masks[0].shape[0], dtype=np.float64)]
    for mask in bool_masks:
        fwd.append(np.clip(fwd[-1] @ mask, 0.0, 1e18))
    bwd = [None] * (len(bool_masks) + 1)
    bwd[-1] = np.ones(bool_masks[-1].shape[1], dtype=np.float64)
    for idx in range(len(bool_masks) - 1, -1, -1):
        bwd[idx] = np.clip(bool_masks[idx] @ bwd[idx + 1], 0.0, 1e18)
    scores = []
    for idx, kernel in enumerate(kernels):
        path_contrib = fwd[idx][:, None] * bwd[idx + 1][None, :]
        abs_w = np.abs(kernel).astype(np.float64)
        denom = float(np.max(abs_w)) + 1e-12
        score = (abs_w / denom) * np.log1p(path_contrib)
        scores.append(score.astype(np.float32))
    return scores


def _mask_has_output_paths(masks: list[np.ndarray]) -> bool:
    if not masks:
        return False
    _, stats = _prune_to_effective_paths(masks)
    return int(stats["reachable_outputs"]) == int(masks[-1].shape[1])


def _apply_structured_masks_with_quant(
    layers,
    masks: list[np.ndarray],
    per_layer_bk: dict[str, float],
    b_floor: float,
    b_ceiling: float,
):
    for layer, mask in zip(layers, masks):
        kernel = layer.kernel.numpy().astype(np.float32)
        kq = layer.kq
        active = mask > 0.5
        kernel_new = (kernel * mask).astype(np.float32)
        active_abs = np.abs(kernel[active])
        if active_abs.size > 0:
            ref_abs = active_abs[active_abs > 0.0]
            if ref_abs.size == 0:
                global_abs = np.abs(kernel[np.abs(kernel) > 0.0])
                ref_abs = global_abs if global_abs.size > 0 else np.array([1e-3], dtype=np.float32)
            mag_floor = float(max(np.percentile(ref_abs, 25) * 0.50, 1e-3))
            weak = active & (np.abs(kernel_new) < mag_floor)
            if np.any(weak):
                signs = np.sign(kernel).astype(np.float32)
                signs[signs == 0.0] = 1.0
                kernel_new[weak] = signs[weak] * mag_floor
        layer.kernel.assign(kernel_new.astype(np.float32))

        b_var = _get_kq_var(kq, "b")
        if b_var is None:
            b_var = _get_kq_var(kq, "f")
        i_var = _get_kq_var(kq, "i")
        k_var = _get_kq_var(kq, "k")

        if b_var is not None:
            b_old = b_var.numpy().astype(np.float32)
            abs_k = np.abs(kernel_new)
            denom = float(np.max(abs_k)) + 1e-12
            importance = abs_k / denom
            b_base = float(np.clip(max(per_layer_bk.get(layer.name, b_floor), 2.0), b_floor, b_ceiling))
            b_active = np.clip(np.maximum(b_old, b_base) * (0.8 + 0.4 * importance), b_floor, b_ceiling)
            b_new = np.where(mask > 0.5, b_active, 0.0)
            b_var.assign(b_new.astype(np.float32))
        if i_var is not None:
            i_old = i_var.numpy().astype(np.float32)
            i_var.assign(np.where(mask > 0.5, np.clip(i_old, -2.0, 6.0), -16.0).astype(np.float32))
        if k_var is not None:
            k_var.assign(np.where(mask > 0.5, 1.0, 0.0).astype(np.float32))

        bq = getattr(layer, "bq", None)
        if bq is not None:
            cols_active = np.sum(mask > 0.5, axis=0) > 0
            bb_var = _get_kq_var(bq, "b")
            if bb_var is None:
                bb_var = _get_kq_var(bq, "f")
            bi_var = _get_kq_var(bq, "i")
            bk_var = _get_kq_var(bq, "k")
            if bb_var is not None:
                bb_old = bb_var.numpy().astype(np.float32)
                bb_floor = float(np.clip(max(float(b_floor), 2.0), b_floor, b_ceiling))
                bb_var.assign(np.where(cols_active, np.clip(np.maximum(bb_old, bb_floor), bb_floor, b_ceiling), 0.0).astype(np.float32))
            if bi_var is not None:
                bi_old = bi_var.numpy().astype(np.float32)
                bi_var.assign(np.where(cols_active, np.clip(bi_old, -2.0, 6.0), -16.0).astype(np.float32))
            if bk_var is not None:
                bk_var.assign(np.where(cols_active, 1.0, 0.0).astype(np.float32))


def spectral_path_prune_to_ebops(
    model,
    target_ebops: float,
    sample_input,
    min_degree: int = 2,
    min_input_width: int = 4,
    min_hidden_width: int = 4,
    b_floor: float = 0.25,
    b_ceiling: float = 6.0,
    near_budget_ratio: float = 1.6,
    high_budget_ratio: float = 0.45,
    verbose: bool = True,
):
    """Connectivity-aware pruning based on a path-connected node subnetwork.

    Strategy:
    - keep all output nodes
    - choose input/hidden node subsets with spectral/path saliency under budget
    - connect consecutive kept node sets as a chain subnetwork

    This guarantees:
    - every output is connected to the selected input set
    - no kept hidden node is only-in or only-out
    - no hidden/output layer collapse
    """
    current_ebops = compute_model_ebops(model, sample_input)
    budget_ratio = float(target_ebops) / max(float(current_ebops), 1.0)
    if current_ebops <= float(target_ebops) * float(near_budget_ratio):
        if verbose:
            print(
                f"[SpectralPathPruner] near-budget preserve mode: "
                f"current={current_ebops:.1f}, target={target_ebops:.1f}"
            )
        return current_ebops, {"effective_paths": 0.0, "reachable_outputs": 0.0, "mode": "preserve"}

    if budget_ratio >= float(high_budget_ratio):
        if verbose:
            print(
                f"[SpectralPathPruner] high-budget fallback to sensitivity: "
                f"target/current={budget_ratio:.3f} >= {high_budget_ratio:.3f}"
            )
        pruner = SensitivityAwarePruner(
            target_ebops=float(target_ebops),
            pruned_threshold=0.1,
            b_k_min=max(float(b_floor), 0.20),
        )
        pruner.prune_to_ebops(model, current_ebops=current_ebops, verbose=verbose)
        measured = compute_model_ebops(model, sample_input)
        return measured, {"effective_paths": 0.0, "reachable_outputs": 0.0, "mode": "sensitivity_fallback"}

    layers = _dense_prunable_layers(model)
    if len(layers) < 2:
        measured = compute_model_ebops(model, sample_input)
        return measured, {"effective_paths": 0.0, "reachable_outputs": 0.0, "mode": "no_layers"}

    weights = [np.array(layer.kernel.numpy(), dtype=np.float32) for layer in layers]
    widths = [int(weights[0].shape[0])] + [int(w.shape[1]) for w in weights]
    hidden_idx = list(range(1, len(widths) - 1))
    removable_idx = list(range(0, len(widths) - 1))  # input + hidden, outputs fixed

    from keras import ops
    _forward_update_ebops_no_bn_drift(model, sample_input)
    edge_costs = []
    for layer in layers:
        if getattr(layer, "_ebops", None) is not None:
            layer_ebops = float(int(ops.convert_to_numpy(layer._ebops)))
        else:
            layer_ebops = 0.0
        active = _layer_active_mask(layer)
        n_active = int(np.sum(active))
        if n_active <= 0:
            n_active = int(np.prod(layer.kernel.shape))
        edge_costs.append(layer_ebops / max(n_active, 1))

    per_layer_bk = compute_bw_aware_degree(
        model,
        target_ebops=float(target_ebops),
        b_a_init=3.0,
        b_k_min=b_floor,
        b_k_max=b_ceiling,
        multiplier=1.2,
        min_degree=min_degree,
        budget_weight="capacity",
        verbose=False,
    )[1]

    saliency = {}
    remove_order = {}
    keep_map = {}
    removed_ptr = {}
    min_keep = {}
    # Input nodes: prefer keeping rows with stronger first-layer support.
    input_score = np.mean(np.abs(weights[0]), axis=1)
    saliency[0] = input_score
    remove_order[0] = np.argsort(input_score).tolist()
    keep_map[0] = np.ones_like(input_score, dtype=bool)
    removed_ptr[0] = 0
    min_keep[0] = int(max(2, min(min_input_width, len(input_score))))
    for h in hidden_idx:
        prev_w = np.abs(weights[h - 1])
        next_w = np.abs(weights[h])
        # Path-aware node score: good support from previous layer and to next layer.
        s = np.mean(prev_w, axis=0) * np.mean(next_w, axis=1)
        saliency[h] = s
        remove_order[h] = np.argsort(s).tolist()
        keep_map[h] = np.ones_like(s, dtype=bool)
        removed_ptr[h] = 0
        min_keep[h] = int(max(2, min(min_hidden_width, len(s))))

    def predict_ebops(curr_widths):
        total = 0.0
        for li, c in enumerate(edge_costs):
            total += c * max(curr_widths[li], curr_widths[li + 1])
        return total

    def pick_next_removal(floor_map):
        best = None
        for h in removable_idx:
            if widths[h] <= floor_map[h]:
                continue
            ptr = removed_ptr[h]
            order = remove_order[h]
            while ptr < len(order) and (not keep_map[h][order[ptr]]):
                ptr += 1
            removed_ptr[h] = ptr
            if ptr >= len(order):
                continue
            n = order[ptr]
            penalty = float(saliency[h][n]) + 1e-9
            next_widths = list(widths)
            next_widths[h] -= 1
            delta = predict_ebops(widths) - predict_ebops(next_widths)
            score = penalty / max(delta, 1e-9)
            if best is None or score < best[0]:
                best = (score, h, n)
        return best

    def build_masks():
        masks_local = []
        for li, layer in enumerate(layers):
            in_dim, out_dim = layer.kernel.shape
            if li == 0:
                row_idx = np.flatnonzero(keep_map[0])
            else:
                row_idx = np.flatnonzero(keep_map[li])
            if li == len(layers) - 1:
                col_idx = np.arange(out_dim, dtype=int)
            else:
                col_idx = np.flatnonzero(keep_map[li + 1])
            mask = np.zeros((in_dim, out_dim), dtype=np.float32)
            if row_idx.size > 0 and col_idx.size > 0:
                # Two directional covering passes: every kept src has an outgoing edge,
                # every kept dst has an incoming edge. This preserves path-connectivity
                # without wasting budget on full bipartite wiring.
                for ridx, src in enumerate(row_idx):
                    dst = col_idx[ridx % len(col_idx)]
                    mask[src, dst] = 1.0
                for cidx, dst in enumerate(col_idx):
                    src = row_idx[cidx % len(row_idx)]
                    mask[src, dst] = 1.0
            masks_local.append(mask)
        return masks_local

    # Greedily remove weakest hidden nodes until the chain subnetwork meets budget.
    guard = 0
    while predict_ebops(widths) > target_ebops * 1.05:
        guard += 1
        if guard > 10000:
            break
        best = pick_next_removal(min_keep)
        if best is None:
            break
        _, h, n = best
        keep_map[h][n] = False
        widths[h] -= 1
        removed_ptr[h] += 1

    masks = build_masks()
    _apply_structured_masks_with_quant(layers, masks, per_layer_bk, b_floor=b_floor, b_ceiling=b_ceiling)
    measured = compute_model_ebops(model, sample_input)
    hard_min_keep = {0: int(max(2, min(2, len(input_score))))}
    for h in hidden_idx:
        hard_min_keep[h] = int(max(1, min(2, len(saliency[h]))))

    refine_guard = 0
    while measured > float(target_ebops) * 1.02:
        refine_guard += 1
        if refine_guard > 10000:
            break
        best = pick_next_removal(hard_min_keep)
        if best is None:
            break
        _, h, n = best
        keep_map[h][n] = False
        widths[h] -= 1
        removed_ptr[h] += 1
        masks = build_masks()
        _apply_structured_masks_with_quant(layers, masks, per_layer_bk, b_floor=b_floor, b_ceiling=b_ceiling)
        measured = compute_model_ebops(model, sample_input)

    input_keep = widths[0]
    hidden_keep = [widths[h] for h in hidden_idx]
    output_keep = widths[-1]
    paths_per_output = float(input_keep * np.prod(hidden_keep)) if hidden_keep else float(input_keep)
    path_stats = {
        "effective_paths": float(paths_per_output * output_keep),
        "reachable_outputs": float(output_keep),
    }
    if verbose:
        print(
            f"[SpectralPathPruner] measured_ebops={measured:.1f}  target={target_ebops:.1f}  "
            f"effective_paths={path_stats['effective_paths']:.1f}  reachable_outputs={path_stats['reachable_outputs']:.0f}"
        )
        for layer, mask in zip(layers, masks):
            col_deg = np.sum(mask > 0.5, axis=0)
            row_deg = np.sum(mask > 0.5, axis=1)
            print(
                f"  [SpectralPathPruner] {layer.name:20s}  "
                f"deg_col_mean={float(np.mean(col_deg)):.2f}  "
                f"deg_row_zero={int(np.sum(row_deg == 0))}"
            )
    return measured, {
        "effective_paths": float(path_stats["effective_paths"]),
        "reachable_outputs": float(path_stats["reachable_outputs"]),
        "selected_inputs": int(input_keep),
        "selected_hidden": [int(v) for v in hidden_keep],
        "mode": "spectral_path",
    }


def _dense_prunable_layers(model):
    layers = []
    for layer in _flatten_layers(model):
        if not hasattr(layer, "kernel") or getattr(layer, "kq", None) is None:
            continue
        if len(layer.kernel.shape) != 2:
            continue
        layers.append(layer)
    return layers


def _layer_active_mask(layer) -> np.ndarray:
    from keras import ops

    try:
        bits = layer.kq.bits_(ops.shape(layer.kernel))
        bits_np = np.array(ops.convert_to_numpy(bits), dtype=np.float32)
        return bits_np > 1e-6
    except Exception:
        b_var = _get_kq_var(layer.kq, "b")
        if b_var is None:
            b_var = _get_kq_var(layer.kq, "f")
        if b_var is None:
            return np.zeros(tuple(layer.kernel.shape), dtype=bool)
        b_np = np.array(b_var.numpy(), dtype=np.float32)
        k_var = _get_kq_var(layer.kq, "k")
        if k_var is None:
            return b_np > 0.0
        k_np = np.array(k_var.numpy(), dtype=np.float32)
        return (b_np > 0.0) | (k_np > 0.0)


def _apply_mask_and_quant(
    layer,
    mask: np.ndarray,
    b_floor: float,
    b_ceiling: float,
):
    kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
    m = mask.astype(np.float32)
    layer.kernel.assign((kernel * m).astype(np.float32))

    kq = layer.kq
    b_var = _get_kq_var(kq, "b")
    if b_var is None:
        b_var = _get_kq_var(kq, "f")
    i_var = _get_kq_var(kq, "i")
    k_var = _get_kq_var(kq, "k")

    if b_var is not None:
        b_old = np.array(b_var.numpy(), dtype=np.float32)
        b_new = np.where(
            m > 0.5,
            np.clip(np.maximum(b_old, b_floor), b_floor, b_ceiling),
            0.0,
        )
        b_var.assign(b_new.astype(np.float32))

    if i_var is not None:
        i_old = np.array(i_var.numpy(), dtype=np.float32)
        i_new = np.where(m > 0.5, np.clip(i_old, -2.0, 6.0), -16.0)
        i_var.assign(i_new.astype(np.float32))

    if k_var is not None:
        k_new = np.where(m > 0.5, 1.0, 0.0).astype(np.float32)
        k_var.assign(k_new)

    # bias quantizer: output-neuron 级别保留
    bq = getattr(layer, "bq", None)
    if bq is not None:
        cols_active = np.sum(m, axis=0) > 0.5
        bb_var = _get_kq_var(bq, "b")
        if bb_var is None:
            bb_var = _get_kq_var(bq, "f")
        bi_var = _get_kq_var(bq, "i")
        bk_var = _get_kq_var(bq, "k")
        if bb_var is not None:
            bb_old = np.array(bb_var.numpy(), dtype=np.float32)
            bb_new = np.where(
                cols_active,
                np.clip(np.maximum(bb_old, b_floor), b_floor, b_ceiling),
                0.0,
            )
            bb_var.assign(bb_new.astype(np.float32))
        if bi_var is not None:
            bi_old = np.array(bi_var.numpy(), dtype=np.float32)
            bi_new = np.where(cols_active, np.clip(bi_old, -2.0, 6.0), -16.0)
            bi_var.assign(bi_new.astype(np.float32))
        if bk_var is not None:
            bk_new = np.where(cols_active, 1.0, 0.0).astype(np.float32)
            bk_var.assign(bk_new)


def _apply_mask_preserve_quant(layer, mask: np.ndarray):
    """Apply a binary mask while preserving surviving quantizer state.

    This is closer to the original pruning baselines, which choose a mask but do
    not re-optimize or re-assign the remaining weights/quantizers at prune time.
    """
    kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
    m = mask.astype(np.float32)
    layer.kernel.assign((kernel * m).astype(np.float32))

    kq = layer.kq
    b_var = _get_kq_var(kq, "b")
    if b_var is None:
        b_var = _get_kq_var(kq, "f")
    i_var = _get_kq_var(kq, "i")
    k_var = _get_kq_var(kq, "k")

    if b_var is not None:
        b_old = np.array(b_var.numpy(), dtype=np.float32)
        b_var.assign(np.where(m > 0.5, b_old, 0.0).astype(np.float32))
    if i_var is not None:
        i_old = np.array(i_var.numpy(), dtype=np.float32)
        i_var.assign(np.where(m > 0.5, i_old, -16.0).astype(np.float32))
    if k_var is not None:
        k_old = np.array(k_var.numpy(), dtype=np.float32)
        k_var.assign(np.where(m > 0.5, k_old, 0.0).astype(np.float32))

    bq = getattr(layer, "bq", None)
    if bq is not None:
        cols_active = np.sum(m, axis=0) > 0.5
        bb_var = _get_kq_var(bq, "b")
        if bb_var is None:
            bb_var = _get_kq_var(bq, "f")
        bi_var = _get_kq_var(bq, "i")
        bk_var = _get_kq_var(bq, "k")
        if bb_var is not None:
            bb_old = np.array(bb_var.numpy(), dtype=np.float32)
            bb_var.assign(np.where(cols_active, bb_old, 0.0).astype(np.float32))
        if bi_var is not None:
            bi_old = np.array(bi_var.numpy(), dtype=np.float32)
            bi_var.assign(np.where(cols_active, bi_old, -16.0).astype(np.float32))
        if bk_var is not None:
            bk_old = np.array(bk_var.numpy(), dtype=np.float32)
            bk_var.assign(np.where(cols_active, bk_old, 0.0).astype(np.float32))


def _collect_named_outputs(model, layer_names, sample_input, training=False):
    outputs = [model.get_layer(n).output for n in layer_names]
    probe = keras.Model(model.inputs, outputs)
    vals = probe(sample_input, training=training)
    if not isinstance(vals, (list, tuple)):
        vals = [vals]
    ret = {}
    for n, v in zip(layer_names, vals):
        ret[n] = np.array(v.numpy(), dtype=np.float32)
    return ret


def _reestimate_quantizer_ranges(
    model,
    b_floor: float = 0.35,
    b_ceiling: float = 6.0,
    i_min: float = -2.0,
    i_max: float = 6.0,
):
    """按当前掩码与权重幅值重估 kq/bq 参数，减少量化饱和与死区。"""
    for layer in _dense_prunable_layers(model):
        kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
        active = _layer_active_mask(layer)

        kq = layer.kq
        b_var = _get_kq_var(kq, "b")
        if b_var is None:
            b_var = _get_kq_var(kq, "f")
        i_var = _get_kq_var(kq, "i")
        k_var = _get_kq_var(kq, "k")

        if b_var is not None:
            b_old = np.array(b_var.numpy(), dtype=np.float32)
            b_new = np.where(active, np.clip(np.maximum(b_old, b_floor), b_floor, b_ceiling), 0.0)
            b_var.assign(b_new.astype(np.float32))

        if i_var is not None:
            i_old = np.array(i_var.numpy(), dtype=np.float32)
            i_req = np.ceil(np.log2(np.maximum(np.abs(kernel), 1e-8))).astype(np.float32) + 1.0
            i_new = np.where(active, np.clip(np.maximum(i_old, i_req), i_min, i_max), -16.0)
            i_var.assign(i_new.astype(np.float32))

        if k_var is not None:
            k_new = np.where(active, 1.0, 0.0).astype(np.float32)
            k_var.assign(k_new)

        cols_active = np.sum(active, axis=0) > 0
        bias = getattr(layer, "bias", None)
        bq = getattr(layer, "bq", None)
        if bq is None:
            continue

        bb_var = _get_kq_var(bq, "b")
        if bb_var is None:
            bb_var = _get_kq_var(bq, "f")
        bi_var = _get_kq_var(bq, "i")
        bk_var = _get_kq_var(bq, "k")

        if bb_var is not None:
            bb_old = np.array(bb_var.numpy(), dtype=np.float32)
            bb_new = np.where(cols_active, np.clip(np.maximum(bb_old, b_floor), b_floor, b_ceiling), 0.0)
            bb_var.assign(bb_new.astype(np.float32))

        if bi_var is not None:
            bi_old = np.array(bi_var.numpy(), dtype=np.float32)
            if bias is not None:
                bias_np = np.array(bias.numpy(), dtype=np.float32)
                bi_req = np.ceil(np.log2(np.maximum(np.abs(bias_np), 1e-8))).astype(np.float32) + 1.0
                bi_new = np.where(cols_active, np.clip(np.maximum(bi_old, bi_req), i_min, i_max), -16.0)
            else:
                bi_new = np.where(cols_active, np.clip(bi_old, i_min, i_max), -16.0)
            bi_var.assign(bi_new.astype(np.float32))

        if bk_var is not None:
            bk_new = np.where(cols_active, 1.0, 0.0).astype(np.float32)
            bk_var.assign(bk_new)


def _bias_active_mask(layer, fallback_mask: np.ndarray | None = None) -> np.ndarray | None:
    bq = getattr(layer, "bq", None)
    if bq is None:
        return fallback_mask

    bb_var = _get_kq_var(bq, "b")
    if bb_var is None:
        bb_var = _get_kq_var(bq, "f")
    bk_var = _get_kq_var(bq, "k")

    if bb_var is None:
        return fallback_mask

    bb_np = np.array(bb_var.numpy(), dtype=np.float32)
    if bk_var is not None:
        bk_np = np.array(bk_var.numpy(), dtype=np.float32)
        return ((bb_np > 0.0) | (bk_np > 0.0)).astype(bool)
    return (bb_np > 0.0).astype(bool)


def _labels_to_sparse(y_np: np.ndarray) -> np.ndarray:
    y_np = np.array(y_np)
    if y_np.ndim >= 2 and y_np.shape[-1] > 1:
        return np.argmax(y_np, axis=-1).astype(np.int32)
    return y_np.reshape(-1).astype(np.int32)


def build_prune_batch(
    model,
    sample_size: int,
    input_h5: str | None,
    teacher_model=None,
):
    """Build a supervised pruning batch; fall back to teacher pseudo labels."""
    if input_h5 and os.path.exists(input_h5):
        (_, _), (x_val, y_val), _ = get_data(input_h5, src='openml')
        n = min(sample_size, len(x_val))
        x = tf.constant(x_val[:n], dtype=tf.float32)
        y = tf.constant(_labels_to_sparse(y_val[:n]), dtype=tf.int32)
        return x, y, f"dataset:{input_h5}"

    x, src = build_sample_input(model, sample_size, input_h5)
    teacher = teacher_model if teacher_model is not None else model
    logits = teacher(x, training=False)
    y = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    return x, y, f"{src}:pseudo"


def _snapshot_prunable_state(model):
    snapshots = []
    for layer in _dense_prunable_layers(model):
        snap = {
            "layer": layer,
            "kernel": np.array(layer.kernel.numpy(), dtype=np.float32),
        }
        bias = getattr(layer, "bias", None)
        if bias is not None:
            snap["bias"] = np.array(bias.numpy(), dtype=np.float32)

        for prefix, qobj in (("kq", getattr(layer, "kq", None)), ("bq", getattr(layer, "bq", None))):
            if qobj is None:
                continue
            for name in ("b", "f", "i", "k"):
                var = _get_kq_var(qobj, name)
                if var is not None:
                    snap[f"{prefix}.{name}"] = (var, np.array(var.numpy(), dtype=np.float32))
        snapshots.append(snap)
    return snapshots


def _restore_prunable_state(snapshots):
    for snap in snapshots:
        layer = snap["layer"]
        layer.kernel.assign(snap["kernel"])
        bias = getattr(layer, "bias", None)
        if bias is not None and "bias" in snap:
            bias.assign(snap["bias"])
        for key, value in snap.items():
            if key in {"layer", "kernel", "bias"}:
                continue
            var, arr = value
            var.assign(arr)


def _active_kernel_mask_from_scores(layer, scores: np.ndarray) -> np.ndarray:
    active = _layer_active_mask(layer).astype(bool)
    if scores.shape != active.shape:
        raise ValueError(f"Score shape mismatch for layer {layer.name}: {scores.shape} vs {active.shape}")
    return active


def _global_topk_masks(score_items, keep_count: int):
    total_active = int(sum(int(np.sum(item["active"])) for item in score_items))
    keep_count = int(np.clip(keep_count, 0, total_active))
    if keep_count <= 0:
        return {id(item["layer"]): np.zeros_like(item["scores"], dtype=np.float32) for item in score_items}

    flat_scores = []
    for idx, item in enumerate(score_items):
        active_pos = np.flatnonzero(item["active"].reshape(-1))
        if active_pos.size == 0:
            continue
        vals = item["scores"].reshape(-1)[active_pos]
        vals = np.nan_to_num(vals, nan=-1e30, posinf=1e30, neginf=-1e30).astype(np.float32)
        for pos, val in zip(active_pos.tolist(), vals.tolist()):
            flat_scores.append((float(val), idx, int(pos)))

    if not flat_scores:
        return {id(item["layer"]): np.zeros_like(item["scores"], dtype=np.float32) for item in score_items}

    flat_scores.sort(key=lambda x: x[0], reverse=True)
    keep_set = {(idx, pos) for _, idx, pos in flat_scores[:keep_count]}
    masks = {}
    for idx, item in enumerate(score_items):
        mask = np.zeros(item["scores"].shape, dtype=np.float32)
        active_pos = np.flatnonzero(item["active"].reshape(-1))
        for pos in active_pos.tolist():
            if (idx, pos) in keep_set:
                mask.reshape(-1)[pos] = 1.0
        masks[id(item["layer"])] = mask
    return masks


def _saliency_loss(logits, labels):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return loss_fn(labels, logits)


def _normalize_scores(score_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    total = 0.0
    for arr in score_map.values():
        total += float(np.sum(np.abs(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))))
    denom = max(total, 1e-12)
    return {
        name: np.abs(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float32) / denom
        for name, arr in score_map.items()
    }


def _saliency_scores_snip(model, dense_layers, x_batch, y_batch):
    params = [layer.kernel for layer in dense_layers]
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=False)
        loss = _saliency_loss(logits, y_batch)
    grads = tape.gradient(loss, params)

    scores = {}
    for layer, grad in zip(dense_layers, grads):
        grad_np = np.zeros(tuple(layer.kernel.shape), dtype=np.float32) if grad is None else np.array(grad.numpy(), dtype=np.float32)
        w_np = np.array(layer.kernel.numpy(), dtype=np.float32)
        scores[layer.name] = np.abs(w_np * grad_np).astype(np.float32)
    return _normalize_scores(scores), {"loss": float(loss.numpy())}


def _saliency_scores_grasp(model, dense_layers, x_batch, y_batch):
    params = [layer.kernel for layer in dense_layers]
    n = int(x_batch.shape[0]) if x_batch.shape[0] is not None else int(tf.shape(x_batch)[0].numpy())
    mid = max(1, n // 2)
    if n >= 2:
        x_a, y_a = x_batch[:mid], y_batch[:mid]
        x_b, y_b = x_batch[mid:], y_batch[mid:]
        if int(x_b.shape[0]) == 0:
            x_b, y_b = x_a, y_a
    else:
        x_a = x_b = x_batch
        y_a = y_b = y_batch

    with tf.GradientTape() as ref_tape:
        logits_b = model(x_b, training=False)
        loss_b = _saliency_loss(logits_b, y_b)
    grads_b = ref_tape.gradient(loss_b, params)

    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            logits_a = model(x_a, training=False)
            loss_a = _saliency_loss(logits_a, y_a)
        grads_a = inner_tape.gradient(loss_a, params)
        z_terms = []
        for ga, gb in zip(grads_a, grads_b):
            if ga is None or gb is None:
                continue
            z_terms.append(tf.reduce_sum(ga * tf.stop_gradient(gb)))
        z = tf.add_n(z_terms) if z_terms else tf.constant(0.0, dtype=loss_a.dtype)
    hg = outer_tape.gradient(z, params)

    scores = {}
    for layer, hgv in zip(dense_layers, hg):
        hvp_np = np.zeros(tuple(layer.kernel.shape), dtype=np.float32) if hgv is None else np.array(hgv.numpy(), dtype=np.float32)
        w_np = np.array(layer.kernel.numpy(), dtype=np.float32)
        scores[layer.name] = (-w_np * hvp_np).astype(np.float32)
    return _normalize_scores(scores), {"loss_a": float(loss_a.numpy()), "loss_b": float(loss_b.numpy())}


def _saliency_scores_synflow(model, dense_layers, sample_input):
    params = [layer.kernel for layer in dense_layers]
    backups = []
    for layer in _flatten_layers(model):
        for attr in ("kernel", "bias"):
            var = getattr(layer, attr, None)
            if var is None:
                continue
            arr = np.array(var.numpy(), dtype=np.float32)
            backups.append((var, arr))
            var.assign(np.abs(arr).astype(np.float32))

    try:
        ones = tf.ones_like(sample_input)
        with tf.GradientTape() as tape:
            logits = model(ones, training=False)
            syn_obj = tf.reduce_sum(logits)
        grads = tape.gradient(syn_obj, params)
        scores = {}
        for layer, grad in zip(dense_layers, grads):
            grad_np = np.zeros(tuple(layer.kernel.shape), dtype=np.float32) if grad is None else np.array(grad.numpy(), dtype=np.float32)
            w_np = np.abs(np.array(layer.kernel.numpy(), dtype=np.float32))
            scores[layer.name] = np.abs(w_np * grad_np).astype(np.float32)
    finally:
        for var, arr in backups:
            var.assign(arr)

    return _normalize_scores(scores), {"synflow_objective": float(syn_obj.numpy())}


def saliency_prune_to_ebops(
    model,
    target_ebops: float,
    sample_input,
    method: str,
    input_h5: str | None = None,
    sample_size: int = 512,
    b_floor: float = 0.35,
    verbose: bool = True,
):
    """Saliency pruning baselines with algorithms closer to original papers.

    - SNIP: one-shot connection sensitivity on a supervised mini-batch.
    - GraSP: one-shot gradient-signal-preservation score with Hessian-gradient proxy.
    - SynFlow: iterative data-free pruning with linearized positive network.
    """
    method = str(method).lower()
    if method not in {"snip", "grasp", "synflow"}:
        raise ValueError(f"Unsupported saliency prune method: {method}")

    dense_layers = _dense_prunable_layers(model)
    if not dense_layers:
        return compute_model_ebops(model, sample_input), {"method": method, "status": "no_prunable_layers"}

    x_batch, y_batch, batch_src = build_prune_batch(
        model,
        sample_size=int(sample_size),
        input_h5=input_h5,
        teacher_model=model,
    )

    def collect_score_items(score_map):
        items = []
        total = 0
        for layer in dense_layers:
            scores = np.array(score_map[layer.name], dtype=np.float32)
            active = _active_kernel_mask_from_scores(layer, scores)
            items.append({"layer": layer, "scores": scores, "active": active})
            total += int(np.sum(active))
        return items, total

    snapshots = _snapshot_prunable_state(model)

    def apply_masks(masks_by_layer):
        _restore_prunable_state(snapshots)
        for item in masks_by_layer:
            _apply_mask_preserve_quant(item["layer"], item["mask"])
        return compute_model_ebops(model, sample_input)

    if method in {"snip", "grasp"}:
        if method == "snip":
            score_map, score_meta = _saliency_scores_snip(model, dense_layers, x_batch, y_batch)
        else:
            score_map, score_meta = _saliency_scores_grasp(model, dense_layers, x_batch, y_batch)

        score_items, total_active = collect_score_items(score_map)
        if total_active <= 0:
            cur = compute_model_ebops(model, sample_input)
            return cur, {
                "method": method,
                "batch_source": batch_src,
                "sample_size": int(x_batch.shape[0]),
                "target_ebops": float(target_ebops),
                "pre_bisect_ebops": float(cur),
                "kept_connections": 0,
                "total_active_connections": 0,
                "score_meta": score_meta,
                "layers": [],
                "status": "no_active_connections",
            }

        def measure_keep(keep_count: int):
            masks = _global_topk_masks(score_items, keep_count=keep_count)
            eb = apply_masks(
                [{"layer": item["layer"], "mask": masks[id(item["layer"])]} for item in score_items]
            )
            return eb, masks

        lo, hi = 0, total_active
        best_keep = total_active
        best_ebops = None
        best_masks = None
        while lo <= hi:
            mid = (lo + hi) // 2
            eb, masks = measure_keep(mid)
            if eb <= float(target_ebops):
                best_keep = mid
                best_ebops = eb
                best_masks = masks
                lo = mid + 1
            else:
                hi = mid - 1

        if best_ebops is None:
            best_keep = max(0, hi)
            best_ebops, best_masks = measure_keep(best_keep)
        else:
            apply_masks(
                [{"layer": item["layer"], "mask": best_masks[id(item["layer"])]} for item in score_items]
            )

        if verbose:
            print(
                f"[{method.upper()}] batch={batch_src}  keep={best_keep}/{total_active}  "
                f"pre_bisect_ebops={best_ebops:.1f}  target={float(target_ebops):.1f}"
            )

        layer_reports = []
        for item in score_items:
            kept_mask = best_masks[id(item["layer"])]
            score_vals = np.nan_to_num(item["scores"][item["active"]], nan=0.0, posinf=0.0, neginf=0.0)
            layer_reports.append(
                {
                    "layer": item["layer"].name,
                    "active_before": int(np.sum(item["active"])),
                    "active_after": int(np.sum(kept_mask > 0.5)),
                    "score_mean": float(np.mean(score_vals)) if score_vals.size else 0.0,
                    "score_max": float(np.max(score_vals)) if score_vals.size else 0.0,
                }
            )

        return float(best_ebops), {
            "method": method,
            "batch_source": batch_src,
            "sample_size": int(x_batch.shape[0]),
            "target_ebops": float(target_ebops),
            "pre_bisect_ebops": float(best_ebops),
            "kept_connections": int(best_keep),
            "total_active_connections": int(total_active),
            "score_meta": score_meta,
            "layers": layer_reports,
        }

    current_ebops = compute_model_ebops(model, sample_input)
    current_masks = {id(layer): _layer_active_mask(layer).astype(np.float32) for layer in dense_layers}
    initial_masks = {k: v.copy() for k, v in current_masks.items()}
    initial_active = int(sum(int(np.sum(m > 0.5)) for m in current_masks.values()))
    step_reports = []
    synflow_steps = 0
    while current_ebops > float(target_ebops):
        synflow_steps += 1
        score_map, score_meta = _saliency_scores_synflow(model, dense_layers, sample_input=x_batch)
        score_items, total_active = collect_score_items(score_map)
        if total_active <= 1:
            break

        overshoot_ratio = max(current_ebops / max(float(target_ebops), 1.0) - 1.0, 0.0)
        prune_frac = float(np.clip(0.02 + 0.20 * overshoot_ratio, 0.01, 0.20))
        drop_count = max(1, int(round(total_active * prune_frac)))
        keep_count = max(0, total_active - drop_count)
        masks = _global_topk_masks(score_items, keep_count=keep_count)
        current_ebops = apply_masks(
            [{"layer": item["layer"], "mask": masks[id(item["layer"])]} for item in score_items]
        )
        current_masks = masks
        step_reports.append(
            {
                "step": synflow_steps,
                "active_connections": int(keep_count),
                "pre_bisect_ebops": float(current_ebops),
                "prune_frac": float(prune_frac),
                "score_meta": score_meta,
            }
        )
        if verbose:
            print(
                f"[SYNFLOW] step={synflow_steps:02d}  prune_frac={prune_frac:.3f}  "
                f"active={keep_count}/{total_active}  ebops={current_ebops:.1f}"
            )
        if synflow_steps >= 64:
            break

    layer_reports = []
    total_active_after = 0
    for layer in dense_layers:
        mask = current_masks[id(layer)]
        total_active_after += int(np.sum(mask > 0.5))
        layer_reports.append(
            {
                "layer": layer.name,
                "active_before": int(np.sum(initial_masks[id(layer)] > 0.5)),
                "active_after": int(np.sum(mask > 0.5)),
            }
        )

    return float(current_ebops), {
        "method": method,
        "batch_source": batch_src,
        "sample_size": int(x_batch.shape[0]),
        "target_ebops": float(target_ebops),
        "pre_bisect_ebops": float(current_ebops),
        "kept_connections": int(total_active_after),
        "total_active_connections": int(initial_active),
        "iterations": int(synflow_steps),
        "layers": layer_reports,
        "steps": step_reports,
    }


def _conjugate_gradient_np(hvp_fn, b: np.ndarray, tol: float = 1e-4, max_iter: int = 25):
    x = np.zeros_like(b, dtype=np.float32)
    r = b.astype(np.float32, copy=True)
    p = r.copy()
    rs_old = float(np.dot(r, r))

    if (not np.isfinite(rs_old)) or rs_old <= tol * tol:
        return x, {"iters": 0, "residual": float(np.sqrt(max(rs_old, 0.0))), "converged": True}

    converged = False
    iters = 0
    for it in range(max(1, int(max_iter))):
        Ap = np.asarray(hvp_fn(p), dtype=np.float32)
        denom = float(np.dot(p, Ap))
        if (not np.isfinite(denom)) or abs(denom) < 1e-12:
            break

        alpha = rs_old / denom
        if not np.isfinite(alpha):
            break

        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(np.dot(r, r))
        iters = it + 1

        if (not np.isfinite(rs_new)) or np.sqrt(max(rs_new, 0.0)) <= tol:
            converged = np.isfinite(rs_new)
            rs_old = rs_new
            break

        beta = rs_new / max(rs_old, 1e-12)
        p = r + beta * p
        rs_old = rs_new

    return x, {
        "iters": int(iters),
        "residual": float(np.sqrt(max(rs_old, 0.0))),
        "converged": bool(converged),
    }


def _snows_make_loss_fn(model, layer_names, teacher_targets, sample_input):
    probe = keras.Model(model.inputs, [model.get_layer(n).output for n in layer_names])
    teacher_targets_tf = [tf.constant(np.array(t, dtype=np.float32)) for t in teacher_targets]

    def loss_fn():
        preds = probe(sample_input, training=False)
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        losses = []
        for pred, target in zip(preds, teacher_targets_tf):
            denom = tf.reduce_mean(tf.square(target)) + tf.constant(1e-6, dtype=pred.dtype)
            losses.append(tf.reduce_mean(tf.square(pred - target)) / denom)
        return tf.add_n(losses) / float(max(len(losses), 1))

    return loss_fn


def _snows_optimize_layer(
    model,
    layer,
    loss_fn,
    newton_steps: int,
    cg_max_iter: int,
    damping: float,
    line_search_decay: float,
    line_search_c: float,
    line_search_max_steps: int,
    verbose: bool,
):
    kernel_mask = _layer_active_mask(layer).astype(bool)
    if kernel_mask.ndim != 2 or int(np.sum(kernel_mask)) == 0:
        return {
            "layer": layer.name,
            "status": "skipped",
            "reason": "no_active_kernel",
        }

    vars_to_opt = [layer.kernel]
    infos = [{
        "name": f"{layer.name}.kernel",
        "shape": tuple(layer.kernel.shape),
        "mask_bool": kernel_mask,
        "mask_float": kernel_mask.astype(np.float32),
        "size": int(np.sum(kernel_mask)),
    }]

    bias = getattr(layer, "bias", None)
    if bias is not None:
        bias_mask = _bias_active_mask(layer, fallback_mask=(np.sum(kernel_mask, axis=0) > 0))
        if bias_mask is not None and int(np.sum(bias_mask)) > 0:
            vars_to_opt.append(bias)
            infos.append({
                "name": f"{layer.name}.bias",
                "shape": tuple(bias.shape),
                "mask_bool": bias_mask.astype(bool),
                "mask_float": bias_mask.astype(np.float32),
                "size": int(np.sum(bias_mask)),
            })

    total_active = int(sum(info["size"] for info in infos))
    if total_active <= 0:
        return {
            "layer": layer.name,
            "status": "skipped",
            "reason": "no_active_params",
        }

    def pack(parts):
        flat = []
        for part, info in zip(parts, infos):
            arr = np.array(part.numpy() if hasattr(part, "numpy") else part, dtype=np.float32)
            if info["size"] > 0:
                flat.append(arr[info["mask_bool"]].reshape(-1))
        return np.concatenate(flat, axis=0) if flat else np.zeros((0,), dtype=np.float32)

    def unpack(vec):
        parts = []
        offset = 0
        for info, var in zip(infos, vars_to_opt):
            full = np.zeros(info["shape"], dtype=np.float32)
            n = info["size"]
            if n > 0:
                full[info["mask_bool"]] = vec[offset:offset + n]
                offset += n
            parts.append(tf.convert_to_tensor(full, dtype=var.dtype))
        return parts

    base_loss = float(loss_fn().numpy())
    report = {
        "layer": layer.name,
        "status": "ok",
        "active_params": total_active,
        "loss_before": base_loss,
        "loss_after": base_loss,
        "steps_run": 0,
        "cg_iters_last": 0,
        "line_search_alpha_last": 0.0,
    }

    for step in range(max(1, int(newton_steps))):
        with tf.GradientTape(persistent=True) as outer_tape:
            with tf.GradientTape() as inner_tape:
                loss = loss_fn()
            grads = inner_tape.gradient(loss, vars_to_opt)
            masked_grads = []
            for grad, var, info in zip(grads, vars_to_opt, infos):
                if grad is None:
                    grad = tf.zeros_like(var)
                masked_grads.append(grad * tf.convert_to_tensor(info["mask_float"], dtype=grad.dtype))

        grad_vec = pack(masked_grads)
        grad_norm = float(np.linalg.norm(grad_vec))
        if (not np.isfinite(grad_norm)) or grad_norm < 1e-8:
            del outer_tape
            break

        def hvp_fn(vec_np: np.ndarray):
            vec_parts = unpack(vec_np)
            dot_terms = [
                tf.reduce_sum(g * tf.stop_gradient(v))
                for g, v in zip(masked_grads, vec_parts)
            ]
            dot = tf.add_n(dot_terms) if dot_terms else tf.constant(0.0, dtype=loss.dtype)
            hvps = outer_tape.gradient(dot, vars_to_opt)
            packed = []
            for hvp, vec_part, var, info in zip(hvps, vec_parts, vars_to_opt, infos):
                if hvp is None:
                    hvp = tf.zeros_like(var)
                masked = (hvp + tf.cast(damping, hvp.dtype) * vec_part) * tf.convert_to_tensor(
                    info["mask_float"], dtype=hvp.dtype
                )
                packed.append(np.array(masked.numpy(), dtype=np.float32)[info["mask_bool"]].reshape(-1))
            return np.concatenate(packed, axis=0) if packed else np.zeros((0,), dtype=np.float32)

        step_vec, cg_info = _conjugate_gradient_np(
            hvp_fn,
            -grad_vec,
            tol=max(1e-6, grad_norm * 1e-2),
            max_iter=cg_max_iter,
        )

        if step_vec.size == 0 or not np.all(np.isfinite(step_vec)):
            del outer_tape
            break

        if float(np.dot(grad_vec, step_vec)) >= 0.0:
            step_vec = -grad_vec

        step_parts = unpack(step_vec)
        base_values = [np.array(var.numpy(), dtype=np.float32) for var in vars_to_opt]
        direction_dot = 0.0
        for grad, step_part in zip(masked_grads, step_parts):
            direction_dot += float(np.sum(np.array(grad.numpy(), dtype=np.float32) * np.array(step_part.numpy(), dtype=np.float32)))

        current_loss = float(loss.numpy())
        accepted = False
        alpha = 1.0
        trial_loss = current_loss
        for _ in range(max(1, int(line_search_max_steps))):
            for var, base, step_part in zip(vars_to_opt, base_values, step_parts):
                delta = np.array(step_part.numpy(), dtype=np.float32)
                var.assign((base + alpha * delta).astype(np.float32))

            trial_loss = float(loss_fn().numpy())
            if trial_loss <= current_loss + line_search_c * alpha * direction_dot:
                accepted = True
                break
            alpha *= line_search_decay

        if not accepted:
            for var, base in zip(vars_to_opt, base_values):
                var.assign(base.astype(np.float32))
            del outer_tape
            break

        report["steps_run"] = int(step + 1)
        report["cg_iters_last"] = int(cg_info["iters"])
        report["line_search_alpha_last"] = float(alpha)
        report["loss_after"] = float(trial_loss)

        if verbose:
            print(
                f"  [SNOWS] {layer.name:20s} step={step+1}/{newton_steps}  "
                f"loss={current_loss:.6f}->{trial_loss:.6f}  "
                f"grad_norm={grad_norm:.4e}  cg_iters={cg_info['iters']}  alpha={alpha:.3f}"
            )

        del outer_tape

    report["loss_after"] = float(loss_fn().numpy())
    return report


def snows_prune_to_ebops(
    model,
    teacher_model,
    target_ebops: float,
    sample_input,
    init_method: str = "sensitivity",
    b_floor: float = 0.30,
    k_step: int = 2,
    newton_steps: int = 2,
    cg_max_iter: int = 25,
    damping: float = 1e-4,
    line_search_decay: float = 0.5,
    line_search_c: float = 1e-4,
    line_search_max_steps: int = 8,
    verbose: bool = True,
):
    """SNOWS-style one-shot pruning for HGQ.

    先用现有预算分配器生成低 EBOPs 初始点，再在固定 bit/mask 上以 K-step
    深层表征重建目标，对每层做 Hessian-free Newton 更新。
    """
    if init_method not in {"sensitivity", "uniform", "spectral_quant"}:
        raise ValueError(f"Unsupported SNOWS init_method: {init_method}")

    current_ebops = compute_model_ebops(model, sample_input)
    used_structured_low_budget = False
    if verbose:
        print(
            f"[SNOWS] init_method={init_method}  current_ebops={current_ebops:.1f}  "
            f"target_ebops={target_ebops:.1f}  k_step={k_step}"
        )

    if init_method == "sensitivity":
        pruner = SensitivityAwarePruner(
            target_ebops=float(target_ebops),
            pruned_threshold=0.1,
            b_k_min=max(float(b_floor), 0.20),
        )
        pruner.prune_to_ebops(model, current_ebops=current_ebops, verbose=verbose)
    elif init_method == "uniform":
        pruner = HighBitPruner(target_ebops=float(target_ebops), pruned_threshold=0.1)
        pruner.prune_to_ebops(model, current_ebops=current_ebops, verbose=verbose)
    else:
        _, used_structured_low_budget = spectral_quant_prune_to_ebops(
            model,
            target_ebops=float(target_ebops),
            sample_input=sample_input,
            min_degree=2,
            b_floor=b_floor,
            low_budget_structured=True,
            low_budget_threshold=900.0,
            min_hidden_width=4,
            near_budget_ratio=1.6,
            high_budget_ratio=0.45,
            verbose=verbose,
        )

    init_pruned_ebops = compute_model_ebops(model, sample_input)
    dense_layers = _dense_prunable_layers(model)
    layer_names = [layer.name for layer in dense_layers]
    teacher_outputs = _collect_named_outputs(teacher_model, layer_names, sample_input, training=False)

    reports = []
    for idx, layer in enumerate(dense_layers):
        horizon_names = layer_names[idx:min(len(layer_names), idx + max(1, int(k_step)) + 1)]
        teacher_targets = [teacher_outputs[name] for name in horizon_names]
        loss_fn = _snows_make_loss_fn(model, horizon_names, teacher_targets, sample_input)
        before = float(loss_fn().numpy())
        rep = _snows_optimize_layer(
            model=model,
            layer=layer,
            loss_fn=loss_fn,
            newton_steps=newton_steps,
            cg_max_iter=cg_max_iter,
            damping=damping,
            line_search_decay=line_search_decay,
            line_search_c=line_search_c,
            line_search_max_steps=line_search_max_steps,
            verbose=verbose,
        )
        rep["horizon"] = list(horizon_names)
        rep["loss_before"] = before
        rep["loss_after"] = float(loss_fn().numpy())
        reports.append(rep)

    final_ebops = compute_model_ebops(model, sample_input)
    total_before = float(sum(float(r.get("loss_before", 0.0)) for r in reports))
    total_after = float(sum(float(r.get("loss_after", 0.0)) for r in reports))
    if verbose:
        print(
            f"[SNOWS] init_pruned_ebops={init_pruned_ebops:.1f}  final_ebops={final_ebops:.1f}  "
            f"repr_loss_sum={total_before:.6f}->{total_after:.6f}"
        )

    return final_ebops, {
        "init_method": init_method,
        "k_step": int(k_step),
        "newton_steps": int(newton_steps),
        "cg_max_iter": int(cg_max_iter),
        "damping": float(damping),
        "used_structured_low_budget": bool(used_structured_low_budget),
        "init_pruned_ebops": float(init_pruned_ebops),
        "final_ebops": float(final_ebops),
        "representation_loss_before_sum": total_before,
        "representation_loss_after_sum": total_after,
        "layers": reports,
    }


def teacher_guided_post_prune_calibration(
    student_model,
    teacher_model,
    sample_input,
    passes: int = 2,
    b_floor: float = 0.35,
    b_ceiling: float = 6.0,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
    shift_clip: float = 8.0,
    verbose: bool = True,
):
    """剪枝后功能校准：逐层输出通道做仿射对齐，提升低预算可训练性。"""
    s_layers = _dense_prunable_layers(student_model)
    if not s_layers:
        return compute_model_ebops(student_model, sample_input)

    layer_names = []
    for l in s_layers:
        try:
            teacher_model.get_layer(l.name)
            layer_names.append(l.name)
        except Exception:
            continue
    if not layer_names:
        return compute_model_ebops(student_model, sample_input)

    for p in range(max(1, int(passes))):
        y_teacher = _collect_named_outputs(teacher_model, layer_names, sample_input, training=False)
        y_student = _collect_named_outputs(student_model, layer_names, sample_input, training=False)
        changes = []

        for name in layer_names:
            layer = student_model.get_layer(name)
            ys = y_student[name]
            yt = y_teacher[name]
            if ys.shape != yt.shape:
                continue

            c = ys.shape[-1]
            ys2 = ys.reshape(-1, c)
            yt2 = yt.reshape(-1, c)
            if ys2.shape[0] < 2:
                continue

            active = _layer_active_mask(layer)
            active_cols = np.sum(active, axis=0) > 0
            if not np.any(active_cols):
                continue

            mu_s = np.mean(ys2, axis=0)
            mu_t = np.mean(yt2, axis=0)
            zs = ys2 - mu_s
            zt = yt2 - mu_t
            var_s = np.mean(zs * zs, axis=0)
            cov_st = np.mean(zs * zt, axis=0)

            scale = cov_st / (var_s + 1e-8)
            scale = np.where(np.isfinite(scale), scale, 1.0).astype(np.float32)
            scale = np.clip(scale, scale_min, scale_max)
            shift = mu_t - scale * mu_s
            shift = np.clip(shift, -shift_clip, shift_clip).astype(np.float32)

            k = np.array(layer.kernel.numpy(), dtype=np.float32)
            k_new = k.copy()
            k_new[:, active_cols] = k[:, active_cols] * scale[active_cols][None, :]
            k_new[:, ~active_cols] = 0.0
            layer.kernel.assign(k_new.astype(np.float32))

            b = getattr(layer, "bias", None)
            if b is not None:
                b_old = np.array(b.numpy(), dtype=np.float32)
                b_new = b_old.copy()
                b_new[active_cols] = b_old[active_cols] * scale[active_cols] + shift[active_cols]
                b_new[~active_cols] = 0.0
                b.assign(b_new.astype(np.float32))

            changes.append(float(np.mean(np.abs(scale[active_cols] - 1.0))))

        _reestimate_quantizer_ranges(student_model, b_floor=b_floor, b_ceiling=b_ceiling)
        if verbose and changes:
            print(
                f"  [TeacherCalib] pass {p+1}/{passes}  "
                f"mean|scale-1|={float(np.mean(changes)):.4f}  layers={len(changes)}"
            )

    return compute_model_ebops(student_model, sample_input)


def structured_chain_prune_to_ebops(
    model,
    target_ebops: float,
    sample_input,
    min_hidden_width: int = 4,
    b_floor: float = 0.35,
    b_ceiling: float = 6.0,
    verbose: bool = True,
):
    """低预算结构化剪枝：先压缩隐藏宽度，保持层间链式连通，再设量化参数。"""
    layers = _dense_prunable_layers(model)
    if len(layers) < 2:
        measured = compute_model_ebops(model, sample_input)
        return measured

    weights = [np.array(layer.kernel.numpy(), dtype=np.float32) for layer in layers]

    # 每层单位连接 EBOPs 成本（来自当前模型实测）
    from keras import ops
    _forward_update_ebops_no_bn_drift(model, sample_input)
    edge_costs = []
    for layer in layers:
        if getattr(layer, "_ebops", None) is not None:
            layer_ebops = float(int(ops.convert_to_numpy(layer._ebops)))
        else:
            layer_ebops = 0.0
        active = _layer_active_mask(layer)
        n_active = int(np.sum(active))
        if n_active <= 0:
            n_active = int(np.prod(layer.kernel.shape))
        edge_costs.append(layer_ebops / max(n_active, 1))

    # widths: [input_dim, hidden_1, ..., hidden_k, output_dim]
    widths = [int(weights[0].shape[0])] + [int(w.shape[1]) for w in weights]
    hidden_idx = list(range(1, len(widths) - 1))

    if not hidden_idx:
        measured = compute_model_ebops(model, sample_input)
        return measured

    saliency = {}
    remove_order = {}
    keep_map = {}
    removed_ptr = {}
    min_keep = {}

    for h in hidden_idx:
        prev_w = np.abs(weights[h - 1])
        next_w = np.abs(weights[h])
        s = np.mean(prev_w, axis=0) * np.mean(next_w, axis=1)
        saliency[h] = s
        remove_order[h] = np.argsort(s).tolist()  # 弱 -> 强
        keep_map[h] = np.ones_like(s, dtype=bool)
        removed_ptr[h] = 0
        min_keep[h] = int(max(1, min(min_hidden_width, len(s))))

    def predict_ebops(curr_widths):
        total = 0.0
        for li, c in enumerate(edge_costs):
            total += c * curr_widths[li] * curr_widths[li + 1]
        return total

    # 贪心删弱神经元：每次优先删“谱贡献低 / 预算收益高”的隐藏单元
    n_guard = 0
    while predict_ebops(widths) > target_ebops * 1.15:
        n_guard += 1
        if n_guard > 10000:
            break
        best = None  # (score, h, neuron)
        for h in hidden_idx:
            if widths[h] <= min_keep[h]:
                continue
            ptr = removed_ptr[h]
            order = remove_order[h]
            while ptr < len(order) and (not keep_map[h][order[ptr]]):
                ptr += 1
            removed_ptr[h] = ptr
            if ptr >= len(order):
                continue
            n = order[ptr]
            penalty = float(saliency[h][n]) + 1e-9
            # 删除隐藏单元 h，对两侧层都省预算
            delta = edge_costs[h - 1] * widths[h - 1] + edge_costs[h] * widths[h + 1]
            score = penalty / max(delta, 1e-9)
            if (best is None) or (score < best[0]):
                best = (score, h, n)

        if best is None:
            break

        _, h, n = best
        if keep_map[h][n]:
            keep_map[h][n] = False
            widths[h] -= 1
        removed_ptr[h] += 1

    # 应用结构化 mask 到每层
    for li, layer in enumerate(layers):
        w = np.array(layer.kernel.numpy(), dtype=np.float32)
        in_dim, out_dim = w.shape
        if li == 0:
            rows_keep = np.ones(in_dim, dtype=bool)
        else:
            rows_keep = keep_map[li]
        if li == len(layers) - 1:
            cols_keep = np.ones(out_dim, dtype=bool)
        else:
            cols_keep = keep_map[li + 1]

        mask = np.outer(rows_keep, cols_keep).astype(np.float32)
        _apply_mask_and_quant(layer, mask, b_floor=b_floor, b_ceiling=b_ceiling)

        if verbose:
            print(
                f"  [StructuredLowBudget] {layer.name:20s}  "
                f"in_keep={int(np.sum(rows_keep))}/{in_dim}  "
                f"out_keep={int(np.sum(cols_keep))}/{out_dim}  "
                f"active={int(np.sum(mask))}"
            )

    measured = compute_model_ebops(model, sample_input)
    if verbose:
        w_desc = ", ".join([str(widths[i]) for i in range(1, len(widths) - 1)])
        print(
            f"[StructuredLowBudget] hidden_widths=[{w_desc}]  "
            f"pred_ebops={predict_ebops(widths):.1f}  measured_ebops={measured:.1f}  "
            f"target={target_ebops:.1f}"
        )
    return measured


def spectral_quant_prune_to_ebops(
    model,
    target_ebops: float,
    sample_input,
    min_degree: int = 2,
    b_floor: float = 0.25,
    b_ceiling: float = 6.0,
    low_budget_structured: bool = True,
    low_budget_threshold: float = 900.0,
    min_hidden_width: int = 4,
    near_budget_ratio: float = 1.6,
    high_budget_ratio: float = 0.45,
    verbose: bool = True,
):
    """谱/拓扑友好的一次性剪枝：低预算走结构化链路，其余走列 top-k。"""
    current_ebops = compute_model_ebops(model, sample_input)
    budget_ratio = float(target_ebops) / max(float(current_ebops), 1.0)
    if current_ebops <= float(target_ebops) * float(near_budget_ratio):
        if verbose:
            print(
                f"[SpectralQuantPruner] near-budget preserve mode: "
                f"current={current_ebops:.1f}, target={target_ebops:.1f}, "
                f"ratio={current_ebops / max(target_ebops, 1.0):.3f}"
            )
        return current_ebops, False

    # 高预算场景只需要温和压缩，避免 top-k 重布线带来不必要的表示退化
    if budget_ratio >= float(high_budget_ratio):
        if verbose:
            print(
                f"[SpectralQuantPruner] high-budget fallback to sensitivity: "
                f"target/current={budget_ratio:.3f} >= {high_budget_ratio:.3f}"
            )
        pruner = SensitivityAwarePruner(
            target_ebops=float(target_ebops),
            pruned_threshold=0.1,
            b_k_min=max(float(b_floor), 0.20),
        )
        pruner.prune_to_ebops(model, current_ebops=current_ebops, verbose=verbose)
        measured = compute_model_ebops(model, sample_input)
        return measured, False

    if low_budget_structured and float(target_ebops) <= float(low_budget_threshold):
        measured = structured_chain_prune_to_ebops(
            model,
            target_ebops=float(target_ebops),
            sample_input=sample_input,
            min_hidden_width=min_hidden_width,
            b_floor=b_floor,
            b_ceiling=b_ceiling,
            verbose=verbose,
        )
        return measured, True

    # 用预算估计每层目标度与建议位宽（capacity 加权）
    per_layer_degree, per_layer_bk = compute_bw_aware_degree(
        model,
        target_ebops=float(target_ebops),
        b_a_init=3.0,
        b_k_min=b_floor,
        b_k_max=b_ceiling,
        multiplier=1.2,
        min_degree=min_degree,
        budget_weight="capacity",
        verbose=False,
    )

    for layer in _flatten_layers(model):
        if not hasattr(layer, "kernel") or getattr(layer, "kq", None) is None:
            continue
        name = layer.name
        kq = layer.kq
        kernel = layer.kernel.numpy().astype(np.float32)

        # 仅处理 2D kernel（当前模型为 Dense/EinsumDense 系列）
        if kernel.ndim != 2:
            continue
        in_dim, out_dim = kernel.shape
        d = int(per_layer_degree.get(name, min_degree))
        d = int(np.clip(d, min_degree, in_dim))

        mask = _build_topk_mask_with_connectivity(kernel, d)
        layer.kernel.assign((kernel * mask).astype(np.float32))

        # 量化参数：活跃连接保底位宽，避免直接落入量化死区
        b_var = _get_kq_var(kq, "b")
        if b_var is None:
            b_var = _get_kq_var(kq, "f")
        i_var = _get_kq_var(kq, "i")
        k_var = _get_kq_var(kq, "k")

        if b_var is not None:
            b_old = b_var.numpy().astype(np.float32)
            abs_k = np.abs(kernel)
            denom = float(np.max(abs_k)) + 1e-12
            importance = abs_k / denom
            b_base = float(np.clip(per_layer_bk.get(name, b_floor), b_floor, b_ceiling))
            b_active = np.clip(np.maximum(b_old, b_base) * (0.8 + 0.4 * importance), b_floor, b_ceiling)
            b_new = np.where(mask > 0.5, b_active, 0.0)
            b_var.assign(b_new.astype(np.float32))

        if i_var is not None:
            i_old = i_var.numpy().astype(np.float32)
            i_new = np.where(mask > 0.5, np.clip(i_old, -2.0, 6.0), -16.0)
            i_var.assign(i_new.astype(np.float32))

        if k_var is not None:
            k_new = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
            k_var.assign(k_new)

        # bias quantizer 保留适度精度，避免偏置全部进入死区
        bq = getattr(layer, "bq", None)
        if bq is not None:
            bb_var = _get_kq_var(bq, "b")
            if bb_var is None:
                bb_var = _get_kq_var(bq, "f")
            bi_var = _get_kq_var(bq, "i")
            bk_var = _get_kq_var(bq, "k")
            if bb_var is not None:
                bb_old = bb_var.numpy().astype(np.float32)
                bb_new = np.clip(np.maximum(bb_old, b_floor), b_floor, b_ceiling)
                bb_var.assign(bb_new.astype(np.float32))
            if bi_var is not None:
                bi_old = bi_var.numpy().astype(np.float32)
                bi_var.assign(np.clip(bi_old, -2.0, 6.0).astype(np.float32))
            if bk_var is not None:
                bk_var.assign(np.ones_like(bk_var.numpy(), dtype=np.float32))

        if verbose:
            col_deg = np.sum(mask, axis=0)
            row_deg = np.sum(mask, axis=1)
            print(
                f"  [SpectralQuantPruner] {name:20s}  d={d}  "
                f"deg_col_mean={float(np.mean(col_deg)):.2f}  "
                f"deg_row_zero={int(np.sum(row_deg==0))}"
            )

    measured = compute_model_ebops(model, sample_input)
    if verbose:
        print(f"[SpectralQuantPruner] post-structural measured_ebops={measured:.1f}  target={target_ebops:.1f}")
    return measured, False


def bisect_ebops_to_target(
    model,
    target_ebops,
    sample_input,
    tolerance=0.02,
    max_iter=30,
    b_k_min=0.01,
    b_k_max=8.0,
    allow_connection_kill=True,
):
    """用二分搜索校准剪枝后位宽，使实测 EBOPs 逼近 target。"""
    snapshots = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None:
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is None:
            continue
        b_snap = b_var.numpy().astype(np.float32).copy()
        k_var = _get_kq_var(kq, 'k')
        k_snap = k_var.numpy().astype(np.float32).copy() if k_var is not None else None
        # bits 常含符号位 k：仅用 b>0 会漏掉仍有成本的连接
        if k_snap is not None:
            active_mask = (b_snap > 0.0) | (k_snap > 0.0)
        else:
            active_mask = b_snap > 0.0
        # b 只对原本有小数位的连接做缩放；其余保持 0
        b_scale_mask = b_snap > 0.0
        # 同时记录 kq.i；死连接 i 置极负，避免残余成本
        i_var = _get_kq_var(kq, 'i')
        i_snap = i_var.numpy().astype(np.float32).copy() if i_var is not None else None
        # bias quantizer 也会计入 ebops，需要同步校准
        bq = getattr(layer, 'bq', None)
        bb_var = _get_kq_var(bq, 'b') if bq is not None else None
        if bb_var is None and bq is not None:
            bb_var = _get_kq_var(bq, 'f')
        bb_snap = bb_var.numpy().astype(np.float32).copy() if bb_var is not None else None
        bk_var = _get_kq_var(bq, 'k') if bq is not None else None
        bk_snap = bk_var.numpy().astype(np.float32).copy() if bk_var is not None else None
        bi_var = _get_kq_var(bq, 'i') if bq is not None else None
        bi_snap = bi_var.numpy().astype(np.float32).copy() if bi_var is not None else None
        if bb_snap is not None:
            if bk_snap is not None:
                bactive_mask = (bb_snap > 0.0) | (bk_snap > 0.0)
            else:
                bactive_mask = bb_snap > 0.0
            bb_scale_mask = bb_snap > 0.0
        else:
            bactive_mask = None
            bb_scale_mask = None
        snapshots.append({
            'b_var': b_var,
            'b_snap': b_snap,
            'b_scale_mask': b_scale_mask,
            'active_mask': active_mask,
            'i_var': i_var,
            'i_snap': i_snap,
            'k_var': k_var,
            'k_snap': k_snap,
            'bb_var': bb_var,
            'bb_snap': bb_snap,
            'bb_scale_mask': bb_scale_mask,
            'bactive_mask': bactive_mask,
            'bi_var': bi_var,
            'bi_snap': bi_snap,
            'bk_var': bk_var,
            'bk_snap': bk_snap,
        })

    if not snapshots:
        return 0.0

    def apply_scale(scale: float):
        for s in snapshots:
            b_var = s['b_var']
            b_snap = s['b_snap']
            b_scale_mask = s['b_scale_mask']
            active_mask = s['active_mask']
            i_var = s['i_var']
            i_snap = s['i_snap']
            k_var = s['k_var']
            k_snap = s['k_snap']

            b_new = np.where(
                b_scale_mask,
                np.clip(b_snap * scale, b_k_min, b_k_max),
                0.0,
            )
            b_var.assign(b_new.astype(np.float32))

            if i_var is not None and i_snap is not None:
                i_new = np.where(active_mask, i_snap, -16.0)
                i_var.assign(i_new.astype(np.float32))
            # 死连接的 kq.k 置 0，去掉符号位成本
            if k_var is not None and k_snap is not None:
                k_new = np.where(active_mask, k_snap, 0.0)
                k_var.assign(k_new.astype(np.float32))
            bb_var = s['bb_var']
            bb_snap = s['bb_snap']
            bb_scale_mask = s['bb_scale_mask']
            bactive_mask = s['bactive_mask']
            bi_var = s['bi_var']
            bi_snap = s['bi_snap']
            bk_var = s['bk_var']
            bk_snap = s['bk_snap']
            if bb_var is not None and bb_snap is not None and bb_scale_mask is not None and bactive_mask is not None:
                bb_new = np.where(
                    bb_scale_mask,
                    np.clip(bb_snap * scale, b_k_min, b_k_max),
                    0.0,
                )
                bb_var.assign(bb_new.astype(np.float32))
                if bi_var is not None and bi_snap is not None:
                    bi_new = np.where(bactive_mask, bi_snap, -16.0)
                    bi_var.assign(bi_new.astype(np.float32))
                if bk_var is not None and bk_snap is not None:
                    bk_new = np.where(bactive_mask, bk_snap, 0.0)
                    bk_var.assign(bk_new.astype(np.float32))

    def measure(scale: float) -> float:
        apply_scale(scale)
        return compute_model_ebops(model, sample_input)

    e_1 = measure(1.0)
    if abs(e_1 - target_ebops) / max(target_ebops, 1.0) <= tolerance:
        return e_1

    if e_1 < target_ebops:
        lo, hi = 1.0, 2.0
        lo_e, hi_e = e_1, measure(hi)
        while hi_e < target_ebops and hi < 1e4:
            lo, lo_e = hi, hi_e
            hi *= 2.0
            hi_e = measure(hi)
        if hi_e < target_ebops:
            apply_scale(hi)
            final_e = compute_model_ebops(model, sample_input)
            print(
                f"  [BisectEBOPs] max-reachable={final_e:.1f} < target={target_ebops:.1f} "
                f"(scale upper bound={hi:.1f})"
            )
            return final_e
    else:
        lo, hi = 0.5, 1.0
        lo_e, hi_e = measure(lo), e_1
        while lo_e > target_ebops and lo > 1e-6:
            hi, hi_e = lo, lo_e
            lo /= 2.0
            lo_e = measure(lo)
        if lo_e > target_ebops:
            if not allow_connection_kill:
                apply_scale(lo)
                final_e = compute_model_ebops(model, sample_input)
                print(
                    f"  [BisectEBOPs] connectivity-preserving floor={final_e:.1f} > "
                    f"target={target_ebops:.1f}; skip weakest-connection killing."
                )
                return final_e
            print(
                f"  [BisectEBOPs] ebops floor={lo_e:.1f} > target={target_ebops:.1f}, "
                f"pruning weakest connections..."
            )
            # 收集所有仍有成本的连接（b>0 或 k>0），按“弱连接优先”剔除
            all_entries = []  # (snap_idx, flat_pos, score)
            for snap_idx, s in enumerate(snapshots):
                b_flat = s['b_snap'].ravel()
                active_flat = s['active_mask'].ravel()
                k_snap = s['k_snap']
                k_flat = k_snap.ravel() if k_snap is not None else None
                for pos in range(len(b_flat)):
                    if not active_flat[pos]:
                        continue
                    score = float(b_flat[pos])
                    if k_flat is not None:
                        score += 1e-4 * float(k_flat[pos])
                    all_entries.append((snap_idx, pos, score))
                bb_snap = s['bb_snap']
                bactive_mask = s['bactive_mask']
                bk_snap = s['bk_snap']
                if bb_snap is not None and bactive_mask is not None:
                    bb_flat = bb_snap.ravel()
                    bactive_flat = bactive_mask.ravel()
                    bk_flat = bk_snap.ravel() if bk_snap is not None else None
                    base_off = 10_000_000
                    for pos in range(len(bb_flat)):
                        if not bactive_flat[pos]:
                            continue
                        score = float(bb_flat[pos])
                        if bk_flat is not None:
                            score += 1e-4 * float(bk_flat[pos])
                        # 用大偏移区分 weight/bias 索引空间
                        all_entries.append((snap_idx, base_off + pos, score))

            all_entries.sort(key=lambda x: x[2])
            if not all_entries:
                final_e = measure(1.0)
                print(
                    f"  [BisectEBOPs] no removable connections, "
                    f"min-reachable={final_e:.1f} (target={target_ebops:.1f})"
                )
                return final_e

            # 每次杀掉 2% 的连接，避免过冲
            kill_step = max(1, len(all_entries) // 50)
            killed = 0
            for i in range(0, len(all_entries), kill_step):
                batch = all_entries[i:i + kill_step]
                for snap_idx, pos, _ in batch:
                    s = snapshots[snap_idx]
                    if pos >= 10_000_000:
                        if s['bb_snap'] is None or s['bb_scale_mask'] is None or s['bactive_mask'] is None:
                            continue
                        bpos = pos - 10_000_000
                        bidx = np.unravel_index(bpos, s['bb_snap'].shape)
                        s['bb_snap'][bidx] = 0.0
                        s['bb_scale_mask'][bidx] = False
                        s['bactive_mask'][bidx] = False
                        if s['bi_snap'] is not None:
                            s['bi_snap'][bidx] = -16.0
                        if s['bk_snap'] is not None:
                            s['bk_snap'][bidx] = 0.0
                    else:
                        idx = np.unravel_index(pos, s['b_snap'].shape)
                        s['b_snap'][idx] = 0.0
                        s['b_scale_mask'][idx] = False
                        s['active_mask'][idx] = False
                        if s['i_snap'] is not None:
                            s['i_snap'][idx] = -16.0
                        if s['k_snap'] is not None:
                            s['k_snap'][idx] = 0.0
                killed += len(batch)

                # 重新应用并测量（同时回写 kq.b/kq.i/kq.k）
                check_e = measure(1.0)

                if check_e <= target_ebops * (1.0 + tolerance):
                    print(
                        f"  [BisectEBOPs] killed {killed}/{len(all_entries)} weakest "
                        f"connections, ebops now={check_e:.1f}"
                    )
                    if abs(check_e - target_ebops) / max(target_ebops, 1.0) <= tolerance:
                        return check_e
                    break
            else:
                final_e = measure(1.0)
                print(
                    f"  [BisectEBOPs] killed all {killed} connections, "
                    f"ebops={final_e:.1f} (target={target_ebops:.1f})"
                )
                return final_e

            # 重新找二分区间
            lo_e = measure(1.0)
            if abs(lo_e - target_ebops) / max(target_ebops, 1.0) <= tolerance:
                return lo_e
            if lo_e < target_ebops:
                lo, hi = 1.0, 2.0
                hi_e = measure(hi)
                while hi_e < target_ebops and hi < 1e4:
                    lo, lo_e = hi, hi_e
                    hi *= 2.0
                    hi_e = measure(hi)
            else:
                lo, hi = 0.5, 1.0
                hi_e = lo_e
                lo_e = measure(lo)
                while lo_e > target_ebops and lo > 1e-6:
                    hi, hi_e = lo, lo_e
                    lo /= 2.0
                    lo_e = measure(lo)

    best_s, best_e = (lo, lo_e) if abs(lo_e - target_ebops) <= abs(hi_e - target_ebops) else (hi, hi_e)

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        mid_e = measure(mid)
        err = abs(mid_e - target_ebops) / max(target_ebops, 1.0)

        if err < abs(best_e - target_ebops) / max(target_ebops, 1.0):
            best_s, best_e = mid, mid_e

        if err <= tolerance:
            break

        if mid_e < target_ebops:
            lo = mid
        else:
            hi = mid

    apply_scale(best_s)
    final_e = compute_model_ebops(model, sample_input)
    print(
        f"  [BisectEBOPs] scale={best_s:.5f}  final_ebops={final_e:.1f}  "
        f"target={target_ebops:.1f}  err={abs(final_e-target_ebops)/max(target_ebops,1.0)*100:.1f}%"
    )
    return final_e


def build_sample_input(model, sample_size: int, input_h5: str | None):
    """优先使用数据集样本；若不可用则回退到 synthetic input。"""
    if input_h5 and os.path.exists(input_h5):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_h5, src='openml')
        del X_train, y_train, X_test, y_test
        n = min(sample_size, len(X_val))
        return tf.constant(X_val[:n], dtype=tf.float32), f"dataset:{input_h5}"

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape is None:
        raise RuntimeError('Cannot infer model.input_shape for synthetic sample input.')

    dyn_shape = [sample_size]
    for d in input_shape[1:]:
        dyn_shape.append(1 if d is None else int(d))

    x = np.random.randn(*dyn_shape).astype(np.float32)
    return tf.constant(x), f"synthetic:{tuple(dyn_shape)}"


def main():
    parser = argparse.ArgumentParser(description='One-shot prune only: prune + forward EBOPs + save weights')
    parser.add_argument('--checkpoint', type=str,
                        default='results/baseline_copy/epoch=3699-val_acc=0.770-ebops=23293-val_loss=0.640.keras')
    parser.add_argument('--target_ebops', type=float, default=200.0)
    parser.add_argument('--prune_method', type=str, default='sensitivity',
                        choices=['uniform', 'sensitivity', 'snip', 'grasp', 'synflow', 'spectral_quant', 'spectral_path', 'snows'])
    parser.add_argument('--input_h5', type=str, default='data/dataset.h5')
    parser.add_argument('--sample_size', type=int, default=512)
    parser.add_argument('--calibrate', action='store_true', default=True,
                        help='Enable bisection calibration after one-shot prune (default: enabled)')
    parser.add_argument('--no_calibrate', action='store_true',
                        help='Disable bisection calibration')
    parser.add_argument('--functional_calibrate', action='store_true', default=False,
                        help='Enable teacher-guided functional calibration after pruning (default: disabled)')
    parser.add_argument('--no_functional_calibrate', action='store_true',
                        help='Disable teacher-guided functional calibration')
    parser.add_argument('--functional_passes', type=int, default=2,
                        help='Number of teacher-guided calibration passes')
    parser.add_argument('--output_dir', type=str, default='results/one_shot_prune_only')
    parser.add_argument('--save_weights', type=str, default='')
    parser.add_argument('--min_degree', type=int, default=2,
                        help='Minimum output degree for spectral_quant method')
    parser.add_argument('--b_floor', type=float, default=0.35,
                        help='Minimum active fractional bits for spectral_quant')
    parser.add_argument('--low_budget_threshold', type=float, default=900.0,
                        help='Use structured low-budget pruning when target_ebops <= this threshold')
    parser.add_argument('--min_hidden_width', type=int, default=4,
                        help='Minimum hidden width kept by structured low-budget pruning')
    parser.add_argument('--near_budget_ratio', type=float, default=1.6,
                        help='Skip structural pruning when current_ebops <= target_ebops * ratio')
    parser.add_argument('--high_budget_ratio', type=float, default=0.45,
                        help='Use sensitivity fallback in spectral_quant when target/current >= this ratio')
    parser.add_argument('--low_budget_structured', action='store_true', default=True,
                        help='Enable structured low-budget pruning branch (default: enabled)')
    parser.add_argument('--no_low_budget_structured', action='store_true',
                        help='Disable structured low-budget pruning branch')
    parser.add_argument('--snows_init_method', type=str, default='sensitivity',
                        choices=['uniform', 'sensitivity', 'spectral_quant'],
                        help='Initial budget allocation method used before SNOWS reconstruction')
    parser.add_argument('--snows_k_step', type=int, default=2,
                        help='SNOWS look-ahead horizon in prunable layers')
    parser.add_argument('--snows_newton_steps', type=int, default=2,
                        help='SNOWS Newton updates per layer')
    parser.add_argument('--snows_cg_iters', type=int, default=25,
                        help='Maximum conjugate-gradient iterations per SNOWS step')
    parser.add_argument('--snows_damping', type=float, default=1e-4,
                        help='Tikhonov damping used in SNOWS Hessian-free solves')
    args, _ = parser.parse_known_args()
    if args.no_calibrate:
        args.calibrate = False
    if args.no_functional_calibrate:
        args.functional_calibrate = False
    if args.no_low_budget_structured:
        args.low_budget_structured = False

    print('=' * 70)
    print('One-shot prune only')
    print(f'  checkpoint  : {args.checkpoint}')
    print(f'  target_ebops: {args.target_ebops}')
    print(f'  prune_method: {args.prune_method}')
    print(f'  calibrate   : {args.calibrate}')
    print(f'  functional_calibrate: {args.functional_calibrate}')
    if args.prune_method == 'spectral_quant':
        print(f'  low_budget_structured: {args.low_budget_structured}')
        print(f'  low_budget_threshold : {args.low_budget_threshold}')
        print(f'  near_budget_ratio    : {args.near_budget_ratio}')
        print(f'  high_budget_ratio    : {args.high_budget_ratio}')
    if args.prune_method == 'snows':
        print(f'  snows_init_method    : {args.snows_init_method}')
        print(f'  snows_k_step         : {args.snows_k_step}')
        print(f'  snows_newton_steps   : {args.snows_newton_steps}')
        print(f'  snows_cg_iters       : {args.snows_cg_iters}')
        print(f'  snows_damping        : {args.snows_damping}')
    print('=' * 70)

    teacher_model = keras.models.load_model(args.checkpoint, compile=False)
    model = keras.models.load_model(args.checkpoint, compile=False)
    print_bk_stats(model, 'loaded')

    sample_input, sample_src = build_sample_input(model, args.sample_size, args.input_h5)
    print(f'  sample_input: {sample_src}')

    baseline_ebops = compute_model_ebops(model, sample_input)
    print(f'  Baseline EBOPs (measured): {baseline_ebops:.1f}')

    used_structured_low_budget = False
    snows_report = None
    saliency_report = None
    spectral_path_report = None

    if args.prune_method == 'sensitivity':
        pruner = SensitivityAwarePruner(target_ebops=args.target_ebops, pruned_threshold=0.1, b_k_min=0.3)
        pruner.prune_to_ebops(model, current_ebops=baseline_ebops, verbose=True)
    elif args.prune_method == 'uniform':
        pruner = HighBitPruner(target_ebops=args.target_ebops, pruned_threshold=0.1)
        pruner.prune_to_ebops(model, current_ebops=baseline_ebops, verbose=True)
    elif args.prune_method in {'snip', 'grasp', 'synflow'}:
        _, saliency_report = saliency_prune_to_ebops(
            model,
            target_ebops=args.target_ebops,
            sample_input=sample_input,
            method=args.prune_method,
            input_h5=args.input_h5,
            sample_size=args.sample_size,
            b_floor=args.b_floor,
            verbose=True,
        )
    elif args.prune_method == 'snows':
        _, snows_report = snows_prune_to_ebops(
            model,
            teacher_model=teacher_model,
            target_ebops=args.target_ebops,
            sample_input=sample_input,
            init_method=args.snows_init_method,
            b_floor=args.b_floor,
            k_step=args.snows_k_step,
            newton_steps=args.snows_newton_steps,
            cg_max_iter=args.snows_cg_iters,
            damping=args.snows_damping,
            verbose=True,
        )
        used_structured_low_budget = bool(snows_report.get('used_structured_low_budget', False))
    elif args.prune_method == 'spectral_path':
        _, spectral_path_report = spectral_path_prune_to_ebops(
            model,
            target_ebops=args.target_ebops,
            sample_input=sample_input,
            min_degree=args.min_degree,
            min_hidden_width=args.min_hidden_width,
            b_floor=args.b_floor,
            near_budget_ratio=args.near_budget_ratio,
            high_budget_ratio=args.high_budget_ratio,
            verbose=True,
        )
        used_structured_low_budget = True
    else:
        _, used_structured_low_budget = spectral_quant_prune_to_ebops(
            model,
            target_ebops=args.target_ebops,
            sample_input=sample_input,
            min_degree=args.min_degree,
            b_floor=args.b_floor,
            low_budget_structured=args.low_budget_structured,
            low_budget_threshold=args.low_budget_threshold,
            min_hidden_width=args.min_hidden_width,
            near_budget_ratio=args.near_budget_ratio,
            high_budget_ratio=args.high_budget_ratio,
            verbose=True,
        )

    post_prune_ebops = compute_model_ebops(model, sample_input)
    print(f'  Post-prune EBOPs (measured): {post_prune_ebops:.1f}')
    if saliency_report is not None:
        print(
            f"  [{args.prune_method.upper()}] kept="
            f"{saliency_report['kept_connections']}/{saliency_report['total_active_connections']}  "
            f"pre_bisect_ebops={saliency_report['pre_bisect_ebops']:.1f}"
        )
    if spectral_path_report is not None:
        print(
            f"  [SPECTRAL_PATH] effective_paths={spectral_path_report['effective_paths']:.1f}  "
            f"reachable_outputs={spectral_path_report['reachable_outputs']:.0f}"
        )

    post_func_calib_ebops = post_prune_ebops
    if args.functional_calibrate:
        post_func_calib_ebops = teacher_guided_post_prune_calibration(
            student_model=model,
            teacher_model=teacher_model,
            sample_input=sample_input,
            passes=args.functional_passes,
            b_floor=args.b_floor,
            b_ceiling=6.0,
            verbose=True,
        )
        print(f'  Post-functional-calib EBOPs (measured): {post_func_calib_ebops:.1f}')

    if args.calibrate:
        preserve_connectivity = (args.prune_method in {'spectral_quant', 'spectral_path'} and used_structured_low_budget)
        near_budget_preserve_case = (
            args.prune_method in {'spectral_quant', 'spectral_path'}
            and (not used_structured_low_budget)
            and baseline_ebops <= float(args.target_ebops) * float(args.near_budget_ratio)
        )
        post_prune_ebops = bisect_ebops_to_target(
            model,
            target_ebops=args.target_ebops,
            sample_input=sample_input,
            tolerance=0.02,
            max_iter=30,
            b_k_min=(0.20 if near_budget_preserve_case else args.b_floor) if args.prune_method in {'spectral_quant', 'spectral_path'} else 0.01,
            allow_connection_kill=(not preserve_connectivity) and (not near_budget_preserve_case),
        )
    else:
        post_prune_ebops = compute_model_ebops(model, sample_input)

    print_bk_stats(model, 'after one-shot prune')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_weights:
        weights_path = Path(args.save_weights)
        if not weights_path.is_absolute():
            weights_path = out_dir / weights_path
    else:
        ckpt_name = Path(args.checkpoint).stem
        weights_path = out_dir / (
            f'{ckpt_name}-oneshot-{args.prune_method}-target{int(args.target_ebops)}-'
            f'ebops{int(round(post_prune_ebops))}.weights.h5'
        )

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(weights_path))

    meta = {
        'checkpoint': args.checkpoint,
        'target_ebops': float(args.target_ebops),
        'prune_method': args.prune_method,
        'min_degree': int(args.min_degree),
        'b_floor': float(args.b_floor),
        'low_budget_structured': bool(args.low_budget_structured),
        'low_budget_threshold': float(args.low_budget_threshold),
        'min_hidden_width': int(args.min_hidden_width),
        'near_budget_ratio': float(args.near_budget_ratio),
        'high_budget_ratio': float(args.high_budget_ratio),
        'used_structured_low_budget': bool(used_structured_low_budget),
        'sample_input': sample_src,
        'baseline_ebops_measured': float(baseline_ebops),
        'post_functional_calib_ebops_measured': float(post_func_calib_ebops),
        'post_prune_ebops_measured': float(post_prune_ebops),
        'calibrated': bool(args.calibrate),
        'functional_calibrated': bool(args.functional_calibrate),
        'functional_passes': int(args.functional_passes),
        'saliency_report': saliency_report,
        'spectral_path_report': spectral_path_report,
        'snows_report': snows_report,
        'weights_path': str(weights_path),
    }
    meta_path = weights_path.with_suffix('.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('=' * 70)
    print(f'Weights saved: {weights_path}')
    print(f'Metadata saved: {meta_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()
