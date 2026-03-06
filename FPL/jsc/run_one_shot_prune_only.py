#!/usr/bin/env python3
"""
run_one_shot_prune_only.py
==========================
仅执行一次性剪枝，不进入任何后续训练阶段。

流程：
1) 加载 checkpoint
2) 执行 one-shot pruning（uniform / sensitivity）
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
    verbose: bool = True,
):
    """谱/拓扑友好的一次性剪枝：低预算走结构化链路，其余走列 top-k。"""
    current_ebops = compute_model_ebops(model, sample_input)
    if current_ebops <= float(target_ebops) * float(near_budget_ratio):
        if verbose:
            print(
                f"[SpectralQuantPruner] near-budget preserve mode: "
                f"current={current_ebops:.1f}, target={target_ebops:.1f}, "
                f"ratio={current_ebops / max(target_ebops, 1.0):.3f}"
            )
        return current_ebops, False

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
                        choices=['uniform', 'sensitivity', 'spectral_quant'])
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
    parser.add_argument('--low_budget_structured', action='store_true', default=True,
                        help='Enable structured low-budget pruning branch (default: enabled)')
    parser.add_argument('--no_low_budget_structured', action='store_true',
                        help='Disable structured low-budget pruning branch')
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
    print('=' * 70)

    teacher_model = keras.models.load_model(args.checkpoint, compile=False)
    model = keras.models.load_model(args.checkpoint, compile=False)
    print_bk_stats(model, 'loaded')

    sample_input, sample_src = build_sample_input(model, args.sample_size, args.input_h5)
    print(f'  sample_input: {sample_src}')

    baseline_ebops = compute_model_ebops(model, sample_input)
    print(f'  Baseline EBOPs (measured): {baseline_ebops:.1f}')

    used_structured_low_budget = False

    if args.prune_method == 'sensitivity':
        pruner = SensitivityAwarePruner(target_ebops=args.target_ebops, pruned_threshold=0.1, b_k_min=0.3)
        pruner.prune_to_ebops(model, current_ebops=baseline_ebops, verbose=True)
    elif args.prune_method == 'uniform':
        pruner = HighBitPruner(target_ebops=args.target_ebops, pruned_threshold=0.1)
        pruner.prune_to_ebops(model, current_ebops=baseline_ebops, verbose=True)
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
            verbose=True,
        )

    post_prune_ebops = compute_model_ebops(model, sample_input)
    print(f'  Post-prune EBOPs (measured): {post_prune_ebops:.1f}')

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
        preserve_connectivity = (args.prune_method == 'spectral_quant' and used_structured_low_budget)
        near_budget_preserve_case = (
            args.prune_method == 'spectral_quant'
            and (not used_structured_low_budget)
            and baseline_ebops <= float(args.target_ebops) * float(args.near_budget_ratio)
        )
        post_prune_ebops = bisect_ebops_to_target(
            model,
            target_ebops=args.target_ebops,
            sample_input=sample_input,
            tolerance=0.02,
            max_iter=30,
            b_k_min=(0.20 if near_budget_preserve_case else args.b_floor) if args.prune_method == 'spectral_quant' else 0.01,
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
        'used_structured_low_budget': bool(used_structured_low_budget),
        'sample_input': sample_src,
        'baseline_ebops_measured': float(baseline_ebops),
        'post_functional_calib_ebops_measured': float(post_func_calib_ebops),
        'post_prune_ebops_measured': float(post_prune_ebops),
        'calibrated': bool(args.calibrate),
        'functional_calibrated': bool(args.functional_calibrate),
        'functional_passes': int(args.functional_passes),
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
