#!/usr/bin/env python3
"""
One-shot prune (target=400) + topology export + spectral/deadzone repair + trainability verification.

核心目标：
1) 先执行一次性剪枝。
2) 立刻用 topology_graph_plot_utils 导出拓扑图（剪枝后）。
3) 用谱理论 + 量化器死区理论做修复，使网络可启动训练。
4) 修复后再次导出拓扑图，并做梯度/短训验证。
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 保证相对路径稳定
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
import random
from pathlib import Path
import re

import keras
import numpy as np
import tensorflow as tf

from data.data import get_data
import model.model  # noqa: F401
from hgq.layers import QLayerBase
from hgq.utils.sugar import FreeEBOPs, ParetoFront
from run_one_shot_prune_only import (
    spectral_quant_prune_to_ebops,
    bisect_ebops_to_target,
    _dense_prunable_layers,
    _layer_active_mask,
    _reestimate_quantizer_ranges,
    teacher_guided_post_prune_calibration,
)
from utils.ramanujan_budget_utils import (
    HighBitPruner,
    SensitivityAwarePruner,
    BetaOnlyBudgetController,
    _flatten_layers,
    _get_kq_var,
)
from utils.topology_graph_plot_utils import TopologyGraphPlotter


DEFAULT_CKPT = "results/baseline/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras"


def _extract_ebops_from_name(name: str):
    m = re.search(r"ebops=(\d+)", name)
    return int(m.group(1)) if m else None


def choose_natural_ckpt_near_target(
    target_ebops: float,
    baseline_dir: str | Path = "results/baseline",
    fallback: str | None = None,
):
    d = Path(baseline_dir)
    if not d.exists():
        return fallback, None, f"baseline dir not found: {d}"

    best = None
    for p in d.glob("*.keras"):
        eb = _extract_ebops_from_name(p.name)
        if eb is None:
            continue
        rec = (abs(float(eb) - float(target_ebops)), eb, str(p))
        if best is None or rec[0] < best[0]:
            best = rec
    if best is None:
        return fallback, None, f"no parseable baseline ckpt under: {d}"
    return best[2], float(best[1]), None


def _broadcast_like(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32)
    if arr.shape == target_shape:
        return arr
    try:
        return np.broadcast_to(arr, target_shape).astype(np.float32)
    except Exception:
        fill = float(np.mean(arr)) if arr.size else 0.0
        return np.full(target_shape, fill, dtype=np.float32)


def _assign_like(var, arr: np.ndarray):
    ref_shape = tuple(var.shape)
    out = _broadcast_like(arr, ref_shape)
    var.assign(out.astype(np.float32))


def _forward_update_ebops_no_bn_drift(model, sample_input):
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
    from keras import ops

    _forward_update_ebops_no_bn_drift(model, sample_input)
    total = 0.0
    for layer in model._flatten_layers():
        if isinstance(layer, QLayerBase) and layer.enable_ebops and layer._ebops is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def _bits_np(layer, kernel_shape: tuple[int, ...]) -> np.ndarray:
    from keras import ops

    bits = None
    try:
        bits = layer.kq.bits_(ops.shape(layer.kernel))
        bits = np.array(ops.convert_to_numpy(bits), dtype=np.float32)
    except Exception:
        bits = None

    if bits is None:
        b_var = _get_kq_var(layer.kq, "b")
        if b_var is None:
            b_var = _get_kq_var(layer.kq, "f")
        if b_var is not None:
            bits = np.array(b_var.numpy(), dtype=np.float32)
    if bits is None:
        bits = np.zeros(kernel_shape, dtype=np.float32)
    return _broadcast_like(bits, kernel_shape)


def _deadzone_mask(layer, kernel: np.ndarray, active: np.ndarray, q_eps: float = 1e-12) -> np.ndarray:
    from keras import ops

    try:
        qker = layer.kq(layer.kernel, training=False)
        q_np = np.array(ops.convert_to_numpy(qker), dtype=np.float32)
        q_np = _broadcast_like(q_np, kernel.shape)
    except Exception:
        q_np = np.zeros_like(kernel, dtype=np.float32)

    return active & (np.abs(kernel) > 1e-8) & (np.abs(q_np) <= q_eps)


def _fused_qkernel_qbias(layer):
    from keras import ops

    kernel_shape = tuple(layer.kernel.shape)
    if hasattr(layer, "get_fused_qkernel_and_qbias") and hasattr(layer, "moving_mean") and hasattr(layer, "moving_variance"):
        qk, qb = layer.get_fused_qkernel_and_qbias(False, layer.moving_mean, layer.moving_variance)
        qk_np = _broadcast_like(np.array(ops.convert_to_numpy(qk), dtype=np.float32), kernel_shape)
        qb_np = np.array(ops.convert_to_numpy(qb), dtype=np.float32)
        return qk_np, qb_np

    qk = layer.kq(layer.kernel, training=False)
    qk_np = _broadcast_like(np.array(ops.convert_to_numpy(qk), dtype=np.float32), kernel_shape)
    b = getattr(layer, "bias", None)
    bq = getattr(layer, "bq", None)
    if b is not None and bq is not None:
        qb = bq(b, training=False)
        qb_np = np.array(ops.convert_to_numpy(qb), dtype=np.float32)
    else:
        qb_np = np.zeros((kernel_shape[1],), dtype=np.float32)
    return qk_np, qb_np


def _fused_deadzone_mask(layer, active: np.ndarray, q_eps: float = 1e-12) -> np.ndarray:
    qk_np, _ = _fused_qkernel_qbias(layer)
    return active & (np.abs(qk_np) <= q_eps)


def _active_submatrix(kernel: np.ndarray, active: np.ndarray):
    col_idx = np.where(np.sum(active, axis=0) > 0)[0]
    if col_idx.size == 0:
        return None
    row_idx = np.where(np.sum(active[:, col_idx], axis=1) > 0)[0]
    if row_idx.size == 0:
        return None
    sub_w = kernel[np.ix_(row_idx, col_idx)].astype(np.float32)
    sub_active = active[np.ix_(row_idx, col_idx)]
    sub_w = np.where(sub_active, sub_w, 0.0).astype(np.float32)
    return sub_w, sub_active, row_idx, col_idx


def _spectral_metrics(kernel: np.ndarray, active: np.ndarray):
    sub = _active_submatrix(kernel, active)
    if sub is None:
        return 0.0, float("inf"), None, None, None, None
    w, _, row_idx, col_idx = sub
    if w.ndim != 2 or np.all(np.abs(w) < 1e-12):
        return 0.0, float("inf"), None, None, row_idx, col_idx
    try:
        u, s, vh = np.linalg.svd(w, full_matrices=False)
    except Exception:
        return 0.0, float("inf"), None, None, row_idx, col_idx
    if s.size == 0:
        return 0.0, float("inf"), None, None, row_idx, col_idx
    sigma_max = float(s[0])
    sigma_min = float(s[-1])
    cond = float(sigma_max / (sigma_min + 1e-12)) if sigma_max > 0 else float("inf")
    u_min = u[:, -1].astype(np.float32)
    v_min = vh[-1, :].astype(np.float32)
    return sigma_min, cond, u_min, v_min, row_idx, col_idx


def _sync_quantizers_for_active(
    layer,
    kernel: np.ndarray,
    active: np.ndarray,
    new_edges: np.ndarray,
    dead_edges: np.ndarray,
    b_floor_new: float,
    b_floor_dead: float,
):
    kq = layer.kq
    b_var = _get_kq_var(kq, "b")
    if b_var is None:
        b_var = _get_kq_var(kq, "f")
    i_var = _get_kq_var(kq, "i")
    k_var = _get_kq_var(kq, "k")

    if b_var is not None:
        b_old = _broadcast_like(np.array(b_var.numpy(), dtype=np.float32), active.shape)
        b_new = b_old.copy()
        b_new = np.where(new_edges, np.maximum(b_new, b_floor_new), b_new)
        b_new = np.where(dead_edges, np.maximum(b_new, b_floor_dead), b_new)
        b_new = np.where(active, b_new, 0.0)
        _assign_like(b_var, b_new)

    if i_var is not None:
        i_old = _broadcast_like(np.array(i_var.numpy(), dtype=np.float32), active.shape)
        i_req = np.ceil(np.log2(np.maximum(np.abs(kernel), 1e-8))).astype(np.float32) + 1.0
        i_new = np.where(active, np.clip(np.maximum(i_old, i_req), -2.0, 2.0), -16.0)
        _assign_like(i_var, i_new)

    if k_var is not None:
        k_old = _broadcast_like(np.array(k_var.numpy(), dtype=np.float32), active.shape)
        k_new = np.where(active, np.maximum(k_old, 1.0), 0.0)
        _assign_like(k_var, k_new)

    bq = getattr(layer, "bq", None)
    if bq is None:
        return
    cols_active = np.sum(active, axis=0) > 0

    bb_var = _get_kq_var(bq, "b")
    if bb_var is None:
        bb_var = _get_kq_var(bq, "f")
    bi_var = _get_kq_var(bq, "i")
    bk_var = _get_kq_var(bq, "k")

    if bb_var is not None:
        bb_old = _broadcast_like(np.array(bb_var.numpy(), dtype=np.float32), cols_active.shape)
        bb_new = np.where(cols_active, np.maximum(bb_old, b_floor_new), 0.0)
        _assign_like(bb_var, bb_new)
    if bi_var is not None:
        bi_old = _broadcast_like(np.array(bi_var.numpy(), dtype=np.float32), cols_active.shape)
        bi_new = np.where(cols_active, np.clip(bi_old, -2.0, 6.0), -16.0)
        _assign_like(bi_var, bi_new)
    if bk_var is not None:
        bk_old = _broadcast_like(np.array(bk_var.numpy(), dtype=np.float32), cols_active.shape)
        bk_new = np.where(cols_active, np.maximum(bk_old, 1.0), 0.0)
        _assign_like(bk_var, bk_new)


def collect_deadzone_stats(model) -> dict[str, float]:
    n_active = 0
    n_deadzone_raw = 0
    n_deadzone_fused = 0
    for layer in _dense_prunable_layers(model):
        kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
        active = _layer_active_mask(layer)
        dead = _deadzone_mask(layer, kernel, active)
        dead_fused = _fused_deadzone_mask(layer, active)
        n_active += int(np.sum(active))
        n_deadzone_raw += int(np.sum(dead))
        n_deadzone_fused += int(np.sum(dead_fused))
    return {
        "active_edges": float(n_active),
        "deadzone_edges_raw": float(n_deadzone_raw),
        "deadzone_ratio_raw": float(n_deadzone_raw / max(n_active, 1)),
        "deadzone_edges_fused": float(n_deadzone_fused),
        "deadzone_ratio_fused": float(n_deadzone_fused / max(n_active, 1)),
        "deadzone_edges": float(n_deadzone_fused),
        "deadzone_ratio": float(n_deadzone_fused / max(n_active, 1)),
    }


def _safe_stable_rank(w: np.ndarray) -> float:
    if w.size == 0:
        return 0.0
    try:
        s = np.linalg.svd(w, compute_uv=False)
    except Exception:
        return 0.0
    if s.size == 0:
        return 0.0
    num = float(np.sum(s * s))
    den = float((s[0] * s[0]) + 1e-12)
    return num / den


def _build_balanced_mask_from_scores(
    score: np.ndarray,
    n_edges: int,
    row_target: int,
    min_col_degree: int = 1,
    min_row_degree: int = 0,
) -> np.ndarray:
    in_dim, out_dim = score.shape
    n_edges = int(np.clip(n_edges, 1, in_dim * out_dim))
    mask = np.zeros((in_dim, out_dim), dtype=bool)
    min_col_degree = int(np.clip(min_col_degree, 1, max(in_dim, 1)))
    min_row_degree = int(np.clip(min_row_degree, 0, max(out_dim, 1)))

    # Step-1: enforce minimum incoming degree for each output.
    for _ in range(min_col_degree):
        for j in range(out_dim):
            if int(np.sum(mask[:, j])) >= min_col_degree:
                continue
            row_penalty = 1.0 + np.sum(mask, axis=1).astype(np.float32)
            adj = score[:, j] / row_penalty
            adj = np.where(mask[:, j], -np.inf, adj)
            i = int(np.argmax(adj))
            if np.isfinite(adj[i]):
                mask[i, j] = True

    # Step-2: encourage row-side coverage to avoid bottlenecked information flow.
    if row_target > 0 and int(np.sum(mask)) < n_edges:
        row_strength = np.max(score, axis=1)
        row_order = np.argsort(row_strength)[::-1]
        for i in row_order:
            if np.any(mask[i, :]):
                continue
            col_penalty = 1.0 + np.sum(mask, axis=0).astype(np.float32)
            j = int(np.argmax(score[i, :] / col_penalty))
            mask[i, j] = True
            if int(np.sum(mask)) >= n_edges:
                break

    # Step-3: enforce minimum outgoing degree for top rows.
    if min_row_degree > 0 and int(np.sum(mask)) < n_edges:
        row_strength = np.max(score, axis=1)
        row_order = np.argsort(row_strength)[::-1]
        for i in row_order:
            while int(np.sum(mask[i, :])) < min_row_degree and int(np.sum(mask)) < n_edges:
                col_penalty = 1.0 + np.sum(mask, axis=0).astype(np.float32)
                adj = score[i, :] / col_penalty
                adj = np.where(mask[i, :], -np.inf, adj)
                j = int(np.argmax(adj))
                if not np.isfinite(adj[j]):
                    break
                mask[i, j] = True

    # Step-4: fill remaining edges with degree-balanced greedy selection.
    while int(np.sum(mask)) < n_edges:
        row_deg = np.sum(mask, axis=1).astype(np.float32)
        col_deg = np.sum(mask, axis=0).astype(np.float32)
        penalty = np.sqrt((row_deg[:, None] + 1.0) * (col_deg[None, :] + 1.0))
        adj = score / penalty
        adj = np.where(mask, -np.inf, adj)
        idx = int(np.argmax(adj))
        if not np.isfinite(adj.flat[idx]):
            break
        i, j = np.unravel_index(idx, adj.shape)
        mask[i, j] = True

    return mask


def _collect_bridge_waste(masks: list[np.ndarray], layer_names: list[str]) -> dict:
    bridges = []
    total_in_only = 0
    total_out_only = 0
    total_active_bridge = 0
    total_nodes = 0
    for i in range(len(masks) - 1):
        prev = masks[i]
        nxt = masks[i + 1]
        in_deg = np.sum(prev, axis=0)
        out_deg = np.sum(nxt, axis=1)
        in_only = int(np.sum((in_deg > 0) & (out_deg == 0)))
        out_only = int(np.sum((in_deg == 0) & (out_deg > 0)))
        isolated = int(np.sum((in_deg == 0) & (out_deg == 0)))
        active_bridge = int(np.sum((in_deg > 0) & (out_deg > 0)))
        node_n = int(in_deg.shape[0])
        bridges.append(
            {
                "bridge": f"{layer_names[i]}->{layer_names[i+1]}",
                "nodes": float(node_n),
                "active": float(active_bridge),
                "in_only": float(in_only),
                "out_only": float(out_only),
                "isolated": float(isolated),
            }
        )
        total_in_only += in_only
        total_out_only += out_only
        total_active_bridge += active_bridge
        total_nodes += node_n
    return {
        "bridges": bridges,
        "total_in_only": float(total_in_only),
        "total_out_only": float(total_out_only),
        "total_active": float(total_active_bridge),
        "total_nodes": float(total_nodes),
    }


def _enforce_bridge_flow_conservation(
    masks: list[np.ndarray],
    scores: list[np.ndarray],
    max_iter: int = 3,
) -> tuple[list[np.ndarray], dict]:
    if len(masks) <= 1:
        return masks, {"iterations": 0.0, "added_in_edges": 0.0, "added_out_edges": 0.0}

    masks = [np.array(m, dtype=bool, copy=True) for m in masks]
    add_in = 0
    add_out = 0
    it_used = 0
    for it in range(max(1, int(max_iter))):
        changed = False
        for i in range(len(masks) - 1):
            prev = masks[i]
            nxt = masks[i + 1]
            prev_score = scores[i]
            nxt_score = scores[i + 1]

            in_deg = np.sum(prev, axis=0)
            out_deg = np.sum(nxt, axis=1)
            in_only_nodes = np.where((in_deg > 0) & (out_deg == 0))[0]
            out_only_nodes = np.where((in_deg == 0) & (out_deg > 0))[0]

            for h in in_only_nodes:
                if np.all(nxt[h, :]):
                    continue
                col_penalty = 1.0 + np.sum(nxt, axis=0).astype(np.float32)
                cand = np.where(~nxt[h, :], nxt_score[h, :] / col_penalty, -np.inf)
                j = int(np.argmax(cand))
                if not np.isfinite(cand[j]):
                    continue
                nxt[h, j] = True
                add_out += 1
                changed = True

            for h in out_only_nodes:
                if np.all(prev[:, h]):
                    continue
                row_penalty = 1.0 + np.sum(prev, axis=1).astype(np.float32)
                cand = np.where(~prev[:, h], prev_score[:, h] / row_penalty, -np.inf)
                r = int(np.argmax(cand))
                if not np.isfinite(cand[r]):
                    continue
                prev[r, h] = True
                add_in += 1
                changed = True

            masks[i] = prev
            masks[i + 1] = nxt
        it_used = it + 1
        if not changed:
            break
    return masks, {
        "iterations": float(it_used),
        "added_in_edges": float(add_in),
        "added_out_edges": float(add_out),
    }


def _safe_norm_score(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    m = float(np.mean(np.abs(x)))
    if not np.isfinite(m) or m < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x / m).astype(np.float32)


def _compute_gradient_saliency(
    model,
    layers,
    sample_input,
    sample_target,
):
    if sample_target is None:
        return [None for _ in layers], [0.0 for _ in layers], None

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        logits = model(sample_input, training=True)
        loss = loss_fn(sample_target, logits)
    grads = tape.gradient(loss, [layer.kernel for layer in layers])

    saliency_maps = []
    layer_strength = []
    for layer, grad in zip(layers, grads):
        if grad is None:
            saliency_maps.append(None)
            layer_strength.append(0.0)
            continue
        g_np = np.abs(np.array(grad.numpy(), dtype=np.float32))
        w_np = np.abs(np.array(layer.kernel.numpy(), dtype=np.float32))
        sal = (g_np * w_np).astype(np.float32)
        saliency_maps.append(sal)
        layer_strength.append(float(np.mean(sal)))
    return saliency_maps, layer_strength, float(_to_float(loss))


def collect_flow_connectivity_stats(model) -> dict:
    layers = _dense_prunable_layers(model)
    if not layers:
        return {
            "has_prunable_layers": False,
            "hidden_in_only_total": 0.0,
            "hidden_out_only_total": 0.0,
            "output_missing_classes": 0.0,
            "bridge": {},
        }
    masks = [_layer_active_mask(layer).astype(bool) for layer in layers]
    layer_names = [layer.name for layer in layers]
    bridge = _collect_bridge_waste(masks, layer_names)
    last_col_deg = np.sum(masks[-1], axis=0)
    output_missing = int(np.sum(last_col_deg == 0))
    return {
        "has_prunable_layers": True,
        "hidden_in_only_total": float(bridge["total_in_only"]),
        "hidden_out_only_total": float(bridge["total_out_only"]),
        "output_missing_classes": float(output_missing),
        "bridge": bridge,
    }


def enforce_model_flow_and_output_connectivity(
    model,
    min_output_degree: int = 1,
    b_floor_new_edge: float = 0.45,
    b_floor_deadzone: float = 0.75,
    deadzone_lift: float = 0.75,
    max_iter: int = 4,
    seed: int = 42,
) -> dict:
    layers = _dense_prunable_layers(model)
    if not layers:
        return {"applied": False, "reason": "no_prunable_layers"}

    rng = np.random.default_rng(seed)
    kernels = [np.array(layer.kernel.numpy(), dtype=np.float32) for layer in layers]
    masks0 = [_layer_active_mask(layer).astype(bool) for layer in layers]
    masks = [m.copy() for m in masks0]
    layer_names = [layer.name for layer in layers]

    scores = []
    for kernel in kernels:
        abs_w = np.abs(kernel)
        row_norm = np.sqrt(np.sum(abs_w, axis=1) + 1e-12)
        col_norm = np.sqrt(np.sum(abs_w, axis=0) + 1e-12)
        s = abs_w * (row_norm[:, None] ** 0.5) * (col_norm[None, :] ** 0.5)
        s += rng.uniform(0.0, 1e-9, size=s.shape).astype(np.float32)
        scores.append(s)

    added_output_edges = 0
    added_backfill_in_edges = 0

    def _ensure_output_cols():
        nonlocal added_output_edges, added_backfill_in_edges
        last = masks[-1]
        if last.size == 0:
            return
        need_deg = max(1, int(min_output_degree))
        for o in range(last.shape[1]):
            while int(np.sum(last[:, o])) < need_deg:
                free_rows = np.where(~last[:, o])[0]
                if free_rows.size == 0:
                    break
                if len(masks) >= 2:
                    prev = masks[-2]
                    in_deg = np.sum(prev, axis=0)
                    preferred = free_rows[in_deg[free_rows] > 0]
                    if preferred.size > 0:
                        cand_rows = preferred
                    else:
                        cand_rows = free_rows
                else:
                    prev = None
                    in_deg = None
                    cand_rows = free_rows
                row_pick = int(cand_rows[np.argmax(scores[-1][cand_rows, o])])

                # If chosen row has no incoming path, backfill one incoming edge first.
                if prev is not None and in_deg is not None and int(in_deg[row_pick]) == 0:
                    row_penalty = 1.0 + np.sum(prev, axis=1).astype(np.float32)
                    prev_cand = np.where(~prev[:, row_pick], scores[-2][:, row_pick] / row_penalty, -np.inf)
                    r = int(np.argmax(prev_cand))
                    if np.isfinite(prev_cand[r]):
                        prev[r, row_pick] = True
                        added_backfill_in_edges += 1
                        masks[-2] = prev
                last[row_pick, o] = True
                added_output_edges += 1
        masks[-1] = last

    before = _collect_bridge_waste(masks, layer_names)
    _ensure_output_cols()
    masks, flow_fix = _enforce_bridge_flow_conservation(masks, scores, max_iter=max_iter)
    _ensure_output_cols()
    after = _collect_bridge_waste(masks, layer_names)

    layer_reports = []
    total_new_edges = 0
    total_dead_fixed = 0
    for li, layer in enumerate(layers):
        kernel = kernels[li]
        active0 = masks0[li]
        active = masks[li]
        in_dim = kernel.shape[0]
        new_edges = active & (~active0)
        n_new = int(np.sum(new_edges))
        if n_new > 0:
            std = float(math.sqrt(2.0 / max(in_dim, 1)))
            idx = np.where(new_edges)
            old_vals = kernel[idx]
            rand_vals = rng.normal(0.0, std, size=n_new).astype(np.float32)
            kernel[idx] = np.where(np.abs(old_vals) > 1e-8, old_vals, rand_vals).astype(np.float32)

        kernel_new = np.where(active, kernel, 0.0).astype(np.float32)
        bits = _bits_np(layer, kernel_new.shape)
        q_step = np.power(2.0, -np.clip(bits, 0.0, 12.0)).astype(np.float32)
        min_abs = float(deadzone_lift) * q_step
        active_abs = np.abs(kernel_new[active])
        cap = float(np.percentile(active_abs, 60.0)) if active_abs.size > 0 else 0.05
        min_abs = np.minimum(min_abs, max(cap, 1e-3)).astype(np.float32)
        sign = np.sign(kernel_new).astype(np.float32)
        rand_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=kernel_new.shape)
        sign = np.where(np.abs(sign) > 1e-8, sign, rand_sign)
        dead_before = _deadzone_mask(layer, kernel_new, active)
        kernel_new = np.where(dead_before, sign * np.maximum(np.abs(kernel_new), min_abs), kernel_new).astype(np.float32)

        layer.kernel.assign(kernel_new)
        _sync_quantizers_for_active(
            layer=layer,
            kernel=kernel_new,
            active=active,
            new_edges=new_edges,
            dead_edges=dead_before,
            b_floor_new=float(b_floor_new_edge),
            b_floor_dead=float(b_floor_deadzone),
        )

        total_new_edges += n_new
        total_dead_fixed += int(np.sum(dead_before))
        layer_reports.append(
            {
                "layer": layer.name,
                "new_edges": float(n_new),
                "deadzone_fixed": float(np.sum(dead_before)),
                "edges_after": float(np.sum(active)),
            }
        )

    last_missing = int(np.sum(np.sum(masks[-1], axis=0) == 0))
    return {
        "applied": True,
        "bridge_before": before,
        "bridge_after": after,
        "flow_fix": flow_fix,
        "added_output_edges": float(added_output_edges),
        "added_backfill_in_edges": float(added_backfill_in_edges),
        "total_new_edges": float(total_new_edges),
        "total_deadzone_fixed": float(total_dead_fixed),
        "output_missing_classes_after": float(last_missing),
        "layers": layer_reports,
    }


def prune_by_topology_quant_theory(
    model,
    target_ebops: float,
    sample_input,
    sample_target=None,
    b_floor: float = 0.35,
    b_ceiling: float = 6.0,
    deadzone_lift: float = 0.80,
    row_target_scale: float = 1.0,
    grad_mix: float = 0.60,
    seed: int = 42,
) -> dict:
    """One-shot prune driven by topology + quantizer theory.

    Design constraints:
    - No baseline topology is used in pruning decisions.
    - Preserve per-layer output connectivity, spread row coverage, and avoid deadzone weights.
    """
    rng = np.random.default_rng(seed)
    layers = _dense_prunable_layers(model)
    if not layers:
        return {"layers": [], "note": "no_prunable_layers"}

    current_ebops = compute_model_ebops(model, sample_input)
    grad_saliency, layer_sensitivity, grad_probe_loss = _compute_gradient_saliency(
        model=model,
        layers=layers,
        sample_input=sample_input,
        sample_target=sample_target,
    )
    sens_arr = np.array(layer_sensitivity, dtype=np.float32)
    sens_mean = float(np.mean(sens_arr)) if sens_arr.size > 0 else 0.0
    layer_density_factors = []
    for s in sens_arr:
        if sens_mean <= 1e-12:
            layer_density_factors.append(1.0)
        else:
            layer_density_factors.append(float(np.clip(math.sqrt(float(s) / sens_mean), 0.70, 1.60)))
    shrink_ratio = math.sqrt(float(target_ebops) / max(float(current_ebops), 1.0))
    width_ratio = float(np.clip(1.40 * shrink_ratio, 0.16, 0.60))
    density_ratio = float(np.clip(0.60 * shrink_ratio, 0.05, 0.22))

    # Select a budget-aware critical neuron backbone without using baseline topology.
    selected_outputs = []
    kernels = [np.array(l.kernel.numpy(), dtype=np.float32) for l in layers]
    for li, kernel in enumerate(kernels):
        _, out_dim = kernel.shape
        if li == len(kernels) - 1:
            selected_outputs.append(np.arange(out_dim, dtype=np.int32))
            continue
        width_tgt = int(np.clip(round(out_dim * width_ratio), 6, out_dim))
        col_imp_abs = np.mean(np.abs(kernel), axis=0)
        g_map = grad_saliency[li] if li < len(grad_saliency) else None
        if g_map is not None:
            col_imp_grad = np.mean(np.array(g_map, dtype=np.float32), axis=0)
            col_imp = (
                (1.0 - float(grad_mix)) * _safe_norm_score(col_imp_abs)
                + float(grad_mix) * _safe_norm_score(col_imp_grad)
            )
        else:
            col_imp = col_imp_abs
        if li < len(kernels) - 1:
            next_w = np.abs(kernels[li + 1])
            if next_w.shape[0] == out_dim:
                col_imp = col_imp * np.sqrt(np.mean(next_w, axis=1) + 1e-12)
        idx = np.argsort(col_imp)[-width_tgt:]
        selected_outputs.append(np.sort(idx).astype(np.int32))

    layer_names = [l.name for l in layers]
    reports = []
    full_scores = []
    masks_init = []
    meta = []
    for li, layer in enumerate(layers):
        kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
        in_dim, out_dim = kernel.shape

        row_idx = (
            np.arange(in_dim, dtype=np.int32)
            if li == 0
            else selected_outputs[li - 1]
        )
        col_idx = selected_outputs[li]
        sub_in = int(row_idx.size)
        sub_out = int(col_idx.size)
        if sub_in <= 0 or sub_out <= 0:
            continue

        abs_w = np.abs(kernel)
        row_norm = np.sqrt(np.sum(abs_w, axis=1) + 1e-12)
        col_norm = np.sqrt(np.sum(abs_w, axis=0) + 1e-12)
        topo_score = abs_w * (row_norm[:, None] ** 0.5) * (col_norm[None, :] ** 0.5)
        g_map = grad_saliency[li] if li < len(grad_saliency) else None
        if g_map is not None:
            grad_score = np.abs(np.array(g_map, dtype=np.float32))
            score = (
                (1.0 - float(grad_mix)) * _safe_norm_score(topo_score)
                + float(grad_mix) * _safe_norm_score(grad_score)
            )
        else:
            score = topo_score
        score += rng.uniform(0.0, 1e-9, size=score.shape).astype(np.float32)
        full_scores.append(score)

        sub_score = score[np.ix_(row_idx, col_idx)]
        min_col_degree = 1 if li < len(layers) - 1 else 2
        min_col_degree = int(min(min_col_degree, sub_in))
        min_row_degree = 1 if li > 0 else 0
        density_ratio_layer = float(
            np.clip(
                density_ratio * (layer_density_factors[li] if li < len(layer_density_factors) else 1.0),
                0.04,
                0.30,
            )
        )
        target_edges = max(sub_out * min_col_degree, int(round(sub_in * sub_out * density_ratio_layer)))
        target_edges = int(min(target_edges, sub_in * sub_out))
        row_target = int(min(sub_in, max(1, round(row_target_scale * math.sqrt(sub_in)))))
        sub_mask = _build_balanced_mask_from_scores(
            sub_score,
            n_edges=target_edges,
            row_target=row_target,
            min_col_degree=min_col_degree,
            min_row_degree=min_row_degree,
        )
        active = np.zeros((in_dim, out_dim), dtype=bool)
        active[np.ix_(row_idx, col_idx)] = sub_mask
        masks_init.append(active)
        meta.append(
            {
                "row_idx": row_idx,
                "col_idx": col_idx,
                "target_edges": target_edges,
                "density_ratio_layer": density_ratio_layer,
                "kernel": kernel,
                "layer": layer,
                "med": float(max(np.median(abs_w[active]) if np.any(active) else 1e-4, 1e-8)),
            }
        )

    bridge_waste_before = _collect_bridge_waste(masks_init, layer_names)
    masks, flow_fix = _enforce_bridge_flow_conservation(masks_init, full_scores, max_iter=4)
    bridge_waste_after = _collect_bridge_waste(masks, layer_names)

    for li, layer in enumerate(layers):
        kernel = np.array(meta[li]["kernel"], dtype=np.float32)
        active = masks[li]
        target_edges = int(meta[li]["target_edges"])
        med = float(meta[li]["med"])
        abs_w = np.abs(kernel)

        # Quantizer deadzone theory: keep active weights away from quantization zero cell.
        active_abs = abs_w[active]
        bits_curr = np.clip(_bits_np(layer, kernel.shape), 0.0, 8.0)
        b_active = np.clip(
            max(float(b_floor), 0.45) + 0.50 * np.log2(1.0 + abs_w / med),
            0.45,
            float(min(b_ceiling, 3.5)),
        ).astype(np.float32)
        q_step = np.power(2.0, -np.clip(np.maximum(bits_curr, b_active), 0.0, 12.0)).astype(np.float32)
        min_abs = float(deadzone_lift) * q_step
        cap = float(np.percentile(active_abs, 60.0)) if active_abs.size > 0 else 0.05
        min_abs = np.minimum(min_abs, max(cap, 1e-3)).astype(np.float32)
        sign = np.sign(kernel).astype(np.float32)
        rand_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=kernel.shape)
        sign = np.where(np.abs(sign) > 1e-9, sign, rand_sign)
        kernel_new = np.where(active, kernel, 0.0).astype(np.float32)
        dead_edges = active & (np.abs(kernel_new) < min_abs)
        kernel_new = np.where(dead_edges, sign * min_abs, kernel_new).astype(np.float32)
        layer.kernel.assign(kernel_new)

        # Theory-consistent quantizer reset: keep active edges alive while reducing
        # integer-bit burden so EBOPs calibration does not collapse connectivity.
        kq = layer.kq
        b_var = _get_kq_var(kq, "b")
        if b_var is None:
            b_var = _get_kq_var(kq, "f")
        i_var = _get_kq_var(kq, "i")
        k_var = _get_kq_var(kq, "k")
        if b_var is not None:
            b_target = np.where(
                active,
                np.clip(
                    1.20 + 0.45 * np.log2(1.0 + np.abs(kernel_new) / (med + 1e-8)),
                    1.40,
                    float(min(b_ceiling, 3.5)),
                ),
                0.0,
            ).astype(np.float32)
            _assign_like(b_var, b_target)
        if i_var is not None:
            i_req = np.ceil(np.log2(np.maximum(np.abs(kernel_new), 1e-6))).astype(np.float32) + 1.0
            i_target = np.where(active, np.clip(i_req, -2.0, 2.0), -16.0).astype(np.float32)
            _assign_like(i_var, i_target)
        if k_var is not None:
            k_target = np.where(active, 1.0, 0.0).astype(np.float32)
            _assign_like(k_var, k_target)

        bq = getattr(layer, "bq", None)
        if bq is not None:
            cols_active = np.sum(active, axis=0) > 0
            bb_var = _get_kq_var(bq, "b")
            if bb_var is None:
                bb_var = _get_kq_var(bq, "f")
            bi_var = _get_kq_var(bq, "i")
            bk_var = _get_kq_var(bq, "k")
            if bb_var is not None:
                bb_target = np.where(cols_active, 1.8, 0.0).astype(np.float32)
                _assign_like(bb_var, bb_target)
            if bi_var is not None:
                bi_target = np.where(cols_active, 0.0, -16.0).astype(np.float32)
                _assign_like(bi_var, bi_target)
            if bk_var is not None:
                bk_target = np.where(cols_active, 1.0, 0.0).astype(np.float32)
                _assign_like(bk_var, bk_target)

        row_deg = np.sum(active, axis=1)
        col_deg = np.sum(active, axis=0)
        reports.append(
            {
                "layer": layer.name,
                "selected_rows": float(meta[li]["row_idx"].size),
                "selected_cols": float(meta[li]["col_idx"].size),
                "target_edges": float(target_edges),
                "density_ratio_layer": float(meta[li]["density_ratio_layer"]),
                "actual_edges": float(np.sum(active)),
                "row_zero_ratio": float(np.mean(row_deg == 0)),
                "col_zero_ratio": float(np.mean(col_deg == 0)),
                "stable_rank_before": float(_safe_stable_rank(meta[li]["kernel"])),
                "stable_rank_after": float(_safe_stable_rank(kernel_new)),
            }
        )

    # Keep theory-prune quantizer assignment explicit; global re-estimation may
    # inflate integer bits and destroy low-budget sparsity geometry.
    measured = compute_model_ebops(model, sample_input)
    return {
        "layers": reports,
        "theory_constraints": {
            "strict_no_baseline_in_pruning": True,
            "budget_aware_neuron_backbone": True,
            "row_coverage_balancing": True,
            "deadzone_avoidance": True,
            "bridge_flow_conservation": True,
        },
        "bridge_waste_before": bridge_waste_before,
        "bridge_waste_after": bridge_waste_after,
        "flow_fix": flow_fix,
        "shrink_ratio": float(shrink_ratio),
        "width_ratio": float(width_ratio),
        "density_ratio": float(density_ratio),
        "grad_mix": float(grad_mix),
        "grad_probe_loss": (None if grad_probe_loss is None else float(grad_probe_loss)),
        "layer_sensitivity": [float(v) for v in layer_sensitivity],
        "layer_density_factors": [float(v) for v in layer_density_factors],
        "post_theory_prune_ebops_measured": float(measured),
    }


def collect_topology_profile(model) -> dict:
    layer_metrics = []
    per_layer_edges = []
    for layer in _dense_prunable_layers(model):
        kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
        if kernel.ndim != 2:
            continue
        active = _layer_active_mask(layer)
        edge_n = int(np.sum(active))
        per_layer_edges.append(edge_n)
        row_deg = np.sum(active, axis=1).astype(np.float32)
        col_deg = np.sum(active, axis=0).astype(np.float32)
        sigma_min, cond, _, _, _, _ = _spectral_metrics(kernel, active)
        layer_metrics.append(
            {
                "layer": layer.name,
                "shape": [int(kernel.shape[0]), int(kernel.shape[1])],
                "edges": float(edge_n),
                "density": float(edge_n / max(kernel.size, 1)),
                "row_zero_ratio": float(np.mean(row_deg == 0)),
                "col_zero_ratio": float(np.mean(col_deg == 0)),
                "row_deg_mean": float(np.mean(row_deg)),
                "row_deg_cv": float(np.std(row_deg) / (np.mean(row_deg) + 1e-12)),
                "col_deg_mean": float(np.mean(col_deg)),
                "col_deg_cv": float(np.std(col_deg) / (np.mean(col_deg) + 1e-12)),
                "stable_rank_active": float(_safe_stable_rank(np.where(active, kernel, 0.0))),
                "sigma_min_active": float(sigma_min),
                "cond_active": float(cond),
            }
        )

    edges_arr = np.array(per_layer_edges, dtype=np.float64)
    edge_total = float(np.sum(edges_arr))
    if edge_total > 0 and edges_arr.size > 1:
        p = edges_arr / edge_total
        ent = -float(np.sum(p * np.log(p + 1e-12)))
        ent_norm = ent / float(np.log(len(p)))
    else:
        ent_norm = 0.0

    return {
        "layers": layer_metrics,
        "global": {
            "active_edges_total": float(edge_total),
            "layer_edge_entropy_norm": float(ent_norm),
            "n_layers": float(len(layer_metrics)),
        },
    }


def topology_closeness_score(profile_a: dict, profile_b: dict) -> dict:
    la = {x["layer"]: x for x in profile_a.get("layers", [])}
    lb = {x["layer"]: x for x in profile_b.get("layers", [])}
    common = sorted(set(la) & set(lb))
    if not common:
        return {"score": 0.0, "detail": {}, "common_layers": []}

    feat_names = [
        "density",
        "row_zero_ratio",
        "col_zero_ratio",
        "row_deg_cv",
        "col_deg_cv",
        "stable_rank_active",
        "sigma_min_active",
    ]
    feat_diffs = {k: [] for k in feat_names}

    for name in common:
        a = la[name]
        b = lb[name]
        for fn in feat_names:
            av = float(a.get(fn, 0.0))
            bv = float(b.get(fn, 0.0))
            diff = abs(av - bv) / (abs(bv) + 1e-6)
            feat_diffs[fn].append(float(np.clip(diff, 0.0, 5.0)))

    feat_mean = {k: float(np.mean(v)) for k, v in feat_diffs.items()}
    local_gap = float(np.mean(list(feat_mean.values())))
    local_score = float(np.exp(-local_gap))

    # Layer edge allocation similarity
    ea = np.array([float(la[n]["edges"]) for n in common], dtype=np.float64)
    eb = np.array([float(lb[n]["edges"]) for n in common], dtype=np.float64)
    if np.sum(ea) > 0:
        ea = ea / np.sum(ea)
    if np.sum(eb) > 0:
        eb = eb / np.sum(eb)
    alloc_l1 = float(np.sum(np.abs(ea - eb)) / 2.0)
    alloc_score = 1.0 - alloc_l1

    score = float(np.clip(0.7 * local_score + 0.3 * alloc_score, 0.0, 1.0))
    return {
        "score": score,
        "detail": {
            "local_score": local_score,
            "alloc_score": float(alloc_score),
            "feature_gap_mean": feat_mean,
            "edge_alloc_l1": alloc_l1,
        },
        "common_layers": common,
    }


def boost_active_bits(model, scale: float = 1.25, b_min: float = 0.2, b_max: float = 8.0):
    for layer in _flatten_layers(model):
        kq = getattr(layer, "kq", None)
        if kq is None:
            continue
        b_var = _get_kq_var(kq, "b")
        if b_var is None:
            b_var = _get_kq_var(kq, "f")
        if b_var is None:
            continue
        b_arr = np.array(b_var.numpy(), dtype=np.float32)
        active = b_arr > 0.0
        if not np.any(active):
            continue
        b_new = b_arr.copy()
        b_new[active] = np.clip(b_new[active] * float(scale), float(b_min), float(b_max))
        _assign_like(b_var, b_new)


class EBOPsFloorRescueCallback(keras.callbacks.Callback):
    """When EBOPs collapses below floor, force a small bitwidth rescue step."""

    def __init__(
        self,
        target_ebops: float,
        floor_ratio: float = 0.90,
        rescue_scale: float = 1.10,
        b_min: float = 0.20,
        b_max: float = 8.0,
    ):
        super().__init__()
        self.target_ebops = float(target_ebops)
        self.floor_ratio = float(floor_ratio)
        self.rescue_scale = float(rescue_scale)
        self.b_min = float(b_min)
        self.b_max = float(b_max)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        eb = float(logs.get("ebops", float("nan")))
        if (not np.isfinite(eb)) or eb <= 0:
            return
        floor = self.target_ebops * self.floor_ratio
        if eb < floor:
            boost_active_bits(self.model, scale=self.rescue_scale, b_min=self.b_min, b_max=self.b_max)
            logs["ebops_floor_rescue"] = 1.0


def repair_by_spectral_and_deadzone(
    model,
    sigma_min_target: float = 0.02,
    spectral_steps: int = 3,
    spectral_lr: float = 0.8,
    min_degree: int = 2,
    b_floor_new_edge: float = 0.45,
    b_floor_deadzone: float = 0.75,
    deadzone_lift: float = 0.75,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    repaired_layers = []
    total_new_edges = 0
    total_dead_fixed = 0

    dense_layers = _dense_prunable_layers(model)
    for li, layer in enumerate(dense_layers):
        kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
        if kernel.ndim != 2:
            continue
        in_dim, out_dim = kernel.shape

        active0 = _layer_active_mask(layer)
        active = active0.copy()
        active_cols0 = np.sum(active0, axis=0) > 0

        # 1) 连通性修复：最后一层必须保证每个输出类别可达；其余层仅补活跃子图。
        if li == len(dense_layers) - 1:
            cols_for_repair = np.arange(out_dim, dtype=np.int32)
        else:
            cols_for_repair = np.where(active_cols0)[0]
        degree_floor = int(min_degree)
        if li == len(dense_layers) - 1:
            degree_floor = max(degree_floor, 3)
        for o in cols_for_repair:
            cur_deg = int(np.sum(active[:, o]))
            need = max(0, degree_floor - cur_deg)
            if need <= 0:
                continue
            cands = np.where(~active[:, o])[0]
            if cands.size == 0:
                continue
            order = cands[np.argsort(np.abs(kernel[cands, o]))[::-1]]
            chosen = order[:need]
            active[chosen, o] = True

        new_edges = active & (~active0)
        n_new = int(np.sum(new_edges))
        if n_new > 0:
            std = float(math.sqrt(2.0 / max(in_dim, 1)))
            idx = np.where(new_edges)
            old_vals = kernel[idx]
            rand_vals = rng.normal(0.0, std, size=n_new).astype(np.float32)
            init_vals = np.where(np.abs(old_vals) > 1e-8, old_vals, rand_vals)
            kernel[idx] = init_vals.astype(np.float32)

        # 2) 死区修复：active 且 Q(w)=0 的连接抬出死区
        dead_before = _deadzone_mask(layer, kernel, active)
        bits = _bits_np(layer, kernel.shape)
        q_step = np.power(2.0, -np.clip(bits, 0.0, 12.0)).astype(np.float32)
        min_abs = deadzone_lift * q_step
        sign = np.sign(kernel).astype(np.float32)
        rand_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=kernel.shape)
        sign = np.where(np.abs(sign) > 1e-8, sign, rand_sign)
        kernel = np.where(dead_before, sign * np.maximum(np.abs(kernel), min_abs), kernel).astype(np.float32)

        # 3) 谱修复：sigma_min 不足时沿最小奇异向量方向做 rank-boost
        sigma_before, cond_before, _, _, _, _ = _spectral_metrics(kernel, active)
        sigma_after = sigma_before
        cond_after = cond_before
        for _ in range(max(0, int(spectral_steps))):
            sigma_cur, cond_cur, u_min, v_min, row_idx, col_idx = _spectral_metrics(kernel, active)
            sigma_after, cond_after = sigma_cur, cond_cur
            if u_min is None or v_min is None or row_idx is None or col_idx is None:
                break
            if sigma_cur >= float(sigma_min_target):
                break
            deficit = float(sigma_min_target - sigma_cur)
            delta_sub = (float(spectral_lr) * deficit * np.outer(u_min, v_min)).astype(np.float32)
            sub_active = active[np.ix_(row_idx, col_idx)].astype(np.float32)
            kernel[np.ix_(row_idx, col_idx)] = (
                kernel[np.ix_(row_idx, col_idx)] + delta_sub * sub_active
            ).astype(np.float32)

        layer.kernel.assign(kernel)
        _sync_quantizers_for_active(
            layer=layer,
            kernel=kernel,
            active=active,
            new_edges=new_edges,
            dead_edges=dead_before,
            b_floor_new=float(b_floor_new_edge),
            b_floor_dead=float(b_floor_deadzone),
        )

        total_new_edges += n_new
        total_dead_fixed += int(np.sum(dead_before))
        repaired_layers.append(
            {
                "layer": layer.name,
                "new_edges": float(n_new),
                "deadzone_fixed": float(np.sum(dead_before)),
                "sigma_min_before": float(sigma_before),
                "sigma_min_after": float(sigma_after),
                "cond_before": float(cond_before),
                "cond_after": float(cond_after),
            }
        )

    # 重新估计量化范围，避免修复后 i/b 与权重幅值失配
    _reestimate_quantizer_ranges(
        model,
        b_floor=max(0.5, b_floor_new_edge),
        b_ceiling=6.0,
        i_min=-2.0,
        i_max=2.0,
    )

    return {
        "layers": repaired_layers,
        "total_new_edges": float(total_new_edges),
        "total_deadzone_fixed": float(total_dead_fixed),
    }


def save_topology_plots(
    model,
    out_dir: Path,
    prefix: str,
    symmetric_topology_plot: bool = False,
    mirror_edges: bool = False,
    plot_matrix: bool = False,
    strict_original_connections: bool = True,
) -> dict[str, str | None]:
    plotter = TopologyGraphPlotter(
        symmetric_topology_plot=bool(symmetric_topology_plot),
        mirror_edges=bool(mirror_edges),
        plot_matrix=bool(plot_matrix),
        strict_original_connections=bool(strict_original_connections),
    )
    layers = plotter.extract_layer_graph_data(model)
    if not layers:
        raise RuntimeError("No quantized layers found for topology plotting.")

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / f"{prefix}_weighted_topology_matrix.png"
    circle_path = out_dir / f"{prefix}_weighted_circle_graph.png"
    if bool(plot_matrix):
        plotter.plot_weighted_topology_matrices(layers, matrix_path)
        matrix_out = str(matrix_path)
    else:
        matrix_out = None
    plotter.plot_circle_graph(layers, circle_path)
    return {
        "matrix_path": matrix_out,
        "circle_path": str(circle_path),
    }


def _to_float(x) -> float:
    try:
        return float(x.numpy())
    except Exception:
        return float(x)


def quick_eval(model, x, y) -> tuple[float, float]:
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    logits = model(x, training=False)
    loss = _to_float(loss_fn(y, logits))
    pred = tf.argmax(logits, axis=-1)
    acc = _to_float(tf.reduce_mean(tf.cast(tf.equal(pred, tf.cast(y, pred.dtype)), tf.float32)))
    return float(loss), float(acc)


def activation_variance_wakeup(
    model,
    sample_input,
    target_std: float = 0.45,
    min_std: float = 0.08,
    max_gain: float = 4.0,
    b_floor_deadzone: float = 0.75,
) -> dict:
    layers = _dense_prunable_layers(model)
    if not layers:
        return {"applied": False, "reason": "no_prunable_layers"}

    probe = keras.Model(model.inputs, [layer.output for layer in layers])
    acts = probe(sample_input, training=False)
    if not isinstance(acts, (list, tuple)):
        acts = [acts]

    layer_reports = []
    for layer, act in zip(layers, acts):
        act_np = np.array(act.numpy(), dtype=np.float32)
        std_before = float(np.std(act_np))
        gain = 1.0
        if np.isfinite(std_before) and std_before < float(min_std):
            gain = float(np.clip(float(target_std) / max(std_before, 1e-6), 1.0, float(max_gain)))
        if gain <= 1.0001:
            layer_reports.append(
                {
                    "layer": layer.name,
                    "std_before": std_before,
                    "gain": 1.0,
                }
            )
            continue

        kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
        active = _layer_active_mask(layer)
        kernel_new = np.where(active, kernel * gain, 0.0).astype(np.float32)
        dead_before = _deadzone_mask(layer, kernel_new, active)
        layer.kernel.assign(kernel_new)
        _sync_quantizers_for_active(
            layer=layer,
            kernel=kernel_new,
            active=active,
            new_edges=np.zeros_like(active, dtype=bool),
            dead_edges=dead_before,
            b_floor_new=0.45,
            b_floor_dead=float(b_floor_deadzone),
        )
        layer_reports.append(
            {
                "layer": layer.name,
                "std_before": std_before,
                "gain": float(gain),
                "deadzone_fixed": float(np.sum(dead_before)),
            }
        )

    acts_after = probe(sample_input, training=False)
    if not isinstance(acts_after, (list, tuple)):
        acts_after = [acts_after]
    std_after = []
    for layer, act in zip(layers, acts_after):
        std_after.append({"layer": layer.name, "std_after": float(np.std(np.array(act.numpy(), dtype=np.float32)))})

    return {
        "applied": True,
        "target_std": float(target_std),
        "min_std": float(min_std),
        "max_gain": float(max_gain),
        "layers": layer_reports,
        "layers_after": std_after,
    }


def break_output_symmetry(
    model,
    noise_scale: float = 0.02,
    bias_noise_scale: float = 0.05,
    seed: int = 42,
) -> dict:
    layers = _dense_prunable_layers(model)
    if not layers:
        return {"applied": False, "reason": "no_prunable_layers"}
    layer = layers[-1]
    rng = np.random.default_rng(seed)
    kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
    active = _layer_active_mask(layer)
    if kernel.ndim != 2 or not np.any(active):
        return {"applied": False, "reason": "no_active_output_edges"}

    k_std = float(np.std(kernel[active])) if np.any(active) else 0.0
    base_std = max(k_std, 1e-2)
    noise_std = float(noise_scale) * base_std
    k_noise = rng.normal(0.0, noise_std, size=kernel.shape).astype(np.float32)
    # Class-wise signed bias to avoid equal-logit lock.
    class_sign = np.sign(np.linspace(-1.0, 1.0, kernel.shape[1], dtype=np.float32))
    class_sign = np.where(class_sign == 0.0, 1.0, class_sign).astype(np.float32)
    kernel_new = np.where(active, kernel + k_noise * class_sign[None, :], 0.0).astype(np.float32)
    layer.kernel.assign(kernel_new)

    b = getattr(layer, "bias", None)
    if b is not None:
        b_np = np.array(b.numpy(), dtype=np.float32)
        b_noise = rng.normal(0.0, float(bias_noise_scale) * base_std, size=b_np.shape).astype(np.float32)
        b.assign((b_np + b_noise * class_sign).astype(np.float32))

    dead_before = _deadzone_mask(layer, kernel_new, active)
    _sync_quantizers_for_active(
        layer=layer,
        kernel=kernel_new,
        active=active,
        new_edges=np.zeros_like(active, dtype=bool),
        dead_edges=dead_before,
        b_floor_new=0.45,
        b_floor_dead=0.75,
    )
    return {
        "applied": True,
        "layer": layer.name,
        "noise_std": float(noise_std),
        "kernel_std_active": float(base_std),
        "deadzone_fixed": float(np.sum(dead_before)),
    }


def rescue_dead_hidden_layers(
    model,
    sample_input,
    min_std: float = 1e-3,
    min_nz_ratio: float = 1e-4,
    kernel_jitter_std: float = 0.25,
    bias_boost: float = 0.35,
    seed: int = 42,
) -> dict:
    layers = _dense_prunable_layers(model)
    if not layers:
        return {"applied": False, "reason": "no_prunable_layers"}

    rng = np.random.default_rng(seed)
    probe = keras.Model(model.inputs, [layer.output for layer in layers])
    vals_before = probe(sample_input, training=False)
    if not isinstance(vals_before, (list, tuple)):
        vals_before = [vals_before]

    reports = []
    changed = 0
    for li, (layer, v) in enumerate(zip(layers, vals_before)):
        # Keep output layer untouched here; it is handled separately.
        if li == len(layers) - 1:
            continue
        v_np = np.array(v.numpy(), dtype=np.float32)
        std_before = float(np.std(v_np))
        nz_before = float(np.mean(np.abs(v_np) > 1e-12))
        if std_before >= float(min_std) and nz_before >= float(min_nz_ratio):
            reports.append(
                {
                    "layer": layer.name,
                    "std_before": std_before,
                    "nz_ratio_before": nz_before,
                    "rescued": False,
                }
            )
            continue

        kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
        active0 = _layer_active_mask(layer).astype(bool)
        if not np.any(active0):
            reports.append(
                {
                    "layer": layer.name,
                    "std_before": std_before,
                    "nz_ratio_before": nz_before,
                    "rescued": False,
                    "reason": "no_active_edges",
                }
            )
            continue

        active = active0.copy()
        # Ensure dead layer receives inputs from actually active upstream neurons.
        if li > 0:
            prev_v = np.array(vals_before[li - 1].numpy(), dtype=np.float32)
            row_activity = np.mean(np.abs(prev_v), axis=0)
            row_order = np.argsort(row_activity)[::-1]
            top_pool = row_order[: max(1, min(16, row_order.size))]
            cols_keep = np.where(np.sum(active, axis=0) > 0)[0]
            if cols_keep.size == 0:
                cols_keep = np.arange(active.shape[1], dtype=np.int32)
            for c in cols_keep:
                if np.any(active[top_pool, c]):
                    continue
                for r in top_pool:
                    if not active[r, c]:
                        active[r, c] = True
                        break

        jitter = rng.normal(0.0, float(kernel_jitter_std), size=kernel.shape).astype(np.float32)
        tiny = np.abs(kernel) < 1e-3
        kernel_new = kernel.copy()
        kernel_new = np.where(active & tiny, jitter, kernel_new).astype(np.float32)
        if float(np.max(np.abs(kernel_new[active]))) < 1e-3:
            kernel_new = np.where(active, jitter, kernel_new).astype(np.float32)
        kernel_new = np.where(active, kernel_new, 0.0).astype(np.float32)

        b = getattr(layer, "bias", None)
        if b is not None:
            b_np = np.array(b.numpy(), dtype=np.float32)
            cols_active = np.sum(active, axis=0) > 0
            b_np[cols_active] = np.where(
                b_np[cols_active] <= 0.0,
                float(bias_boost),
                b_np[cols_active] + 0.2 * float(bias_boost),
            ).astype(np.float32)
            b.assign(b_np.astype(np.float32))

        layer.kernel.assign(kernel_new.astype(np.float32))
        dead_before = _deadzone_mask(layer, kernel_new, active)
        _sync_quantizers_for_active(
            layer=layer,
            kernel=kernel_new,
            active=active,
            new_edges=(active & (~active0)),
            dead_edges=dead_before,
            b_floor_new=0.45,
            b_floor_dead=0.75,
        )
        changed += 1
        reports.append(
            {
                "layer": layer.name,
                "std_before": std_before,
                "nz_ratio_before": nz_before,
                "rescued": True,
                "deadzone_fixed": float(np.sum(dead_before)),
            }
        )

    vals_after = probe(sample_input, training=False)
    if not isinstance(vals_after, (list, tuple)):
        vals_after = [vals_after]
    after_stats = []
    for layer, v in zip(layers, vals_after):
        v_np = np.array(v.numpy(), dtype=np.float32)
        after_stats.append(
            {
                "layer": layer.name,
                "std_after": float(np.std(v_np)),
                "nz_ratio_after": float(np.mean(np.abs(v_np) > 1e-12)),
            }
        )
    return {
        "applied": True,
        "changed_layers": float(changed),
        "layers": reports,
        "after": after_stats,
    }


def repair_fused_deadzone_and_bias(
    model,
    sample_input,
    max_iter: int = 4,
    fused_b_floor: float = 2.0,
    kernel_gain: float = 1.8,
    bias_floor: float = 1.5,
    min_hidden_nz_ratio: float = 1e-3,
    seed: int = 42,
) -> dict:
    layers = _dense_prunable_layers(model)
    if not layers:
        return {"applied": False, "reason": "no_prunable_layers"}

    rng = np.random.default_rng(seed)
    probe = keras.Model(model.inputs, [layer.output for layer in layers])
    layer_reports = []
    total_changed = 0

    for it in range(max(1, int(max_iter))):
        vals = probe(sample_input, training=False)
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        changed_this_iter = 0

        for li, (layer, val) in enumerate(zip(layers, vals)):
            active = _layer_active_mask(layer).astype(bool)
            if not np.any(active):
                continue

            v_np = np.array(val.numpy(), dtype=np.float32)
            nz_ratio = float(np.mean(np.abs(v_np) > 1e-12))
            hidden_dead = (li < len(layers) - 1) and (nz_ratio < float(min_hidden_nz_ratio))

            fused_dead = _fused_deadzone_mask(layer, active)
            n_fused_dead = int(np.sum(fused_dead))
            if (not hidden_dead) and n_fused_dead <= 0:
                continue

            kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
            gain_mask = fused_dead.copy()
            if hidden_dead:
                gain_mask |= active

            if np.any(gain_mask):
                jitter = rng.normal(0.0, 0.1, size=kernel.shape).astype(np.float32)
                kernel_new = np.where(gain_mask, kernel * float(kernel_gain), kernel).astype(np.float32)
                kernel_new = np.where(
                    gain_mask & (np.abs(kernel_new) < 1e-4),
                    jitter,
                    kernel_new,
                ).astype(np.float32)
                kernel_new = np.where(active, kernel_new, 0.0).astype(np.float32)
                layer.kernel.assign(kernel_new)
            else:
                kernel_new = kernel

            b_var = _get_kq_var(layer.kq, "b")
            if b_var is None:
                b_var = _get_kq_var(layer.kq, "f")
            if b_var is not None:
                b_np = _broadcast_like(np.array(b_var.numpy(), dtype=np.float32), active.shape)
                b_np = np.where(gain_mask, np.maximum(b_np, float(fused_b_floor)), b_np)
                b_np = np.where(active, b_np, 0.0).astype(np.float32)
                _assign_like(b_var, b_np)

            i_var = _get_kq_var(layer.kq, "i")
            if i_var is not None:
                i_np = _broadcast_like(np.array(i_var.numpy(), dtype=np.float32), active.shape)
                i_np = np.where(active, np.clip(i_np, -2.0, 2.0), -16.0).astype(np.float32)
                _assign_like(i_var, i_np)

            k_var = _get_kq_var(layer.kq, "k")
            if k_var is not None:
                k_np = _broadcast_like(np.array(k_var.numpy(), dtype=np.float32), active.shape)
                k_np = np.where(active, 1.0, 0.0).astype(np.float32)
                _assign_like(k_var, k_np)

            cols_active = np.sum(active, axis=0) > 0
            b = getattr(layer, "bias", None)
            if b is not None and hidden_dead:
                b_np = np.array(b.numpy(), dtype=np.float32)
                b_np[cols_active] = np.maximum(b_np[cols_active], float(bias_floor))
                b.assign(b_np.astype(np.float32))

            bq = getattr(layer, "bq", None)
            if bq is not None:
                bb_var = _get_kq_var(bq, "b")
                if bb_var is None:
                    bb_var = _get_kq_var(bq, "f")
                if bb_var is not None:
                    bb_np = _broadcast_like(np.array(bb_var.numpy(), dtype=np.float32), cols_active.shape)
                    bb_np = np.where(cols_active, np.maximum(bb_np, float(fused_b_floor)), 0.0)
                    _assign_like(bb_var, bb_np)
                bk_var = _get_kq_var(bq, "k")
                if bk_var is not None:
                    bk_np = _broadcast_like(np.array(bk_var.numpy(), dtype=np.float32), cols_active.shape)
                    bk_np = np.where(cols_active, 1.0, 0.0).astype(np.float32)
                    _assign_like(bk_var, bk_np)

            fused_dead_after = int(np.sum(_fused_deadzone_mask(layer, active)))
            layer_reports.append(
                {
                    "iter": float(it + 1),
                    "layer": layer.name,
                    "hidden_dead": bool(hidden_dead),
                    "nz_ratio_before": float(nz_ratio),
                    "fused_dead_before": float(n_fused_dead),
                    "fused_dead_after": float(fused_dead_after),
                }
            )
            changed_this_iter += 1

        total_changed += changed_this_iter
        if changed_this_iter <= 0:
            break

    vals_after = probe(sample_input, training=False)
    if not isinstance(vals_after, (list, tuple)):
        vals_after = [vals_after]
    after = []
    for layer, val in zip(layers, vals_after):
        v_np = np.array(val.numpy(), dtype=np.float32)
        after.append(
            {
                "layer": layer.name,
                "std_after": float(np.std(v_np)),
                "nz_ratio_after": float(np.mean(np.abs(v_np) > 1e-12)),
            }
        )
    return {
        "applied": True,
        "changed_steps": float(total_changed),
        "fused_b_floor": float(fused_b_floor),
        "kernel_gain": float(kernel_gain),
        "layers": layer_reports,
        "after": after,
    }


def refit_output_head_from_features(
    model,
    x_ref,
    y_ref,
    ridge_lambda: float = 1e-2,
) -> dict:
    layers = _dense_prunable_layers(model)
    if len(layers) < 2:
        return {"applied": False, "reason": "insufficient_layers"}

    out_layer = layers[-1]
    prev_layer = layers[-2]
    active = _layer_active_mask(out_layer).astype(bool)
    if not np.any(active):
        return {"applied": False, "reason": "no_active_output_edges"}

    feat_model = keras.Model(model.inputs, prev_layer.output)
    feats = np.array(feat_model(x_ref, training=False).numpy(), dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] < 8:
        return {"applied": False, "reason": "invalid_feature_shape"}
    if float(np.std(feats)) < 1e-9:
        return {"applied": False, "reason": "zero_feature_variance"}

    y_np = np.array(y_ref.numpy(), dtype=np.int32).reshape(-1)
    n_class = int(out_layer.kernel.shape[1])
    if y_np.size != feats.shape[0]:
        n = min(int(y_np.size), int(feats.shape[0]))
        y_np = y_np[:n]
        feats = feats[:n, :]
    y_oh = np.eye(n_class, dtype=np.float32)[np.clip(y_np, 0, n_class - 1)]

    x_aug = np.concatenate([feats, np.ones((feats.shape[0], 1), dtype=np.float32)], axis=1)
    gram = x_aug.T @ x_aug
    gram += float(ridge_lambda) * np.eye(gram.shape[0], dtype=np.float32)
    rhs = x_aug.T @ y_oh
    try:
        wb = np.linalg.solve(gram, rhs).astype(np.float32)
    except Exception:
        wb = np.linalg.lstsq(x_aug, y_oh, rcond=None)[0].astype(np.float32)

    w_fit = wb[:-1, :]
    b_fit = wb[-1, :]
    w_fit = np.clip(w_fit, -8.0, 8.0).astype(np.float32)
    b_fit = np.clip(b_fit, -8.0, 8.0).astype(np.float32)

    k_new = np.where(active, w_fit, 0.0).astype(np.float32)
    out_layer.kernel.assign(k_new)
    if getattr(out_layer, "bias", None) is not None:
        out_layer.bias.assign(b_fit.astype(np.float32))

    dead = _deadzone_mask(out_layer, k_new, active)
    _sync_quantizers_for_active(
        layer=out_layer,
        kernel=k_new,
        active=active,
        new_edges=np.zeros_like(active, dtype=bool),
        dead_edges=dead,
        b_floor_new=2.0,
        b_floor_dead=2.0,
    )

    logits = np.array(model(x_ref, training=False).numpy(), dtype=np.float32)
    pred = np.argmax(logits, axis=-1)
    acc = float(np.mean(pred == y_np))
    bins = np.bincount(pred, minlength=n_class).astype(np.int32).tolist()
    return {
        "applied": True,
        "ridge_lambda": float(ridge_lambda),
        "ref_acc_on_sample": float(acc),
        "pred_hist_on_sample": [int(v) for v in bins],
        "logits_std_on_sample": float(np.std(logits)),
    }


def gradient_probe(model, x, y, one_step_lr: float = 1e-3) -> dict:
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)

    g_sumsq = 0.0
    g_total = 0
    g_near_zero = 0
    kernel_grad_norms = []

    for v, g in zip(model.trainable_variables, grads):
        if g is None:
            continue
        g_np = np.array(g.numpy(), dtype=np.float32)
        g_sumsq += float(np.sum(g_np * g_np))
        g_total += int(g_np.size)
        g_near_zero += int(np.sum(np.abs(g_np) < 1e-8))
        if "kernel" in v.name:
            kernel_grad_norms.append(float(np.linalg.norm(g_np)))

    first_last = 0.0
    if kernel_grad_norms:
        first_last = float(kernel_grad_norms[-1] / (kernel_grad_norms[0] + 1e-12))

    pre_loss = float(_to_float(loss))
    backups = [v.numpy().copy() for v in model.trainable_variables]
    opt = keras.optimizers.SGD(learning_rate=float(one_step_lr), momentum=0.0)
    grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    if grads_vars:
        opt.apply_gradients(grads_vars)
    post_logits = model(x, training=True)
    post_loss = float(_to_float(loss_fn(y, post_logits)))
    for v, b in zip(model.trainable_variables, backups):
        v.assign(b)

    one_step_drop = float(pre_loss - post_loss)
    grad_norm = float(math.sqrt(g_sumsq))
    near_zero_ratio = float(g_near_zero / max(g_total, 1))
    logits_std = float(np.std(np.array(logits.numpy(), dtype=np.float32)))

    trainable = (
        np.isfinite(grad_norm)
        and grad_norm > 1e-10
        and near_zero_ratio < 0.99995
        and logits_std > 1e-9
        and one_step_drop > -5e-2
    )

    return {
        "grad_global_norm": grad_norm,
        "grad_near_zero_ratio": near_zero_ratio,
        "grad_first_last_ratio": float(first_last),
        "logits_std": float(logits_std),
        "one_step_loss_drop": one_step_drop,
        "trainable_proxy": bool(trainable),
    }


def _set_all_beta(model, beta_value: float):
    bv = tf.constant(beta_value, dtype=tf.float32)
    for layer in _flatten_layers(model):
        if hasattr(layer, "_beta"):
            try:
                layer._beta.assign(tf.ones_like(layer._beta) * bv)
            except Exception:
                pass


def run_smoke_train(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    target_ebops: float,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    output_dir: str | Path,
    min_steps_per_epoch: int = 8,
    enable_budget_control: bool = False,
    freeze_quantizers: bool = True,
    grad_clip_norm: float | None = None,
    freeze_batchnorm: bool = False,
):
    if int(epochs) <= 0:
        return {}

    # Verification focuses on trainability; disable residual EBOPs penalty from checkpoint.
    _set_all_beta(model, 0.0)

    if bool(freeze_quantizers):
        for layer in _flatten_layers(model):
            kq = getattr(layer, "kq", None)
            if kq is not None:
                kq.trainable = False
            bq = getattr(layer, "bq", None)
            if bq is not None:
                bq.trainable = False

    if bool(freeze_batchnorm):
        for layer in _flatten_layers(model):
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(int(len(x_train)), seed=42, reshuffle_each_iteration=True)
        .repeat()
        .batch(int(batch_size), drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(int(batch_size), drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    effective_steps_per_epoch = max(1, int(steps_per_epoch))
    if int(epochs) >= 100 and effective_steps_per_epoch < int(min_steps_per_epoch):
        effective_steps_per_epoch = int(min_steps_per_epoch)
        print(
            f"  [TrainAdjust] steps_per_epoch too small for long run, "
            f"auto raise to {effective_steps_per_epoch}"
        )

    opt = keras.optimizers.Adam(
        learning_rate=float(learning_rate),
        clipnorm=(None if grad_clip_norm is None else float(grad_clip_norm)),
    )
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        steps_per_execution=1,
    )

    ebops_cb = FreeEBOPs()
    pareto_cb = ParetoFront(
        str(output_dir),
        ["val_accuracy", "ebops"],
        [1, -1],
        fname_format="epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras",
    )
    callbacks = [ebops_cb, pareto_cb]
    if bool(enable_budget_control):
        beta_ctrl = BetaOnlyBudgetController(
            target_ebops=float(target_ebops),
            margin=0.10,
            beta_init=3e-6,
            beta_min=1e-8,
            beta_max=5e-4,
            adjust_factor=1.2,
            ema_alpha=0.25,
            warmup_epochs=max(20, int(epochs) // 10),
            max_change_ratio=1.5,
            init_ebops=float(target_ebops),
            rescue_threshold=0.10,
            rescue_rate=0.60,
            rescue_max_alpha=1.20,
        )
        floor_rescue_cb = EBOPsFloorRescueCallback(
            target_ebops=float(target_ebops),
            floor_ratio=0.90,
            rescue_scale=1.10,
            b_min=0.20,
            b_max=8.0,
        )
        callbacks += [beta_ctrl, floor_rescue_cb]

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(epochs),
        steps_per_epoch=effective_steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    ).history

    return {
        "history": {
            "loss": [float(v) for v in hist.get("loss", [])],
            "accuracy": [float(v) for v in hist.get("accuracy", [])],
            "val_loss": [float(v) for v in hist.get("val_loss", [])],
            "val_accuracy": [float(v) for v in hist.get("val_accuracy", [])],
            "ebops": [float(v) for v in hist.get("ebops", [])],
        }
    }


def main():
    parser = argparse.ArgumentParser(description="One-shot prune + spectral/deadzone repair for 400 EBOPs")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--target_ebops", type=float, default=400.0)
    parser.add_argument(
        "--prune_method",
        type=str,
        default="spectral_quant",
        choices=["uniform", "sensitivity", "spectral_quant", "topo_quant_theory"],
    )
    parser.add_argument("--input_h5", type=str, default="data/dataset.h5")
    parser.add_argument("--output_dir", type=str, default="results/prune_repair_400")
    parser.add_argument("--sample_size", type=int, default=2048)
    parser.add_argument("--min_degree", type=int, default=2)
    parser.add_argument("--b_floor", type=float, default=0.35)
    parser.add_argument("--low_budget_threshold", type=float, default=900.0)
    parser.add_argument("--min_hidden_width", type=int, default=4)
    parser.add_argument("--near_budget_ratio", type=float, default=1.6)
    parser.add_argument("--high_budget_ratio", type=float, default=0.45)
    parser.add_argument("--sigma_min_target", type=float, default=0.02)
    parser.add_argument("--spectral_steps", type=int, default=4)
    parser.add_argument("--spectral_lr", type=float, default=0.8)
    parser.add_argument("--deadzone_lift", type=float, default=0.75)
    parser.add_argument("--deadzone_b_floor", type=float, default=0.75)
    parser.add_argument("--calib_tolerance", type=float, default=0.03)
    parser.add_argument("--calib_b_k_min", type=float, default=0.20)
    parser.add_argument("--verify_train_samples", type=int, default=33200)
    parser.add_argument("--verify_val_samples", type=int, default=8192)
    parser.add_argument("--verify_batch_size", type=int, default=4096)
    parser.add_argument("--verify_epochs", type=int, default=3)
    parser.add_argument("--verify_steps_per_epoch", type=int, default=2)
    parser.add_argument("--verify_lr", type=float, default=3e-4)
    parser.add_argument("--verify_enable_budget_control", action="store_true", default=False)
    parser.add_argument("--verify_freeze_quantizers", action="store_true", default=False)
    parser.add_argument("--teacher_calib_passes", type=int, default=0)
    parser.add_argument("--min_steps_per_epoch_longrun", type=int, default=8)
    parser.add_argument("--natural_baseline_dir", type=str, default="results/baseline")
    parser.add_argument("--topo_row_target_scale", type=float, default=1.0)
    parser.add_argument("--topo_grad_mix", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("One-shot prune -> topology plot -> spectral/deadzone repair -> verify")
    print(f"  checkpoint      : {args.checkpoint}")
    print(f"  target_ebops    : {args.target_ebops}")
    print(f"  prune_method    : {args.prune_method}")
    print(f"  output_dir      : {out_dir}")
    print("=" * 72)

    print("\n[1/7] Loading data...")
    (x_train, y_train), (x_val, y_val), _ = get_data(args.input_h5, src="openml")
    n_sample = min(int(args.sample_size), len(x_train))
    sample_input = tf.constant(x_train[:n_sample], dtype=tf.float32)
    sample_target = tf.constant(y_train[:n_sample], dtype=tf.int32)

    print("\n[2/7] Loading model...")
    model = keras.models.load_model(args.checkpoint, compile=False)
    baseline_ebops = compute_model_ebops(model, sample_input)
    print(f"  baseline_ebops(measured) = {baseline_ebops:.1f}")
    natural_ckpt, natural_ebops_hint, natural_pick_warn = choose_natural_ckpt_near_target(
        target_ebops=float(args.target_ebops),
        baseline_dir=args.natural_baseline_dir,
        fallback=args.checkpoint,
    )
    if natural_pick_warn:
        print(f"  [WARN] {natural_pick_warn}")
    print(f"  Natural baseline ckpt for plotting: {natural_ckpt} (ebops≈{natural_ebops_hint})")
    natural_model = keras.models.load_model(natural_ckpt, compile=False)
    print("  Saving baseline (natural, near-target) topology plots...")
    baseline_plot = save_topology_plots(
        natural_model,
        out_dir,
        prefix=f"baseline_natural_target{int(args.target_ebops)}",
    )
    baseline_topology_profile = collect_topology_profile(natural_model)
    if baseline_plot["matrix_path"] is not None:
        print(f"  saved: {baseline_plot['matrix_path']}")
    print(f"  saved: {baseline_plot['circle_path']}")

    print("\n[3/7] One-shot pruning...")
    used_structured_low_budget = False
    theory_prune_report = None
    if args.prune_method == "sensitivity":
        pruner = SensitivityAwarePruner(
            target_ebops=float(args.target_ebops),
            pruned_threshold=0.1,
            b_k_min=max(float(args.b_floor), 0.3),
        )
        pruner.prune_to_ebops(model, current_ebops=baseline_ebops, verbose=True)
    elif args.prune_method == "uniform":
        pruner = HighBitPruner(target_ebops=float(args.target_ebops), pruned_threshold=0.1)
        pruner.prune_to_ebops(model, current_ebops=baseline_ebops, verbose=True)
    elif args.prune_method == "topo_quant_theory":
        used_structured_low_budget = True
        theory_prune_report = prune_by_topology_quant_theory(
            model,
            target_ebops=float(args.target_ebops),
            sample_input=sample_input,
            sample_target=sample_target,
            b_floor=max(float(args.b_floor), 0.35),
            b_ceiling=6.0,
            deadzone_lift=max(float(args.deadzone_lift), 0.8),
            row_target_scale=float(args.topo_row_target_scale),
            grad_mix=float(args.topo_grad_mix),
            seed=int(args.seed),
        )
    else:
        _, used_structured_low_budget = spectral_quant_prune_to_ebops(
            model,
            target_ebops=float(args.target_ebops),
            sample_input=sample_input,
            min_degree=int(args.min_degree),
            b_floor=float(args.b_floor),
            low_budget_structured=True,
            low_budget_threshold=float(args.low_budget_threshold),
            min_hidden_width=int(args.min_hidden_width),
            near_budget_ratio=float(args.near_budget_ratio),
            high_budget_ratio=float(args.high_budget_ratio),
            verbose=True,
        )

    post_prune_ebops = compute_model_ebops(model, sample_input)
    print(f"  post_prune_ebops(measured) = {post_prune_ebops:.1f}")
    post_prune_ebops = bisect_ebops_to_target(
        model,
        target_ebops=float(args.target_ebops),
        sample_input=sample_input,
        tolerance=float(args.calib_tolerance),
        max_iter=30,
        b_k_min=float(args.calib_b_k_min),
        b_k_max=8.0,
        allow_connection_kill=(not used_structured_low_budget),
    )
    print(f"  post_prune_ebops(calibrated) = {post_prune_ebops:.1f}")

    print("\n[4/7] Saving topology plot after one-shot pruning...")
    prune_plot = save_topology_plots(model, out_dir, prefix="post_prune")
    post_prune_model = out_dir / "model_post_prune.keras"
    post_prune_weights = out_dir / "model_post_prune.weights.h5"
    model.save(post_prune_model)
    model.save_weights(post_prune_weights)
    post_prune_topology_profile = collect_topology_profile(model)
    prune_vs_baseline = topology_closeness_score(post_prune_topology_profile, baseline_topology_profile)
    if prune_plot["matrix_path"] is not None:
        print(f"  saved: {prune_plot['matrix_path']}")
    print(f"  saved: {prune_plot['circle_path']}")
    print(f"  topology closeness(post_prune vs baseline_natural) = {prune_vs_baseline['score']:.4f}")

    print("\n[5/7] Spectral + deadzone repair...")
    deadzone_before = collect_deadzone_stats(model)
    repair_report = repair_by_spectral_and_deadzone(
        model,
        sigma_min_target=float(args.sigma_min_target),
        spectral_steps=int(args.spectral_steps),
        spectral_lr=float(args.spectral_lr),
        min_degree=int(args.min_degree),
        b_floor_new_edge=max(float(args.b_floor), 0.45),
        b_floor_deadzone=float(args.deadzone_b_floor),
        deadzone_lift=float(args.deadzone_lift),
        seed=int(args.seed),
    )
    repaired_ebops = bisect_ebops_to_target(
        model,
        target_ebops=float(args.target_ebops),
        sample_input=sample_input,
        tolerance=float(args.calib_tolerance),
        max_iter=30,
        b_k_min=float(args.calib_b_k_min),
        b_k_max=8.0,
        allow_connection_kill=False,
    )
    if repaired_ebops < float(args.target_ebops) * 0.90:
        for _ in range(4):
            boost_active_bits(model, scale=1.35, b_min=float(args.calib_b_k_min), b_max=8.0)
            repaired_ebops = bisect_ebops_to_target(
                model,
                target_ebops=float(args.target_ebops),
                sample_input=sample_input,
                tolerance=float(args.calib_tolerance),
                max_iter=30,
                b_k_min=float(args.calib_b_k_min),
                b_k_max=8.0,
                allow_connection_kill=False,
            )
            if repaired_ebops >= float(args.target_ebops) * 0.90:
                break

    connectivity_fix_rounds = []
    for round_idx in range(3):
        conn_before = collect_flow_connectivity_stats(model)
        n_bad = (
            conn_before["hidden_in_only_total"]
            + conn_before["hidden_out_only_total"]
            + conn_before["output_missing_classes"]
        )
        if n_bad <= 0:
            break
        print(
            f"  [ConnectivityFix] round={round_idx + 1} "
            f"in_only={conn_before['hidden_in_only_total']:.0f} "
            f"out_only={conn_before['hidden_out_only_total']:.0f} "
            f"output_missing={conn_before['output_missing_classes']:.0f}"
        )
        fix_report = enforce_model_flow_and_output_connectivity(
            model,
            min_output_degree=1,
            b_floor_new_edge=max(float(args.b_floor), 0.45),
            b_floor_deadzone=float(args.deadzone_b_floor),
            deadzone_lift=float(args.deadzone_lift),
            max_iter=5,
            seed=int(args.seed) + round_idx,
        )
        repaired_ebops = bisect_ebops_to_target(
            model,
            target_ebops=float(args.target_ebops),
            sample_input=sample_input,
            tolerance=float(args.calib_tolerance),
            max_iter=30,
            b_k_min=float(args.calib_b_k_min),
            b_k_max=8.0,
            allow_connection_kill=False,
        )
        connectivity_fix_rounds.append(
            {
                "round": float(round_idx + 1),
                "before": conn_before,
                "fix_report": fix_report,
                "post_recalib_ebops": float(repaired_ebops),
            }
        )

    connectivity_final = collect_flow_connectivity_stats(model)
    if connectivity_final["output_missing_classes"] > 0:
        raise RuntimeError(
            "Final repaired model still has disconnected output classes: "
            f"{int(connectivity_final['output_missing_classes'])}"
        )
    if (
        connectivity_final["hidden_in_only_total"] > 0
        or connectivity_final["hidden_out_only_total"] > 0
    ):
        print(
            "  [WARN] hidden bridge mismatch remains: "
            f"in_only={connectivity_final['hidden_in_only_total']:.0f}, "
            f"out_only={connectivity_final['hidden_out_only_total']:.0f}"
        )

    teacher_calib_report = {
        "enabled": bool(int(args.teacher_calib_passes) > 0),
        "passes": float(max(0, int(args.teacher_calib_passes))),
    }
    if int(args.teacher_calib_passes) > 0:
        teacher_ebops = teacher_guided_post_prune_calibration(
            student_model=model,
            teacher_model=natural_model,
            sample_input=sample_input,
            passes=int(args.teacher_calib_passes),
            b_floor=max(float(args.b_floor), 0.35),
            b_ceiling=6.0,
            verbose=True,
        )
        teacher_calib_report["post_teacher_calib_ebops"] = float(teacher_ebops)

    wakeup_report = activation_variance_wakeup(
        model,
        sample_input=sample_input,
        target_std=0.45,
        min_std=0.08,
        max_gain=4.0,
        b_floor_deadzone=float(args.deadzone_b_floor),
    )
    output_symmetry_report = break_output_symmetry(
        model,
        noise_scale=0.02,
        bias_noise_scale=0.05,
        seed=int(args.seed),
    )
    dead_layer_rescue_report = rescue_dead_hidden_layers(
        model,
        sample_input=sample_input,
        min_std=1e-3,
        min_nz_ratio=1e-4,
        kernel_jitter_std=0.25,
        bias_boost=0.35,
        seed=int(args.seed),
    )
    fused_deadzone_repair_report = repair_fused_deadzone_and_bias(
        model,
        sample_input=sample_input,
        max_iter=4,
        fused_b_floor=2.0,
        kernel_gain=1.8,
        bias_floor=1.5,
        min_hidden_nz_ratio=1e-3,
        seed=int(args.seed),
    )
    repaired_ebops = bisect_ebops_to_target(
        model,
        target_ebops=float(args.target_ebops),
        sample_input=sample_input,
        tolerance=float(args.calib_tolerance),
        max_iter=30,
        b_k_min=float(args.calib_b_k_min),
        b_k_max=8.0,
        allow_connection_kill=False,
    )
    deadzone_after = collect_deadzone_stats(model)
    print(f"  repaired_ebops(calibrated) = {repaired_ebops:.1f}")
    print(
        "  deadzone ratio: "
        f"{deadzone_before['deadzone_ratio']:.4f} -> {deadzone_after['deadzone_ratio']:.4f}"
    )
    print(
        "  connectivity: "
        f"in_only={connectivity_final['hidden_in_only_total']:.0f}, "
        f"out_only={connectivity_final['hidden_out_only_total']:.0f}, "
        f"output_missing={connectivity_final['output_missing_classes']:.0f}"
    )
    if dead_layer_rescue_report.get("applied", False):
        print(
            "  dead_layer_rescue: "
            f"changed_layers={dead_layer_rescue_report.get('changed_layers', 0):.0f}"
        )
    if fused_deadzone_repair_report.get("applied", False):
        print(
            "  fused_deadzone_repair: "
            f"changed_steps={fused_deadzone_repair_report.get('changed_steps', 0):.0f}"
        )

    print("\n[6/7] Saving topology plot after repair...")
    repair_plot = save_topology_plots(model, out_dir, prefix="post_repair")
    post_repair_topology_profile = collect_topology_profile(model)
    repair_vs_baseline = topology_closeness_score(post_repair_topology_profile, baseline_topology_profile)
    if repair_plot["matrix_path"] is not None:
        print(f"  saved: {repair_plot['matrix_path']}")
    print(f"  saved: {repair_plot['circle_path']}")
    print(f"  topology closeness(post_repair vs baseline_natural) = {repair_vs_baseline['score']:.4f}")

    print("\n[7/7] Trainability verification...")
    n_g = min(int(args.verify_val_samples), len(x_val))
    x_grad = tf.constant(x_val[:n_g], dtype=tf.float32)
    y_grad = tf.constant(y_val[:n_g], dtype=tf.int32)

    low_budget_auto_stable = float(args.target_ebops) <= 600.0
    verify_freeze_quantizers = bool(args.verify_freeze_quantizers)
    verify_enable_budget_control = bool(
        args.verify_enable_budget_control or (low_budget_auto_stable and (not verify_freeze_quantizers))
    )
    verify_lr_use = float(args.verify_lr)
    verify_grad_clip = None
    verify_freeze_batchnorm = False
    if low_budget_auto_stable:
        verify_lr_use = float(args.verify_lr)
        verify_grad_clip = 1.0
        verify_freeze_batchnorm = False
    print(
        "  verify_train_policy: "
        f"enable_budget_control={verify_enable_budget_control}, "
        f"freeze_quantizers={verify_freeze_quantizers}, "
        f"lr={verify_lr_use:.1e}, "
        f"grad_clip={verify_grad_clip}, "
        f"freeze_batchnorm={verify_freeze_batchnorm}"
    )

    probe_before = gradient_probe(model, x_grad, y_grad, one_step_lr=max(float(verify_lr_use), 1e-4))
    val_loss_before, val_acc_before = quick_eval(model, x_grad, y_grad)
    print(
        f"  probe_before: grad_norm={probe_before['grad_global_norm']:.3e}, "
        f"near_zero={probe_before['grad_near_zero_ratio']:.4f}, "
        f"one_step_drop={probe_before['one_step_loss_drop']:.4e}, "
        f"trainable_proxy={probe_before['trainable_proxy']}"
    )

    n_tr = min(int(args.verify_train_samples), len(x_train))
    n_v = min(int(args.verify_val_samples), len(x_val))
    smoke = run_smoke_train(
        model=model,
        x_train=tf.constant(x_train[:n_tr], dtype=tf.float32),
        y_train=tf.constant(y_train[:n_tr], dtype=tf.int32),
        x_val=tf.constant(x_val[:n_v], dtype=tf.float32),
        y_val=tf.constant(y_val[:n_v], dtype=tf.int32),
        target_ebops=float(args.target_ebops),
        epochs=int(args.verify_epochs),
        steps_per_epoch=int(args.verify_steps_per_epoch),
        batch_size=int(args.verify_batch_size),
        learning_rate=float(verify_lr_use),
        output_dir=out_dir,
        min_steps_per_epoch=int(args.min_steps_per_epoch_longrun),
        enable_budget_control=bool(verify_enable_budget_control),
        freeze_quantizers=bool(verify_freeze_quantizers),
        grad_clip_norm=verify_grad_clip,
        freeze_batchnorm=bool(verify_freeze_batchnorm),
    )

    val_loss_after, val_acc_after = quick_eval(model, x_grad, y_grad)
    probe_after = gradient_probe(model, x_grad, y_grad, one_step_lr=max(float(verify_lr_use), 1e-4))

    repaired_model = out_dir / "model_post_repair.keras"
    repaired_weights = out_dir / "model_post_repair.weights.h5"
    model.save(repaired_model)
    model.save_weights(repaired_weights)

    report = {
        "checkpoint": args.checkpoint,
        "natural_baseline_plot_ckpt": natural_ckpt,
        "natural_baseline_plot_ebops_hint": natural_ebops_hint,
        "target_ebops": float(args.target_ebops),
        "prune_method": args.prune_method,
        "strict_design_constraints": {
            "no_baseline_topology_used_for_pruning_decision": True,
            "topology_theory_driven": True,
            "quantizer_deadzone_theory_driven": True,
        },
        "baseline_ebops_measured": float(baseline_ebops),
        "post_prune_ebops_measured": float(post_prune_ebops),
        "post_repair_ebops_measured": float(repaired_ebops),
        "theory_prune_report": theory_prune_report,
        "deadzone_before": deadzone_before,
        "deadzone_after": deadzone_after,
        "connectivity_fix_rounds": connectivity_fix_rounds,
        "connectivity_final": connectivity_final,
        "teacher_calibration": teacher_calib_report,
        "activation_wakeup_report": wakeup_report,
        "output_symmetry_report": output_symmetry_report,
        "dead_layer_rescue_report": dead_layer_rescue_report,
        "fused_deadzone_repair_report": fused_deadzone_repair_report,
        "verify_train_policy": {
            "auto_low_budget_stable": bool(low_budget_auto_stable),
            "enable_budget_control": bool(verify_enable_budget_control),
            "freeze_quantizers": bool(verify_freeze_quantizers),
            "verify_lr": float(verify_lr_use),
            "grad_clip_norm": (None if verify_grad_clip is None else float(verify_grad_clip)),
            "freeze_batchnorm": bool(verify_freeze_batchnorm),
        },
        "repair_report": repair_report,
        "topology_profiles": {
            "baseline_natural": baseline_topology_profile,
            "post_prune": post_prune_topology_profile,
            "post_repair": post_repair_topology_profile,
        },
        "topology_closeness": {
            "post_prune_vs_baseline_natural": prune_vs_baseline,
            "post_repair_vs_baseline_natural": repair_vs_baseline,
        },
        "probe_before": probe_before,
        "probe_after": probe_after,
        "val_before": {"loss": float(val_loss_before), "acc": float(val_acc_before)},
        "val_after": {"loss": float(val_loss_after), "acc": float(val_acc_after)},
        "smoke_train": smoke,
        "topology_plots": {
            "baseline_natural": baseline_plot,
            "post_prune": prune_plot,
            "post_repair": repair_plot,
        },
        "saved_model": str(repaired_model),
        "saved_weights": str(repaired_weights),
        "saved_post_prune_model": str(post_prune_model),
        "saved_post_prune_weights": str(post_prune_weights),
    }

    report_path = out_dir / "repair_verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Done.")
    print(f"  repaired_model  : {repaired_model}")
    print(f"  repaired_weights: {repaired_weights}")
    print(f"  report          : {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
