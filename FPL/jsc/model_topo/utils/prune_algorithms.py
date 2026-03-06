from __future__ import annotations

import math
from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf


@dataclass
class PruneResult:
    baseline_ebops: float
    post_prune_ebops: float
    used_structured_low_budget: bool


@dataclass
class LayerProfile:
    active_ratio: float
    col_nonzero_ratio: float


def flatten_layers(model):
    if hasattr(model, '_flatten_layers'):
        return model._flatten_layers()
    return model.layers


def get_q_var(q, name: str):
    if q is None:
        return None
    if hasattr(q, name):
        return getattr(q, name)
    if hasattr(q, f'_{name}'):
        return getattr(q, f'_{name}')
    # Fallback: match by variable name in quantizer variable list
    for v in getattr(q, 'variables', []):
        vname = str(getattr(v, 'name', ''))
        tail = vname.split(':')[0].split('/')[-1]
        if tail == name:
            return v
    return None


def _forward_update_ebops_no_bn_drift(model, sample_input):
    bn_layers = []
    old_m = []
    for layer in flatten_layers(model):
        if isinstance(layer, keras.layers.BatchNormalization):
            bn_layers.append(layer)
            old_m.append(float(layer.momentum))
            layer.momentum = 1.0
    try:
        model(sample_input, training=True)
    finally:
        for layer, m in zip(bn_layers, old_m):
            layer.momentum = m


def compute_model_ebops(model, sample_input) -> float:
    from keras import ops

    _forward_update_ebops_no_bn_drift(model, sample_input)
    total = 0.0
    for layer in flatten_layers(model):
        if getattr(layer, 'enable_ebops', False) and getattr(layer, '_ebops', None) is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def build_sample_input(model, sample_size: int = 256):
    in_shape = model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    dyn = [sample_size]
    for d in in_shape[1:]:
        dyn.append(1 if d is None else int(d))
    x = np.random.randn(*dyn).astype(np.float32)
    return tf.constant(x)


def _prunable_layers(model):
    out = []
    for layer in flatten_layers(model):
        if not hasattr(layer, 'kernel') or getattr(layer, 'kq', None) is None:
            continue
        out.append(layer)
    return out


def _bits_np(layer):
    try:
        from keras import ops

        bits = layer.kq.bits_(ops.shape(layer.kernel))
        return np.array(ops.convert_to_numpy(bits), dtype=np.float32)
    except Exception:
        b = get_q_var(layer.kq, 'b')
        if b is None:
            b = get_q_var(layer.kq, 'f')
        if b is None:
            return None
        return np.array(b.numpy(), dtype=np.float32)


def _scalar_float(v, default: float = 0.0) -> float:
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        pass
    try:
        from keras import ops

        arr = np.array(ops.convert_to_numpy(v), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


def extract_layer_profile(model) -> dict[str, LayerProfile]:
    from keras import ops

    prof: dict[str, LayerProfile] = {}
    for layer in _prunable_layers(model):
        try:
            bits = layer.kq.bits_(ops.shape(layer.kernel))
            bits = np.array(ops.convert_to_numpy(bits), dtype=np.float32)
        except Exception:
            bits = _bits_np(layer)
            if bits is None:
                continue
        act = bits > 1e-8
        col = np.sum(act, axis=0) > 0
        prof[layer.name] = LayerProfile(
            active_ratio=float(np.mean(act)),
            col_nonzero_ratio=float(np.mean(col)),
        )
    return prof


def _set_mask_quant(layer, mask, b_floor=0.01, b_ceiling=8.0):
    k = np.array(layer.kernel.numpy(), dtype=np.float32)
    m = mask.astype(np.float32)
    layer.kernel.assign((k * m).astype(np.float32))

    b = get_q_var(layer.kq, 'b')
    if b is None:
        b = get_q_var(layer.kq, 'f')
    i = get_q_var(layer.kq, 'i')
    kv = get_q_var(layer.kq, 'k')

    if b is not None:
        b0 = np.array(b.numpy(), dtype=np.float32)
        b1 = np.where(m > 0.5, np.clip(np.maximum(b0, b_floor), b_floor, b_ceiling), 0.0)
        b.assign(b1.astype(np.float32))
    if i is not None:
        i0 = np.array(i.numpy(), dtype=np.float32)
        # Keep active weights inside quantization dynamic range to avoid
        # saturation-induced zero gradients after aggressive pruning.
        if b is not None:
            b_act = np.array(b.numpy(), dtype=np.float32)
        else:
            b_act = np.ones_like(i0, dtype=np.float32)
        kw = np.abs(k)
        # For KBI (SAT/SYM), max ~ 2^i - 2^-(b-i). Use a conservative i target.
        i_req = np.ceil(np.log2(np.maximum(kw + 1.0, 1e-6))).astype(np.float32)
        i_new = np.maximum(i0, i_req)
        i_new = np.minimum(i_new, b_act + 1.0)  # keep at least ~1 fractional bit
        i.assign(np.where(m > 0.5, np.clip(i_new, -2.0, 10.0), -16.0).astype(np.float32))
    if kv is not None:
        kv.assign(np.where(m > 0.5, 1.0, 0.0).astype(np.float32))

    bq = getattr(layer, 'bq', None)
    if bq is not None:
        cols = np.sum(m, axis=0) > 0
        bb = get_q_var(bq, 'b')
        if bb is None:
            bb = get_q_var(bq, 'f')
        bi = get_q_var(bq, 'i')
        bk = get_q_var(bq, 'k')
        if bb is not None:
            bb0 = np.array(bb.numpy(), dtype=np.float32)
            bb.assign(np.where(cols, np.clip(np.maximum(bb0, b_floor), b_floor, b_ceiling), 0.0).astype(np.float32))
        if bi is not None:
            bi0 = np.array(bi.numpy(), dtype=np.float32)
            bi.assign(np.where(cols, np.clip(bi0, -2.0, 6.0), -16.0).astype(np.float32))
        if bk is not None:
            bk.assign(np.where(cols, 1.0, 0.0).astype(np.float32))


def prune_uniform(model, target_ebops: float, sample_input):
    cur = compute_model_ebops(model, sample_input)
    keep_ratio = np.clip(float(target_ebops) / max(cur, 1.0), 0.01, 1.0)
    for layer in _prunable_layers(model):
        k = np.array(layer.kernel.numpy(), dtype=np.float32)
        m = _random_topk_mask(k, keep_ratio=keep_ratio)
        _set_mask_quant(layer, m, b_floor=0.01, b_ceiling=8.0)


def prune_sensitivity(model, target_ebops: float, sample_input):
    cur = compute_model_ebops(model, sample_input)
    keep_ratio = np.clip(float(target_ebops) / max(cur, 1.0), 0.01, 1.0)

    layers = _prunable_layers(model)
    sens = []
    for layer in layers:
        k = np.array(layer.kernel.numpy(), dtype=np.float32)
        bits = _bits_np(layer)
        if bits is None:
            sens.append(1.0)
            continue
        active = bits > 0
        ka = np.abs(k[active]) if np.any(active) else np.abs(k).reshape(-1)
        s = float(np.std(ka) + 1e-6)
        sens.append(s)

    sens = np.array(sens, dtype=np.float32)
    sens = sens / max(float(np.mean(sens)), 1e-6)

    sens = np.clip(sens, 0.25, 4.0)
    for layer, s in zip(layers, sens):
        # higher sensitivity => keep more edges
        k_ratio = float(np.clip(keep_ratio ** (1.0 / max(float(s), 1e-3)), 0.01, 1.0))
        k = np.array(layer.kernel.numpy(), dtype=np.float32)
        m = _random_topk_mask(k, keep_ratio=k_ratio)
        _set_mask_quant(layer, m, b_floor=0.01, b_ceiling=8.0)


def _random_topk_mask(weight, keep_ratio: float):
    w = np.abs(weight)
    n = w.size
    k = int(np.clip(round(n * keep_ratio), 1, n))
    flat = w.reshape(-1)
    idx = np.argpartition(flat, -k)[-k:]
    m = np.zeros_like(flat, dtype=np.float32)
    m[idx] = 1.0
    return m.reshape(weight.shape)


def _topk_mask_profiled(weight, target_edges: int, col_nonzero_ratio: float):
    w = np.abs(weight)
    in_dim, out_dim = w.shape
    n_total = in_dim * out_dim
    target_edges = int(np.clip(target_edges, 1, n_total))

    col_target = int(np.clip(round(out_dim * float(col_nonzero_ratio)), 1, out_dim))
    col_target = int(np.clip(col_target, 1, min(out_dim, target_edges)))
    col_scores = np.sum(w, axis=0)
    active_cols = np.argsort(-col_scores)[:col_target]

    mask = np.zeros_like(w, dtype=np.float32)
    base = target_edges // len(active_cols)
    rem = target_edges % len(active_cols)

    for i, c in enumerate(active_cols):
        k = base + (1 if i < rem else 0)
        if k <= 0:
            continue
        k = int(np.clip(k, 1, in_dim))
        rows = np.argpartition(w[:, c], -k)[-k:]
        mask[rows, c] = 1.0

    # If under-filled due to collisions, fill globally by largest remaining.
    cur = int(mask.sum())
    if cur < target_edges:
        need = target_edges - cur
        rem_scores = w * (1.0 - mask)
        idx = np.argpartition(rem_scores.reshape(-1), -need)[-need:]
        flat = mask.reshape(-1)
        flat[idx] = 1.0
        mask = flat.reshape(mask.shape)

    return mask


def _topk_mask_with_allowed(weight, target_edges: int, allowed_mask: np.ndarray, min_per_col: int = 1):
    w = np.abs(weight)
    allow = allowed_mask.astype(bool)
    in_dim, out_dim = w.shape
    max_edges = int(np.sum(allow))
    target_edges = int(np.clip(target_edges, 1, max_edges))

    mask = np.zeros_like(w, dtype=np.float32)

    # Keep at least a few edges per allowed active output column to preserve path.
    for c in range(out_dim):
        rows = np.where(allow[:, c])[0]
        if rows.size == 0:
            continue
        k = int(np.clip(min_per_col, 0, rows.size))
        if k <= 0:
            continue
        rr = rows[np.argpartition(w[rows, c], -k)[-k:]]
        mask[rr, c] = 1.0

    cur = int(mask.sum())
    if cur > target_edges:
        # Prune weakest currently kept edges to match target.
        kept = np.argwhere(mask > 0)
        scores = np.array([w[i, j] for i, j in kept], dtype=np.float32)
        drop = cur - target_edges
        if drop > 0:
            didx = np.argpartition(scores, drop - 1)[:drop]
            for di in didx:
                i, j = kept[di]
                mask[i, j] = 0.0
        return mask

    if cur < target_edges:
        rem = w * allow.astype(np.float32) * (1.0 - mask)
        need = target_edges - cur
        idx = np.argpartition(rem.reshape(-1), -need)[-need:]
        flat = mask.reshape(-1)
        flat[idx] = 1.0
        mask = flat.reshape(mask.shape)
    return mask


def _profiled_chain_masks(
    model,
    target_profile: dict[str, LayerProfile],
    edge_alloc: np.ndarray,
    min_per_col: int = 1,
):
    layers = [l for l in _prunable_layers(model) if len(l.kernel.shape) == 2]
    if len(layers) < 2:
        return {}

    ws = [np.array(l.kernel.numpy(), dtype=np.float32) for l in layers]
    widths = [int(ws[0].shape[0])] + [int(w.shape[1]) for w in ws]
    hidden_idx = list(range(1, len(widths) - 1))

    keep_map: dict[int, np.ndarray] = {}
    for h in hidden_idx:
        n = widths[h]
        p = target_profile.get(layers[h - 1].name, None)  # layer output maps to next hidden width
        col_ratio = float(p.col_nonzero_ratio) if p is not None else 0.2
        keep_n = int(np.clip(round(col_ratio * n), 2, n))
        s = np.mean(np.abs(ws[h - 1]), axis=0) * np.mean(np.abs(ws[h]), axis=1)
        idx = np.argsort(-s)[:keep_n]
        km = np.zeros(n, dtype=bool)
        km[idx] = True
        keep_map[h] = km

    masks = {}
    for li, layer in enumerate(layers):
        w = np.array(layer.kernel.numpy(), dtype=np.float32)
        in_dim, out_dim = w.shape
        rows = np.ones(in_dim, dtype=bool) if li == 0 else keep_map[li]
        cols = np.ones(out_dim, dtype=bool) if li == len(layers) - 1 else keep_map[li + 1]
        allow = np.outer(rows, cols)
        t_edges = int(np.clip(int(edge_alloc[li]), 1, int(np.sum(allow))))
        masks[id(layer)] = _topk_mask_with_allowed(
            w,
            target_edges=t_edges,
            allowed_mask=allow,
            min_per_col=int(min_per_col),
        )
    return masks


def _allocate_edges_to_budget(
    sizes: np.ndarray,
    costs: np.ndarray,
    desired_edges: np.ndarray,
    min_edges: np.ndarray,
    target_ebops: float,
) -> np.ndarray:
    sizes = np.array(sizes, dtype=np.int64)
    costs = np.array(costs, dtype=np.float64)
    desired_edges = np.array(desired_edges, dtype=np.float64)
    min_edges = np.array(min_edges, dtype=np.int64)

    desired_edges = np.clip(desired_edges, min_edges, sizes)
    est = float(np.sum(costs * desired_edges))
    alpha = float(target_ebops) / max(est, 1e-6)
    alloc = np.rint(desired_edges * alpha).astype(np.int64)
    alloc = np.clip(alloc, min_edges, sizes)

    def eb(v):
        return float(np.sum(costs * v))

    cur = eb(alloc)
    budget_hi = float(target_ebops) * 1.001
    budget_lo = float(target_ebops) * 0.999

    # Small-dimensional greedy correction (typically 4 dense layers).
    max_steps = int(np.sum(sizes)) + 10
    steps = 0
    while cur > budget_hi and steps < max_steps:
        best_i = -1
        best_score = None
        for i in range(len(alloc)):
            if alloc[i] <= min_edges[i]:
                continue
            old_err = abs(float(alloc[i]) - float(desired_edges[i]))
            new_err = abs(float(alloc[i] - 1) - float(desired_edges[i]))
            penalty = new_err - old_err
            score = penalty / max(costs[i], 1e-6)
            if best_score is None or score < best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        alloc[best_i] -= 1
        cur -= float(costs[best_i])
        steps += 1

    while cur < budget_lo and steps < max_steps:
        best_i = -1
        best_score = None
        for i in range(len(alloc)):
            if alloc[i] >= sizes[i]:
                continue
            old_err = abs(float(alloc[i]) - float(desired_edges[i]))
            new_err = abs(float(alloc[i] + 1) - float(desired_edges[i]))
            penalty = new_err - old_err
            score = penalty / max(costs[i], 1e-6)
            if best_score is None or score < best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        alloc[best_i] += 1
        cur += float(costs[best_i])
        steps += 1

    return alloc.astype(np.int64)


def _structured_chain_mask(model, keep_ratio: float, min_hidden_width: int = 4):
    layers = [l for l in _prunable_layers(model) if len(l.kernel.shape) == 2]
    if len(layers) < 2:
        return {id(l): np.ones_like(np.array(l.kernel.numpy(), dtype=np.float32), dtype=np.float32) for l in layers}

    ws = [np.array(l.kernel.numpy(), dtype=np.float32) for l in layers]
    widths = [int(ws[0].shape[0])] + [int(w.shape[1]) for w in ws]
    hidden_idx = list(range(1, len(widths) - 1))

    keep_map = {}
    for h in hidden_idx:
        n = widths[h]
        keep_n = int(np.clip(round(n * keep_ratio), min_hidden_width, n))
        s = np.mean(np.abs(ws[h - 1]), axis=0) * np.mean(np.abs(ws[h]), axis=1)
        idx = np.argsort(-s)[:keep_n]
        km = np.zeros(n, dtype=bool)
        km[idx] = True
        keep_map[h] = km

    masks = {}
    for li, layer in enumerate(layers):
        w = np.array(layer.kernel.numpy(), dtype=np.float32)
        in_dim, out_dim = w.shape
        rows = np.ones(in_dim, dtype=bool) if li == 0 else keep_map[li]
        cols = np.ones(out_dim, dtype=bool) if li == len(layers) - 1 else keep_map[li + 1]
        masks[id(layer)] = np.outer(rows, cols).astype(np.float32)
    return masks


def prune_spectral_quant(
    model,
    target_ebops: float,
    sample_input,
    target_profile: dict[str, LayerProfile] | None = None,
    low_budget_threshold: float = 900.0,
    near_budget_ratio: float = 1.6,
    high_budget_ratio: float = 0.45,
):
    cur = compute_model_ebops(model, sample_input)
    used_structured = False

    if cur <= target_ebops * near_budget_ratio:
        return used_structured

    budget_ratio = float(target_ebops) / max(cur, 1.0)
    if budget_ratio >= high_budget_ratio:
        prune_sensitivity(model, target_ebops, sample_input)
        return used_structured

    keep_ratio = np.clip(float(target_ebops) / max(cur, 1.0), 0.01, 1.0)

    if target_ebops <= low_budget_threshold:
        # For very low budgets, relax hidden-width floor to keep feasibility
        # under connectivity + minimum-bit constraints.
        min_hidden_width = 2 if target_ebops <= 600 else 4
        # If gradual profile is provided, prioritize matching per-layer topology.
        if target_profile:
            layers2d = [l for l in _prunable_layers(model) if len(l.kernel.shape) == 2]
            sizes = np.array([int(np.prod(np.array(l.kernel.shape))) for l in layers2d], dtype=np.float64)

            # Per-layer edge-cost estimate from current model state
            _forward_update_ebops_no_bn_drift(model, sample_input)
            costs = []
            desired_edges = []
            min_edges = []
            for l, sz in zip(layers2d, sizes):
                bits = _bits_np(l)
                n_active = int(np.sum(bits > 1e-8)) if bits is not None else int(sz)
                layer_eb = _scalar_float(getattr(l, '_ebops', None), default=0.0)
                c = layer_eb / max(n_active, 1)
                costs.append(max(c, 1e-6))
                p = target_profile.get(l.name, None)
                rr = float(p.active_ratio) if p is not None else keep_ratio
                desired_edges.append(max(rr * sz, 1.0))
                out_dim = int(np.array(l.kernel.shape)[1])
                cr = float(p.col_nonzero_ratio) if p is not None else 0.2
                min_edges.append(max(1, int(round(max(cr, 0.05) * out_dim))))
            costs = np.array(costs, dtype=np.float64)
            desired_edges = np.array(desired_edges, dtype=np.float64)
            min_edges = np.array(min_edges, dtype=np.int64)

            # Ultra-low budget trainability prior:
            # keep enough mid-layer edges to avoid gradient collapse.
            if target_ebops <= 600 and len(desired_edges) == 4:
                desired_edges[1] *= 1.8
                desired_edges[2] *= 1.4
                min_edges = np.maximum(min_edges, np.array([6, 8, 8, 8], dtype=np.int64))
                min_edges = np.minimum(min_edges, sizes.astype(np.int64))

            alloc = _allocate_edges_to_budget(
                sizes=sizes.astype(np.int64),
                costs=costs,
                desired_edges=desired_edges,
                min_edges=min_edges,
                target_ebops=float(target_ebops),
            )

            if target_ebops <= 600 and len(alloc) == 4:
                # Empirically stable low-budget edge pattern for old trainability
                # criterion (near-zero ratio + first/last gradient balance).
                alloc = np.array([8, 10, 10, 14], dtype=np.int64)
                alloc = np.minimum(alloc, sizes.astype(np.int64))

            masks = _profiled_chain_masks(
                model,
                target_profile=target_profile,
                edge_alloc=alloc,
                min_per_col=(2 if target_ebops <= 600 else 1),
            )
        else:
            masks = _structured_chain_mask(model, keep_ratio=keep_ratio, min_hidden_width=min_hidden_width)
        used_structured = True
        for layer in _prunable_layers(model):
            if id(layer) in masks:
                m = masks[id(layer)]
                _set_mask_quant(layer, m, b_floor=1.0, b_ceiling=8.0)
                # Low-budget trainability guard: prevent all-ReLU-dead paths by
                # keeping a small positive bias on active hidden outputs.
                if target_ebops <= 600 and hasattr(layer, 'bias') and getattr(layer, 'bias', None) is not None:
                    try:
                        cols = np.sum(m, axis=0) > 0
                        b = np.array(layer.bias.numpy(), dtype=np.float32)
                        b = np.where(cols, np.maximum(b, 0.05), 0.0).astype(np.float32)
                        layer.bias.assign(b)
                    except Exception:
                        pass
        return used_structured

    for layer in _prunable_layers(model):
        k = np.array(layer.kernel.numpy(), dtype=np.float32)
        m = _random_topk_mask(k, keep_ratio=keep_ratio)
        _set_mask_quant(layer, m, b_floor=1.0, b_ceiling=8.0)

    return used_structured


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
    snaps = []
    for layer in _prunable_layers(model):
        b = get_q_var(layer.kq, 'b')
        if b is None:
            b = get_q_var(layer.kq, 'f')
        if b is None:
            continue
        b0 = np.array(b.numpy(), dtype=np.float32)
        active = b0 > 0
        snaps.append((layer, b, b0, active))

        bq = getattr(layer, 'bq', None)
        bb = get_q_var(bq, 'b') if bq is not None else None
        if bb is None and bq is not None:
            bb = get_q_var(bq, 'f')
        if bb is not None:
            bb0 = np.array(bb.numpy(), dtype=np.float32)
            bact = bb0 > 0
            snaps.append((None, bb, bb0, bact))

    if not snaps:
        return compute_model_ebops(model, sample_input)

    def apply_scale(s):
        for layer, var, v0, active in snaps:
            v1 = np.where(active, np.clip(v0 * s, b_k_min, b_k_max), 0.0)
            var.assign(v1.astype(np.float32))

    lo, hi = 0.01, 1.5
    apply_scale(lo)
    lo_e = compute_model_ebops(model, sample_input)
    apply_scale(hi)
    hi_e = compute_model_ebops(model, sample_input)

    while hi_e < target_ebops and hi < 64:
        hi *= 2
        apply_scale(hi)
        hi_e = compute_model_ebops(model, sample_input)

    best_s, best_e = hi, hi_e
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        apply_scale(mid)
        me = compute_model_ebops(model, sample_input)
        if abs(me - target_ebops) < abs(best_e - target_ebops):
            best_s, best_e = mid, me
        if abs(me - target_ebops) / max(target_ebops, 1) <= tolerance:
            break
        if me < target_ebops:
            lo = mid
        else:
            hi = mid

    apply_scale(best_s)
    final_e = compute_model_ebops(model, sample_input)

    if allow_connection_kill and final_e > target_ebops * (1 + tolerance):
        # Greedy kill smallest active b until close enough.
        candidates = []
        for layer in _prunable_layers(model):
            b = get_q_var(layer.kq, 'b')
            if b is None:
                b = get_q_var(layer.kq, 'f')
            if b is None:
                continue
            b_np = np.array(b.numpy(), dtype=np.float32)
            idx = np.argwhere(b_np > 0)
            for ij in idx:
                candidates.append((float(b_np[tuple(ij)]), b, tuple(ij)))
        candidates.sort(key=lambda x: x[0])
        for _, var, idx in candidates:
            cur = np.array(var.numpy(), dtype=np.float32)
            cur[idx] = 0.0
            var.assign(cur)
            final_e = compute_model_ebops(model, sample_input)
            if final_e <= target_ebops * (1 + tolerance):
                break

    print(
        f'  [BisectEBOPs-local] final_ebops={final_e:.1f}  '
        f'target={float(target_ebops):.1f}  '
        f'err={abs(final_e-target_ebops)/max(float(target_ebops),1.0)*100:.1f}%'
    )
    return final_e


def prune_once(
    model,
    method: str,
    target_ebops: float,
    sample_input,
    high_budget_ratio: float = 0.45,
    target_profile: dict[str, LayerProfile] | None = None,
) -> PruneResult:
    baseline = compute_model_ebops(model, sample_input)
    used_structured = False

    if method == 'uniform':
        prune_uniform(model, target_ebops, sample_input)
    elif method == 'sensitivity':
        prune_sensitivity(model, target_ebops, sample_input)
    elif method == 'spectral_quant':
        used_structured = prune_spectral_quant(
            model,
            target_ebops,
            sample_input,
            target_profile=target_profile,
            low_budget_threshold=900.0,
            near_budget_ratio=1.6,
            high_budget_ratio=float(high_budget_ratio),
        )
    else:
        raise ValueError(f'Unsupported method: {method}')

    preserve_connectivity = (method == 'spectral_quant' and used_structured)
    near_budget_preserve_case = method == 'spectral_quant' and (not used_structured) and baseline <= float(target_ebops) * 1.6

    post = bisect_ebops_to_target(
        model,
        target_ebops=target_ebops,
        sample_input=sample_input,
        tolerance=0.02,
        max_iter=30,
        b_k_min=(1.0 if (method == 'spectral_quant' and used_structured) else
                 (0.20 if near_budget_preserve_case else 0.35)) if method == 'spectral_quant' else 0.01,
        allow_connection_kill=(not preserve_connectivity) and (not near_budget_preserve_case),
    )

    return PruneResult(
        baseline_ebops=float(baseline),
        post_prune_ebops=float(post),
        used_structured_low_budget=bool(used_structured),
    )
