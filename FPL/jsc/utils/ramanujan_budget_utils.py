# ramanujan_budget_utils.py
"""
BW-aware Ramanujan 初始化与恒 EBOPS 约束工具。

主要 API
--------
compute_bw_aware_degree(model, target_ebops, b_a_init, b_k_init, ...)
    联合求解每层的 (d_l, b_k_l)，使得初始化时 Σ EBOPS ≈ target_ebops。
    度 d_l 由拉玛努金谱条件（sqrt(N_in) * multiplier）决定（保证谱间隙）；
    位宽 b_k_l 由 EBOPS 预算分配反推。

apply_ramanujan_bw_init(model, ...)
    在 apply_ramanujan_init 基础上同时写入 per-layer 初始位宽到 kq.b / kq.i。

EBOPsConstantProjector (Callback)
    Phase-2 使用：每 epoch 结束后将全部 kq.b 等比缩放，使总 EBOPS 保持在
    target_ebops 附近，从而在恒计算预算下纯粹最大化精度。

BetaZeroCallback (Callback)
    Phase-2 使用：强制将所有 HGQ 层的 _beta 置为 0，关闭 EBOPS 惩罚。
"""

import math
from typing import Iterable

import numpy as np
import tensorflow as tf
import keras


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _get_kq_var(kq, name: str):
    """Robustly fetch a quantizer variable by logical name.

    In HGQ/Brevitas-style quantizers, variables are typically named like:
      ".../b:0", ".../i:0", ".../f:0"
    Comparing with raw variable.name (which includes scopes and ":0") fails.
    This helper matches by the last path component.

    It also supports the common pattern where the variable is exposed directly
    as an attribute, e.g. kq.b / kq.i.
    """
    # 1) Direct attribute (best case)
    if hasattr(kq, name):
        v = getattr(kq, name)
        if isinstance(v, tf.Variable):
            return v

    # 2) Search by suffix in kq.variables
    cand = []
    for v in getattr(kq, "variables", []):
        leaf = v.name.split("/")[-1]          # e.g. "b:0"
        leaf = leaf.split(":")[0]            # e.g. "b"
        if leaf == name:
            cand.append(v)
    if len(cand) == 1:
        return cand[0]
    if len(cand) > 1:
        cand.sort(key=lambda x: len(x.name))
        return cand[0]

    # 3) Fallback: contains match (rare scopes)
    for v in getattr(kq, "variables", []):
        if ("/" + name + ":") in v.name:
            return v
    return None


def _flatten_layers(model: keras.Model) -> Iterable[keras.layers.Layer]:
    """Iterate all layers (including nested) robustly across Keras versions."""
    if hasattr(model, "_flatten_layers"):
        try:
            return list(model._flatten_layers(include_self=False, recursive=True))
        except TypeError:
            return list(model._flatten_layers())
    out = []
    stack = list(getattr(model, "layers", []))
    while stack:
        l = stack.pop(0)
        out.append(l)
        stack.extend(getattr(l, "layers", []))
    return out


def _ramanujan_like_mask(shape, degree, rng):
    """生成 d-regular 式稀疏掩膜（1=保留, 0=剪掉）。"""
    if len(shape) == 2:
        in_dim, out_dim = int(shape[0]), int(shape[1])
        degree = min(degree, in_dim)
        mask = np.zeros((in_dim, out_dim), dtype=np.float32)
        for o in range(out_dim):
            idx = rng.choice(in_dim, size=degree, replace=False)
            mask[idx, o] = 1.0
        return mask
    elif len(shape) == 4:
        kh, kw, in_ch, out_ch = [int(x) for x in shape]
        degree = min(degree, in_ch)
        base = np.zeros((in_ch, out_ch), dtype=np.float32)
        for o in range(out_ch):
            idx = rng.choice(in_ch, size=degree, replace=False)
            base[idx, o] = 1.0
        return np.broadcast_to(base[None, None], (kh, kw, in_ch, out_ch)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported kernel shape: {shape}")


# ---------------------------------------------------------------------------
# compute_bw_aware_degree
# ---------------------------------------------------------------------------

def compute_bw_aware_degree(
    model: keras.Model,
    target_ebops: float,
    b_a_init: float = 3.0,
    b_k_min: float = 1.0,
    b_k_max: float = 8.0,
    multiplier: float = 1.5,
    min_degree: int = 4,
    budget_weight: str = 'capacity',   # 'capacity' | 'uniform'
    verbose: bool = True,
) -> tuple[dict, dict]:
    """给定目标 EBOPS，联合求解每层的拉玛努金度和初始 kernel 位宽。

    算法
    ----
    1. 对每个 HGQ 层用谱条件计算 d_l = clamp(round(sqrt(N_in_l) * multiplier), min_degree, N_in_l)
    2. 按 budget_weight 分配各层 EBOPS 预算 E_l^∗
       - 'capacity'  : E_l^∗ ∝ N_in_l * N_out_l  （容量大的层获得更多预算）
       - 'uniform'   : E_l^∗ = target / n_layers  （等分）
    3. 反推每层 kernel 位宽  b_k_l = E_l^∗ / (d_l * N_out_l * b_a_init)
       并 clamp 到 [b_k_min, b_k_max]
    4. 打印各层实际预期 EBOPS 及总和（与 target 对比），便于调试。

    返回
    ----
    per_layer_degree : dict {layer_name: d_l}
    per_layer_bk     : dict {layer_name: b_k_l}   (初始 fractional bits)
    """
    # (layer, N_in_eff, N_out, d_l, conn_mul)
    # Dense: conn_mul=1
    # Conv2D: our mask selects channels and broadcasts across spatial kernel
    #         positions, so active connections per output = d_l * (kh*kw)
    layers_info = []

    for layer in _flatten_layers(model):
        if getattr(layer, 'kq', None) is None or not hasattr(layer, 'kernel'):
            continue

        shape = layer.kernel.shape
        if len(shape) == 2:
            in_dim, out_dim = int(shape[0]), int(shape[1])
            conn_mul = 1
            Nin_eff = in_dim
            Nout = out_dim
            deg_base = in_dim
        elif len(shape) == 4:
            kh, kw, in_ch, out_ch = [int(x) for x in shape]
            conn_mul = kh * kw
            Nin_eff = in_ch * conn_mul
            Nout = out_ch
            deg_base = in_ch
        else:
            continue

        d_l = int(round(math.sqrt(deg_base) * multiplier))
        d_l = max(d_l, min_degree)
        d_l = min(d_l, deg_base)
        layers_info.append((layer, Nin_eff, Nout, d_l, conn_mul))

    if not layers_info:
        raise ValueError("No HGQ layers found in model.")

    # 分配预算权重
    if budget_weight == 'capacity':
        weights = [N_in_eff * N_out for _, N_in_eff, N_out, _, _ in layers_info]
    elif budget_weight == 'uniform':
        weights = [1.0] * len(layers_info)
    else:
        raise ValueError(f"Unknown budget_weight='{budget_weight}'")

    total_weight = sum(weights)

    per_layer_degree = {}
    per_layer_bk     = {}
    actual_ebops_sum = 0.0

    if verbose:
        print(f"\n[compute_bw_aware_degree] target_ebops={target_ebops:.2f}, "
              f"b_a_init={b_a_init}, budget_weight={budget_weight}")
        print(f"{'layer':20s}  {'N_in_eff':>8}  {'N_out':>5}  {'d_l':>4}  "
              f"{'conn_mul':>8}  {'sparsity':>8}  {'b_k_l':>6}  {'E_l_pred':>12}")

    for (layer, N_in_eff, N_out, d_l, conn_mul), w in zip(layers_info, weights):
        E_l = target_ebops * (w / total_weight)
        b_k_l = E_l / max((d_l * conn_mul) * N_out * b_a_init, 1e-9)
        b_k_l = float(np.clip(b_k_l, b_k_min, b_k_max))

        # 实际预期 EBOPS（受 clamp 影响）
        E_l_actual = (d_l * conn_mul) * N_out * b_k_l * b_a_init
        actual_ebops_sum += E_l_actual

        per_layer_degree[layer.name] = d_l
        per_layer_bk[layer.name]     = b_k_l

        if verbose:
            if len(layer.kernel.shape) == 2:
                deg_base = int(layer.kernel.shape[0])
            else:
                deg_base = int(layer.kernel.shape[2])
            sparsity = 1.0 - (d_l / max(deg_base, 1))
            print(f"  {layer.name:20s}  {N_in_eff:8d}  {N_out:5d}  {d_l:4d}  "
                  f"{conn_mul:8d}  {sparsity:7.1%}  {b_k_l:6.3f}  {E_l_actual:12.1f}")

    if verbose:
        print(f"  {'TOTAL':20s}  {'':>5}  {'':>5}  {'':>4}  {'':>8}  {'':>6}  "
              f"{actual_ebops_sum:12.1f}  (target={target_ebops:.1f}, "
              f"ratio={actual_ebops_sum/target_ebops:.3f})\n")

    return per_layer_degree, per_layer_bk


# ---------------------------------------------------------------------------
# apply_ramanujan_bw_init
# ---------------------------------------------------------------------------

def apply_ramanujan_bw_init(
    model: keras.Model,
    per_layer_degree: dict,
    per_layer_bk: dict,
    seed: int = 42,
    pruned_frac_bits: float = 0.0,
    pruned_int_bits:  float = 0.0,
    active_int_bits:  float = 1.0,    # 保留连接的整数位宽（符号 + 整数）
    also_zero_kernel: bool = True,
    verbose: bool = True,
):
    """BW-aware 拉玛努金稀疏初始化。

    相比 apply_ramanujan_init，本函数额外将保留连接的 kq.b 设置为
    per_layer_bk 中给出的位宽，而不是沿用全局统一初始值。
    这样初始化后全模型的预期 EBOPS ≈ target_ebops（由 compute_bw_aware_degree 保证）。

    逻辑:
      kq.b[i,o]:  被剪连接 → pruned_frac_bits;  保留连接 → per_layer_bk[layer_name]
      kq.i[i,o]:  被剪连接 → pruned_int_bits;   保留连接 → active_int_bits
      kernel[i,o]: 被剪连接可选清零（also_zero_kernel=True）
    """
    rng = np.random.RandomState(seed)

    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None or not hasattr(layer, 'kernel'):
            continue

        kernel_var = layer.kernel
        shape = kernel_var.shape

        d_l   = per_layer_degree.get(layer.name)
        b_k_l = per_layer_bk.get(layer.name)
        if d_l is None:
            continue   # 不在计划内的层跳过

        mask   = _ramanujan_like_mask(shape, d_l, rng)   # 1=保留, 0=剪掉
        pruned = 1.0 - mask

        # ---- kq.b （小数位宽）：保留 → b_k_l，剪掉 → pruned_frac_bits --------
        # HGQ quantizers sometimes use 'b' or 'f' to represent fractional bits.
        b_var = _get_kq_var(kq, 'b')
        f_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            b_new = mask * b_k_l + pruned * pruned_frac_bits
            b_var.assign(b_new.astype(np.float32))
        elif f_var is not None:
            f_new = mask * b_k_l + pruned * pruned_frac_bits
            f_var.assign(f_new.astype(np.float32))

        # ---- kq.i （整数位宽）：保留 → active_int_bits，剪掉 → pruned_int_bits
        i_var = _get_kq_var(kq, 'i')
        if i_var is not None:
            i_new = mask * active_int_bits + pruned * pruned_int_bits
            i_var.assign(i_new.astype(np.float32))

        # ---- 可选：清零浮点 kernel ------------------------------------------------
        if also_zero_kernel:
            kernel_var.assign(kernel_var.numpy() * mask)

        # 将 mask 挂在 layer 上，供 RamanujanMaskEnforcer 使用
        layer.ramanujan_mask = tf.constant(mask, dtype=tf.float32)

        if verbose:
            active = int(mask.sum())
            total  = int(mask.size)
            print(f"[RamanujanBWInit] {layer.name:20s}  shape={list(shape)}  "
                  f"d={d_l}  b_k={b_k_l:.3f}  "
                  f"active={active}/{total}  sparsity={1-mask.mean():.3f}")


# ---------------------------------------------------------------------------
# 3D Ramanujan Allocation — per-connection bitwidth optimization
# ---------------------------------------------------------------------------

def compute_3d_ramanujan_allocation(
    model: keras.Model,
    target_ebops: float,
    b_a_init: float = 3.0,
    b_k_viable: float = 1.0,
    b_k_max: float = 6.0,
    init_bk: float | None = None,
    base_multiplier: float = 2.0,
    min_per_layer: int = 5,
    budget_weight: str = 'capacity',
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """3D Ramanujan graph init: per-connection bitwidth allocation under EBOPS budget.

    核心思路
    --------
    标准 Ramanujan 初始化用 d-regular 拓扑 + 均匀位宽，在极低 EBOPS 预算下
    所有连接的位宽都太低（< 1 bit, ~0.1）导致 STE 梯度全是噪声、无法训练。

    本方法将预算 **集中** 到少量高重要性连接上，使每条存活连接都有足够位宽
    (≥ b_k_viable) 进行有效训练，同时放弃低价值连接。

    "三维" 指联合优化三个维度：
      1. 深度维（层）  : 保证每层至少 min_per_layer 条活跃连接（梯度流贯通）
      2. 宽度维（神经元）: 在层内按 |kernel| 重要性分数选择哪些连接存活
      3. 精度维（位宽）  : 存活连接按重要性非均匀分配 b_k ∈ [b_k_viable, b_k_max]

    两阶段策略（init_bk 不为 None 时）
    ----------------------------------
    **拓扑选择** 仍基于 target_ebops 预算（决定存活连接数量），但实际写入的
    b_k = init_bk（通常 = 模型原始位宽，如 3.0），使初始 EBOPS > target。
    训练时由 BetaOnlyBudgetController 自然压缩到 target_ebops。
    这样保证：
      - 初始拓扑来自 Ramanujan 谱优选（连接质量）
      - 初始位宽足够高，Glorot 权重不会被量化为全 0（可训练性）
      - 渐进压缩沿用 baseline 经验证的路径（梯度自然分配位宽）

    参数
    ----
    b_k_viable       : 拓扑选择时的计算位宽（决定存活数量），建议 ≥ 1.0
    b_k_max          : 最高位宽上限（仅在 init_bk=None 时用于非均匀分配）
    init_bk          : 存活连接的实际初始 b_k。
                       None → 用预算优化的非均匀位宽（可能很低）。
                       float → 所有存活连接统一设为此值（推荐 = 模型原始 f0）。
    base_multiplier  : 基础 Ramanujan 度 = round(sqrt(N_in) * mult)
    min_per_layer    : 每层至少保留的连接数
    budget_weight    : 'capacity' (按 N_in*N_out 加权) | 'uniform'

    返回
    ----
    per_layer_bk_map : dict {layer_name: ndarray[float32] 形状同 kernel}
        每个元素 = 该连接的 b_k（0 = 被剪, > 0 = 存活）
    """
    rng = np.random.RandomState(seed)

    # ── Phase 1: 收集层信息 + 生成基础 Ramanujan 掩膜 ────────────────────
    layers_info = []
    for layer in _flatten_layers(model):
        if getattr(layer, 'kq', None) is None or not hasattr(layer, 'kernel'):
            continue
        shape = layer.kernel.shape
        kernel_np = layer.kernel.numpy()

        if len(shape) == 2:
            in_dim, out_dim = int(shape[0]), int(shape[1])
            conn_mul = 1
            deg_base = in_dim
        elif len(shape) == 4:
            kh, kw, in_ch, out_ch = [int(x) for x in shape]
            conn_mul = kh * kw
            in_dim, out_dim = in_ch, out_ch
            deg_base = in_ch
        else:
            continue

        d_l = int(round(math.sqrt(deg_base) * base_multiplier))
        d_l = max(d_l, min(2, deg_base))
        d_l = min(d_l, deg_base)

        mask = _ramanujan_like_mask(shape, d_l, rng)
        layers_info.append({
            'name': layer.name, 'layer': layer, 'shape': shape,
            'in_dim': in_dim, 'out_dim': out_dim,
            'd_l': d_l, 'conn_mul': conn_mul,
            'mask': mask, 'kernel_np': kernel_np,
        })

    if not layers_info:
        raise ValueError("No HGQ layers with kq found in model.")

    # ── Phase 2: 按 |kernel| 为候选连接打分 ─────────────────────────────
    for info in layers_info:
        k = info['kernel_np']
        m = info['mask']
        if len(info['shape']) == 4:
            # Conv: 聚合空间维 → (in_ch, out_ch) 分数
            scores = (np.abs(k).sum(axis=(0, 1))
                      * (m.sum(axis=(0, 1)) > 0).astype(np.float32))
        else:
            scores = np.abs(k) * m
        info['scores_2d'] = scores        # (in_dim, out_dim)

    # ── Phase 3: 按层分配 EBOPS 预算（保底 + 容量加权）──────────────────
    if budget_weight == 'capacity':
        caps = [info['in_dim'] * info['out_dim'] for info in layers_info]
    else:
        caps = [1.0] * len(layers_info)
    total_cap = sum(caps)

    floor_budget = min_per_layer * b_k_viable * b_a_init
    total_floor  = floor_budget * len(layers_info)
    if total_floor > target_ebops * 0.95:
        min_per_layer = max(1, int(
            target_ebops * 0.95 / (len(layers_info) * b_k_viable * b_a_init)))
        floor_budget = min_per_layer * b_k_viable * b_a_init
        total_floor  = floor_budget * len(layers_info)

    distributable = target_ebops - total_floor
    budgets = [floor_budget + distributable * (c / total_cap) for c in caps]

    # ── Phase 4: 逐层剪枝 + 非均匀位宽分配 ──────────────────────────────
    per_layer_bk_map = {}
    actual_total = 0.0

    if verbose:
        print(f"\n[3D-Ramanujan] target_ebops={target_ebops:.1f}, "
              f"b_k_viable={b_k_viable}, b_k_max={b_k_max}, "
              f"b_a_init={b_a_init}")
        print(f"  {'layer':12s}  {'budget':>8s}  {'d_base':>6s}  "
              f"{'n_cand':>6s}  {'n_surv':>6s}  "
              f"{'mean_bk':>8s}  {'max_bk':>7s}  {'E_est':>8s}")

    for info, budget_l in zip(layers_info, budgets):
        layer  = info['layer']
        shape  = info['shape']
        mask   = info['mask']
        scores = info['scores_2d']
        cm     = info['conn_mul']
        in_dim, out_dim = info['in_dim'], info['out_dim']

        # 活跃候选（channel 级别）
        if len(shape) == 4:
            ch_mask = (mask.sum(axis=(0, 1)) > 0).astype(np.float32)
        else:
            ch_mask = (mask > 0).astype(np.float32)
        active_pos = np.argwhere(ch_mask > 0)   # [(i, j), ...]
        n_active = len(active_pos)

        # 预算内可存活的最大连接数
        ebops_per_conn = b_k_viable * b_a_init * cm
        n_max = max(min_per_layer, int(budget_l / ebops_per_conn))

        # 按分数排序，保留 top n_max
        if n_active <= n_max:
            surv_pos = active_pos
        else:
            pos_sc  = np.array([scores[i, j] for i, j in active_pos])
            top_idx = np.argsort(-pos_sc)[:n_max]
            surv_pos = active_pos[top_idx]
        n_survive = len(surv_pos)

        # ── 位宽分配：base = b_k_viable，富余按重要性加权 ────────────
        surv_sc = np.array([scores[i, j] for i, j in surv_pos])
        sc_sum  = surv_sc.sum()
        sc_frac = (surv_sc / sc_sum if sc_sum > 0
                   else np.ones(n_survive) / max(n_survive, 1))

        if init_bk is not None:
            # 两阶段策略：拓扑由预算决定，位宽固定为 init_bk
            bk_arr = np.full(n_survive, float(init_bk), dtype=np.float32)
        else:
            # 原始策略：位宽也由预算约束
            bk_arr    = np.full(n_survive, b_k_viable, dtype=np.float32)
            base_cost = n_survive * ebops_per_conn
            extra     = max(0.0, budget_l - base_cost)
            if extra > 0 and n_survive > 0:
                bk_arr = bk_arr + extra * sc_frac / (b_a_init * cm)
                bk_arr = np.clip(bk_arr, b_k_viable, b_k_max).astype(np.float32)

        # 构建全形状 b_k 图
        if len(shape) == 2:
            bk_map = np.zeros(shape, dtype=np.float32)
            for idx, (i, j) in enumerate(surv_pos):
                bk_map[i, j] = bk_arr[idx]
        else:   # 4-D Conv
            bk_base = np.zeros((in_dim, out_dim), dtype=np.float32)
            for idx, (i, j) in enumerate(surv_pos):
                bk_base[i, j] = bk_arr[idx]
            bk_map = np.broadcast_to(
                bk_base[np.newaxis, np.newaxis], shape
            ).copy().astype(np.float32)

        ebops_est = float(bk_arr.sum()) * b_a_init * cm
        actual_total += ebops_est
        per_layer_bk_map[layer.name] = bk_map

        if verbose:
            m_bk = float(bk_arr.mean()) if n_survive else 0.0
            x_bk = float(bk_arr.max())  if n_survive else 0.0
            print(f"  {layer.name:12s}  {budget_l:8.1f}  "
                  f"{info['d_l']:6d}  {n_active:6d}  {n_survive:6d}  "
                  f"{m_bk:8.3f}  {x_bk:7.3f}  {ebops_est:8.1f}")

    if verbose:
        print(f"  {'TOTAL':12s}  {target_ebops:8.1f}  {'':6s}  {'':6s}  "
              f"{'':6s}  {'':8s}  {'':7s}  {actual_total:8.1f}")
        if init_bk is not None:
            print(f"  init_bk={init_bk} (initial EBOPS={actual_total:.0f}, "
                  f"target={target_ebops:.0f}, beta controller will compress)")
        print(f"  ratio = {actual_total / target_ebops:.3f}\n")

    return per_layer_bk_map


def apply_3d_ramanujan_init(
    model: keras.Model,
    per_layer_bk_map: dict[str, np.ndarray],
    active_int_bits: float = 1.0,
    pruned_int_bits: float = 0.0,
    pruned_frac_bits: float = 0.0,
    also_zero_kernel: bool = True,
    rescale_kernel: bool = True,
    verbose: bool = True,
):
    """将 compute_3d_ramanujan_allocation 的 b_k map 写入模型。

    对每个层：
      kq.b[i,j] = bk_map[i,j]           (0 → 被剪, ≥ b_k_viable → 存活)
      kq.i[i,j] = active/pruned_int_bits (按存活/剪除区分)
      kernel[i,j] *= (bk_map > 0)        (清零被剪连接的浮点 kernel)
      layer.ramanujan_mask = (bk_map > 0) (供 RamanujanMaskEnforcer 使用)

    rescale_kernel : 若存活连接的 kernel stddev 相对于量化步长过小
        (stddev < step * 0.5)，适当放大使权重不会全被量化为 0。
        这是从头训练+稀疏+低位宽场景的关键保障。
    """
    for layer in _flatten_layers(model):
        bk_map = per_layer_bk_map.get(layer.name)
        if bk_map is None:
            continue

        kq = getattr(layer, 'kq', None)
        if kq is None:
            continue

        kernel_var = layer.kernel
        mask   = (bk_map > 0).astype(np.float32)
        pruned = 1.0 - mask

        # ── kq.b / kq.f: 写入逐连接位宽 ────────────────────────────────
        b_var = _get_kq_var(kq, 'b')
        f_var = _get_kq_var(kq, 'f')
        target_var = b_var if b_var is not None else f_var
        if target_var is not None:
            target_var.assign(
                (bk_map + pruned * pruned_frac_bits).astype(np.float32))

        # ── kq.i: 整数位宽 ──────────────────────────────────────────────
        i_var = _get_kq_var(kq, 'i')
        if i_var is not None:
            i_new = mask * active_int_bits + pruned * pruned_int_bits
            i_var.assign(i_new.astype(np.float32))

        # ── 清零被剪 kernel ──────────────────────────────────────────────
        if also_zero_kernel:
            kernel_var.assign(kernel_var.numpy() * mask)

        # ── 量化感知 kernel 重缩放 ──────────────────────────────────────
        if rescale_kernel:
            active_bk = bk_map[bk_map > 0]
            if len(active_bk) > 0:
                mean_step = float(2.0 ** (-active_bk.mean()))
                k_np = kernel_var.numpy()
                active_vals = k_np[mask > 0]
                if len(active_vals) > 0:
                    current_std = float(np.std(active_vals))
                    # 若 kernel stddev < 1.5 倍量化步长，放大以覆盖多个量化级
                    target_std = mean_step * 3.0
                    if current_std < mean_step * 1.5 and current_std > 1e-9:
                        scale = target_std / current_std
                        k_np = k_np * mask * scale  # 只缩放存活连接
                        kernel_var.assign(k_np.astype(np.float32))
                        if verbose:
                            print(f"    [rescale] {layer.name}: "
                                  f"std {current_std:.4f} → {target_std:.4f} "
                                  f"(×{scale:.2f}, step={mean_step:.4f})")

        # ── 挂载 mask 给 RamanujanMaskEnforcer ──────────────────────────
        layer.ramanujan_mask = tf.constant(mask, dtype=tf.float32)

        if verbose:
            n_active = int(mask.sum())
            n_total  = int(mask.size)
            active_bk = bk_map[bk_map > 0]
            if len(active_bk) > 0:
                print(f"  [3D-RamInit] {layer.name:12s}  "
                      f"shape={list(kernel_var.shape)}  "
                      f"active={n_active}/{n_total} "
                      f"({n_active / n_total:.1%})  "
                      f"bk: mean={active_bk.mean():.3f}  "
                      f"min={active_bk.min():.3f}  "
                      f"max={active_bk.max():.3f}")
            else:
                print(f"  [3D-RamInit] {layer.name:12s}  ALL PRUNED")


# ---------------------------------------------------------------------------
# EBOPsConstantProjector  — Phase-2 恒 EBOPS callback
# ---------------------------------------------------------------------------

class EBOPsConstantProjector(keras.callbacks.Callback):
    """Phase-2 Callback：每 epoch 结束后将全部 kq.b 等比缩放，
    使总 EBOPS 维持在 target_ebops 附近。

    原理
    ----
    EBOPS ≈ Σ_l Σ_{(i,o)∈mask} kq.b[i,o] * b_a[i]
    在 b_a 不变的假设下，将全部 kq.b 乘以 α = target_ebops / current_ebops
    可以等比压缩/扩张到目标预算：
        new_kq.b[i,o] = α * kq.b[i,o]        (保留连接)
        new_kq.b[i,o] = 0                     (剪掉连接，不改变)

    被剪掉的连接（值 ≤ pruned_threshold）不参与缩放，避免将 0 放大。
    缩放后 kq.b 被 clamp 到 [b_k_min, b_k_max]，避免极端值。

    振荡抑制
    --------
    裸投影 α = target/current 会引发正反馈振荡：
      投影压低 kq.b → gradient（β=0）把位宽推高 → EBOPS 蹿高 →
      下一 epoch α 极小再次压低 → 死循环。

    三层阻尼机制：
    1. 幂次阻尼 (alpha_gamma)：
         α = (target/current)^gamma，gamma < 1 时只做部分修正。
         gamma=0.3 → 每 epoch 修正 30% 的超差量，约 10 epoch 收敛。
    2. alpha 限幅 (alpha_min/alpha_max)：
         clip(α, alpha_min, alpha_max)，防止单步大幅跳动。
    3. EMA 平滑 (ema_alpha)：
         对 current_ebops 做指数移动平均，减少噪声驱动的过调。
         ema_alpha=0.3 → 以 0.3:0.7 权重混合新旧 EBOPS 估计。

    参数
    ----
    target_ebops     : 目标 EBOPS
    b_k_min          : kq.b 下限（默认 0.5）
    b_k_max          : kq.b 上限（默认 8.0）
    pruned_threshold : kq.b ≤ 该阈值的连接视为剪掉，不参与缩放（默认 0.1）
    start_epoch      : 开始投影的 epoch
    alpha_gamma      : 幂次阻尼指数，越小越保守（默认 0.3）
    alpha_min        : alpha 下限（默认 0.85）
    alpha_max        : alpha 上限（默认 1.15）
    ema_alpha        : EBOPS EMA 平滑系数（默认 0.3，即新值权重 30%）
    log_scale        : 是否打印每 epoch 的缩放信息（默认 False）
    """

    def __init__(
        self,
        target_ebops: float,
        b_k_min:          float = 0.5,
        b_k_max:          float = 8.0,
        pruned_threshold: float = 0.1,
        start_epoch:      int   = 0,
        alpha_gamma:      float = 0.3,
        alpha_min:        float = 0.85,
        alpha_max:        float = 1.15,
        ema_alpha:        float = 0.3,
        project_activation: bool = True,
        log_scale:        bool  = False,
    ):
        super().__init__()
        self.target_ebops     = float(target_ebops)
        self.b_k_min          = b_k_min
        self.b_k_max          = b_k_max
        self.pruned_threshold = pruned_threshold
        self.start_epoch      = start_epoch
        self.alpha_gamma      = alpha_gamma
        self.alpha_min        = alpha_min
        self.alpha_max        = alpha_max
        self.ema_alpha        = ema_alpha
        self.project_activation = project_activation
        self.log_scale        = log_scale
        self._ebops_ema       = None   # EMA 状态

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return

        logs = logs or {}
        raw_ebops = logs.get('ebops', None)
        if raw_ebops is None:
            raw_ebops = logs.get('val_ebops', float('nan'))
        raw_ebops = float(raw_ebops)
        if not math.isfinite(raw_ebops) or raw_ebops <= 0:
            return

        # EMA 平滑（首个 epoch 直接用原始值初始化）
        if self._ebops_ema is None:
            self._ebops_ema = raw_ebops
        else:
            self._ebops_ema = (self.ema_alpha * raw_ebops
                               + (1.0 - self.ema_alpha) * self._ebops_ema)
        current_ebops = self._ebops_ema

        # 幂次阻尼投影 + 限幅
        raw_alpha = self.target_ebops / current_ebops
        alpha = raw_alpha ** self.alpha_gamma
        alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))

        for layer in self.model.layers:
            kq = getattr(layer, 'kq', None)
            if kq is None:
                continue

            b_var = _get_kq_var(kq, 'b')
            f_var = _get_kq_var(kq, 'f')
            target_var = b_var if b_var is not None else f_var
            if target_var is None:
                continue

            b_arr = target_var.numpy()
            # 只缩放"保留连接"（b > pruned_threshold），被剪连接不动
            active_mask = (b_arr > self.pruned_threshold).astype(np.float32)
            b_new = np.where(
                active_mask > 0,
                np.clip(b_arr * alpha, self.b_k_min, self.b_k_max),
                b_arr,
            )
            target_var.assign(b_new.astype(np.float32))

            if self.project_activation:
                aq = getattr(layer, 'aq', None)
                if aq is not None:
                    ab_var = _get_kq_var(aq, 'b')
                    af_var = _get_kq_var(aq, 'f')
                    a_target = ab_var if ab_var is not None else af_var
                    if a_target is not None:
                        a_arr = a_target.numpy()
                        # activation 没有“剪枝连接”的概念：全部视为 active
                        a_new = np.clip(a_arr * alpha, self.b_k_min, self.b_k_max)
                        a_target.assign(a_new.astype(np.float32))

        if self.log_scale:
            print(f"[EBOPsConstantProjector] epoch={epoch}  "
                  f"raw={raw_ebops:.1f}  ema={current_ebops:.1f}  "
                  f"target={self.target_ebops:.1f}  "
                  f"raw_alpha={raw_alpha:.4f}  alpha={alpha:.4f}")


# ---------------------------------------------------------------------------
# BetaZeroCallback  — Phase-2 关闭 EBOPS 惩罚
# ---------------------------------------------------------------------------

class BetaZeroCallback(keras.callbacks.Callback):
    """每个 epoch 开始时将所有 HGQ 层的 _beta 强制设置为 0，
    等效于完全关闭 EBOPS 惩罚项，使优化器专注于提升精度。

    需排在 BetaScheduler & EBOPsAdaptiveBeta 之后，以覆盖它们的赋值。

    参数
    ----
    start_epoch : 开始置零的 epoch（= phase1_epochs）
    """

    def __init__(self, start_epoch: int = 0):
        super().__init__()
        self.start_epoch = start_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        for layer in _flatten_layers(self.model):
            if hasattr(layer, '_beta'):
                try:
                    layer._beta.assign(tf.zeros_like(layer._beta))
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 一次性剪枝基线：HighBitPruner
# ---------------------------------------------------------------------------


class ActivationBitsFixer(keras.callbacks.Callback):
    """在训练过程中将所有层的 activation quantizer 位宽固定为常数。

    适用场景
    --------
    在 HGQ 中，activation 位宽也可能是可训练变量。若你将 _beta 置 0，
    activation 位宽会被梯度推高；若你只投影 kernel 位宽去维持 EBOPS，
    会导致 kernel 位宽被压到下限，从而出现量化过硬 / dead-STE / acc 不涨。

    本 callback 在每个 epoch 开始时把 aq.b / aq.f 重新设为常数，从而：
      1) EBOPS 预算主要由 kernel 位宽承担（投影更稳定）
      2) 训练早期避免 activation 位宽漂移导致的“预算挤压”

    参数
    ----
    b_a_fixed  : 固定的 activation fractional bits（例如 3.0）
    start_epoch: 从哪个 epoch 开始固定
    """

    def __init__(self, b_a_fixed: float = 3.0, start_epoch: int = 0):
        super().__init__()
        self.b_a_fixed = float(b_a_fixed)
        self.start_epoch = int(start_epoch)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        for layer in _flatten_layers(self.model):
            aq = getattr(layer, 'aq', None)
            if aq is None:
                continue
            b_var = _get_kq_var(aq, 'b')
            f_var = _get_kq_var(aq, 'f')
            target_var = b_var if b_var is not None else f_var
            if target_var is None:
                continue
            arr = target_var.numpy()
            target_var.assign((np.ones_like(arr, dtype=np.float32) * self.b_a_fixed))


class HighBitPruner:
    """一次性剪枝基线（对比实验用）。

    流程：先在高位宽下训练收敛（或直接用已有 checkpoint），
    然后调用 prune_to_ebops() 将全模型 kq.b 按百分位阈值剪枝，
    使总 EBOPS 降至 target_ebops。

    与 BW-aware Ramanujan init 的对比：
      - 本类在已有知识的基础上剪枝，理论上保留了更多已学习的表示。
      - Ramanujan init 从头开始，受益于谱性质的梯度流保证。
    """

    def __init__(self, target_ebops: float, pruned_threshold: float = 0.1):
        self.target_ebops     = float(target_ebops)
        self.pruned_threshold = pruned_threshold

    def _collect_active_bk(self, model):
        """收集所有活跃连接的 kq.b 值。"""
        all_b = []
        for layer in _flatten_layers(model):
            kq = getattr(layer, 'kq', None)
            if kq is None:
                continue
            b_var = _get_kq_var(kq, 'b')
            if b_var is not None:
                b_arr = b_var.numpy().ravel()
                all_b.extend(b_arr[b_arr > self.pruned_threshold].tolist())
        return np.array(all_b, dtype=np.float32)

    def prune_to_ebops(self, model, current_ebops: float, verbose: bool = True):
        """将全模型 kq.b 等比缩放至 target_ebops 对应的位宽（一次性，不投影）。

        参数
        ----
        current_ebops : 当前实际 EBOPS（从 FreeEBOPs 或 logs 读取）

        注意
        ----
        只缩放 kq.b（小数位宽），**不**缩放 kq.i（整数位宽）。
        kq.i 控制量化器的动态范围，缩放它会导致大量激活/权重被截断，
        是 finetune 精度无法恢复的重要原因之一。
        """
        if current_ebops <= 0:
            raise ValueError("current_ebops must be > 0")

        alpha = self.target_ebops / current_ebops

        for layer in model.layers:
            kq = getattr(layer, 'kq', None)
            if kq is None:
                continue
            b_var = _get_kq_var(kq, 'b')
            f_var = _get_kq_var(kq, 'f')
            target_var = b_var if b_var is not None else f_var
            if target_var is None:
                continue
            b_arr = target_var.numpy()
            active_mask = (b_arr > self.pruned_threshold).astype(np.float32)
            # b_k_min 不能设太高（如 0.5），否则大比例剪枝时 clamp 会阻止 kq.b 降到足够低
            # 例如 target=300, current=19899 → alpha=0.015 → 3.0*0.015=0.045 → clip(0.5)=0.5 远超预期
            b_new = np.where(active_mask > 0, np.clip(b_arr * alpha, 0.01, 8.0), 0.0)
            target_var.assign(b_new.astype(np.float32))

            # 注意：不缩放 kq.i（整数位宽） —— 它控制动态范围，不是计算代价

        if verbose:
            print(f"[HighBitPruner] current_ebops={current_ebops:.1f}  "
                  f"target={self.target_ebops:.1f}  alpha={alpha:.4f}")


class SensitivityAwarePruner:
    """基于每层 EBOPS 贡献敏感度的剪枝器。

    改进 HighBitPruner 的两个问题：
    1. 不缩放 kq.i（整数位宽）
    2. 按每层 EBOPS 贡献比例分配预算，而不是均匀缩放

    原理
    ----
    对每层 l 计算其当前 EBOPs 贡献 E_l。
    目标 EBOPs 总量按比例分配到各层：E_l_target = E_l * (target / current)
    但位宽较低的层少剪（保护精度敏感层），位宽较高的层多剪。
    具体做法：alpha_l = (target / current) ^ (mean_bk_l / global_mean_bk)
    位宽高于均值的层 alpha 更小（剪更多），反之更大（保护）。

    参数
    ----
    target_ebops     : 目标 EBOPS
    pruned_threshold : kq.b ≤ 该阈值视为已剪
    protect_power    : 保护指数（0=均匀，1=完全按位宽比例，默认 0.5）
    b_k_min/max      : 剪枝后 kq.b 的范围
    """

    def __init__(
        self,
        target_ebops: float,
        pruned_threshold: float = 0.1,
        protect_power: float = 0.5,
        b_k_min: float = 0.01,
        b_k_max: float = 8.0,
    ):
        self.target_ebops     = float(target_ebops)
        self.pruned_threshold = pruned_threshold
        self.protect_power    = protect_power
        self.b_k_min          = b_k_min
        self.b_k_max          = b_k_max

    def prune_to_ebops(self, model, current_ebops: float, verbose: bool = True):
        if current_ebops <= 0:
            raise ValueError("current_ebops must be > 0")

        global_alpha = self.target_ebops / current_ebops

        # Step 1: 收集每层的平均活跃位宽
        layer_info = []
        all_active_bk = []
        for layer in _flatten_layers(model):
            kq = getattr(layer, 'kq', None)
            if kq is None:
                continue
            b_var = _get_kq_var(kq, 'b')
            f_var = _get_kq_var(kq, 'f')
            target_var = b_var if b_var is not None else f_var
            if target_var is None:
                continue
            b_arr = target_var.numpy()
            active = b_arr[b_arr > self.pruned_threshold]
            mean_bk = float(active.mean()) if len(active) > 0 else 0.0
            layer_info.append((layer, target_var, b_arr, mean_bk))
            all_active_bk.extend(active.tolist())

        if not all_active_bk:
            return

        global_mean_bk = float(np.mean(all_active_bk))

        # Step 2: 按层敏感度分配 per-layer alpha
        for layer, target_var, b_arr, mean_bk in layer_info:
            if global_mean_bk > 0 and mean_bk > 0:
                # 位宽高于均值的层 → ratio > 1 → alpha_l < global_alpha（多剪）
                # 位宽低于均值的层 → ratio < 1 → alpha_l > global_alpha（少剪）
                ratio = mean_bk / global_mean_bk
                alpha_l = global_alpha ** (ratio ** self.protect_power)
            else:
                alpha_l = global_alpha

            active_mask = (b_arr > self.pruned_threshold).astype(np.float32)
            b_new = np.where(
                active_mask > 0,
                np.clip(b_arr * alpha_l, self.b_k_min, self.b_k_max),
                0.0,
            )
            target_var.assign(b_new.astype(np.float32))

            if verbose:
                print(f"  [SensitivityPruner] {layer.name:20s}  "
                      f"mean_bk={mean_bk:.3f}  alpha_l={alpha_l:.4f}")

        if verbose:
            print(f"[SensitivityPruner] current={current_ebops:.1f}  "
                  f"target={self.target_ebops:.1f}  global_alpha={global_alpha:.4f}")


class BetaOnlyBudgetController(keras.callbacks.Callback):
    """用 beta 自然维持 EBOPs 预算，不做均匀投影。

    核心思路
    --------
    Baseline 在连续训练中靠 beta（EBOPS 惩罚系数）+ 梯度自然达到目标 EBOPs，
    同时保持最优的 per-layer 位宽分配。

    本 callback 复制这个机制：
    - warmup 阶段（epoch < warmup_epochs）不介入，让 BetaScheduler 控制
    - warmup 结束后，从 BetaScheduler 当前值接管
    - 当 EBOPs > target * (1 + margin) 时，自适应增大 beta 施加压缩力
    - 当 EBOPs < target * (1 - margin) 时，减小 beta 释放压力
    - 当 EBOPs 在容忍带内时，保持 beta 不变

    与 EBOPsConstantProjector 的区别
    ------
    - 投影器直接修改 kq.b，用同一个 alpha 缩放所有层 → 破坏跨层分配
    - 本 callback 只调整 beta 标量 → HGQ 梯度自行决定各层怎么分配位宽

    参数
    ----
    target_ebops    : 目标 EBOPS
    warmup_epochs   : warmup 期间不介入（让 BetaScheduler 控制），之后再开始调控
    margin          : 容忍带宽度（默认 0.15 = ±15%）
    beta_min        : beta 下限
    beta_max        : beta 上限
    adjust_factor   : 基础乘法调整因子（默认 1.15）
    max_step_factor : 单步最大乘法因子上限（默认 1.5），防止 beta 暴涨
    ema_alpha       : EBOPS EMA 平滑系数
    """

    def __init__(
        self,
        target_ebops: float,
        warmup_epochs: int = 0,
        margin: float = 0.15,
        beta_min: float = 1e-8,
        beta_max: float = 1e-3,
        adjust_factor: float = 1.15,
        max_step_factor: float = 1.5,
        ema_alpha: float = 0.3,
    ):
        super().__init__()
        self.target_ebops    = float(target_ebops)
        self.warmup_epochs   = warmup_epochs
        self.margin          = margin
        self.beta_current    = None  # 将从 BetaScheduler 值接管
        self.beta_min        = beta_min
        self.beta_max        = beta_max
        self.adjust_factor   = adjust_factor
        self.max_step_factor = max_step_factor
        self.ema_alpha       = ema_alpha
        self._ebops_ema      = None

    def _read_beta(self):
        """从模型当前 _beta 变量读取 beta 值。"""
        for layer in _flatten_layers(self.model):
            if hasattr(layer, '_beta'):
                try:
                    return float(layer._beta.numpy().flat[0])
                except Exception:
                    pass
        return 1e-6

    def _set_beta(self, value):
        bv = tf.constant(value, dtype=tf.float32)
        for layer in _flatten_layers(self.model):
            if hasattr(layer, '_beta'):
                try:
                    layer._beta.assign(tf.ones_like(layer._beta) * bv)
                except Exception:
                    pass

    def on_epoch_begin(self, epoch, logs=None):
        # warmup 阶段不介入，让 BetaScheduler 控制 beta
        if epoch < self.warmup_epochs:
            return
        # 首次接管：从 BetaScheduler 已设置的值初始化
        if self.beta_current is None:
            self.beta_current = self._read_beta()
        self._set_beta(self.beta_current)

    def on_epoch_end(self, epoch, logs=None):
        # warmup 阶段不调整
        if epoch < self.warmup_epochs:
            return
        logs = logs or {}
        raw_ebops = float(logs.get('ebops', float('nan')))
        if not math.isfinite(raw_ebops) or raw_ebops <= 0:
            return

        # 首次接管时，也读取当前 beta 以防 on_epoch_begin 没执行
        if self.beta_current is None:
            self.beta_current = self._read_beta()

        if self._ebops_ema is None:
            self._ebops_ema = raw_ebops
        else:
            self._ebops_ema = (self.ema_alpha * raw_ebops
                               + (1.0 - self.ema_alpha) * self._ebops_ema)

        upper = self.target_ebops * (1.0 + self.margin)
        lower = self.target_ebops * (1.0 - self.margin)

        if self._ebops_ema > upper:
            # EBOPs 超标 → 增大 beta，按超额比例加速，但限制单步幅度
            overshoot = self._ebops_ema / upper
            factor = min(self.adjust_factor * overshoot, self.max_step_factor)
            self.beta_current = min(
                self.beta_current * factor,
                self.beta_max,
            )
        elif self._ebops_ema < lower:
            # EBOPs 不足 → 减小 beta 释放压力，同样限制单步幅度
            undershoot = lower / max(self._ebops_ema, 1.0)
            factor = min(self.adjust_factor * undershoot, self.max_step_factor)
            self.beta_current = max(
                self.beta_current / factor,
                self.beta_min,
            )
        # else: 在容忍带内，保持不变

        logs['beta_budget'] = self.beta_current


# ---------------------------------------------------------------------------
# 结构化剪枝：从 baseline 高精度 checkpoint 剪到 target EBOPS
# ---------------------------------------------------------------------------

def structured_prune_for_budget(
    model: keras.Model,
    target_ebops: float,
    b_a: float = 3.0,
    init_bk: float = 3.0,
    min_per_layer: int = 5,
    budget_weight: str = 'capacity',
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """从已训练模型中选择最重要的连接，保留高位宽，其余清零。

    与均匀缩放剪枝（SensitivityAwarePruner）的关键区别
    ------------------------------------------------
    均匀缩放：所有连接 b_k *= α → 全部变成 ~0.04 → STE 梯度全死
    结构化剪枝：选 ~100 条最重要的连接保留 b_k=3.0，其余 b_k=0 → 可训练

    重要性评分
    ----------
    每条连接的重要性 = |kernel[i,j]| × b_k[i,j]
    这衡量了该连接在当前模型中的信息传递能力：
      - |kernel| 大 → 权重本身重要
      - b_k 大 → 量化精度高，传递的信息保真度高
    baseline 训练过程中，梯度已经自然地将重要性集中到少量连接上，
    这个分数恰好捕捉了这一分布。

    参数
    ----
    model        : 已训练的 HGQ 模型（含 kq.b 和 kernel）
    target_ebops : 目标 EBOPS 预算
    b_a          : 激活量化位宽（用于计算 EBOPS = b_k × b_a）
    init_bk      : 存活连接的初始 b_k（通常 = 模型训练时的 f0）
    min_per_layer: 每层至少保留的连接数（保证梯度流贯通）
    budget_weight: 'capacity'（按 N_in × N_out 加权）| 'uniform'

    返回
    ----
    per_layer_bk_map : dict {layer_name: ndarray[float32] 同 kernel 形状}
    """
    # ── 收集层信息 ─────────────────────────────────────────────────────
    layers_info = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None or not hasattr(layer, 'kernel'):
            continue
        shape = layer.kernel.shape
        kernel_np = layer.kernel.numpy()

        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        bk_np = b_var.numpy() if b_var is not None else np.ones(shape, dtype=np.float32)

        if len(shape) == 2:
            in_dim, out_dim = int(shape[0]), int(shape[1])
            conn_mul = 1
        elif len(shape) == 4:
            kh, kw, in_ch, out_ch = [int(x) for x in shape]
            conn_mul = kh * kw
            in_dim, out_dim = in_ch, out_ch
        else:
            continue

        # 重要性 = |kernel| × b_k，2D 形式 (in_dim, out_dim)
        if len(shape) == 4:
            scores = (np.abs(kernel_np) * np.maximum(bk_np, 0)).sum(axis=(0, 1))
        else:
            scores = np.abs(kernel_np) * np.maximum(bk_np, 0)

        layers_info.append({
            'name': layer.name, 'layer': layer, 'shape': shape,
            'in_dim': in_dim, 'out_dim': out_dim,
            'conn_mul': conn_mul, 'scores_2d': scores,
            'kernel_np': kernel_np, 'bk_np': bk_np,
        })

    if not layers_info:
        raise ValueError("No HGQ layers with kq found.")

    # ── 按层分配 EBOPS 预算 ───────────────────────────────────────────
    if budget_weight == 'capacity':
        caps = [info['in_dim'] * info['out_dim'] for info in layers_info]
    else:
        caps = [1.0] * len(layers_info)
    total_cap = sum(caps)

    ebops_per_conn = init_bk * b_a
    floor_budget = min_per_layer * ebops_per_conn
    total_floor  = floor_budget * len(layers_info)
    if total_floor > target_ebops * 0.95:
        min_per_layer = max(1, int(
            target_ebops * 0.95 / (len(layers_info) * ebops_per_conn)))
        floor_budget = min_per_layer * ebops_per_conn
        total_floor  = floor_budget * len(layers_info)

    distributable = target_ebops - total_floor
    budgets = [floor_budget + distributable * (c / total_cap) for c in caps]

    # ── 逐层选择 top-N 连接 ───────────────────────────────────────────
    per_layer_bk_map = {}
    actual_total = 0.0

    if verbose:
        print(f"\n[StructuredPrune] target_ebops={target_ebops:.1f}, "
              f"init_bk={init_bk}, b_a={b_a}")
        print(f"  {'layer':12s}  {'budget':>8s}  "
              f"{'n_total':>7s}  {'n_surv':>6s}  {'pct':>5s}  "
              f"{'mean_bk':>8s}  {'E_est':>8s}")

    for info, budget_l in zip(layers_info, budgets):
        layer  = info['layer']
        shape  = info['shape']
        scores = info['scores_2d']
        cm     = info['conn_mul']
        in_dim, out_dim = info['in_dim'], info['out_dim']

        # 预算内存活数
        n_max = max(min_per_layer, int(budget_l / (ebops_per_conn * cm)))

        # 按重要性选 top-N
        flat_sc = scores.ravel()
        n_total = len(flat_sc)
        n_survive = min(n_max, int((flat_sc > 1e-12).sum()))
        n_survive = max(min_per_layer, n_survive)

        top_idx = np.argsort(-flat_sc)[:n_survive]
        surv_mask_flat = np.zeros(n_total, dtype=np.float32)
        surv_mask_flat[top_idx] = 1.0
        surv_mask_2d = surv_mask_flat.reshape(in_dim, out_dim)

        # 构建 b_k map
        if len(shape) == 2:
            bk_map = surv_mask_2d * init_bk
        else:
            bk_map = np.broadcast_to(
                (surv_mask_2d * init_bk)[np.newaxis, np.newaxis], shape
            ).copy().astype(np.float32)

        ebops_est = float(n_survive * ebops_per_conn * cm)
        actual_total += ebops_est
        per_layer_bk_map[layer.name] = bk_map

        if verbose:
            pct = n_survive / n_total * 100
            print(f"  {layer.name:12s}  {budget_l:8.1f}  "
                  f"{n_total:7d}  {n_survive:6d}  {pct:4.1f}%  "
                  f"{init_bk:8.3f}  {ebops_est:8.1f}")

    if verbose:
        print(f"  {'TOTAL':12s}  {target_ebops:8.1f}  {'':7s}  {'':6s}  {'':5s}  "
              f"{'':8s}  {actual_total:8.1f}")
        print(f"  ratio = {actual_total / target_ebops:.3f}\n")

    return per_layer_bk_map