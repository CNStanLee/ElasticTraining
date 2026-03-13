"""
谱约束一次性剪枝工具集
==========================================

核心 API
--------
spectral_quant_prune_to_ebops(model, target_ebops, sample_input, ...)
    谱 / 拓扑友好的一次性剪枝（默认方法）

bisect_ebops_to_target(model, target_ebops, sample_input, ...)
    二分搜索校准位宽使 eBOPs 精确命中 target

snap_active_bk(model, snap_min=1.0)
    消除 limbo 连接：b ∈ (0, snap_min) → snap_min

compute_bw_aware_degree(model, target_ebops, ...)
    联合求解每层 Ramanujan 度 + 初始位宽

apply_ramanujan_bw_init(model, ...)
    应用双正则掩膜 + 位宽到 kq.b / kq.i / kernel

RamanujanMaskEnforcer (Callback)
    训练中钳位被剪连接，可选渐进放开
"""

from __future__ import annotations

import math

import keras
import numpy as np
import tensorflow as tf

from . import _get_kq_var, _flatten_layers


# ═══════════════════════════════════════════════════════════════════════════════
# 双正则二部图掩膜生成
# ═══════════════════════════════════════════════════════════════════════════════

def _biregular_2d_mask(in_dim: int, out_dim: int, col_degree: int, rng) -> np.ndarray:
    """生成双正则二部图掩膜。

    保证：
    - 每个输出列恰好连接 col_degree 个输入
    - 每个输入行度数近似均匀
    - 所有输入节点至少有 1 条连接 (当 E >= in_dim)
    """
    col_degree = min(col_degree, in_dim)
    total_edges = col_degree * out_dim

    if col_degree >= in_dim:
        return np.ones((in_dim, out_dim), dtype=np.float32)

    # 计算目标行度
    d_row_floor = total_edges // in_dim
    remainder = total_edges % in_dim
    row_degrees = np.full(in_dim, d_row_floor, dtype=int)
    if remainder > 0:
        extra = rng.choice(in_dim, size=remainder, replace=False)
        row_degrees[extra] += 1
    row_degrees = np.minimum(row_degrees, out_dim)

    # 贪心行优先分配 (Erdős–Gallai ordering)
    mask = np.zeros((in_dim, out_dim), dtype=np.float32)
    col_remaining = np.full(out_dim, col_degree, dtype=int)
    row_order = np.argsort(-row_degrees)

    for i in row_order:
        d_i = int(row_degrees[i])
        if d_i == 0:
            continue
        available = np.where(col_remaining > 0)[0]
        if len(available) <= d_i:
            selected = available
        else:
            weights = col_remaining[available].astype(np.float64)
            weights /= weights.sum()
            selected = rng.choice(available, size=d_i, replace=False, p=weights)
        mask[i, selected] = 1.0
        col_remaining[selected] -= 1

    # 修补列度偏差
    for j in range(out_dim):
        actual = int(mask[:, j].sum())
        deficit = col_degree - actual
        if deficit > 0:
            zeros = np.where(mask[:, j] == 0)[0]
            if len(zeros) > 0:
                fill = rng.choice(zeros, size=min(deficit, len(zeros)), replace=False)
                mask[fill, j] = 1.0

    return mask


def _ramanujan_like_mask(shape, degree: int, rng) -> np.ndarray:
    """生成双正则 Ramanujan 式稀疏掩膜 (1=保留, 0=剪掉)。"""
    if len(shape) == 2:
        return _biregular_2d_mask(int(shape[0]), int(shape[1]), degree, rng)
    elif len(shape) == 4:
        kh, kw, in_ch, out_ch = [int(x) for x in shape]
        base = _biregular_2d_mask(in_ch, out_ch, degree, rng)
        return np.broadcast_to(base[None, None], (kh, kw, in_ch, out_ch)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported kernel shape: {shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# compute_bw_aware_degree — 联合求解每层度 + 位宽
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bw_aware_degree(
    model: keras.Model,
    target_ebops: float,
    b_a_init: float = 3.0,
    b_k_min: float = 1.0,
    b_k_max: float = 8.0,
    multiplier: float = 1.5,
    min_degree: int = 4,
    budget_weight: str = 'capacity',
    verbose: bool = True,
) -> tuple[dict, dict]:
    """联合求解每层 Ramanujan 度和初始 kernel 位宽。

    算法：
    1. d_l = clamp(round(sqrt(N_in) * multiplier), min_degree, N_in)  (谱条件)
    2. 按 capacity/uniform 分配 eBOPs 预算 E_l
    3. 反推 b_k_l = E_l / (d_l * N_out * b_a_init)

    Returns: (per_layer_degree, per_layer_bk)
    """
    layers_info = []
    for layer in _flatten_layers(model):
        if getattr(layer, 'kq', None) is None or not hasattr(layer, 'kernel'):
            continue
        shape = layer.kernel.shape
        if len(shape) == 2:
            in_dim, out_dim = int(shape[0]), int(shape[1])
            conn_mul, Nin_eff, Nout, deg_base = 1, in_dim, out_dim, in_dim
        elif len(shape) == 4:
            kh, kw, in_ch, out_ch = [int(x) for x in shape]
            conn_mul = kh * kw
            Nin_eff, Nout, deg_base = in_ch * conn_mul, out_ch, in_ch
        else:
            continue
        d_l = max(min_degree, min(int(round(math.sqrt(deg_base) * multiplier)), deg_base))
        layers_info.append((layer, Nin_eff, Nout, d_l, conn_mul))

    if not layers_info:
        raise ValueError("No HGQ layers found in model.")

    # 预算权重
    if budget_weight == 'capacity':
        weights = [N_in * N_out for _, N_in, N_out, _, _ in layers_info]
    elif budget_weight == 'uniform':
        weights = [1.0] * len(layers_info)
    else:
        raise ValueError(f"Unknown budget_weight='{budget_weight}'")
    total_weight = sum(weights)

    per_layer_degree, per_layer_bk = {}, {}
    actual_ebops_sum = 0.0

    if verbose:
        print(f"\n[compute_bw_aware_degree] target_ebops={target_ebops:.0f}, "
              f"b_a_init={b_a_init}, budget_weight={budget_weight}")
        print(f"  {'layer':20s}  {'N_in':>6}  {'N_out':>5}  {'d_l':>4}  "
              f"{'sparsity':>8}  {'b_k':>6}  {'E_l':>10}")

    for (layer, Nin_eff, Nout, d_l, conn_mul), w in zip(layers_info, weights):
        E_l = target_ebops * (w / total_weight)
        b_k_l = float(np.clip(
            E_l / max((d_l * conn_mul) * Nout * b_a_init, 1e-9),
            b_k_min, b_k_max
        ))
        E_l_actual = (d_l * conn_mul) * Nout * b_k_l * b_a_init
        actual_ebops_sum += E_l_actual

        per_layer_degree[layer.name] = d_l
        per_layer_bk[layer.name] = b_k_l

        if verbose:
            deg_base = int(layer.kernel.shape[0]) if len(layer.kernel.shape) == 2 else int(layer.kernel.shape[2])
            sparsity = 1.0 - (d_l / max(deg_base, 1))
            print(f"  {layer.name:20s}  {Nin_eff:6d}  {Nout:5d}  {d_l:4d}  "
                  f"{sparsity:7.1%}  {b_k_l:6.3f}  {E_l_actual:10.1f}")

    if verbose:
        print(f"  {'TOTAL':20s}  {'':>6}  {'':>5}  {'':>4}  {'':>8}  {'':>6}  "
              f"{actual_ebops_sum:10.1f}  (target={target_ebops:.0f}, "
              f"ratio={actual_ebops_sum / target_ebops:.3f})\n")

    return per_layer_degree, per_layer_bk


# ═══════════════════════════════════════════════════════════════════════════════
# apply_ramanujan_bw_init — 应用双正则掩膜 + 位宽
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ramanujan_bw_init(
    model: keras.Model,
    per_layer_degree: dict,
    per_layer_bk: dict,
    seed: int = 42,
    pruned_frac_bits: float = 0.0,
    pruned_int_bits: float = 0.0,
    active_int_bits: float = 1.0,
    also_zero_kernel: bool = True,
    verbose: bool = True,
):
    """位宽感知 Ramanujan 稀疏初始化。

    保留连接 → kq.b=per_layer_bk, kq.i=active_int_bits
    剪掉连接 → kq.b=0, kq.i=0, kernel=0
    """
    rng = np.random.RandomState(seed)

    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None or not hasattr(layer, 'kernel'):
            continue

        kernel_var = layer.kernel
        shape = kernel_var.shape
        d_l = per_layer_degree.get(layer.name)
        b_k_l = per_layer_bk.get(layer.name)
        if d_l is None:
            continue

        mask = _ramanujan_like_mask(shape, d_l, rng)
        pruned = 1.0 - mask

        # kq.b (fractional bits)
        b_var = _get_kq_var(kq, 'b')
        f_var = _get_kq_var(kq, 'f')
        target_bvar = b_var if b_var is not None else f_var
        if target_bvar is not None:
            target_bvar.assign((mask * b_k_l + pruned * pruned_frac_bits).astype(np.float32))

        # kq.i (integer bits)
        i_var = _get_kq_var(kq, 'i')
        if i_var is not None:
            i_var.assign((mask * active_int_bits + pruned * pruned_int_bits).astype(np.float32))

        # kernel 清零
        if also_zero_kernel:
            kernel_var.assign(kernel_var.numpy() * mask)

        # 挂 mask 到 layer，供 MaskEnforcer 使用
        layer.ramanujan_mask = tf.constant(mask, dtype=tf.float32)

        if verbose:
            active = int(mask.sum())
            total = int(mask.size)
            print(f"[RamanujanBWInit] {layer.name:20s}  shape={list(shape)}  "
                  f"d={d_l}  b_k={b_k_l:.3f}  "
                  f"active={active}/{total}  sparsity={1 - mask.mean():.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# RamanujanMaskEnforcer — 训练中维持稀疏拓扑
# ═══════════════════════════════════════════════════════════════════════════════

class RamanujanMaskEnforcer(keras.callbacks.Callback):
    """训练中钳位被剪连接的 kq.b/kq.i 为 0，防止 optimizer 恢复。

    支持渐进放开：
    - [0, release_epoch): 完全固定
    - [release_epoch, release_epoch+fade_epochs): 线性衰减强度
    - 之后: 完全放开
    """

    def __init__(
        self,
        release_epoch: int | None = None,
        fade_epochs: int = 0,
        enforce_frac_bits: float = 0.0,
        enforce_int_bits: float = 0.0,
        min_active_frac_bits: float | None = None,
    ):
        super().__init__()
        self.release_epoch = release_epoch
        self.fade_epochs = max(fade_epochs, 1)
        self.enforce_frac_bits = enforce_frac_bits
        self.enforce_int_bits = enforce_int_bits
        self.min_active_frac_bits = min_active_frac_bits
        self._current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def _enforce_strength(self) -> float:
        if self.release_epoch is None:
            return 1.0
        epoch = self._current_epoch
        if epoch < self.release_epoch:
            return 1.0
        progress = (epoch - self.release_epoch) / self.fade_epochs
        return max(0.0, 1.0 - progress)

    def on_train_batch_end(self, batch, logs=None):
        strength = self._enforce_strength()
        if strength <= 0.0:
            return

        for layer in self.model.layers:
            mask = getattr(layer, "ramanujan_mask", None)
            if mask is None:
                continue
            kq = getattr(layer, "kq", None)
            if kq is None:
                continue

            pruned = 1.0 - mask.numpy()

            b_var = _get_kq_var(kq, "b")
            if b_var is not None:
                b_arr = b_var.numpy()
                if strength >= 1.0:
                    b_arr = np.where(pruned > 0, self.enforce_frac_bits, b_arr)
                else:
                    target = np.where(pruned > 0, self.enforce_frac_bits, b_arr)
                    b_arr = strength * target + (1.0 - strength) * b_arr
                # 活跃连接下限保护
                if self.min_active_frac_bits is not None and strength > 0.0:
                    effective_floor = self.min_active_frac_bits * strength
                    active = mask.numpy() > 0
                    b_arr = np.where(active & (b_arr < effective_floor), effective_floor, b_arr)
                b_var.assign(b_arr)

            i_var = _get_kq_var(kq, "i")
            if i_var is not None:
                i_arr = i_var.numpy()
                if strength >= 1.0:
                    i_arr = np.where(pruned > 0, self.enforce_int_bits, i_arr)
                else:
                    target = np.where(pruned > 0, self.enforce_int_bits, i_arr)
                    i_arr = strength * target + (1.0 - strength) * i_arr
                i_var.assign(i_arr)


# ═══════════════════════════════════════════════════════════════════════════════
# eBOPs 测量工具
# ═══════════════════════════════════════════════════════════════════════════════

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
    from hgq.layers import QLayerBase

    _forward_update_ebops_no_bn_drift(model, sample_input)
    total = 0
    for layer in model._flatten_layers():
        if isinstance(layer, QLayerBase) and layer.enable_ebops and layer._ebops is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def _dense_prunable_layers(model):
    """返回模型中所有可剪枝的 2D-kernel HGQ 层。"""
    layers = []
    for layer in _flatten_layers(model):
        if not hasattr(layer, "kernel") or getattr(layer, "kq", None) is None:
            continue
        if len(layer.kernel.shape) != 2:
            continue
        layers.append(layer)
    return layers


def _layer_active_mask(layer) -> np.ndarray:
    """获取层当前活跃连接掩膜。"""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Top-K 连通掩膜
# ═══════════════════════════════════════════════════════════════════════════════

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


def _apply_mask_and_quant(layer, mask: np.ndarray, b_floor: float, b_ceiling: float):
    """对层应用结构化掩膜并同步量化参数。"""
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

    # bias quantizer: 保留活跃输出的偏置量化
    bq = getattr(layer, "bq", None)
    if bq is not None:
        cols_active = np.sum(m, axis=0) > 0.5
        bb_var = _get_kq_var(bq, "b")
        if bb_var is None:
            bb_var = _get_kq_var(bq, "f")
        bi_var = _get_kq_var(bq, "i")
        bk_var = _get_kq_var(bq, "k")
        if bb_var is not None:
            bb_old = bb_var.numpy().astype(np.float32)
            bb_new = np.where(cols_active, np.clip(np.maximum(bb_old, b_floor), b_floor, b_ceiling), 0.0)
            bb_var.assign(bb_new.astype(np.float32))
        if bi_var is not None:
            bi_old = bi_var.numpy().astype(np.float32)
            bi_var.assign(np.where(cols_active, np.clip(bi_old, -2.0, 6.0), -16.0).astype(np.float32))
        if bk_var is not None:
            bk_var.assign(np.where(cols_active, 1.0, 0.0).astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# SensitivityAwarePruner
# ═══════════════════════════════════════════════════════════════════════════════

class SensitivityAwarePruner:
    """基于每层 eBOPs 贡献敏感度的剪枝器。"""

    def __init__(
        self,
        target_ebops: float,
        pruned_threshold: float = 0.1,
        protect_power: float = 0.5,
        b_k_min: float = 0.01,
        b_k_max: float = 8.0,
    ):
        self.target_ebops = float(target_ebops)
        self.pruned_threshold = pruned_threshold
        self.protect_power = protect_power
        self.b_k_min = b_k_min
        self.b_k_max = b_k_max

    def prune_to_ebops(self, model, current_ebops: float, verbose: bool = True):
        if current_ebops <= 0:
            raise ValueError("current_ebops must be > 0")
        global_alpha = self.target_ebops / current_ebops

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

        for layer, target_var, b_arr, mean_bk in layer_info:
            if global_mean_bk > 0 and mean_bk > 0:
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


# ═══════════════════════════════════════════════════════════════════════════════
# structured_chain_prune_to_ebops
# ═══════════════════════════════════════════════════════════════════════════════

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

    widths = [int(weights[0].shape[0])] + [int(w.shape[1]) for w in weights]
    hidden_idx = list(range(1, len(widths) - 1))
    if not hidden_idx:
        return compute_model_ebops(model, sample_input)

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
        remove_order[h] = np.argsort(s).tolist()
        keep_map[h] = np.ones_like(s, dtype=bool)
        removed_ptr[h] = 0
        min_keep[h] = int(max(1, min(min_hidden_width, len(s))))

    def predict_ebops(curr_widths):
        total = 0.0
        for li, c in enumerate(edge_costs):
            total += c * curr_widths[li] * curr_widths[li + 1]
        return total

    n_guard = 0
    while predict_ebops(widths) > target_ebops * 1.15:
        n_guard += 1
        if n_guard > 10000:
            break
        best = None
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

    for li, layer in enumerate(layers):
        w = np.array(layer.kernel.numpy(), dtype=np.float32)
        in_dim, out_dim = w.shape
        rows_keep = np.ones(in_dim, dtype=bool) if li == 0 else keep_map[li]
        cols_keep = np.ones(out_dim, dtype=bool) if li == len(layers) - 1 else keep_map[li + 1]
        mask = np.outer(rows_keep, cols_keep).astype(np.float32)
        _apply_mask_and_quant(layer, mask, b_floor=b_floor, b_ceiling=b_ceiling)
        if verbose:
            print(f"  [StructuredLowBudget] {layer.name:20s}  "
                  f"in_keep={int(np.sum(rows_keep))}/{in_dim}  "
                  f"out_keep={int(np.sum(cols_keep))}/{out_dim}  "
                  f"active={int(np.sum(mask))}")

    measured = compute_model_ebops(model, sample_input)
    if verbose:
        w_desc = ", ".join([str(widths[i]) for i in range(1, len(widths) - 1)])
        print(f"[StructuredLowBudget] hidden_widths=[{w_desc}]  "
              f"pred_ebops={predict_ebops(widths):.1f}  measured_ebops={measured:.1f}  "
              f"target={target_ebops:.1f}")
    return measured


# ═══════════════════════════════════════════════════════════════════════════════
# spectral_quant_prune_to_ebops — 谱/拓扑友好的一次性剪枝
# ═══════════════════════════════════════════════════════════════════════════════

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

    # 高预算场景只需要温和压缩
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

    # 用预算估计每层目标度与建议位宽
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

        if kernel.ndim != 2:
            continue
        in_dim, out_dim = kernel.shape
        d = int(per_layer_degree.get(name, min_degree))
        d = int(np.clip(d, min_degree, in_dim))

        mask = _build_topk_mask_with_connectivity(kernel, d)
        layer.kernel.assign((kernel * mask).astype(np.float32))

        # 量化参数：活跃连接保底位宽
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
            # 保留预训练位宽与预算计算值的较大者，
            # 后续由 bisect 全局缩放到目标 eBOPs
            b_base = float(np.clip(per_layer_bk.get(name, b_floor), b_floor, b_ceiling))
            b_active = np.clip(
                np.maximum(b_old, b_base) * (0.8 + 0.4 * importance),
                b_floor, b_ceiling,
            )
            b_new = np.where(mask > 0.5, b_active, 0.0)
            b_var.assign(b_new.astype(np.float32))

        if i_var is not None:
            i_old = i_var.numpy().astype(np.float32)
            i_new = np.where(mask > 0.5, np.clip(i_old, -2.0, 6.0), -16.0)
            i_var.assign(i_new.astype(np.float32))

        if k_var is not None:
            k_new = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
            k_var.assign(k_new)

        # bias quantizer
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
                f"deg_row_zero={int(np.sum(row_deg == 0))}"
            )

    measured = compute_model_ebops(model, sample_input)
    if verbose:
        print(f"[SpectralQuantPruner] post-structural measured_ebops={measured:.1f}  target={target_ebops:.1f}")
    return measured, False


# ═══════════════════════════════════════════════════════════════════════════════
# spectral_sensitivity_prune_to_ebops — 结构化谱剪枝 + 敏感度重分配
# ═══════════════════════════════════════════════════════════════════════════════

def spectral_sensitivity_prune_to_ebops(
    model,
    target_ebops: float,
    sample_input,
    min_degree: int = 2,
    b_floor: float = 0.25,
    b_ceiling: float = 6.0,
    pruned_threshold: float = 0.1,
    protect_power: float = 0.5,
    verbose: bool = True,
):
    """先做谱/结构剪枝，再做 sensitivity 位宽重分配。"""
    measured, used_structured = spectral_quant_prune_to_ebops(
        model,
        target_ebops=target_ebops,
        sample_input=sample_input,
        min_degree=min_degree,
        b_floor=b_floor,
        b_ceiling=b_ceiling,
        verbose=verbose,
    )

    current_ebops = compute_model_ebops(model, sample_input)
    pruner = SensitivityAwarePruner(
        target_ebops=float(target_ebops),
        pruned_threshold=pruned_threshold,
        protect_power=protect_power,
        b_k_min=max(float(b_floor), 0.01),
        b_k_max=float(b_ceiling),
    )
    pruner.prune_to_ebops(model, current_ebops=current_ebops, verbose=verbose)
    measured = compute_model_ebops(model, sample_input)

    if verbose:
        print(
            f"[Spectral+Sensitivity] after refine measured_ebops={measured:.1f}  "
            f"target={target_ebops:.1f}  structured={used_structured}"
        )
    return measured, used_structured


# ═══════════════════════════════════════════════════════════════════════════════
# bisect_ebops_to_target — 二分搜索校准位宽
# ═══════════════════════════════════════════════════════════════════════════════

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
        if k_snap is not None:
            active_mask = (b_snap > 0.0) | (k_snap > 0.0)
        else:
            active_mask = b_snap > 0.0
        b_scale_mask = b_snap > 0.0
        i_var = _get_kq_var(kq, 'i')
        i_snap = i_var.numpy().astype(np.float32).copy() if i_var is not None else None
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
            'b_var': b_var, 'b_snap': b_snap, 'b_scale_mask': b_scale_mask,
            'active_mask': active_mask,
            'i_var': i_var, 'i_snap': i_snap,
            'k_var': k_var, 'k_snap': k_snap,
            'bb_var': bb_var, 'bb_snap': bb_snap, 'bb_scale_mask': bb_scale_mask,
            'bactive_mask': bactive_mask,
            'bi_var': bi_var, 'bi_snap': bi_snap,
            'bk_var': bk_var, 'bk_snap': bk_snap,
        })

    if not snapshots:
        return 0.0

    def apply_scale(scale: float):
        for s in snapshots:
            b_new = np.where(
                s['b_scale_mask'],
                np.clip(s['b_snap'] * scale, b_k_min, b_k_max),
                0.0,
            )
            s['b_var'].assign(b_new.astype(np.float32))
            if s['i_var'] is not None and s['i_snap'] is not None:
                i_new = np.where(s['active_mask'], s['i_snap'], -16.0)
                s['i_var'].assign(i_new.astype(np.float32))
            if s['k_var'] is not None and s['k_snap'] is not None:
                k_new = np.where(s['active_mask'], s['k_snap'], 0.0)
                s['k_var'].assign(k_new.astype(np.float32))
            if (s['bb_var'] is not None and s['bb_snap'] is not None
                    and s['bb_scale_mask'] is not None and s['bactive_mask'] is not None):
                bb_new = np.where(
                    s['bb_scale_mask'],
                    np.clip(s['bb_snap'] * scale, b_k_min, b_k_max),
                    0.0,
                )
                s['bb_var'].assign(bb_new.astype(np.float32))
                if s['bi_var'] is not None and s['bi_snap'] is not None:
                    s['bi_var'].assign(
                        np.where(s['bactive_mask'], s['bi_snap'], -16.0).astype(np.float32))
                if s['bk_var'] is not None and s['bk_snap'] is not None:
                    s['bk_var'].assign(
                        np.where(s['bactive_mask'], s['bk_snap'], 0.0).astype(np.float32))

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
            print(f"  [BisectEBOPs] max-reachable={final_e:.1f} < target={target_ebops:.1f}")
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
                print(f"  [BisectEBOPs] floor={final_e:.1f} > target={target_ebops:.1f}")
                return final_e
            print(f"  [BisectEBOPs] ebops floor={lo_e:.1f} > target={target_ebops:.1f}, "
                  f"pruning weakest connections...")
            # 收集可杀连接
            all_entries = []
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
                        all_entries.append((snap_idx, base_off + pos, score))

            all_entries.sort(key=lambda x: x[2])
            if not all_entries:
                return measure(1.0)

            kill_step = max(1, len(all_entries) // 50)
            killed = 0
            for i in range(0, len(all_entries), kill_step):
                batch = all_entries[i:i + kill_step]
                for snap_idx, pos, _ in batch:
                    s = snapshots[snap_idx]
                    if pos >= 10_000_000:
                        if s['bb_snap'] is None:
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
                check_e = measure(1.0)
                if check_e <= target_ebops * (1.0 + tolerance):
                    print(f"  [BisectEBOPs] killed {killed}/{len(all_entries)} weakest, ebops={check_e:.1f}")
                    if abs(check_e - target_ebops) / max(target_ebops, 1.0) <= tolerance:
                        return check_e
                    break
            else:
                final_e = measure(1.0)
                print(f"  [BisectEBOPs] killed all {killed}, ebops={final_e:.1f}")
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
    print(f"  [BisectEBOPs] scale={best_s:.5f}  final_ebops={final_e:.1f}  "
          f"target={target_ebops:.1f}  err={abs(final_e - target_ebops) / max(target_ebops, 1.0) * 100:.1f}%")
    return final_e


# ═══════════════════════════════════════════════════════════════════════════════
# snap_active_bk — 消除 limbo 连接
# ═══════════════════════════════════════════════════════════════════════════════

def snap_active_bk(
    model,
    snap_min: float = 1.0,
    min_degree: int = 2,
    verbose: bool = True,
) -> tuple[int, int]:
    """谱感知 limbo 消除：只在列度不足 min_degree 时保留 1-bit 连接，其余杀掉。

    bisect_ebops_to_target 通过全局缩放 bk 来命中目标 eBOPs，可能把某些 _b 压到
    (0, 1) 区间。这些 limbo 连接在 round_conv 后可能死亡。

    策略：
    - 对每个 2D 层的每个输出列 j，统计 solid 连接 (b >= snap_min) 的列度
    - 若 solid_degree(j) >= min_degree：该列的 limbo 连接直接杀掉 (b→0)
    - 若 solid_degree(j) < min_degree：从该列 limbo 中选最重要的补到 min_degree，
      snap 到 1-bit，剩余杀掉

    Returns: (n_snapped, n_killed) — 保留为 1-bit 的数量 和 杀掉的数量
    """
    n_snapped = 0
    n_killed = 0

    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None:
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is None:
            continue

        b = b_var.numpy().astype(np.float32)
        limbo = (b > 0.0) & (b < float(snap_min))
        if not limbo.any():
            continue

        kernel = getattr(layer, 'kernel', None)
        # 只对 2D 权重做列度感知 snap
        if kernel is not None and len(kernel.shape) == 2:
            in_dim, out_dim = kernel.shape
            abs_w = np.abs(kernel.numpy().astype(np.float32))
            solid = b >= float(snap_min)  # 已经稳固的连接
            b_new = b.copy()

            for j in range(out_dim):
                col_limbo = np.where(limbo[:, j])[0]  # 该列的 limbo 行
                if len(col_limbo) == 0:
                    continue
                solid_deg = int(solid[:, j].sum())
                deficit = max(0, min_degree - solid_deg)

                if deficit > 0 and len(col_limbo) > 0:
                    # 按 |kernel| 降序选最重要的 deficit 个 snap 到 1-bit
                    importance = abs_w[col_limbo, j]
                    keep_count = min(deficit, len(col_limbo))
                    keep_idx = col_limbo[np.argsort(-importance)[:keep_count]]
                    kill_idx = np.setdiff1d(col_limbo, keep_idx)
                    b_new[keep_idx, j] = float(snap_min)
                    b_new[kill_idx, j] = 0.0
                    n_snapped += keep_count
                    n_killed += len(kill_idx)
                else:
                    # 列度足够，所有 limbo 直接杀掉
                    b_new[col_limbo, j] = 0.0
                    n_killed += len(col_limbo)

            b_var.assign(b_new.astype(np.float32))

            # 同步 kernel：被杀掉的连接清零
            killed_mask = (b == 0.0) | ((limbo) & (b_new == 0.0))
            if kernel is not None:
                k_np = kernel.numpy().astype(np.float32)
                # 只清零新杀掉的
                new_dead = limbo & (b_new == 0.0)
                if new_dead.any():
                    k_np[new_dead] = 0.0
                    kernel.assign(k_np)

            # 同步 kq.k (keep mask)
            k_var = _get_kq_var(kq, 'k')
            if k_var is not None:
                k_arr = k_var.numpy().astype(np.float32)
                new_dead = limbo & (b_new == 0.0)
                if new_dead.any():
                    k_arr[new_dead] = 0.0
                    k_var.assign(k_arr)

            # 同步 kq.i
            i_var = _get_kq_var(kq, 'i')
            if i_var is not None:
                i_arr = i_var.numpy().astype(np.float32)
                new_dead = limbo & (b_new == 0.0)
                if new_dead.any():
                    i_arr[new_dead] = -16.0
                    i_var.assign(i_arr)

            if verbose:
                print(f'  [snap] {layer.name:20s}  snapped={int((limbo & (b_new >= snap_min)).sum())}  '
                      f'killed={int((limbo & (b_new == 0.0)).sum())}  '
                      f'col_deg_min={int(np.sum(b_new[:, :] > 0, axis=0).min())}')
        else:
            # 非 2D 层（如输出层）：limbo 全杀
            layer_killed = int(limbo.sum())
            b_var.assign(np.where(limbo, 0.0, b).astype(np.float32))
            n_killed += layer_killed

    return n_snapped, n_killed

# ═══════════════════════════════════════════════════════════════════════════════
# Ablation 剪枝方法: random / magnitude (用于对比实验)
# ═══════════════════════════════════════════════════════════════════════════════

def random_prune_to_ebops(model, target_ebops: float, sample_input, b_floor=0.35,
                          b_ceiling=6.0, seed=42, verbose=True):
    """随机剪枝到目标 eBOPs (无谱约束, 无连通保证)。

    对每层随机选择保留的连接 (同等稀疏率), 不考虑拓扑连通性。
    用于 ablation 对比, 展示谱约束的必要性。
    """
    rng = np.random.RandomState(seed)
    current_ebops = compute_model_ebops(model, sample_input)
    if current_ebops <= 0:
        return
    keep_ratio = min(1.0, float(target_ebops) / float(current_ebops))

    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None or not hasattr(layer, 'kernel'):
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is None:
            continue
        shape = layer.kernel.shape
        mask = (rng.rand(*shape) < keep_ratio).astype(np.float32)
        _apply_mask_and_quant(layer, mask, b_floor=b_floor, b_ceiling=b_ceiling)
        if verbose:
            print(f'  [RandomPrune] {layer.name:20s}  '
                  f'keep={int(mask.sum())}/{int(np.prod(shape))}  '
                  f'sparsity={1 - mask.mean():.3f}')

    measured = compute_model_ebops(model, sample_input)
    if verbose:
        print(f'[RandomPrune] target={target_ebops:.0f}  measured={measured:.0f}')
    return measured


def magnitude_prune_to_ebops(model, target_ebops: float, sample_input, b_floor=0.35,
                             b_ceiling=6.0, verbose=True):
    """权重幅值剪枝到目标 eBOPs (全局 top-k, 无谱约束)。

    全局收集所有权重绝对值, 找到阈值使保留连接的 eBOPs ≈ target。
    用于 ablation 对比, 展示谱约束对极低预算的优势。
    """
    current_ebops = compute_model_ebops(model, sample_input)
    if current_ebops <= 0:
        return
    keep_ratio = min(1.0, float(target_ebops) / float(current_ebops))

    # 收集所有权重绝对值
    all_abs = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None or not hasattr(layer, 'kernel'):
            continue
        all_abs.extend(np.abs(layer.kernel.numpy()).ravel().tolist())

    if not all_abs:
        return
    all_abs = np.array(all_abs)
    n_keep = max(1, int(len(all_abs) * keep_ratio))
    threshold = float(np.sort(all_abs)[-n_keep]) if n_keep < len(all_abs) else 0.0

    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None or not hasattr(layer, 'kernel'):
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is None:
            continue
        w = np.abs(layer.kernel.numpy())
        mask = (w >= threshold).astype(np.float32)
        _apply_mask_and_quant(layer, mask, b_floor=b_floor, b_ceiling=b_ceiling)
        if verbose:
            print(f'  [MagnitudePrune] {layer.name:20s}  '
                  f'keep={int(mask.sum())}/{int(np.prod(w.shape))}  '
                  f'sparsity={1 - mask.mean():.3f}')

    measured = compute_model_ebops(model, sample_input)
    if verbose:
        print(f'[MagnitudePrune] threshold={threshold:.6f}  '
              f'target={target_ebops:.0f}  measured={measured:.0f}')
    return measured


def random_init_prune_to_ebops(model, target_ebops: float, sample_input, b_floor=0.35,
                               b_ceiling=6.0, seed=42, verbose=True):
    """随机重初始化权重后按比例剪枝 (最弱 baseline)。

    先用 Glorot uniform 重新初始化所有可剪枝层的 kernel 权重，
    然后按全局均匀稀疏率随机选择保留连接。
    用于 ablation 对比，展示预训练权重的重要性。
    """
    rng = np.random.RandomState(seed)
    current_ebops = compute_model_ebops(model, sample_input)
    if current_ebops <= 0:
        return
    keep_ratio = min(1.0, float(target_ebops) / float(current_ebops))

    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None or not hasattr(layer, 'kernel'):
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is None:
            continue
        shape = layer.kernel.shape
        # Glorot uniform re-initialization
        fan_in, fan_out = int(np.prod(shape[:-1])), int(shape[-1])
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        new_kernel = rng.uniform(-limit, limit, size=shape).astype(np.float32)
        layer.kernel.assign(new_kernel)
        # Random proportional pruning
        mask = (rng.rand(*shape) < keep_ratio).astype(np.float32)
        _apply_mask_and_quant(layer, mask, b_floor=b_floor, b_ceiling=b_ceiling)
        if verbose:
            print(f'  [RandomInitPrune] {layer.name:20s}  '
                  f'keep={int(mask.sum())}/{int(np.prod(shape))}  '
                  f'sparsity={1 - mask.mean():.3f}')

    measured = compute_model_ebops(model, sample_input)
    if verbose:
        print(f'[RandomInitPrune] target={target_ebops:.0f}  measured={measured:.0f}')
    return measured


def snip_prune_to_ebops(model, target_ebops: float, sample_input, sample_labels,
                        b_floor=0.35, b_ceiling=6.0, verbose=True):
    """SNIP: 单次连接敏感度剪枝 (Lee et al., ICLR 2019)。

    计算每个连接的敏感度 |w * dL/dw|，全局排序后保留 top-k。
    只需单次前向+反向传播即可决定剪枝掩膜。
    """
    current_ebops = compute_model_ebops(model, sample_input)
    if current_ebops <= 0:
        return
    keep_ratio = min(1.0, float(target_ebops) / float(current_ebops))

    # 收集可剪枝层
    prunable = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None or not hasattr(layer, 'kernel'):
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            prunable.append(layer)

    if not prunable:
        return

    # 单次前向+反向计算 connection sensitivity
    # 使用 model.trainable_variables (自动被 tape 追踪) 兼容 Keras 3
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        logits = model(sample_input, training=False)
        loss = loss_fn(sample_labels, logits)
    all_grads = tape.gradient(loss, model.trainable_variables)
    grad_map = {id(v): g for v, g in zip(model.trainable_variables, all_grads)}
    grads = [grad_map.get(id(l.kernel), None) for l in prunable]

    # 全局 connection sensitivity: |w * grad|
    all_scores = []
    for layer, g in zip(prunable, grads):
        if g is None:
            s = np.zeros_like(layer.kernel.numpy())
        else:
            s = np.abs(layer.kernel.numpy() * g.numpy())
        all_scores.append(s.ravel())
    all_scores = np.concatenate(all_scores)

    # 求全局阈值
    n_keep = max(1, int(len(all_scores) * keep_ratio))
    threshold = float(np.sort(all_scores)[-n_keep]) if n_keep < len(all_scores) else 0.0

    # 应用掩膜
    idx = 0
    for layer, g in zip(prunable, grads):
        w = layer.kernel.numpy()
        if g is None:
            scores = np.zeros_like(w)
        else:
            scores = np.abs(w * g.numpy())
        mask = (scores >= threshold).astype(np.float32)
        _apply_mask_and_quant(layer, mask, b_floor=b_floor, b_ceiling=b_ceiling)
        if verbose:
            print(f'  [SNIP] {layer.name:20s}  '
                  f'keep={int(mask.sum())}/{int(np.prod(w.shape))}  '
                  f'sparsity={1 - mask.mean():.3f}')

    measured = compute_model_ebops(model, sample_input)
    if verbose:
        print(f'[SNIP] threshold={threshold:.8f}  target={target_ebops:.0f}  measured={measured:.0f}')
    return measured


def grasp_prune_to_ebops(model, target_ebops: float, sample_input, sample_labels,
                         b_floor=0.35, b_ceiling=6.0, verbose=True):
    """GraSP: 梯度信号保持剪枝 (Wang et al., ICLR 2020)。

    保留能最大化梯度流的连接。分数 = -H·g = -(w * grad(grad·w))，
    即权重对 Hessian-gradient 乘积的贡献。保留分数最高的连接。
    """
    current_ebops = compute_model_ebops(model, sample_input)
    if current_ebops <= 0:
        return
    keep_ratio = min(1.0, float(target_ebops) / float(current_ebops))

    prunable = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None or not hasattr(layer, 'kernel'):
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            prunable.append(layer)

    if not prunable:
        return

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 第一步: 计算梯度 g = dL/dw
    # 使用 model.trainable_variables (自动被 tape 追踪) 兼容 Keras 3
    with tf.GradientTape() as tape1:
        logits1 = model(sample_input, training=False)
        loss1 = loss_fn(sample_labels, logits1)
    all_grads1 = tape1.gradient(loss1, model.trainable_variables)
    grad_map1 = {id(v): g for v, g in zip(model.trainable_variables, all_grads1)}
    grads1 = [grad_map1.get(id(l.kernel), None) for l in prunable]

    # 第二步: 计算 Hessian-gradient 乘积的近似
    # 对 gradient·weight 关于weight再求梯度 → Hg
    with tf.GradientTape() as tape2:
        logits2 = model(sample_input, training=False)
        loss2 = loss_fn(sample_labels, logits2)
    all_grads2 = tape2.gradient(loss2, model.trainable_variables)
    grad_map2 = {id(v): g for v, g in zip(model.trainable_variables, all_grads2)}
    grads2 = [grad_map2.get(id(l.kernel), None) for l in prunable]

    # GraSP score: -w * Hg ≈ -w * grad (简化为 Heuristic)
    # 完整的 Hessian 计算代价太大，使用一次梯度近似
    # score = -g * (g * w) 的符号可以分辨保持/损害梯度流
    all_scores = []
    for layer, g1, g2 in zip(prunable, grads1, grads2):
        w = layer.kernel.numpy()
        if g1 is None or g2 is None:
            s = np.zeros_like(w)
        else:
            g1_np = g1.numpy()
            g2_np = g2.numpy()
            # Hessian-gradient product approximation: score = -w * g2
            # 负号使得高分表示保留该连接对梯度流有利
            s = -(w * g2_np)
        all_scores.append(s.ravel())
    all_scores = np.concatenate(all_scores)

    # top-k 保留
    n_keep = max(1, int(len(all_scores) * keep_ratio))
    threshold = float(np.sort(all_scores)[-n_keep]) if n_keep < len(all_scores) else float('-inf')

    offset = 0
    for layer in prunable:
        w = layer.kernel.numpy()
        n = int(np.prod(w.shape))
        scores = all_scores[offset:offset + n].reshape(w.shape)
        offset += n
        mask = (scores >= threshold).astype(np.float32)
        _apply_mask_and_quant(layer, mask, b_floor=b_floor, b_ceiling=b_ceiling)
        if verbose:
            print(f'  [GraSP] {layer.name:20s}  '
                  f'keep={int(mask.sum())}/{int(np.prod(w.shape))}  '
                  f'sparsity={1 - mask.mean():.3f}')

    measured = compute_model_ebops(model, sample_input)
    if verbose:
        print(f'[GraSP] target={target_ebops:.0f}  measured={measured:.0f}')
    return measured


def synflow_prune_to_ebops(model, target_ebops: float, sample_input, b_floor=0.35,
                           b_ceiling=6.0, n_iters=100, verbose=True):
    """SynFlow: 数据无关迭代剪枝 (Tanaka et al., NeurIPS 2020)。

    不需要训练数据标签。使用全 1 输入，迭代计算 synaptic saliency
    S_j = |θ_j * (dR/dθ_j)| 其中 R = 1^T · f(1)，然后逐步剪去最低分数连接。
    多次迭代避免 layer-collapse。
    """
    current_ebops = compute_model_ebops(model, sample_input)
    if current_ebops <= 0:
        return
    target_keep_ratio = min(1.0, float(target_ebops) / float(current_ebops))

    prunable = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None or not hasattr(layer, 'kernel'):
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            prunable.append(layer)

    if not prunable:
        return

    # 保存原始权重
    original_kernels = [l.kernel.numpy().copy() for l in prunable]

    # 初始化掩膜 (全 1)
    masks = [np.ones(l.kernel.shape, dtype=np.float32) for l in prunable]

    # SynFlow 使用全 1 输入 (data-free)
    ones_input = tf.ones_like(sample_input[:1])

    for iteration in range(n_iters):
        # 计算当前迭代的保留比例 (指数衰减)
        # fraction = target^((it+1)/n_iters)
        frac = target_keep_ratio ** ((iteration + 1) / n_iters)

        # 用 |w| 替代 w 做前向 (SynFlow 要求非负)
        for i, layer in enumerate(prunable):
            abs_w = np.abs(original_kernels[i]) * masks[i]
            layer.kernel.assign(abs_w.astype(np.float32))

        # 前向传播计算 R = sum(output)
        # 使用 model.trainable_variables (自动被 tape 追踪) 兼容 Keras 3
        with tf.GradientTape() as tape:
            out = model(ones_input, training=False)
            R = tf.reduce_sum(out)
        all_grads = tape.gradient(R, model.trainable_variables)
        grad_map = {id(v): g for v, g in zip(model.trainable_variables, all_grads)}
        grads = [grad_map.get(id(l.kernel), None) for l in prunable]

        # Synaptic saliency: S = |θ * dR/dθ|
        all_scores = []
        for i, (layer, g) in enumerate(zip(prunable, grads)):
            w_abs = np.abs(original_kernels[i]) * masks[i]
            if g is None:
                s = np.zeros_like(w_abs)
            else:
                s = w_abs * np.abs(g.numpy())
            # 已剪掉的连接分数设为 0
            s *= masks[i]
            all_scores.append(s.ravel())
        all_scores_flat = np.concatenate(all_scores)

        # 仅对存活连接排序
        alive = all_scores_flat[all_scores_flat > 0]
        if len(alive) == 0:
            break
        n_total_alive = len(alive)
        n_keep = max(1, int(frac * sum(np.prod(m.shape) for m in masks)))
        n_keep = min(n_keep, n_total_alive)

        if n_keep < n_total_alive:
            threshold = float(np.sort(alive)[-(n_keep)])
        else:
            threshold = 0.0

        # 更新掩膜
        offset = 0
        for i, layer in enumerate(prunable):
            n = int(np.prod(masks[i].shape))
            scores = all_scores_flat[offset:offset + n].reshape(masks[i].shape)
            offset += n
            masks[i] = ((scores >= threshold) & (masks[i] > 0.5)).astype(np.float32)

    # 恢复原始权重并应用最终掩膜
    for i, layer in enumerate(prunable):
        layer.kernel.assign(original_kernels[i].astype(np.float32))
        _apply_mask_and_quant(layer, masks[i], b_floor=b_floor, b_ceiling=b_ceiling)
        if verbose:
            print(f'  [SynFlow] {layer.name:20s}  '
                  f'keep={int(masks[i].sum())}/{int(np.prod(masks[i].shape))}  '
                  f'sparsity={1 - masks[i].mean():.3f}')

    measured = compute_model_ebops(model, sample_input)
    if verbose:
        print(f'[SynFlow] iters={n_iters}  target={target_ebops:.0f}  measured={measured:.0f}')
    return measured
