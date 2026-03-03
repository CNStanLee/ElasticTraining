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
            b_new = np.where(active_mask > 0, np.clip(b_arr * alpha, 0.5, 8.0), 0.0)
            target_var.assign(b_new.astype(np.float32))

            # 同步 kq.i
            i_var = _get_kq_var(kq, 'i')
            if i_var is not None:
                i_arr = i_var.numpy()
                i_new = np.where(active_mask > 0, np.clip(i_arr * alpha, 0.5, 8.0), 0.0)
                i_var.assign(i_new.astype(np.float32))

        if verbose:
            print(f"[HighBitPruner] current_ebops={current_ebops:.1f}  "
                  f"target={self.target_ebops:.1f}  alpha={alpha:.4f}")