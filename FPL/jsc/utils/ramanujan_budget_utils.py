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


def _biregular_2d_mask(in_dim, out_dim, col_degree, rng):
    """生成双正则二部图掩膜 (bi-regular bipartite graph mask)。

    保证：
    - 每个输出（列）恰好连接 col_degree 个输入
    - 每个输入（行）连接 floor(E/in_dim) 或 ceil(E/in_dim) 个输出
      其中 E = col_degree * out_dim 为总边数
    - 所有输入节点都至少有 1 条连接（当 E >= in_dim 时）

    算法：贪心行优先分配，按行度降序处理，权重采样保持列度均衡。
    """
    col_degree = min(col_degree, in_dim)
    total_edges = col_degree * out_dim

    if col_degree >= in_dim:
        return np.ones((in_dim, out_dim), dtype=np.float32)

    # ── 计算目标行度 ──────────────────────────────────────────────────────
    d_row_floor = total_edges // in_dim
    remainder   = total_edges % in_dim

    row_degrees = np.full(in_dim, d_row_floor, dtype=int)
    if remainder > 0:
        extra = rng.choice(in_dim, size=remainder, replace=False)
        row_degrees[extra] += 1

    # 安全检查：行度不超过 out_dim
    row_degrees = np.minimum(row_degrees, out_dim)

    # ── 贪心行优先分配 ────────────────────────────────────────────────────
    mask = np.zeros((in_dim, out_dim), dtype=np.float32)
    col_remaining = np.full(out_dim, col_degree, dtype=int)

    # 按行度降序处理（高约束行先分配，保证可行性 — Erdős–Gallai）
    row_order = np.argsort(-row_degrees)

    for i in row_order:
        d_i = int(row_degrees[i])
        if d_i == 0:
            continue

        available = np.where(col_remaining > 0)[0]
        if len(available) <= d_i:
            # 可用列不够，全部取走
            selected = available
        else:
            # 权重 = 列剩余容量，倾向于填充尚有空余的列
            weights = col_remaining[available].astype(np.float64)
            weights /= weights.sum()
            selected = rng.choice(available, size=d_i, replace=False, p=weights)

        mask[i, selected] = 1.0
        col_remaining[selected] -= 1

    # ── 修补列度（如果贪心过程中有微小偏差）────────────────────────────────
    for j in range(out_dim):
        actual = int(mask[:, j].sum())
        deficit = col_degree - actual
        if deficit > 0:
            zeros = np.where(mask[:, j] == 0)[0]
            if len(zeros) > 0:
                fill = rng.choice(zeros, size=min(deficit, len(zeros)), replace=False)
                mask[fill, j] = 1.0

    return mask


def _ramanujan_like_mask(shape, degree, rng):
    """生成双正则 (bi-regular) 拉马努金式稀疏掩膜（1=保留, 0=剪掉）。

    相比旧版仅保证列正则（column-regular），新版同时保证行侧度数近似均匀，
    确保所有输入节点都被连接（当总边数 >= in_dim 时）。
    """
    if len(shape) == 2:
        in_dim, out_dim = int(shape[0]), int(shape[1])
        return _biregular_2d_mask(in_dim, out_dim, degree, rng)
    elif len(shape) == 4:
        kh, kw, in_ch, out_ch = [int(x) for x in shape]
        base = _biregular_2d_mask(in_ch, out_ch, degree, rng)
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


class KQBStabilizer(keras.callbacks.Callback):
    """剪枝后 kq.b 稳定器：防止初期梯度导致 EBOPs 大幅偏离目标。

    问题背景
    --------
    一次性剪枝后，kq.b 被精确校准到目标 EBOPs。但训练一开始，分类损失的
    梯度会把 kq.b 往上推（更多位宽 = 更低量化噪声），导致 EBOPs 在第一个
    epoch 就远超目标。BetaOnlyBudgetController 只能在 epoch 结束后反应，已经
    太晚了。

    机制
    ----
    在每个 epoch 结束后，将当前 kq.b 与剪枝快照做加权混合：
        b_new = alpha * b_snapshot + (1 - alpha) * b_current

    - hold_epochs 内：alpha = hold_strength（接近 1.0，几乎冻结 kq.b）
    - release_epochs 内：alpha 从 hold_strength 线性衰减到 0（逐渐放开）
    - 之后：不再干预，完全由梯度 + beta 控制

    这让权重有时间适应低精度量化点，同时保持 EBOPs 稳定在目标附近。

    参数
    ----
    hold_epochs     : 完全保持快照的 epoch 数（默认 30）
    release_epochs  : 从保持到释放的过渡 epoch 数（默认 100）
    hold_strength   : 保持强度，1.0 = 完全冻结（默认 0.8）
    """

    def __init__(self, hold_epochs: int = 30, release_epochs: int = 100,
                 hold_strength: float = 0.8):
        super().__init__()
        self.hold_epochs = hold_epochs
        self.release_epochs = release_epochs
        self.hold_strength = hold_strength
        self._snapshot = None  # type: dict | None

    def capture_snapshot(self, model):
        """保存当前 kq.b 值作为参考点（在剪枝校准完成后调用）。"""
        self._snapshot = {}
        for layer in _flatten_layers(model):
            kq = getattr(layer, 'kq', None)
            if kq is None:
                continue
            b_var = _get_kq_var(kq, 'b')
            if b_var is None:
                b_var = _get_kq_var(kq, 'f')
            if b_var is not None:
                self._snapshot[id(layer)] = (b_var, b_var.numpy().copy())

    def on_epoch_end(self, epoch, logs=None):
        if self._snapshot is None:
            return
        total = self.hold_epochs + self.release_epochs
        if epoch >= total:
            return

        if epoch < self.hold_epochs:
            alpha = self.hold_strength
        else:
            progress = (epoch - self.hold_epochs) / max(self.release_epochs, 1)
            alpha = self.hold_strength * (1.0 - progress)

        for b_var, b_snap in self._snapshot.values():
            current = b_var.numpy()
            b_new = alpha * b_snap + (1.0 - alpha) * current
            b_var.assign(b_new.astype(np.float32))


class BetaOnlyBudgetController(keras.callbacks.Callback):
    """用 beta + rescue 投影维持 EBOPs 预算。

    核心思路
    --------
    1. **beta 调节（下行通道）**：
       - EBOPs > target → 增大 beta → 梯度压低 kq.b → EBOPs 下降
       - EBOPs < target → 减小 beta → 释放压力

    2. **rescue 投影（上行通道）**：
       beta=0 无法 *主动* 推高 kq.b，深度剪枝后分类梯度太弱，EBOPs 会自由落体。
       rescue 每 epoch 持续检测并施加温和的上行修正，像弹簧一样把 EBOPs 拉回目标:
         alpha = 1 + rescue_rate * max(0, 1 - ema_ebops / target)
       EBOPs 越低 → alpha 越大 → 修正越强，但被 rescue_max_alpha clamp。
       触发条件: EBOPs < target * (1 - rescue_threshold)

    参数
    ----
    target_ebops      : 目标 EBOPS
    margin            : 容忍带宽度（默认 0.1 = ±10%）
    beta_init/min/max : beta 范围
    adjust_factor     : beta 乘法调整因子
    ema_alpha         : EBOPs EMA 平滑系数
    warmup_epochs     : 预热期 epoch 数
    max_change_ratio  : 每 epoch beta 最大变化倍率
    init_ebops        : 初始化 EMA 的 EBOPs 值
    rescue_threshold  : 触发 rescue 的偏差阈值（默认 0.15 = EBOPs < 85% target）
    rescue_rate       : rescue 修正强度系数（默认 0.3）
    rescue_max_alpha  : rescue 单步最大放大倍率（默认 1.08）
    rescue_b_min      : kq.b 活跃判定阈值
    rescue_b_max      : kq.b 放大后上限
    """

    def __init__(
        self,
        target_ebops: float,
        margin: float = 0.1,
        beta_init: float = 1e-5,
        beta_min: float = 1e-7,
        beta_max: float = 1e-3,
        adjust_factor: float = 1.3,
        ema_alpha: float = 0.3,
        warmup_epochs: int = 0,
        max_change_ratio: float = 0,
        init_ebops: float = None,
        rescue_threshold: float = 0.15,
        rescue_rate: float = 0.3,
        rescue_max_alpha: float = 1.08,
        rescue_b_min: float = 0.05,
        rescue_b_max: float = 8.0,
    ):
        super().__init__()
        self.target_ebops     = float(target_ebops)
        self.margin           = margin
        self.beta_current     = float(beta_init)
        self.beta_min         = beta_min
        self.beta_max         = beta_max
        self.adjust_factor    = adjust_factor
        self.ema_alpha        = ema_alpha
        self.warmup_epochs    = warmup_epochs
        self.max_change_ratio = max_change_ratio
        self._ebops_ema       = float(init_ebops) if init_ebops is not None else None
        self._epoch_counter   = 0
        # rescue 参数
        self.rescue_threshold = rescue_threshold
        self.rescue_rate      = rescue_rate
        self.rescue_max_alpha = rescue_max_alpha
        self.rescue_b_min     = rescue_b_min
        self.rescue_b_max     = rescue_b_max

    def _set_beta(self, value):
        bv = tf.constant(value, dtype=tf.float32)
        for layer in _flatten_layers(self.model):
            if hasattr(layer, '_beta'):
                try:
                    layer._beta.assign(tf.ones_like(layer._beta) * bv)
                except Exception:
                    pass

    def _rescue_scale_kqb(self, alpha: float):
        """直接放大所有活跃 kq.b，为 EBOPs 提供上行动力。"""
        for layer in _flatten_layers(self.model):
            kq = getattr(layer, 'kq', None)
            if kq is None:
                continue
            b_var = _get_kq_var(kq, 'b')
            if b_var is None:
                b_var = _get_kq_var(kq, 'f')
            if b_var is None:
                continue
            b_arr = b_var.numpy()
            active = b_arr > self.rescue_b_min
            if not np.any(active):
                continue
            b_new = np.where(
                active,
                np.clip(b_arr * alpha, self.rescue_b_min, self.rescue_b_max),
                b_arr,
            )
            b_var.assign(b_new.astype(np.float32))

    def on_epoch_begin(self, epoch, logs=None):
        self._set_beta(self.beta_current)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        raw_ebops = float(logs.get('ebops', float('nan')))
        if not math.isfinite(raw_ebops) or raw_ebops <= 0:
            return

        self._epoch_counter += 1

        if self._ebops_ema is None:
            self._ebops_ema = raw_ebops
        else:
            self._ebops_ema = (self.ema_alpha * raw_ebops
                               + (1.0 - self.ema_alpha) * self._ebops_ema)

        upper = self.target_ebops * (1.0 + self.margin)
        lower = self.target_ebops * (1.0 - self.margin)

        old_beta = self.beta_current

        # 预热：前 warmup_epochs 内用更温和的调整力度
        if self.warmup_epochs > 0 and self._epoch_counter <= self.warmup_epochs:
            warmup_scale = self._epoch_counter / self.warmup_epochs
        else:
            warmup_scale = 1.0
        effective_factor = 1.0 + (self.adjust_factor - 1.0) * warmup_scale

        if self._ebops_ema > upper:
            # EBOPs 超标 → 增大 beta 施加压缩力
            self.beta_current = min(
                self.beta_current * effective_factor,
                self.beta_max,
            )
        elif self._ebops_ema < lower:
            # EBOPs 不足 → 减小 beta 释放压力
            self.beta_current = max(
                self.beta_current / effective_factor,
                self.beta_min,
            )
        # else: 在容忍带内，保持不变

        # 每 epoch beta 变化倍率限制，防止单步跳变过大
        if self.max_change_ratio > 1.0 and old_beta > 0:
            self.beta_current = float(np.clip(
                self.beta_current,
                old_beta / self.max_change_ratio,
                old_beta * self.max_change_ratio,
            ))

        # ── rescue 投影：持续温和的上行修正 ──────────────────────────────
        # EBOPs 大幅低于目标时，每 epoch 施加一小步 kq.b 放大。
        # 修正强度与偏差成正比（弹簧模型），避免一次性大幅跳变。
        rescue_floor = self.target_ebops * (1.0 - self.rescue_threshold)
        if self._ebops_ema < rescue_floor:
            # 偏差比: 1 - ema/target, e.g. ema=900, target=1500 → deviation=0.4
            deviation = 1.0 - self._ebops_ema / self.target_ebops
            alpha = 1.0 + self.rescue_rate * max(deviation, 0.0)
            alpha = float(np.clip(alpha, 1.0, self.rescue_max_alpha))
            if alpha > 1.001:
                self._rescue_scale_kqb(alpha)
                # rescue 后把 beta 压到最低，避免刚推上来又被压下去
                self.beta_current = self.beta_min
                logs['rescue_alpha'] = alpha

        logs['beta_budget'] = self.beta_current