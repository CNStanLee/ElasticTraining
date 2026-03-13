"""
eBOPs 预算控制 — Beta 调度 + KQB 稳定器 + 激活位宽固定
====================================================

BetaOnlyBudgetController
    通过 beta 调节 + rescue 投影维持 eBOPs 在目标附近

KQBStabilizer
    剪枝后锚定 kq.b 防止初期漂移

ActivationBitsFixer
    固定激活位宽防漂移

ProgressiveBudgetController
    课程式 EBOPs：target 从 warmup_ebops 指数衰减到 final_ebops，
    避免一次性剪枝的冷启动冲击

BetaCurriculumController
    监控 acc 停滞 → beta 退火重启，打破 beta→bk→STE 死锁

AdaptiveLRBiwidthScaler
    低位宽时自动放大 LR 补偿 STE 信噪比下降
"""

from __future__ import annotations

import math

import keras
import numpy as np
import tensorflow as tf

from . import _get_kq_var, _flatten_layers


# ═══════════════════════════════════════════════════════════════════════════════
# 共享小工具
# ═══════════════════════════════════════════════════════════════════════════════

def _set_all_beta(model, value: float):
    """设置模型所有 HGQ 层的 _beta 值。"""
    bv = tf.constant(value, dtype=tf.float32)
    for layer in _flatten_layers(model):
        if hasattr(layer, '_beta'):
            try:
                layer._beta.assign(tf.ones_like(layer._beta) * bv)
            except Exception:
                pass


def _get_active_bk_mean(model, pruned_threshold: float = 0.1) -> float:
    """返回所有活跃（b > threshold）kq.b 的均值。"""
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
    if not all_b:
        return 0.0
    arr = np.array(all_b)
    active = arr[arr > pruned_threshold]
    return float(active.mean()) if len(active) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# BetaOnlyBudgetController — 主预算控制器
# ═══════════════════════════════════════════════════════════════════════════════

class BetaOnlyBudgetController(keras.callbacks.Callback):
    """用 beta + rescue 投影维持 eBOPs 预算。

    下行通道 (beta 调节)：
      eBOPs > target → 增大 beta → 梯度压低 kq.b
      eBOPs < target → 减小 beta → 释放压力

    上行通道 (rescue 投影)：
      eBOPs 大幅低于目标时，直接放大活跃 kq.b (弹簧模型)
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
        self.target_ebops = float(target_ebops)
        self.margin = margin
        self.beta_current = float(beta_init)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.adjust_factor = adjust_factor
        self.ema_alpha = ema_alpha
        self.warmup_epochs = warmup_epochs
        self.max_change_ratio = max_change_ratio
        self._ebops_ema = float(init_ebops) if init_ebops is not None else None
        self._epoch_counter = 0
        self.rescue_threshold = rescue_threshold
        self.rescue_rate = rescue_rate
        self.rescue_max_alpha = rescue_max_alpha
        self.rescue_b_min = rescue_b_min
        self.rescue_b_max = rescue_b_max

    def _set_beta(self, value):
        bv = tf.constant(value, dtype=tf.float32)
        for layer in _flatten_layers(self.model):
            if hasattr(layer, '_beta'):
                try:
                    layer._beta.assign(tf.ones_like(layer._beta) * bv)
                except Exception:
                    pass

    def _rescue_scale_kqb(self, alpha: float):
        """直接放大活跃 kq.b，为 eBOPs 提供上行动力。"""
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
            self._ebops_ema = self.ema_alpha * raw_ebops + (1.0 - self.ema_alpha) * self._ebops_ema

        upper = self.target_ebops * (1.0 + self.margin)
        lower = self.target_ebops * (1.0 - self.margin)
        old_beta = self.beta_current

        # 预热
        if self.warmup_epochs > 0 and self._epoch_counter <= self.warmup_epochs:
            warmup_scale = self._epoch_counter / self.warmup_epochs
        else:
            warmup_scale = 1.0
        effective_factor = 1.0 + (self.adjust_factor - 1.0) * warmup_scale

        if self._ebops_ema > upper:
            self.beta_current = min(self.beta_current * effective_factor, self.beta_max)
        elif self._ebops_ema < lower:
            self.beta_current = max(self.beta_current / effective_factor, self.beta_min)

        # 变化倍率限制
        if self.max_change_ratio > 1.0 and old_beta > 0:
            self.beta_current = float(np.clip(
                self.beta_current,
                old_beta / self.max_change_ratio,
                old_beta * self.max_change_ratio,
            ))

        # rescue 投影
        rescue_floor = self.target_ebops * (1.0 - self.rescue_threshold)
        if self._ebops_ema < rescue_floor:
            deviation = 1.0 - self._ebops_ema / self.target_ebops
            alpha = float(np.clip(
                1.0 + self.rescue_rate * max(deviation, 0.0),
                1.0, self.rescue_max_alpha
            ))
            if alpha > 1.001:
                self._rescue_scale_kqb(alpha)
                self.beta_current = self.beta_min
                logs['rescue_alpha'] = alpha

        logs['beta'] = self.beta_current


# ═══════════════════════════════════════════════════════════════════════════════
# KQBStabilizer — 剪枝后位宽稳定器
# ═══════════════════════════════════════════════════════════════════════════════

class KQBStabilizer(keras.callbacks.Callback):
    """剪枝后锚定 kq.b 防止初期梯度导致 eBOPs 偏离。

    每 epoch 结束后将当前 kq.b 与剪枝快照混合：
      b_new = alpha * b_snapshot + (1-alpha) * b_current

    - hold_epochs 内: alpha = hold_strength (近似冻结)
    - release_epochs 内: alpha 线性衰减到 0
    """

    def __init__(self, hold_epochs: int = 30, release_epochs: int = 100,
                 hold_strength: float = 0.8):
        super().__init__()
        self.hold_epochs = hold_epochs
        self.release_epochs = release_epochs
        self.hold_strength = hold_strength
        self._snapshot = None

    def capture_snapshot(self, model):
        """保存当前 kq.b 作为参考点（剪枝后调用）。"""
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
            b_var.assign((alpha * b_snap + (1.0 - alpha) * current).astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# ActivationBitsFixer — 固定激活位宽
# ═══════════════════════════════════════════════════════════════════════════════

class ActivationBitsFixer(keras.callbacks.Callback):
    """每 epoch 将所有层 aq.b/aq.f 固定为常数，防止激活位宽漂移。"""

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
            target_var.assign(np.ones_like(target_var.numpy(), dtype=np.float32) * self.b_a_fixed)


# ═══════════════════════════════════════════════════════════════════════════════
# SoftDeathFloor — 软死亡下限（防止连接永久死亡）
# ═══════════════════════════════════════════════════════════════════════════════

class SoftDeathFloor(keras.callbacks.Callback):
    """给"死连接"的 kq.b 保持一个极小正值下限，阻止永久死亡。

    HGQ 中 round_conv(b < 0.5) = 0 → 前向输出恒为 0 → 任务梯度为 0
    → b 只受 beta*eBOPs 梯度压低 → 永远无法复活。

    本 callback 在每 epoch 结束后，将 b < alive_threshold 的连接
    钳位到 b_floor，使它们保持微弱信号。

    关键设计：
    - b_floor 极小（默认 0.05），round_conv(0.05) ≈ 0，
      对 eBOPs 影响可忽略 (<0.1% per connection)
    - 但 HGQ 的 STE (straight-through estimator) 仍然传梯度给 b，
      所以只要 beta 下降，b 可以自然回升
    - alive_threshold (默认 0.4) 区分"存活"和"濒死"连接

    注意：对绘图不产生影响（plotting 使用 > 0.4 作为可见阈值）。

    Parameters
    ----------
    b_floor : float
        死连接的 kq.b 下限 (默认 0.05)
    alive_threshold : float
        b >= 此值视为存活，不干预 (默认 0.4)
    apply_every : int
        每 N epoch 执行一次 (默认 1)
    protect_kernel : bool
        是否同步把 kernel 权重从 0 恢复为微小随机值 (默认 True)
    kernel_init_scale : float
        恢复 kernel 时的初始化标准差 (默认 0.01)
    """

    def __init__(
        self,
        b_floor: float = 0.05,
        alive_threshold: float = 0.4,
        apply_every: int = 1,
        protect_kernel: bool = True,
        kernel_init_scale: float = 0.01,
    ):
        super().__init__()
        self.b_floor = float(b_floor)
        self.alive_threshold = float(alive_threshold)
        self.apply_every = max(1, int(apply_every))
        self.protect_kernel = bool(protect_kernel)
        self.kernel_init_scale = float(kernel_init_scale)
        self._n_clamped_last = 0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.apply_every != 0:
            return

        n_clamped = 0
        for layer in _flatten_layers(self.model):
            kq = getattr(layer, 'kq', None)
            if kq is None:
                continue
            b_var = _get_kq_var(kq, 'b')
            if b_var is None:
                b_var = _get_kq_var(kq, 'f')
            if b_var is None:
                continue

            b = b_var.numpy().astype(np.float32)
            # 找出"濒死/已死"连接: 0 <= b < alive_threshold
            needs_floor = (b >= 0.0) & (b < self.alive_threshold)
            if not needs_floor.any():
                continue

            count = int(needs_floor.sum())
            n_clamped += count
            b_new = np.where(needs_floor, self.b_floor, b)
            b_var.assign(b_new.astype(np.float32))

            # 同步恢复 kernel：如果 kernel 为 0 而 b 被恢复，
            # 需要给 kernel 一个微小值否则梯度仍为 0
            if self.protect_kernel:
                kernel = getattr(layer, 'kernel', None)
                if kernel is not None:
                    k = kernel.numpy().astype(np.float32)
                    dead_kernel = needs_floor & (np.abs(k) < 1e-10)
                    if dead_kernel.any():
                        noise = np.random.randn(*k.shape).astype(np.float32) * self.kernel_init_scale
                        k = np.where(dead_kernel, noise, k)
                        kernel.assign(k)

        self._n_clamped_last = n_clamped
        if logs is not None:
            logs['soft_floor_clamped'] = n_clamped


# ═══════════════════════════════════════════════════════════════════════════════
# ProgressiveBudgetController — 渐进式预算衰减
# ═══════════════════════════════════════════════════════════════════════════════
# TopologyRescueCallback — 停滞触发、结构感知、预算中性的拓扑修复
# ═══════════════════════════════════════════════════════════════════════════════

class TopologyRescueCallback(keras.callbacks.Callback):
    """停滞触发、结构感知、预算中性的拓扑修复。

    SoftDeathFloor 的问题:
      - 无条件周期性执行，不管网络是否需要帮助
      - 把所有濒死连接钳位到 b_floor=0.05 (< 0.5)，前向仍为 0，
        只靠 STE 传梯度，但在如此低 b 值下 STE 噪声极大
      - 与 beta 优化器形成对抗循环: beta 压低 b → SDF 抬高 b → beta 更用力压低
      - 实验 D 证明: SDF 越频繁/越激进，精度越低

    TopologyRescue 的改进:
      1. 停滞触发: 只在 val_accuracy 停滞 stall_patience epoch 后才干预
      2. 结构感知: 只复活满足谱条件（节点度不足）的死连接
      3. 有效复活: revival_b_val 设为有意义的位宽 (如 1.0)，
         round_conv(1.0) > 0 → 前向有信号 → 梯度有效
      4. 预算中性: 复活时同时杀掉等量最弱活跃连接 (swap-kill)，
         不增加 eBOPs → 不与 beta 调度器冲突
      5. 衰减冷却: 每次干预后长 cool_down，干预强度随次数递减

    本质是"拓扑扰动"策略:
      - 正常训练时完全不干预，让优化器自由塑造拓扑
      - 陷入停滞时，通过局部 swap 探索替代拓扑
      - 优化器在新拓扑上重新优化，可能逃逸局部最优

    Parameters
    ----------
    revival_b_val : float
        复活连接的目标位宽 (默认 1.0 = 有效 1-bit)
    check_interval : int
        每 N epoch 检查是否需要干预
    stall_patience : int
        val_accuracy 停滞多少 epoch 后触发
    min_delta : float
        判定改善的最小增量
    max_swap_per_layer : int
        每次每层最多 swap 多少连接
    swap_budget_neutral : bool
        是否 swap-kill (复活+杀死等量弱连接)
    min_degree : int
        谱条件: 节点最小入度/出度
    cool_down : int
        两次干预的最小间隔 epoch
    max_interventions : int
        总干预上限
    alive_threshold : float
        b < 此值视为死连接
    decay_factor : float
        每次干预后 max_swap 乘以此因子 (衰减)
    kernel_init_scale : float
        复活 kernel 为 0 时的初始化标准差
    """

    def __init__(
        self,
        revival_b_val: float = 1.0,
        check_interval: int = 50,
        stall_patience: int = 300,
        min_delta: float = 5e-5,
        max_swap_per_layer: int = 4,
        swap_budget_neutral: bool = True,
        min_degree: int = 2,
        cool_down: int = 200,
        max_interventions: int = 10,
        alive_threshold: float = 0.5,
        decay_factor: float = 0.85,
        kernel_init_scale: float = 0.01,
    ):
        super().__init__()
        self.revival_b_val = float(revival_b_val)
        self.check_interval = max(1, int(check_interval))
        self.stall_patience = int(stall_patience)
        self.min_delta = float(min_delta)
        self.max_swap_per_layer = int(max_swap_per_layer)
        self.swap_budget_neutral = bool(swap_budget_neutral)
        self.min_degree = int(min_degree)
        self.cool_down = int(cool_down)
        self.max_interventions = int(max_interventions)
        self.alive_threshold = float(alive_threshold)
        self.decay_factor = float(decay_factor)
        self.kernel_init_scale = float(kernel_init_scale)

        self._best_acc = -1.0
        self._stall_counter = 0
        self._last_intervention_epoch = -cool_down
        self._n_interventions = 0
        self._current_max_swap = float(max_swap_per_layer)
        self._total_revived = 0
        self._total_killed = 0

    def _spectral_candidates(self, b_arr: np.ndarray) -> list[tuple[int, int]]:
        """找出因入度/出度不足需复活的死连接候选 (row, col)。"""
        if b_arr.ndim == 1:
            return []
        if b_arr.ndim > 2:
            b_arr = b_arr.reshape(b_arr.shape[0], -1)

        dead = (b_arr < self.alive_threshold)
        active = ~dead
        out_degree = active.sum(axis=1)   # 每行 = input node 出度
        in_degree = active.sum(axis=0)    # 每列 = output node 入度

        candidates = set()
        # 找出度不足的 input 节点，其死连接是候选
        for r in np.where(out_degree < self.min_degree)[0]:
            for c in np.where(dead[r])[0]:
                candidates.add((int(r), int(c)))
        # 找入度不足的 output 节点
        for c in np.where(in_degree < self.min_degree)[0]:
            for r in np.where(dead[:, c])[0]:
                candidates.add((int(r), int(c)))
        return list(candidates)

    def _rank_by_kernel_magnitude(self, layer, candidates, orig_shape):
        """用 |kernel| 为复活候选排序 (大 → 有潜力)。"""
        kernel = getattr(layer, 'kernel', None)
        scored = []
        if kernel is not None:
            try:
                w = np.abs(kernel.numpy())
                w_2d = w.reshape(w.shape[0], -1) if w.ndim > 2 else w
                for (r, c) in candidates:
                    if r < w_2d.shape[0] and c < w_2d.shape[1]:
                        scored.append((float(w_2d[r, c]), r, c))
                    else:
                        scored.append((0.0, r, c))
            except Exception:
                scored = [(np.random.rand(), r, c) for (r, c) in candidates]
        else:
            scored = [(np.random.rand(), r, c) for (r, c) in candidates]
        scored.sort(reverse=True)
        return scored

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = float(logs.get('val_accuracy', logs.get('val_acc', -1.0)))
        if val_acc < 0:
            return

        # ── 停滞检测 ────────────────────────────────────────────────
        if val_acc > self._best_acc + self.min_delta:
            self._best_acc = val_acc
            self._stall_counter = 0
        else:
            self._stall_counter += 1

        # ── 检查触发条件 ───────────────────────────────────────────
        if self._stall_counter < self.stall_patience:
            return
        if (epoch + 1) % self.check_interval != 0:
            return
        if (epoch - self._last_intervention_epoch) < self.cool_down:
            return
        if self._n_interventions >= self.max_interventions:
            return

        effective_max_swap = max(1, int(self._current_max_swap))

        # ── 遍历各层执行拓扑修复 ─────────────────────────────────
        total_revived = 0
        total_killed = 0

        for layer in _flatten_layers(self.model):
            kq = getattr(layer, 'kq', None)
            if kq is None or not hasattr(layer, 'kernel'):
                continue
            b_var = _get_kq_var(kq, 'b')
            if b_var is None:
                b_var = _get_kq_var(kq, 'f')
            if b_var is None:
                continue

            b_arr = b_var.numpy()
            orig_shape = b_arr.shape
            b_2d = b_arr.reshape(b_arr.shape[0], -1) if b_arr.ndim > 2 else b_arr

            # 谱条件查找候选
            candidates = self._spectral_candidates(b_2d)
            if not candidates:
                continue

            # 按 |kernel| 排序
            ranked = self._rank_by_kernel_magnitude(layer, candidates, orig_shape)
            top_k = ranked[:effective_max_swap]
            if not top_k:
                continue

            # ── Swap-kill: 找等量最弱活跃连接 ──────────────────
            kill_set = []
            if self.swap_budget_neutral:
                revive_set = {(r, c) for (_, r, c) in top_k}
                alive_conns = []
                for r in range(b_2d.shape[0]):
                    for c in range(b_2d.shape[1]):
                        if b_2d[r, c] >= self.alive_threshold and (r, c) not in revive_set:
                            alive_conns.append((float(b_2d[r, c]), r, c))
                alive_conns.sort()  # 按 b 从小到大 → 最弱在前
                kill_set = [(r, c) for (_, r, c) in alive_conns[:len(top_k)]]

                # 杀死
                for (r, c) in kill_set:
                    b_2d[r, c] = 0.0

            # ── 复活 ──────────────────────────────────────────────
            for (score, r, c) in top_k:
                b_2d[r, c] = self.revival_b_val

            # 写回
            if b_arr.ndim > 2:
                b_new = b_2d.reshape(orig_shape)
            else:
                b_new = b_2d
            b_var.assign(b_new.astype(np.float32))

            # 同步恢复 kernel
            kernel = getattr(layer, 'kernel', None)
            if kernel is not None:
                k = kernel.numpy().astype(np.float32)
                k_2d = k.reshape(k.shape[0], -1) if k.ndim > 2 else k
                for (score, r, c) in top_k:
                    if r < k_2d.shape[0] and c < k_2d.shape[1]:
                        if abs(k_2d[r, c]) < 1e-10:
                            k_2d[r, c] = np.random.randn() * self.kernel_init_scale
                if k.ndim > 2:
                    k = k_2d.reshape(k.shape)
                else:
                    k = k_2d
                kernel.assign(k.astype(np.float32))

            total_revived += len(top_k)
            total_killed += len(kill_set)

            lname = getattr(layer, 'name', str(id(layer)))
            print(f'  [TopoRescue] {lname}: revived={len(top_k)} killed={len(kill_set)} '
                  f'(candidates={len(candidates)}, '
                  f'top |w|={top_k[0][0]:.3e}..{top_k[-1][0]:.3e})')

        # ── 更新状态 ─────────────────────────────────────────────
        if total_revived > 0:
            self._last_intervention_epoch = epoch
            self._n_interventions += 1
            self._total_revived += total_revived
            self._total_killed += total_killed
            self._current_max_swap *= self.decay_factor  # 衰减
            self._stall_counter = 0  # 重置停滞计数器

            print(f'  [TopoRescue] epoch={epoch}  intervention #{self._n_interventions}'
                  f'/{self.max_interventions}  revived={total_revived} killed={total_killed}'
                  f'  total_revived={self._total_revived}  stall_was={self.stall_patience}ep'
                  f'  next_max_swap={max(1, int(self._current_max_swap))}')

            logs['topo_rescue_revived'] = total_revived
            logs['topo_rescue_killed'] = total_killed
            logs['topo_rescue_interventions'] = self._n_interventions


# ═══════════════════════════════════════════════════════════════════════════════

class ProgressiveBudgetController(keras.callbacks.Callback):
    """课程式 EBOPs：target 从 warmup_ebops 指数衰减到 final_ebops。

    避免从高 eBOPs baseline 一次性剪枝到极低目标造成的冷启动量化冲击
    （几乎所有 b_k 被强制到极低值，STE 完全失效）。

    搭配 BetaOnlyBudgetController 使用：每 epoch 更新其 target_ebops，
    让 beta 平滑施压，模型自然降低 b_k。

    Parameters
    ----------
    budget_ctrl : BetaOnlyBudgetController
        被管理的预算控制器实例
    warmup_ebops : float
        初始（宽松）eBOPs 目标
    final_ebops : float
        最终 eBOPs 目标
    decay_epochs : int
        从 warmup 衰减到 final 的 epoch 数（指数衰减 = 对数空间线性）
    start_epoch : int
        从哪个 epoch 开始衰减
    """

    def __init__(
        self,
        budget_ctrl: BetaOnlyBudgetController,
        warmup_ebops: float,
        final_ebops: float,
        decay_epochs: int,
        start_epoch: int = 0,
    ):
        super().__init__()
        self.budget_ctrl = budget_ctrl
        self.warmup_ebops = float(warmup_ebops)
        self.final_ebops = float(final_ebops)
        self.decay_epochs = max(1, int(decay_epochs))
        self.start_epoch = int(start_epoch)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        t = min((epoch - self.start_epoch) / self.decay_epochs, 1.0)
        # 指数插值（对数空间线性插值）
        log_w = math.log(self.warmup_ebops)
        log_f = math.log(self.final_ebops)
        current_target = math.exp(log_w + t * (log_f - log_w))
        self.budget_ctrl.target_ebops = current_target


# ═══════════════════════════════════════════════════════════════════════════════
# BetaCurriculumController — Beta 退火重启
# ═══════════════════════════════════════════════════════════════════════════════

class BetaCurriculumController(keras.callbacks.Callback):
    """监控 acc 停滞，自动进行 beta 退火重启。

    状态机
    ------
    COMPRESS  → 正常运行，BetaOnlyBudgetController 控制 beta
    RECOVER   → acc 停滞 stall_patience epoch → beta=0 让 acc 先恢复
    RESTART   → recover_epochs 后重置 beta = beta_at_stall * restart_decay

    这打破了 beta↑ → b_k↓ → STE 失效 → acc 停 → beta 不变 的正反馈死锁。

    Parameters
    ----------
    budget_ctrl : BetaOnlyBudgetController
        被管理的预算控制器
    stall_patience : int
        acc 停滞多少 epoch 后触发 RECOVER
    recover_epochs : int
        RECOVER 阶段持续多少 epoch (beta=0)
    min_delta : float
        acc 增长判定阈值
    restart_decay : float
        重启 beta 缩放因子
    max_restarts : int
        最多重启次数
    """

    STATE_COMPRESS = 'COMPRESS'
    STATE_RECOVER = 'RECOVER'

    def __init__(
        self,
        budget_ctrl: BetaOnlyBudgetController,
        stall_patience: int = 800,
        recover_epochs: int = 400,
        min_delta: float = 5e-5,
        restart_decay: float = 0.3,
        max_restarts: int = 6,
        recover_beta_floor: float = None,
    ):
        super().__init__()
        self.budget_ctrl = budget_ctrl
        self.stall_patience = stall_patience
        self.recover_epochs = recover_epochs
        self.min_delta = min_delta
        self.restart_decay = restart_decay
        self.max_restarts = max_restarts
        # RECOVER 期间 beta 最低值: 不完全归零, 防止 eBOPs 爆炸
        # 默认 = None → 使用 budget_ctrl.beta_min (向后兼容)
        self.recover_beta_floor = recover_beta_floor

        self._state = self.STATE_COMPRESS
        self._best_acc = -1.0
        self._stall_counter = 0
        self._recover_counter = 0
        self._n_restarts = 0
        self._beta_at_stall = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = float(logs.get('val_accuracy', logs.get('val_acc', -1.0)))
        if val_acc < 0:
            return

        if self._state == self.STATE_COMPRESS:
            if val_acc > self._best_acc + self.min_delta:
                self._best_acc = val_acc
                self._stall_counter = 0
            else:
                self._stall_counter += 1

            if (self._stall_counter >= self.stall_patience
                    and self._n_restarts < self.max_restarts):
                self._beta_at_stall = self.budget_ctrl.beta_current
                # RECOVER beta: 不完全归零, 保留最小压力防止 eBOPs 爆炸
                floor = self.recover_beta_floor
                if floor is None:
                    floor = self.budget_ctrl.beta_min
                print(f'\n  [BetaCurriculum] epoch={epoch}  STALL detected '
                      f'({self._stall_counter} ep)  '
                      f'beta={self._beta_at_stall:.2e}  → RECOVER phase '
                      f'(floor={floor:.2e})')
                self._state = self.STATE_RECOVER
                self._recover_counter = 0
                self._stall_counter = 0
                _set_all_beta(self.model, floor)
                self.budget_ctrl.beta_current = floor

        elif self._state == self.STATE_RECOVER:
            self._recover_counter += 1
            if val_acc > self._best_acc + self.min_delta:
                self._best_acc = val_acc

            # RECOVER 期间仍维持 floor beta, 防止 budget_ctrl 将其调为 0
            floor = self.recover_beta_floor
            if floor is None:
                floor = self.budget_ctrl.beta_min
            if self.budget_ctrl.beta_current < floor:
                self.budget_ctrl.beta_current = floor
                _set_all_beta(self.model, floor)

            if self._recover_counter >= self.recover_epochs:
                beta_new = self._beta_at_stall * self.restart_decay
                beta_new = max(beta_new, self.budget_ctrl.beta_min)
                self._n_restarts += 1
                print(f'\n  [BetaCurriculum] epoch={epoch}  RECOVER done  '
                      f'best_acc={self._best_acc:.4f}  '
                      f'restart beta: {self._beta_at_stall:.2e} → {beta_new:.2e}  '
                      f'(restart #{self._n_restarts}/{self.max_restarts})')
                self.budget_ctrl.beta_current = beta_new
                self.budget_ctrl._ebops_ema = None
                _set_all_beta(self.model, beta_new)
                self._state = self.STATE_COMPRESS
                self._stall_counter = 0

        logs['beta_curriculum_restarts'] = float(self._n_restarts)


# ═══════════════════════════════════════════════════════════════════════════════
# AdaptiveLRBiwidthScaler — 自适应 LR 缩放
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveLRBiwidthScaler(keras.callbacks.Callback):
    """当 mean b_k 低于阈值时，自动放大当前 LR。

    STE 有效信噪比 ∝ 2^b_k。b_k 低时梯度更新噪声大，
    需要更大 LR 才能越过量化边界。

    缩放公式:
        lr_effective = lr_base * max(1, (bk_threshold / mean_bk)^scale_power)

    Parameters
    ----------
    bk_threshold : float
        触发 LR 放大的位宽阈值
    scale_power : float
        缩放指数
    lr_max_factor : float
        LR 最大放大倍数
    """

    def __init__(
        self,
        bk_threshold: float = 2.0,
        scale_power: float = 0.5,
        lr_max_factor: float = 5.0,
        pruned_threshold: float = 0.1,
        log: bool = False,
    ):
        super().__init__()
        self.bk_threshold = bk_threshold
        self.scale_power = scale_power
        self.lr_max_factor = lr_max_factor
        self.pruned_threshold = pruned_threshold
        self.log = log

    @staticmethod
    def _get_lr(opt) -> float:
        lr = opt.learning_rate
        try:
            return float(lr.numpy())
        except Exception:
            return float(lr)

    @staticmethod
    def _set_lr(opt, value: float):
        try:
            opt.learning_rate.assign(float(value))
        except Exception:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        mean_bk = _get_active_bk_mean(self.model, self.pruned_threshold)
        factor = max(1.0, (self.bk_threshold / max(mean_bk, 0.1)) ** self.scale_power)
        factor = min(factor, self.lr_max_factor)

        opt = self.model.optimizer
        base_lr = self._get_lr(opt)
        new_lr = base_lr * factor
        self._set_lr(opt, new_lr)

        if self.log:
            print(f'  [AdaptiveLR] epoch={epoch}  '
                  f'mean_bk={mean_bk:.3f}  factor={factor:.2f}  lr={new_lr:.2e}')


# ═══════════════════════════════════════════════════════════════════════════════
# EBOPsConstantProjector — 直接投影维持 eBOPs (极低预算必备)
# ═══════════════════════════════════════════════════════════════════════════════

class EBOPsConstantProjector(keras.callbacks.Callback):
    """每 epoch 结束后将全部活跃 kq.b 等比缩放，使总 eBOPs 维持在 target。

    设计用于极低预算场景 (ebops ≤ 500)，替代 beta 调控:
      - beta 在 1-bit 下完全失效 (STE 梯度噪声 >> 任务梯度)
      - 直接投影无需 beta，避免连接被压死

    振荡抑制:
      1. 幂次阻尼: α = (target/current)^gamma, gamma<1 做部分修正
      2. alpha 限幅: clip(α, alpha_min, alpha_max)
      3. EMA 平滑: 减少噪声驱动的过调

    Parameters
    ----------
    target_ebops : float
        目标 eBOPs
    b_k_min : float
        kq.b 下限 (默认 0.5，保证至少 1-bit)
    b_k_max : float
        kq.b 上限
    pruned_threshold : float
        kq.b ≤ 该阈值视为剪掉，不参与缩放
    alpha_gamma : float
        幂次阻尼指数，越小越保守 (默认 0.5)
    alpha_min, alpha_max : float
        alpha 限幅
    ema_alpha : float
        eBOPs EMA 平滑系数
    """

    def __init__(
        self,
        target_ebops: float,
        b_k_min: float = 0.5,
        b_k_max: float = 8.0,
        pruned_threshold: float = 0.1,
        start_epoch: int = 0,
        alpha_gamma: float = 0.5,
        alpha_min: float = 0.80,
        alpha_max: float = 1.25,
        ema_alpha: float = 0.3,
        project_activation: bool = False,
        log_scale: bool = False,
    ):
        super().__init__()
        self.target_ebops = float(target_ebops)
        self.b_k_min = b_k_min
        self.b_k_max = b_k_max
        self.pruned_threshold = pruned_threshold
        self.start_epoch = start_epoch
        self.alpha_gamma = alpha_gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ema_alpha = ema_alpha
        self.project_activation = project_activation
        self.log_scale = log_scale
        self._ebops_ema = None

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return

        logs = logs or {}
        raw_ebops = float(logs.get('ebops', float('nan')))
        if not math.isfinite(raw_ebops) or raw_ebops <= 0:
            return

        if self._ebops_ema is None:
            self._ebops_ema = raw_ebops
        else:
            self._ebops_ema = (self.ema_alpha * raw_ebops
                               + (1.0 - self.ema_alpha) * self._ebops_ema)
        current_ebops = self._ebops_ema

        raw_alpha = self.target_ebops / current_ebops
        alpha = raw_alpha ** self.alpha_gamma
        alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))

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
            active_mask = (b_arr > self.pruned_threshold).astype(np.float32)
            b_new = np.where(
                active_mask > 0,
                np.clip(b_arr * alpha, self.b_k_min, self.b_k_max),
                b_arr,
            )
            b_var.assign(b_new.astype(np.float32))

            if self.project_activation:
                aq = getattr(layer, 'aq', None)
                if aq is not None:
                    ab_var = _get_kq_var(aq, 'b')
                    if ab_var is None:
                        ab_var = _get_kq_var(aq, 'f')
                    if ab_var is not None:
                        a_arr = ab_var.numpy()
                        a_new = np.clip(a_arr * alpha, self.b_k_min, self.b_k_max)
                        ab_var.assign(a_new.astype(np.float32))

        if self.log_scale:
            print(f'  [Projector] ep={epoch}  raw={raw_ebops:.0f}  '
                  f'ema={current_ebops:.0f}  target={self.target_ebops:.0f}  '
                  f'α={alpha:.4f}')
