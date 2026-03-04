# low_budget_utils.py
"""
极低预算训练工具集（目标 EBOPs <= 500）

核心问题：β 过大 → b_k 被压到 1–2 bit → STE 梯度噪声剧增 → acc 不涨 →
          EBOPs 无法收敛 → β 继续增大 → 死循环。

解决策略（三条主线）
====================
1. Ramanujan 稀疏角度
   - TopologyWarmupCallback:  从高 degree 出发，随训练进度逐步按梯度剪边，
                               避免随机低度初始化造成的秩塌缩。
   - EdgeRewiringCallback:    周期性对死亡神经元（梯度范数极小）重采样新边，
                               同时维持 d-regular 结构与谱间隙。

2. 谱约束角度
   - MinSingularValueRegularizer: 通过幂迭代估计 σ_min，在训练 loss 中加惩罚，
                                   防止活跃权重子矩阵秩塌缩。
   - SpectralLossCallback:        在 on_train_batch_begin 前把惩罚暴露为 metric，
                                   便于监控。

3. STE / β 优化角度
   - BetaCurriculumController:   监控 acc 停滞 → 临时归零 β 让 acc 恢复 →
                                  从较小 β 重新启动压缩（打破正反馈）。
   - AdaptiveLRBiwidthScaler:    当 mean b_k 低于阈值时自动升高 LR，对抗 STE
                                  信噪比下降。
   - ProgressiveBudgetScheduler: 课程式学习：target 从暖身值逐步衰减到最终值，
                                  避免量化冷启动冲击。
"""

import math
import numpy as np
import tensorflow as tf
import keras

from .ramanujan_budget_utils import (
    _flatten_layers,
    _get_kq_var,
    _ramanujan_like_mask,
)


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────────────



def _get_active_bk_mean(model: keras.Model, pruned_threshold: float = 0.1) -> float:
    """返回全模型活跃连接的平均 kernel 位宽。"""
    all_b = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None:
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            arr = b_var.numpy().ravel().astype(np.float32)
            all_b.extend(arr[arr > pruned_threshold].tolist())
    return float(np.mean(all_b)) if all_b else 4.0


def _set_all_beta(model: keras.Model, beta_value: float):
    bv = tf.constant(beta_value, dtype=tf.float32)
    for layer in _flatten_layers(model):
        if hasattr(layer, '_beta'):
            try:
                layer._beta.assign(tf.ones_like(layer._beta) * bv)
            except Exception:
                pass


def _power_iteration_min_sv(w: np.ndarray, n_iter: int = 5) -> float:
    """用幂迭代近似计算矩阵 w 的最小奇异值（= (W^T W) 最小特征值的平方根）。

    注意：这是最小奇异值的下界近似，用于判断是否出现秩塌缩。
    精确计算需 SVD，代价过高；幂迭代仅需若干次矩阵乘。
    始终操作 tall 形式（rows >= cols），使 Gram 矩阵维度与 eye 维度一致。
    """
    if w.ndim > 2:
        w = w.reshape(-1, w.shape[-1])
    # 确保 tall (rows >= cols)
    if w.shape[0] < w.shape[1]:
        w = w.T
    m, n = w.shape   # m >= n
    ww = w.T @ w                          # (n, n)
    lam_max = float(np.linalg.norm(ww, 'fro') / max(n, 1)) + 1e-6
    ww_shifted = lam_max * np.eye(n, dtype=np.float32) - ww.astype(np.float32)
    v = np.random.randn(n).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(n_iter):
        v = ww_shifted @ v
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            break
        v /= norm
    rayleigh = float(v @ ww.astype(np.float32) @ v)
    return max(rayleigh, 0.0) ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Ramanujan 稀疏 — 拓扑热身 + 边重连
# ─────────────────────────────────────────────────────────────────────────────

class TopologyWarmupCallback(keras.callbacks.Callback):
    """从高度出发按梯度逐步降低度，保护早期梯度流。

    原理
    ----
    - 初始 degree = degree_start（较高，确保梯度流）
    - 每 step_interval 个 epoch 将每层度下降 1，直到 degree_final
    - 删边策略：每次删掉梯度绝对值最小的连接（而非随机）
    - 始终维持 d-regular 结构（每列恰好 d 条活跃边）

    参数
    ----
    degree_start    : 起始度（默认 max(sqrt(N_in)*3, min_degree)）
    degree_final    : 最终目标度（应与 ramanujan_budget_utils 计算的 d_l 一致）
    step_interval   : 每几个 epoch 减度一次（默认 200）
    min_degree      : 绝对下限，保证 Ramanujan 谱条件（默认 2）
    """

    def __init__(
        self,
        per_layer_degree_final: dict,      # {layer_name: d_final}
        degree_warmup_mul: float = 3.0,    # 初始度 = d_final * warmup_mul（上限 N_in）
        step_interval: int = 200,
        min_degree: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.degree_final     = per_layer_degree_final
        self.warmup_mul       = degree_warmup_mul
        self.step_interval    = step_interval
        self.min_degree       = min_degree
        self._rng             = np.random.RandomState(seed)
        self._current_degree  = {}   # {layer_name: current d}
        self._masks           = {}   # {layer_name: np.ndarray mask}

    # ── 初始化 ──────────────────────────────────────────────────────────────
    def on_train_begin(self, logs=None):
        for layer in _flatten_layers(self.model):
            if getattr(layer, 'kq', None) is None or not hasattr(layer, 'kernel'):
                continue
            name = layer.name
            if name not in self.degree_final:
                continue
            d_final = self.degree_final[name]
            shape   = layer.kernel.shape

            if len(shape) == 2:
                in_dim = int(shape[0])
            elif len(shape) == 4:
                in_dim = int(shape[2])
            else:
                continue

            d_start = min(int(round(d_final * self.warmup_mul)), in_dim)
            d_start = max(d_start, d_final)
            self._current_degree[name] = d_start
            mask = _ramanujan_like_mask(shape, d_start, self._rng)
            self._masks[name] = mask

            # 将零权重初始化为 ramanujan mask 掩盖
            w = layer.kernel.numpy()
            layer.kernel.assign((w * mask).astype(np.float32))

            # 设置 b_k：被剪连接 → 0，活跃连接保持现有值
            kq = layer.kq
            b_var = _get_kq_var(kq, 'b')
            if b_var is None:
                b_var = _get_kq_var(kq, 'f')
            if b_var is not None:
                b_arr = b_var.numpy()
                b_var.assign((b_arr * mask).astype(np.float32))

            layer.ramanujan_mask = tf.constant(mask, dtype=tf.float32)

    # ── 每 step_interval epoch 剪一次 ─────────────────────────────────────
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or epoch % self.step_interval != 0:
            return
        for layer in _flatten_layers(self.model):
            if getattr(layer, 'kq', None) is None or not hasattr(layer, 'kernel'):
                continue
            name = layer.name
            if name not in self._current_degree:
                continue
            d_cur   = self._current_degree[name]
            d_final = self.degree_final[name]
            if d_cur <= max(d_final, self.min_degree):
                continue

            # 降度 1
            d_new = d_cur - 1
            self._current_degree[name] = d_new
            self._prune_weakest_edges(layer, d_new)

    def _prune_weakest_edges(self, layer, new_degree: int):
        """对每个输出神经元：从活跃边中删掉梯度最小的，直到度 = new_degree。"""
        w = layer.kernel.numpy()
        mask = self._masks[layer.name].copy()

        if w.ndim == 2:
            in_dim, out_dim = w.shape
            for o in range(out_dim):
                active_idx = np.where(mask[:, o] > 0.5)[0]
                if len(active_idx) <= new_degree:
                    continue
                # 按权重绝对值（梯度代理）升序排列，删最弱的
                vals = np.abs(w[active_idx, o])
                n_prune = len(active_idx) - new_degree
                prune_pos = np.argsort(vals)[:n_prune]
                mask[active_idx[prune_pos], o] = 0.0

            # 应用 mask
            layer.kernel.assign((w * mask).astype(np.float32))
            kq = layer.kq
            b_var = _get_kq_var(kq, 'b')
            if b_var is None:
                b_var = _get_kq_var(kq, 'f')
            if b_var is not None:
                b_arr = b_var.numpy()
                b_var.assign((b_arr * mask).astype(np.float32))

        elif w.ndim == 4:
            kh, kw, in_ch, out_ch = w.shape
            for o in range(out_ch):
                active_idx = np.where(mask[0, 0, :, o] > 0.5)[0]
                if len(active_idx) <= new_degree:
                    continue
                vals = np.abs(w[:, :, active_idx, o]).mean(axis=(0, 1))
                n_prune = len(active_idx) - new_degree
                prune_pos = np.argsort(vals)[:n_prune]
                mask[:, :, active_idx[prune_pos], o] = 0.0

            layer.kernel.assign((w * mask).astype(np.float32))
            kq = layer.kq
            b_var = _get_kq_var(kq, 'b')
            if b_var is None:
                b_var = _get_kq_var(kq, 'f')
            if b_var is not None:
                b_arr = b_var.numpy()
                b_var.assign((b_arr * mask).astype(np.float32))

        self._masks[layer.name] = mask
        layer.ramanujan_mask = tf.constant(mask, dtype=tf.float32)


class EdgeRewiringCallback(keras.callbacks.Callback):
    """周期性对死亡神经元重采样新边，打破梯度阻断。

    死亡判定：某输出神经元所有输入权重的绝对值均 < dead_threshold。
    重连策略：用目前未连接（mask=0）的输入中随机选一条替换最弱边。
    不变量：维持 d-regular 结构（每列恰好 d 条活跃边）。
    """

    def __init__(
        self,
        per_layer_degree: dict,       # 当前度（动态更新若有 TopologyWarmup）
        rewire_interval: int = 500,
        dead_threshold: float = 0.01,
        seed: int = 99,
    ):
        super().__init__()
        self.per_layer_degree = per_layer_degree
        self.rewire_interval  = rewire_interval
        self.dead_threshold   = dead_threshold
        self._rng             = np.random.RandomState(seed)

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or epoch % self.rewire_interval != 0:
            return
        n_rewired = 0
        for layer in _flatten_layers(self.model):
            if not hasattr(layer, 'ramanujan_mask'):
                continue
            if not hasattr(layer, 'kernel'):
                continue
            w    = layer.kernel.numpy()
            mask = layer.ramanujan_mask.numpy()
            d    = self.per_layer_degree.get(layer.name, 2)
            n_rewired += self._rewire_layer(layer, w, mask, d)
        if n_rewired > 0:
            print(f'  [EdgeRewiring] epoch={epoch}  rewired={n_rewired} edges')

    def _rewire_layer(self, layer, w: np.ndarray, mask: np.ndarray, d: int) -> int:
        if w.ndim != 2:
            return 0   # 仅处理 Dense 层（Conv 扩展略）
        in_dim, out_dim = w.shape
        mask = mask.copy()
        n_rewired = 0
        for o in range(out_dim):
            active_idx = np.where(mask[:, o] > 0.5)[0]
            if len(active_idx) == 0:
                continue
            abs_w = np.abs(w[active_idx, o])
            if abs_w.max() >= self.dead_threshold:
                continue
            # 死亡神经元：替换最弱的边
            inactive_idx = np.where(mask[:, o] < 0.5)[0]
            if len(inactive_idx) == 0:
                continue
            weakest = active_idx[np.argmin(abs_w)]
            new_src = self._rng.choice(inactive_idx)
            mask[weakest, o] = 0.0
            mask[new_src,  o] = 1.0
            # 新权重用 He 初始化
            w[weakest, o] = 0.0
            w[new_src,  o] = float(self._rng.randn() * math.sqrt(2.0 / in_dim))
            n_rewired += 1

        layer.kernel.assign(w.astype(np.float32))
        layer.ramanujan_mask = tf.constant(mask, dtype=tf.float32)
        return n_rewired


# ─────────────────────────────────────────────────────────────────────────────
# 2.  谱约束 — 最小奇异值正则化
# ─────────────────────────────────────────────────────────────────────────────

class SpectralRegularizationCallback(keras.callbacks.Callback):
    """每 `check_interval` epoch 诊断每层最小奇异值，
    自动调整 Keras 层的 kernel_regularizer 强度（不修改 HGQ 量化器）。

    实现方案：直接对权重做就地惩罚更新（additive gradient step），
    避免修改计算图：
        W ← W + lr_spec * max(0, σ_min_target - σ_min(W)) * u * v^T
    其中 (u, v) 是最小奇异向量（通过幂迭代估计）。
    """

    def __init__(
        self,
        sigma_min_target: float = 0.01,
        lr_spec: float = 1e-4,
        check_interval: int = 100,
        pruned_threshold: float = 0.1,
    ):
        super().__init__()
        self.sigma_min_target  = sigma_min_target
        self.lr_spec           = lr_spec
        self.check_interval    = check_interval
        self.pruned_threshold  = pruned_threshold

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.check_interval != 0:
            return

        total_penalty = 0.0
        n_layers = 0

        for layer in _flatten_layers(self.model):
            kq = getattr(layer, 'kq', None)
            if kq is None or not hasattr(layer, 'kernel'):
                continue

            w = layer.kernel.numpy()

            # 仅对活跃子矩阵操作
            mask = getattr(layer, 'ramanujan_mask', None)
            if mask is not None:
                mask_np = mask.numpy()
                w_active = w * mask_np
            else:
                w_active = w

            # 展平为 2D
            orig_shape = w_active.shape
            if w_active.ndim > 2:
                w_2d = w_active.reshape(-1, w_active.shape[-1])
            else:
                w_2d = w_active

            sigma_min = _power_iteration_min_sv(w_2d)
            deficit = max(0.0, self.sigma_min_target - sigma_min)

            if deficit > 1e-8:
                # 始终操作 tall 形式，保证维度一致
                transposed = w_2d.shape[0] < w_2d.shape[1]
                w_tall = w_2d.T if transposed else w_2d        # (m_t, n_t), m_t >= n_t
                m_t, n_t = w_tall.shape
                ww = w_tall.T @ w_tall                         # (n_t, n_t)
                lam_max = float(np.linalg.norm(ww, 'fro') + 1e-6)
                v = np.random.randn(n_t).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-12
                shifted = lam_max * np.eye(n_t, dtype=np.float32) - ww.astype(np.float32)
                for _ in range(8):
                    v = shifted @ v
                    v /= np.linalg.norm(v) + 1e-12
                u = (w_tall @ v).astype(np.float32)            # (m_t,)
                u /= np.linalg.norm(u) + 1e-12

                # 梯度步：沿最小奇异向量方向"推开"
                delta = self.lr_spec * deficit * np.outer(u, v)  # (m_t, n_t)
                w_tall_new = (w_tall + delta)                    # (m_t, n_t)
                w_2d_new   = w_tall_new.T if transposed else w_tall_new
                if w_2d_new.shape == w_2d.shape:
                    w_updated = w_2d_new.reshape(orig_shape)
                    if mask is not None:
                        w_updated = w_updated * mask_np
                    layer.kernel.assign(w_updated.astype(np.float32))

                total_penalty += deficit
                n_layers += 1

        if n_layers > 0:
            print(f'  [SpectralReg] epoch={epoch}  '
                  f'n_layers_penalized={n_layers}  '
                  f'total_deficit={total_penalty:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# 3-A.  β 课程控制器（打破正反馈死循环）
# ─────────────────────────────────────────────────────────────────────────────

class BetaCurriculumController(keras.callbacks.Callback):
    """监控 acc 停滞，自动进行 β 退火重启（β Curriculum with Panic Recovery）。

    状态机
    ------
    COMPRESS  → 正常运行，BetaOnlyBudgetController 控制 β
    RECOVER   → acc 停滞 stall_patience epoch → β=0，让 acc 先爬升
    RESTART   → recover_epochs 后 → β 重置为 beta_restart，重启压缩

    关键参数
    --------
    stall_patience   : acc 在此 epoch 数内不增长 min_delta → 触发 RECOVER
    recover_epochs   : RECOVER 阶段持续 epoch 数
    beta_restart     : 退火后重启 β 值（比停滞时的 β 小 restart_decay 倍）
    restart_decay    : β 重启时缩放因子（默认 0.3）
    max_restarts     : 最多重启次数（默认 5）
    budget_ctrl      : 指向被托管的 BetaOnlyBudgetController（同步 beta_current）
    """

    STATE_COMPRESS = 'COMPRESS'
    STATE_RECOVER  = 'RECOVER'
    STATE_RESTART  = 'RESTART'

    def __init__(
        self,
        budget_ctrl,               # BetaOnlyBudgetController 实例
        stall_patience: int  = 800,
        recover_epochs: int  = 400,
        min_delta: float     = 5e-5,
        restart_decay: float = 0.3,
        max_restarts: int    = 6,
    ):
        super().__init__()
        self.budget_ctrl    = budget_ctrl
        self.stall_patience = stall_patience
        self.recover_epochs = recover_epochs
        self.min_delta      = min_delta
        self.restart_decay  = restart_decay
        self.max_restarts   = max_restarts

        self._state          = self.STATE_COMPRESS
        self._best_acc       = -1.0
        self._stall_counter  = 0
        self._recover_counter= 0
        self._n_restarts     = 0
        self._beta_at_stall  = None

    def on_epoch_end(self, epoch, logs=None):
        logs     = logs or {}
        val_acc  = float(logs.get('val_accuracy', logs.get('val_acc', -1.0)))
        if val_acc < 0:
            return

        if self._state == self.STATE_COMPRESS:
            # 检测停滞
            if val_acc > self._best_acc + self.min_delta:
                self._best_acc      = val_acc
                self._stall_counter = 0
            else:
                self._stall_counter += 1

            if (self._stall_counter >= self.stall_patience
                    and self._n_restarts < self.max_restarts):
                # 触发 RECOVER
                self._beta_at_stall = self.budget_ctrl.beta_current
                print(f'\n  [βCurriculum] epoch={epoch}  STALL detected '
                      f'({self._stall_counter} ep)  '
                      f'β={self._beta_at_stall:.2e}  → RECOVER phase')
                self._state           = self.STATE_RECOVER
                self._recover_counter = 0
                self._stall_counter   = 0
                _set_all_beta(self.model, 0.0)
                self.budget_ctrl.beta_current = 0.0

        elif self._state == self.STATE_RECOVER:
            self._recover_counter += 1
            # 追踪恢复期间 acc 改善
            if val_acc > self._best_acc + self.min_delta:
                self._best_acc = val_acc

            if self._recover_counter >= self.recover_epochs:
                beta_new = self._beta_at_stall * self.restart_decay
                beta_new = max(beta_new, self.budget_ctrl.beta_min)
                self._n_restarts += 1
                print(f'\n  [βCurriculum] epoch={epoch}  RECOVER done  '
                      f'best_acc={self._best_acc:.4f}  '
                      f'restart β: {self._beta_at_stall:.2e} → {beta_new:.2e}  '
                      f'(restart #{self._n_restarts}/{self.max_restarts})')
                self.budget_ctrl.beta_current = beta_new
                self.budget_ctrl._ebops_ema   = None   # 重置 EMA
                _set_all_beta(self.model, beta_new)
                self._state          = self.STATE_COMPRESS
                self._stall_counter  = 0

        # 用数值编码，不写字符串到 logs（Keras 内部会对 log values 做 np.mean）
        logs['beta_curriculum_restarts'] = float(self._n_restarts)


# ─────────────────────────────────────────────────────────────────────────────
# 3-B.  自适应 LR × 位宽缩放
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveLRBiwidthScaler(keras.callbacks.Callback):
    """当 mean b_k 低于阈值时，自动放大当前 LR。

    原理
    ----
    STE 的有效信噪比 ∝ 2^b_k。b_k 越低，每步梯度更新越噪，
    需要更大的 LR 才能越过量化边界。
    缩放公式：
        lr_effective = lr_base * max(1, (bk_threshold / mean_bk)^scale_power)
    额外保护：lr_effective ≤ lr_max_factor * lr_base（防止过大）

    参数
    ----
    bk_threshold  : 触发 LR 放大的位宽阈值（默认 2.0）
    scale_power   : 缩放指数（默认 0.5）
    lr_max_factor : LR 最大放大倍数（默认 5.0）
    base_lr       : 基础 LR（None 则从 optimizer 读取）
    """

    def __init__(
        self,
        bk_threshold: float  = 2.0,
        scale_power: float   = 0.5,
        lr_max_factor: float = 5.0,
        base_lr: float       = None,
        pruned_threshold: float = 0.1,
        log: bool = False,
    ):
        super().__init__()
        self.bk_threshold     = bk_threshold
        self.scale_power      = scale_power
        self.lr_max_factor    = lr_max_factor
        self._base_lr         = base_lr
        self.pruned_threshold = pruned_threshold
        self.log              = log
        self._init_lr         = None

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
            pass   # schedule objects are read-only; skip silently

    def on_train_begin(self, logs=None):
        opt = self.model.optimizer
        if self._base_lr is None:
            self._init_lr = self._get_lr(opt)
        else:
            self._init_lr = self._base_lr

    def on_epoch_begin(self, epoch, logs=None):
        mean_bk = _get_active_bk_mean(self.model, self.pruned_threshold)
        factor  = max(1.0, (self.bk_threshold / max(mean_bk, 0.1)) ** self.scale_power)
        factor  = min(factor, self.lr_max_factor)

        # 读取调度器已设置的 LR，再乘 factor
        opt     = self.model.optimizer
        base_lr = self._get_lr(opt)
        new_lr  = base_lr * factor
        self._set_lr(opt, new_lr)

        if self.log:
            print(f'  [AdaptiveLR] epoch={epoch}  '
                  f'mean_bk={mean_bk:.3f}  factor={factor:.2f}  lr={new_lr:.2e}')


# ─────────────────────────────────────────────────────────────────────────────
# 3-C.  渐进式预算调度器
# ─────────────────────────────────────────────────────────────────────────────

class ProgressiveBudgetController(keras.callbacks.Callback):
    """课程式 EBOPs：target 从 warmup_ebops 指数衰减到 final_ebops。

    这避免了从 baseline (>>20k EBOPs) 直接剪枝到 500 EBOPs 造成的
    冷启动量化冲击（几乎所有 b_k 被强制到 0.1 以下，STE 完全失效）。

    用法
    ----
    1. 配合 BetaOnlyBudgetController（实时传入 target 更新）。
    2. 不需要手动剪枝；通过逐步减小 budget_ctrl.target_ebops，
       让 β 平滑施压，模型自然降低 b_k。

    参数
    ----
    budget_ctrl    : BetaOnlyBudgetController 实例
    warmup_ebops   : 起始目标 EBOPs（例如 4× final）
    final_ebops    : 最终目标
    decay_epochs   : 在多少 epoch 内从 warmup 降到 final（指数衰减）
    start_epoch    : 从哪个 epoch 开始（default=0）
    """

    def __init__(
        self,
        budget_ctrl,
        warmup_ebops: float,
        final_ebops:  float,
        decay_epochs: int,
        start_epoch:  int = 0,
    ):
        super().__init__()
        self.budget_ctrl   = budget_ctrl
        self.warmup_ebops  = float(warmup_ebops)
        self.final_ebops   = float(final_ebops)
        self.decay_epochs  = int(decay_epochs)
        self.start_epoch   = int(start_epoch)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        t = min((epoch - self.start_epoch) / max(self.decay_epochs, 1), 1.0)
        # 指数插值（对数空间线性插值）
        log_w = math.log(self.warmup_ebops)
        log_f = math.log(self.final_ebops)
        current_target = math.exp(log_w + t * (log_f - log_w))
        self.budget_ctrl.target_ebops = current_target


# ─────────────────────────────────────────────────────────────────────────────
# 3-D.  β 梯度截断（防止 EBOPs 梯度淹没精度梯度）
# ─────────────────────────────────────────────────────────────────────────────

class BetaGradClipCallback(keras.callbacks.Callback):
    """限制 β 单步变化量，防止 EBOPs 梯度突变导致 b_k 雪崩。

    实现方式：在 BetaOnlyBudgetController 之后运行，限制 β 的相对变化率。
    即：控制器计算的新 β 与旧 β 的比值不超过 max_ratio（每 epoch）。

    参数
    ----
    budget_ctrl  : BetaOnlyBudgetController 实例
    max_ratio    : 单 epoch β 最大变化倍数（默认 1.5）
    """

    def __init__(self, budget_ctrl, max_ratio: float = 1.5):
        super().__init__()
        self.budget_ctrl = budget_ctrl
        self.max_ratio   = max_ratio
        self._prev_beta  = None

    def on_epoch_begin(self, epoch, logs=None):
        if self._prev_beta is None:
            self._prev_beta = self.budget_ctrl.beta_current
            return
        prev   = self._prev_beta
        cur    = self.budget_ctrl.beta_current
        if prev > 0:
            ratio  = cur / prev
            if ratio > self.max_ratio:
                clipped = prev * self.max_ratio
                self.budget_ctrl.beta_current = clipped
                _set_all_beta(self.model, clipped)
            elif ratio < 1.0 / self.max_ratio:
                clipped = prev / self.max_ratio
                self.budget_ctrl.beta_current = clipped
                _set_all_beta(self.model, clipped)
        self._prev_beta = self.budget_ctrl.beta_current
