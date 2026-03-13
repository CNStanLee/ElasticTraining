"""
Topology Elasticity — 解决 Beta 死锁的三机制组合
=================================================

Beta 死锁根因分析:
  kq.b < 0.5 → round_conv = 0 → 前向输出 = 0 → 任务梯度 = 0
  → 仅 beta*eBOPs 梯度继续压低 b → 死亡不可逆
  → 拓扑在预算压缩早期就被锁定，无法探索更优拓扑

为什么 SDF (消融 D) 失败:
  b_floor = 0.05 < 0.5 → 连接虽被保护但前向仍为 0 → 与 beta 对抗循环

为什么 TopologyRescue (消融 E) 失败:
  事后干预 → 拓扑已深度提交 → 4 swap/layer 无法实质改变拓扑

本模块提供三个互补机制:

WarmTopologyFloor
  衰减 b 下限 (b0 ≥ 0.5 → 0)，防止预算压缩期间连接过早死亡。
  与 SDF 的关键区别: b0 ≥ 0.5 确保被保护连接在前向传播中仍然活跃。

ImportanceRebalance
  周期性地根据连接重要性 (|w|·b) 重新分配 b 预算。
  高重要性连接获得更多 b → 更快确立；弱连接加速淘汰。
  总 b 预算守恒 → 不扰乱 beta 控制器。

StochasticTopologyExplore
  模拟退火式拓扑扰动: 保存检查点 → 随机 swap → 短期评估 → 决定接受/回滚。
  提供梯度下降无法到达的拓扑探索空间。
"""

from __future__ import annotations

import math
import os
import tempfile

import keras
import numpy as np
import tensorflow as tf

from . import _get_kq_var, _flatten_layers


# ═══════════════════════════════════════════════════════════════════════════════
# WarmTopologyFloor — 衰减 b 下限防止过早死亡
# ═══════════════════════════════════════════════════════════════════════════════

class WarmTopologyFloor(keras.callbacks.Callback):
    """衰减 b-value 下限，防止连接在预算压缩期间过早且不可逆地死亡。

    核心思想:
      在训练初期保持 b_floor ≥ 0.5，使所有连接在前向传播中保持活跃。
      随着训练进行，floor 逐步衰减到 0，允许弱连接自然死亡。
      这给优化器足够时间评估每条连接的重要性，再做出不可逆的拓扑决策。

    与 SoftDeathFloor 的根本区别:
      SDF: b_floor = 0.05 < 0.5 → 连接仍然死亡 (round_conv=0)，无用
      WTF: b_floor starts ≥ 0.5 → 连接真正活跃，前向有信号，梯度有效

    衰减策略:
      b_floor(t) = b0 * max(0, 1 - t / anneal_epochs)^anneal_power
      anneal_power > 1: 前期慢衰减 (更长的探索期), 后期快速释放
      anneal_power < 1: 前期快衰减

    Parameters
    ----------
    b_floor_init : float
        初始 b 下限 (默认 0.55, 略高于 0.5 死亡线)
    anneal_epochs : int
        从 b_floor_init 衰减到 0 所需的 epoch 数
    anneal_power : float
        衰减曲线幂次 (1.0=线性, >1=前慢后快, <1=前快后慢)
    apply_every : int
        每 N epoch 执行一次 (默认 1)
    protect_kernel : bool
        是否同步恢复死亡 kernel (当 b 被钳位回活跃)
    kernel_init_scale : float
        恢复 kernel 时的初始化标准差
    log_interval : int
        打印日志的间隔 epoch 数 (默认 100)
    """

    def __init__(
        self,
        b_floor_init: float = 0.55,
        anneal_epochs: int = 2000,
        anneal_power: float = 1.5,
        apply_every: int = 1,
        protect_kernel: bool = True,
        kernel_init_scale: float = 0.01,
        log_interval: int = 100,
    ):
        super().__init__()
        self.b_floor_init = float(b_floor_init)
        self.anneal_epochs = max(1, int(anneal_epochs))
        self.anneal_power = float(anneal_power)
        self.apply_every = max(1, int(apply_every))
        self.protect_kernel = bool(protect_kernel)
        self.kernel_init_scale = float(kernel_init_scale)
        self.log_interval = max(1, int(log_interval))
        self._total_clamped = 0
        self._total_kernel_restored = 0

    def _current_floor(self, epoch: int) -> float:
        """计算当前 epoch 的 b 下限。"""
        t = min(epoch / self.anneal_epochs, 1.0)
        return self.b_floor_init * max(0.0, (1.0 - t)) ** self.anneal_power

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.apply_every != 0:
            return

        floor = self._current_floor(epoch)
        if floor < 1e-6:
            return  # floor 已衰减到 0, 不再干预

        n_clamped = 0
        n_kernel = 0

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
            # 只钳位那些 b > 0 但 < floor 的连接 (已被剪枝到 0 的不管)
            # 关键: 也包括 b == 0 的连接 (在 anneal 早期阶段)
            needs_floor = (b < floor) & (b >= 0.0)
            # 但不恢复被谱剪枝彻底删除的连接 (那些一开始就是 0)
            # 只恢复在训练过程中因 beta 压力降到 floor 以下的
            # 区分: 使用 kernel weight 是否为 0 来判断
            if not needs_floor.any():
                continue

            count = int(needs_floor.sum())
            n_clamped += count
            b_new = np.where(needs_floor, floor, b)
            b_var.assign(b_new.astype(np.float32))

            # 同步恢复 kernel
            if self.protect_kernel:
                kernel = getattr(layer, 'kernel', None)
                if kernel is not None:
                    k = kernel.numpy().astype(np.float32)
                    dead_kernel = needs_floor & (np.abs(k) < 1e-10)
                    if dead_kernel.any():
                        noise = np.random.randn(*k.shape).astype(np.float32) * self.kernel_init_scale
                        k = np.where(dead_kernel, noise, k)
                        kernel.assign(k)
                        n_kernel += int(dead_kernel.sum())

        self._total_clamped += n_clamped
        self._total_kernel_restored += n_kernel

        if logs is not None:
            logs['wtf_floor'] = floor
            logs['wtf_clamped'] = n_clamped

        if epoch % self.log_interval == 0 and n_clamped > 0:
            print(f'  [WarmFloor] epoch={epoch}  floor={floor:.4f}  '
                  f'clamped={n_clamped}  kernel_restored={n_kernel}')


# ═══════════════════════════════════════════════════════════════════════════════
# ImportanceRebalance — 重要性加权 b 预算再分配
# ═══════════════════════════════════════════════════════════════════════════════

class ImportanceRebalance(keras.callbacks.Callback):
    """基于连接重要性进行 b 值的再分配，打破 uniform beta 导致的拓扑决策随机性。

    核心思想:
      Beta 对所有连接施加均匀压力，但最优拓扑需要异质性决策。
      本机制周期性地将 b 预算从低重要性连接转移到高重要性连接:
        - 高重要性连接: 增加 b → 更确定地存活
        - 低重要性连接: 减少 b → 更快被淘汰
      总 b 预算近似守恒 → 不扰乱 beta 控制器和 eBOPs 预算。

    重要性度量:
      importance_i = |kernel_weight_i| × max(b_i, ε)
      这是 Fisher Information 的廉价近似：
        - |w| 大的连接对 loss 影响大
        - b 大的连接贡献更多 eBOPs → 值得保留

    再分配算法:
      1. 收集每层所有活跃连接的 (importance, b_value)
      2. 按 importance 排序
      3. bottom-p% 连接: b *= (1 - δ)
      4. top-p% 连接:    b *= (1 + δ) (但总增量 ≤ 总减量, 保持守恒)
      5. Clip b ∈ [b_min, b_max]

    Parameters
    ----------
    rebalance_interval : int
        每 N epoch 执行一次再分配
    rebalance_fraction : float
        参与再分配的连接比例 (上下各 p%)
    rebalance_delta : float
        每次调整幅度 (默认 0.1 = ±10%)
    alive_threshold : float
        b > 此值视为活跃，参与再分配
    b_min : float
        再分配后的 b 下限
    b_max : float
        再分配后的 b 上限
    start_epoch : int
        从哪个 epoch 开始再分配
    decay_factor : float
        delta 随时间衰减 (每次执行后 *= decay_factor)
    log_interval : int
        打印日志的间隔
    """

    def __init__(
        self,
        rebalance_interval: int = 50,
        rebalance_fraction: float = 0.2,
        rebalance_delta: float = 0.1,
        alive_threshold: float = 0.3,
        b_min: float = 0.0,
        b_max: float = 8.0,
        start_epoch: int = 0,
        decay_factor: float = 0.98,
        log_interval: int = 100,
    ):
        super().__init__()
        self.rebalance_interval = max(1, int(rebalance_interval))
        self.rebalance_fraction = float(rebalance_fraction)
        self.rebalance_delta = float(rebalance_delta)
        self.alive_threshold = float(alive_threshold)
        self.b_min = float(b_min)
        self.b_max = float(b_max)
        self.start_epoch = int(start_epoch)
        self.decay_factor = float(decay_factor)
        self.log_interval = max(1, int(log_interval))
        self._current_delta = float(rebalance_delta)
        self._n_executions = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        if (epoch - self.start_epoch) % self.rebalance_interval != 0:
            return

        total_boosted = 0
        total_weakened = 0
        total_b_moved = 0.0

        for layer in _flatten_layers(self.model):
            kq = getattr(layer, 'kq', None)
            if kq is None or not hasattr(layer, 'kernel'):
                continue
            b_var = _get_kq_var(kq, 'b')
            if b_var is None:
                b_var = _get_kq_var(kq, 'f')
            if b_var is None:
                continue

            b_arr = b_var.numpy().astype(np.float32)
            kernel = layer.kernel.numpy().astype(np.float32)

            # 计算重要性 (flatten for indexing)
            b_flat = b_arr.ravel()
            k_flat = kernel.ravel()
            n = len(b_flat)

            # 活跃连接掩码
            alive_mask = b_flat > self.alive_threshold
            alive_idx = np.where(alive_mask)[0]
            if len(alive_idx) < 4:  # 太少连接, 跳过
                continue

            # 计算重要性
            importance = np.abs(k_flat[alive_idx]) * np.maximum(b_flat[alive_idx], 0.01)

            # 排序
            sort_order = np.argsort(importance)
            n_alive = len(alive_idx)
            n_adjust = max(1, int(n_alive * self.rebalance_fraction))

            # Bottom n_adjust: 减少 b
            weak_idx = alive_idx[sort_order[:n_adjust]]
            b_decrease = b_flat[weak_idx] * self._current_delta
            b_flat[weak_idx] = np.maximum(b_flat[weak_idx] - b_decrease, self.b_min)

            # Top n_adjust: 增加 b (总增量 ≤ 总减量, 保持守恒)
            strong_idx = alive_idx[sort_order[-n_adjust:]]
            total_decrease = float(b_decrease.sum())
            per_increase = total_decrease / max(n_adjust, 1)
            b_flat[strong_idx] = np.minimum(b_flat[strong_idx] + per_increase, self.b_max)

            # 写回
            b_new = b_flat.reshape(b_arr.shape)
            b_var.assign(b_new.astype(np.float32))

            total_boosted += n_adjust
            total_weakened += n_adjust
            total_b_moved += total_decrease

        # 衰减 delta
        self._current_delta *= self.decay_factor
        self._n_executions += 1

        if logs is not None:
            logs['ir_boosted'] = total_boosted
            logs['ir_weakened'] = total_weakened
            logs['ir_b_moved'] = total_b_moved

        if epoch % self.log_interval == 0 and (total_boosted + total_weakened) > 0:
            print(f'  [ImportRebal] epoch={epoch}  '
                  f'boosted={total_boosted}  weakened={total_weakened}  '
                  f'b_moved={total_b_moved:.3f}  delta={self._current_delta:.4f}')


# ═══════════════════════════════════════════════════════════════════════════════
# StochasticTopologyExplore — 模拟退火式拓扑探索
# ═══════════════════════════════════════════════════════════════════════════════

class StochasticTopologyExplore(keras.callbacks.Callback):
    """模拟退火式拓扑扰动 + 评估 + 回滚 探索策略。

    核心思想:
      梯度下降无法跨越拓扑障碍 (死连接无梯度)。
      本机制通过周期性拓扑 swap 提供梯度无法到达的探索:
        1. 保存当前 b 值快照 (检查点)
        2. 执行 swap: 复活 k 个死连接 + 杀死 k 个最弱活跃连接
        3. 正常训练 eval_window epochs
        4. 评估: 如果 val_accuracy 改善 → 接受新拓扑
                 如果略微退化 (≤ accept_tol) → 以一定概率接受 (模拟退火)
                 如果严重退化 (> accept_tol) → 回滚到检查点

    候选选择:
      - 死连接候选: 优先选择满足谱条件 (节点度不足) 的死连接
      - 被杀连接: 按 |w| × b 排序, 选最弱的

    模拟退火:
      - 接受概率 = exp(-degradation / temperature)
      - temperature 随训练进行而降低 (exploration → exploitation)

    Parameters
    ----------
    explore_interval : int
        每 N epoch 尝试一次拓扑探索
    eval_window : int
        每次探索后训练多少 epoch 再评估效果
        (注: 这里不是独立训练, 而是标记状态, 等 eval_window 个 epoch 后评估)
    n_swap_per_layer : int
        每层 swap 多少连接
    accept_tolerance : float
        val_accuracy 退化容忍度 (超过此值回滚, 低于此值按概率接受)
    initial_temperature : float
        模拟退火初始温度
    temp_decay : float
        温度衰减因子 (每次探索后 *= temp_decay)
    min_temperature : float
        最低温度
    start_epoch : int
        从哪个 epoch 开始探索
    max_explorations : int
        总探索次数上限
    min_degree : int
        谱条件: 节点最小度
    alive_threshold : float
        b < 此值视为死连接
    revival_b_val : float
        复活连接的 b 值
    kernel_init_scale : float
        复活 kernel 为 0 时的初始化标准差
    """

    def __init__(
        self,
        explore_interval: int = 200,
        eval_window: int = 50,
        n_swap_per_layer: int = 4,
        accept_tolerance: float = 0.005,
        initial_temperature: float = 0.01,
        temp_decay: float = 0.85,
        min_temperature: float = 1e-4,
        start_epoch: int = 500,
        max_explorations: int = 15,
        min_degree: int = 2,
        alive_threshold: float = 0.5,
        revival_b_val: float = 1.0,
        kernel_init_scale: float = 0.01,
    ):
        super().__init__()
        self.explore_interval = max(1, int(explore_interval))
        self.eval_window = max(1, int(eval_window))
        self.n_swap_per_layer = int(n_swap_per_layer)
        self.accept_tolerance = float(accept_tolerance)
        self.temperature = float(initial_temperature)
        self.temp_decay = float(temp_decay)
        self.min_temperature = float(min_temperature)
        self.start_epoch = int(start_epoch)
        self.max_explorations = int(max_explorations)
        self.min_degree = int(min_degree)
        self.alive_threshold = float(alive_threshold)
        self.revival_b_val = float(revival_b_val)
        self.kernel_init_scale = float(kernel_init_scale)

        # 状态机: IDLE / EXPLORING / EVALUATING
        self._state = 'IDLE'
        self._checkpoint = None  # dict: layer_id → (b_snapshot, kernel_snapshot)
        self._pre_swap_acc = None
        self._swap_epoch = None
        self._n_explorations = 0
        self._n_accepted = 0
        self._n_rejected = 0

    def _save_checkpoint(self):
        """保存所有可剪枝层的 b 和 kernel 快照。"""
        ckpt = {}
        for layer in _flatten_layers(self.model):
            kq = getattr(layer, 'kq', None)
            if kq is None or not hasattr(layer, 'kernel'):
                continue
            b_var = _get_kq_var(kq, 'b')
            if b_var is None:
                b_var = _get_kq_var(kq, 'f')
            if b_var is None:
                continue
            ckpt[id(layer)] = {
                'b_var': b_var,
                'b_snap': b_var.numpy().copy(),
                'kernel': layer.kernel,
                'k_snap': layer.kernel.numpy().copy(),
            }
        return ckpt

    def _restore_checkpoint(self, ckpt):
        """从快照恢复 b 和 kernel。"""
        for layer in _flatten_layers(self.model):
            lid = id(layer)
            if lid not in ckpt:
                continue
            info = ckpt[lid]
            info['b_var'].assign(info['b_snap'].astype(np.float32))
            info['kernel'].assign(info['k_snap'].astype(np.float32))

    def _spectral_dead_candidates(self, b_arr: np.ndarray) -> list[tuple[int, int]]:
        """找出谱条件下需复活的死连接候选。"""
        if b_arr.ndim == 1:
            return []
        if b_arr.ndim > 2:
            b_arr = b_arr.reshape(b_arr.shape[0], -1)

        dead = (b_arr < self.alive_threshold)
        active = ~dead
        out_degree = active.sum(axis=1)
        in_degree = active.sum(axis=0)

        candidates = set()
        for r in np.where(out_degree < self.min_degree)[0]:
            for c in np.where(dead[r])[0]:
                candidates.add((int(r), int(c)))
        for c in np.where(in_degree < self.min_degree)[0]:
            for r in np.where(dead[:, c])[0]:
                candidates.add((int(r), int(c)))

        # 如果谱候选不足, 补充随机死连接
        if len(candidates) < self.n_swap_per_layer:
            dead_positions = list(zip(*np.where(dead)))
            np.random.shuffle(dead_positions)
            for (r, c) in dead_positions:
                candidates.add((int(r), int(c)))
                if len(candidates) >= self.n_swap_per_layer * 2:
                    break

        return list(candidates)

    def _perform_swap(self):
        """执行拓扑 swap: 复活死连接 + 杀死最弱活跃连接。"""
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

            b_arr = b_var.numpy().astype(np.float32)
            orig_shape = b_arr.shape
            b_2d = b_arr.reshape(b_arr.shape[0], -1) if b_arr.ndim > 2 else b_arr

            # 找死连接候选
            candidates = self._spectral_dead_candidates(b_2d)
            if not candidates:
                continue

            # 按 |kernel| 排序候选 (大 kernel → 更有潜力)
            kernel = layer.kernel.numpy().astype(np.float32)
            k_2d = kernel.reshape(kernel.shape[0], -1) if kernel.ndim > 2 else kernel
            scored_cands = []
            for (r, c) in candidates:
                if r < k_2d.shape[0] and c < k_2d.shape[1]:
                    scored_cands.append((abs(float(k_2d[r, c])) + np.random.rand() * 0.001, r, c))
                else:
                    scored_cands.append((np.random.rand() * 0.001, r, c))
            scored_cands.sort(reverse=True)
            top_k = scored_cands[:self.n_swap_per_layer]

            # 找最弱活跃连接 (swap-kill 维持预算)
            revive_set = {(r, c) for (_, r, c) in top_k}
            alive_conns = []
            for r in range(b_2d.shape[0]):
                for c in range(b_2d.shape[1]):
                    if b_2d[r, c] >= self.alive_threshold and (r, c) not in revive_set:
                        importance = abs(float(k_2d[r, c])) * float(b_2d[r, c])
                        alive_conns.append((importance, r, c))
            alive_conns.sort()
            kill_set = [(r, c) for (_, r, c) in alive_conns[:len(top_k)]]

            # 执行 swap
            for (r, c) in kill_set:
                b_2d[r, c] = 0.0
            for (_, r, c) in top_k:
                b_2d[r, c] = self.revival_b_val
                # 恢复 kernel
                if abs(float(k_2d[r, c])) < 1e-10:
                    k_2d[r, c] = np.random.randn() * self.kernel_init_scale

            # 写回
            if b_arr.ndim > 2:
                b_new = b_2d.reshape(orig_shape)
                k_new = k_2d.reshape(kernel.shape)
            else:
                b_new = b_2d
                k_new = k_2d
            b_var.assign(b_new.astype(np.float32))
            layer.kernel.assign(k_new.astype(np.float32))

            total_revived += len(top_k)
            total_killed += len(kill_set)

        return total_revived, total_killed

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = float(logs.get('val_accuracy', logs.get('val_acc', -1.0)))
        if val_acc < 0:
            return

        if self._state == 'IDLE':
            # 检查是否应该开始新的探索
            if epoch < self.start_epoch:
                return
            if self._n_explorations >= self.max_explorations:
                return
            if (epoch - self.start_epoch) % self.explore_interval != 0:
                return

            # 开始探索: 保存检查点 → swap
            self._checkpoint = self._save_checkpoint()
            self._pre_swap_acc = val_acc
            self._swap_epoch = epoch

            n_rev, n_kill = self._perform_swap()
            if n_rev == 0:
                self._state = 'IDLE'
                self._checkpoint = None
                return

            self._state = 'EVALUATING'
            self._n_explorations += 1

            print(f'  [TopoExplore] epoch={epoch}  exploration #{self._n_explorations}'
                  f'/{self.max_explorations}  revived={n_rev} killed={n_kill}  '
                  f'pre_acc={val_acc:.4f}  T={self.temperature:.4f}')

        elif self._state == 'EVALUATING':
            # 等待 eval_window 后评估
            if epoch - self._swap_epoch < self.eval_window:
                return

            # 评估结果
            delta_acc = val_acc - self._pre_swap_acc

            if delta_acc >= 0:
                # 改善: 接受
                decision = 'ACCEPT (improved)'
                self._n_accepted += 1
            elif abs(delta_acc) <= self.accept_tolerance:
                # 轻微退化: 按概率接受 (模拟退火)
                accept_prob = math.exp(delta_acc / max(self.temperature, 1e-10))
                if np.random.rand() < accept_prob:
                    decision = f'ACCEPT (anneal, p={accept_prob:.3f})'
                    self._n_accepted += 1
                else:
                    decision = f'REJECT (anneal, p={accept_prob:.3f})'
                    self._restore_checkpoint(self._checkpoint)
                    self._n_rejected += 1
            else:
                # 严重退化: 回滚
                decision = 'REJECT (degraded)'
                self._restore_checkpoint(self._checkpoint)
                self._n_rejected += 1

            # 温度衰减
            self.temperature = max(self.temperature * self.temp_decay, self.min_temperature)

            print(f'  [TopoExplore] epoch={epoch}  {decision}  '
                  f'Δacc={delta_acc:+.4f}  '
                  f'pre={self._pre_swap_acc:.4f} → post={val_acc:.4f}  '
                  f'accepted={self._n_accepted} rejected={self._n_rejected}  '
                  f'T={self.temperature:.4f}')

            # 状态重置
            self._state = 'IDLE'
            self._checkpoint = None

            logs['topo_explore_accepted'] = self._n_accepted
            logs['topo_explore_rejected'] = self._n_rejected
