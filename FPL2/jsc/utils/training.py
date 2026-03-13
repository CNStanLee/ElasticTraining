"""
训练工具 — LR 调度、训练轨迹记录、智能早停、SWA
================================================
"""

from __future__ import annotations

import os
import time
from math import cos, pi

import h5py
import keras
import numpy as np
import tensorflow as tf
from hgq.layers import QLayerBase
from keras import ops


# ═══════════════════════════════════════════════════════════════════════════════
# LR 调度：Cosine Decay with Warm Restarts
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_decay_restarts_schedule(
    initial_learning_rate: float,
    first_decay_steps: int,
    t_mul: float = 1.0,
    m_mul: float = 1.0,
    alpha: float = 0.0,
    alpha_steps: int = 0,
):
    """Cosine annealing with warm restarts (Loshchilov & Hutter)。

    返回一个 callable: epoch -> lr
    """
    def schedule(global_step):
        cycle_step = global_step
        cycle_len = first_decay_steps
        n_cycle = 1
        while cycle_step >= cycle_len:
            cycle_step -= cycle_len
            cycle_len *= t_mul
            n_cycle += 1
        cycle_t = min(cycle_step / (cycle_len - alpha_steps), 1)
        lr = alpha + 0.5 * (initial_learning_rate - alpha) * (1 + cos(pi * cycle_t)) * m_mul ** max(n_cycle - 1, 0)
        return lr
    return schedule


# ═══════════════════════════════════════════════════════════════════════════════
# TrainingTraceToH5 — 每 epoch 追加 HDF5 训练轨迹
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingTraceToH5(keras.callbacks.Callback):
    """将每 epoch 的标量指标 + HGQ 位宽分布追加写入 HDF5 文件。

    Parameters
    ----------
    write_every : int
        每 N epoch 才实际写入 h5 文件 (默认 10)。
        中间 epoch 的数据会缓存在内存中，减少 h5 open/flush 开销。
        训练结束时 (on_train_end) 自动 flush 剩余数据。
    """

    def __init__(self, output_dir: str, filename: str = "training_trace.h5",
                 max_bits: int = 8, beta_callback=None, write_every: int = 10):
        super().__init__()
        self.output_dir = output_dir
        self.filename = filename
        self.path = os.path.join(output_dir, filename)
        self.max_bits = int(max_bits)
        self.beta_callback = beta_callback
        self.write_every = max(1, int(write_every))
        self.t0 = None
        self._buffer = []  # list of dicts, one per buffered epoch

    # ── 位宽分布收集 ──────────────────────────────────────────────────────

    def _collect_bits(self):
        all_bits = []
        for layer in self.model._flatten_layers():
            if not isinstance(layer, QLayerBase):
                continue
            for qname in ("_iq", "kq", "_oq"):
                q = getattr(layer, qname, None)
                if q is not None and getattr(q, "built", False):
                    b = ops.convert_to_numpy(q.bits).reshape(-1)
                    all_bits.extend(b.tolist())
        return all_bits

    def _bw_distribution_pct(self, all_bits):
        if not all_bits:
            return np.zeros((self.max_bits + 1,), dtype=np.float64)
        counts = np.zeros((self.max_bits + 1,), dtype=np.int64)
        for b in all_bits:
            idx = max(0, min(int(round(float(b))), self.max_bits))
            counts[idx] += 1
        return (counts.astype(np.float64) * 100.0) / float(len(all_bits))

    # ── 标量辅助 ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_float(x):
        try:
            return float(ops.convert_to_numpy(x))
        except Exception:
            try:
                return float(x)
            except Exception:
                return float("nan")

    def _get_lr(self, logs):
        if logs:
            for k in ("lr", "learning_rate"):
                if k in logs:
                    return self._to_float(logs[k])
        opt = getattr(self.model, "optimizer", None)
        if opt:
            lr = getattr(opt, "learning_rate", None)
            if lr is not None:
                return self._to_float(lr)
        return float("nan")

    def _get_beta(self, logs):
        if logs and "beta" in logs:
            return self._to_float(logs["beta"])
        cb = self.beta_callback
        if cb:
            for attr in ("beta", "current_beta", "value"):
                if hasattr(cb, attr):
                    v = getattr(cb, attr)
                    if v is not None:
                        return self._to_float(v)
        return float("nan")

    # ── HDF5 追加 ─────────────────────────────────────────────────────────

    @staticmethod
    def _append_1d(hf, name, value, dtype=np.float64):
        if name not in hf:
            hf.create_dataset(name, data=np.array([value], dtype=dtype), maxshape=(None,), chunks=True)
        else:
            d = hf[name]
            d.resize((d.shape[0] + 1,))
            d[-1] = value

    @staticmethod
    def _append_2d_row(hf, name, row: np.ndarray):
        row = np.asarray(row, dtype=np.float64).reshape(1, -1)
        if name not in hf:
            hf.create_dataset(name, data=row, maxshape=(None, row.shape[1]), chunks=True)
        else:
            d = hf[name]
            d.resize((d.shape[0] + 1, d.shape[1]))
            d[-1, :] = row

    def on_train_begin(self, logs=None):
        self.t0 = time.time()
        self._buffer = []
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        t_sec = time.time() - (self.t0 or time.time())

        loss     = self._to_float(logs.get("loss", float("nan")))
        val_loss = self._to_float(logs.get("val_loss", float("nan")))
        acc      = self._to_float(logs.get("accuracy", float("nan")))
        val_acc  = self._to_float(logs.get("val_accuracy", float("nan")))
        ebops    = self._to_float(logs.get("ebops", float("nan")))
        lr       = self._get_lr(logs)
        beta     = self._get_beta(logs)

        # 仅在即将 flush 时收集位宽分布 (避免每 epoch GPU→CPU 传输)
        needs_flush = (len(self._buffer) + 1 >= self.write_every)
        if needs_flush:
            bw_pct = self._bw_distribution_pct(self._collect_bits())
            for b in range(self.max_bits + 1):
                logs[f"bw_{b}bit_pct"] = float(bw_pct[b])
        else:
            bw_pct = None

        # 缓存到内存
        self._buffer.append(dict(
            epoch=int(epoch), time_sec=float(t_sec),
            loss=loss, val_loss=val_loss,
            accuracy=acc, val_accuracy=val_acc,
            ebops=ebops, lr=lr, beta=beta,
            bw_pct=bw_pct.copy() if bw_pct is not None else None,
        ))

        # 每 write_every 个 epoch 批量写入
        if len(self._buffer) >= self.write_every:
            self._flush_buffer()

    def on_train_end(self, logs=None):
        """训练结束时 flush 剩余缓冲数据。"""
        if self._buffer:
            self._flush_buffer()

    def _flush_buffer(self):
        """将缓冲数据批量写入 h5 文件。"""
        if not self._buffer:
            return
        buf = self._buffer
        self._buffer = []

        # 对没有 bw_pct 的行, 用最近一个有效值填充 (或收集当前值)
        last_bw = None
        for row in reversed(buf):
            if row["bw_pct"] is not None:
                last_bw = row["bw_pct"]
                break
        if last_bw is None:
            last_bw = self._bw_distribution_pct(self._collect_bits())
        for row in buf:
            if row["bw_pct"] is None:
                row["bw_pct"] = last_bw

        try:
            with h5py.File(self.path, "a") as hf:
                for row in buf:
                    self._append_1d(hf, "epochs", row["epoch"], dtype=np.int32)
                    self._append_1d(hf, "time_sec", row["time_sec"])
                    self._append_1d(hf, "loss", row["loss"])
                    self._append_1d(hf, "val_loss", row["val_loss"])
                    self._append_1d(hf, "accuracy", row["accuracy"])
                    self._append_1d(hf, "val_accuracy", row["val_accuracy"])
                    self._append_1d(hf, "ebops", row["ebops"])
                    self._append_1d(hf, "lr", row["lr"])
                    self._append_1d(hf, "beta", row["beta"])
                    self._append_2d_row(hf, "bw_pct", row["bw_pct"])
                if "bits" not in hf:
                    hf.create_dataset("bits", data=np.arange(0, self.max_bits + 1, dtype=np.int32))
                hf.flush()
        except Exception:
            pass  # 不因写盘失败中断训练


# ═══════════════════════════════════════════════════════════════════════════════
# BudgetAwareEarlyStopping — 精度平台 + eBOPs 达标 联合早停
# ═══════════════════════════════════════════════════════════════════════════════

class BudgetAwareEarlyStopping(keras.callbacks.Callback):
    """精度平台期 + eBOPs 达标时的联合早停。

    触发条件（AND）：
    1. val_accuracy 连续 patience epoch 无 min_delta 提升
    2. ebops <= ebops_budget
    """

    def __init__(
        self,
        ebops_budget: float = float('inf'),
        patience: int = 300,
        min_delta: float = 1e-4,
        min_epoch: int = 0,
        restore_best_weights: bool = True,
    ):
        super().__init__()
        self.ebops_budget = ebops_budget
        self.patience = patience
        self.min_delta = min_delta
        self.min_epoch = min_epoch
        self.restore_best_weights = restore_best_weights
        self._best_val_acc = -float('inf')
        self._plateau_count = 0
        self._best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = float(logs.get('val_accuracy', float('nan')))
        current_ebops = float(logs.get('ebops', float('nan')))

        if val_acc != val_acc or epoch < self.min_epoch:
            return

        if val_acc > self._best_val_acc + self.min_delta:
            self._best_val_acc = val_acc
            self._plateau_count = 0
            if self.restore_best_weights:
                self._best_weights = [v.numpy() for v in self.model.weights]
        else:
            self._plateau_count += 1

        ebops_ok = (current_ebops != current_ebops) or (current_ebops <= self.ebops_budget)

        if self._plateau_count >= self.patience and ebops_ok:
            print(
                f'\n[EarlyStopping] Accuracy plateau {self._plateau_count} epochs '
                f'(best={self._best_val_acc:.4f}), '
                f'eBOPs={current_ebops:.0f} ≤ budget={self.ebops_budget:.0f}. '
                f'Stopping @ epoch {epoch}.'
            )
            if self.restore_best_weights and self._best_weights is not None:
                for var, val in zip(self.model.weights, self._best_weights):
                    var.assign(val)
                print(f'[EarlyStopping] Restored best weights (val_acc={self._best_val_acc:.4f})')
            self.model.stop_training = True


# ═══════════════════════════════════════════════════════════════════════════════
# StochasticWeightAveraging — SWA for flatter minima
# ═══════════════════════════════════════════════════════════════════════════════

class StochasticWeightAveraging(keras.callbacks.Callback):
    """Stochastic Weight Averaging (Izmailov et al. 2018).

    在 LR 余弦退火每个周期的谷底收集权重快照，积累均值。
    训练结束后用均值替换模型权重，得到更平坦（泛化更好）的极小值。

    对 HGQ 的特殊处理:
      - kq.b 和 kernel 都参与平均
      - 平均后的 kq.b 可能处于不同的 eBOPs 点，但更稳定
      - SWA 不改变拓扑结构（不复活/杀死连接）

    Parameters
    ----------
    swa_start : int
        开始收集快照的 epoch（通常设为 Phase 2 起始）
    cycle_length : int
        LR 余弦周期长度（在周期末尾收集快照）
    apply_on_train_end : bool
        训练结束时是否自动应用 SWA 权重
    """

    def __init__(
        self,
        swa_start: int = 0,
        cycle_length: int = 500,
        apply_on_train_end: bool = False,
    ):
        super().__init__()
        self.swa_start = int(swa_start)
        self.cycle_length = max(1, int(cycle_length))
        self.apply_on_train_end = bool(apply_on_train_end)
        self._swa_weights = None
        self._n_averaged = 0

    @property
    def n_averaged(self):
        return self._n_averaged

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.swa_start:
            return

        effective_epoch = epoch - self.swa_start
        # 在周期末尾收集快照（LR 最低点，模型最稳定）
        if effective_epoch > 0 and effective_epoch % self.cycle_length == 0:
            self._collect_snapshot()

    def _collect_snapshot(self):
        """收集当前权重到 SWA 均值。"""
        if self._swa_weights is None:
            self._swa_weights = [w.numpy().copy() for w in self.model.trainable_variables]
            self._n_averaged = 1
        else:
            self._n_averaged += 1
            for i, w in enumerate(self.model.trainable_variables):
                self._swa_weights[i] += (w.numpy() - self._swa_weights[i]) / self._n_averaged
        print(f'  [SWA] Collected snapshot #{self._n_averaged}')

    def apply_swa_weights(self):
        """将 SWA 均值写入模型。"""
        if self._swa_weights is None or self._n_averaged == 0:
            print('  [SWA] No snapshots collected, skipping.')
            return False
        for w, swa_w in zip(self.model.trainable_variables, self._swa_weights):
            w.assign(swa_w.astype(np.float32))
        print(f'  [SWA] Applied weight average of {self._n_averaged} snapshots.')
        return True

    def get_swa_weights(self):
        """返回 SWA 均值权重列表（不修改模型）。"""
        if self._swa_weights is None:
            return None
        return [w.copy() for w in self._swa_weights]

    def on_train_end(self, logs=None):
        if self.apply_on_train_end:
            self.apply_swa_weights()


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Distillation Helper — Phase 间知识迁移
# ═══════════════════════════════════════════════════════════════════════════════

def generate_self_distillation_labels(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    alpha: float = 0.3,
    temperature: float = 3.0,
    batch_size: int = 33200,
) -> np.ndarray:
    """从当前模型生成自蒸馏软标签。

    Phase-Transfer Self-Distillation:
      在 Phase 1 结束时，模型在较高 eBOPs 下达到了最佳精度。
      用该模型的 soft predictions 混合 hard labels，
      作为 Phase 2 的训练标签。

    好处:
      - 保留 Phase 1 高 eBOPs 模型的 "dark knowledge"
      - 类间相对概率编码了数据流形结构信息
      - 不改变拓扑、不影响 eBOPs、不扰乱 beta
      - 帮助 Phase 2 在压缩过程中保持精度

    Parameters
    ----------
    model : keras.Model
        当前模型（Phase 1 结束后）
    X_train : np.ndarray
        训练数据
    y_train : np.ndarray
        硬标签（integer labels）
    num_classes : int
        类别数
    alpha : float
        软标签混合比例: y_mix = (1-α)*one_hot + α*soft_labels
    temperature : float
        温度参数: 更高 → soft labels 更平滑
    batch_size : int
        预测批大小

    Returns
    -------
    y_soft : np.ndarray
        混合标签, shape=(N, num_classes), dtype=float32
    """
    print(f'  [SelfDistill] Generating soft labels: α={alpha}, T={temperature}...')

    # 生成 teacher logits
    teacher_logits = model.predict(X_train, batch_size=batch_size, verbose=0)

    # Softmax with temperature
    teacher_probs = tf.nn.softmax(
        tf.constant(teacher_logits, dtype=tf.float32) / temperature
    ).numpy().astype(np.float32)

    # One-hot hard labels
    y_onehot = tf.one_hot(y_train, num_classes).numpy().astype(np.float32)

    # Mix: (1-α)*hard + α*soft
    y_soft = ((1.0 - alpha) * y_onehot + alpha * teacher_probs).astype(np.float32)

    # 验证
    teacher_acc = float(np.mean(np.argmax(teacher_logits, axis=-1) == y_train))
    print(f'  [SelfDistill] Teacher accuracy: {teacher_acc:.4f}')
    print(f'  [SelfDistill] Soft label entropy: {-np.mean(np.sum(y_soft * np.log(y_soft + 1e-10), axis=-1)):.3f}')

    return y_soft
