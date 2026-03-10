"""
训练工具 — LR 调度、训练轨迹记录、智能早停
==========================================
"""

from __future__ import annotations

import os
import time
from math import cos, pi

import h5py
import keras
import numpy as np
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
    """将每 epoch 的标量指标 + HGQ 位宽分布追加写入 HDF5 文件。"""

    def __init__(self, output_dir: str, filename: str = "training_trace.h5",
                 max_bits: int = 8, beta_callback=None):
        super().__init__()
        self.output_dir = output_dir
        self.filename = filename
        self.path = os.path.join(output_dir, filename)
        self.max_bits = int(max_bits)
        self.beta_callback = beta_callback
        self.t0 = None

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

        bw_pct = self._bw_distribution_pct(self._collect_bits())
        for b in range(self.max_bits + 1):
            logs[f"bw_{b}bit_pct"] = float(bw_pct[b])

        try:
            with h5py.File(self.path, "a") as hf:
                self._append_1d(hf, "epochs", int(epoch), dtype=np.int32)
                self._append_1d(hf, "time_sec", float(t_sec))
                self._append_1d(hf, "loss", loss)
                self._append_1d(hf, "val_loss", val_loss)
                self._append_1d(hf, "accuracy", acc)
                self._append_1d(hf, "val_accuracy", val_acc)
                self._append_1d(hf, "ebops", ebops)
                self._append_1d(hf, "lr", lr)
                self._append_1d(hf, "beta", beta)
                self._append_2d_row(hf, "bw_pct", bw_pct)
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
