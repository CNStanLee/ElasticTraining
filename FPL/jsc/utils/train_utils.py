import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # jax, torch, tensorflow
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import time

import tensorflow as tf
import keras
import numpy as np
import h5py

# --- HGQ layer base (for identifying quantized layers) ---
from hgq.layers import QLayerBase
from keras import ops
from math import cos, pi

class EBOPsAdaptiveBeta(keras.callbacks.Callback):
    """
    在 warmup 结束后，动态调整 beta 使 EBOPs 在 total_epochs 结束时趋近于 0。

    时序设计（关键）
    ----------------
    错误做法：在 on_epoch_end 写 layer._beta
              → BetaScheduler.on_epoch_begin 在下一 epoch 开始时立刻覆盖，boost 无效。

    正确做法：
      - on_epoch_end   : 读当前 EBOPs，计算目标 beta，存入 self._next_beta
      - on_epoch_begin : （排在 BetaScheduler 之后）用 self._next_beta 覆盖写入 layer._beta

    参数
    ----
    beta_scheduler   : 原有的 BetaScheduler 实例
    warmup_end_epoch : beta 开始上升的 epoch（= beta_sch_1）
    total_epochs     : 训练总 epoch
    beta_hard_max    : beta 绝对上限（建议 1e-1，12000 epoch 内压到 0 需要足够大）
    boost_power      : 超额比例指数，越大越激进（建议 0.5~1.0）
    target_ebops     : 最终目标 EBOPs，默认 0
    """

    def __init__(
        self,
        beta_scheduler,
        warmup_end_epoch: int,
        total_epochs: int,
        beta_hard_max: float = 1e-1,
        boost_power: float = 0.7,
        target_ebops: float = 0.0,
    ):
        super().__init__()
        self.beta_scheduler = beta_scheduler
        self.warmup_end_epoch = warmup_end_epoch
        self.total_epochs = total_epochs
        self.beta_hard_max = beta_hard_max
        self.boost_power = boost_power
        self.target_ebops = target_ebops
        self._ebops_at_warmup_end = None
        self._next_beta = None  # on_epoch_end 计算，on_epoch_begin 施加

    def _assign_beta(self, beta_value):
        """直接写入所有量化层的 _beta 变量。"""
        from keras import ops
        for layer in self.model._flatten_layers():
            if hasattr(layer, '_beta'):
                layer._beta.assign(
                    ops.convert_to_tensor(beta_value, dtype=layer._beta.dtype)
                )

    def on_epoch_begin(self, epoch, logs=None):
        """BetaScheduler 先赋值后本 callback 覆盖——须排在 callbacks 列表 BetaScheduler 之后。"""
        if self._next_beta is not None:
            self._assign_beta(self._next_beta)

    def on_epoch_end(self, epoch, logs=None):
        """读取本 epoch EBOPs，计算下一个 epoch 应施加的 beta。"""
        logs = logs or {}
        if epoch < self.warmup_end_epoch:
            return

        current_ebops = float(logs.get('ebops', float('nan')))
        if current_ebops != current_ebops:  # nan check
            return

        if self._ebops_at_warmup_end is None:
            self._ebops_at_warmup_end = max(current_ebops, 1.0)

        compress_span = self.total_epochs - self.warmup_end_epoch
        elapsed = epoch - self.warmup_end_epoch
        progress = min(elapsed / max(compress_span, 1), 1.0)

        ebops_range = self._ebops_at_warmup_end - self.target_ebops
        target = self._ebops_at_warmup_end - ebops_range * progress

        beta_scheduled = float(self.beta_scheduler.beta_fn(epoch))

        if target <= 0 or current_ebops <= target:
            self._next_beta = None
            return

        surplus_ratio = current_ebops / max(target, 1e-6)
        beta_boosted = beta_scheduled * (surplus_ratio ** self.boost_power)
        self._next_beta = min(beta_boosted, self.beta_hard_max)
        logs['beta_adaptive'] = self._next_beta


def cosine_decay_restarts_schedule(
    initial_learning_rate: float, first_decay_steps: int, t_mul=1.0, m_mul=1.0, alpha=0.0, alpha_steps=0
):
    def schedule(global_step):
        n_cycle = 1
        cycle_step = global_step
        cycle_len = first_decay_steps
        while cycle_step >= cycle_len:
            cycle_step -= cycle_len
            cycle_len *= t_mul
            n_cycle += 1

        cycle_t = min(cycle_step / (cycle_len - alpha_steps), 1)
        lr = alpha + 0.5 * (initial_learning_rate - alpha) * (1 + cos(pi * cycle_t)) * m_mul ** max(n_cycle - 1, 0)
        return lr

    return schedule

class TrainingTraceToH5(keras.callbacks.Callback):
    """
    Append per-epoch training details + HGQ bitwidth distribution to one HDF5 file.

    Writes: results/training_trace.h5
      - epochs                 [N]
      - time_sec               [N]
      - loss, val_loss         [N]
      - accuracy, val_accuracy [N]
      - ebops                  [N]
      - lr                     [N]
      - beta                   [N]
      - bw_pct                 [N, max_bits+1]  (percentage at each integer bit)
      - bits                   [max_bits+1]     (0..max_bits)
    """

    def __init__(self, output_dir: str, filename: str = "training_trace.h5",
                 max_bits: int = 8, beta_callback=None):
        super().__init__()
        self.output_dir = output_dir
        self.filename = filename
        self.path = os.path.join(output_dir, filename)
        self.max_bits = int(max_bits)
        self.beta_callback = beta_callback
        self.t0 = None

    # ---------- bitwidth ----------
    def _collect_bits(self):
        all_bits = []
        for layer in self.model._flatten_layers():
            if not isinstance(layer, QLayerBase):
                continue

            # input quantizer
            if hasattr(layer, "_iq") and layer._iq is not None and getattr(layer._iq, "built", False):
                b = ops.convert_to_numpy(layer._iq.bits).reshape(-1)
                all_bits.extend(b.tolist())

            # weight/kernel quantizer
            if hasattr(layer, "kq") and layer.kq is not None and getattr(layer.kq, "built", False):
                b = ops.convert_to_numpy(layer.kq.bits).reshape(-1)
                all_bits.extend(b.tolist())

            # output quantizer
            if hasattr(layer, "_oq") and layer._oq is not None and getattr(layer._oq, "built", False):
                b = ops.convert_to_numpy(layer._oq.bits).reshape(-1)
                all_bits.extend(b.tolist())

        return all_bits

    def _bw_distribution_pct(self, all_bits):
        if not all_bits:
            return np.zeros((self.max_bits + 1,), dtype=np.float64)

        counts = np.zeros((self.max_bits + 1,), dtype=np.int64)
        for b in all_bits:
            idx = int(round(float(b)))
            idx = max(0, min(idx, self.max_bits))
            counts[idx] += 1

        return (counts.astype(np.float64) * 100.0) / float(len(all_bits))

    # ---------- scalar helpers ----------
    def _to_float(self, x):
        try:
            return float(ops.convert_to_numpy(x))
        except Exception:
            try:
                return float(x)
            except Exception:
                return float("nan")

    def _get_lr(self, logs):
        if logs is not None:
            if "lr" in logs:
                return self._to_float(logs["lr"])
            if "learning_rate" in logs:
                return self._to_float(logs["learning_rate"])

        opt = getattr(self.model, "optimizer", None)
        if opt is not None:
            lr = getattr(opt, "learning_rate", None)
            if lr is not None:
                return self._to_float(lr)
        return float("nan")

    def _get_beta(self, logs):
        if logs is not None and "beta" in logs:
            return self._to_float(logs["beta"])

        cb = self.beta_callback
        if cb is not None:
            for attr in ("beta", "current_beta", "value"):
                if hasattr(cb, attr):
                    v = getattr(cb, attr)
                    if v is not None:
                        return self._to_float(v)
        return float("nan")

    # ---------- HDF5 append ----------
    def _append_1d(self, hf, name, value, dtype=np.float64):
        if name not in hf:
            hf.create_dataset(
                name,
                data=np.array([value], dtype=dtype),
                maxshape=(None,),
                chunks=True,
            )
        else:
            d = hf[name]
            d.resize((d.shape[0] + 1,))
            d[-1] = value

    def _append_2d_row(self, hf, name, row: np.ndarray):
        row = np.asarray(row, dtype=np.float64).reshape(1, -1)
        if name not in hf:
            hf.create_dataset(
                name,
                data=row,
                maxshape=(None, row.shape[1]),
                chunks=True,
            )
        else:
            d = hf[name]
            d.resize((d.shape[0] + 1, d.shape[1]))
            d[-1, :] = row

    def on_train_begin(self, logs=None):
        self.t0 = time.time()
        os.makedirs(self.output_dir, exist_ok=True)
        # If you want to always start fresh, uncomment:
        # if os.path.exists(self.path):
        #     os.remove(self.path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        t_sec = time.time() - (self.t0 or time.time())

        # scalar metrics
        loss = self._to_float(logs.get("loss", float("nan")))
        val_loss = self._to_float(logs.get("val_loss", float("nan")))
        acc = self._to_float(logs.get("accuracy", float("nan")))
        val_acc = self._to_float(logs.get("val_accuracy", float("nan")))
        ebops = self._to_float(logs.get("ebops", float("nan")))
        lr = self._get_lr(logs)
        beta = self._get_beta(logs)

        # bitwidth distribution
        bw_pct = self._bw_distribution_pct(self._collect_bits())

        # also expose bw into keras logs (so it appears in history.history)
        for b in range(self.max_bits + 1):
            logs[f"bw_{b}bit_pct"] = float(bw_pct[b])

        # append to H5
        try:
            with h5py.File(self.path, "a") as hf:
                self._append_1d(hf, "epochs", int(epoch), dtype=np.int32)
                self._append_1d(hf, "time_sec", float(t_sec), dtype=np.float64)

                self._append_1d(hf, "loss", loss)
                self._append_1d(hf, "val_loss", val_loss)
                self._append_1d(hf, "accuracy", acc)
                self._append_1d(hf, "val_accuracy", val_acc)
                self._append_1d(hf, "ebops", ebops)
                self._append_1d(hf, "lr", lr)
                self._append_1d(hf, "beta", beta)

                self._append_2d_row(hf, "bw_pct", bw_pct)

                # static metadata once
                if "bits" not in hf:
                    hf.create_dataset("bits", data=np.arange(0, self.max_bits + 1, dtype=np.int32))

                hf.flush()
        except Exception:
            # don't crash training if disk write fails
            pass


class GradientNormLogger(keras.callbacks.Callback):
    """每隔若干 epoch 计算并打印（及可选写入 h5）各层 kernel 的梯度 L2 范数。

    用于验证 HGQ 深层网络的梯度消失 / 爆炸问题：
      - 梯度范数随层序号单调递减 → 梯度消失（靠近输入的层学不到东西）
      - 梯度范数随层序号单调递增 → 梯度爆炸

    HGQ 特有现象：
      当某层 kq.b（小数位宽）被压到接近 0 时，STE 的有效缩放因子趋近 0，
      该层以前的所有层都收不到任何梯度 —— 即使层数不多也会出现"量化梯度消失"。

    参数:
        log_every   : 每隔多少 epoch 记录一次（默认 100）
        output_path : h5 文件路径，None 则只打印不存储
        tape_batch  : 用于计算梯度的批次数（默认 1，节省时间）
    """

    def __init__(self, dataset_for_grad, log_every: int = 100, output_path: str | None = None):
        super().__init__()
        self.dataset_for_grad = dataset_for_grad  # 传入一个小批次 (X, y) 或 tf.data batch
        self.log_every = log_every
        self.output_path = output_path
        self._records = {}   # {layer_name: [norm_epoch0, norm_epoch1, ...]}
        self._epochs  = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_every != 0:
            return

        # 取一个 batch 计算梯度（始终用 sparse CE，避免 compiled_loss 跨 Keras 版本问题）
        x, y = self.dataset_for_grad
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(y, y_pred, from_logits=True)
            )

        grads = tape.gradient(loss, self.model.trainable_variables)

        # 按层汇总 kernel 梯度范数（用 id() 作 key，兼容 Keras 3 / TF2 所有版本）
        var_to_grad = {id(v): g for v, g in zip(self.model.trainable_variables, grads) if g is not None}

        print(f"\n[GradientNormLogger] epoch={epoch}")
        self._epochs.append(epoch)
        for layer in self.model.layers:
            if not hasattr(layer, 'kernel'):
                continue
            if id(layer.kernel) not in var_to_grad:
                continue
            norm = float(tf.norm(var_to_grad[id(layer.kernel)]).numpy())
            print(f"  {layer.name:12s}  kernel grad norm = {norm:.4e}")
            if layer.name not in self._records:
                self._records[layer.name] = []
            self._records[layer.name].append(norm)

        # 可选写入 h5
        if self.output_path is not None:
            try:
                with h5py.File(self.output_path, 'a') as hf:
                    hf.require_dataset('epochs', shape=(0,), maxshape=(None,), dtype=np.int32)
                    for name, norms in self._records.items():
                        key = f'grad_norm/{name}'
                        ds = hf.require_dataset(key, shape=(0,), maxshape=(None,), dtype=np.float32)
                        ds.resize(len(norms), axis=0)
                        ds[:] = norms
                    ep_ds = hf['epochs']
                    ep_ds.resize(len(self._epochs), axis=0)
                    ep_ds[:] = self._epochs
            except Exception:
                pass