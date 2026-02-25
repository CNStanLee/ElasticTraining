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