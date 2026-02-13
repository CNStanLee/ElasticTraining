import os

os.environ['KERAS_BACKEND'] = 'tensorflow'  # jax, torch, tensorflow
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import random
from math import cos, pi
import math
import tensorflow as tf
try:
    # Detect GPUs and enable memory growth if present. If no GPUs are found,
    # leave TensorFlow configured to use CPU only.
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # Best-effort: if memory growth setting fails, continue.
                pass
            device = 'GPU:0'
            print(f'GPU(s) detected ({len(gpus)}), using {device}')
        else:
            # No GPUs available: explicitly hide GPUs to ensure CPU-only execution
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            device = 'cpu:0'
            print('No GPUs detected, using CPU')
    except Exception:
        # Fallback to CPU if any error occurs during GPU detection
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
        device = 'cpu:0'
        print('GPU detection error, falling back to CPU')
except Exception:
    # If TensorFlow import/config fails, set a safe default device
    device = 'cpu:0'

import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import h5py
import json
import time
import glob
import re
from data.data import get_data
from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PBar, PieceWiseSchedule
from keras.callbacks import LearningRateScheduler
from model.model import get_model_hgq, get_model_hgqt
from utils.train_utils import cosine_decay_restarts_schedule

np.random.seed(42)
random.seed(42)

# paras
input_folder = 'data/dataset.h5'
output_folder = 'results/'
batch_size = 33200
learning_rate = 5e-3
fig_update_epochs = 2000
plot_window_epochs = 4000  # only plot the last N epochs to keep plotting fast
# how often to write full .mat/.npz data files (reduce I/O by increasing)
# Set to 1 to save every epoch (slow), larger values reduce disk writes.
data_save_epochs = 1000
# Toggle writing full .mat/.npz files. Set to False to disable heavy disk writes.
save_mat = False
# beta scheduling
epochs = 200000 # >50
# beta 0, 4000, 200000
beta_sch_0 = 0
beta_sch_1 = epochs // 50  # start ramping later (10% of training)
beta_sch_2 = epochs
# Use smaller max beta for shorter runs to avoid getting stuck
beta_max = min(1e-3, 5e-7 * (epochs / 100))  # scale max beta with epochs
#



def dynamic_cosine_restart_params(
    total_steps: int,
    base_m_mul: float = 0.94,
    base_T: int = 200_000,
    f0: float = 0.025,     # first cycle = 2.5% of training
    t_mul: float = 1.7,    # cycle length growth
):
    T = max(1, int(total_steps))

    first_decay_steps = max(2, int(round(f0 * T)))

    # Preserve similar overall decay strength when you change training length
    # But clamp m_mul to avoid it going too low for short runs
    m_mul = base_m_mul ** min(10, base_T / T)  # clamp exponent to max 10
    m_mul = max(0.5, m_mul)  # floor at 0.5 to avoid LR collapsing

    # Higher LR floor for short runs; approaches 1e-6 for long runs
    alpha = max(1e-6, 0.02 / T)

    # Smooth transition into alpha over ~1% of training
    alpha_steps = max(5, int(round(0.01 * T)))

    return dict(
        first_decay_steps=first_decay_steps,
        t_mul=t_mul,
        m_mul=m_mul,
        alpha=alpha,
        alpha_steps=alpha_steps,
    )

device = 'cpu:0'  # your own flag; TF already forced to CPU

os.makedirs(os.path.dirname(input_folder), exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

from hgq.layers import QLayerBase
from keras import ops


class BitwidthDistributionMonitor(keras.callbacks.Callback):
    """Monitor and plot the distribution of bitwidths (percentage at each bit level) during training.

    This callback keeps the latest plot in memory and writes a single PNG/.mat pair at
    the end of training, overwriting previous files so the `results/` folder only contains
    the most recent images.
    """
    
    def __init__(self, max_bits: int = 8, output_dir: str = output_folder, resume: bool = False):
        super().__init__()
        self.max_bits = max_bits
        self.output_dir = output_dir
        self.resume = resume
        self.update_every = fig_update_epochs  # update figure/PNG every N epochs
        os.makedirs(self.output_dir, exist_ok=True)

        # Time-series data
        self.epochs: list[int] = []
        self.bw_history: list[list[float]] = []  # per-epoch percentage for bits 0..max_bits
        self.ebops_values: list[float] = []

        # Figure elements
        self.fig = None
        self.ax_bw = None
        self.ax_ebops = None
        self.bw_lines = []  # one line per bit (0..max_bits)
        self.ebops_line = None

    def _collect_all_bits(self) -> list[float]:
        """Collect all individual bitwidth values from quantizers in the model."""
        all_bits = []
        for layer in self.model._flatten_layers():
            if isinstance(layer, QLayerBase):
                # Input quantizer
                if hasattr(layer, '_iq') and layer._iq is not None and layer._iq.built:
                    bits = ops.convert_to_numpy(layer._iq.bits).flatten()
                    all_bits.extend(bits.tolist())
                # Kernel quantizer
                if hasattr(layer, 'kq') and layer.kq is not None and layer.kq.built:
                    bits = ops.convert_to_numpy(layer.kq.bits).flatten()
                    all_bits.extend(bits.tolist())
                # Output quantizer
                if hasattr(layer, '_oq') and layer._oq is not None and layer._oq.built:
                    bits = ops.convert_to_numpy(layer._oq.bits).flatten()
                    all_bits.extend(bits.tolist())
        return all_bits

    def _compute_distribution(self, all_bits: list[float]) -> list[float]:
        """Compute percentage distribution for each bit level (0, 1, 2, ..., max_bits)."""
        if not all_bits:
            return [0.0] * (self.max_bits + 1)
        
        total = len(all_bits)
        counts = [0] * (self.max_bits + 1)
        for b in all_bits:
            idx = int(round(b))
            idx = max(0, min(idx, self.max_bits))
            counts[idx] += 1
        
        return [100.0 * c / total for c in counts]

    def on_train_begin(self, logs=None):
        # If resuming, try to restore previous history from disk
        if self.resume:
            mat_path = os.path.join(self.output_dir, 'bitwidth.mat')
            npz_path = os.path.join(self.output_dir, 'bitwidth.mat.npz')
            restored = False
            try:
                if os.path.exists(mat_path):
                    data = sio.loadmat(mat_path)
                    if 'epoch' in data and 'bw_pct' in data:
                        epoch = np.array(data['epoch']).astype(int).flatten()
                        bw_pct = np.array(data['bw_pct']).astype(float)
                        if bw_pct.ndim == 2 and bw_pct.shape[1] == self.max_bits + 1:
                            self.epochs = epoch.tolist()
                            self.bw_history = bw_pct.tolist()
                            eb = data.get('ebops', None)
                            if eb is not None:
                                self.ebops_values = np.array(eb).astype(float).flatten().tolist()
                            restored = True
            except Exception:
                restored = False
            if not restored:
                try:
                    if os.path.exists(npz_path):
                        data = np.load(npz_path)
                        if 'epoch' in data and 'bw_pct' in data:
                            epoch = np.array(data['epoch']).astype(int).flatten()
                            bw_pct = np.array(data['bw_pct']).astype(float)
                            if bw_pct.ndim == 2 and bw_pct.shape[1] == self.max_bits + 1:
                                self.epochs = epoch.tolist()
                                self.bw_history = bw_pct.tolist()
                                if 'ebops' in data:
                                    self.ebops_values = np.array(data['ebops']).astype(float).flatten().tolist()
                                restored = True
                except Exception:
                    pass

        plt.ion()
        self.fig, (self.ax_bw, self.ax_ebops) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

        # Top: per-bit percentage over epochs
        self.ax_bw.set_title('Bitwidth Percentages per Epoch')
        self.ax_bw.set_ylabel('Percentage (%)')
        self.ax_bw.set_ylim(0.0, 100.0)

        epochs_label = 'Epoch'
        self.ax_ebops.set_xlabel(epochs_label)
        self.ax_ebops.set_ylabel('EBOPs (overall)')

        # One line per bit (0..max_bits)
        self.bw_lines = []
        for b in range(0, self.max_bits + 1):
            (line,) = self.ax_bw.plot([], [], label=f'{b}-bit')
            self.bw_lines.append(line)
        self.ax_bw.legend(loc='upper right', ncol=2, fontsize=8)

        # EBOPs line (bottom subplot)
        (self.ebops_line,) = self.ax_ebops.plot([], [], color='tab:red', label='EBOPs')
        self.ax_ebops.legend(loc='best')

        self.fig.tight_layout()
        # If we restored history, draw a first window now
        if self.epochs:
            x = np.array(self.epochs, dtype=np.int32)
            bw_arr = np.array(self.bw_history, dtype=np.float64)
            ebops_arr = np.array(self.ebops_values, dtype=np.float64) if self.ebops_values else np.array([], dtype=np.float64)

            if x.size > 0:
                if x.size > plot_window_epochs:
                    x_win = x[-plot_window_epochs:]
                    bw_win = bw_arr[-plot_window_epochs:, :]
                    ebops_win = ebops_arr[-plot_window_epochs:] if ebops_arr.size else ebops_arr
                else:
                    x_win = x
                    bw_win = bw_arr
                    ebops_win = ebops_arr

                if bw_win.size > 0:
                    for idx, line in enumerate(self.bw_lines):
                        if idx < bw_win.shape[1]:
                            line.set_data(x_win, bw_win[:, idx])
                    self.ax_bw.set_xlim(x_win.min(), x_win.max() + 1)
                    self.ax_bw.set_ylim(0.0, 100.0)

                if ebops_win.size:
                    valid_ebops = ebops_win[np.isfinite(ebops_win)]
                    if valid_ebops.size:
                        self.ebops_line.set_data(x_win, ebops_win)
                        vmin = valid_ebops.min()
                        vmax = valid_ebops.max()
                        if vmin == vmax:
                            vmin *= 0.5
                            vmax *= 1.5
                        self.ax_ebops.set_xlim(x_win.min(), x_win.max() + 1)
                        self.ax_ebops.set_ylim(max(vmin * 0.8, 1e-10), vmax * 1.2)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_end(self, epoch, logs=None):
        all_bits = self._collect_all_bits()
        distribution = self._compute_distribution(all_bits)
        
        if logs is not None:
            for i, pct in enumerate(distribution):
                logs[f'bw_{i}bit_pct'] = pct
        # Capture EBOPs from logs if available
        ebops_val = None
        if logs is not None and 'ebops' in logs:
            try:
                ebops_val = float(logs['ebops'])
            except Exception:
                ebops_val = None

        # Store history (use NaN if ebops is missing)
        self.epochs.append(int(epoch))
        self.bw_history.append(distribution)
        self.ebops_values.append(ebops_val if ebops_val is not None else float('nan'))

        # Fast per-epoch append to HDF5 (keeps per-epoch I/O small)
        try:
            h5_path = os.path.join(self.output_dir, 'bitwidth.h5')
            with h5py.File(h5_path, 'a') as hf:
                # epochs dataset
                if 'epochs' not in hf:
                    hf.create_dataset('epochs', data=np.array(self.epochs, dtype=np.int32), maxshape=(None,), chunks=True)
                else:
                    d = hf['epochs']
                    d.resize(d.shape[0] + 1, axis=0)
                    d[-1] = int(epoch)

                # bw_pct dataset (2D)
                bw_arr_row = np.array(distribution, dtype=np.float64).reshape(1, -1)
                if 'bw_pct' not in hf:
                    hf.create_dataset('bw_pct', data=np.array(self.bw_history, dtype=np.float64), maxshape=(None, bw_arr_row.shape[1]), chunks=True)
                else:
                    d = hf['bw_pct']
                    d.resize(d.shape[0] + 1, axis=0)
                    d[-1, :] = bw_arr_row

                # ebops dataset
                eb_val = ebops_val if ebops_val is not None else float('nan')
                if 'ebops' not in hf:
                    hf.create_dataset('ebops', data=np.array(self.ebops_values, dtype=np.float64), maxshape=(None,), chunks=True)
                else:
                    d = hf['ebops']
                    d.resize(d.shape[0] + 1, axis=0)
                    d[-1] = eb_val
                hf.flush()
        except Exception:
            # HDF5 not available or append failed; continue and fall back to periodic .mat saves
            pass

        # lightweight resume metadata (small, cheap write) so we can resume at any epoch
        try:
            resume_meta = {'epoch': int(epoch), 'timestamp': time.time(), 'n': len(self.epochs)}
            with open(os.path.join(self.output_dir, 'resume.json'), 'w') as f:
                json.dump(resume_meta, f)
        except Exception:
            pass

        # Only update figure and PNG every `update_every` epochs or on last epoch
        do_update = (
            (epoch % self.update_every == 0)
            or (hasattr(self, 'params') and isinstance(self.params, dict) and epoch + 1 == self.params.get('epochs'))
        )

        if do_update and self.fig is not None and self.ax_bw is not None and self.ax_ebops is not None:
            x = np.array(self.epochs, dtype=np.int32)
            bw_arr = np.array(self.bw_history, dtype=np.float64)  # shape: [n_epochs, max_bits+1]
            ebops_arr = np.array(self.ebops_values, dtype=np.float64)

            # Restrict plotting to last plot_window_epochs points to keep it fast
            if x.size > 0:
                if x.size > plot_window_epochs:
                    x_win = x[-plot_window_epochs:]
                    bw_win = bw_arr[-plot_window_epochs:, :]
                    ebops_win = ebops_arr[-plot_window_epochs:]
                else:
                    x_win = x
                    bw_win = bw_arr
                    ebops_win = ebops_arr

                # Update per-bit percentage lines for bits 0..max_bits
                if bw_win.size > 0:
                    for idx, line in enumerate(self.bw_lines):
                        if idx < bw_win.shape[1]:  # guard in case of mismatch
                            line.set_data(x_win, bw_win[:, idx])

                    self.ax_bw.set_xlim(x_win.min(), x_win.max() + 1)
                    self.ax_bw.set_ylim(0.0, 100.0)

                # Update EBOPs line
                valid_ebops = ebops_win[np.isfinite(ebops_win)]
                if valid_ebops.size > 0:
                    self.ebops_line.set_data(x_win, ebops_win)
                    vmin = valid_ebops.min()
                    vmax = valid_ebops.max()
                    if vmin == vmax:
                        vmin *= 0.5
                        vmax *= 1.5
                    self.ax_ebops.set_xlim(x_win.min(), x_win.max() + 1)
                    self.ax_ebops.set_ylim(max(vmin * 0.8, 1e-10), vmax * 1.2)

            self.fig.canvas.draw_idle()
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass

            # Save PNG on update epochs
            png_path = os.path.join(self.output_dir, 'bitwidth.png')
            try:
                self.fig.savefig(png_path, bbox_inches='tight')
            except Exception:
                pass

        # Save full history files only at configured intervals to reduce I/O
        should_save = (
            (epoch % data_save_epochs == 0)
            or (hasattr(self, 'params') and isinstance(self.params, dict) and epoch + 1 == self.params.get('epochs'))
        )

        if should_save and save_mat:
            mat_path = os.path.join(self.output_dir, 'bitwidth.mat')
            try:
                bw_arr = np.array(self.bw_history, dtype=np.float64)
                if bw_arr.size == 0:
                    bw_pct = np.empty((0, self.max_bits + 1), dtype=np.float64)
                else:
                    # keep percentages for bits 0..max_bits
                    bw_pct = bw_arr[:, : self.max_bits + 1]
                sio.savemat(
                    mat_path,
                    {
                        'epoch': np.array(self.epochs, dtype=np.int32),
                        'bw_pct': bw_pct,
                        'bits': np.arange(0, self.max_bits + 1, dtype=np.int32),
                        'ebops': np.array(self.ebops_values, dtype=np.float64),
                    },
                )
            except Exception:
                try:
                    bw_arr = np.array(self.bw_history, dtype=np.float64)
                    if bw_arr.size == 0:
                        bw_pct = np.empty((0, self.max_bits + 1), dtype=np.float64)
                    else:
                        bw_pct = bw_arr[:, : self.max_bits + 1]
                    np.savez_compressed(
                        os.path.join(self.output_dir, 'bitwidth.mat.npz'),
                        epoch=np.array(self.epochs, dtype=np.int32),
                        bw_pct=bw_pct,
                        bits=np.arange(0, self.max_bits + 1, dtype=np.int32),
                        ebops=np.array(self.ebops_values, dtype=np.float64),
                    )
                except Exception:
                    pass

    def on_train_end(self, logs=None):
        # All data and plots are handled in on_epoch_end (with throttling)
        return


class NormalizedAccEbopsScatter(keras.callbacks.Callback):
    def __init__(self, use_validation: bool = True, output_dir: str = output_folder, resume: bool = False):
        super().__init__()
        self.use_validation = use_validation
        self.acc_values: list[float] = []
        self.ebops_values: list[float] = []
        self.acc_min = float('inf')
        self.acc_max = float('-inf')
        self.fig = None
        self.ax = None
        self.scatter = None
        self.output_dir = output_dir
        self.resume = resume
        self.update_every = fig_update_epochs  # update figure/PNG every N epochs
        os.makedirs(self.output_dir, exist_ok=True)

    def _compute_pareto_mask(self, acc_vals=None, ebops_vals=None) -> list[bool]:
        """Compute which points are on the Pareto frontier.
        
        Pareto optimal: higher accuracy AND lower ebops is better.
        A point is on the frontier if no other point dominates it.
        """
        if acc_vals is None or ebops_vals is None:
            acc_vals = self.acc_values
            ebops_vals = self.ebops_values

        n = len(acc_vals)
        is_pareto = [True] * n
        
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # Point j dominates point i if j has higher acc AND lower ebops
                # (or equal in one and strictly better in the other)
                if (acc_vals[j] >= acc_vals[i] and 
                    ebops_vals[j] <= ebops_vals[i] and
                    (acc_vals[j] > acc_vals[i] or 
                     ebops_vals[j] < ebops_vals[i])):
                    is_pareto[i] = False
                    break
        
        return is_pareto

    def on_train_begin(self, logs=None):
        # If resuming, try to restore previous history from disk
        if self.resume:
            mat_path = os.path.join(self.output_dir, 'acc_ebops.mat')
            npz_path = os.path.join(self.output_dir, 'acc_ebops.mat.npz')
            restored = False
            try:
                if os.path.exists(mat_path):
                    data = sio.loadmat(mat_path)
                    if 'acc_values' in data and 'ebops_values' in data:
                        self.acc_values = np.array(data['acc_values']).astype(float).flatten().tolist()
                        self.ebops_values = np.array(data['ebops_values']).astype(float).flatten().tolist()
                        restored = True
            except Exception:
                restored = False
            if not restored:
                try:
                    if os.path.exists(npz_path):
                        data = np.load(npz_path)
                        if 'acc_values' in data and 'ebops_values' in data:
                            self.acc_values = np.array(data['acc_values']).astype(float).flatten().tolist()
                            self.ebops_values = np.array(data['ebops_values']).astype(float).flatten().tolist()
                            restored = True
                except Exception:
                    pass

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title('Accuracy vs EBOPs')
        self.ax.set_xlabel('EBOPs')
        self.ax.set_ylabel('Accuracy')
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_box_aspect(1)  # 1:1 box shape without forcing equal data units
        self.scatter = self.ax.scatter([], [], s=18, alpha=0.8)

        # If we restored history, draw a first window now
        if self.acc_values and self.ebops_values:
            acc_arr = np.array(self.acc_values, dtype=np.float64)
            ebops_arr = np.array(self.ebops_values, dtype=np.float64)
            n = acc_arr.size
            if n > 0:
                if n > plot_window_epochs:
                    acc_win = acc_arr[-plot_window_epochs:]
                    ebops_win = ebops_arr[-plot_window_epochs:]
                else:
                    acc_win = acc_arr
                    ebops_win = ebops_arr

                self.scatter.set_offsets(np.column_stack([ebops_win, acc_win]))
                pareto_mask = self._compute_pareto_mask(acc_win.tolist(), ebops_win.tolist())
                colors = ['red' if is_p else 'blue' for is_p in pareto_mask]
                self.scatter.set_facecolors(colors)

                if ebops_win.size:
                    ebops_min = float(ebops_win.min())
                    ebops_max = float(ebops_win.max())
                    margin = (ebops_max - ebops_min) * 0.1 if ebops_max > ebops_min else ebops_max * 0.1
                    self.ax.set_xlim(max(0, ebops_min - margin), ebops_max + margin)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        acc_key = 'val_accuracy' if self.use_validation and 'val_accuracy' in logs else 'accuracy'
        if acc_key not in logs or 'ebops' not in logs:
            return
        acc = float(logs[acc_key])
        ebops = float(logs['ebops'])

        self.acc_values.append(acc)
        self.ebops_values.append(ebops)

        # Fast per-epoch append to HDF5 and write resume metadata
        try:
            h5_path = os.path.join(self.output_dir, 'acc_ebops.h5')
            with h5py.File(h5_path, 'a') as hf:
                # epochs dataset
                if 'epochs' not in hf:
                    hf.create_dataset('epochs', data=np.arange(len(self.acc_values), dtype=np.int32), maxshape=(None,), chunks=True)
                else:
                    d = hf['epochs']
                    d.resize(d.shape[0] + 1, axis=0)
                    d[-1] = int(epoch)

                # acc dataset
                acc_row = np.array([acc], dtype=np.float64)
                if 'acc' not in hf:
                    hf.create_dataset('acc', data=np.array(self.acc_values, dtype=np.float64), maxshape=(None,), chunks=True)
                else:
                    d = hf['acc']
                    d.resize(d.shape[0] + 1, axis=0)
                    d[-1] = acc_row[0]

                # ebops dataset
                eb_row = np.array([ebops], dtype=np.float64)
                if 'ebops' not in hf:
                    hf.create_dataset('ebops', data=np.array(self.ebops_values, dtype=np.float64), maxshape=(None,), chunks=True)
                else:
                    d = hf['ebops']
                    d.resize(d.shape[0] + 1, axis=0)
                    d[-1] = eb_row[0]
                hf.flush()
        except Exception:
            pass

        try:
            resume_meta = {'epoch': int(epoch), 'timestamp': time.time(), 'n': len(self.acc_values)}
            with open(os.path.join(self.output_dir, 'resume.json'), 'w') as f:
                json.dump(resume_meta, f)
        except Exception:
            pass

        # Only update figure and save PNG every `update_every` epochs or on last epoch
        do_update = (
            (epoch % self.update_every == 0)
            or (hasattr(self, 'params') and isinstance(self.params, dict) and epoch + 1 == self.params.get('epochs'))
        )

        if do_update and self.scatter is not None:
            acc_arr = np.array(self.acc_values, dtype=np.float64)
            ebops_arr = np.array(self.ebops_values, dtype=np.float64)

            # Restrict plotting to last plot_window_epochs points
            n = acc_arr.size
            if n > 0:
                if n > plot_window_epochs:
                    acc_win = acc_arr[-plot_window_epochs:]
                    ebops_win = ebops_arr[-plot_window_epochs:]
                else:
                    acc_win = acc_arr
                    ebops_win = ebops_arr

                self.scatter.set_offsets(np.column_stack([ebops_win, acc_win]))

                # Compute Pareto frontier and set colors (on the windowed data only)
                pareto_mask = self._compute_pareto_mask(acc_win.tolist(), ebops_win.tolist())
                colors = ['red' if is_p else 'blue' for is_p in pareto_mask]
                self.scatter.set_facecolors(colors)

                # Set x-axis limits based on ebops range
                if ebops_win.size:
                    ebops_min = float(ebops_win.min())
                    ebops_max = float(ebops_win.max())
                    margin = (ebops_max - ebops_min) * 0.1 if ebops_max > ebops_min else ebops_max * 0.1
                    self.ax.set_xlim(max(0, ebops_min - margin), ebops_max + margin)

            self.ax.set_title(f'{acc_key} vs EBOPs (epoch {epoch + 1})')
            self.fig.canvas.draw_idle()
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass
            # Save current scatter image (PNG) on update epochs
            png_path = os.path.join(self.output_dir, 'acc_ebops.png')
            try:
                if self.fig is not None:
                    self.fig.savefig(png_path, bbox_inches='tight')
            except Exception:
                pass

        # Save full history files only at configured intervals to reduce I/O
        should_save = (
            (epoch % data_save_epochs == 0)
            or (hasattr(self, 'params') and isinstance(self.params, dict) and epoch + 1 == self.params.get('epochs'))
        )

        if should_save and save_mat:
            mat_path = os.path.join(self.output_dir, 'acc_ebops.mat')
            try:
                pareto_full = self._compute_pareto_mask()
                sio.savemat(
                    mat_path,
                    {
                        'acc_values': np.array(self.acc_values),
                        'ebops_values': np.array(self.ebops_values),
                        'pareto_mask': np.array(pareto_full),
                        'epoch': int(epoch),
                    },
                )
            except Exception:
                try:
                    np.savez_compressed(
                        os.path.join(self.output_dir, 'acc_ebops.mat.npz'),
                        acc_values=np.array(self.acc_values),
                        ebops_values=np.array(self.ebops_values),
                        epoch=int(epoch),
                    )
                except Exception:
                    pass
    def on_train_end(self, logs=None):
        # Nothing extra to do on train end since we're saving each epoch (files overwritten)
        return


class LrBetaMonitor(keras.callbacks.Callback):
    """Record learning rate and beta per epoch and save to a .mat file.

    Also maintains a figure with LR/Beta vs epoch. To avoid slowing
    training, the figure/PNG are only updated every ``update_every`` epochs
    (data are still written to the .mat file every epoch).
    """

    def __init__(self, beta_callback=None, output_dir: str = output_folder, resume: bool = False, update_every: int = fig_update_epochs):
        super().__init__()
        self.beta_callback = beta_callback
        self.output_dir = output_dir
        self.resume = resume
        self.update_every = max(1, int(update_every))
        os.makedirs(self.output_dir, exist_ok=True)
        self.lrs: list[float] = []
        self.betas: list[float] = []
        self.epochs: list[int] = []
        self.fig = None
        self.ax_lr = None
        self.ax_beta = None
        self.lr_line = None
        self.beta_line = None

    def _to_float_safe(self, value):
        try:
            return float(ops.convert_to_numpy(value))
        except Exception:
            try:
                return float(value)
            except Exception:
                return None

    def on_train_begin(self, logs=None):
        # If resuming, try to restore previous history from disk
        if self.resume:
            mat_path = os.path.join(self.output_dir, 'lr_beta.mat')
            npz_path = os.path.join(self.output_dir, 'lr_beta.mat.npz')
            restored = False
            try:
                if os.path.exists(mat_path):
                    data = sio.loadmat(mat_path)
                    if 'epoch' in data and 'lr' in data and 'beta' in data:
                        self.epochs = np.array(data['epoch']).astype(int).flatten().tolist()
                        self.lrs = np.array(data['lr']).astype(float).flatten().tolist()
                        self.betas = np.array(data['beta']).astype(float).flatten().tolist()
                        restored = True
            except Exception:
                restored = False
            if not restored:
                try:
                    if os.path.exists(npz_path):
                        data = np.load(npz_path)
                        if 'epoch' in data and 'lr' in data and 'beta' in data:
                            self.epochs = np.array(data['epoch']).astype(int).flatten().tolist()
                            self.lrs = np.array(data['lr']).astype(float).flatten().tolist()
                            self.betas = np.array(data['beta']).astype(float).flatten().tolist()
                            restored = True
                except Exception:
                    pass

        plt.ion()
        self.fig, (self.ax_lr, self.ax_beta) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        self.ax_lr.set_title('Learning Rate and Beta vs Epoch')
        self.ax_lr.set_ylabel('LR')
        self.ax_beta.set_ylabel('Beta')
        self.ax_beta.set_xlabel('Epoch')

        # Use log scale as these values are typically small
        self.ax_lr.set_yscale('log')
        self.ax_beta.set_yscale('log')

        (self.lr_line,) = self.ax_lr.plot([], [], color='tab:blue', label='lr')
        (self.beta_line,) = self.ax_beta.plot([], [], color='tab:orange', label='beta')

        self.ax_lr.legend(loc='best')
        self.ax_beta.legend(loc='best')

        self.fig.tight_layout()
        # If we restored history on resume, draw it once now
        if self.epochs:
            x = np.array(self.epochs, dtype=np.int32)
            lr_arr = np.array(self.lrs, dtype=np.float64)
            beta_arr = np.array(self.betas, dtype=np.float64)
            self.lr_line.set_data(x, lr_arr)
            self.beta_line.set_data(x, beta_arr)

            if len(x) > 0:
                self.ax_lr.set_xlim(x.min(), x.max() + 1)

                def _set_ylim(ax, arr):
                    valid = arr[np.isfinite(arr)]
                    if valid.size == 0:
                        return
                    vmin = valid.min()
                    vmax = valid.max()
                    if vmin == vmax:
                        vmin *= 0.5
                        vmax *= 1.5
                    ax.set_ylim(max(vmin * 0.8, 1e-10), vmax * 1.2)

                _set_ylim(self.ax_lr, lr_arr)
                _set_ylim(self.ax_beta, beta_arr)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_end(self, epoch, logs=None):
        lr_val = None
        beta_val = None

        if logs is not None:
            if 'lr' in logs:
                lr_val = logs['lr']
            elif 'learning_rate' in logs:
                lr_val = logs['learning_rate']
            if 'beta' in logs:
                beta_val = logs['beta']

        # Fallback: read current optimizer learning rate
        if lr_val is None and getattr(self.model, 'optimizer', None) is not None:
            lr_val = getattr(self.model.optimizer, 'learning_rate', None)

        # Optional: try to read beta from the scheduler callback if provided
        if beta_val is None and self.beta_callback is not None:
            for attr in ('beta', 'current_beta', 'value'):
                candidate = getattr(self.beta_callback, attr, None)
                if candidate is not None:
                    beta_val = candidate
                    break

        lr_val = self._to_float_safe(lr_val)
        beta_val = self._to_float_safe(beta_val)

        self.epochs.append(int(epoch))
        self.lrs.append(lr_val if lr_val is not None else float('nan'))
        self.betas.append(beta_val if beta_val is not None else float('nan'))
        # Update in-memory plot less frequently to avoid slowdown
        do_plot_update = (
            (epoch % self.update_every == 0)
            or (hasattr(self, 'params') and isinstance(self.params, dict) and epoch + 1 == self.params.get('epochs'))
        )

        if do_plot_update and self.fig is not None and self.lr_line is not None and self.beta_line is not None:
            x = np.array(self.epochs, dtype=np.int32)
            lr_arr = np.array(self.lrs, dtype=np.float64)
            beta_arr = np.array(self.betas, dtype=np.float64)

            # Restrict plotting to last plot_window_epochs points
            if x.size > 0:
                if x.size > plot_window_epochs:
                    x_win = x[-plot_window_epochs:]
                    lr_win = lr_arr[-plot_window_epochs:]
                    beta_win = beta_arr[-plot_window_epochs:]
                else:
                    x_win = x
                    lr_win = lr_arr
                    beta_win = beta_arr

                self.lr_line.set_data(x_win, lr_win)
                self.beta_line.set_data(x_win, beta_win)

                self.ax_lr.set_xlim(x_win.min(), x_win.max() + 1)

                # Avoid issues when all values are NaN
                def _set_ylim(ax, arr):
                    valid = arr[np.isfinite(arr)]
                    if valid.size == 0:
                        return
                    vmin = valid.min()
                    vmax = valid.max()
                    if vmin == vmax:
                        vmin *= 0.5
                        vmax *= 1.5
                    ax.set_ylim(max(vmin * 0.8, 1e-10), vmax * 1.2)

                _set_ylim(self.ax_lr, lr_win)
                _set_ylim(self.ax_beta, beta_win)

            self.fig.canvas.draw_idle()

        # Save full history files only at configured intervals to reduce I/O
        should_save = (
            (epoch % data_save_epochs == 0)
            or (hasattr(self, 'params') and isinstance(self.params, dict) and epoch + 1 == self.params.get('epochs'))
        )

        png_path = os.path.join(self.output_dir, 'lr_beta.png')
        mat_path = os.path.join(self.output_dir, 'lr_beta.mat')
        try:
            if do_plot_update and self.fig is not None:
                try:
                    self.fig.savefig(png_path, bbox_inches='tight')
                except Exception:
                    pass
            # Append to HDF5 per-epoch (fast)
            try:
                h5_path = os.path.join(self.output_dir, 'lr_beta.h5')
                with h5py.File(h5_path, 'a') as hf:
                    if 'epochs' not in hf:
                        hf.create_dataset('epochs', data=np.array(self.epochs, dtype=np.int32), maxshape=(None,), chunks=True)
                    else:
                        d = hf['epochs']
                        d.resize(d.shape[0] + 1, axis=0)
                        d[-1] = int(epoch)

                    if 'lr' not in hf:
                        hf.create_dataset('lr', data=np.array(self.lrs, dtype=np.float64), maxshape=(None,), chunks=True)
                    else:
                        d = hf['lr']
                        d.resize(d.shape[0] + 1, axis=0)
                        d[-1] = float(self.lrs[-1])

                    if 'beta' not in hf:
                        hf.create_dataset('beta', data=np.array(self.betas, dtype=np.float64), maxshape=(None,), chunks=True)
                    else:
                        d = hf['beta']
                        d.resize(d.shape[0] + 1, axis=0)
                        d[-1] = float(self.betas[-1])
                    hf.flush()
            except Exception:
                pass

            # lightweight resume metadata
            try:
                resume_meta = {'epoch': int(epoch), 'timestamp': time.time(), 'n': len(self.epochs)}
                with open(os.path.join(self.output_dir, 'resume.json'), 'w') as f:
                    json.dump(resume_meta, f)
            except Exception:
                pass

            # Still write .mat/npz on configured intervals if enabled
            if should_save and save_mat:
                try:
                    sio.savemat(
                        mat_path,
                        {
                            'epoch': np.array(self.epochs, dtype=np.int32),
                            'lr': np.array(self.lrs, dtype=np.float64),
                            'beta': np.array(self.betas, dtype=np.float64),
                        },
                    )
                except Exception:
                    try:
                        np.savez_compressed(
                            os.path.join(self.output_dir, 'lr_beta.mat.npz'),
                            epoch=np.array(self.epochs, dtype=np.int32),
                            lr=np.array(self.lrs, dtype=np.float64),
                            beta=np.array(self.betas, dtype=np.float64),
                        )
                    except Exception:
                        pass
        except Exception:
            pass

    def on_train_end(self, logs=None):
        # Final save to make sure latest plot/data are on disk
        png_path = os.path.join(self.output_dir, 'lr_beta.png')
        mat_path = os.path.join(self.output_dir, 'lr_beta.mat')
        try:
            if self.fig is not None:
                try:
                    self.fig.savefig(png_path, bbox_inches='tight')
                except Exception:
                    pass
            if save_mat:
                sio.savemat(
                    mat_path,
                    {
                        'epoch': np.array(self.epochs, dtype=np.int32),
                        'lr': np.array(self.lrs, dtype=np.float64),
                        'beta': np.array(self.betas, dtype=np.float64),
                    },
                )
        except Exception:
            try:
                if save_mat:
                    np.savez_compressed(
                        os.path.join(self.output_dir, 'lr_beta.mat.npz'),
                        epoch=np.array(self.epochs, dtype=np.int32),
                        lr=np.array(self.lrs, dtype=np.float64),
                        beta=np.array(self.betas, dtype=np.float64),
                    )
            except Exception:
                pass


class PBarWithOffset(PBar):
    """Wrapper around hgq.utils.sugar.PBar that supports resume.

    It initializes the underlying tqdm progress bar with a non-zero
    starting epoch so that, when resuming training, the left-hand
    counter reflects the true epoch count instead of starting from 0.
    """

    def __init__(self, *args, start_epoch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_epoch = max(0, int(start_epoch))
        self._offset_applied = False

    def on_epoch_begin(self, epoch, logs=None):
        # Let the original PBar create tqdm (with total=self.params['epochs'])
        super().on_epoch_begin(epoch, logs)
        # Once, immediately after tqdm is created, set its starting position
        if (
            not self._offset_applied
            and self._start_epoch > 0
            and getattr(self, "pbar", None) is not None
        ):
            try:
                # Move tqdm's internal counter to the resume epoch
                self.pbar.n = self._start_epoch
                # Keep tqdm's printing state consistent
                if hasattr(self.pbar, "last_print_n"):
                    self.pbar.last_print_n = self.pbar.n
                self.pbar.refresh()
            except Exception:
                # If anything goes wrong, fall back to default behavior
                pass
            self._offset_applied = True


def hgq_training_run(new_run: bool = False):
    print('Running HGQ training...')
    src = 'openml'  # 'cernbox'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src=src)
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val = Dataset(X_val, y_val, batch_size, device)
    ebops = FreeEBOPs()
    pareto = ParetoFront(
        output_folder,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )

    # beta and lr schedulers
    beta_sched = BetaScheduler(PieceWiseSchedule([(beta_sch_0, 5e-7, 'constant'), (beta_sch_1, 5e-7, 'log'), (beta_sch_2, beta_max, 'constant')]))
    p = dynamic_cosine_restart_params(epochs)

    lr_sched = LearningRateScheduler(
        cosine_decay_restarts_schedule(
            learning_rate,
            p["first_decay_steps"],
            t_mul=p["t_mul"],
            m_mul=p["m_mul"],
            alpha=p["alpha"],
            alpha_steps=p["alpha_steps"],
        )
    )

    # Resume logic: by default resume unless `new_run` is True
    resume = not bool(new_run)
    latest_ckpt = None
    last_epoch = -1
    if resume:
        ckpts = glob.glob(os.path.join(output_folder, '*.keras'))
        if ckpts:
            # Prefer parsing epoch=NN from filename; fallback to modification time
            max_epoch = -1
            chosen = None
            for c in ckpts:
                m = re.search(r'epoch=(\d+)', os.path.basename(c))
                if m:
                    e = int(m.group(1))
                    if e > max_epoch:
                        max_epoch = e
                        chosen = c
            if chosen is None:
                # fallback to newest file
                chosen = max(ckpts, key=os.path.getmtime)
                max_epoch = -1
            latest_ckpt = chosen
            last_epoch = max_epoch
            print(f'Found checkpoint {latest_ckpt} (epoch {last_epoch})')
        else:
            resume = False

    # Model loading: prefer loading full saved model (restores optimizer state). If that fails, load weights.
    model = None
    if resume and latest_ckpt is not None:
        try:
            model = keras.models.load_model(latest_ckpt)
            print(f'Loaded full model+optimizer from {latest_ckpt}')
        except Exception:
            try:
                model = get_model_hgq(3, 3)
                model.load_weights(latest_ckpt)
                print(f'Loaded weights from {latest_ckpt} into new model')
            except Exception:
                print('Failed to load checkpoint; starting new model')
                model = None

    if model is None:
        model = get_model_hgq(3, 3)

    # Compile if needed (models loaded with keras.models.load_model are usually compiled)
    try:
        _ = model.optimizer
        compiled = True
    except Exception:
        compiled = False

    opt = keras.optimizers.Adam()
    metrics = ['accuracy']
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if not compiled:
        model.compile(optimizer=opt, loss=loss, metrics=metrics, steps_per_execution=32)

    # Compute initial_epoch for resuming
    if resume and last_epoch >= 0:
        initial_epoch = last_epoch + 1
    else:
        initial_epoch = 0

    # Prepare callbacks (plot callbacks will only write final files)
    pbar = PBarWithOffset(
        'loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.4f}/{val_accuracy:.4f} - lr: {learning_rate:.2e} - beta: {beta:.1e}',
        start_epoch=initial_epoch,
    )
    lr_beta_monitor = LrBetaMonitor(beta_callback=beta_sched, output_dir=output_folder, resume=resume, update_every=fig_update_epochs)
    bw_monitor = BitwidthDistributionMonitor(max_bits=8, output_dir=output_folder, resume=resume)
    acc_ebops_monitor = NormalizedAccEbopsScatter(output_dir=output_folder, resume=resume)

    callbacks = [ebops, lr_sched, beta_sched, pbar, pareto, lr_beta_monitor, acc_ebops_monitor, bw_monitor]

    # Start/continue training
    model.fit(dataset_train, epochs=epochs, initial_epoch=initial_epoch, validation_data=dataset_val, callbacks=callbacks, verbose=0)  # type: ignore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new', action='store_true', help='Start a new training run from scratch (do not resume)')
    args = parser.parse_args()
    hgq_training_run(new_run=args.new)
