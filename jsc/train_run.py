import os

os.environ['KERAS_BACKEND'] = 'tensorflow'  # jax, torch, tensorflow
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import random
from math import cos, pi
import math
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import keras
import matplotlib.pyplot as plt
import numpy as np
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
# beta scheduling
epochs = 1000 # >50
# beta 0, 4000, 200000
beta_sch_0 = 0
beta_sch_1 = epochs // 10  # start ramping later (10% of training)
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
    """Monitor and plot the distribution of bitwidths (percentage at each bit level) during training."""
    
    def __init__(self, max_bits: int = 8):
        super().__init__()
        self.max_bits = max_bits
        self.fig = None
        self.ax = None
        self.bars = None

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
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.set_title('Bitwidth Distribution')
        self.ax.set_xlabel('Bitwidth')
        self.ax.set_ylabel('Percentage (%)')
        self.ax.set_ylim(0.0, 100.0)
        self.ax.set_xlim(-0.5, self.max_bits + 0.5)
        self.ax.set_xticks(range(self.max_bits + 1))
        
        # Initialize bars with zero height
        self.bars = self.ax.bar(range(self.max_bits + 1), [0] * (self.max_bits + 1), 
                                 color='steelblue', edgecolor='black', alpha=0.8)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_end(self, epoch, logs=None):
        all_bits = self._collect_all_bits()
        distribution = self._compute_distribution(all_bits)
        
        if logs is not None:
            for i, pct in enumerate(distribution):
                logs[f'bw_{i}bit_pct'] = pct
        
        if self.bars is not None:
            for bar, height in zip(self.bars, distribution):
                bar.set_height(height)
            
            # Add percentage labels on bars
            self.ax.set_title(f'Bitwidth Distribution (epoch {epoch + 1})')
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()


class NormalizedAccEbopsScatter(keras.callbacks.Callback):
    def __init__(self, use_validation: bool = True):
        super().__init__()
        self.use_validation = use_validation
        self.acc_values: list[float] = []
        self.ebops_values: list[float] = []
        self.acc_min = float('inf')
        self.acc_max = float('-inf')
        self.fig = None
        self.ax = None
        self.scatter = None

    def _compute_pareto_mask(self) -> list[bool]:
        """Compute which points are on the Pareto frontier.
        
        Pareto optimal: higher accuracy AND lower ebops is better.
        A point is on the frontier if no other point dominates it.
        """
        n = len(self.acc_values)
        is_pareto = [True] * n
        
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # Point j dominates point i if j has higher acc AND lower ebops
                # (or equal in one and strictly better in the other)
                if (self.acc_values[j] >= self.acc_values[i] and 
                    self.ebops_values[j] <= self.ebops_values[i] and
                    (self.acc_values[j] > self.acc_values[i] or 
                     self.ebops_values[j] < self.ebops_values[i])):
                    is_pareto[i] = False
                    break
        
        return is_pareto

    def on_train_begin(self, logs=None):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title('Accuracy vs EBOPs')
        self.ax.set_xlabel('EBOPs')
        self.ax.set_ylabel('Accuracy')
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_box_aspect(1)  # 1:1 box shape without forcing equal data units
        self.scatter = self.ax.scatter([], [], s=18, alpha=0.8)
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

        if self.scatter is not None:
            self.scatter.set_offsets(np.column_stack([self.ebops_values, self.acc_values]))
            
            # Compute Pareto frontier and set colors
            pareto_mask = self._compute_pareto_mask()
            colors = ['red' if is_p else 'blue' for is_p in pareto_mask]
            self.scatter.set_facecolors(colors)
            
            # Set x-axis limits based on ebops range
            if self.ebops_values:
                ebops_min = min(self.ebops_values)
                ebops_max = max(self.ebops_values)
                margin = (ebops_max - ebops_min) * 0.1 if ebops_max > ebops_min else ebops_max * 0.1
                self.ax.set_xlim(max(0, ebops_min - margin), ebops_max + margin)
            
            self.ax.set_title(f'{acc_key} vs EBOPs (epoch {epoch + 1})')
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()


def hgq_training_run():
    print('Running HGQ training...')
    src = 'openml'  # 'cernbox'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src=src)
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val = Dataset(X_val, y_val, batch_size, device)
    model = get_model_hgq(3, 3)
    pbar = PBar(
        'loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.4f}/{val_accuracy:.4f} - lr: {learning_rate:.2e} - beta: {beta:.1e}'
    )
    ebops = FreeEBOPs()
    pareto = ParetoFront(
        output_folder,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )
    # beta_sched = BetaScheduler(PieceWiseSchedule([(0, 5e-7, 'constant'), (4000, 5e-7, 'log'), (200000, 1e-3, 'constant')]))
    beta_sched = BetaScheduler(PieceWiseSchedule([(beta_sch_0, 5e-7, 'constant'), (beta_sch_1, 5e-7, 'log'), (beta_sch_2, beta_max, 'constant')]))
    # lr_sched = LearningRateScheduler(
    #     cosine_decay_restarts_schedule(learning_rate, 4000, t_mul=1.0, m_mul=0.94, alpha=1e-6, alpha_steps=50)
    # )
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
    callbacks = [ebops, lr_sched, beta_sched, pbar, pareto]
    callbacks.append(NormalizedAccEbopsScatter())
    callbacks.append(BitwidthDistributionMonitor(max_bits=8))

    opt = keras.optimizers.Adam()
    metrics = ['accuracy']
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, steps_per_execution=32)

    model.fit(dataset_train, epochs=epochs, validation_data=dataset_val, callbacks=callbacks, verbose=0)  # type: ignore


if __name__ == '__main__':
    hgq_training_run()
