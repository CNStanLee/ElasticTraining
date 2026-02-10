import os

os.environ['KERAS_BACKEND'] = 'tensorflow'  # jax, torch, tensorflow
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import random
from math import cos, pi

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
device = 'cpu:0'  # your own flag; TF already forced to CPU

os.makedirs(os.path.dirname(input_folder), exist_ok=True)
os.makedirs(output_folder, exist_ok=True)


class NormalizedAccEbopsScatter(keras.callbacks.Callback):
    def __init__(self, use_validation: bool = True):
        super().__init__()
        self.use_validation = use_validation
        self.acc_values: list[float] = []
        self.ebops_values: list[float] = []
        self.acc_min = float('inf')
        self.acc_max = float('-inf')
        self.ebops_min = float('inf')
        self.ebops_max = float('-inf')
        self.fig = None
        self.ax = None
        self.scatter = None

    def on_train_begin(self, logs=None):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_title('Normalized Accuracy vs Normalized EBOPs')
        self.ax.set_xlabel('Normalized EBOPs')
        self.ax.set_ylabel('Normalized Accuracy')
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
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

        self.acc_min = min(self.acc_min, acc)
        self.acc_max = max(self.acc_max, acc)
        self.ebops_min = min(self.ebops_min, ebops)
        self.ebops_max = max(self.ebops_max, ebops)

        def normalize(value: float, vmin: float, vmax: float) -> float:
            if vmax <= vmin:
                return 0.0
            return (value - vmin) / (vmax - vmin)

        acc_norm = normalize(acc, self.acc_min, self.acc_max)
        ebops_norm = normalize(ebops, self.ebops_min, self.ebops_max)
        self.acc_values.append(acc)
        self.ebops_values.append(ebops)

        if self.scatter is not None:
            acc_norm_series = [normalize(v, self.acc_min, self.acc_max) for v in self.acc_values]
            ebops_norm_series = [normalize(v, self.ebops_min, self.ebops_max) for v in self.ebops_values]
            self.scatter.set_offsets(np.column_stack([ebops_norm_series, acc_norm_series]))
            self.ax.set_title(f'Normalized {acc_key} vs Normalized EBOPs (epoch {epoch + 1})')
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
    beta_sched = BetaScheduler(PieceWiseSchedule([(0, 5e-7, 'constant'), (4000, 5e-7, 'log'), (200000, 1e-3, 'constant')]))
    lr_sched = LearningRateScheduler(
        cosine_decay_restarts_schedule(learning_rate, 4000, t_mul=1.0, m_mul=0.94, alpha=1e-6, alpha_steps=50)
    )
    callbacks = [ebops, lr_sched, beta_sched, pbar, pareto]
    callbacks.append(NormalizedAccEbopsScatter())

    opt = keras.optimizers.Adam()
    metrics = ['accuracy']
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, steps_per_execution=32)

    model.fit(dataset_train, epochs=300, validation_data=dataset_val, callbacks=callbacks, verbose=0)  # type: ignore


if __name__ == '__main__':
    hgq_training_run()
