import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # jax, torch, tensorflow
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random

import keras
import numpy as np

from data.data import get_data
from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PieceWiseSchedule
from keras.callbacks import LearningRateScheduler
from model.model import get_model_hgq
from utils.train_utils import cosine_decay_restarts_schedule
from utils.tf_device import get_tf_device
from utils.train_utils import TrainingTraceToH5


np.random.seed(42)
random.seed(42)

input_folder = 'data/dataset.h5'
output_folder = 'results/ram/'
batch_size = 33200
learning_rate = 5e-3
epochs = 20000

beta_sch_0 = 0
beta_sch_1 = epochs // 50
beta_sch_2 = epochs
beta_max = min(1e-3, 5e-7 * (epochs / 100))

device = get_tf_device()
os.makedirs(output_folder, exist_ok=True)


def ramanujan_sparse_matrix(fan_in: int, fan_out: int, degree: int, rng: np.random.Generator):
    """
    Build a fixed-degree bipartite sparse mask of shape (fan_in, fan_out).
    Each output neuron has 'degree' incoming edges (or fan_in if fan_in < degree).

    This is an expander-style sparse pattern; if you have a true Ramanujan
    adjacency for a given (fan_in, fan_out), you can replace this function
    by loading that adjacency.
    """
    degree_eff = min(degree, fan_in)
    mask = np.zeros((fan_in, fan_out), dtype=np.float32)

    for j in range(fan_out):
        # Choose 'degree_eff' distinct input indices for column j
        idx = rng.choice(fan_in, size=degree_eff, replace=False)
        mask[idx, j] = 1.0

    return mask


def apply_ramanujan_init(model: keras.Model, degree: int = 8, seed: int = 42):
    """
    Apply Ramanujan-style sparse initialization to all layers with a 'kernel'.

    For each kernel W:
      1) Flatten to (fan_in, fan_out).
      2) Build a sparse bipartite mask M using ramanujan_sparse_matrix.
      3) Sample values from N(0, sigma^2) with Glorot-like sigma.
      4) Set W = M * sampled_values, then reshape back and write to the layer.
    """
    rng = np.random.default_rng(seed)

    for layer in model.layers:
        if not hasattr(layer, 'kernel'):
            continue

        weights = layer.get_weights()
        if len(weights) == 0:
            continue

        kernel = weights[0]
        kernel_shape = kernel.shape

        # Handle kernels of arbitrary rank >= 2 (Dense, Conv, etc.)
        if kernel.ndim < 2:
            # Nothing to do; unexpected for standard layers
            continue

        fan_out = kernel_shape[-1]
        fan_in = int(np.prod(kernel_shape[:-1]))

        # Flatten kernel to 2D
        kernel_flat = kernel.reshape((fan_in, fan_out))

        # Build sparse mask and Glorot-like scaling
        mask = ramanujan_sparse_matrix(fan_in, fan_out, degree, rng)
        # Effective fan-in/out given sparsity; keep scale stable
        nonzero_per_col = mask.sum(axis=0).mean()
        eff_fan_in = max(1.0, nonzero_per_col)
        eff_fan_out = fan_out  # each output neuron kept
        sigma = np.sqrt(2.0 / (eff_fan_in + eff_fan_out))

        # Sample values only where mask != 0
        sampled = rng.normal(loc=0.0, scale=sigma, size=(fan_in, fan_out)).astype(np.float32)
        kernel_flat_new = sampled * mask

        # Reshape back and set weights
        kernel_new = kernel_flat_new.reshape(kernel_shape)
        weights[0] = kernel_new
        layer.set_weights(weights)

    print(f"Applied Ramanujan-style sparse initialization with degree={degree}, seed={seed}.")


if __name__ == '__main__':
    print('Starting training...')

    print('get dataset...')
    src = 'openml'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src=src)
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val = Dataset(X_val, y_val, batch_size, device)

    print('init EBOPs and pareto...')
    ebops_cb = FreeEBOPs()
    pareto_cb = ParetoFront(
        output_folder,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )

    print("get model...")
    model = get_model_hgq(3, 3)
    model.summary()

    # ---- Ramanujan-style sparse initialization ----
    # You can tune 'degree' if you want denser/sparser init.
    apply_ramanujan_init(model, degree=8, seed=42)

    print('set hyperparameters...')
    beta_sched = BetaScheduler(
        PieceWiseSchedule([
            (beta_sch_0, 5e-7, 'constant'),
            (beta_sch_1, 5e-7, 'log'),
            (beta_sch_2, beta_max, 'constant')
        ])
    )
    lr_scheduler = LearningRateScheduler(cosine_decay_restarts_schedule(learning_rate, epochs))

    trace_cb = TrainingTraceToH5(
        output_dir=output_folder,
        filename="training_trace.h5",
        max_bits=8,
        beta_callback=beta_sched,
    )

    callbacks = [ebops_cb, pareto_cb, beta_sched, lr_scheduler, trace_cb]

    opt = keras.optimizers.Adam()
    metrics = ['accuracy']
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, steps_per_execution=32)

    print('start training...')
    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print('training completed.')
    print(f"Trace saved to: {os.path.join(output_folder, 'training_trace.h5')}")