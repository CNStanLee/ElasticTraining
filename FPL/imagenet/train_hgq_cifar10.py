"""
train_hgq_cifar10.py – Train a HGQ-quantized ResNet-18 on CIFAR-10 and
diagnose gradient health (vanishing / exploding) during training.

Gradient diagnostics
--------------------
* At the start of each epoch, one mini-batch is run through GradientTape.
* Per-layer gradient L2 norms are recorded.
* Layers whose gradient norm falls below ``--grad-vanish-thresh`` are
  flagged as "vanishing"; those above ``--grad-explode-thresh`` as
  "exploding".
* A summary is printed each epoch and the full history is written to
  ``<output>/grad_norms.json``.

HGQ quantization
----------------
* Weights    : SAT_SYM, initial fractional bits = ``--bw-k``
* Activations: initial fractional bits = ``--bw-a``
* Beta (EBOPs regularisation) linearly ramps from 0 → ``--beta-max``
  after a warmup of ``--warmup-epochs`` epochs.

Example
-------
    python train_hgq_cifar10.py
    python train_hgq_cifar10.py --epochs 50 --bw-k 4 --bw-a 4
    python train_hgq_cifar10.py --grad-check-every 5
"""

# python train_hgq_cifar10.py --bw-k 6 --bw-a 6 --grad-check-every 5 --epochs 100

import os

os.environ['KERAS_BACKEND']        = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import re
import math

import numpy as np
import tensorflow as tf
import keras
from keras import layers

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for _gpu in gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
    print(f'GPUs available: {len(gpus)}')
else:
    print('No GPU found, using CPU')

from hgq.config import QuantizerConfigScope
from hgq.layers import (
    QConv2D,
    QBatchNormalization,
    QDense,
    QGlobalAveragePooling2D,
    QMaxPooling2D,
)
from hgq.utils.sugar import BetaScheduler, FreeEBOPs, PieceWiseSchedule

from utils.train_utils import WarmupCosineDecay


# ---------------------------------------------------------------------------
# HGQ ResNet-18 for CIFAR-10 (32×32 → lightweight stem)
# ---------------------------------------------------------------------------

def _q_basic_block(x, filters, stride=1, name='block'):
    """HGQ-quantized BasicBlock (two 3×3 QConv2D + QBatchNormalization)."""
    shortcut = x

    x = QConv2D(filters, 3, strides=stride, padding='same',
                use_bias=False, name=f'{name}_conv1')(x)
    x = QBatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)

    x = QConv2D(filters, 3, strides=1, padding='same',
                use_bias=False, name=f'{name}_conv2')(x)
    x = QBatchNormalization(name=f'{name}_bn2')(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = QConv2D(filters, 1, strides=stride, padding='same',
                           use_bias=False, name=f'{name}_proj')(shortcut)
        shortcut = QBatchNormalization(name=f'{name}_proj_bn')(shortcut)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name}_relu2')(x)
    return x


def _q_make_layer(x, filters, num_blocks, stride, name):
    x = _q_basic_block(x, filters, stride=stride, name=f'{name}_0')
    for i in range(1, num_blocks):
        x = _q_basic_block(x, filters, stride=1, name=f'{name}_{i}')
    return x


def get_hgq_resnet18_cifar(num_classes: int = 10,
                            init_bw_k: int = 4,
                            init_bw_a: int = 4):
    """
    HGQ-quantized ResNet-18 with a CIFAR-10-friendly stem.

    Parameters
    ----------
    num_classes : int
    init_bw_k   : initial fractional bits for weights
    init_bw_a   : initial fractional bits for activations

    Notes
    -----
    Weight quantizer uses KBI with b0=8, i0=1:
      step = 2^-(b-i-k) = 2^-(8-1-1) = 2^-6 ≈ 0.016
      Glorot-initialised weights (~±0.1) are representable without being
      rounded to 0.  The HGQ default KBI b0=4, i0=2 gives step=0.5 which
      rounds all Glorot weights to 0 → zero conv outputs → no gradient flow.
    Datalane uses i0=4 so the representable range is [-16, 16], safely
      covering BN-normalised activations throughout the network.
    """
    with (
        # KBI weight: b0=8, i0=1 → step=2^-6≈0.016, range [-2, ~2]
        QuantizerConfigScope(place='weight', q_type='kbi', b0=8, i0=1,
                             overflow_mode='SAT_SYM', trainable=True),
        QuantizerConfigScope(place='bias',   q_type='kbi', b0=8, i0=1,
                             overflow_mode='WRAP',    trainable=True),
        # KIF datalane: i0=4 → range [-16, 16]
        QuantizerConfigScope(place='datalane', overflow_mode='SAT',
                             i0=4, f0=init_bw_a),
    ):
        inputs = keras.Input(shape=(32, 32, 3), name='input')

        # Lightweight stem: single 3×3, stride-1, no maxpool
        x = QConv2D(64, 3, strides=1, padding='same',
                    use_bias=False, name='stem_conv')(inputs)
        x = QBatchNormalization(name='stem_bn')(x)
        x = layers.ReLU(name='stem_relu')(x)

        # Residual stages
        x = _q_make_layer(x, 64,  num_blocks=2, stride=1, name='layer1')
        x = _q_make_layer(x, 128, num_blocks=2, stride=2, name='layer2')
        x = _q_make_layer(x, 256, num_blocks=2, stride=2, name='layer3')
        x = _q_make_layer(x, 512, num_blocks=2, stride=2, name='layer4')

        x = QGlobalAveragePooling2D(name='gap')(x)
        outputs = QDense(num_classes, name='fc')(x)

    return keras.Model(inputs, outputs, name='hgq_resnet18_cifar10')


# ---------------------------------------------------------------------------
# CIFAR-10 dataloaders  (same as train_cifar10.py)
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_cifar10_dataloaders(batch_size: int = 128, num_workers: int = 4):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    mean = tf.constant(CIFAR10_MEAN, dtype=tf.float32)
    std  = tf.constant(CIFAR10_STD,  dtype=tf.float32)

    def normalize(img):
        img = tf.cast(img, tf.float32) / 255.0
        return (img - mean) / std

    def augment(img, label):
        img = tf.image.pad_to_bounding_box(img, 4, 4, 40, 40)
        img = tf.image.random_crop(img, size=[32, 32, 3])
        img = tf.image.random_flip_left_right(img)
        return normalize(img), label

    def preprocess(img, label):
        return normalize(img), label

    au = tf.data.AUTOTUNE if num_workers == 0 else num_workers

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(50000, reshuffle_each_iteration=True)
        .map(augment,    num_parallel_calls=au)
        .batch(batch_size, drop_remainder=True)
        .prefetch(au)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(preprocess, num_parallel_calls=au)
        .batch(batch_size, drop_remainder=False)
        .prefetch(au)
    )
    return train_ds, val_ds, 10


# ---------------------------------------------------------------------------
# Activation range monitor
# ---------------------------------------------------------------------------

# Key layer name fragments to monitor (one match per stage is enough)
_ACT_MONITOR_KEYWORDS = [
    'stem_relu',
    'layer1_1_relu2',
    'layer2_1_relu2',
    'layer3_1_relu2',
    'layer4_1_relu2',
    'gap',
    'fc',
]


class ActivationRangeMonitor(keras.callbacks.Callback):
    """
    After every ``check_every`` epochs, run ONE forward pass through a
    pre-built multi-output model and record per-layer activation stats.

    Using a single model avoids the OOM caused by building N sub-models
    (one per layer) that all stay alive simultaneously.
    Monitored layers are a small representative subset defined by
    ``_ACT_MONITOR_KEYWORDS``.
    """

    def __init__(self, probe_x, output_dir: str = 'results_hgq_cifar10/',
                 check_every: int = 1):
        super().__init__()
        self.probe_x      = probe_x
        self.output_dir   = output_dir
        self.check_every  = check_every
        self.history: list[dict] = []
        self._probe_model = None   # built lazily in on_train_begin
        self._probe_names: list[str] = []

    # ------------------------------------------------------------------
    def on_train_begin(self, logs=None):
        """Build a single multi-output probe model once."""
        selected_layers = []
        for layer in self.model.layers:
            if any(kw in layer.name for kw in _ACT_MONITOR_KEYWORDS):
                try:
                    # Check output is a single tensor (not a list)
                    if isinstance(layer.output, tf.Tensor):
                        selected_layers.append(layer)
                except Exception:
                    pass
        if not selected_layers:
            # fallback: pick every 5th layer
            selected_layers = self.model.layers[::5]

        outputs = [l.output for l in selected_layers]
        self._probe_names = [l.name for l in selected_layers]
        self._probe_model = keras.Model(
            inputs=self.model.input, outputs=outputs
        )
        print(f'[ActRange] monitoring {len(self._probe_names)} layers: '
              f'{self._probe_names}')

    # ------------------------------------------------------------------
    def _collect(self):
        if self._probe_model is None:
            return {}
        raw = self._probe_model(self.probe_x, training=False)
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        stats = {}
        for name, tensor in zip(self._probe_names, raw):
            a = tensor.numpy()
            stats[name] = {
                'min':       float(a.min()),
                'max':       float(a.max()),
                'mean':      float(a.mean()),
                'std':       float(a.std()),
                'zero_frac': float((a == 0).mean()),
            }
        return stats

    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.check_every != 0:
            return

        stats = self._collect()
        all_zero = [k for k, v in stats.items() if v['zero_frac'] > 0.99]

        print(f'\n[ActRange epoch {epoch+1}]  '
              f'layers_checked={len(stats)}  all_zero={len(all_zero)}')
        if all_zero:
            print(f'  ALL-ZERO (quantizer saturated?): {all_zero}')
        for k, v in stats.items():
            print(f'  {k:35s}  zero={v["zero_frac"]*100:5.1f}%  '
                  f'range=[{v["min"]:7.3f}, {v["max"]:7.3f}]  '
                  f'mean={v["mean"]:7.4f}')

        entry = {'epoch': epoch + 1, 'all_zero': all_zero, 'stats': stats}
        self.history.append(entry)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'act_ranges.json'), 'w') as f:
            json.dump(self.history, f, indent=2)


# ---------------------------------------------------------------------------
# Gradient diagnostics callback
# ---------------------------------------------------------------------------

class GradientDiagnostics(keras.callbacks.Callback):
    """
    Comprehensive gradient health monitor.

    Checks performed
    ----------------
    * ``on_train_begin``    – initial probe before any weight updates
    * ``on_train_batch_end``– probe for the first ``watch_first_batches`` steps
    * ``on_epoch_end``      – full probe every ``check_every`` epochs

    Metrics per layer
    -----------------
    * Gradient L2 norm
    * Weight L2 norm
    * grad/weight ratio  (< 1e-3 → update is negligible relative to weight magnitude)
    * % near-zero gradients  (STE saturation indicator)

    Flags
    -----
    Vanishing  : gradient norm < vanish_thresh
    Exploding  : gradient norm > explode_thresh
    """

    def __init__(self,
                 probe_batch,
                 loss_fn,
                 output_dir: str = 'results_hgq_cifar10/',
                 check_every: int = 1,
                 watch_first_batches: int = 0,
                 vanish_thresh: float = 1e-6,
                 explode_thresh: float = 1e2):
        super().__init__()
        self.probe_x, self.probe_y = probe_batch
        self.loss_fn             = loss_fn
        self.output_dir          = output_dir
        self.check_every         = check_every
        self.watch_first_batches = watch_first_batches
        self.vanish_thresh       = vanish_thresh
        self.explode_thresh      = explode_thresh
        self.history: list[dict] = []
        self._epoch              = 0

    # ------------------------------------------------------------------
    def _compute_stats(self):
        """
        Run GradientTape on the probe batch using training=True so that
        HGQ's straight-through estimator (STE) is active.
        Returns a dict: layer_name → {grad_norm, weight_norm, ratio, zero_frac}.
        """
        x, y = self.probe_x, self.probe_y
        tvars = self.model.trainable_variables
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss   = self.loss_fn(y, logits)
        grads = tape.gradient(loss, tvars)

        layer_data: dict[str, dict] = {}
        for var, grad in zip(tvars, grads):
            lname = var.name.split('/')[0]
            entry = layer_data.setdefault(
                lname, {'grad_norms': [], 'weight_norms': [], 'zero_fracs': []}
            )
            w_norm = float(tf.norm(var).numpy())
            entry['weight_norms'].append(w_norm)
            if grad is not None:
                g = grad.numpy().ravel()
                entry['grad_norms'].append(float(np.linalg.norm(g)))
                entry['zero_fracs'].append(float((np.abs(g) < 1e-10).mean()))
            else:
                entry['grad_norms'].append(0.0)
                entry['zero_fracs'].append(1.0)

        # Reduce to per-layer scalars
        result = {}
        for lname, d in layer_data.items():
            g  = float(np.sqrt(np.mean(np.square(d['grad_norms']))))
            w  = float(np.sqrt(np.mean(np.square(d['weight_norms']))))
            result[lname] = {
                'grad_norm':   g,
                'weight_norm': w,
                'ratio':       float(g / (w + 1e-12)),
                'zero_frac':   float(np.mean(d['zero_fracs'])),
            }
        return result, float(loss.numpy())

    # ------------------------------------------------------------------
    def _report(self, tag: str, stats: dict, loss: float):
        norms      = {k: v['grad_norm'] for k, v in stats.items()}
        vanishing  = [k for k, v in stats.items() if v['grad_norm'] < self.vanish_thresh]
        exploding  = [k for k, v in stats.items() if v['grad_norm'] > self.explode_thresh]
        dead_ste   = [k for k, v in stats.items() if v['zero_frac'] > 0.9]  # STE mostly inactive
        has_issue  = bool(vanishing or exploding)

        status = 'ISSUE' if has_issue else 'OK'
        global_norm = float(np.sqrt(sum(v['grad_norm']**2 for v in stats.values())))
        print(f'\n[GradDiag {tag}] {status}  loss={loss:.4f}  '
              f'global_norm={global_norm:.3e}  '
              f'layers={len(stats)}  vanishing={len(vanishing)}  '
              f'exploding={len(exploding)}  dead_STE={len(dead_ste)}')
        if vanishing:
            print(f'  VANISHING  (grad_norm < {self.vanish_thresh:.0e}): {vanishing}')
        if exploding:
            print(f'  EXPLODING  (grad_norm > {self.explode_thresh:.0e}): {exploding}')
        if dead_ste:
            print(f'  DEAD STE   (>90% zero grads): {dead_ste[:8]}')

        sorted_s = sorted(stats.items(), key=lambda x: x[1]['grad_norm'])
        print('  Bottom-5 grad norm / ratio / dead%:')
        for k, v in sorted_s[:5]:
            print(f'    {k:40s}  '
                  f'gnorm={v["grad_norm"]:.2e}  '
                  f'ratio={v["ratio"]:.2e}  '
                  f'zero={v["zero_frac"]*100:.0f}%')
        print('  Top-5 grad norm / ratio / dead%:')
        for k, v in sorted_s[-5:]:
            print(f'    {k:40s}  '
                  f'gnorm={v["grad_norm"]:.2e}  '
                  f'ratio={v["ratio"]:.2e}  '
                  f'zero={v["zero_frac"]*100:.0f}%')

        return {
            'has_issue':  has_issue,
            'global_norm': global_norm,
            'loss':       loss,
            'vanishing':  vanishing,
            'exploding':  exploding,
            'dead_ste':   dead_ste,
            'layer_stats': stats,
        }

    # ------------------------------------------------------------------
    def on_train_begin(self, logs=None):
        """Probe before any weight update to catch issues at init."""
        print('\n[GradDiag] === Pre-training gradient probe ===')
        stats, loss = self._compute_stats()
        entry = self._report('pre-train', stats, loss)
        entry['epoch'] = 0
        self.history.append(entry)
        self._save()

    # ------------------------------------------------------------------
    def on_train_batch_end(self, batch, logs=None):
        """Monitor first ``watch_first_batches`` steps (disabled by default).

        WARNING: each call runs a full GradientTape pass — very expensive
        for large models.  Keep watch_first_batches=0 unless debugging
        gradient issues in the very first steps.
        """
        if self.watch_first_batches == 0 or batch >= self.watch_first_batches:
            return
        stats, loss = self._compute_stats()
        self._report(f'epoch{self._epoch+1}_batch{batch+1}', stats, loss)

    # ------------------------------------------------------------------
    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.check_every != 0:
            return
        stats, loss = self._compute_stats()
        entry = self._report(f'epoch {epoch+1}', stats, loss)
        entry['epoch'] = epoch + 1
        self.history.append(entry)
        self._save()

    # ------------------------------------------------------------------
    def _save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # Serialize: layer_stats dicts may contain non-serialisable floats
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(i) for i in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj
        with open(os.path.join(self.output_dir, 'grad_norms.json'), 'w') as f:
            json.dump(_clean(self.history), f, indent=2)


# ---------------------------------------------------------------------------
# JSON logger (same as train_cifar10.py)
# ---------------------------------------------------------------------------

class JSONLogger(keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path    = path
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        entry = {'epoch': epoch + 1}
        if logs:
            entry.update({k: float(v) for k, v in logs.items()})
        self.history.append(entry)
        with open(self.path, 'w') as f:
            json.dump(self.history, f, indent=2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='HGQ ResNet-18 CIFAR-10 training + gradient diagnostics'
    )
    p.add_argument('--output',   type=str, default='results_hgq_cifar10/')
    p.add_argument('--workers',  type=int, default=4)

    # Training hyper-parameters
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--batch-size',    type=int,   default=128)
    p.add_argument('--lr',            type=float, default=0.1)
    p.add_argument('--momentum',      type=float, default=0.9)
    p.add_argument('--weight-decay',  type=float, default=5e-4)
    p.add_argument('--warmup-epochs', type=int,   default=5)

    # HGQ quantization
    p.add_argument('--bw-k',      type=int,   default=4,
                   help='Initial fractional bits for weights')
    p.add_argument('--bw-a',      type=int,   default=4,
                   help='Initial fractional bits for activations')
    p.add_argument('--beta-max',  type=float, default=1e-5,
                   help='Peak EBOPs regularisation coefficient')

    # Gradient diagnostics
    p.add_argument('--grad-check-every',   type=int,   default=1,
                   help='Run gradient check every N epochs')
    p.add_argument('--grad-vanish-thresh', type=float, default=1e-6,
                   help='Gradient norm below this → vanishing')
    p.add_argument('--grad-explode-thresh',type=float, default=1e2,
                   help='Gradient norm above this → exploding')

    # Resume
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--seed',   type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print('Loading CIFAR-10 ...')
    train_ds, val_ds, num_classes = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Cache a small fixed probe batch for diagnostics.
    # Using only 16 samples (instead of the full training batch) keeps the
    # GradientTape pass and the multi-output ActivationRangeMonitor model
    # well within GPU memory even after the main training graph is allocated.
    _raw_x, _raw_y = next(iter(train_ds))
    probe_x = _raw_x[:16]
    probe_y = _raw_y[:16]
    probe_batch = (probe_x, probe_y)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Using MirroredStrategy across {strategy.num_replicas_in_sync} GPUs')
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = get_hgq_resnet18_cifar(
            num_classes=num_classes,
            init_bw_k=args.bw_k,
            init_bw_a=args.bw_a,
        )

        steps_per_epoch = sum(1 for _ in train_ds)
        total_steps     = steps_per_epoch * args.epochs
        warmup_steps    = steps_per_epoch * args.warmup_epochs

        lr_schedule = WarmupCosineDecay(
            base_lr=args.lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=1e-5,
        )

        optimizer = keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')],
        )

    model.summary()

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    initial_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f'Resuming from {args.resume}')
            model.load_weights(args.resume)
            m = re.search(r'epoch[=_](\d+)', args.resume)
            if m:
                initial_epoch = int(m.group(1))
                print(f'  Starting at epoch {initial_epoch}')
        else:
            print(f'Checkpoint not found: {args.resume}')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    # HGQ beta (EBOPs regularisation): ramp from 0 after warmup
    beta_scheduler = BetaScheduler(
        PieceWiseSchedule([
            (0,                   5e-10,       'constant'),   # flat start
            (args.warmup_epochs,  5e-10,       'log'),        # ramp up
            (args.epochs,         args.beta_max, 'constant'), # peak
        ])
    )

    # HGQ EBOPs counter (adds 'ebops' to logs)
    ebops_cb = FreeEBOPs()

    # Activation range monitor (catches quantizer saturation)
    act_monitor = ActivationRangeMonitor(
        probe_x=probe_x,
        output_dir=args.output,
        check_every=args.grad_check_every,
    )

    # Gradient diagnostics
    grad_diag = GradientDiagnostics(
        probe_batch=probe_batch,
        loss_fn=loss_fn,
        output_dir=args.output,
        check_every=args.grad_check_every,
        watch_first_batches=0,   # set >0 only for early-step debugging
        vanish_thresh=args.grad_vanish_thresh,
        explode_thresh=args.grad_explode_thresh,
    )

    callbacks = [
        beta_scheduler,
        ebops_cb,
        act_monitor,
        grad_diag,
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output, 'best.keras'),
            monitor='val_acc',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                args.output,
                'epoch={epoch:03d}-val_acc={val_acc:.4f}.keras'
            ),
            save_freq='epoch',
            verbose=0,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output, 'tb_logs'),
            histogram_freq=0,
            update_freq='epoch',
        ),
        JSONLogger(os.path.join(args.output, 'history.json')),
        keras.callbacks.TerminateOnNaN(),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model.fit(
        train_ds,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Final gradient health report
    # ------------------------------------------------------------------
    issues = [e for e in grad_diag.history if e['has_issue']]
    print(f'\n=== Gradient Health Summary ===')
    print(f'Total epochs checked : {len(grad_diag.history)}')
    print(f'Epochs with issues   : {len(issues)}')
    if issues:
        print('Issue epochs:')
        for e in issues:
            print(f'  epoch {e["epoch"]:4d}  '
                  f'vanishing={e["vanishing"]}  exploding={e["exploding"]}')
    else:
        print('No gradient issues detected throughout training.')
    print(f'Full gradient norm history saved to: '
          f'{os.path.join(args.output, "grad_norms.json")}')

    print('\nTraining complete.')


# python train_hgq_cifar10.py --bw-k 6 --bw-a 6 --grad-check-every 5 --epochs 100
if __name__ == '__main__':
    main()
