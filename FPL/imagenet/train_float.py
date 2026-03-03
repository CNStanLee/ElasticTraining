"""
train_float.py – Train a float32 ResNet-18 on ImageNet with TensorFlow / Keras.

Expected dataset layout:
    <data_dir>/
        train/
            n01440764/  (one folder per class)
            ...
        val/
            n01440764/
            ...

Default data path: /mnt/data6/dataset/imagenet
If it does not exist, ImageNet is downloaded automatically from Kaggle.

Example:
    python train_float.py                              # uses data/imagenet (auto-resolved)
    python train_float.py --data /other/imagenet       # override path
    python train_float.py --epochs 90 --batch-size 256
"""

import os

os.environ['KERAS_BACKEND']       = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json

import tensorflow as tf
import keras

# GPU memory growth – must be set before any TF ops
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for _gpu in gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
    print(f'GPUs available: {len(gpus)}')
else:
    print('No GPU found, using CPU')

from data.data import get_imagenet_dataloaders, resolve_data_dir
from model.model import get_resnet18
from utils.train_utils import WarmupCosineDecay


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Float32 ResNet-18 ImageNet training (TF/Keras)')

    # Data – defaults to /mnt/data6/dataset/imagenet; auto-downloaded there if absent
    parser.add_argument('--data', type=str,
                        default='/mnt/data6/dataset/imagenet',
                        help='Path to ImageNet root directory (contains train/ and val/)')
    parser.add_argument('--output', type=str, default='results/',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of tf.data parallel-map workers (0 = AUTOTUNE)')

    # Training hyper-parameters
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Peak learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of linear warm-up epochs')
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # Checkpoint / resume
    parser.add_argument('--resume', type=str, default='',
                        help='Path to a .keras checkpoint to resume from')

    # Misc
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class JSONLogger(keras.callbacks.Callback):
    """Appends per-epoch metrics to a JSON file after every epoch."""

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
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_dir = resolve_data_dir(args.data)
    print('Loading ImageNet from:', data_dir)
    train_ds, val_ds, num_classes = get_imagenet_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(f'Classes: {num_classes}')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # Use MirroredStrategy for multi-GPU if GPUs are available
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Using MirroredStrategy across {strategy.num_replicas_in_sync} GPUs')
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = get_resnet18(num_classes=num_classes)

        # Estimate steps per epoch from the dataset
        steps_per_epoch = sum(1 for _ in train_ds)  # count batches
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

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                label_smoothing=args.label_smoothing,
            ),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name='acc'),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='acc_top5'),
            ],
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
            # Try to infer starting epoch from filename, e.g. epoch=27-...keras
            import re
            m = re.search(r'epoch[=_](\d+)', args.resume)
            if m:
                initial_epoch = int(m.group(1))
                print(f'  Starting at epoch {initial_epoch}')
        else:
            print(f'Checkpoint not found: {args.resume}')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output, 'best.keras'),
            monitor='val_acc',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # Save latest checkpoint every epoch
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                args.output,
                'epoch={epoch:02d}-val_acc={val_acc:.3f}.keras'
            ),
            save_freq='epoch',
            save_weights_only=False,
            verbose=0,
        ),
        # TensorBoard logs
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output, 'tb_logs'),
            histogram_freq=0,
            update_freq='epoch',
        ),
        # JSON history
        JSONLogger(os.path.join(args.output, 'history.json')),
        # Reduce LR on plateau (optional safety net)
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

    print('\nTraining complete.')

# conda run -n py312tf python train_float.py --data /path/to/imagenet --epochs 90 --batch-size 256
if __name__ == '__main__':
    main()
