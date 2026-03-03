"""
train_cifar10.py – Train a float32 ResNet-18 on CIFAR-10 with TensorFlow / Keras.

CIFAR-10: 60 000 colour images (32×32), 10 classes.
Dataset is downloaded automatically by Keras on first run.

Example:
    python train_cifar10.py
    python train_cifar10.py --epochs 100 --batch-size 128
    python train_cifar10.py --resume results_cifar10/best.keras
"""

import os

os.environ['KERAS_BACKEND']        = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import re

import tensorflow as tf
import keras
from keras import layers

# GPU memory growth – must be set before any TF ops
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for _gpu in gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
    print(f'GPUs available: {len(gpus)}')
else:
    print('No GPU found, using CPU')

from utils.train_utils import WarmupCosineDecay


# ---------------------------------------------------------------------------
# CIFAR-10 adapted ResNet-18
# ---------------------------------------------------------------------------
# For 32×32 inputs the standard ImageNet stem (7×7 conv, stride-2, maxpool)
# is too aggressive.  Replace it with a single 3×3 conv (stride-1, no pool).
# ---------------------------------------------------------------------------

def _basic_block(x, filters, stride=1, name='block'):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False, name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)

    x = layers.Conv2D(filters, 3, strides=1, padding='same',
                      use_bias=False, name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 use_bias=False, name=f'{name}_proj')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_proj_bn')(shortcut)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name}_relu2')(x)
    return x


def _make_layer(x, filters, num_blocks, stride, name):
    x = _basic_block(x, filters, stride=stride, name=f'{name}_0')
    for i in range(1, num_blocks):
        x = _basic_block(x, filters, stride=1, name=f'{name}_{i}')
    return x


def get_resnet18_cifar(num_classes: int = 10):
    """ResNet-18 with a CIFAR-friendly stem (3×3 conv, stride-1, no maxpool)."""
    inputs = keras.Input(shape=(32, 32, 3), name='input')

    # Lightweight stem for small images
    x = layers.Conv2D(64, 3, strides=1, padding='same',
                      use_bias=False, name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)
    # No maxpool — keeps spatial size at 32×32 through layer1

    x = _make_layer(x, 64,  num_blocks=2, stride=1, name='layer1')
    x = _make_layer(x, 128, num_blocks=2, stride=2, name='layer2')
    x = _make_layer(x, 256, num_blocks=2, stride=2, name='layer3')
    x = _make_layer(x, 512, num_blocks=2, stride=2, name='layer4')

    x = layers.GlobalAveragePooling2D(name='gap')(x)
    outputs = layers.Dense(num_classes, name='fc')(x)

    return keras.Model(inputs, outputs, name='resnet18_cifar10')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_cifar10_dataloaders(batch_size: int = 128, num_workers: int = 4):
    """
    Returns (train_ds, val_ds, num_classes) as tf.data.Dataset objects.

    Training pipeline:
        RandomFlip → RandomCrop (with 4-pixel padding) → Normalize

    Validation pipeline:
        Normalize only
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    mean = tf.constant(CIFAR10_MEAN, dtype=tf.float32)
    std  = tf.constant(CIFAR10_STD,  dtype=tf.float32)

    def normalize(img):
        img = tf.cast(img, tf.float32) / 255.0
        return (img - mean) / std

    def augment(img, label):
        # Pad 4 pixels on each side, then random crop back to 32×32
        img = tf.image.pad_to_bounding_box(img, 4, 4, 40, 40)
        img = tf.image.random_crop(img, size=[32, 32, 3])
        img = tf.image.random_flip_left_right(img)
        img = normalize(img)
        return img, label

    def preprocess(img, label):
        img = normalize(img)
        return img, label

    autotune = tf.data.AUTOTUNE if num_workers == 0 else num_workers

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(50000, reshuffle_each_iteration=True)
        .map(augment, num_parallel_calls=autotune)
        .batch(batch_size, drop_remainder=True)
        .prefetch(autotune)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(preprocess, num_parallel_calls=autotune)
        .batch(batch_size, drop_remainder=False)
        .prefetch(autotune)
    )

    return train_ds, val_ds, 10


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Float32 ResNet-18 CIFAR-10 training (TF/Keras)'
    )
    parser.add_argument('--output', type=str, default='results_cifar10/',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel-map workers (0 = AUTOTUNE)')

    # Hyper-parameters (CIFAR-10 defaults)
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--batch-size',      type=int,   default=128)
    parser.add_argument('--lr',              type=float, default=0.1)
    parser.add_argument('--momentum',        type=float, default=0.9)
    parser.add_argument('--weight-decay',    type=float, default=5e-4)
    parser.add_argument('--warmup-epochs',   type=int,   default=5)

    parser.add_argument('--resume', type=str, default='',
                        help='Path to a .keras checkpoint to resume from')
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
    print('Loading CIFAR-10 ...')
    train_ds, val_ds, num_classes = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(f'Classes: {num_classes}')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Using MirroredStrategy across {strategy.num_replicas_in_sync} GPUs')
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = get_resnet18_cifar(num_classes=num_classes)

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

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
            ),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name='acc'),
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
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output, 'best.keras'),
            monitor='val_acc',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                args.output,
                'epoch={epoch:03d}-val_acc={val_acc:.4f}.keras'
            ),
            save_freq='epoch',
            save_weights_only=False,
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

    print('\nTraining complete.')


# conda run -n py312tf python train_cifar10.py --epochs 100 --batch-size 128
if __name__ == '__main__':
    main()
