from pathlib import Path
import math

import keras
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def _preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = keras.applications.resnet.preprocess_input(images)
    return images, labels


def _augment(images, labels):
    images = tf.image.random_flip_left_right(images)
    return images, labels


def load_imagenet1k(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 256,
    train_subset: int | None = None,
    val_subset: int | None = None,
    expected_num_classes: int | None = None,
    strict_num_classes: bool = False,
):
    root = Path(data_root)
    train_dir = root / 'train'
    val_dir = root / 'val'

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f'ImageNet-1K directory not found. Expect: {train_dir} and {val_dir}'
        )

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='int',
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    num_classes = len(train_ds.class_names)
    train_count_full = sum(1 for p in train_dir.rglob('*') if p.is_file())
    val_count_full = sum(1 for p in val_dir.rglob('*') if p.is_file())
    if expected_num_classes is not None and num_classes != int(expected_num_classes):
        msg = (
            f'Class count mismatch: found {num_classes}, expected {expected_num_classes}. '
            f'Dataset root={root}'
        )
        if strict_num_classes:
            raise ValueError(msg)
        print(f'[WARN] {msg}')

    if train_subset is not None and train_subset > 0:
        train_ds = train_ds.unbatch().take(int(train_subset)).batch(batch_size)
    if val_subset is not None and val_subset > 0:
        val_ds = val_ds.unbatch().take(int(val_subset)).batch(batch_size)

    if train_subset is not None and train_subset > 0:
        train_count = int(train_subset)
    else:
        train_count = int(train_count_full)

    if val_subset is not None and val_subset > 0:
        val_count = int(val_subset)
    else:
        val_count = int(val_count_full)

    train_ds = train_ds.map(_augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    train_steps = max(1, math.ceil(train_count / int(batch_size)))
    val_steps = max(1, math.ceil(val_count / int(batch_size)))

    return train_ds, val_ds, num_classes, train_steps, val_steps
