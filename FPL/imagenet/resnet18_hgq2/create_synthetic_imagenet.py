#!/usr/bin/env python3
"""Generate a synthetic ImageNet-style dataset for testing ResNet-18 HGQ2.

Creates train/ and val/ directories with N classes of synthetic images.
Each class has distinguishable spatial frequency / color patterns so a
CNN *can* learn them, making gradient and bitwidth diagnostics meaningful.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _make_pattern(cls_id: int, img_idx: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a distinguishable pattern for a given class.

    Each class is characterised by:
    - a dominant colour channel bias
    - a set of spatial frequency bands (stripes / checkerboard / blobs)
    - random noise overlay for variety
    """
    img = rng.integers(0, 30, size=(size, size, 3), dtype=np.uint8)

    # Class-dependent colour channel emphasis
    ch = cls_id % 3
    img[:, :, ch] = np.clip(img[:, :, ch].astype(np.int16) + 80, 0, 255).astype(np.uint8)

    # Class-dependent spatial frequency
    freq = 2 + (cls_id * 3) % 17
    ys = np.arange(size).reshape(-1, 1)
    xs = np.arange(size).reshape(1, -1)

    pattern_type = cls_id % 4
    if pattern_type == 0:
        # Horizontal stripes
        stripe = ((ys * freq // size) % 2).astype(np.uint8) * 120
        img[:, :, (cls_id + 1) % 3] = np.clip(img[:, :, (cls_id + 1) % 3].astype(np.int16) + stripe, 0, 255).astype(np.uint8)
    elif pattern_type == 1:
        # Vertical stripes
        stripe = ((xs * freq // size) % 2).astype(np.uint8) * 120
        img[:, :, (cls_id + 2) % 3] = np.clip(img[:, :, (cls_id + 2) % 3].astype(np.int16) + stripe, 0, 255).astype(np.uint8)
    elif pattern_type == 2:
        # Checkerboard
        check = (((ys * freq // size) + (xs * freq // size)) % 2).astype(np.uint8) * 100
        img[:, :, ch] = np.clip(img[:, :, ch].astype(np.int16) + check, 0, 255).astype(np.uint8)
    else:
        # Diagonal stripes
        diag = (((ys + xs) * freq // size) % 2).astype(np.uint8) * 110
        img[:, :, (ch + 1) % 3] = np.clip(img[:, :, (ch + 1) % 3].astype(np.int16) + diag, 0, 255).astype(np.uint8)

    # Random per-image augmentation (brightness shift)
    shift = rng.integers(-20, 20)
    img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

    return img


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic ImageNet-style directory')
    parser.add_argument('--output', type=str, default='/tmp/imagenet_synth',
                        help='Output root directory')
    parser.add_argument('--num-classes', type=int, default=20,
                        help='Number of classes')
    parser.add_argument('--train-per-class', type=int, default=200,
                        help='Training images per class')
    parser.add_argument('--val-per-class', type=int, default=50,
                        help='Validation images per class')
    parser.add_argument('--image-size', type=int, default=64,
                        help='Image size (square)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.output)

    for split, count in [('train', args.train_per_class), ('val', args.val_per_class)]:
        for cls_id in range(args.num_classes):
            cls_name = f'n{cls_id:08d}'
            cls_dir = root / split / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            for img_idx in range(count):
                img_arr = _make_pattern(cls_id, img_idx, args.image_size, rng)
                img = Image.fromarray(img_arr)
                img.save(cls_dir / f'{img_idx:05d}.JPEG')

    total = args.num_classes * (args.train_per_class + args.val_per_class)
    print(f'Created {total} images ({args.num_classes} classes) at {root}')
    print(f'  train: {args.num_classes * args.train_per_class} images')
    print(f'  val:   {args.num_classes * args.val_per_class} images')


if __name__ == '__main__':
    main()
