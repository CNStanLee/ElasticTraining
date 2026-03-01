import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
from pathlib import Path

import keras

from imagenet_data import load_imagenet1k
from model_resnet18 import build_resnet18_fp32


def get_callbacks(output_dir: Path):
    ckpt_path = output_dir / 'best_fp32.keras'
    csv_path = output_dir / 'history_fp32.csv'

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(csv_path), append=False),
    ]
    return callbacks


def main():
    parser = argparse.ArgumentParser(description='Train float ResNet-18 on ImageNet-1K')
    parser.add_argument('--data-root', type=str, required=True, help='ImageNet root dir containing train/ and val/')
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--expected-classes', type=int, default=1000)
    parser.add_argument('--strict-classes', action='store_true', help='Raise error when class count mismatches expected-classes')
    parser.add_argument('--train-subset', type=int, default=0, help='Use first N train images (0 = full train set)')
    parser.add_argument('--val-subset', type=int, default=0, help='Use first N val images (0 = full val set)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='results/resnet18_hgq2/fp32')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, num_classes, train_steps, val_steps = load_imagenet1k(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        expected_num_classes=args.expected_classes,
        strict_num_classes=args.strict_classes,
    )

    model = build_resnet18_fp32(input_shape=(args.image_size, args.image_size, 3), num_classes=num_classes)

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=max(1, int(train_steps) * args.epochs),
        alpha=1e-3,
    )
    optimizer = keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds.repeat(),
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=get_callbacks(output_dir),
        verbose=1,
    )

    summary = {
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_classes': int(num_classes),
        'image_size': args.image_size,
        'data_root': args.data_root,
        'expected_classes': args.expected_classes,
        'strict_classes': args.strict_classes,
        'train_subset': args.train_subset,
        'val_subset': args.val_subset,
    }
    with open(output_dir / 'summary_fp32.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('Training finished. Summary:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
