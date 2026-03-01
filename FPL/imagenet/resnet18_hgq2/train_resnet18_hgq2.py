import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
from pathlib import Path

import keras
from hgq.utils.sugar import BetaScheduler, FreeEBOPs, PieceWiseSchedule

from imagenet_data import load_imagenet1k
from model_resnet18 import build_resnet18_hgq2


def main():
    parser = argparse.ArgumentParser(description='Train HGQ2-quantized ResNet-18 on ImageNet-1K')
    parser.add_argument('--data-root', type=str, required=True, help='ImageNet root dir containing train/ and val/')
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--expected-classes', type=int, default=1000)
    parser.add_argument('--strict-classes', action='store_true', help='Raise error when class count mismatches expected-classes')
    parser.add_argument('--train-subset', type=int, default=0, help='Use first N train images (0 = full train set)')
    parser.add_argument('--val-subset', type=int, default=0, help='Use first N val images (0 = full val set)')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--init-bw-k', type=int, default=2)
    parser.add_argument('--init-bw-a', type=int, default=2)
    parser.add_argument('--beta-max', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default='results/resnet18_hgq2/hgq2')
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

    model = build_resnet18_hgq2(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=num_classes,
        init_bw_k=args.init_bw_k,
        init_bw_a=args.init_bw_a,
    )

    total_steps = max(1, int(train_steps) * args.epochs)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=1e-3,
    )

    optimizer = keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    beta_sched = BetaScheduler(
        PieceWiseSchedule([
            (0, 5e-7, 'constant'),
            (max(1, args.epochs // 10), 5e-7, 'log'),
            (args.epochs, args.beta_max, 'constant'),
        ])
    )

    callbacks = [
        FreeEBOPs(),
        beta_sched,
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_hgq2.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / 'history_hgq2.csv'), append=False),
    ]

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=min(8, train_steps),
    )

    history = model.fit(
        train_ds.repeat(),
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    summary = {
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_ebops': float(history.history.get('ebops', [float('nan')])[-1]),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'init_bw_k': args.init_bw_k,
        'init_bw_a': args.init_bw_a,
        'beta_max': args.beta_max,
        'num_classes': int(num_classes),
        'image_size': args.image_size,
        'data_root': args.data_root,
        'expected_classes': args.expected_classes,
        'strict_classes': args.strict_classes,
        'train_subset': args.train_subset,
        'val_subset': args.val_subset,
    }
    with open(output_dir / 'summary_hgq2.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('Training finished. Summary:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
