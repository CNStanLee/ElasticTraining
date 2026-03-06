#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def pick_latest_uniform_meta(pruned_root: Path, target: int) -> Path:
    d = pruned_root / 'uniform'
    cands = sorted(d.glob(f'*-oneshot-uniform-target{target}-ebops*.weights.meta.json'))
    if not cands:
        raise FileNotFoundError(f'No uniform meta found under {d} for target={target}')
    return sorted(cands, key=lambda p: p.stat().st_mtime)[-1]


def _flatten_layers(model):
    if hasattr(model, '_flatten_layers'):
        return model._flatten_layers()
    return model.layers


def _forward_update_ebops_no_bn_drift(model, sample_input):
    bn_layers = []
    old_m = []
    for layer in _flatten_layers(model):
        if isinstance(layer, keras.layers.BatchNormalization):
            bn_layers.append(layer)
            old_m.append(float(layer.momentum))
            layer.momentum = 1.0
    try:
        model(sample_input, training=True)
    finally:
        for layer, m in zip(bn_layers, old_m):
            layer.momentum = m


def compute_model_ebops(model, sample_input) -> float:
    from keras import ops

    _forward_update_ebops_no_bn_drift(model, sample_input)
    total = 0.0
    for layer in _flatten_layers(model):
        if getattr(layer, 'enable_ebops', False) and getattr(layer, '_ebops', None) is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def set_all_beta(model, beta_value: float):
    bv = tf.constant(beta_value, dtype=tf.float32)
    for layer in _flatten_layers(model):
        if hasattr(layer, '_beta'):
            try:
                layer._beta.assign(tf.ones_like(layer._beta) * bv)
            except Exception:
                pass


class MeasuredEbopsCallback(keras.callbacks.Callback):
    def __init__(self, sample_input):
        super().__init__()
        self.sample_input = sample_input

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        eb = compute_model_ebops(self.model, self.sample_input)
        logs['ebops_measured'] = float(eb)
        print(f'  [EBOPs measured] epoch={epoch+1} ebops={eb:.1f}')


def main() -> None:
    here = Path(__file__).resolve().parent
    jsc_root = here.parent
    if str(jsc_root) not in sys.path:
        sys.path.insert(0, str(jsc_root))

    import model.model  # noqa: F401
    from data.data import get_data
    from hgq.utils.sugar import FreeEBOPs
    rbu_path = jsc_root / 'utils' / 'ramanujan_budget_utils.py'
    spec = importlib.util.spec_from_file_location('ramanujan_budget_utils', rbu_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {rbu_path}')
    rbu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rbu)
    BetaOnlyBudgetController = rbu.BetaOnlyBudgetController
    EBOPsConstantProjector = rbu.EBOPsConstantProjector

    parser = argparse.ArgumentParser(description='Train uniform-pruned model for 500 epochs and observe accuracy')
    parser.add_argument('--target', type=int, default=400)
    parser.add_argument('--target_ebops', type=float, default=None)
    parser.add_argument(
        '--base_model',
        type=str,
        default='baseline/epoch=2236-val_acc=0.770-ebops=23589-val_loss=0.641.keras',
    )
    parser.add_argument('--pruned_root', type=str, default='pruned_models')
    parser.add_argument('--data_h5', type=str, default='../data/dataset.h5')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=33200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta_init', type=float, default=1e-5)
    parser.add_argument('--beta_min', type=float, default=1e-7)
    parser.add_argument('--beta_max', type=float, default=5e-4)
    parser.add_argument('--budget_margin', type=float, default=0.08)
    parser.add_argument('--beta_adjust_factor', type=float, default=1.15)
    parser.add_argument('--beta_ema_alpha', type=float, default=0.2)
    parser.add_argument('--use_projector', action='store_true', default=True)
    parser.add_argument('--no_use_projector', action='store_true')
    parser.add_argument('--projector_gamma', type=float, default=0.5)
    parser.add_argument('--projector_alpha_min', type=float, default=0.80)
    parser.add_argument('--projector_alpha_max', type=float, default=1.25)
    parser.add_argument('--projector_ema_alpha', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='uniform_train500')
    args = parser.parse_args()
    if args.no_use_projector:
        args.use_projector = False

    tf.random.set_seed(int(args.seed))
    np.random.seed(int(args.seed))

    base_model = here / args.base_model
    pruned_root = here / args.pruned_root
    data_h5 = (here / args.data_h5).resolve()
    out_dir = here / args.out_dir / f'target{int(args.target)}'
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = pick_latest_uniform_meta(pruned_root=pruned_root, target=int(args.target))
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    weights_path = Path(meta['weights_path'])
    target_ebops = float(args.target_ebops) if args.target_ebops is not None else float(args.target)

    model = keras.models.load_model(base_model, compile=False)
    model.load_weights(weights_path)

    (x_train, y_train), (x_val, y_val), _ = get_data(data_h5)
    x_train = tf.constant(x_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.int32)
    x_val = tf.constant(x_val, dtype=tf.float32)
    y_val = tf.constant(y_val, dtype=tf.int32)
    sample_input = tf.constant(x_val[:512], dtype=tf.float32)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(args.lr)),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')],
    )

    csv_path = out_dir / 'history.csv'
    best_path = out_dir / 'best.keras'
    ebops_cb = FreeEBOPs()
    measured_ebops_cb = MeasuredEbopsCallback(sample_input=sample_input)
    set_all_beta(model, float(args.beta_init))
    budget_ctrl = BetaOnlyBudgetController(
        target_ebops=target_ebops,
        margin=float(args.budget_margin),
        beta_init=float(args.beta_init),
        beta_min=float(args.beta_min),
        beta_max=float(args.beta_max),
        adjust_factor=float(args.beta_adjust_factor),
        ema_alpha=float(args.beta_ema_alpha),
    )
    projector = None
    if args.use_projector:
        projector = EBOPsConstantProjector(
            target_ebops=target_ebops,
            b_k_min=0.25,
            b_k_max=8.0,
            pruned_threshold=0.1,
            start_epoch=0,
            alpha_gamma=float(args.projector_gamma),
            alpha_min=float(args.projector_alpha_min),
            alpha_max=float(args.projector_alpha_max),
            ema_alpha=float(args.projector_ema_alpha),
            project_activation=False,
            log_scale=False,
        )
    callbacks = [
        ebops_cb,
        budget_ctrl,
        *( [projector] if projector is not None else [] ),
        measured_ebops_cb,
        keras.callbacks.CSVLogger(str(csv_path)),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor='val_acc',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    print(f'[LOAD] meta={meta_path.name}')
    print(f'[LOAD] weights={weights_path.name}')
    print(f'[TRAIN] epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}')
    print(
        f'[BUDGET] target_ebops={target_ebops:.1f} '
        f'beta=[{args.beta_min:.1e},{args.beta_max:.1e}] init={args.beta_init:.1e} '
        f'margin={args.budget_margin:.2f}'
    )
    if args.use_projector:
        print(
            f'[PROJECTOR] on gamma={args.projector_gamma:.2f} '
            f'alpha=[{args.projector_alpha_min:.2f},{args.projector_alpha_max:.2f}]'
        )

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        verbose=2,
        callbacks=callbacks,
    )

    h = hist.history
    tr_acc = np.array(h.get('acc', []), dtype=np.float32)
    va_acc = np.array(h.get('val_acc', []), dtype=np.float32)
    epochs = np.arange(1, len(tr_acc) + 1, dtype=np.int32)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, tr_acc, label='train acc')
    plt.plot(epochs, va_acc, label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Opt Train Curve (target={args.target})')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plot_path = out_dir / 'acc_curve.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    summary_path = out_dir / 'summary.txt'
    best_val = float(np.max(va_acc)) if va_acc.size else float('nan')
    last_val = float(va_acc[-1]) if va_acc.size else float('nan')
    final_ebops = compute_model_ebops(model, sample_input)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'meta={meta_path}\n')
        f.write(f'weights={weights_path}\n')
        f.write(f'epochs={args.epochs}\n')
        f.write(f'batch_size={args.batch_size}\n')
        f.write(f'lr={args.lr}\n')
        f.write(f'target_ebops={target_ebops:.1f}\n')
        f.write(f'final_ebops_measured={final_ebops:.1f}\n')
        f.write(f'best_val_acc={best_val:.6f}\n')
        f.write(f'last_val_acc={last_val:.6f}\n')

    print('[DONE]')
    print(f'history={csv_path}')
    print(f'plot   ={plot_path}')
    print(f'best   ={best_path}')
    print(f'best_val_acc={best_val:.6f} last_val_acc={last_val:.6f} final_ebops={final_ebops:.1f}')


if __name__ == '__main__':
    main()
