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


def _get_q_var(q, name: str):
    if q is None:
        return None
    if hasattr(q, name):
        return getattr(q, name)
    if hasattr(q, f'_{name}'):
        return getattr(q, f'_{name}')
    for v in getattr(q, 'variables', []):
        vname = str(getattr(v, 'name', ''))
        tail = vname.split(':')[0].split('/')[-1]
        if tail == name:
            return v
    return None


def pick_latest_opt_meta(pruned_root: Path, target: int) -> Path:
    d = pruned_root / 'spectral_quant'
    cands = sorted(d.glob(f'*-oneshot-spectral_quant-target{target}-ebops*.weights.meta.json'))
    if not cands:
        raise FileNotFoundError(f'No opt meta found under {d} for target={target}')
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
    def __init__(self, sample_input, log_every: int = 20, measure_every: int = 1):
        super().__init__()
        self.sample_input = sample_input
        self.log_every = int(log_every)
        self.measure_every = max(1, int(measure_every))
        self._last_eb = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        do_measure = (epoch < 5) or (((epoch + 1) % self.measure_every) == 0)
        if do_measure or (self._last_eb is None):
            eb = compute_model_ebops(self.model, self.sample_input)
            self._last_eb = float(eb)
        logs['ebops_measured'] = float(self._last_eb)
        if (epoch + 1) % self.log_every == 0 or epoch < 5:
            print(f'  [EBOPs measured] epoch={epoch+1} ebops={float(self._last_eb):.1f}')


class BudgetSwingCallback(keras.callbacks.Callback):
    """Periodically swing target ebops to escape flat local minima."""

    def __init__(self, controller, base_target: float, amp_ratio: float = 0.08, period: int = 80):
        super().__init__()
        self.controller = controller
        self.base = float(base_target)
        self.amp = float(amp_ratio)
        self.period = int(period)

    def on_epoch_begin(self, epoch, logs=None):
        if self.period <= 0 or self.amp <= 0:
            return
        phase = 2.0 * np.pi * (float(epoch) / float(self.period))
        tgt = self.base * (1.0 + self.amp * np.sin(phase))
        self.controller.target_ebops = float(max(tgt, 1.0))
        if logs is not None:
            logs['swing_target'] = self.controller.target_ebops


class DynamicRewireCallback(keras.callbacks.Callback):
    """Periodic prune-regrow to alter topology while keeping sparsity scale."""

    def __init__(
        self,
        start_epoch: int = 80,
        interval: int = 20,
        swap_rate: float = 0.02,
        pruned_threshold: float = 1e-6,
        seed: int = 42,
    ):
        super().__init__()
        self.start_epoch = int(start_epoch)
        self.interval = int(interval)
        self.swap_rate = float(swap_rate)
        self.pruned_threshold = float(pruned_threshold)
        self.rng = np.random.RandomState(int(seed))

    def _rewire_layer(self, layer) -> int:
        if not hasattr(layer, 'kernel') or getattr(layer, 'kq', None) is None:
            return 0
        if len(layer.kernel.shape) != 2:
            return 0
        k = np.array(layer.kernel.numpy(), dtype=np.float32)
        b_var = _get_q_var(layer.kq, 'b')
        if b_var is None:
            b_var = _get_q_var(layer.kq, 'f')
        if b_var is None:
            return 0
        i_var = _get_q_var(layer.kq, 'i')
        k_var = _get_q_var(layer.kq, 'k')
        b = np.array(b_var.numpy(), dtype=np.float32)
        i = np.array(i_var.numpy(), dtype=np.float32) if i_var is not None else None
        kv = np.array(k_var.numpy(), dtype=np.float32) if k_var is not None else None

        try:
            from keras import ops
            bits = layer.kq.bits_(ops.shape(layer.kernel))
            bits = np.array(ops.convert_to_numpy(bits), dtype=np.float32)
            active = bits > 1e-8
        except Exception:
            active = b > self.pruned_threshold

        active_idx = np.argwhere(active)
        inact_idx = np.argwhere(~active)
        if active_idx.size == 0 or inact_idx.size == 0:
            return 0
        nswap = int(round(active_idx.shape[0] * self.swap_rate))
        nswap = int(np.clip(nswap, 1, min(active_idx.shape[0], inact_idx.shape[0])))
        if nswap <= 0:
            return 0

        act_abs = np.abs(k[active[:, :]])
        drop_rel = np.argsort(act_abs)[:nswap]
        drop_idx = active_idx[drop_rel]
        grow_sel = self.rng.choice(inact_idx.shape[0], size=nswap, replace=False)
        grow_idx = inact_idx[grow_sel]
        for (d0, d1), (g0, g1) in zip(drop_idx, grow_idx):
            wd = float(k[d0, d1])
            bd = float(b[d0, d1])
            idv = float(i[d0, d1]) if i is not None else 0.0
            k[g0, g1] = wd if abs(wd) > 1e-8 else (0.01 * np.std(k) + 1e-3)
            if self.rng.rand() < 0.5:
                k[g0, g1] *= -1.0
            k[d0, d1] = 0.0
            b[g0, g1] = max(bd, self.pruned_threshold)
            b[d0, d1] = 0.0
            if i is not None:
                i[g0, g1] = idv
                i[d0, d1] = -16.0
            if kv is not None:
                kv[g0, g1] = 1.0
                kv[d0, d1] = 0.0

        layer.kernel.assign(k.astype(np.float32))
        b_var.assign(b.astype(np.float32))
        if i_var is not None:
            i_var.assign(i.astype(np.float32))
        if k_var is not None:
            k_var.assign(kv.astype(np.float32))
        return int(nswap)

    def on_epoch_end(self, epoch, logs=None):
        e = int(epoch) + 1
        if e < self.start_epoch:
            return
        if self.interval <= 0 or ((e - self.start_epoch) % self.interval != 0):
            return
        total = 0
        for layer in _flatten_layers(self.model):
            total += self._rewire_layer(layer)
        if total > 0:
            print(f'  [Rewire] epoch={e} swaps={total}')
            if logs is not None:
                logs['rewire_swaps'] = float(total)


class CollapseRecoveryCallback(keras.callbacks.Callback):
    """Recover to best-known weights when validation collapses for multiple epochs."""

    def __init__(
        self,
        monitor: str = 'val_acc',
        min_epoch: int = 120,
        drop_tol: float = 0.05,
        patience: int = 12,
        cooldown: int = 40,
    ):
        super().__init__()
        self.monitor = str(monitor)
        self.min_epoch = int(min_epoch)
        self.drop_tol = float(drop_tol)
        self.patience = int(patience)
        self.cooldown = int(cooldown)
        self.best = -np.inf
        self.best_weights = None
        self.bad = 0
        self.last_recover_epoch = -10**9

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get(self.monitor)
        if val is None:
            return
        val = float(val)
        if val > self.best:
            self.best = val
            self.best_weights = self.model.get_weights()
            self.bad = 0
            return
        if (epoch + 1) < self.min_epoch:
            return
        if (epoch + 1) - self.last_recover_epoch < self.cooldown:
            return
        if val < (self.best - self.drop_tol):
            self.bad += 1
        else:
            self.bad = 0
        if self.bad >= self.patience and self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            self.last_recover_epoch = int(epoch) + 1
            self.bad = 0
            print(
                f'  [Recover] epoch={epoch+1} restore_best={self.best:.4f} '
                f'current={val:.4f}'
            )
            logs['recovered'] = 1.0


class CarryForwardValMetricsCallback(keras.callbacks.Callback):
    """Keep val metrics non-NA when validation is not run every epoch."""

    def __init__(self):
        super().__init__()
        self.last_val_acc = None
        self.last_val_loss = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        v_acc = logs.get('val_acc')
        v_loss = logs.get('val_loss')
        if v_acc is not None:
            self.last_val_acc = float(v_acc)
        if v_loss is not None:
            self.last_val_loss = float(v_loss)

        if logs.get('val_acc') is None:
            if self.last_val_acc is not None:
                logs['val_acc'] = float(self.last_val_acc)
            elif logs.get('acc') is not None:
                logs['val_acc'] = float(logs['acc'])
        if logs.get('val_loss') is None:
            if self.last_val_loss is not None:
                logs['val_loss'] = float(self.last_val_loss)
            elif logs.get('loss') is not None:
                logs['val_loss'] = float(logs['loss'])


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

    parser = argparse.ArgumentParser(description='Train opt-pruned model for 500 epochs and observe accuracy')
    parser.add_argument('--target', type=int, default=400)
    parser.add_argument('--target_ebops', type=float, default=None)
    parser.add_argument('--init_model', type=str, default=None, help='Optional full model path to continue training from')
    parser.add_argument('--meta_path', type=str, default=None, help='Optional one-shot meta json path to initialize from')
    parser.add_argument(
        '--base_model',
        type=str,
        default='baseline/epoch=2236-val_acc=0.770-ebops=23589-val_loss=0.641.keras',
    )
    parser.add_argument('--pruned_root', type=str, default='pruned_models')
    parser.add_argument('--data_h5', type=str, default='../data/dataset.h5')
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--batch_size', type=int, default=33200)
    parser.add_argument('--validation_freq', type=int, default=1)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--lr_cycle', type=int, default=120)
    parser.add_argument('--beta_init', type=float, default=1e-5)
    parser.add_argument('--beta_min', type=float, default=1e-7)
    parser.add_argument('--beta_max', type=float, default=5e-4)
    parser.add_argument('--budget_margin', type=float, default=0.12)
    parser.add_argument('--beta_adjust_factor', type=float, default=1.25)
    parser.add_argument('--beta_ema_alpha', type=float, default=0.2)
    parser.add_argument('--use_projector', action='store_true', default=True)
    parser.add_argument('--no_use_projector', action='store_true')
    parser.add_argument('--projector_gamma', type=float, default=0.5)
    parser.add_argument('--projector_alpha_min', type=float, default=0.80)
    parser.add_argument('--projector_alpha_max', type=float, default=1.25)
    parser.add_argument('--projector_ema_alpha', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='opt_train500')
    parser.add_argument('--swing_amp', type=float, default=0.08)
    parser.add_argument('--swing_period', type=int, default=80)
    parser.add_argument('--rewire_start', type=int, default=80)
    parser.add_argument('--rewire_interval', type=int, default=20)
    parser.add_argument('--rewire_rate', type=float, default=0.02)
    parser.add_argument('--no_rewire', action='store_true')
    parser.add_argument('--recover_start', type=int, default=120)
    parser.add_argument('--recover_drop_tol', type=float, default=0.05)
    parser.add_argument('--recover_patience', type=int, default=12)
    parser.add_argument('--recover_cooldown', type=int, default=40)
    parser.add_argument('--no_recover', action='store_true')
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--measure_every', type=int, default=1)
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

    meta_path = None
    weights_path = None
    target_ebops = float(args.target_ebops) if args.target_ebops is not None else float(args.target)

    if args.init_model:
        model = keras.models.load_model((here / args.init_model), compile=False)
    else:
        meta_path = (here / args.meta_path).resolve() if args.meta_path else pick_latest_opt_meta(pruned_root=pruned_root, target=int(args.target))
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        weights_path = Path(meta['weights_path'])
        model = keras.models.load_model(base_model, compile=False)
        model.load_weights(weights_path)

    (x_train, y_train), (x_val, y_val), _ = get_data(data_h5)
    x_train = tf.constant(x_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.int32)
    x_val = tf.constant(x_val, dtype=tf.float32)
    y_val = tf.constant(y_val, dtype=tf.int32)
    sample_input = tf.constant(x_val[:512], dtype=tf.float32)

    lr_sched = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=float(args.lr),
        first_decay_steps=max(int(args.lr_cycle), 10),
        t_mul=1.0,
        m_mul=0.95,
        alpha=float(args.lr_min / max(args.lr, 1e-12)),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_sched),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')],
    )

    csv_path = out_dir / 'history.csv'
    best_path = out_dir / 'best.keras'
    ebops_cb = FreeEBOPs()
    measured_ebops_cb = MeasuredEbopsCallback(
        sample_input=sample_input,
        log_every=int(args.log_every),
        measure_every=int(args.measure_every),
    )
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
    swing_cb = BudgetSwingCallback(
        controller=budget_ctrl,
        base_target=target_ebops,
        amp_ratio=float(args.swing_amp),
        period=int(args.swing_period),
    )
    rewire_cb = DynamicRewireCallback(
        start_epoch=int(args.rewire_start),
        interval=int(args.rewire_interval),
        swap_rate=float(args.rewire_rate),
        pruned_threshold=1e-6,
        seed=int(args.seed),
    )
    recover_cb = CollapseRecoveryCallback(
        monitor='val_acc',
        min_epoch=int(args.recover_start),
        drop_tol=float(args.recover_drop_tol),
        patience=int(args.recover_patience),
        cooldown=int(args.recover_cooldown),
    )
    carry_val_cb = CarryForwardValMetricsCallback()
    callbacks = [
        ebops_cb,
        swing_cb,
        budget_ctrl,
        *( [projector] if projector is not None else [] ),
        *( [] if args.no_rewire else [rewire_cb] ),
        *( [] if args.no_recover else [recover_cb] ),
        carry_val_cb,
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

    if args.init_model:
        print(f'[LOAD] init_model={args.init_model}')
    else:
        print(f'[LOAD] meta={meta_path.name}')
        print(f'[LOAD] weights={weights_path.name}')
    print(f'[TRAIN] epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}')
    print(
        f'[BUDGET] target_ebops={target_ebops:.1f} '
        f'beta=[{args.beta_min:.1e},{args.beta_max:.1e}] init={args.beta_init:.1e} '
        f'margin={args.budget_margin:.2f}'
    )
    print(f'[SWING] amp={args.swing_amp:.3f} period={args.swing_period}')
    if not args.no_rewire:
        print(f'[REWIRE] start={args.rewire_start} every={args.rewire_interval} rate={args.rewire_rate:.3f}')
    if not args.no_recover:
        print(
            f'[RECOVER] start={args.recover_start} drop_tol={args.recover_drop_tol:.3f} '
            f'patience={args.recover_patience} cooldown={args.recover_cooldown}'
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
        validation_freq=max(1, int(args.validation_freq)),
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
        if meta_path is not None:
            f.write(f'meta={meta_path}\n')
        if weights_path is not None:
            f.write(f'weights={weights_path}\n')
        if args.init_model:
            f.write(f'init_model={args.init_model}\n')
        f.write(f'epochs={args.epochs}\n')
        f.write(f'batch_size={args.batch_size}\n')
        f.write(f'validation_freq={max(1, int(args.validation_freq))}\n')
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
