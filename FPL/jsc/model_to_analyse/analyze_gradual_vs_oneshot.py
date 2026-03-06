#!/usr/bin/env python3
"""
Analyse trainability at matched EBOPs for:
- gradual training checkpoints
- original one-shot prune
- optimized one-shot prune (spectral_quant)

Outputs:
- analysis/metrics.csv
- analysis/pairing.csv
- analysis/plots/*.png
- analysis/summary.md
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
import argparse
import re
import math
import json
import sys

import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
JSC_ROOT = HERE.parent
GRADUAL_DIR = HERE / 'gradual_training'
ONESHOT_DIR = HERE / 'oneshot'
ONESHOT_OPT_DIR = HERE / 'oneshot_opt'
OUT_DIR = HERE / 'analysis'
PLOTS_DIR = OUT_DIR / 'plots'

BASELINE_CKPT = GRADUAL_DIR / 'epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras'
STATE_TARGETS = {300, 2005, 5011}

METHODS = [
    ('gradual', 'Gradual training', '#1f77b4', 'o'),
    ('oneshot', 'One-shot baseline', '#d62728', 's'),
    ('oneshot_opt', 'One-shot optimized', '#2ca02c', '^'),
    ('random_init', 'Random init (same mask)', '#ff7f0e', 'D'),
]

if str(JSC_ROOT) not in sys.path:
    sys.path.insert(0, str(JSC_ROOT))

from data.data import get_data
from hgq.layers import QLayerBase, QEinsumDenseBatchnorm  # noqa: F401
from utils.ramanujan_budget_utils import _flatten_layers, _get_kq_var


def parse_gradual_checkpoints():
    recs = []
    pat = re.compile(r'epoch=(\d+)-val_acc=([0-9.]+)-ebops=(\d+)-val_loss=([0-9.]+)\.keras$')
    for p in sorted(GRADUAL_DIR.glob('*.keras')):
        m = pat.search(p.name)
        if not m:
            continue
        recs.append({
            'path': p,
            'epoch': int(m.group(1)),
            'val_acc_from_name': float(m.group(2)),
            'target_ebops': int(m.group(3)),
            'val_loss_from_name': float(m.group(4)),
            'name': p.name,
        })
    recs.sort(key=lambda x: x['target_ebops'])
    return recs


def parse_weight_records(dir_path: Path):
    recs = []
    pat = re.compile(r'target(\d+)-ebops(\d+)\.weights\.h5$')
    for p in sorted(dir_path.glob('*.weights.h5')):
        m = pat.search(p.name)
        if not m:
            continue
        recs.append({
            'path': p,
            'target_ebops': int(m.group(1)),
            'measured_ebops_from_name': int(m.group(2)),
            'mtime': p.stat().st_mtime,
            'name': p.name,
        })
    return recs


def pick_weight_for_target(target, recs):
    exact = [r for r in recs if r['target_ebops'] == target and 'epoch=7789-val_acc=0.770-ebops=19899' in r['name']]
    if not exact:
        exact = [r for r in recs if r['target_ebops'] == target]
    if exact:
        return sorted(exact, key=lambda x: x['mtime'])[-1]
    return None


def to_float(x):
    try:
        return float(x.numpy())
    except Exception:
        return float(x)


def compute_model_ebops(model, sample_input):
    from keras import ops
    model(sample_input, training=True)
    total = 0.0
    for layer in model._flatten_layers():
        if isinstance(layer, QLayerBase) and getattr(layer, 'enable_ebops', False) and layer._ebops is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def _bits_np(q, shape_like):
    if q is None:
        return None
    try:
        from keras import ops
        bits = q.bits_(ops.shape(shape_like))
        return np.array(ops.convert_to_numpy(bits), dtype=np.float32)
    except Exception:
        return None


def _flatten_kernel_and_mask(kernel_np, mask_np):
    if kernel_np.ndim > 2:
        k2 = kernel_np.reshape(-1, kernel_np.shape[-1])
        m2 = mask_np.reshape(-1, mask_np.shape[-1]) if mask_np is not None else None
    else:
        k2 = kernel_np
        m2 = mask_np
    return k2, m2


def _kurtosis(arr):
    if arr.size == 0:
        return 0.0
    m = float(arr.mean())
    s = float(arr.std())
    if s < 1e-12:
        return 0.0
    z = (arr - m) / s
    return float(np.mean(z**4))


def quick_eval(model, x, y):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    logits = model(x, training=False)
    loss = to_float(loss_fn(y, logits))
    pred = tf.argmax(logits, axis=-1)
    acc = to_float(tf.reduce_mean(tf.cast(tf.equal(pred, tf.cast(y, pred.dtype)), tf.float32)))
    return loss, acc


def _grad_stats(model, x_grad, y_grad):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        logits = model(x_grad, training=True)
        loss = loss_fn(y_grad, logits)
    grads = tape.gradient(loss, model.trainable_variables)

    g_sumsq = 0.0
    g_total = 0
    g_near_zero = 0
    kernel_grad_norms = []
    per_layer = {}

    for v, g in zip(model.trainable_variables, grads):
        if g is None:
            continue
        g_np = np.array(g.numpy(), dtype=np.float32)
        g_sumsq += float(np.sum(g_np * g_np))
        g_total += g_np.size
        g_near_zero += int(np.sum(np.abs(g_np) < 1e-8))

        vname = v.name
        lname = vname.split('/')[0]
        per_layer.setdefault(lname, 0.0)
        per_layer[lname] += float(np.linalg.norm(g_np))

        if 'kernel' in vname:
            kernel_grad_norms.append(float(np.linalg.norm(g_np)))

    return {
        'loss': float(to_float(loss)),
        'grads': grads,
        'grad_global_norm': float(math.sqrt(g_sumsq)),
        'grad_near_zero_ratio': float(g_near_zero / max(g_total, 1)),
        'grad_first_last_ratio': float(kernel_grad_norms[-1] / (kernel_grad_norms[0] + 1e-12)) if kernel_grad_norms else 0.0,
        'grad_log_std': float(np.std(np.log(np.array(kernel_grad_norms) + 1e-12))) if kernel_grad_norms else 0.0,
        'grad_layer_norms': per_layer,
    }


def analyse_model(model, x_eval, y_eval, x_grad, y_grad):
    rec = {}

    rec['measured_ebops'] = compute_model_ebops(model, x_eval)
    val_loss, val_acc = quick_eval(model, x_eval, y_eval)
    rec['val_loss_eval'] = val_loss
    rec['val_acc_eval'] = val_acc

    n_w_total = 0
    n_w_dead = 0
    n_w_low = 0
    bits_active = []

    out_total = 0
    out_disconnected = 0
    layer_deg_means = []

    active_w_values = []
    sigma_min_list = []
    sigma_max_list = []
    cond_list = []
    stable_rank_list = []

    for layer in _flatten_layers(model):
        if not hasattr(layer, 'kernel') or getattr(layer, 'kq', None) is None:
            continue

        ker = np.array(layer.kernel.numpy(), dtype=np.float32)
        bits = _bits_np(layer.kq, layer.kernel)
        if bits is None:
            continue

        active = bits > 1e-6
        low = bits <= 0.1

        n_w_total += int(bits.size)
        n_w_dead += int(np.sum(~active))
        n_w_low += int(np.sum(low))
        if np.any(active):
            bits_active.extend(bits[active].ravel().tolist())

        k2, m2 = _flatten_kernel_and_mask(ker, active.astype(np.float32))
        if m2 is None:
            continue

        deg = np.sum(m2 > 0.5, axis=0)
        out_total += int(deg.size)
        out_disconnected += int(np.sum(deg == 0))
        layer_deg_means.append(float(np.mean(deg)))

        w_active = k2[m2 > 0.5]
        if w_active.size > 0:
            active_w_values.append(w_active.astype(np.float32))

        wm = k2 * m2
        if np.all(np.abs(wm) < 1e-12):
            sigma_max = 0.0
            sigma_min = 0.0
            cond = np.inf
            stable_rank = 0.0
        else:
            s = np.linalg.svd(wm, compute_uv=False, full_matrices=False)
            sigma_max = float(s[0]) if s.size > 0 else 0.0
            sigma_min = float(s[-1]) if s.size > 0 else 0.0
            cond = float(sigma_max / (sigma_min + 1e-12)) if sigma_max > 0 else np.inf
            fro2 = float(np.sum(wm * wm))
            stable_rank = float(fro2 / (sigma_max**2 + 1e-12)) if sigma_max > 0 else 0.0

        sigma_min_list.append(sigma_min)
        sigma_max_list.append(sigma_max)
        cond_list.append(cond)
        stable_rank_list.append(stable_rank)

    rec['dead_ratio_kernel'] = float(n_w_dead / max(n_w_total, 1))
    rec['low_bits_ratio_kernel'] = float(n_w_low / max(n_w_total, 1))
    rec['mean_bits_active'] = float(np.mean(bits_active)) if bits_active else 0.0

    rec['disconnected_output_ratio'] = float(out_disconnected / max(out_total, 1))
    rec['mean_output_degree'] = float(np.mean(layer_deg_means)) if layer_deg_means else 0.0

    if active_w_values:
        w = np.concatenate(active_w_values)
        rec['weight_abs_mean'] = float(np.mean(np.abs(w)))
        rec['weight_abs_std'] = float(np.std(np.abs(w)))
        rec['weight_near_zero_ratio'] = float(np.mean(np.abs(w) < 1e-3))
        rec['weight_kurtosis'] = _kurtosis(w)
    else:
        rec['weight_abs_mean'] = 0.0
        rec['weight_abs_std'] = 0.0
        rec['weight_near_zero_ratio'] = 1.0
        rec['weight_kurtosis'] = 0.0

    rec['spectral_sigma_min_min'] = float(np.min(sigma_min_list)) if sigma_min_list else 0.0
    rec['spectral_sigma_max_mean'] = float(np.mean(sigma_max_list)) if sigma_max_list else 0.0
    finite_conds = [c for c in cond_list if np.isfinite(c)]
    rec['spectral_condition_median'] = float(np.median(finite_conds)) if finite_conds else 1e12
    rec['spectral_stable_rank_mean'] = float(np.mean(stable_rank_list)) if stable_rank_list else 0.0

    gstat = _grad_stats(model, x_grad, y_grad)
    rec['grad_global_norm'] = gstat['grad_global_norm']
    rec['grad_near_zero_ratio'] = gstat['grad_near_zero_ratio']
    rec['grad_first_last_ratio'] = gstat['grad_first_last_ratio']
    rec['grad_log_std'] = gstat['grad_log_std']
    rec['grad_batch_loss'] = gstat['loss']

    # one-step trainability proxy: use smaller LR for stability
    before = gstat['loss']
    backups = [v.numpy().copy() for v in model.trainable_variables]
    opt = keras.optimizers.SGD(learning_rate=1e-5)
    opt.apply_gradients([(g, v) for g, v in zip(gstat['grads'], model.trainable_variables) if g is not None])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    after = to_float(loss_fn(y_grad, model(x_grad, training=True)))
    for v, b in zip(model.trainable_variables, backups):
        v.assign(b)

    drop = float(before - after)
    rec['one_step_loss_drop'] = drop if np.isfinite(drop) else -1e9

    return rec


def reinit_active_kernels(model, seed=0):
    rng = np.random.default_rng(seed)
    for layer in _flatten_layers(model):
        if not hasattr(layer, 'kernel') or getattr(layer, 'kq', None) is None:
            continue
        ker = np.array(layer.kernel.numpy(), dtype=np.float32)
        bits = _bits_np(layer.kq, layer.kernel)
        if bits is None:
            continue
        active = bits > 1e-6
        fan_in = int(np.prod(ker.shape[:-1])) if ker.ndim >= 2 else int(ker.size)
        std = float(math.sqrt(2.0 / max(fan_in, 1)))
        rand = rng.normal(loc=0.0, scale=std, size=ker.shape).astype(np.float32)
        ker_new = np.where(active, rand, 0.0).astype(np.float32)
        layer.kernel.assign(ker_new)
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.assign(np.zeros_like(layer.bias.numpy(), dtype=np.float32))


def trainability_200(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=200,
    batch_size=4096,
    learning_rate=3e-4,
    seed=42,
):
    tf.keras.utils.set_random_seed(seed)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        int(x_train.shape[0]), seed=seed, reshuffle_each_iteration=True
    ).repeat().batch(int(batch_size), drop_remainder=False)
    val_bs = int(min(1024, x_val.shape[0]))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(val_bs, drop_remainder=False)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    init_loss, init_acc = quick_eval(model, x_val, y_val)
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=1,
        epochs=int(epochs),
        verbose=0,
    ).history

    val_acc_hist = [float(v) for v in hist.get('val_accuracy', [])]
    val_loss_hist = [float(v) for v in hist.get('val_loss', [])]

    final_acc = float(val_acc_hist[-1]) if val_acc_hist else float(init_acc)
    best_acc = float(max(val_acc_hist)) if val_acc_hist else float(init_acc)
    final_loss = float(val_loss_hist[-1]) if val_loss_hist else float(init_loss)

    return {
        'train200_init_acc': float(init_acc),
        'train200_final_acc': final_acc,
        'train200_best_acc': best_acc,
        'train200_acc_gain': float(final_acc - init_acc),
        'train200_init_loss': float(init_loss),
        'train200_final_loss': final_loss,
        'train200_loss_drop': float(init_loss - final_loss),
    }


def collect_state(model, x_grad, y_grad):
    layer_names = []
    masks = []
    bits_mean = []
    dead_ratio = []
    b_mean = []

    for layer in _flatten_layers(model):
        if not hasattr(layer, 'kernel') or getattr(layer, 'kq', None) is None:
            continue
        bits = _bits_np(layer.kq, layer.kernel)
        if bits is None:
            continue

        active = bits > 1e-6
        _, m2 = _flatten_kernel_and_mask(np.array(layer.kernel.numpy(), dtype=np.float32), active.astype(np.float32))

        layer_names.append(layer.name)
        masks.append(m2)
        bits_mean.append(float(np.mean(bits)))
        dead_ratio.append(float(np.mean(~active)))

        b_var = _get_kq_var(layer.kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(layer.kq, 'f')
        b_mean.append(float(np.mean(b_var.numpy())) if b_var is not None else 0.0)

    gstat = _grad_stats(model, x_grad, y_grad)
    grad_by_layer = {ln: gstat['grad_layer_norms'].get(ln, 0.0) for ln in layer_names}

    return {
        'layer_names': layer_names,
        'masks': masks,
        'bits_mean': bits_mean,
        'dead_ratio': dead_ratio,
        'b_mean': b_mean,
        'grad_by_layer': grad_by_layer,
    }


def save_csv(records, path):
    if not records:
        return
    keys = list(records[0].keys())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for r in records:
            vals = []
            for k in keys:
                v = r.get(k, '')
                if isinstance(v, float):
                    vals.append(f'{v:.8g}')
                else:
                    vals.append(str(v))
            f.write(','.join(vals) + '\n')


def to_cols(records):
    keys = records[0].keys()
    cols = {k: [] for k in keys}
    for r in records:
        for k in keys:
            cols[k].append(r[k])
    return cols


def plot_metric_multi(cols_by_method, y_key, title, ylabel, out_path, log_x=True):
    plt.figure(figsize=(6.4, 4.2))
    for key, label, color, marker in METHODS:
        if key not in cols_by_method:
            continue
        c = cols_by_method[key]
        plt.plot(c['measured_ebops'], c[y_key], marker=marker, color=color, label=label)
    if log_x:
        plt.xscale('log')
    plt.xlabel('Measured EBOPs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_state_connectivity(target, states_by_method, out_path):
    methods_present = [m for m in METHODS if m[0] in states_by_method]
    if not methods_present:
        return
    layer_names = states_by_method[methods_present[0][0]]['layer_names']
    nrows = len(methods_present)
    ncols = len(layer_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)

    for r, (mkey, mlabel, color, marker) in enumerate(methods_present):
        st = states_by_method[mkey]
        for c, lname in enumerate(layer_names):
            ax = axes[r, c]
            m = st['masks'][c]
            ax.imshow(m, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
            if r == 0:
                ax.set_title(lname)
            if c == 0:
                ax.set_ylabel(mlabel)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f'Connectivity Masks by Layer (target={target})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_state_gradients(target, states_by_method, out_path):
    methods_present = [m for m in METHODS if m[0] in states_by_method]
    if not methods_present:
        return
    layer_names = states_by_method[methods_present[0][0]]['layer_names']
    x = np.arange(len(layer_names))

    series = {}
    has_positive = False
    for mkey, _, _, _ in methods_present:
        st = states_by_method[mkey]
        y_raw = np.array([st['grad_by_layer'].get(ln, 0.0) for ln in layer_names], dtype=np.float32)
        series[mkey] = y_raw
        has_positive = has_positive or bool(np.any(y_raw > 0.0))

    plt.figure(figsize=(7.2, 4.2))
    for mkey, mlabel, color, marker in methods_present:
        y_raw = series[mkey]
        y_plot = np.maximum(y_raw, 1e-12) if has_positive else y_raw
        plt.plot(x, y_plot, marker=marker, color=color, label=mlabel)
    if has_positive:
        plt.yscale('log')
    plt.xticks(x, layer_names)
    plt.ylabel('Per-layer grad norm (log)' if has_positive else 'Per-layer grad norm')
    plt.title(f'Gradient Flow by Layer (target={target})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_state_quant(target, states_by_method, out_path):
    methods_present = [m for m in METHODS if m[0] in states_by_method]
    if not methods_present:
        return
    layer_names = states_by_method[methods_present[0][0]]['layer_names']
    x = np.arange(len(layer_names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
    axes = axes[0]

    for mkey, mlabel, color, marker in methods_present:
        st = states_by_method[mkey]
        axes[0].plot(x, st['bits_mean'], marker=marker, color=color, label=mlabel)
        axes[1].plot(x, st['dead_ratio'], marker=marker, color=color, label=mlabel)
        axes[2].plot(x, st['b_mean'], marker=marker, color=color, label=mlabel)

    axes[0].set_title('Mean bits (kq.bits)')
    axes[1].set_title('Dead ratio (kq.bits<=0)')
    axes[2].set_title('Mean b/f parameter')
    for ax in axes:
        ax.set_xticks(x, layer_names)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(f'Quantizer Parameters by Layer (target={target})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(pairing, out_path):
    lines = ['# Gradual vs One-shot vs Optimized One-shot', '']

    def _avg_delta(ps, a, b, key):
        if not ps:
            return 0.0
        arr = [p[f'{a}_{key}'] - p[f'{b}_{key}'] for p in ps]
        return float(np.mean(arr))

    low = [p for p in pairing if p['target_ebops'] <= 700]
    mid = [p for p in pairing if 700 < p['target_ebops'] <= 5000]
    high = [p for p in pairing if p['target_ebops'] > 5000]

    for name, ps in [('low(<=700)', low), ('mid(700,5000]', mid), ('high(>5000)', high), ('all', pairing)]:
        if not ps:
            continue
        lines.append(f'## {name}')
        lines.append(f"- oneshot_opt - oneshot (val_acc_eval): {_avg_delta(ps,'oneshot_opt','oneshot','val_acc_eval'):+.4f}")
        lines.append(f"- oneshot_opt - gradual (val_acc_eval): {_avg_delta(ps,'oneshot_opt','gradual','val_acc_eval'):+.4f}")
        lines.append(f"- oneshot_opt - oneshot (one_step_loss_drop): {_avg_delta(ps,'oneshot_opt','oneshot','one_step_loss_drop'):+.4e}")
        lines.append(f"- oneshot_opt - oneshot (dead_ratio_kernel): {_avg_delta(ps,'oneshot_opt','oneshot','dead_ratio_kernel'):+.4f}")
        lines.append(f"- oneshot_opt - oneshot (disconnected_output_ratio): {_avg_delta(ps,'oneshot_opt','oneshot','disconnected_output_ratio'):+.4f}")
        lines.append(f"- oneshot_opt - oneshot (spectral_stable_rank_mean): {_avg_delta(ps,'oneshot_opt','oneshot','spectral_stable_rank_mean'):+.4f}")
        if f'random_init_train200_best_acc' in ps[0]:
            lines.append(f"- random_init - oneshot_opt (train200_best_acc): {_avg_delta(ps,'random_init','oneshot_opt','train200_best_acc'):+.4f}")
            lines.append(f"- oneshot_opt - oneshot (train200_best_acc): {_avg_delta(ps,'oneshot_opt','oneshot','train200_best_acc'):+.4f}")
            lines.append(f"- oneshot_opt - gradual (train200_best_acc): {_avg_delta(ps,'oneshot_opt','gradual','train200_best_acc'):+.4f}")
        lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Analyse gradual/oneshot/optimized/random-init trainability')
    parser.add_argument('--train200_epochs', type=int, default=200)
    parser.add_argument('--train200_train_samples', type=int, default=4096)
    parser.add_argument('--train200_val_samples', type=int, default=2048)
    parser.add_argument('--train200_batch_size', type=int, default=4096)
    parser.add_argument('--train200_lr', type=float, default=3e-4)
    args = parser.parse_args()

    os.chdir(JSC_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    gradual = parse_gradual_checkpoints()
    oneshot = parse_weight_records(ONESHOT_DIR)
    oneshot_opt = parse_weight_records(ONESHOT_OPT_DIR)

    if not gradual:
        raise RuntimeError('No gradual checkpoints found.')

    (x_train, y_train), (x_val, y_val), _ = get_data('data/dataset.h5', src='openml')
    x_eval = tf.constant(x_val[:4096], dtype=tf.float32)
    y_eval = tf.constant(y_val[:4096], dtype=tf.int32)
    x_grad = tf.constant(x_train[:2048], dtype=tf.float32)
    y_grad = tf.constant(y_train[:2048], dtype=tf.int32)
    n_train200 = int(min(args.train200_train_samples, x_train.shape[0]))
    n_val200 = int(min(args.train200_val_samples, x_val.shape[0]))
    x_train200 = tf.constant(x_train[:n_train200], dtype=tf.float32)
    y_train200 = tf.constant(y_train[:n_train200], dtype=tf.int32)
    x_val200 = tf.constant(x_val[:n_val200], dtype=tf.float32)
    y_val200 = tf.constant(y_val[:n_val200], dtype=tf.int32)

    records = []
    pairing = []

    for g in gradual:
        target = g['target_ebops']
        o = pick_weight_for_target(target, oneshot)
        p = pick_weight_for_target(target, oneshot_opt)

        model_map = {}

        # gradual model
        mg = keras.models.load_model(g['path'], compile=False)
        gm = analyse_model(mg, x_eval, y_eval, x_grad, y_grad)
        records.append({'kind': 'gradual', 'target_ebops': target, 'source_file': g['name'], **gm})
        model_map['gradual'] = mg

        pair_row = {'target_ebops': target, 'gradual_file': g['name']}
        pair_row.update({f'gradual_{k}': v for k, v in gm.items()})

        if o is not None:
            mo = keras.models.load_model(BASELINE_CKPT, compile=False)
            mo.load_weights(o['path'])
            om = analyse_model(mo, x_eval, y_eval, x_grad, y_grad)
            records.append({'kind': 'oneshot', 'target_ebops': target, 'source_file': o['name'], **om})
            model_map['oneshot'] = mo
            pair_row['oneshot_file'] = o['name']
            pair_row.update({f'oneshot_{k}': v for k, v in om.items()})

        if p is not None:
            mp = keras.models.load_model(BASELINE_CKPT, compile=False)
            mp.load_weights(p['path'])
            pm = analyse_model(mp, x_eval, y_eval, x_grad, y_grad)
            mp_train = keras.models.load_model(BASELINE_CKPT, compile=False)
            mp_train.load_weights(p['path'])
            pm.update(
                trainability_200(
                    mp_train,
                    x_train200,
                    y_train200,
                    x_val200,
                    y_val200,
                    epochs=args.train200_epochs,
                    batch_size=args.train200_batch_size,
                    learning_rate=args.train200_lr,
                    seed=1000 + int(target),
                )
            )
            records.append({'kind': 'oneshot_opt', 'target_ebops': target, 'source_file': p['name'], **pm})
            model_map['oneshot_opt'] = mp
            pair_row['oneshot_opt_file'] = p['name']
            pair_row.update({f'oneshot_opt_{k}': v for k, v in pm.items()})
            # random-init baseline on the same quantization/mask topology as oneshot_opt
            mr = keras.models.load_model(BASELINE_CKPT, compile=False)
            mr.load_weights(p['path'])
            reinit_active_kernels(mr, seed=2000 + int(target))
            rm = analyse_model(mr, x_eval, y_eval, x_grad, y_grad)
            mr_train = keras.models.load_model(BASELINE_CKPT, compile=False)
            mr_train.load_weights(p['path'])
            reinit_active_kernels(mr_train, seed=2000 + int(target))
            rm.update(
                trainability_200(
                    mr_train,
                    x_train200,
                    y_train200,
                    x_val200,
                    y_val200,
                    epochs=args.train200_epochs,
                    batch_size=args.train200_batch_size,
                    learning_rate=args.train200_lr,
                    seed=3000 + int(target),
                )
            )
            records.append({'kind': 'random_init', 'target_ebops': target, 'source_file': f'random-init-from-{p["name"]}', **rm})
            model_map['random_init'] = mr
            pair_row['random_init_file'] = f'random-init-from-{p["name"]}'
            pair_row.update({f'random_init_{k}': v for k, v in rm.items()})

        if o is not None:
            # oneshot train200 validation
            mo_train = keras.models.load_model(BASELINE_CKPT, compile=False)
            mo_train.load_weights(o['path'])
            ot = trainability_200(
                mo_train,
                x_train200,
                y_train200,
                x_val200,
                y_val200,
                epochs=args.train200_epochs,
                batch_size=args.train200_batch_size,
                learning_rate=args.train200_lr,
                seed=4000 + int(target),
            )
            for k, v in ot.items():
                pair_row[f'oneshot_{k}'] = v
            for r in records:
                if r.get('kind') == 'oneshot' and int(r.get('target_ebops', -1)) == int(target):
                    r.update(ot)
                    break

        # gradual train200 validation
        mg_train = keras.models.load_model(g['path'], compile=False)
        gt = trainability_200(
            mg_train,
            x_train200,
            y_train200,
            x_val200,
            y_val200,
            epochs=args.train200_epochs,
            batch_size=args.train200_batch_size,
            learning_rate=args.train200_lr,
            seed=5000 + int(target),
        )
        for k, v in gt.items():
            pair_row[f'gradual_{k}'] = v
        for r in records:
            if r.get('kind') == 'gradual' and int(r.get('target_ebops', -1)) == int(target):
                r.update(gt)
                break

        pairing.append(pair_row)

        if target in STATE_TARGETS and all(k in model_map for k in ['gradual', 'oneshot', 'oneshot_opt', 'random_init']):
            states = {k: collect_state(v, x_grad, y_grad) for k, v in model_map.items()}
            plot_state_connectivity(target, states, PLOTS_DIR / f'09_connectivity_target{target}.png')
            plot_state_gradients(target, states, PLOTS_DIR / f'10_gradients_target{target}.png')
            plot_state_quant(target, states, PLOTS_DIR / f'11_quant_params_target{target}.png')

        print(f"[Analyse] target={target} done")

    save_csv(records, OUT_DIR / 'metrics.csv')
    save_csv(pairing, OUT_DIR / 'pairing.csv')

    cols_by_method = {}
    for key, _, _, _ in METHODS:
        rs = [r for r in records if r['kind'] == key]
        if rs:
            rs = sorted(rs, key=lambda x: x['measured_ebops'])
            cols_by_method[key] = to_cols(rs)

    plot_metric_multi(cols_by_method, 'val_acc_eval',
                      'Validation Accuracy vs EBOPs', 'Val Accuracy',
                      PLOTS_DIR / '01_val_acc_vs_ebops.png')
    plot_metric_multi(cols_by_method, 'one_step_loss_drop',
                      'One-step Loss Drop (Trainability Proxy)', 'Loss Drop after 1 SGD step',
                      PLOTS_DIR / '02_one_step_trainability.png')
    plot_metric_multi(cols_by_method, 'train200_best_acc',
                      '200-step Validation Best Accuracy', 'Best val acc in 200 steps',
                      PLOTS_DIR / '12_train200_best_acc.png')
    plot_metric_multi(cols_by_method, 'train200_acc_gain',
                      '200-step Validation Accuracy Gain', 'Final val acc - init val acc',
                      PLOTS_DIR / '13_train200_acc_gain.png')
    plot_metric_multi(cols_by_method, 'train200_loss_drop',
                      '200-step Validation Loss Drop', 'Init val loss - final val loss',
                      PLOTS_DIR / '14_train200_loss_drop.png')
    plot_metric_multi(cols_by_method, 'dead_ratio_kernel',
                      'Quantization Dead-zone Ratio', 'Dead Ratio (kernel bits<=0)',
                      PLOTS_DIR / '03_quant_deadzone.png')
    plot_metric_multi(cols_by_method, 'grad_global_norm',
                      'Gradient Global Norm', 'Global Grad Norm',
                      PLOTS_DIR / '04_gradient_flow.png')
    plot_metric_multi(cols_by_method, 'disconnected_output_ratio',
                      'Topology Disconnected Output Ratio', 'Disconnected Output Ratio',
                      PLOTS_DIR / '05_topology_connectivity.png')
    plot_metric_multi(cols_by_method, 'weight_near_zero_ratio',
                      'Weight Near-zero Ratio', 'P(|w|<1e-3) on active weights',
                      PLOTS_DIR / '06_weight_distribution.png')
    plot_metric_multi(cols_by_method, 'spectral_sigma_min_min',
                      'Spectral Minimum Singular Value', 'min sigma_min across layers',
                      PLOTS_DIR / '07_spectral_sigma_min.png')
    plot_metric_multi(cols_by_method, 'spectral_condition_median',
                      'Spectral Condition Number', 'Median condition number',
                      PLOTS_DIR / '08_spectral_condition.png')

    # dashboard
    specs = [
        ('val_acc_eval', 'Val Acc'),
        ('train200_best_acc', 'Train200 Best Acc'),
        ('dead_ratio_kernel', 'Dead Ratio'),
        ('grad_global_norm', 'Grad Norm'),
        ('disconnected_output_ratio', 'Disconnected Out Ratio'),
        ('weight_near_zero_ratio', 'Near-zero Weight Ratio'),
        ('spectral_sigma_min_min', 'min sigma_min'),
        ('spectral_condition_median', 'Cond Median'),
        ('spectral_stable_rank_mean', 'Stable Rank Mean'),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (k, ylab) in zip(axes.ravel(), specs):
        for mkey, mlabel, color, marker in METHODS:
            if mkey not in cols_by_method:
                continue
            c = cols_by_method[mkey]
            ax.plot(c['measured_ebops'], c[k], marker=marker, color=color, label=mlabel)
        ax.set_xscale('log')
        ax.set_xlabel('Measured EBOPs')
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / '00_dashboard.png', dpi=180)
    plt.close(fig)

    write_summary(pairing, OUT_DIR / 'summary.md')

    meta = {
        'n_gradual': len(gradual),
        'n_oneshot': len(oneshot),
        'n_oneshot_opt': len(oneshot_opt),
        'n_random_init': len([r for r in records if r['kind'] == 'random_init']),
        'n_records_by_kind': {k: len([r for r in records if r['kind'] == k]) for k, _, _, _ in METHODS},
        'n_pairs': len(pairing),
        'baseline_ckpt': str(BASELINE_CKPT),
        'eval_samples': int(x_eval.shape[0]),
        'grad_samples': int(x_grad.shape[0]),
        'train200_epochs': int(args.train200_epochs),
        'train200_train_samples': int(n_train200),
        'train200_val_samples': int(n_val200),
        'train200_batch_size': int(args.train200_batch_size),
        'train200_lr': float(args.train200_lr),
        'state_targets': sorted(list(STATE_TARGETS)),
    }
    with open(OUT_DIR / 'run_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Analysis done. Outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
