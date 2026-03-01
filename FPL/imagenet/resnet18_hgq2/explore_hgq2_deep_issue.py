import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
from pathlib import Path

import keras
from keras import ops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from hgq.layers import QLayerBase
from hgq.utils.sugar import BetaScheduler, FreeEBOPs, PieceWiseSchedule

from imagenet_data import load_imagenet1k
from model_resnet18 import build_resnet18_fp32, build_resnet18_hgq2


class GradientProbe(keras.callbacks.Callback):
    def __init__(self, x_probe, y_probe, layer_names, log_every=1):
        super().__init__()
        self.x_probe = x_probe
        self.y_probe = y_probe
        self.layer_names = layer_names
        self.log_every = max(1, int(log_every))
        self.epochs = []
        self.records = {name: [] for name in layer_names}

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_every != 0:
            return

        with tf.GradientTape() as tape:
            logits = self.model(self.x_probe, training=True)
            loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(self.y_probe, logits, from_logits=True)
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        grad_map = {id(v): g for v, g in zip(self.model.trainable_variables, grads) if g is not None}

        self.epochs.append(int(epoch))
        for name in self.layer_names:
            layer = self.model.get_layer(name)
            value = float('nan')
            # Primary: try layer.kernel directly
            if hasattr(layer, 'kernel') and id(layer.kernel) in grad_map:
                value = float(tf.norm(grad_map[id(layer.kernel)]).numpy())
            else:
                # Fallback: search through layer's trainable_weights for kernel variable
                for w in layer.trainable_weights:
                    if 'kernel' in w.name and id(w) in grad_map:
                        value = float(tf.norm(grad_map[id(w)]).numpy())
                        break
            # Guard against NaN / Inf
            if not np.isfinite(value):
                value = float('nan')
            self.records[name].append(value)


class HGQBitwidthProbe(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.mean_bits = []
        self.min_bits = []
        self.pct_le_1 = []
        self.pct_le_2 = []

    def _collect_bits(self):
        def extract_bits(qobj):
            vals = []
            if qobj is None:
                return vals
            if hasattr(qobj, 'bits') and getattr(qobj, 'built', False):
                try:
                    vals.extend(ops.convert_to_numpy(qobj.bits).reshape(-1).tolist())
                except Exception:
                    pass
                return vals
            if hasattr(qobj, 'quantizers'):
                for sub_q in getattr(qobj, 'quantizers', []):
                    vals.extend(extract_bits(sub_q))
            return vals

        all_bits = []
        for layer in self.model._flatten_layers():
            if not isinstance(layer, QLayerBase):
                continue
            if hasattr(layer, '_iq'):
                all_bits.extend(extract_bits(layer._iq))
            if hasattr(layer, 'kq'):
                all_bits.extend(extract_bits(layer.kq))
            if hasattr(layer, '_oq'):
                all_bits.extend(extract_bits(layer._oq))
        return np.array(all_bits, dtype=np.float32)

    def on_epoch_end(self, epoch, logs=None):
        bits = self._collect_bits()
        self.epochs.append(int(epoch))
        if bits.size == 0:
            self.mean_bits.append(float('nan'))
            self.min_bits.append(float('nan'))
            self.pct_le_1.append(float('nan'))
            self.pct_le_2.append(float('nan'))
            return

        self.mean_bits.append(float(np.mean(bits)))
        self.min_bits.append(float(np.min(bits)))
        self.pct_le_1.append(float(np.mean(bits <= 1.0)))
        self.pct_le_2.append(float(np.mean(bits <= 2.0)))


def pick_probe_layers(model):
    kernel_layers = []
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            kernel_layers.append(layer.name)
        elif any('kernel' in w.name for w in getattr(layer, 'trainable_weights', [])):
            kernel_layers.append(layer.name)
    if len(kernel_layers) <= 6:
        return kernel_layers
    # Pick spread-out layers: first, 1/4, 1/2, 3/4, last
    n = len(kernel_layers)
    indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    return [kernel_layers[i] for i in indices]


def get_probe_batch(dataset, probe_batch: int):
    probe_batch = max(1, int(probe_batch))
    xs = []
    ys = []
    for images, labels in dataset.unbatch().take(probe_batch):
        xs.append(images)
        ys.append(labels)
    if not xs:
        raise ValueError('Probe batch is empty. Please check dataset path and subset settings.')
    x_probe = tf.stack(xs, axis=0)
    y_probe = tf.stack(ys, axis=0)
    return x_probe, y_probe


def train_fp32(args, train_ds, val_ds, num_classes: int, train_steps: int, val_steps: int, output_dir: Path, x_probe, y_probe):
    model = build_resnet18_fp32(input_shape=(args.image_size, args.image_size, 3), num_classes=num_classes)
    probe_layers = pick_probe_layers(model)
    grad_probe = GradientProbe(
        x_probe=x_probe,
        y_probe=y_probe,
        layer_names=probe_layers,
        log_every=args.log_every,
    )

    total_steps = max(1, int(train_steps) * args.epochs)
    optimizer = keras.optimizers.SGD(
        learning_rate=keras.optimizers.schedules.CosineDecay(args.lr_fp32, total_steps, alpha=1e-3),
        momentum=0.9,
        weight_decay=args.wd_fp32,
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
        callbacks=[grad_probe],
        verbose=1,
    )

    model.save(output_dir / 'fp32_last.keras')
    return history.history, grad_probe


def train_hgq2(args, train_ds, val_ds, num_classes: int, train_steps: int, val_steps: int, output_dir: Path, x_probe, y_probe):
    model = build_resnet18_hgq2(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=num_classes,
        init_bw_k=args.init_bw_k,
        init_bw_a=args.init_bw_a,
    )
    probe_layers = pick_probe_layers(model)
    grad_probe = GradientProbe(
        x_probe=x_probe,
        y_probe=y_probe,
        layer_names=probe_layers,
        log_every=args.log_every,
    )
    bw_probe = HGQBitwidthProbe()

    total_steps = max(1, int(train_steps) * args.epochs)
    optimizer = keras.optimizers.SGD(
        learning_rate=keras.optimizers.schedules.CosineDecay(args.lr_hgq2, total_steps, alpha=1e-3),
        momentum=0.9,
        weight_decay=args.wd_hgq2,
        nesterov=True,
    )

    beta_sched = BetaScheduler(
        PieceWiseSchedule([
            (0, 5e-7, 'constant'),
            (max(1, args.epochs // 10), 5e-7, 'log'),
            (args.epochs, args.beta_max, 'constant'),
        ])
    )

    spe = min(8, train_steps)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=spe,
    )

    history = model.fit(
        train_ds.repeat(),
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=[FreeEBOPs(), beta_sched, grad_probe, bw_probe],
        verbose=1,
    )

    model.save(output_dir / 'hgq2_last.keras')
    return history.history, grad_probe, bw_probe


def _safe_last(arr):
    if not arr:
        return float('nan')
    return float(arr[-1])


def _final_grad_ratio(probe: GradientProbe):
    """Ratio of first-layer / last-layer gradient norm (last epoch)."""
    if not probe.layer_names or not probe.records:
        return float('nan')
    first = probe.records[probe.layer_names[0]]
    last = probe.records[probe.layer_names[-1]]
    if not first or not last:
        return float('nan')
    num = float(first[-1])
    denom = float(last[-1])
    if not np.isfinite(num) or not np.isfinite(denom) or abs(denom) < 1e-12:
        return float('nan')
    return float(num / denom)


def plot_diagnostics(fp_probe: GradientProbe, hgq_probe: GradientProbe, bw_probe: HGQBitwidthProbe,
                     fp_hist: dict, hgq_hist: dict, output_dir: Path):
    # 1. Gradient comparison plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in fp_probe.layer_names:
        vals = fp_probe.records[name]
        valid = [(e, v) for e, v in zip(fp_probe.epochs, vals) if np.isfinite(v) and v > 0]
        if valid:
            ax.semilogy([e for e, _ in valid], [v for _, v in valid], linestyle='--', marker='o', markersize=3, label=f'fp32:{name}')
    for name in hgq_probe.layer_names:
        vals = hgq_probe.records[name]
        valid = [(e, v) for e, v in zip(hgq_probe.epochs, vals) if np.isfinite(v) and v > 0]
        if valid:
            ax.semilogy([e for e, _ in valid], [v for _, v in valid], linestyle='-', marker='s', markersize=3, label=f'hgq2:{name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Kernel gradient L2 norm (log)')
    ax.set_title('Gradient propagation comparison (FP32 vs HGQ2)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / 'grad_compare.png', dpi=150)
    plt.close(fig)

    # 2. Bitwidth evolution plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(bw_probe.epochs, bw_probe.mean_bits, label='mean bits', marker='o', markersize=3)
    ax.plot(bw_probe.epochs, bw_probe.min_bits, label='min bits', marker='s', markersize=3)
    ax.plot(bw_probe.epochs, bw_probe.pct_le_1, label='pct(bits<=1)', marker='^', markersize=3)
    ax.plot(bw_probe.epochs, bw_probe.pct_le_2, label='pct(bits<=2)', marker='v', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_title('HGQ2 bitwidth evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'bitwidth_evolution.png', dpi=150)
    plt.close(fig)

    # 3. Accuracy & loss comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 3a. Val accuracy
    fp_vacc = fp_hist.get('val_accuracy', [])
    hgq_vacc = hgq_hist.get('val_accuracy', [])
    if fp_vacc:
        axes[0].plot(range(1, len(fp_vacc) + 1), fp_vacc, label='FP32', linewidth=2)
    if hgq_vacc:
        axes[0].plot(range(1, len(hgq_vacc) + 1), hgq_vacc, label='HGQ2', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Val Accuracy')
    axes[0].set_title('Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 3b. Train loss
    fp_loss = fp_hist.get('loss', [])
    hgq_loss = hgq_hist.get('loss', [])
    if fp_loss:
        axes[1].semilogy(range(1, len(fp_loss) + 1), fp_loss, label='FP32', linewidth=2)
    if hgq_loss:
        axes[1].semilogy(range(1, len(hgq_loss) + 1), hgq_loss, label='HGQ2', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Train Loss (log)')
    axes[1].set_title('Training Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3c. EBOPs
    ebops = hgq_hist.get('ebops', [])
    if ebops:
        axes[2].semilogy(range(1, len(ebops) + 1), ebops, label='EBOPs', linewidth=2, color='red')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('EBOPs (log)')
        axes[2].set_title('HGQ2 EBOPs evolution')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / 'accuracy_loss_compare.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Explore deep-network training issues of HGQ2 ResNet-18 (ImageNet-1K)')
    parser.add_argument('--data-root', type=str, required=True, help='ImageNet root dir containing train/ and val/')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--expected-classes', type=int, default=1000)
    parser.add_argument('--strict-classes', action='store_true', help='Raise error when class count mismatches expected-classes')
    parser.add_argument('--train-subset', type=int, default=20000)
    parser.add_argument('--val-subset', type=int, default=5000)
    parser.add_argument('--probe-batch', type=int, default=128)
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--lr-fp32', type=float, default=0.08)
    parser.add_argument('--wd-fp32', type=float, default=1e-4)
    parser.add_argument('--lr-hgq2', type=float, default=0.05)
    parser.add_argument('--wd-hgq2', type=float, default=2e-4)
    parser.add_argument('--init-bw-k', type=int, default=2)
    parser.add_argument('--init-bw-a', type=int, default=2)
    parser.add_argument('--beta-max', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default='results/resnet18_hgq2/explore')
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

    x_probe, y_probe = get_probe_batch(val_ds, args.probe_batch)

    fp_hist, fp_probe = train_fp32(args, train_ds, val_ds, num_classes, train_steps, val_steps, output_dir, x_probe, y_probe)
    hgq_hist, hgq_probe, bw_probe = train_hgq2(args, train_ds, val_ds, num_classes, train_steps, val_steps, output_dir, x_probe, y_probe)

    fp_ratio = _final_grad_ratio(fp_probe)
    hgq_ratio = _final_grad_ratio(hgq_probe)
    val_acc_gap = _safe_last(fp_hist.get('val_accuracy', [])) - _safe_last(hgq_hist.get('val_accuracy', []))

    findings = []
    if np.isfinite(val_acc_gap) and val_acc_gap > 0.05:
        findings.append(f'HGQ2 val_accuracy 低于 FP32 {val_acc_gap:.1%}，深层量化训练存在显著性能损失。')
    if np.isfinite(fp_ratio) and np.isfinite(hgq_ratio) and hgq_ratio < fp_ratio * 0.3:
        findings.append(f'HGQ2 的前层/后层梯度比({hgq_ratio:.4f})明显低于 FP32({fp_ratio:.4f})，前层梯度衰减更严重。')
    elif not np.isfinite(hgq_ratio) and np.isfinite(fp_ratio):
        # Check if HGQ2 gradients are all zero/NaN
        hgq_all_nan = all(
            (not hgq_probe.records[n] or not np.isfinite(hgq_probe.records[n][-1]) or abs(hgq_probe.records[n][-1]) < 1e-20)
            for n in hgq_probe.layer_names
        )
        if hgq_all_nan:
            findings.append('HGQ2 所有probe层的梯度范数为 0 或 NaN，网络梯度完全消失(dead network)。')
        else:
            findings.append('HGQ2 梯度比不可计算(NaN)，部分层梯度异常。')
    if bw_probe.pct_le_1 and np.isfinite(bw_probe.pct_le_1[-1]) and bw_probe.pct_le_1[-1] > 0.02:
        findings.append(f'{bw_probe.pct_le_1[-1]:.1%} 的量化比特降到 <=1 bit，存在位宽坍缩现象。')
    if bw_probe.min_bits and np.isfinite(bw_probe.min_bits[-1]) and bw_probe.min_bits[-1] < 0.5:
        findings.append(f'最低位宽降至 {bw_probe.min_bits[-1]:.2f} bit，部分通道几乎完全关闭。')
    # Check EBOPs collapse
    ebops_list = hgq_hist.get('ebops', [])
    if len(ebops_list) >= 2:
        initial_ebops = ebops_list[0]
        final_ebops = ebops_list[-1]
        if initial_ebops > 0 and final_ebops / initial_ebops < 0.01:
            findings.append(f'EBOPs 从 {initial_ebops:.0f} 骤降到 {final_ebops:.0f} (仅剩 {final_ebops/initial_ebops:.2%})，计算量几乎归零。')
    if not findings:
        findings.append('当前配置下未观察到强烈异常，可尝试提高 beta_max 或增大训练轮数。')

    # Per-layer gradient history for debugging
    fp_grad_history = {name: fp_probe.records[name] for name in fp_probe.layer_names}
    hgq_grad_history = {name: hgq_probe.records[name] for name in hgq_probe.layer_names}

    report = {
        'config': vars(args),
        'fp32': {
            'final_train_acc': _safe_last(fp_hist.get('accuracy', [])),
            'final_val_acc': _safe_last(fp_hist.get('val_accuracy', [])),
            'final_grad_ratio_first_last': fp_ratio,
            'probe_layers': fp_probe.layer_names,
            'grad_history': fp_grad_history,
        },
        'hgq2': {
            'final_train_acc': _safe_last(hgq_hist.get('accuracy', [])),
            'final_val_acc': _safe_last(hgq_hist.get('val_accuracy', [])),
            'final_ebops': _safe_last(hgq_hist.get('ebops', [])),
            'final_grad_ratio_first_last': hgq_ratio,
            'final_mean_bits': _safe_last(bw_probe.mean_bits),
            'final_min_bits': _safe_last(bw_probe.min_bits),
            'final_pct_bits_le_1': _safe_last(bw_probe.pct_le_1),
            'final_pct_bits_le_2': _safe_last(bw_probe.pct_le_2),
            'probe_layers': hgq_probe.layer_names,
            'grad_history': hgq_grad_history,
            'bitwidth_history': {
                'epochs': bw_probe.epochs,
                'mean_bits': bw_probe.mean_bits,
                'min_bits': bw_probe.min_bits,
                'pct_le_1': bw_probe.pct_le_1,
                'pct_le_2': bw_probe.pct_le_2,
            },
            'ebops_history': ebops_list,
        },
        'num_classes_detected': int(num_classes),
        'val_acc_gap_fp32_minus_hgq2': val_acc_gap,
        'findings': findings,
    }

    with open(output_dir / 'diagnostic_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_diagnostics(fp_probe, hgq_probe, bw_probe, fp_hist, hgq_hist, output_dir)

    print('Diagnostic finished. Report saved to:')
    print(output_dir / 'diagnostic_report.json')
    for msg in findings:
        print('-', msg)


if __name__ == '__main__':
    main()
