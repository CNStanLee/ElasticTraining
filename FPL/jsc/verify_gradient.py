"""
verify_gradient.py — 验证 HGQ 模型随层数增加时的梯度消失/爆炸问题

实验设计：
  - 测试多种深度：3 / 5 / 7 / 9 / 11 层
  - 每种深度训练固定 epoch（默认 500，足以观察梯度行为但不浪费时间）
  - 每隔 log_every epoch 记录各层 kernel 梯度 L2 范数
  - 最终输出两张图：
      1. 每种深度的"层序号 vs 最终梯度范数"对比（看梯度是否随深度消失）
      2. 每种深度的"epoch vs 梯度范数"训练曲线（看梯度动态）

运行方式：
    cd /home/changhong/prj/ElasticTraining/FPL/jsc
    conda activate py12tf
    python verify_gradient.py
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

from data.data import get_data
from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, PieceWiseSchedule
from hgq.config import QuantizerConfigScope
from hgq.layers import QDense, QEinsumDenseBatchnorm
from keras.callbacks import LearningRateScheduler
from utils.train_utils import cosine_decay_restarts_schedule, GradientNormLogger
from utils.tf_device import get_tf_device

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_FOLDER  = 'data/dataset.h5'
OUTPUT_FOLDER = 'results/grad_verify/'
BATCH_SIZE    = 33200
TRAIN_EPOCHS  = 1000       # 每种深度训练的 epoch 数（短训，只看梯度行为）
LOG_EVERY     = 50         # 每隔多少 epoch 记录一次梯度范数
LEARNING_RATE = 5e-3
BETA_MAX      = 1e-4       # 压缩力适中，让 HGQ 量化激活但不过激压缩

# 测试的深度配置：{标签: widths 列表}
DEPTH_CONFIGS = {
    ' 3-layer': [64, 64, 32],
    ' 5-layer': [64, 64, 64, 64, 32],
    ' 7-layer': [64] * 6 + [32],
    ' 9-layer': [64] * 8 + [32],
    '11-layer': [64] * 10 + [32],
}
# ── End Config ───────────────────────────────────────────────────────────────


def build_model(widths, init_bw_k=3, init_bw_a=3):
    with (
        QuantizerConfigScope(place='weight',   overflow_mode='SAT_SYM', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place='bias',     overflow_mode='WRAP',    f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place='datalane', i0=1, f0=init_bw_a),
    ):
        inp = keras.Input(shape=(16,))
        out = inp
        for i, w in enumerate(widths):
            out = QEinsumDenseBatchnorm(
                'bc,cC->bC', w, name=f't{i}', bias_axes='C', activation='relu'
            )(out)
        out = QDense(5, name='out')(out)
    return keras.Model(inp, out)


def run_one_depth(label, widths, X_train, y_train, X_val, y_val, device):
    print(f'\n{"="*60}')
    print(f'  Running: {label}  widths={widths}')
    print(f'{"="*60}')

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    h5_path = os.path.join(OUTPUT_FOLDER, f'grad_{label.strip()}.h5')

    dataset_train = Dataset(X_train, y_train, BATCH_SIZE, device, shuffle=True)
    dataset_val   = Dataset(X_val,   y_val,   BATCH_SIZE, device)

    model = build_model(widths)
    model.summary()

    # Beta schedule（轻度压缩，让量化发挥作用）
    beta_sched = BetaScheduler(
        PieceWiseSchedule([
            (0,            5e-7,    'constant'),
            (TRAIN_EPOCHS, BETA_MAX, 'log'),
        ])
    )
    lr_sched = LearningRateScheduler(
        cosine_decay_restarts_schedule(LEARNING_RATE, 500, t_mul=1.0, m_mul=0.95, alpha=1e-6)
    )

    # 用第一个 batch 作为梯度计算的固定探针
    sample_x = X_train[:BATCH_SIZE]
    sample_y = y_train[:BATCH_SIZE]
    grad_logger = GradientNormLogger(
        dataset_for_grad=(sample_x, sample_y),
        log_every=LOG_EVERY,
        output_path=h5_path,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=16,
    )
    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=TRAIN_EPOCHS,
        callbacks=[FreeEBOPs(), beta_sched, lr_sched, grad_logger],
        verbose=0,
    )

    # 返回最后一次记录的各层梯度范数 {layer_name: norm}
    final_norms = {name: vals[-1] for name, vals in grad_logger._records.items()}
    all_records = dict(grad_logger._records)
    epochs_logged = list(grad_logger._epochs)
    print(f'\n  Final gradient norms for {label}:')
    for k, v in final_norms.items():
        print(f'    {k:12s}: {v:.4e}')
    return final_norms, all_records, epochs_logged


def plot_results(all_final, all_records, all_epochs_logged):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ── 图1：各深度最终梯度范数 vs 层序号 ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, final_norms in all_final.items():
        names = list(final_norms.keys())
        norms = list(final_norms.values())
        ax.semilogy(range(len(norms)), norms, marker='o', label=label.strip())
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Layer (input → output)')
    ax.set_ylabel('Kernel gradient L2 norm (log scale)')
    ax.set_title('HGQ Gradient norm per layer — final epoch\n'
                 'Monotone decrease toward input = vanishing gradient')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(OUTPUT_FOLDER, 'grad_norm_final.png')
    plt.savefig(out1, dpi=150)
    print(f'\nSaved: {out1}')
    plt.close()

    # ── 图2：训练曲线 — 每种深度选第一层（最易消失）和最后层对比 ────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, records in all_records.items():
        epochs = all_epochs_logged[label]
        layer_names = list(records.keys())
        first_layer = layer_names[0]   # 最靠近输入（最易消失）
        last_layer  = layer_names[-1]  # 最靠近输出（梯度最强）
        axes[0].semilogy(epochs, records[first_layer], label=label.strip())
        axes[1].semilogy(epochs, records[last_layer],  label=label.strip())
    for ax, title in zip(axes, ['First hidden layer (t0) — vanishing risk',
                                 'Last hidden layer — reference']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient L2 norm')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
    plt.suptitle('HGQ gradient norm training curves')
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_FOLDER, 'grad_norm_curves.png')
    plt.savefig(out2, dpi=150)
    print(f'Saved: {out2}')
    plt.close()


if __name__ == '__main__':
    device = get_tf_device()

    print('Loading data...')
    (X_train, y_train), (X_val, y_val), _ = get_data(INPUT_FOLDER, src='openml')

    all_final        = {}
    all_records      = {}
    all_epochs_logged = {}

    for label, widths in DEPTH_CONFIGS.items():
        final_norms, records, epochs_logged = run_one_depth(
            label, widths, X_train, y_train, X_val, y_val, device
        )
        all_final[label]         = final_norms
        all_records[label]       = records
        all_epochs_logged[label] = epochs_logged

    print('\n\nPlotting...')
    plot_results(all_final, all_records, all_epochs_logged)
    print('\nDone. Results saved to:', OUTPUT_FOLDER)
