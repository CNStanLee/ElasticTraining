"""
结构化剪枝 + finetune：从 baseline checkpoint 剪到 EBOPS=300。

与 train_run_prune_finetune.py 的关键区别
-----------------------------------------
旧方案：均匀缩放所有 b_k × α → 全部 ~0.04 → STE 梯度全死 → 训不动
本方案：选 ~100 条最重要的连接保留 b_k=3.0，其余 b_k=0 → 可训练

原理
----
baseline 经过 200k epoch 训练，权重和位宽已经自然地将信息集中到少量连接。
利用 |kernel × b_k| 作为重要性分数，保留 top-N 条连接（刚好满足 EBOPS 预算），
其余清零。这样做的优势：
  1. 存活连接的权重是 baseline 精心训练出的（比 Glorot 随机好得多）
  2. 存活连接有足够位宽（b_k=3）→ STE 梯度有信号 → 可训练
  3. 起点已经有一定精度（比从零开始快得多）
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import argparse

import keras
import numpy as np
import tensorflow as tf

from data.data import get_data
from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PieceWiseSchedule
from keras.callbacks import LearningRateScheduler
from model.model import get_model_hgq
from utils.train_utils import cosine_decay_restarts_schedule, TrainingTraceToH5
from utils.tf_device import get_tf_device
from utils.ramanujan_budget_utils import (
    structured_prune_for_budget,
    apply_3d_ramanujan_init,
    BetaOnlyBudgetController,
    _flatten_layers,
    _get_kq_var,
)

np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

# Baseline checkpoint（高精度、高 EBOPS）
BASELINE_CKPT = 'results/baseline/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras'

TARGET_EBOPS = 300

input_folder = 'data/dataset.h5'
batch_size   = 33200

# ── 剪枝参数 ─────────────────────────────────────────────────────────────────
PRUNE_INIT_BK      = 3.0     # 存活连接的初始 b_k
PRUNE_B_A           = 3.0     # 激活位宽（用于 EBOPS 计算）
PRUNE_MIN_PER_LAYER = 5       # 每层至少保留的连接数

# ── 训练参数 ──────────────────────────────────────────────────────────────────
EPOCHS       = 30000
LR           = 5e-3
LR_CYCLE     = 2000
LR_MMUL      = 0.94

# Beta: warmup(beta=0) → log ramp → 恒定
BETA_WARMUP  = 300            # 固定值，不与 EPOCHS 成比例
BETA_MAX     = 3e-3

# Budget controller
BUDGET_CTRL_MARGIN = 0.15


# ── 命令行参数覆盖 ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Structured prune from baseline + finetune at EBOPS budget')
parser.add_argument('--target_ebops',  type=float, default=TARGET_EBOPS)
parser.add_argument('--epochs',        type=int,   default=EPOCHS)
parser.add_argument('--checkpoint',    type=str,   default=BASELINE_CKPT)
parser.add_argument('--init_bk',       type=float, default=PRUNE_INIT_BK)
args, _ = parser.parse_known_args()

TARGET_EBOPS  = args.target_ebops
EPOCHS        = args.epochs
BASELINE_CKPT = args.checkpoint
PRUNE_INIT_BK = args.init_bk

output_folder = f'results/structured_prune_{int(TARGET_EBOPS)}/'
device = get_tf_device()
os.makedirs(output_folder, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def print_bk_stats(model, label=''):
    all_b = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None:
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            all_b.extend(b_var.numpy().ravel().tolist())
    if all_b:
        arr = np.array(all_b)
        print(f'  [bk_stats {label}]  '
              f'mean={arr.mean():.3f}  std={arr.std():.3f}  '
              f'min={arr.min():.3f}  max={arr.max():.3f}  '
              f'n_active(>0.01)={int((arr>0.01).sum())}/{len(arr)}')


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('  Structured Prune + Finetune')
    print(f'  Baseline     : {BASELINE_CKPT}')
    print(f'  Target EBOPs : {TARGET_EBOPS}')
    print(f'  Init b_k     : {PRUNE_INIT_BK}')
    print(f'  Epochs       : {EPOCHS}')
    print(f'  Warmup       : {BETA_WARMUP} (beta=0)')
    print(f'  Beta max     : {BETA_MAX:.1e}')
    print(f'  Output       : {output_folder}')
    print('=' * 70)

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/5] Loading dataset...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src='openml')
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val   = Dataset(X_val,   y_val,   batch_size, device)

    # ── 2. 加载 baseline checkpoint ──────────────────────────────────────────
    print(f'\n[2/5] Loading baseline: {BASELINE_CKPT}')
    model = keras.models.load_model(BASELINE_CKPT)
    model.summary()
    print_bk_stats(model, 'baseline (before prune)')

    # 剪枝前评估
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )
    res = model.evaluate(dataset_val, verbose=0)
    print(f'  Baseline  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}')

    # ── 3. 结构化剪枝 ────────────────────────────────────────────────────────
    print(f'\n[3/5] Structured pruning to EBOPS={TARGET_EBOPS}...')
    per_layer_bk_map = structured_prune_for_budget(
        model,
        target_ebops=TARGET_EBOPS,
        b_a=PRUNE_B_A,
        init_bk=PRUNE_INIT_BK,
        min_per_layer=PRUNE_MIN_PER_LAYER,
        budget_weight='capacity',
        verbose=True,
    )

    # 应用剪枝（复用 apply_3d_ramanujan_init —— 写入 b_k map + 清零 kernel）
    print('Applying structured prune...')
    apply_3d_ramanujan_init(
        model,
        per_layer_bk_map,
        active_int_bits=1.0,
        pruned_int_bits=0.0,
        pruned_frac_bits=0.0,
        also_zero_kernel=True,
        rescale_kernel=False,    # baseline 权重已经训练好，不需要重缩放
        verbose=True,
    )
    print_bk_stats(model, 'after structured prune')

    # 剪枝后评估
    res = model.evaluate(dataset_val, verbose=0)
    print(f'  Post-prune  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}')

    # ── 4. 设置 Callbacks ────────────────────────────────────────────────────
    print(f'\n[4/5] Setting up training callbacks...')

    ebops_cb = FreeEBOPs()
    pareto_cb = ParetoFront(
        output_folder,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )

    # Beta: warmup 阶段 beta=0，之后 log ramp
    beta_sched = BetaScheduler(
        PieceWiseSchedule([
            (0,            0, 'constant'),
            (BETA_WARMUP,  1e-7, 'log'),
            (EPOCHS,       BETA_MAX, 'constant'),
        ])
    )

    lr_sched = LearningRateScheduler(
        cosine_decay_restarts_schedule(
            LR, LR_CYCLE, t_mul=1.0, m_mul=LR_MMUL,
            alpha=1e-6, alpha_steps=50,
        )
    )

    budget_ctrl = BetaOnlyBudgetController(
        target_ebops    = TARGET_EBOPS,
        warmup_epochs   = BETA_WARMUP,
        margin          = BUDGET_CTRL_MARGIN,
        beta_min        = 1e-8,
        beta_max        = BETA_MAX,
        adjust_factor   = 1.2,
        max_step_factor = 2.0,
        ema_alpha       = 0.3,
    )

    trace_cb = TrainingTraceToH5(
        output_dir=output_folder,
        filename='training_trace.h5',
        max_bits=8,
        beta_callback=beta_sched,
    )

    callbacks = [
        ebops_cb,
        pareto_cb,
        beta_sched,
        budget_ctrl,
        lr_sched,
        trace_cb,
    ]

    # ── 5. 训练 ──────────────────────────────────────────────────────────────
    print(f'\n[5/5] Training for {EPOCHS} epochs...')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 完成 ──────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('Training complete.')
    print(f'Pareto checkpoints : {output_folder}')
    print(f'Training trace     : {os.path.join(output_folder, "training_trace.h5")}')

    res = model.evaluate(dataset_val, verbose=0)
    print(f'Final  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}')
    print_bk_stats(model, 'final')
    print('=' * 70)
