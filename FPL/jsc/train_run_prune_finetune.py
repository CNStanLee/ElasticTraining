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
from hgq.utils.sugar import Dataset, FreeEBOPs, ParetoFront
from keras.callbacks import LearningRateScheduler
from utils.train_utils import cosine_decay_restarts_schedule, TrainingTraceToH5, BudgetAwareEarlyStopping
from utils.tf_device import get_tf_device
from utils.ramanujan_budget_utils import (
    EBOPsConstantProjector,
    BetaZeroCallback,
    HighBitPruner,
    SensitivityAwarePruner,
    BetaOnlyBudgetController,
    _flatten_layers,
    _get_kq_var,
)

np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

# baseline epoch=2236 最高精度检查点（val_acc=0.770，ebops=23589）
BASELINE_CKPT  = 'results/baseline/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras'
BASELINE_EBOPS = 19899      # 从文件名读，作为剪枝前参考值

TARGET_EBOPS = 3000

# ── 剪枝方式 ──────────────────────────────────────────────────────────────────
# 'uniform'     : 旧方案 HighBitPruner (均匀缩放，有精度损失)
# 'sensitivity' : 新方案 SensitivityAwarePruner (按层敏感度分配预算)
PRUNE_METHOD = 'sensitivity'

input_folder = 'data/dataset.h5'
batch_size   = 33200

# ── Phase 1：恢复期 ─────────────────────────────────────────────────────────
# 一次性位宽剪枝后，权重与量化工作点失配。
# 需要在高 LR 下让权重重新适应低精度量化点。
# **关键**：用 BetaOnlyBudgetController 双向控制 EBOPs：
#   - EBOPs > target → 增大 beta → 梯度压低位宽
#   - EBOPs < target → 减小 beta（趋近于 0）→ 停止压缩，允许位宽回升
# 仅用 ConstantBeta 会导致 EBOPs 单调下降（只有压力没有释放）。
PHASE1_EPOCHS    = 5000
PHASE1_LR        = 2e-3      # 高 LR：加快重适应
PHASE1_LR_CYCLE  = 2000
PHASE1_LR_MMUL   = 0.9
PHASE1_BETA_INIT = 3e-5      # controller 起始值
PHASE1_BETA_MIN  = 1e-8      # 近零下限：EBOPs 低于目标时几乎不压缩
PHASE1_BETA_MAX  = 2e-4      # 上限：防止过度压缩

# ── Phase 2：精度最大化 ──────────────────────────────────────────────────────
# 核心改进：完全取消均匀投影器。
# 原因：均匀投影锁死各层相对位宽比例，HGQ 无法按重要性跨层重分配位宽，
#       而 baseline 0.765 正是靠 ~80k epoch 训练出的自适应分配实现的。
#
# 新方案：用 BetaOnlyBudgetController 自适应调整 beta。
#   - EBOPs 超标时 → 自动增大 beta → 梯度自然压低位宽
#   - EBOPs 低于目标时 → 自动减小 beta → 允许位宽回升
#   - HGQ 梯度全程自由决定 per-layer 分配，不受均匀约束
PHASE2_EPOCHS    = 30000     # 增加 epoch：给 HGQ 足够时间找到最优分配
PHASE2_LR        = 5e-4
PHASE2_LR_CYCLE  = 800
PHASE2_LR_MMUL   = 0.95
# Phase 2 beta 由 BetaOnlyBudgetController 自动管理
PHASE2_BETA_INIT = 1e-5      # controller 起始值
PHASE2_BETA_MIN  = 1e-7
PHASE2_BETA_MAX  = 5e-4      # 不超过 baseline 的 beta_max=1e-3

EARLYSTOP_PATIENCE = 5000
EARLYSTOP_BUDGET   = TARGET_EBOPS * 1.5   # 50% 弹性窗口


# ── 命令行参数覆盖 ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='One-shot prune + two-phase finetune')
parser.add_argument('--target_ebops',  type=float, default=TARGET_EBOPS)
parser.add_argument('--phase1_epochs', type=int,   default=PHASE1_EPOCHS)
parser.add_argument('--phase2_epochs', type=int,   default=PHASE2_EPOCHS)
parser.add_argument('--checkpoint',    type=str,   default=BASELINE_CKPT)
parser.add_argument('--prune_method',  type=str,   default=PRUNE_METHOD,
                    choices=['uniform', 'sensitivity'])
args, _ = parser.parse_known_args()

TARGET_EBOPS    = args.target_ebops
PHASE1_EPOCHS   = args.phase1_epochs
PHASE2_EPOCHS   = args.phase2_epochs
BASELINE_CKPT   = args.checkpoint
PRUNE_METHOD    = args.prune_method
EARLYSTOP_BUDGET = TARGET_EBOPS * 1.5

output_folder = f'results/prune_finetune_{int(TARGET_EBOPS)}/'
device = get_tf_device()
os.makedirs(output_folder, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def set_all_beta(model, beta_value: float):
    """强制把所有 HGQ 层的 _beta 设为固定值。"""
    bv = tf.constant(beta_value, dtype=tf.float32)
    for layer in _flatten_layers(model):
        if hasattr(layer, '_beta'):
            try:
                layer._beta.assign(tf.ones_like(layer._beta) * bv)
            except Exception:
                pass


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
              f'p5={np.percentile(arr,5):.3f}  p95={np.percentile(arr,95):.3f}  '
              f'n_dead(<=0.1)={int((arr<=0.1).sum())}/{len(arr)}')


def make_lr_scheduler(lr_init, cycle, mmul, offset=0):
    """构造 LearningRateScheduler，offset 用于 phase2（epoch 从 PHASE1_EPOCHS 开始）。"""
    fn = cosine_decay_restarts_schedule(lr_init, cycle, t_mul=1.0, m_mul=mmul,
                                        alpha=1e-6, alpha_steps=50)
    def schedule(epoch):
        return fn(max(0, epoch - offset))
    return LearningRateScheduler(schedule)


class ConstantBetaCallback(keras.callbacks.Callback):
    """每个 epoch 开始时把 beta 固定为常数（用于 phase1 恢复期）。"""
    def __init__(self, beta_value: float):
        super().__init__()
        self.beta_value = beta_value

    def on_epoch_begin(self, epoch, logs=None):
        set_all_beta(self.model, self.beta_value)


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

    print('=' * 65)
    print(f'  One-shot Prune + Two-phase Finetune')
    print(f'  Source      : {BASELINE_CKPT}')
    print(f'  Target      : ebops = {TARGET_EBOPS}')
    print(f'  Prune method: {PRUNE_METHOD}')
    print(f'  Phase1      : {PHASE1_EPOCHS} epochs  (recovery,  beta auto [{PHASE1_BETA_INIT:.1e}~{PHASE1_BETA_MAX:.1e}])')
    print(f'  Phase2      : {PHASE2_EPOCHS} epochs  (acc-max,   beta auto [{PHASE2_BETA_INIT:.1e}~{PHASE2_BETA_MAX:.1e}])')
    print(f'  Output      : {output_folder}')
    print('=' * 65)

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/5] Loading dataset...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src='openml')
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val   = Dataset(X_val,   y_val,   batch_size, device)

    # ── 2. 加载最优权重 ──────────────────────────────────────────────────────
    print(f'\n[2/5] Loading checkpoint: {BASELINE_CKPT}')
    model = keras.models.load_model(BASELINE_CKPT)
    model.summary()
    print_bk_stats(model, 'before pruning')

    # ── 3. 一次性位宽剪枝 ────────────────────────────────────────────────────
    print(f'\n[3/5] One-shot bit-width pruning ({PRUNE_METHOD}): {BASELINE_EBOPS} -> {TARGET_EBOPS}')
    if PRUNE_METHOD == 'sensitivity':
        pruner = SensitivityAwarePruner(target_ebops=TARGET_EBOPS, pruned_threshold=0.1)
    else:
        pruner = HighBitPruner(target_ebops=TARGET_EBOPS, pruned_threshold=0.1)
    pruner.prune_to_ebops(model, current_ebops=BASELINE_EBOPS, verbose=True)
    print_bk_stats(model, 'after pruning')

    # 剪枝后快速评估（预期精度有明显跌落，属正常现象）
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )
    res = model.evaluate(dataset_val, verbose=0)
    print(f'  Post-pruning  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}')

    # ── 共享 Callbacks（两个 phase 都用） ─────────────────────────────────────
    ebops_cb  = FreeEBOPs()
    pareto_cb = ParetoFront(
        output_folder,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )
    trace_cb = TrainingTraceToH5(
        output_dir=output_folder,
        filename='training_trace.h5',
        max_bits=8,
        beta_callback=None,
    )

    # 重新编译（重置 Adam 动量，不带旧初始化动量）
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1：恢复期
    # 高 LR + BetaOnlyBudgetController 双向控制：
    #   - 权重重新适应低精度量化工作点
    #   - EBOPs 低于 target 时自动减小 beta，防止过度压缩
    #   - EBOPs 高于 target 时自动增大 beta，维持预算
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n[4/5] PHASE 1  Recovery  ({PHASE1_EPOCHS} epochs, '
          f'lr={PHASE1_LR:.1e}, beta=auto [{PHASE1_BETA_INIT:.1e}~{PHASE1_BETA_MAX:.1e}])')

    set_all_beta(model, PHASE1_BETA_INIT)

    phase1_budget_ctrl = BetaOnlyBudgetController(
        target_ebops  = TARGET_EBOPS,
        margin        = 0.15,          # ±15% 容忍带
        beta_init     = PHASE1_BETA_INIT,
        beta_min      = PHASE1_BETA_MIN,
        beta_max      = PHASE1_BETA_MAX,
        adjust_factor = 1.3,
        ema_alpha     = 0.3,
    )

    phase1_callbacks = [
        ebops_cb,
        pareto_cb,
        # 双向 budget controller：EBOPs 过低时减 beta，过高时加 beta
        phase1_budget_ctrl,
        make_lr_scheduler(PHASE1_LR, PHASE1_LR_CYCLE, PHASE1_LR_MMUL, offset=0),
        trace_cb,
    ]

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=PHASE1_EPOCHS,
        callbacks=phase1_callbacks,
        verbose=1,
    )

    print_bk_stats(model, 'end of phase1')
    res = model.evaluate(dataset_val, verbose=0)
    print(f'  Phase1 end   val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}')

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2：精度最大化
    # 核心改进：完全取消均匀投影器，改用 BetaOnlyBudgetController。
    #
    # baseline 的 Pareto 前沿是靠 beta + 梯度在 ~80k epoch 中自然形成的：
    #   beta 提供压缩动力 → 梯度决定各层保留多少位宽 → 最优 per-layer 分配
    # 均匀投影器破坏了这个机制（强制所有层同比例缩放）。
    #
    # BetaOnlyBudgetController 复制 baseline 的机制：
    #   - EBOPs > target → 自动增大 beta → 梯度自然压低位宽
    #   - EBOPs < target → 自动减小 beta → 允许位宽回升
    #   - HGQ 梯度全程自由决定 per-layer 分配
    #
    # initial_epoch=PHASE1_EPOCHS：epoch 计数连续，checkpoint 文件名不重置
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n[5/5] PHASE 2  Accuracy Maximization  ({PHASE2_EPOCHS} epochs, '
          f'lr={PHASE2_LR:.1e}, beta=auto)')

    # 关键：继承 Phase 1 controller 找到的均衡 beta，而不是硬编码重置。
    # Phase 1 结束时 controller 已经找到维持 EBOPs≈target 的 beta 值，
    # 直接沿用可以避免 Phase 2 开头的 EBOPs 漂移。
    phase1_final_beta = phase1_budget_ctrl.beta_current
    print(f'  Inheriting Phase1 equilibrium beta: {phase1_final_beta:.2e}')

    set_all_beta(model, phase1_final_beta)

    budget_ctrl = BetaOnlyBudgetController(
        target_ebops  = TARGET_EBOPS,
        margin        = 0.05,          # ±5% 紧容忍带（精度最大化阶段需要精确控制）
        beta_init     = phase1_final_beta,  # 从 Phase 1 均衡点出发
        beta_min      = PHASE2_BETA_MIN,
        beta_max      = PHASE2_BETA_MAX,
        adjust_factor = 1.15,          # 更温和：Phase 2 已在目标附近，不需要大幅调整
        ema_alpha     = 0.15,          # 慢 EMA：减少噪声驱动的过调
    )

    early_stop_cb = BudgetAwareEarlyStopping(
        ebops_budget         = EARLYSTOP_BUDGET,
        patience             = EARLYSTOP_PATIENCE,
        min_delta            = 5e-5,
        min_epoch            = PHASE1_EPOCHS + 1000,
        restore_best_weights = True,
    )

    phase2_callbacks = [
        ebops_cb,
        pareto_cb,
        # 核心：用自适应 beta 驱动 HGQ 自然维持 EBOPs，同时允许跨层重分配
        budget_ctrl,
        make_lr_scheduler(PHASE2_LR, PHASE2_LR_CYCLE, PHASE2_LR_MMUL,
                          offset=PHASE1_EPOCHS),
        trace_cb,
        early_stop_cb,
    ]

    # 重新编译（重置动量，避免 phase1 积累的动量干扰 phase2 精细收敛）
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        initial_epoch=PHASE1_EPOCHS,
        epochs=TOTAL_EPOCHS,
        callbacks=phase2_callbacks,
        verbose=1,
    )

    print('\n' + '=' * 65)
    print('Training complete.')
    print(f'Pareto checkpoints : {output_folder}')
    print(f'Training trace     : {os.path.join(output_folder, "training_trace.h5")}')
    print('=' * 65)