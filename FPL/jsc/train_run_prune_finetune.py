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
    _flatten_layers,
    _get_kq_var,
)

np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

# baseline epoch=2236 最高精度检查点（val_acc=0.770，ebops=23589）
BASELINE_CKPT  = 'results/baseline/epoch=2236-val_acc=0.770-ebops=23589-val_loss=0.641.keras'
BASELINE_EBOPS = 23589      # 从文件名读，作为剪枝前参考值

TARGET_EBOPS = 3000

input_folder = 'data/dataset.h5'
batch_size   = 33200

# ── Phase 1：恢复期 ─────────────────────────────────────────────────────────
# 一次性位宽剪枝后，alpha≈0.127（切了87%位宽），权重与量化工作点严重失配。
# 需要在小 beta + 高 LR 下让权重重新适应低精度量化点。
# 投影器宽松（±40%），优先让权重稳定，不强迫精确卡在 target_ebops。
PHASE1_EPOCHS    = 5000
PHASE1_LR        = 2e-3      # 高 LR：加快重适应
PHASE1_LR_CYCLE  = 2000
PHASE1_LR_MMUL   = 0.9
PHASE1_BETA      = 3e-5      # 小但非零：维持量化侧正则，不过度压缩
                              # (baseline beta_max=1e-3，此处只有 3% 的压缩力)
PHASE1_PROJ_AMIN = 0.75      # 宽松：允许 ±40% 波动
PHASE1_PROJ_AMAX = 1.25
PHASE1_PROJ_GAMMA= 0.25      # 极保守阻尼

# ── Phase 2：精度最大化 ──────────────────────────────────────────────────────
# 核心修复：不用 beta=0 + 严格投影器。
# 原因：均匀投影锁死各层相对位宽比例，HGQ 无法按重要性跨层重分配位宽，
#      而 baseline 0.759 正是靠 ~80k epoch 训练出的自适应分配实现的。
#
# 新方案：用小 beta（~1e-5）自然维持预算。
#   依据：baseline 在 ebops≈3000 处的 beta 约 8e-6~1e-5（从 log 调度反推）。
#   在此 beta 下，HGQ 自发将总 EBOPs 稳定在 ~3000，同时自由重分配各层位宽。
#   投影器降为宽松安全网（±50%），只防极端漂移，不干预跨层分配。
PHASE2_EPOCHS    = 20000
PHASE2_LR        = 5e-4
PHASE2_LR_CYCLE  = 800
PHASE2_LR_MMUL   = 0.95
PHASE2_BETA      = 1e-5      # baseline 在 ebops≈3000 的自然均衡 beta
PHASE2_PROJ_AMIN = 0.60      # 宽松安全网：只防止 EBOPs 漂移超 ±50%
PHASE2_PROJ_AMAX = 1.40
PHASE2_PROJ_GAMMA= 0.15      # 极低阻尼：几乎不干预，只截断极端情况

EARLYSTOP_PATIENCE = 3000
EARLYSTOP_BUDGET   = TARGET_EBOPS * 1.5   # 50% 弹性窗口（配合宽松安全网）


# ── 命令行参数覆盖 ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='One-shot prune + two-phase finetune')
parser.add_argument('--target_ebops',  type=float, default=TARGET_EBOPS)
parser.add_argument('--phase1_epochs', type=int,   default=PHASE1_EPOCHS)
parser.add_argument('--phase2_epochs', type=int,   default=PHASE2_EPOCHS)
parser.add_argument('--checkpoint',    type=str,   default=BASELINE_CKPT)
args, _ = parser.parse_known_args()

TARGET_EBOPS    = args.target_ebops
PHASE1_EPOCHS   = args.phase1_epochs
PHASE2_EPOCHS   = args.phase2_epochs
BASELINE_CKPT   = args.checkpoint
EARLYSTOP_BUDGET = TARGET_EBOPS * 1.25

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
    print(f'  Phase1      : {PHASE1_EPOCHS} epochs  (recovery,  beta={PHASE1_BETA:.1e})')
    print(f'  Phase2      : {PHASE2_EPOCHS} epochs  (acc-max,   beta=0)')
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
    # HighBitPruner 按比例缩放 kq.b：alpha = target / current ≈ 0.127
    # 幅度大，权重与量化工作点严重失配 → Phase1 恢复期重适应
    print(f'\n[3/5] One-shot bit-width pruning: {BASELINE_EBOPS} -> {TARGET_EBOPS}')
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
    # 小 beta + 高 LR + 宽松投影：让权重重新适应低精度量化工作点
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n[4/5] PHASE 1  Recovery  ({PHASE1_EPOCHS} epochs, '
          f'lr={PHASE1_LR:.1e}, beta={PHASE1_BETA:.1e})')

    set_all_beta(model, PHASE1_BETA)   # 手动初始化 beta（ConstantBetaCallback 在每 epoch 维持）

    phase1_callbacks = [
        ebops_cb,
        pareto_cb,
        EBOPsConstantProjector(
            target_ebops      = TARGET_EBOPS,
            b_k_min           = 0.2,
            b_k_max           = 8.0,
            pruned_threshold  = 0.1,
            start_epoch       = 0,
            alpha_gamma       = PHASE1_PROJ_GAMMA,
            alpha_min         = PHASE1_PROJ_AMIN,
            alpha_max         = PHASE1_PROJ_AMAX,
            ema_alpha         = 0.2,
            project_activation= True,
            log_scale         = False,
        ),
        ConstantBetaCallback(PHASE1_BETA),   # on_epoch_begin 维持小 beta
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
    # beta=0，严格投影维持预算，专注提升 val_accuracy
    # initial_epoch=PHASE1_EPOCHS：epoch 计数连续，checkpoint 文件名不重置
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n[5/5] PHASE 2  Accuracy Maximization  ({PHASE2_EPOCHS} epochs, '
          f'lr={PHASE2_LR:.1e}, beta={PHASE2_BETA:.1e} [free bit redistribution])')

    # 手动设 beta 为均衡值，ConstantBetaCallback 在 epoch_begin 维持
    set_all_beta(model, PHASE2_BETA)

    early_stop_cb = BudgetAwareEarlyStopping(
        ebops_budget         = EARLYSTOP_BUDGET,
        patience             = EARLYSTOP_PATIENCE,
        min_delta            = 5e-5,
        min_epoch            = PHASE1_EPOCHS + 500,
        restore_best_weights = True,
    )

    phase2_callbacks = [
        ebops_cb,
        pareto_cb,
        # 宽松安全网投影器：只截断 EBOPs 极端漂移（±50%），不干预跨层位宽分配
        EBOPsConstantProjector(
            target_ebops      = TARGET_EBOPS,
            b_k_min           = 0.2,
            b_k_max           = 8.0,
            pruned_threshold  = 0.1,
            start_epoch       = PHASE1_EPOCHS,
            alpha_gamma       = PHASE2_PROJ_GAMMA,
            alpha_min         = PHASE2_PROJ_AMIN,
            alpha_max         = PHASE2_PROJ_AMAX,
            ema_alpha         = 0.15,   # 慢热 EMA：投影器更少响应短期波动
            project_activation= True,
            log_scale         = False,
        ),
        # 关键：用均衡 beta 驱动 HGQ 自然维持 EBOPs，同时允许跨层重分配
        ConstantBetaCallback(PHASE2_BETA),
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
