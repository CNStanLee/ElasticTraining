import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import argparse
import re
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from data.data import get_data
import model.model  # noqa: F401  # 注册自定义层
from hgq.layers import QLayerBase
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
from run_one_shot_prune_only import (
    spectral_quant_prune_to_ebops,
    bisect_ebops_to_target,
    teacher_guided_post_prune_calibration,
)

np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

# baseline epoch=2236 最高精度检查点（val_acc=0.770，ebops=23589）
# BASELINE_CKPT  = 'results/baseline/epoch=2236-val_acc=0.770-ebops=23589-val_loss=0.641.keras'
# BASELINE_EBOPS = 23589      # 从文件名读，作为剪枝前参考值
BASELINE_CKPT = "results/baseline/epoch=3699-val_acc=0.770-ebops=23293-val_loss=0.640.keras"
BASELINE_EBOPS = 23293

TARGET_EBOPS = 1000

# ── 剪枝方式 ──────────────────────────────────────────────────────────────────
# 'uniform'        : 旧方案 HighBitPruner (均匀缩放，有精度损失)
# 'sensitivity'    : 按层敏感度分配预算
# 'spectral_quant' : 谱/拓扑友好剪枝（低预算会自动走结构化子网络）
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
PHASE2_EPOCHS    = 10000     # 增加 epoch：给 HGQ 足够时间找到最优分配
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
                    choices=['uniform', 'sensitivity', 'spectral_quant'])
parser.add_argument('--auto_warm_start', action='store_true', default=True,
                    help='Auto pick a near-budget warm-start checkpoint for low-budget training')
parser.add_argument('--no_auto_warm_start', action='store_true',
                    help='Disable near-budget warm-start checkpoint selection')
parser.add_argument('--warm_start_dir', type=str, default='model_to_analyse/gradual_training',
                    help='Directory used by auto warm-start checkpoint selection')
parser.add_argument('--warm_start_max_ratio', type=float, default=2.0,
                    help='Only consider warm-start checkpoints with ebops <= target * ratio')
parser.add_argument('--functional_calibrate', action='store_true', default=False,
                    help='Enable teacher-guided functional calibration after pruning (default: disabled)')
parser.add_argument('--no_functional_calibrate', action='store_true',
                    help='Disable teacher-guided functional calibration')
parser.add_argument('--functional_passes', type=int, default=2,
                    help='Number of teacher-guided calibration passes after pruning')
args, _ = parser.parse_known_args()

TARGET_EBOPS    = args.target_ebops
PHASE1_EPOCHS   = args.phase1_epochs
PHASE2_EPOCHS   = args.phase2_epochs
BASELINE_CKPT   = args.checkpoint
PRUNE_METHOD    = args.prune_method
EARLYSTOP_BUDGET = TARGET_EBOPS * 1.5
if args.no_auto_warm_start:
    args.auto_warm_start = False
if args.no_functional_calibrate:
    args.functional_calibrate = False


def _extract_ebops_from_name(name: str):
    m = re.search(r'ebops=(\d+)', name)
    return int(m.group(1)) if m else None


def _choose_warm_start_ckpt(
    user_ckpt: str,
    target_ebops: float,
    enabled: bool,
    warm_dir: str,
    max_ratio: float,
):
    if not enabled:
        return user_ckpt, None

    user_name = Path(user_ckpt).name
    user_ebops = _extract_ebops_from_name(user_name)
    if user_ebops is not None and user_ebops <= target_ebops * max_ratio:
        return user_ckpt, None

    d = Path(warm_dir)
    if not d.exists():
        return user_ckpt, None

    cands = []
    for p in d.glob('*.keras'):
        e = _extract_ebops_from_name(p.name)
        if e is None:
            continue
        if e < target_ebops:
            continue
        if e > target_ebops * max_ratio:
            continue
        cands.append((abs(e - target_ebops), e, str(p)))

    if not cands:
        return user_ckpt, None

    cands.sort(key=lambda x: (x[0], x[1]))
    chosen = cands[0][2]
    msg = f'auto warm-start: {user_ckpt} -> {chosen}'
    return chosen, msg


BASELINE_CKPT, warm_start_msg = _choose_warm_start_ckpt(
    BASELINE_CKPT,
    TARGET_EBOPS,
    enabled=(args.auto_warm_start and PRUNE_METHOD == 'spectral_quant' and TARGET_EBOPS <= 800),
    warm_dir=args.warm_start_dir,
    max_ratio=args.warm_start_max_ratio,
)

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


def _forward_update_ebops_no_bn_drift(model, sample_input):
    """刷新 layer._ebops，同时避免 BatchNorm moving stats 被更新。"""
    bn_layers = []
    old_momentum = []
    for layer in _flatten_layers(model):
        if isinstance(layer, keras.layers.BatchNormalization):
            bn_layers.append(layer)
            old_momentum.append(float(layer.momentum))
            layer.momentum = 1.0
    try:
        model(sample_input, training=True)
    finally:
        for layer, m in zip(bn_layers, old_momentum):
            layer.momentum = m


def compute_model_ebops(model, sample_input) -> float:
    """通过一次前向传播实测模型当前 EBOPs（不依赖 checkpoint 文件名）。

    注意：这里会以 training=True 刷新 _ebops，但临时冻结 BN 动量，避免污染统计量。
    """
    from keras import ops
    _forward_update_ebops_no_bn_drift(model, sample_input)
    total = 0
    for layer in model._flatten_layers():
        if isinstance(layer, QLayerBase) and layer.enable_ebops and layer._ebops is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def correct_pruning_ebops(
    model,
    actual_ebops: float,
    target_ebops: float,
    sample_input,
    tolerance: float = 0.03,
    max_iter: int = 20,
    b_k_min: float = 0.01,
    b_k_max: float = 8.0,
) -> float:
    """用二分搜索将剪枝后的 kq.b 调整到使 EBOPs 命中目标。

    问题背景
    --------
    剪枝后所有 kq.b 可能被 b_k_min 钉平（典型值全部 = 0.3）。此时普通
    比例迭代会陷入"scale 上→脱底→爆涨 / scale 下→重新钉底→骤降"的
    死循环（无穷振荡）。

    二分法策略
    ----------
    1. 快照保存剪枝后的 kq.b
    2. 确定 scale 的上下界（低 → EBOPs < target，高 → EBOPs > target）
    3. 每次 mid = (lo + hi) / 2，apply saved * mid，实测 EBOPs，收窄区间
    4. 收敛后一次性写入
    """
    # ── 快照 ────────────────────────────────────────────────────────────────
    snapshots = {}  # layer_name → (b_var, b_arr_snapshot)
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None:
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is None:
            continue
        snapshots[id(layer)] = (b_var, b_var.numpy().copy())

    def apply_scale(s: float):
        for b_var, b_snap in snapshots.values():
            b_new = np.where(
                b_snap > 0.1,
                np.clip(b_snap * s, b_k_min, b_k_max),
                0.0,
            )
            b_var.assign(b_new.astype(np.float32))

    def measure_ebops(s: float) -> float:
        apply_scale(s)
        return float(compute_model_ebops(model, sample_input))

    # ── 快速检查：当前是否已经满足 ─────────────────────────────────────────
    if abs(actual_ebops - target_ebops) / max(target_ebops, 1.0) <= tolerance:
        print(f'  [EBOPs correction] already within tolerance: '
              f'actual={actual_ebops:.1f}  target={target_ebops:.1f}')
        return actual_ebops

    # ── 确定二分上下界 ────────────────────────────────────────────────────
    # 从 scale=1.0（当前快照）开始寻找 lo/hi
    e_at_1 = measure_ebops(1.0)
    if e_at_1 < target_ebops:
        # 需要增大 scale
        lo, hi = 1.0, 1.0
        hi_e = e_at_1
        for _ in range(20):
            hi *= 2.0
            hi_e = measure_ebops(hi)
            if hi_e >= target_ebops:
                break
        lo_e = e_at_1
    else:
        # 需要减小 scale
        lo, hi = 1.0, 1.0
        lo_e = e_at_1
        for _ in range(20):
            lo /= 2.0
            lo_e = measure_ebops(lo)
            if lo_e <= target_ebops:
                break
        hi_e = e_at_1

    print(f'  [EBOPs correction] bisect range: '
          f'scale=[{lo:.4f},{hi:.4f}]  ebops=[{lo_e:.0f},{hi_e:.0f}]  target={target_ebops:.0f}')

    # ── 二分搜索 ───────────────────────────────────────────────────────────
    best_s, best_e = lo, lo_e
    for i in range(max_iter):
        mid = (lo + hi) / 2.0
        mid_e = measure_ebops(mid)
        err = abs(mid_e - target_ebops) / target_ebops
        print(f'  [EBOPs correction] iter {i+1:2d}: '
              f'scale={mid:.5f}  ebops={mid_e:.1f}  err={err*100:.1f}%')
        if err < abs(best_e - target_ebops) / target_ebops:
            best_s, best_e = mid, mid_e
        if err <= tolerance:
            break
        if mid_e < target_ebops:
            lo = mid
        else:
            hi = mid

    # ── 写入最优 scale ─────────────────────────────────────────────────────
    apply_scale(best_s)
    final_e = measure_ebops(best_s)
    print(f'  [EBOPs correction] final: scale={best_s:.5f}  '
          f'ebops={final_e:.1f}  target={target_ebops:.1f}  '
          f'err={abs(final_e-target_ebops)/target_ebops*100:.1f}%')
    return final_e


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
    if warm_start_msg:
        print(f'  Warm-start  : {warm_start_msg}')
    print(f'  Target      : ebops = {TARGET_EBOPS}')
    print(f'  Prune method: {PRUNE_METHOD}')
    print(f'  Func-calib  : {args.functional_calibrate} (passes={args.functional_passes})')
    print(f'  Phase1      : {PHASE1_EPOCHS} epochs  (recovery,  beta auto [{PHASE1_BETA_INIT:.1e}~{PHASE1_BETA_MAX:.1e}])')
    print(f'  Phase2      : {PHASE2_EPOCHS} epochs  (acc-max,   beta auto [{PHASE2_BETA_INIT:.1e}~{PHASE2_BETA_MAX:.1e}])')
    print(f'  Output      : {output_folder}')
    print('=' * 65)

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/5] Loading dataset...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src='openml')
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val   = Dataset(X_val,   y_val,   batch_size, device)

    # 用于实测 EBOPs 的样本（训练模式，512 条与 FreeEBOPs 行为一致）
    _sample_input = tf.constant(X_val[:512], dtype=tf.float32)

    # ── 2. 加载最优权重 ──────────────────────────────────────────────────────
    print(f'\n[2/5] Loading checkpoint: {BASELINE_CKPT}')
    teacher_model = keras.models.load_model(BASELINE_CKPT, compile=False)
    model = keras.models.load_model(BASELINE_CKPT, compile=False)
    model.summary()
    print_bk_stats(model, 'before pruning')

    # 实测加载后的 EBOPs（不依赖文件名中的硬编码值）
    actual_baseline_ebops = compute_model_ebops(model, _sample_input)
    print(f'  Actual baseline EBOPs (measured): {actual_baseline_ebops:.1f}  '
          f'(hardcoded ref: {BASELINE_EBOPS})')

    # ── 3. 一次性位宽剪枝 ────────────────────────────────────────────────────
    print(f'\n[3/5] One-shot bit-width pruning ({PRUNE_METHOD}): '
          f'{actual_baseline_ebops:.1f} -> {TARGET_EBOPS}')
    used_structured_low_budget = False
    if PRUNE_METHOD == 'sensitivity':
        pruner = SensitivityAwarePruner(target_ebops=TARGET_EBOPS, pruned_threshold=0.1, b_k_min=0.3)
        pruner.prune_to_ebops(model, current_ebops=actual_baseline_ebops, verbose=True)
    elif PRUNE_METHOD == 'uniform':
        pruner = HighBitPruner(target_ebops=TARGET_EBOPS, pruned_threshold=0.1)
        pruner.prune_to_ebops(model, current_ebops=actual_baseline_ebops, verbose=True)
    else:
        _, used_structured_low_budget = spectral_quant_prune_to_ebops(
            model,
            target_ebops=TARGET_EBOPS,
            sample_input=_sample_input,
            min_degree=2,
            b_floor=0.35,
            low_budget_structured=True,
            low_budget_threshold=900.0,
            min_hidden_width=4,
            near_budget_ratio=1.6,
            verbose=True,
        )
    print_bk_stats(model, 'after pruning')

    # 实测剪枝后 EBOPs，迭代修正到目标 ±3%
    post_prune_ebops = compute_model_ebops(model, _sample_input)
    print(f'  Post-prune EBOPs (measured): {post_prune_ebops:.1f}  target: {TARGET_EBOPS}')
    near_budget_preserve_case = (
        PRUNE_METHOD == 'spectral_quant'
        and (not used_structured_low_budget)
        and actual_baseline_ebops <= float(TARGET_EBOPS) * 1.6
    )
    if args.functional_calibrate:
        post_func_ebops = teacher_guided_post_prune_calibration(
            student_model=model,
            teacher_model=teacher_model,
            sample_input=_sample_input,
            passes=args.functional_passes,
            b_floor=0.35,
            b_ceiling=6.0,
            verbose=True,
        )
        print(f'  Post-functional-calib EBOPs (measured): {post_func_ebops:.1f}')

    if PRUNE_METHOD == 'spectral_quant':
        post_prune_ebops = bisect_ebops_to_target(
            model,
            target_ebops=TARGET_EBOPS,
            sample_input=_sample_input,
            tolerance=0.03,
            max_iter=24,
            b_k_min=0.20 if near_budget_preserve_case else 0.35,
            allow_connection_kill=(not used_structured_low_budget) and (not near_budget_preserve_case),
        )
    else:
        post_prune_ebops = correct_pruning_ebops(
            model, post_prune_ebops, TARGET_EBOPS, _sample_input,
            tolerance=0.03, max_iter=20,
        )
    print_bk_stats(model, 'after correction')

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
    use_const_projector = (
        PRUNE_METHOD == 'spectral_quant'
        and TARGET_EBOPS <= 800
        and used_structured_low_budget
    )
    const_projector = None
    if use_const_projector:
        const_projector = EBOPsConstantProjector(
            target_ebops=TARGET_EBOPS,
            b_k_min=0.25,
            b_k_max=8.0,
            pruned_threshold=0.1,
            start_epoch=0,
            alpha_gamma=0.5,
            alpha_min=0.80,
            alpha_max=1.25,
            ema_alpha=0.3,
            project_activation=False,
            log_scale=False,
        )
        print('  [Low-budget] Enable EBOPsConstantProjector to prevent EBOPs collapse.')

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
    phase1_lr_use = PHASE1_LR
    phase2_lr_use = PHASE2_LR
    if use_const_projector:
        # 低预算稳定模式：避免高 LR + beta 压缩把模型直接拉到随机猜测
        phase1_lr_use = min(PHASE1_LR, 3e-4)
        phase2_lr_use = min(PHASE2_LR, 1.5e-4)
        print('\n  [Low-budget stable mode] disable beta compression, use projector-only budget control.')
    elif near_budget_preserve_case:
        # 近预算微调：主要是小步恢复，不需要激进学习率
        phase1_lr_use = min(PHASE1_LR, 1e-3)
        phase2_lr_use = min(PHASE2_LR, 2e-4)
        print('\n  [Near-budget mode] use gentler LR to preserve pretrained accuracy.')

    print(f'\n[4/5] PHASE 1  Recovery  ({PHASE1_EPOCHS} epochs, '
          f'lr={phase1_lr_use:.1e}, beta={"fixed 0" if use_const_projector else f"auto [{PHASE1_BETA_INIT:.1e}~{PHASE1_BETA_MAX:.1e}]"} )')

    if use_const_projector:
        set_all_beta(model, 0.0)
        phase1_budget_ctrl = None
    else:
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
        *( [const_projector] if const_projector is not None else [] ),
        make_lr_scheduler(phase1_lr_use, PHASE1_LR_CYCLE, PHASE1_LR_MMUL, offset=0),
        trace_cb,
    ]
    if phase1_budget_ctrl is not None:
        # 双向 budget controller：EBOPs 过低时减 beta，过高时加 beta
        phase1_callbacks.insert(2, phase1_budget_ctrl)

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
          f'lr={phase2_lr_use:.1e}, beta={"fixed 0" if use_const_projector else "auto"})')

    # 关键：继承 Phase 1 controller 找到的均衡 beta，而不是硬编码重置。
    # Phase 1 结束时 controller 已经找到维持 EBOPs≈target 的 beta 值，
    # 直接沿用可以避免 Phase 2 开头的 EBOPs 漂移。
    if use_const_projector:
        phase1_final_beta = 0.0
        set_all_beta(model, 0.0)
        budget_ctrl = None
        print('  Inheriting Phase1 beta: 0.00e+00 (projector-only mode)')
    else:
        phase1_final_beta = phase1_budget_ctrl.beta_current
        print(f'  Inheriting Phase1 equilibrium beta: {phase1_final_beta:.2e}')
        set_all_beta(model, phase1_final_beta)
        phase2_margin = 0.05
        phase2_adjust_factor = 1.15
        phase2_ema = 0.15
        phase2_beta_max_use = PHASE2_BETA_MAX
        if near_budget_preserve_case:
            phase2_margin = 0.08
            phase2_adjust_factor = 1.08
            phase2_ema = 0.20
            phase2_beta_max_use = min(PHASE2_BETA_MAX, 2e-4)
        budget_ctrl = BetaOnlyBudgetController(
            target_ebops  = TARGET_EBOPS,
            margin        = phase2_margin,  # 紧容忍带（近预算时更平滑）
            beta_init     = phase1_final_beta,  # 从 Phase 1 均衡点出发
            beta_min      = PHASE2_BETA_MIN,
            beta_max      = phase2_beta_max_use,
            adjust_factor = phase2_adjust_factor,
            ema_alpha     = phase2_ema,
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
        *( [const_projector] if const_projector is not None else [] ),
        make_lr_scheduler(phase2_lr_use, PHASE2_LR_CYCLE, PHASE2_LR_MMUL,
                          offset=PHASE1_EPOCHS),
        trace_cb,
        early_stop_cb,
    ]
    if budget_ctrl is not None:
        # 核心：用自适应 beta 驱动 HGQ 自然维持 EBOPs，同时允许跨层重分配
        phase2_callbacks.insert(2, budget_ctrl)

    # 重新编译（重置动量，避免 phase1 积累的动量干扰 phase2 精细收敛）
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=phase2_lr_use),
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
