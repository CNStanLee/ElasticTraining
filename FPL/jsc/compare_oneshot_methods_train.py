#!/usr/bin/env python3
"""
compare_oneshot_methods_train.py
=================================
对快速对比筛选出的一次性剪枝方法进行**全量训练**对比实验。

流程:
  1. 从 baseline checkpoint 加载模型
  2. 用指定的一次性剪枝方法裁剪到目标 eBOPs
  3. Phase 1 恢复期 + Phase 2 精度最大化 (与 train_run_prune_finetune.py 相同)
  4. 保存 Pareto 前沿检查点和训练轨迹

用法示例:
  # 单个实验
  python compare_oneshot_methods_train.py \
    --prune_method snip --target_ebops 1000

  # 批量运行所有对比实验 (用 run_all_training_comparisons.sh)
  bash run_all_training_comparisons.sh

输出:
  results/oneshot_train_{target_ebops}_{method}/
    ├── epoch=xxx-val_acc=xxx-ebops=xxx-val_loss=xxx.keras
    ├── training_trace.h5
    └── experiment_meta.json
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 保证相对路径基于脚本目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import random
import argparse
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
    BetaOnlyBudgetController,
    _flatten_layers,
    _get_kq_var,
)
from run_one_shot_prune_only import (
    compute_model_ebops,
    bisect_ebops_to_target,
    spectral_quant_prune_to_ebops,
    saliency_prune_to_ebops,
    snows_prune_to_ebops,
    teacher_guided_post_prune_calibration,
    _dense_prunable_layers,
)
from utils.ramanujan_budget_utils import (
    HighBitPruner,
    SensitivityAwarePruner,
)

np.random.seed(42)
random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 默认配置
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_CKPT  = "results/baseline/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras"
BASELINE_EBOPS = 19899
INPUT_H5       = "data/dataset.h5"
BATCH_SIZE     = 33200

# ── Phase 1：恢复期 ─────────────────────────────────────────────────────────
PHASE1_EPOCHS    = 5000
PHASE1_LR        = 2e-3
PHASE1_LR_CYCLE  = 2000
PHASE1_LR_MMUL   = 0.9
PHASE1_BETA_INIT = 5e-7
PHASE1_BETA_MIN  = 1e-8
PHASE1_BETA_MAX  = 2e-4

# ── Phase 2：精度最大化 ──────────────────────────────────────────────────────
PHASE2_EPOCHS    = 10000
PHASE2_LR        = 5e-4
PHASE2_LR_CYCLE  = 800
PHASE2_LR_MMUL   = 0.95
PHASE2_BETA_INIT = 1e-5
PHASE2_BETA_MIN  = 1e-7
PHASE2_BETA_MAX  = 5e-4

# 校准
CALIBRATION_OVERSHOOT = 1.10

# EarlyStop
EARLYSTOP_PATIENCE = 5000


# ══════════════════════════════════════════════════════════════════════════════
# 命令行参数
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Train comparison: one-shot prune + two-phase finetune')
parser.add_argument('--prune_method', type=str, required=True,
                    choices=['uniform', 'sensitivity', 'snip', 'grasp', 'synflow',
                             'spectral_quant', 'snows'],
                    help='One-shot pruning method')
parser.add_argument('--target_ebops', type=float, required=True,
                    help='Target eBOPs budget')
parser.add_argument('--checkpoint', type=str, default=BASELINE_CKPT)
parser.add_argument('--phase1_epochs', type=int, default=PHASE1_EPOCHS)
parser.add_argument('--phase2_epochs', type=int, default=PHASE2_EPOCHS)
parser.add_argument('--output_dir', type=str, default='',
                    help='Output directory (auto-generated if empty)')
parser.add_argument('--input_h5', type=str, default=INPUT_H5)
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('--functional_calibrate', action='store_true', default=False)
parser.add_argument('--functional_passes', type=int, default=2)
parser.add_argument('--calib_overshoot', type=float, default=CALIBRATION_OVERSHOOT)
parser.add_argument('--sample_size', type=int, default=512)

args, _ = parser.parse_known_args()

TARGET_EBOPS    = args.target_ebops
PHASE1_EPOCHS   = args.phase1_epochs
PHASE2_EPOCHS   = args.phase2_epochs
PRUNE_METHOD    = args.prune_method
CALIBRATION_OVERSHOOT = args.calib_overshoot
EARLYSTOP_BUDGET = TARGET_EBOPS * 1.5

# 自动生成输出目录
if not args.output_dir:
    output_folder = f'results/oneshot_train_{int(TARGET_EBOPS)}_{PRUNE_METHOD}'
else:
    output_folder = args.output_dir

device = get_tf_device()
os.makedirs(output_folder, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def set_all_beta(model, beta_value: float):
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
              f'n_dead(<=0.1)={int((arr<=0.1).sum())}/{len(arr)}')


def _forward_update_ebops_no_bn_drift(model, sample_input):
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


def local_compute_model_ebops(model, sample_input) -> float:
    from keras import ops
    _forward_update_ebops_no_bn_drift(model, sample_input)
    total = 0
    for layer in model._flatten_layers():
        if isinstance(layer, QLayerBase) and layer.enable_ebops and layer._ebops is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def correct_pruning_ebops(model, actual_ebops, target_ebops, sample_input,
                          tolerance=0.03, max_iter=20, b_k_min=0.01, b_k_max=8.0):
    snapshots = {}
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

    def apply_scale(s):
        for b_var, b_snap in snapshots.values():
            b_new = np.where(b_snap > 0.1, np.clip(b_snap * s, b_k_min, b_k_max), 0.0)
            b_var.assign(b_new.astype(np.float32))

    def measure_ebops(s):
        apply_scale(s)
        return float(local_compute_model_ebops(model, sample_input))

    if abs(actual_ebops - target_ebops) / max(target_ebops, 1.0) <= tolerance:
        return actual_ebops

    e_at_1 = measure_ebops(1.0)
    if e_at_1 < target_ebops:
        lo, hi = 1.0, 1.0
        for _ in range(20):
            hi *= 2.0
            if measure_ebops(hi) >= target_ebops:
                break
    else:
        lo, hi = 1.0, 1.0
        for _ in range(20):
            lo /= 2.0
            if measure_ebops(lo) <= target_ebops:
                break

    best_s, best_e = lo, measure_ebops(lo)
    for i in range(max_iter):
        mid = (lo + hi) / 2.0
        mid_e = measure_ebops(mid)
        err = abs(mid_e - target_ebops) / target_ebops
        if err < abs(best_e - target_ebops) / target_ebops:
            best_s, best_e = mid, mid_e
        if err <= tolerance:
            break
        if mid_e < target_ebops:
            lo = mid
        else:
            hi = mid

    apply_scale(best_s)
    final_e = local_compute_model_ebops(model, sample_input)
    print(f'  [EBOPs correction] final: scale={best_s:.5f}  '
          f'ebops={final_e:.1f}  target={target_ebops:.1f}')
    return final_e


def make_lr_scheduler(lr_init, cycle, mmul, offset=0):
    fn = cosine_decay_restarts_schedule(lr_init, cycle, t_mul=1.0, m_mul=mmul,
                                        alpha=1e-6, alpha_steps=50)
    def schedule(epoch):
        return fn(max(0, epoch - offset))
    return LearningRateScheduler(schedule)


# ══════════════════════════════════════════════════════════════════════════════
# 剪枝执行
# ══════════════════════════════════════════════════════════════════════════════

def execute_pruning(model, target_ebops, sample_input, method, teacher_model=None):
    """执行一次性剪枝，返回 (used_structured_low_budget, prune_report)。"""
    current_ebops = local_compute_model_ebops(model, sample_input)
    used_structured = False
    report = {'method': method}

    if method == 'uniform':
        pruner = HighBitPruner(target_ebops=target_ebops, pruned_threshold=0.1)
        pruner.prune_to_ebops(model, current_ebops=current_ebops, verbose=True)

    elif method == 'sensitivity':
        pruner = SensitivityAwarePruner(
            target_ebops=target_ebops, pruned_threshold=0.1, b_k_min=0.3)
        pruner.prune_to_ebops(model, current_ebops=current_ebops, verbose=True)

    elif method in ('snip', 'grasp', 'synflow'):
        _, sal_report = saliency_prune_to_ebops(
            model, target_ebops=target_ebops,
            sample_input=sample_input, method=method,
            input_h5=args.input_h5, sample_size=args.sample_size,
            b_floor=0.35, verbose=True,
        )
        report['saliency_report'] = sal_report

    elif method == 'spectral_quant':
        _, used_structured = spectral_quant_prune_to_ebops(
            model, target_ebops=target_ebops,
            sample_input=sample_input,
            min_degree=2, b_floor=0.35,
            low_budget_structured=True,
            low_budget_threshold=900.0,
            min_hidden_width=4,
            near_budget_ratio=1.6,
            verbose=True,
        )

    elif method == 'snows':
        if teacher_model is None:
            raise ValueError("SNOWS requires a teacher model")
        _, snows_report = snows_prune_to_ebops(
            model, teacher_model=teacher_model,
            target_ebops=target_ebops,
            sample_input=sample_input,
            init_method='sensitivity',
            b_floor=0.30, k_step=2, newton_steps=2,
            verbose=True,
        )
        report['snows_report'] = snows_report
        used_structured = bool(snows_report.get('used_structured_low_budget', False))

    return used_structured, report


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

    print('=' * 70)
    print(f'  One-shot Prune + Two-phase Finetune  [Method Comparison]')
    print(f'  Source      : {args.checkpoint}')
    print(f'  Target      : ebops = {TARGET_EBOPS}')
    print(f'  Prune method: {PRUNE_METHOD}')
    print(f'  Func-calib  : {args.functional_calibrate} (passes={args.functional_passes})')
    print(f'  Phase1      : {PHASE1_EPOCHS} epochs  (recovery)')
    print(f'  Phase2      : {PHASE2_EPOCHS} epochs  (acc-max)')
    print(f'  Output      : {output_folder}')
    print('=' * 70)

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/5] Loading dataset...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(args.input_h5, src='openml')
    dataset_train = Dataset(X_train, y_train, args.batch_size, device, shuffle=True)
    dataset_val   = Dataset(X_val,   y_val,   args.batch_size, device)
    _sample_input = tf.constant(X_train[:min(2048, len(X_train))], dtype=tf.float32)

    # ── 2. 加载模型 ──────────────────────────────────────────────────────────
    print(f'\n[2/5] Loading checkpoint: {args.checkpoint}')
    teacher_model = keras.models.load_model(args.checkpoint, compile=False)
    model = keras.models.load_model(args.checkpoint, compile=False)
    model.summary()
    print_bk_stats(model, 'before pruning')
    actual_baseline_ebops = local_compute_model_ebops(model, _sample_input)
    print(f'  Actual baseline EBOPs (measured): {actual_baseline_ebops:.1f}')

    # ── 3. 一次性剪枝 ────────────────────────────────────────────────────────
    print(f'\n[3/5] One-shot pruning ({PRUNE_METHOD}): '
          f'{actual_baseline_ebops:.1f} -> {TARGET_EBOPS}')

    used_structured_low_budget, prune_report = execute_pruning(
        model, TARGET_EBOPS, _sample_input, PRUNE_METHOD,
        teacher_model=teacher_model,
    )
    print_bk_stats(model, 'after pruning')

    post_prune_ebops = local_compute_model_ebops(model, _sample_input)
    print(f'  Post-prune EBOPs (measured): {post_prune_ebops:.1f}  target: {TARGET_EBOPS}')

    near_budget_preserve_case = (
        PRUNE_METHOD == 'spectral_quant'
        and (not used_structured_low_budget)
        and actual_baseline_ebops <= float(TARGET_EBOPS) * 1.6
    )

    # 功能校准 (可选)
    if args.functional_calibrate:
        post_func_ebops = teacher_guided_post_prune_calibration(
            student_model=model, teacher_model=teacher_model,
            sample_input=_sample_input,
            passes=args.functional_passes,
            b_floor=0.35, b_ceiling=6.0, verbose=True,
        )
        print(f'  Post-functional-calib EBOPs: {post_func_ebops:.1f}')

    # 校准目标
    calib_target = TARGET_EBOPS * CALIBRATION_OVERSHOOT
    print(f'  Calibrating to {calib_target:.1f} '
          f'(target={TARGET_EBOPS:.1f} × overshoot={CALIBRATION_OVERSHOOT})')

    if PRUNE_METHOD == 'spectral_quant':
        post_prune_ebops = bisect_ebops_to_target(
            model, target_ebops=calib_target, sample_input=_sample_input,
            tolerance=0.03, max_iter=24,
            b_k_min=0.20 if near_budget_preserve_case else 0.35,
            allow_connection_kill=(not used_structured_low_budget) and (not near_budget_preserve_case),
        )
    else:
        post_prune_ebops = bisect_ebops_to_target(
            model, target_ebops=calib_target, sample_input=_sample_input,
            tolerance=0.05, max_iter=30,
        )
    print_bk_stats(model, 'after correction')

    # 剪枝后快速评估
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )
    res = model.evaluate(dataset_val, verbose=0)
    print(f'  Post-pruning  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}')

    # 再验证
    post_compile_ebops = local_compute_model_ebops(model, _sample_input)
    if abs(post_compile_ebops - calib_target) / max(calib_target, 1.0) > 0.05:
        print(f'  [WARN] Post-compile eBOPs drifted: {post_compile_ebops:.1f}. Re-calibrating...')
        if PRUNE_METHOD == 'spectral_quant':
            post_compile_ebops = bisect_ebops_to_target(
                model, target_ebops=calib_target, sample_input=_sample_input,
                tolerance=0.03, max_iter=24,
                b_k_min=0.20 if near_budget_preserve_case else 0.35,
                allow_connection_kill=(not used_structured_low_budget) and (not near_budget_preserve_case),
            )
        else:
            post_compile_ebops = bisect_ebops_to_target(
                model, target_ebops=calib_target, sample_input=_sample_input,
                tolerance=0.05, max_iter=30,
            )
    else:
        print(f'  Post-compile EBOPs verified: {post_compile_ebops:.1f}  (OK)')

    # ── Callbacks ────────────────────────────────────────────────────────────
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
            target_ebops=TARGET_EBOPS, b_k_min=0.25, b_k_max=8.0,
            pruned_threshold=0.1, start_epoch=0, alpha_gamma=0.5,
            alpha_min=0.80, alpha_max=1.25, ema_alpha=0.3,
            project_activation=False, log_scale=False,
        )
        print('  [Low-budget] Enable EBOPsConstantProjector.')

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1：恢复期
    # ══════════════════════════════════════════════════════════════════════════
    phase1_lr_use = PHASE1_LR
    phase2_lr_use = PHASE2_LR
    if use_const_projector:
        phase1_lr_use = min(PHASE1_LR, 3e-4)
        phase2_lr_use = min(PHASE2_LR, 1.5e-4)
    elif near_budget_preserve_case:
        phase1_lr_use = min(PHASE1_LR, 1e-3)
        phase2_lr_use = min(PHASE2_LR, 2e-4)

    print(f'\n[4/5] PHASE 1  Recovery  ({PHASE1_EPOCHS} epochs, lr={phase1_lr_use:.1e})')

    if use_const_projector:
        set_all_beta(model, 0.0)
        phase1_budget_ctrl = None
    else:
        set_all_beta(model, PHASE1_BETA_INIT)
        phase1_budget_ctrl = BetaOnlyBudgetController(
            target_ebops=TARGET_EBOPS, margin=0.20,
            beta_init=PHASE1_BETA_INIT, beta_min=PHASE1_BETA_MIN, beta_max=PHASE1_BETA_MAX,
            adjust_factor=1.15, ema_alpha=0.25, warmup_epochs=100,
            max_change_ratio=1.5, init_ebops=TARGET_EBOPS,
        )

    phase1_callbacks = [
        ebops_cb, pareto_cb,
        *([const_projector] if const_projector is not None else []),
        make_lr_scheduler(phase1_lr_use, PHASE1_LR_CYCLE, PHASE1_LR_MMUL, offset=0),
        trace_cb,
    ]
    if phase1_budget_ctrl is not None:
        phase1_callbacks.insert(2, phase1_budget_ctrl)

    model.fit(
        dataset_train, validation_data=dataset_val,
        epochs=PHASE1_EPOCHS, callbacks=phase1_callbacks, verbose=1,
    )

    print_bk_stats(model, 'end of phase1')
    res = model.evaluate(dataset_val, verbose=0)
    print(f'  Phase1 end   val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}')

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2：精度最大化
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n[5/5] PHASE 2  Accuracy Maximization  ({PHASE2_EPOCHS} epochs, lr={phase2_lr_use:.1e})')

    if use_const_projector:
        set_all_beta(model, 0.0)
        budget_ctrl = None
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
            target_ebops=TARGET_EBOPS, margin=phase2_margin,
            beta_init=phase1_final_beta, beta_min=PHASE2_BETA_MIN, beta_max=phase2_beta_max_use,
            adjust_factor=phase2_adjust_factor, ema_alpha=phase2_ema,
            warmup_epochs=0, max_change_ratio=1.5, init_ebops=TARGET_EBOPS,
        )

    early_stop_cb = BudgetAwareEarlyStopping(
        ebops_budget=EARLYSTOP_BUDGET, patience=EARLYSTOP_PATIENCE,
        min_delta=5e-5, min_epoch=PHASE1_EPOCHS + 1000, restore_best_weights=True,
    )

    phase2_callbacks = [
        ebops_cb, pareto_cb,
        *([const_projector] if const_projector is not None else []),
        make_lr_scheduler(phase2_lr_use, PHASE2_LR_CYCLE, PHASE2_LR_MMUL, offset=PHASE1_EPOCHS),
        trace_cb, early_stop_cb,
    ]
    if budget_ctrl is not None:
        phase2_callbacks.insert(2, budget_ctrl)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=phase2_lr_use),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    model.fit(
        dataset_train, validation_data=dataset_val,
        initial_epoch=PHASE1_EPOCHS, epochs=TOTAL_EPOCHS,
        callbacks=phase2_callbacks, verbose=1,
    )

    # ── 保存实验元信息 ───────────────────────────────────────────────────────
    meta = {
        'prune_method': PRUNE_METHOD,
        'target_ebops': float(TARGET_EBOPS),
        'checkpoint': args.checkpoint,
        'phase1_epochs': PHASE1_EPOCHS,
        'phase2_epochs': PHASE2_EPOCHS,
        'total_epochs': TOTAL_EPOCHS,
        'calibration_overshoot': CALIBRATION_OVERSHOOT,
        'used_structured_low_budget': used_structured_low_budget,
        'functional_calibrate': args.functional_calibrate,
        'post_prune_ebops': post_prune_ebops,
        'post_prune_val_acc': float(res[1]) if 'res' in dir() else 0.0,
    }
    if 'saliency_report' in prune_report:
        meta['saliency_report'] = prune_report['saliency_report']
    if 'snows_report' in prune_report:
        meta['snows_report'] = prune_report['snows_report']

    meta_path = Path(output_folder) / 'experiment_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print('\n' + '=' * 70)
    print('Training complete.')
    print(f'  Prune method     : {PRUNE_METHOD}')
    print(f'  Target eBOPs     : {TARGET_EBOPS}')
    print(f'  Pareto checkpoints: {output_folder}')
    print(f'  Training trace    : {os.path.join(output_folder, "training_trace.h5")}')
    print(f'  Experiment meta   : {meta_path}')
    print('=' * 70)
