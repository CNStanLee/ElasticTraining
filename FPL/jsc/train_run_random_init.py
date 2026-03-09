"""
随机初始化 → BetaOnlyBudgetController 训练验证脚本
=====================================================
假说：在网络容量足够时（如 2000 EBOPs），随机初始化的网络也能训练到
与「预训练→剪枝→微调」相近的性能。

流程
----
1. 从零构建 HGQ 模型（随机初始化，init_bw_k=3）
2. 不做任何剪枝，直接用 BetaOnlyBudgetController 训练
   - beta 自动调节：EBOPs 高于目标时增大 beta 压缩，低于目标时减小 beta 释放
   - HGQ 梯度全程自由决定 per-layer 位宽分配
3. 使用与 prune_finetune 相同的 lr schedule、EarlyStopping、ParetoFront

与 train_run_prune_finetune.py 的区别
--------------------------------------
- 无 baseline checkpoint、无剪枝步骤
- 无 Phase 1/Phase 2 分割（只有一个连续训练阶段）
- beta_init 较高（需要从全精度 ~19900 EBOPs 压到目标，初始需要较强压缩力）
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
import model.model as model_module  # noqa: F401  注册自定义层
from model.model import get_model_hgq
from hgq.layers import QLayerBase
from hgq.utils.sugar import Dataset, FreeEBOPs, ParetoFront
from keras.callbacks import LearningRateScheduler
from utils.train_utils import cosine_decay_restarts_schedule, TrainingTraceToH5, BudgetAwareEarlyStopping
from utils.tf_device import get_tf_device
from utils.ramanujan_budget_utils import (
    BetaOnlyBudgetController,
    _flatten_layers,
    _get_kq_var,
)

np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

TARGET_EBOPS = 3162

input_folder = 'data/dataset.h5'
batch_size   = 33200

# ── 训练参数 ──────────────────────────────────────────────────────────────────
TOTAL_EPOCHS = 50000
LR_INIT      = 2e-3
LR_CYCLE     = 2000
LR_MMUL      = 0.95

# ── Beta 控制参数 ──────────────────────────────────────────────────────────────
# 随机初始化模型的 EBOPs 约 19900（全 3-bit），需要 beta 压缩到目标。
# 初始 beta 需要比 prune_finetune 高，因为要从 ~19900 降到 ~2000（10x 压缩）。
BETA_INIT = 1e-4
BETA_MIN  = 1e-8
BETA_MAX  = 1e-3

EARLYSTOP_PATIENCE = 5000

# ── 命令行参数覆盖 ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Random-init HGQ training with beta budget control')
parser.add_argument('--target_ebops',  type=float, default=TARGET_EBOPS)
parser.add_argument('--total_epochs',  type=int,   default=TOTAL_EPOCHS)
parser.add_argument('--lr',            type=float, default=LR_INIT)
parser.add_argument('--beta_init',     type=float, default=BETA_INIT)
parser.add_argument('--init_bw_k',     type=int,   default=3,
                    help='Initial kernel bitwidth for HGQ model (default: 3)')
args, _ = parser.parse_known_args()

TARGET_EBOPS = args.target_ebops
TOTAL_EPOCHS = args.total_epochs
LR_INIT      = args.lr
BETA_INIT    = args.beta_init
EARLYSTOP_BUDGET = TARGET_EBOPS * 1.5

output_folder = f'results/random_init_{int(TARGET_EBOPS)}/'
device = get_tf_device()
os.makedirs(output_folder, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def set_all_beta(mdl, beta_value: float):
    bv = tf.constant(beta_value, dtype=tf.float32)
    for layer in _flatten_layers(mdl):
        if hasattr(layer, '_beta'):
            try:
                layer._beta.assign(tf.ones_like(layer._beta) * bv)
            except Exception:
                pass


def print_bk_stats(mdl, label=''):
    all_b = []
    for layer in _flatten_layers(mdl):
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


def compute_model_ebops(mdl, sample_input) -> float:
    from keras import ops
    # 前向传播刷新 _ebops，冻结 BN 动量
    bn_layers = []
    old_momentum = []
    for layer in _flatten_layers(mdl):
        if isinstance(layer, keras.layers.BatchNormalization):
            bn_layers.append(layer)
            old_momentum.append(float(layer.momentum))
            layer.momentum = 1.0
    try:
        mdl(sample_input, training=True)
    finally:
        for layer, m in zip(bn_layers, old_momentum):
            layer.momentum = m
    total = 0
    for layer in mdl._flatten_layers():
        if isinstance(layer, QLayerBase) and layer.enable_ebops and layer._ebops is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def make_lr_scheduler(lr_init, cycle, mmul):
    fn = cosine_decay_restarts_schedule(lr_init, cycle, t_mul=1.0, m_mul=mmul,
                                        alpha=1e-6, alpha_steps=50)
    return LearningRateScheduler(lambda epoch: fn(epoch))


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 65)
    print(f'  Random-Init HGQ Training (beta budget control)')
    print(f'  Target      : ebops = {TARGET_EBOPS}')
    print(f'  Epochs      : {TOTAL_EPOCHS}')
    print(f'  LR          : {LR_INIT:.1e}  (cycle={LR_CYCLE}, mmul={LR_MMUL})')
    print(f'  Beta        : init={BETA_INIT:.1e}  range=[{BETA_MIN:.1e}, {BETA_MAX:.1e}]')
    print(f'  Init BW     : k={args.init_bw_k}')
    print(f'  Output      : {output_folder}')
    print('=' * 65)

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/3] Loading dataset...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src='openml')
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val   = Dataset(X_val,   y_val,   batch_size, device)

    _sample_input = tf.constant(X_train[:min(2048, len(X_train))], dtype=tf.float32)

    # ── 2. 构建随机初始化模型 ────────────────────────────────────────────────
    print(f'\n[2/3] Building fresh HGQ model (init_bw_k={args.init_bw_k})...')
    model = get_model_hgq(init_bw_k=args.init_bw_k, init_bw_a=3)
    model.summary()

    # 测量随机初始化时的 EBOPs
    init_ebops = compute_model_ebops(model, _sample_input)
    print(f'  Initial EBOPs (random init, full bitwidth): {init_ebops:.1f}')
    print_bk_stats(model, 'random init')

    # ── 缩放 kq.b 到目标附近 ────────────────────────────────────────────────
    # 随机初始化后 kq.b 全部 = init_bw_k+1（如 4.0），EBOPs 远高于目标。
    # 如果不缩放，beta 需要 ~100 epoch 才能把 EBOPs 压下来，浪费训练时间。
    #
    # 注意: 均匀缩放全同的 kq.b 会遇到量化阶梯问题——所有连接同时跨越整数
    # 位宽边界，导致 EBOPs 在 ~660 和 ~40990 之间没有中间值。
    # 解决方案: 先给 kq.b 加入随机噪声（±20%），打破对齐，使 EBOPs 成为
    # 平滑函数后再做二分搜索。
    calib_target = TARGET_EBOPS * 1.10  # 同 prune_finetune 的 CALIBRATION_OVERSHOOT

    # 快照所有 kq.b
    snapshots = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, 'kq', None)
        if kq is None:
            continue
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            snapshots.append((b_var, b_var.numpy().copy()))

    # 加入 per-element 随机噪声，打破 kq.b 全同导致的量化阶梯
    rng = np.random.RandomState(42)
    for b_var, b_snap in snapshots:
        noise = rng.uniform(0.7, 1.3, size=b_snap.shape).astype(np.float32)
        b_var.assign(b_snap * noise)
    # 重新快照（含噪声的值作为新基准）
    snapshots = [(bv, bv.numpy().copy()) for bv, _ in snapshots]

    noisy_ebops = compute_model_ebops(model, _sample_input)
    print(f'  After adding noise to kq.b: EBOPs = {noisy_ebops:.1f}')
    print(f'  Scaling kq.b to reach ~{calib_target:.0f} EBOPs (target={TARGET_EBOPS:.0f} × 1.10)...')

    def _apply_scale(s):
        for b_var, b_snap in snapshots:
            b_var.assign((b_snap * s).astype(np.float32))

    def _measure(s):
        _apply_scale(s)
        return compute_model_ebops(model, _sample_input)

    # 二分搜索 scale
    # kq.b 已有噪声，EBOPs 是 scale 的平滑函数，二分法可正常收敛。
    lo, hi = 0.001, 1.0
    best_scale, best_ebops, best_err = hi, noisy_ebops, abs(noisy_ebops - calib_target)
    for i in range(40):
        mid = (lo + hi) / 2.0
        e = _measure(mid)
        err = abs(e - calib_target)
        if err < best_err:
            best_scale, best_ebops, best_err = mid, e, err
        if err / calib_target < 0.03:
            best_scale, best_ebops = mid, e
            break
        if e > calib_target:
            hi = mid
        else:
            lo = mid

    scaled_ebops = _measure(best_scale)
    print(f'  After kq.b scaling: EBOPs = {scaled_ebops:.1f}  (scale = {best_scale:.5f}, target = {calib_target:.0f})')
    print_bk_stats(model, 'after scaling')

    # ── 3. 编译 & 训练 ───────────────────────────────────────────────────────
    print(f'\n[3/3] Training  ({TOTAL_EPOCHS} epochs)')
    print(f'  Strategy: BetaOnlyBudgetController maintains EBOPs ≈ {TARGET_EBOPS:.0f}')

    set_all_beta(model, BETA_INIT)

    # Budget controller 配置
    # kq.b 已缩放到目标附近，controller 只需维持 EBOPs 在目标带内。
    budget_ctrl = BetaOnlyBudgetController(
        target_ebops    = TARGET_EBOPS,
        margin          = 0.20,         # ±20%
        beta_init       = BETA_INIT,
        beta_min        = BETA_MIN,
        beta_max        = BETA_MAX,
        adjust_factor   = 1.15,
        ema_alpha       = 0.25,
        warmup_epochs   = 100,          # 温和起步
        max_change_ratio = 1.5,
        init_ebops      = TARGET_EBOPS, # EMA 从目标值开始（kq.b 已校准到附近）
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_INIT),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

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
    early_stop_cb = BudgetAwareEarlyStopping(
        ebops_budget         = EARLYSTOP_BUDGET,
        patience             = EARLYSTOP_PATIENCE,
        min_delta            = 5e-5,
        min_epoch            = 1000,          # 至少训练 1000 epoch 再考虑 early stop
        restore_best_weights = True,
    )

    callbacks = [
        ebops_cb,
        budget_ctrl,
        pareto_cb,
        make_lr_scheduler(LR_INIT, LR_CYCLE, LR_MMUL),
        trace_cb,
        early_stop_cb,
    ]

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=TOTAL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 最终统计 ──────────────────────────────────────────────────────────────
    print_bk_stats(model, 'final')
    final_ebops = compute_model_ebops(model, _sample_input)
    res = model.evaluate(dataset_val, verbose=0)
    print(f'\n  Final:  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}  ebops={final_ebops:.1f}')

    print('\n' + '=' * 65)
    print('Training complete.')
    print(f'Pareto checkpoints : {output_folder}')
    print(f'Training trace     : {os.path.join(output_folder, "training_trace.h5")}')
    print('=' * 65)
