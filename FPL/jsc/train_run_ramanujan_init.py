"""
Ramanujan 图初始化 → BetaOnlyBudgetController 训练脚本
=====================================================
假说：随机初始化在低 EBOPs（如 2000）下无法训练，是因为稀疏网络的梯度流
被随机拓扑切断——大量输入/输出节点成为孤岛，信号在前向/反向传播中衰减为零。

Ramanujan 图的谱间隙性质保证：
  λ_2 ≤ 2√(d-1)   (d = 正则度)
这意味着图的连通性接近完全图的最优水平，即使在极度稀疏（d << N）时仍然
保持高效信息传播。具体到神经网络：
  1. 前向传播：每个输出节点都能"看到"充分多样的输入特征组合
  2. 反向传播：梯度能沿高连通性路径回传到所有权重，避免死区
  3. 位宽分配：按 EBOPS 预算反推每层位宽，使初始化时总 EBOPS ≈ 目标

流程
----
1. 从零构建 HGQ 模型（随机初始化，init_bw_k=3）
2. 用 compute_bw_aware_degree 联合求解每层的 Ramanujan 度和初始位宽
3. 用 apply_ramanujan_bw_init 应用 Ramanujan 拓扑 + 位宽初始化
4. RamanujanMaskEnforcer 在 warmup 期间保护拓扑，之后渐进放开
5. BetaOnlyBudgetController 维持 EBOPS ≈ 目标

与 train_run_random_init.py 的区别
-----------------------------------
- 使用 Ramanujan 图拓扑初始化，而非完全随机
- BW-aware 度/位宽联合分配，初始化即在目标 EBOPS 附近
- 不需要二分搜索缩放 kq.b（已由 compute_bw_aware_degree 精确分配）
- 添加 RamanujanMaskEnforcer 保护稀疏拓扑

与 train_run_ram.py 的区别
---------------------------
- 使用 BetaOnlyBudgetController 精确控制 EBOPS 预算，而非 BetaScheduler
- 目标明确：2000 EBOPS（可通过命令行参数修改）
- 使用 BW-aware 度分配（compute_bw_aware_degree），而非固定度
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
    compute_bw_aware_degree,
    apply_ramanujan_bw_init,
    _flatten_layers,
    _get_kq_var,
)
from utils.ramaujian_utils import RamanujanMaskEnforcer

np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

TARGET_EBOPS = 2001

input_folder = 'data/dataset.h5'
batch_size   = 33200

# ── 训练参数 ──────────────────────────────────────────────────────────────────
TOTAL_EPOCHS = 50000
LR_INIT      = 2e-3
LR_CYCLE     = 2000
LR_MMUL      = 0.95

# ── Beta 控制参数 ──────────────────────────────────────────────────────────────
# Ramanujan 初始化后 EBOPs 已在目标附近，beta 只需温和维持。
BETA_INIT = 1e-5
BETA_MIN  = 1e-8
BETA_MAX  = 1e-3

EARLYSTOP_PATIENCE = 5000

# ── Ramanujan 参数 ────────────────────────────────────────────────────────────
# 度的乘数：degree = round(sqrt(in_dim) * multiplier)
# multiplier=1.5 保证谱间隙余量
RAM_MULTIPLIER = 1.5
RAM_MIN_DEGREE = 4
# 位宽分配策略：'capacity' = 按层容量分配预算, 'uniform' = 等分
RAM_BUDGET_WEIGHT = 'capacity'

# ── Mask 保护参数 ─────────────────────────────────────────────────────────────
# warmup 期间完全固定 Ramanujan 拓扑，之后渐进放开
MASK_HOLD_EPOCHS = 500        # 完全固定的 epoch 数
MASK_FADE_EPOCHS = 2000       # 从固定到完全放开的过渡 epoch 数

# ── 命令行参数覆盖 ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Ramanujan-init HGQ training with beta budget control')
parser.add_argument('--target_ebops',    type=float, default=TARGET_EBOPS)
parser.add_argument('--total_epochs',    type=int,   default=TOTAL_EPOCHS)
parser.add_argument('--lr',              type=float, default=LR_INIT)
parser.add_argument('--beta_init',       type=float, default=BETA_INIT)
parser.add_argument('--init_bw_k',       type=int,   default=3,
                    help='Initial kernel bitwidth for HGQ model (default: 3)')
parser.add_argument('--ram_multiplier',  type=float, default=RAM_MULTIPLIER,
                    help='Ramanujan degree multiplier on sqrt(in_dim)')
parser.add_argument('--ram_min_degree',  type=int,   default=RAM_MIN_DEGREE)
parser.add_argument('--mask_hold',       type=int,   default=MASK_HOLD_EPOCHS,
                    help='Epochs to fully hold Ramanujan mask')
parser.add_argument('--mask_fade',       type=int,   default=MASK_FADE_EPOCHS,
                    help='Epochs to fade from hold to release')
args, _ = parser.parse_known_args()

TARGET_EBOPS     = args.target_ebops
TOTAL_EPOCHS     = args.total_epochs
LR_INIT          = args.lr
BETA_INIT        = args.beta_init
MASK_HOLD_EPOCHS = args.mask_hold
MASK_FADE_EPOCHS = args.mask_fade
EARLYSTOP_BUDGET = TARGET_EBOPS * 1.5

output_folder = f'results/ramanujan_init_{int(TARGET_EBOPS)}/'
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
        active = arr[arr > 0.1]
        dead   = arr[arr <= 0.1]
        print(f'  [bk_stats {label}]  '
              f'mean={arr.mean():.3f}  std={arr.std():.3f}  '
              f'min={arr.min():.3f}  max={arr.max():.3f}  '
              f'p5={np.percentile(arr,5):.3f}  p95={np.percentile(arr,95):.3f}  '
              f'n_dead(<=0.1)={len(dead)}/{len(arr)} ({100*len(dead)/len(arr):.1f}%)  '
              f'active_mean={active.mean():.3f}' if len(active) > 0 else '')


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


def print_ramanujan_topology_stats(mdl):
    """打印 Ramanujan 初始化后每层的拓扑统计。"""
    print('\n  ┌─ Ramanujan Topology Summary ────────────────────────────────┐')
    for layer in _flatten_layers(mdl):
        mask = getattr(layer, 'ramanujan_mask', None)
        if mask is None:
            continue
        m = mask.numpy()
        active = int(m.sum())
        total  = int(m.size)
        sparsity = 1.0 - m.mean()

        # 分析连通性
        if m.ndim == 2:
            row_deg = m.sum(axis=1)  # 每个输入节点的度
            col_deg = m.sum(axis=0)  # 每个输出节点的度
            isolated_in  = int((row_deg == 0).sum())
            isolated_out = int((col_deg == 0).sum())
            print(f'  │ {layer.name:20s}  active={active:5d}/{total:5d}  '
                  f'sparsity={sparsity:.1%}  '
                  f'row_deg=[{row_deg.min():.0f},{row_deg.max():.0f}]  '
                  f'col_deg=[{col_deg.min():.0f},{col_deg.max():.0f}]  '
                  f'isolated_in={isolated_in}  isolated_out={isolated_out}')
        else:
            print(f'  │ {layer.name:20s}  active={active:5d}/{total:5d}  '
                  f'sparsity={sparsity:.1%}')
    print('  └─────────────────────────────────────────────────────────────┘\n')


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print(f'  Ramanujan-Init HGQ Training (BW-aware + beta budget control)')
    print(f'  Target      : ebops = {TARGET_EBOPS}')
    print(f'  Epochs      : {TOTAL_EPOCHS}')
    print(f'  LR          : {LR_INIT:.1e}  (cycle={LR_CYCLE}, mmul={LR_MMUL})')
    print(f'  Beta        : init={BETA_INIT:.1e}  range=[{BETA_MIN:.1e}, {BETA_MAX:.1e}]')
    print(f'  Init BW     : k={args.init_bw_k}')
    print(f'  Ramanujan   : multiplier={args.ram_multiplier}, min_degree={args.ram_min_degree}')
    print(f'  Mask protect: hold={MASK_HOLD_EPOCHS}, fade={MASK_FADE_EPOCHS}')
    print(f'  Output      : {output_folder}')
    print('=' * 70)

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/4] Loading dataset...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src='openml')
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val   = Dataset(X_val,   y_val,   batch_size, device)

    _sample_input = tf.constant(X_train[:min(2048, len(X_train))], dtype=tf.float32)

    # ── 2. 构建模型（先用默认初始化）─────────────────────────────────────────
    print(f'\n[2/4] Building fresh HGQ model (init_bw_k={args.init_bw_k})...')
    model = get_model_hgq(init_bw_k=args.init_bw_k, init_bw_a=3)
    model.summary()

    # 测量随机初始化时的 EBOPs（作为参照）
    init_ebops = compute_model_ebops(model, _sample_input)
    print(f'  Random-init EBOPs (full bitwidth): {init_ebops:.1f}')
    print_bk_stats(model, 'before Ramanujan init')

    # ── 3. Ramanujan BW-aware 初始化 ─────────────────────────────────────────
    print(f'\n[3/4] Applying Ramanujan BW-aware initialization...')
    print(f'  Solving joint degree + bitwidth allocation for target={TARGET_EBOPS:.0f} EBOPs...')

    # 联合求解每层的 Ramanujan 度和初始位宽（线性公式估算）
    per_layer_degree, per_layer_bk = compute_bw_aware_degree(
        model,
        target_ebops=TARGET_EBOPS,
        b_a_init=3.0,        # activation 初始位宽
        b_k_min=0.5,         # kernel 位宽下限
        b_k_max=8.0,         # kernel 位宽上限
        multiplier=args.ram_multiplier,
        min_degree=args.ram_min_degree,
        budget_weight=RAM_BUDGET_WEIGHT,
        verbose=True,
    )

    # 应用 Ramanujan 拓扑 + 位宽初始化
    # 先用较高的初始 b_k（3.0），然后通过二分搜索校准到目标 EBOPs
    # 因为 HGQ 的 EBOPS 与 kq.b 不是线性关系，线性公式估算会有很大偏差。
    INIT_BK_FOR_ACTIVE = 3.0   # 先给所有活跃连接 3-bit，再统一缩放
    apply_ramanujan_bw_init(
        model,
        per_layer_degree=per_layer_degree,
        per_layer_bk={name: INIT_BK_FOR_ACTIVE for name in per_layer_degree},
        seed=42,
        pruned_frac_bits=0.0,    # 被剪连接位宽 = 0 → HGQ 输出恒为 0
        pruned_int_bits=0.0,
        active_int_bits=1.0,     # 保留连接的整数位宽
        also_zero_kernel=True,   # 同步清零浮点 kernel
        verbose=True,
    )

    # Ramanujan 拓扑已应用，现在校准 kq.b 到目标 EBOPs
    ram_ebops_raw = compute_model_ebops(model, _sample_input)
    print(f'\n  Ramanujan topology applied (b_k={INIT_BK_FOR_ACTIVE}): EBOPs = {ram_ebops_raw:.1f}')
    print_bk_stats(model, 'after Ramanujan (before calibration)')
    print_ramanujan_topology_stats(model)

    # ── 校准 kq.b：加噪声 + 二分搜索缩放到目标 EBOPs ────────────────────────
    # HGQ 的 EBOPS 跟 kq.b 是非线性关系（量化阶梯效应）。
    # 如果所有活跃连接的 kq.b 完全相同（如 3.0），它们会同时跨越整数位宽
    # 边界，导致 EBOPs 在少数几个离散值之间跳变，二分搜索无法收敛。
    # 解决：先给 kq.b 加 ±20% 随机噪声打破对齐，使 EBOPs 成为平滑函数。
    calib_target = TARGET_EBOPS * 1.10

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

    # 加入 per-element 随机噪声，只对活跃连接（b > 0.05）
    rng = np.random.RandomState(42)
    for b_var, b_snap in snapshots:
        active = b_snap > 0.05
        noise = rng.uniform(0.8, 1.2, size=b_snap.shape).astype(np.float32)
        b_noisy = np.where(active, b_snap * noise, b_snap)
        b_var.assign(b_noisy.astype(np.float32))
    # 重新快照（含噪声的值作为新基准）
    snapshots = [(bv, bv.numpy().copy()) for bv, _ in snapshots]

    noisy_ebops = compute_model_ebops(model, _sample_input)
    print(f'  After adding noise to active kq.b: EBOPs = {noisy_ebops:.1f}')
    print(f'  Calibrating kq.b to reach ~{calib_target:.0f} EBOPs (target={TARGET_EBOPS:.0f} × 1.10)...')

    def _apply_scale(s):
        for b_var, b_snap in snapshots:
            active = b_snap > 0.05
            b_new = np.where(active, (b_snap * s).clip(0.1, 8.0), b_snap)
            b_var.assign(b_new.astype(np.float32))

    def _measure(s):
        _apply_scale(s)
        return compute_model_ebops(model, _sample_input)

    # 二分搜索 scale
    lo, hi = 0.01, 3.0
    best_scale, best_ebops, best_err = 1.0, noisy_ebops, abs(noisy_ebops - calib_target)
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
        if i % 10 == 0:
            print(f'    [bisect iter={i}]  scale={mid:.5f}  ebops={e:.1f}  target={calib_target:.0f}')

    _apply_scale(best_scale)
    calibrated_ebops = compute_model_ebops(model, _sample_input)
    print(f'  After calibration: EBOPs = {calibrated_ebops:.1f}  '
          f'(scale={best_scale:.5f}, target={calib_target:.0f})')
    print_bk_stats(model, 'after calibration')

    # ── 4. 编译 & 训练 ───────────────────────────────────────────────────────
    print(f'\n[4/4] Training  ({TOTAL_EPOCHS} epochs)')
    print(f'  Strategy: Ramanujan topology + BetaOnlyBudgetController → EBOPs ≈ {TARGET_EBOPS:.0f}')
    print(f'  Mask: hold {MASK_HOLD_EPOCHS} epochs, then fade over {MASK_FADE_EPOCHS} epochs')

    set_all_beta(model, BETA_INIT)

    # Budget controller
    budget_ctrl = BetaOnlyBudgetController(
        target_ebops    = TARGET_EBOPS,
        margin          = 0.20,         # ±20%
        beta_init       = BETA_INIT,
        beta_min        = BETA_MIN,
        beta_max        = BETA_MAX,
        adjust_factor   = 1.15,
        ema_alpha       = 0.25,
        warmup_epochs   = 200,          # 前 200 epoch 温和调节
        max_change_ratio = 1.5,
        init_ebops      = TARGET_EBOPS, # EMA 从目标值开始
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

    # Ramanujan mask enforcer: 在 warmup 期间保护拓扑结构
    mask_enforcer = RamanujanMaskEnforcer(
        release_epoch=MASK_HOLD_EPOCHS,
        fade_epochs=MASK_FADE_EPOCHS,
    )

    early_stop_cb = BudgetAwareEarlyStopping(
        ebops_budget         = EARLYSTOP_BUDGET,
        patience             = EARLYSTOP_PATIENCE,
        min_delta            = 5e-5,
        min_epoch            = 1000,
        restore_best_weights = True,
    )

    callbacks = [
        ebops_cb,
        budget_ctrl,
        mask_enforcer,       # 保护 Ramanujan 拓扑
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

    # 打印最终拓扑（看看 mask 放开后拓扑变化了多少）
    print_ramanujan_topology_stats(model)

    print('\n' + '=' * 70)
    print('Training complete.')
    print(f'Pareto checkpoints : {output_folder}')
    print(f'Training trace     : {os.path.join(output_folder, "training_trace.h5")}')
    print('=' * 70)
