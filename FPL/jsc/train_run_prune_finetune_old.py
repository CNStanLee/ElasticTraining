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
from model.model import get_model_hgq
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
    KQBStabilizer,
    compute_bw_aware_degree,
    apply_ramanujan_bw_init,
    _flatten_layers,
    _get_kq_var,
)
from utils.ramaujian_utils import RamanujanMaskEnforcer
from utils.topology_graph_plot_utils import LayerGraphData, TopologyGraphPlotter
from run_one_shot_prune_only import (
    saliency_prune_to_ebops,
    spectral_path_prune_to_ebops,
    spectral_quant_prune_to_ebops,
    snows_prune_to_ebops,
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
BASELINE_CKPT = "results/baseline/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras"
BASELINE_EBOPS = 19899

TARGET_EBOPS = 1000

# Ramanujan init v2 tuned defaults (kept separate to avoid polluting other branches)
DEFAULT_INIT_CALIB_MULT = 1.10   # legacy for random_init; Ramanujan branch overrides to 1.0
RAM_CALIB_DEFAULT     = 1.00
RAM_P1_BETA_INIT      = 5e-7
RAM_P1_BETA_MIN       = 1e-8
RAM_P1_BETA_MAX       = 2e-4
RAM_P1_MARGIN         = 0.20
RAM_P1_ADJUST         = 1.15
RAM_P1_EMA            = 0.25
RAM_P1_WARMUP         = 100
RAM_P1_MAX_CHANGE     = 1.50
RAM_P2_MARGIN         = 0.05
RAM_P2_BETA_MIN       = 1e-7
RAM_P2_BETA_MAX       = 5e-4
RAM_P2_ADJUST         = 1.15
RAM_P2_EMA            = 0.15
RAM_P2_WARMUP         = 0
RAM_P2_MAX_CHANGE     = 1.50

# ── 剪枝方式 ──────────────────────────────────────────────────────────────────
# 'uniform'        : 旧方案 HighBitPruner (均匀缩放，有精度损失)
# 'sensitivity'    : 按层敏感度分配预算
# 'snip'           : 单次梯度连接敏感度 |dL/dw * w|
# 'grasp'          : 梯度流保持的 Hessian-gradient saliency
# 'synflow'        : data-free synaptic flow saliency
# 'spectral_quant' : 谱/拓扑友好剪枝（低预算会自动走结构化子网络）
# 'spectral_path'  : 在 spectral_quant 上加入端到端路径连通性约束
# 'snows'          : sensitivity/spectral warm-start + Hessian-free K-step representation reconstruction
# 'random_init'    : 随机初始化 + kq.b 校准到预算附近
# 'ramanujan_init' : Ramanujan 拓扑初始化 + kq.b 校准到预算附近
PRUNE_METHOD = 'spectral_quant'

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
                    choices=['uniform', 'sensitivity', 'snip', 'grasp', 'synflow', 'spectral_quant', 'spectral_path', 'snows', 'random_init', 'ramanujan_init'])
parser.add_argument('--output_dir',    type=str,   default='',
                    help='Optional output directory. Defaults to results/prune_finetune_{target}/')
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
parser.add_argument('--snows_init_method', type=str, default='sensitivity',
                    choices=['uniform', 'sensitivity', 'spectral_quant'],
                    help='Initial budget allocation used before SNOWS reconstruction')
parser.add_argument('--snows_k_step', type=int, default=2,
                    help='SNOWS look-ahead horizon in prunable layers')
parser.add_argument('--snows_newton_steps', type=int, default=2,
                    help='SNOWS Newton updates per layer')
parser.add_argument('--snows_cg_iters', type=int, default=25,
                    help='Maximum conjugate-gradient iterations for SNOWS')
parser.add_argument('--snows_damping', type=float, default=1e-4,
                    help='SNOWS Hessian-free damping')
parser.add_argument('--init_bw_k', type=int, default=3,
                    help='Initial kernel bitwidth for random/ramanujan initialization branches')
parser.add_argument('--init_calib_multiplier', type=float, default=DEFAULT_INIT_CALIB_MULT,
                    help='Calibrate fresh-init models to target * multiplier eBOPs before training')
parser.add_argument('--ram_multiplier', type=float, default=1.5,
                    help='Ramanujan degree multiplier')
parser.add_argument('--ram_min_degree', type=int, default=4,
                    help='Minimum Ramanujan per-layer degree')
parser.add_argument('--ram_mask_hold', type=int, default=None,
                    help='Epochs to hold Ramanujan mask before release (default: phase1_epochs)')
parser.add_argument('--ram_mask_fade', type=int, default=3000,
                    help='Fade epochs for Ramanujan mask release')
args, _ = parser.parse_known_args()

TARGET_EBOPS    = args.target_ebops
PHASE1_EPOCHS   = args.phase1_epochs
PHASE2_EPOCHS   = args.phase2_epochs
BASELINE_CKPT   = args.checkpoint
PRUNE_METHOD    = args.prune_method
EARLYSTOP_BUDGET = TARGET_EBOPS * 1.5
# Ramanujan init v2 使用 1.0× 校准；保持 random_init 旧默认 1.10×
ram_calib_multiplier = args.init_calib_multiplier
if PRUNE_METHOD == 'ramanujan_init' and args.init_calib_multiplier == DEFAULT_INIT_CALIB_MULT:
    ram_calib_multiplier = RAM_CALIB_DEFAULT
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

output_folder = args.output_dir or f'results/prune_finetune_{int(TARGET_EBOPS)}/'
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


def calibrate_kqb_to_target(
    model,
    sample_input,
    target_ebops: float,
    noise_low: float = 0.8,
    noise_high: float = 1.2,
    active_threshold: float = 0.05,
    lo: float = 0.005,
    hi: float = 5.0,
    max_iter: int = 50,
):
    """Uniformly rescale active kq.b with de-alignment noise to hit a target EBOPs."""
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

    if not snapshots:
        return compute_model_ebops(model, sample_input)

    rng = np.random.RandomState(42)
    noisy_snapshots = []
    for b_var, b_snap in snapshots:
        active = b_snap > active_threshold
        noise = rng.uniform(noise_low, noise_high, size=b_snap.shape).astype(np.float32)
        b_noisy = np.where(active, b_snap * noise, b_snap)
        b_var.assign(b_noisy.astype(np.float32))
        noisy_snapshots.append((b_var, b_var.numpy().copy()))

    def _apply_scale(scale: float):
        for b_var, b_snap in noisy_snapshots:
            active = b_snap > active_threshold
            b_new = np.where(active, (b_snap * scale).clip(0.1, 8.0), b_snap)
            b_var.assign(b_new.astype(np.float32))

    def _measure(scale: float) -> float:
        _apply_scale(scale)
        return compute_model_ebops(model, sample_input)

    current_ebops = _measure(1.0)
    best_scale, best_ebops = 1.0, current_ebops
    best_err = abs(current_ebops - target_ebops)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        ebops = _measure(mid)
        err = abs(ebops - target_ebops)
        if err < best_err:
            best_scale, best_ebops, best_err = mid, ebops, err
        if err / max(target_ebops, 1.0) < 0.03:
            best_scale, best_ebops = mid, ebops
            break
        if ebops > target_ebops:
            hi = mid
        else:
            lo = mid

    _apply_scale(best_scale)
    return compute_model_ebops(model, sample_input)


def save_init_artifacts(
    model,
    output_folder: str,
    stem: str,
    title: str,
    sample_input,
):
    """Save current model weights and plot its topology before training."""
    init_dir = Path(output_folder) / 'init_graph'
    init_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(output_folder) / f'{stem}.keras'
    model.save(model_path)

    measured_ebops = compute_model_ebops(model, sample_input)
    subtitle = f'ebops={measured_ebops:.0f}'
    plotter = TopologyGraphPlotter(
        symmetric_topology_plot=False,
        mirror_edges=False,
        plot_matrix=True,
    )
    layers = plotter.extract_layer_graph_data(model)
    matrix_path = init_dir / f'{stem}_weighted_topology_matrix.png'
    circle_path = init_dir / f'{stem}_weighted_circle_graph.png'
    plotter.plot_weighted_topology_matrices(layers, matrix_path, title=title, subtitle=subtitle)
    plotter.plot_circle_graph(layers, circle_path, title=title, subtitle=subtitle)
    return {
        'model_path': str(model_path),
        'matrix_path': str(matrix_path),
        'circle_path': str(circle_path),
        'ebops': measured_ebops,
    }


def save_ramanujan_mask_artifacts(
    model,
    output_folder: str,
    stem: str,
    title: str,
    sample_input,
):
    """Plot the exact Ramanujan mask instead of weight-magnitude-weighted edges."""
    init_dir = Path(output_folder) / 'init_graph'
    init_dir.mkdir(parents=True, exist_ok=True)

    layers = []
    for layer in _flatten_layers(model):
        mask = getattr(layer, 'ramanujan_mask', None)
        if mask is None:
            continue
        m = np.array(mask.numpy(), dtype=np.float32)
        if m.ndim < 2:
            continue
        layers.append(
            LayerGraphData(
                name=layer.name,
                bit_matrix=m,
                weighted_matrix=m,
            )
        )
    if not layers:
        return None

    measured_ebops = compute_model_ebops(model, sample_input)
    subtitle = f'ebops={measured_ebops:.0f} | exact ramanujan mask'
    plotter = TopologyGraphPlotter(
        symmetric_topology_plot=False,
        mirror_edges=False,
        plot_matrix=True,
    )
    matrix_path = init_dir / f'{stem}_weighted_topology_matrix.png'
    circle_path = init_dir / f'{stem}_weighted_circle_graph.png'
    plotter.plot_weighted_topology_matrices(layers, matrix_path, title=title, subtitle=subtitle)
    plotter.plot_circle_graph(layers, circle_path, title=title, subtitle=subtitle)
    return {
        'matrix_path': str(matrix_path),
        'circle_path': str(circle_path),
        'ebops': measured_ebops,
    }


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
    init_only_methods = {'random_init', 'ramanujan_init'}

    print('=' * 65)
    print(f'  One-shot Prune + Two-phase Finetune')
    if PRUNE_METHOD in init_only_methods:
        print(f'  Source      : fresh model initialization')
    else:
        print(f'  Source      : {BASELINE_CKPT}')
    if warm_start_msg:
        print(f'  Warm-start  : {warm_start_msg}')
    print(f'  Target      : ebops = {TARGET_EBOPS}')
    print(f'  Prune method: {PRUNE_METHOD}')
    print(f'  Func-calib  : {args.functional_calibrate} (passes={args.functional_passes})')
    if PRUNE_METHOD == 'snows':
        print(f'  SNOWS init  : {args.snows_init_method}')
        print(f'  SNOWS K     : {args.snows_k_step}')
        print(f'  SNOWS Newton: {args.snows_newton_steps} x CG{args.snows_cg_iters}')
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
    teacher_model = None
    ramanujan_mask_enforcer = None
    ramanujan_kqb_stabilizer = None
    ramanujan_init_report = None
    random_init_report = None
    if PRUNE_METHOD in init_only_methods:
        print(f'\n[2/5] Building fresh HGQ model (init_bw_k={args.init_bw_k})')
        model = get_model_hgq(init_bw_k=args.init_bw_k, init_bw_a=3)
        model.summary()
        print_bk_stats(model, 'fresh init')
        actual_baseline_ebops = compute_model_ebops(model, _sample_input)
        print(f'  Fresh-init EBOPs (measured): {actual_baseline_ebops:.1f}')
    else:
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
    stage_label = 'Initialization' if PRUNE_METHOD in init_only_methods else 'One-shot bit-width pruning'
    print(f'\n[3/5] {stage_label} ({PRUNE_METHOD}): '
          f'{actual_baseline_ebops:.1f} -> {TARGET_EBOPS}')
    used_structured_low_budget = False
    snows_report = None
    spectral_path_report = None
    ramanujan_precalib_report = None
    if PRUNE_METHOD == 'random_init':
        calib_target = TARGET_EBOPS * args.init_calib_multiplier
        post_prune_ebops = calibrate_kqb_to_target(
            model,
            _sample_input,
            calib_target,
            noise_low=0.7,
            noise_high=1.3,
        )
        print(f'  Random-init calibration: target={calib_target:.1f}  actual={post_prune_ebops:.1f}')
        random_init_report = save_init_artifacts(
            model,
            output_folder=output_folder,
            stem='random_init_start',
            title='Random Init Start Topology',
            sample_input=_sample_input,
        )
        print(f'  Random-init graph: {random_init_report["circle_path"]}')
    elif PRUNE_METHOD == 'ramanujan_init':
        per_layer_degree, _ = compute_bw_aware_degree(
            model,
            target_ebops=TARGET_EBOPS,
            b_a_init=3.0,
            b_k_min=0.5,
            b_k_max=8.0,
            multiplier=args.ram_multiplier,
            min_degree=args.ram_min_degree,
            budget_weight='capacity',
            verbose=True,
        )
        apply_ramanujan_bw_init(
            model,
            per_layer_degree=per_layer_degree,
            per_layer_bk={name: 4.0 for name in per_layer_degree},
            seed=42,
            pruned_frac_bits=0.0,
            pruned_int_bits=0.0,
            active_int_bits=1.0,
            also_zero_kernel=True,
            verbose=True,
        )
        raw_ramanujan_ebops = compute_model_ebops(model, _sample_input)
        print(f'  Ramanujan topology init EBOPs: {raw_ramanujan_ebops:.1f}')
        ramanujan_precalib_report = save_ramanujan_mask_artifacts(
            model,
            output_folder=output_folder,
            stem='ramanujan_init_pre_calib',
            title='Ramanujan Init Exact Mask Topology',
            sample_input=_sample_input,
        )
        if ramanujan_precalib_report is not None:
            print(f'  Ramanujan pre-calib mask graph: {ramanujan_precalib_report["circle_path"]}')
        post_prune_ebops = calibrate_kqb_to_target(
            model,
            _sample_input,
            TARGET_EBOPS * ram_calib_multiplier,
        )
        print(f'  Ramanujan calibrated start EBOPs: {post_prune_ebops:.1f} '
              f'(mult={ram_calib_multiplier:.2f}×)')
        ramanujan_init_report = save_init_artifacts(
            model,
            output_folder=output_folder,
            stem='ramanujan_init_start',
            title='Ramanujan Init Start Topology',
            sample_input=_sample_input,
        )
        print(f'  Ramanujan init graph: {ramanujan_init_report["circle_path"]}')
        ramanujan_kqb_stabilizer = KQBStabilizer(
            hold_epochs=50,
            release_epochs=200,
            hold_strength=0.8,
        )
        ramanujan_kqb_stabilizer.capture_snapshot(model)
        ramanujan_mask_enforcer = RamanujanMaskEnforcer(
            release_epoch=(
                args.ram_mask_hold if args.ram_mask_hold is not None
                else (min(2000, PHASE1_EPOCHS) if ram_calib_multiplier > 1.01 else PHASE1_EPOCHS)
            ),
            fade_epochs=args.ram_mask_fade,
            min_active_frac_bits=1.0 if ram_calib_multiplier > 1.01 else None,
        )
    elif PRUNE_METHOD == 'sensitivity':
        pruner = SensitivityAwarePruner(target_ebops=TARGET_EBOPS, pruned_threshold=0.1, b_k_min=0.3)
        pruner.prune_to_ebops(model, current_ebops=actual_baseline_ebops, verbose=True)
    elif PRUNE_METHOD == 'uniform':
        pruner = HighBitPruner(target_ebops=TARGET_EBOPS, pruned_threshold=0.1)
        pruner.prune_to_ebops(model, current_ebops=actual_baseline_ebops, verbose=True)
    elif PRUNE_METHOD in {'snip', 'grasp', 'synflow'}:
        saliency_prune_to_ebops(
            model,
            target_ebops=TARGET_EBOPS,
            sample_input=_sample_input,
            method=PRUNE_METHOD,
            input_h5=input_folder,
            sample_size=min(512, int(_sample_input.shape[0])),
            b_floor=0.35,
            verbose=True,
        )
    elif PRUNE_METHOD == 'snows':
        _, snows_report = snows_prune_to_ebops(
            model,
            teacher_model=teacher_model,
            target_ebops=TARGET_EBOPS,
            sample_input=_sample_input,
            init_method=args.snows_init_method,
            b_floor=0.30,
            k_step=args.snows_k_step,
            newton_steps=args.snows_newton_steps,
            cg_max_iter=args.snows_cg_iters,
            damping=args.snows_damping,
            verbose=True,
        )
        used_structured_low_budget = bool(snows_report.get('used_structured_low_budget', False))
    elif PRUNE_METHOD == 'spectral_path':
        _, spectral_path_report = spectral_path_prune_to_ebops(
            model,
            target_ebops=TARGET_EBOPS,
            sample_input=_sample_input,
            min_degree=2,
            min_hidden_width=4,
            b_floor=0.35,
            near_budget_ratio=1.6,
            high_budget_ratio=0.45,
            verbose=True,
        )
        used_structured_low_budget = True
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
    print_bk_stats(model, 'after init/pruning')

    if PRUNE_METHOD not in init_only_methods:
        # 实测剪枝后 EBOPs，迭代修正到目标 ±3%
        post_prune_ebops = compute_model_ebops(model, _sample_input)
    print(f'  Post-stage EBOPs (measured): {post_prune_ebops:.1f}  target: {TARGET_EBOPS}')
    if snows_report is not None:
        print(
            f"  SNOWS repr-loss sum: "
            f"{snows_report['representation_loss_before_sum']:.6f} -> "
            f"{snows_report['representation_loss_after_sum']:.6f}"
        )
    near_budget_preserve_case = (
        PRUNE_METHOD in {'spectral_quant', 'spectral_path'}
        and (not used_structured_low_budget)
        and actual_baseline_ebops <= float(TARGET_EBOPS) * 1.6
    )
    ramanujan_mode = (PRUNE_METHOD == 'ramanujan_init')
    if args.functional_calibrate and teacher_model is not None:
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

    if PRUNE_METHOD in {'random_init', 'ramanujan_init'}:
        pass
    elif PRUNE_METHOD in {'spectral_quant', 'spectral_path'}:
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

    if use_const_projector:
        phase1_beta_desc = 'fixed 0'
    elif ramanujan_mode:
        phase1_beta_desc = (f'auto [{RAM_P1_BETA_INIT:.1e}~{RAM_P1_BETA_MAX:.1e}] '
                            f'warmup={RAM_P1_WARMUP}')
    else:
        phase1_beta_desc = f'auto [{PHASE1_BETA_INIT:.1e}~{PHASE1_BETA_MAX:.1e}]'

    print(f'\n[4/5] PHASE 1  Recovery  ({PHASE1_EPOCHS} epochs, '
          f'lr={phase1_lr_use:.1e}, beta={phase1_beta_desc} )')

    if use_const_projector:
        set_all_beta(model, 0.0)
        phase1_budget_ctrl = None
    elif ramanujan_mode:
        set_all_beta(model, RAM_P1_BETA_INIT)
        phase1_budget_ctrl = BetaOnlyBudgetController(
            target_ebops     = TARGET_EBOPS,
            margin           = RAM_P1_MARGIN,
            beta_init        = RAM_P1_BETA_INIT,
            beta_min         = RAM_P1_BETA_MIN,
            beta_max         = RAM_P1_BETA_MAX,
            adjust_factor    = RAM_P1_ADJUST,
            ema_alpha        = RAM_P1_EMA,
            warmup_epochs    = RAM_P1_WARMUP,
            max_change_ratio = RAM_P1_MAX_CHANGE,
            init_ebops       = post_prune_ebops,
        )
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
    if ramanujan_kqb_stabilizer is not None:
        phase1_callbacks.insert(2 if phase1_budget_ctrl is None else 3, ramanujan_kqb_stabilizer)
    if ramanujan_mask_enforcer is not None:
        insert_idx = 2 if phase1_budget_ctrl is None else 3
        if ramanujan_kqb_stabilizer is not None:
            insert_idx += 1
        phase1_callbacks.insert(insert_idx, ramanujan_mask_enforcer)

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
    # 显式展示 Phase2 beta 配置，Ramanujan 分支沿用 v2 调参
    if use_const_projector:
        phase2_beta_desc = 'fixed 0'
    elif ramanujan_mode:
        phase2_beta_desc = f'auto (Ramanujan v2, margin=±{int(RAM_P2_MARGIN*100)}%)'
    else:
        phase2_beta_desc = 'auto'
    print(f'\n[5/5] PHASE 2  Accuracy Maximization  ({PHASE2_EPOCHS} epochs, '
          f'lr={phase2_lr_use:.1e}, beta={phase2_beta_desc})')

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
        set_all_beta(model, phase1_final_beta)

        if ramanujan_mode:
            phase2_margin = RAM_P2_MARGIN
            phase2_adjust_factor = RAM_P2_ADJUST
            phase2_ema = RAM_P2_EMA
            phase2_beta_max_use = RAM_P2_BETA_MAX
            phase2_beta_min_use = RAM_P2_BETA_MIN
            phase2_warmup = RAM_P2_WARMUP
            phase2_max_change = RAM_P2_MAX_CHANGE
            phase2_init_ebops = TARGET_EBOPS
            print(f'  Inheriting Phase1 equilibrium beta: {phase1_final_beta:.2e} (Ramanujan v2 tuned)')
        else:
            phase2_margin = 0.05
            phase2_adjust_factor = 1.15
            phase2_ema = 0.15
            phase2_beta_max_use = PHASE2_BETA_MAX
            phase2_beta_min_use = PHASE2_BETA_MIN
            phase2_warmup = 0
            phase2_max_change = 0
            phase2_init_ebops = None
            if near_budget_preserve_case:
                phase2_margin = 0.08
                phase2_adjust_factor = 1.08
                phase2_ema = 0.20
                phase2_beta_max_use = min(PHASE2_BETA_MAX, 2e-4)
            print(f'  Inheriting Phase1 equilibrium beta: {phase1_final_beta:.2e}')

        budget_ctrl = BetaOnlyBudgetController(
            target_ebops     = TARGET_EBOPS,
            margin           = phase2_margin,
            beta_init        = phase1_final_beta,
            beta_min         = phase2_beta_min_use,
            beta_max         = phase2_beta_max_use,
            adjust_factor    = phase2_adjust_factor,
            ema_alpha        = phase2_ema,
            warmup_epochs    = phase2_warmup,
            max_change_ratio = phase2_max_change,
            init_ebops       = phase2_init_ebops,
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
    if ramanujan_mask_enforcer is not None:
        phase2_callbacks.insert(3 if budget_ctrl is not None else 2, ramanujan_mask_enforcer)

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
