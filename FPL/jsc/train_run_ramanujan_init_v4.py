"""
Ramanujan 图初始化 v4 — 只 target 位宽 + 连接复活
===================================================
相比 v2 的简化与改进：

v2 → v4 的改进
--------------
1. **只 target 位宽，不再校准 eBOPs**：
   - 直接用 --init_bw (1/2/3) 初始化活跃连接的 kq.b
   - 去掉 calibrate_kqb_to_target，eBOPs 完全由位宽自然决定
   - 训练中用轻量 beta 控制让 HGQ 自由调节（不追求精确 eBOPs 目标）

2. **连接复活 (SpectralGradientRevivalCallback)**：
   - 每隔 revival_interval epoch，检查谱条件（度不足的节点）
   - 按权重大小排序候选死连接，复活 top-k 并淘汰等量弱连接（swap-kill）
   - 保持 eBOPs 预算不变的同时修复拓扑退化

3. **Sweep over 1,2,3 bit**：
   python train_run_ramanujan_init_v4.py --sweep_bw 1,2,3

理论分析
--------
模型: input(16) → t1(64) → t2(64) → t3(32) → out(5)
Ramanujan 谱条件 d-regular graph: λ₂ ≤ 2√(d-1)

不同位宽下的初始 eBOPs 估算（活跃连接 × b_k × b_a≈3）:
  1-bit:  eBOPs ≈  (384+768+384+40) × 1 × 3 =  4728
  2-bit:  eBOPs ≈  (384+768+384+40) × 2 × 3 =  9456
  3-bit:  eBOPs ≈  (384+768+384+40) × 3 × 3 = 14184
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import argparse
import json
import time

import keras
import numpy as np
import tensorflow as tf

from data.data import get_data
import model.model as model_module  # noqa: F401
from model.model import get_model_hgq
from hgq.layers import QLayerBase
from hgq.utils.sugar import Dataset, FreeEBOPs, ParetoFront
from keras.callbacks import LearningRateScheduler
from utils.train_utils import cosine_decay_restarts_schedule, TrainingTraceToH5, BudgetAwareEarlyStopping
from utils.tf_device import get_tf_device
from utils.ramanujan_budget_utils import (
    ActivationBitsFixer,
    BetaOnlyBudgetController,
    KQBStabilizer,
    SpectralGradientRevivalCallback,
    compute_bw_aware_degree,
    apply_ramanujan_bw_init,
    _flatten_layers,
    _get_kq_var,
)
from utils.ramaujian_utils import RamanujanMaskEnforcer
from utils.topology_graph_plot_utils import TopologyGraphPlotter


# ══════════════════════════════════════════════════════════════════════════════
# 默认配置
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    init_bw         = 1,        # 活跃连接位宽 (1/2/3)

    # Phase 1: 恢复
    phase1_epochs   = 5000,
    phase1_lr       = 2e-3,
    phase1_lr_cycle = 2000,
    phase1_lr_mmul  = 0.9,
    phase1_beta_init = 5e-7,
    phase1_beta_min  = 1e-8,
    phase1_beta_max  = 5e-5,
    phase1_margin    = 0.25,    # 宽 margin: 不死追 eBOPs，让 HGQ 自由学

    # Phase 2: 精调
    phase2_epochs   = 10000,
    phase2_lr       = 5e-4,
    phase2_lr_cycle = 800,
    phase2_lr_mmul  = 0.95,
    phase2_beta_min  = 1e-7,
    phase2_beta_max  = 5e-4,
    phase2_margin    = 0.10,

    # Ramanujan 拓扑
    ram_multiplier  = 0.75,
    ram_min_degree  = 3,
    ram_budget_weight = 'capacity',

    # KQB Stabilizer
    kqb_hold_epochs    = 500,
    kqb_release_epochs = 1000,
    kqb_hold_strength  = 0.8,

    # 连接复活
    revival_enabled   = True,
    revival_interval  = 200,
    revival_max_per_layer = 8,
    revival_b_val     = None,   # None = 和 init_bw 相同
    revival_swap_kill  = False,
    revival_min_degree = 2,

    # Early stopping
    earlystop_patience = 5000,
)

# ── 命令行参数 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Ramanujan-init v4: bitwidth-only targeting + connection revival')
parser.add_argument('--init_bw',         type=int,   default=DEFAULT_CONFIG['init_bw'],
                    help='Active connection kernel bitwidth: 1, 2, or 3 (default: 1)')
parser.add_argument('--init_bw_a',       type=int,   default=None,
                    help='Activation bitwidth (default: same as init_bw)')
parser.add_argument('--phase1_epochs',   type=int,   default=DEFAULT_CONFIG['phase1_epochs'])
parser.add_argument('--phase2_epochs',   type=int,   default=DEFAULT_CONFIG['phase2_epochs'])
parser.add_argument('--phase1_lr',       type=float, default=DEFAULT_CONFIG['phase1_lr'])
parser.add_argument('--phase2_lr',       type=float, default=DEFAULT_CONFIG['phase2_lr'])
parser.add_argument('--ram_multiplier',  type=float, default=DEFAULT_CONFIG['ram_multiplier'])
parser.add_argument('--ram_min_degree',  type=int,   default=DEFAULT_CONFIG['ram_min_degree'])
parser.add_argument('--mask_hold',       type=int,   default=None,
                    help='Epochs to hold mask (default: phase1_epochs)')
parser.add_argument('--mask_fade',       type=int,   default=3000,
                    help='Fade epochs for mask release (default: 3000)')
parser.add_argument('--sweep_bw',        type=str,   default=None,
                    help='Comma-separated bitwidths for sweep, e.g. "1,2,3"')
parser.add_argument('--seed',            type=int,   default=42)
parser.add_argument('--init_only',       action='store_true',
                    help='Only do Ramanujan init + save weights + plot graph, skip training')
parser.add_argument('--output_dir',      type=str, default='',
                    help='Optional output directory for a single run')
# 连接复活参数
parser.add_argument('--revival',         action=argparse.BooleanOptionalAction,
                    default=DEFAULT_CONFIG['revival_enabled'],
                    help='Enable/disable connection revival (default: enabled)')
parser.add_argument('--revival_interval', type=int, default=DEFAULT_CONFIG['revival_interval'],
                    help='Epochs between revival attempts (default: 200)')
parser.add_argument('--revival_max_per_layer', type=int,
                    default=DEFAULT_CONFIG['revival_max_per_layer'],
                    help='Max connections revived per layer per attempt (default: 8)')
parser.add_argument('--revival_b_val',   type=float, default=None,
                    help='kq.b value for revived connections (default: same as init_bw)')
parser.add_argument('--swap_kill',       action=argparse.BooleanOptionalAction,
                    default=DEFAULT_CONFIG['revival_swap_kill'],
                    help='Swap-kill: revive dead + kill weak to keep eBOPs stable (default: on)')
args, _ = parser.parse_known_args()

input_folder = 'data/dataset.h5'
batch_size   = 33200


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
        if len(active) > 0:
            print(f'  [bk_stats {label}]  '
                  f'mean={arr.mean():.3f}  std={arr.std():.3f}  '
                  f'min={arr.min():.3f}  max={arr.max():.3f}  '
                  f'dead={len(dead)}/{len(arr)} ({100*len(dead)/len(arr):.1f}%)  '
                  f'active_mean={active.mean():.3f}  active_std={active.std():.3f}')
        else:
            print(f'  [bk_stats {label}]  ALL DEAD!')


def compute_model_ebops(mdl, sample_input) -> float:
    from keras import ops
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
    print('\n  ┌─ Ramanujan Topology ────────────────────────────────────────┐')
    for layer in _flatten_layers(mdl):
        mask = getattr(layer, 'ramanujan_mask', None)
        if mask is None:
            continue
        m = mask.numpy()
        active = int(m.sum())
        total = int(m.size)
        sparsity = 1.0 - m.mean()
        if m.ndim == 2:
            row_deg = m.sum(axis=1)
            col_deg = m.sum(axis=0)
            print(f'  │ {layer.name:20s}  {active:4d}/{total:5d}  sparse={sparsity:.1%}  '
                  f'row=[{row_deg.min():.0f},{row_deg.max():.0f}]  '
                  f'col=[{col_deg.min():.0f},{col_deg.max():.0f}]')
        else:
            print(f'  │ {layer.name:20s}  {active:4d}/{total:5d}  sparse={sparsity:.1%}')
    print('  └─────────────────────────────────────────────────────────────┘\n')


# ══════════════════════════════════════════════════════════════════════════════
# 单次训练（按位宽目标）
# ══════════════════════════════════════════════════════════════════════════════

def run_single_bw(init_bw: int, cfg: dict):
    """对单个位宽目标执行完整的 Ramanujan 初始化 + 连接复活 + 两阶段训练。

    与 v2 的核心区别：
      - 不做 eBOPs 校准，活跃连接直接赋 kq.b = init_bw
      - eBOPs 是位宽的自然结果，不是显式目标
      - 训练中通过连接复活保持拓扑健康
    """

    np.random.seed(cfg.get('seed', 42))
    random.seed(cfg.get('seed', 42))

    phase1_epochs = cfg['phase1_epochs']
    phase2_epochs = cfg['phase2_epochs']
    total_epochs  = phase1_epochs + phase2_epochs

    output_folder = cfg.get('output_dir', '') or f'results/ramanujan_v4_{init_bw}bit/'
    os.makedirs(output_folder, exist_ok=True)
    device = get_tf_device()

    revival_b_val = cfg.get('revival_b_val') or float(init_bw)

    print('=' * 72)
    print(f'  Ramanujan-Init v4  |  init_bw = {init_bw}-bit')
    print(f'  Phase 1: {phase1_epochs} epochs, lr={cfg["phase1_lr"]:.1e}, margin=±{cfg["phase1_margin"]*100:.0f}%')
    print(f'  Phase 2: {phase2_epochs} epochs, lr={cfg["phase2_lr"]:.1e}, margin=±{cfg["phase2_margin"]*100:.0f}%')
    print(f'  Ramanujan: mult={cfg["ram_multiplier"]}, min_degree={cfg["ram_min_degree"]}')
    print(f'  Revival: enabled={cfg["revival_enabled"]}  interval={cfg["revival_interval"]}  '
          f'max_per_layer={cfg["revival_max_per_layer"]}  swap_kill={cfg["revival_swap_kill"]}  '
          f'b_val={revival_b_val}')
    print(f'  Output: {output_folder}')
    print('=' * 72)

    t0 = time.time()

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/5] Loading dataset...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src='openml')
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val   = Dataset(X_val,   y_val,   batch_size, device)
    _sample = tf.constant(X_train[:min(2048, len(X_train))], dtype=tf.float32)
    _sample_y = tf.constant(y_train[:min(2048, len(y_train))], dtype=tf.int64)

    # ── 2. 构建模型 ──────────────────────────────────────────────────────────
    # 用 init_bw 作为模型构造时的初始位宽
    init_bw_a = cfg.get('init_bw_a', None) or init_bw  # 默认与 kernel 位宽一致
    print(f'\n[2/5] Building HGQ model (init_bw_k={init_bw}, init_bw_a={init_bw_a})...')
    model = get_model_hgq(init_bw_k=init_bw, init_bw_a=init_bw_a)

    init_ebops = compute_model_ebops(model, _sample)
    print(f'  Random-init eBOPs: {init_ebops:.0f}')

    # ── 3. Ramanujan 初始化（只设位宽，不做 eBOPs 校准）─────────────────────
    # 使用一个参考 target_ebops 来决定 Ramanujan 拓扑的度数，
    # 但不再对 kq.b 做校准缩放
    ref_target_ebops = init_ebops * 0.5  # 参考值，只影响拓扑度计算
    print(f'\n[3/5] Ramanujan BW-aware init → {init_bw}-bit (no eBOPs calibration)...')

    per_layer_degree, _per_layer_bk = compute_bw_aware_degree(
        model,
        target_ebops=ref_target_ebops,
        b_a_init=float(init_bw_a),
        b_k_min=0.5,
        b_k_max=8.0,
        multiplier=cfg['ram_multiplier'],
        min_degree=cfg['ram_min_degree'],
        budget_weight=cfg['ram_budget_weight'],
        verbose=True,
    )

    # 直接用 init_bw 作为所有活跃连接的位宽 — 不校准
    INIT_BK = float(init_bw)
    apply_ramanujan_bw_init(
        model,
        per_layer_degree=per_layer_degree,
        per_layer_bk={name: INIT_BK for name in per_layer_degree},
        seed=cfg.get('seed', 42),
        pruned_frac_bits=0.0,
        pruned_int_bits=0.0,
        active_int_bits=1.0,
        also_zero_kernel=True,
        verbose=True,
    )

    raw_ebops = compute_model_ebops(model, _sample)
    print(f'  After Ramanujan init (b_k={init_bw}): eBOPs = {raw_ebops:.0f}')
    print(f'  (eBOPs is a natural consequence of {init_bw}-bit, not a target)')
    print_ramanujan_topology_stats(model)
    print_bk_stats(model, f'{init_bw}-bit init')

    # ── 3.5 保存初始化权重 + 绘制拓扑图 ──────────────────────────────────────
    init_model_path = os.path.join(output_folder, f'ramanujan_init_{init_bw}bit.keras')
    model.save(init_model_path)
    print(f'  Init weights saved: {init_model_path}')

    topo_plot_dir = os.path.join(output_folder, 'init_graph')
    plotter = TopologyGraphPlotter(symmetric_topology_plot=False, mirror_edges=False)
    topo_outputs = plotter.plot_from_keras(
        keras_path=init_model_path,
        output_dir=topo_plot_dir,
    )
    print(f'  Ramanujan topology matrix: {topo_outputs["matrix_path"]}')
    print(f'  Ramanujan topology circle: {topo_outputs["circle_path"]}')

    if cfg.get('init_only', False):
        print(f'\n  [init_only] Done. Skipping training.')
        return {
            'init_bw':       init_bw,
            'init_ebops':    raw_ebops,
            'init_model':    init_model_path,
            'matrix_plot':   str(topo_outputs['matrix_path']),
            'circle_plot':   str(topo_outputs['circle_path']),
        }

    # ── 4. Phase 1: 恢复训练 + 连接复活 ──────────────────────────────────────
    print(f'\n[4/5] Phase 1: Recovery + Revival  ({phase1_epochs} epochs)')

    set_all_beta(model, cfg['phase1_beta_init'])

    # KQB Stabilizer: 锚定 kq.b 防止初期漂移
    kqb_stab = KQBStabilizer(
        hold_epochs=cfg['kqb_hold_epochs'],
        release_epochs=cfg['kqb_release_epochs'],
        hold_strength=cfg['kqb_hold_strength'],
    )
    kqb_stab.capture_snapshot(model)

    # Budget controller: 轻量控制，用较宽 margin
    # 以初始 eBOPs 为参考目标（不强制压缩/扩张，只防止极端漂移）
    phase1_budget_ctrl = BetaOnlyBudgetController(
        target_ebops    = raw_ebops,  # 以初始化后的自然 eBOPs 为目标
        margin          = cfg['phase1_margin'],
        beta_init       = cfg['phase1_beta_init'],
        beta_min        = cfg['phase1_beta_min'],
        beta_max        = cfg['phase1_beta_max'],
        adjust_factor   = 1.15,
        ema_alpha       = 0.25,
        warmup_epochs   = 100,
        max_change_ratio = 1.5,
        init_ebops      = raw_ebops,
    )

    # 连接复活回调
    # 用小批量训练数据做梯度探针，决定复活优先级
    _revival_probe_n = min(512, len(X_train))
    _revival_probe_x = tf.constant(X_train[:_revival_probe_n], dtype=tf.float32)
    _revival_probe_y = tf.constant(y_train[:_revival_probe_n], dtype=tf.int64)

    revival_cb = None
    if cfg['revival_enabled']:
        revival_cb = SpectralGradientRevivalCallback(
            target_ebops=raw_ebops,
            probe_x=_revival_probe_x,
            probe_y=_revival_probe_y,
            min_degree=cfg['revival_min_degree'],
            revival_b_val=revival_b_val,
            max_revival_per_layer=cfg['revival_max_per_layer'],
            revival_interval=cfg['revival_interval'],
            ebops_deficit_threshold=0.20,
            dead_fraction_threshold=0.85,
            grad_min_threshold=0.0,
            cool_down=cfg['revival_interval'] // 2,
            swap_kill=cfg['revival_swap_kill'],
        )
        print(f'  [Revival] enabled: interval={cfg["revival_interval"]}  '
              f'max_per_layer={cfg["revival_max_per_layer"]}  '
              f'swap_kill={cfg["revival_swap_kill"]}  '
              f'b_val={revival_b_val}')

    # Mask enforcer: 保护 Ramanujan 拓扑
    # min_active_frac_bits 防止活跃连接被 beta 正则化压死
    # 对 1-bit init，floor=0.5 确保 round(kq.b) >= 1
    min_active_floor = max(0.5, float(init_bw) * 0.5)
    mask_hold = cfg.get('mask_hold', phase1_epochs)
    mask_fade = cfg.get('mask_fade', 3000)
    mask_enforcer = RamanujanMaskEnforcer(
        release_epoch=mask_hold,
        fade_epochs=mask_fade,
        min_active_frac_bits=min_active_floor,
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

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg['phase1_lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    # Activation bits fixer: 固定激活位宽，防止训练中漂移
    act_bits_fixer = ActivationBitsFixer(b_a_fixed=float(init_bw_a), start_epoch=0)

    phase1_callbacks = [
        ebops_cb,
        phase1_budget_ctrl,
        kqb_stab,
        mask_enforcer,
        act_bits_fixer,
        pareto_cb,
        make_lr_scheduler(cfg['phase1_lr'], cfg['phase1_lr_cycle'], cfg['phase1_lr_mmul']),
        trace_cb,
    ]
    if revival_cb is not None:
        # 复活回调放 mask_enforcer 之后，让复活的连接立刻受 mask 保护
        phase1_callbacks.insert(4, revival_cb)

    print(f'  Starting Phase 1...')
    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=phase1_epochs,
        callbacks=phase1_callbacks,
        verbose=1,
    )

    phase1_ebops = compute_model_ebops(model, _sample)
    phase1_res = model.evaluate(dataset_val, verbose=0)
    phase1_beta = phase1_budget_ctrl.beta_current
    print(f'\n  Phase 1 done:  val_acc={phase1_res[1]:.4f}  eBOPs={phase1_ebops:.0f}  '
          f'beta={phase1_beta:.2e}')
    print_bk_stats(model, 'end of Phase 1')

    # ── 5. Phase 2: 精调 + 连接复活继续 ──────────────────────────────────────
    print(f'\n[5/5] Phase 2: Accuracy maximization + Revival  ({phase2_epochs} epochs)')
    print(f'  Recompiling with fresh Adam optimizer...')

    phase2_budget_ctrl = BetaOnlyBudgetController(
        target_ebops    = phase1_ebops,  # 以 Phase 1 结束时的 eBOPs 为目标
        margin          = cfg['phase2_margin'],
        beta_init       = phase1_beta,
        beta_min        = cfg['phase2_beta_min'],
        beta_max        = cfg['phase2_beta_max'],
        adjust_factor   = 1.15,
        ema_alpha       = 0.15,
        warmup_epochs   = 0,
        max_change_ratio = 1.5,
        init_ebops      = phase1_ebops,
    )

    # Phase 2 也可以继续复活
    revival_cb_p2 = None
    if cfg['revival_enabled']:
        revival_cb_p2 = SpectralGradientRevivalCallback(
            target_ebops=phase1_ebops,
            probe_x=_revival_probe_x,
            probe_y=_revival_probe_y,
            min_degree=cfg['revival_min_degree'],
            revival_b_val=revival_b_val,
            max_revival_per_layer=cfg['revival_max_per_layer'],
            revival_interval=cfg['revival_interval'] * 2,  # Phase 2 间隔更大
            ebops_deficit_threshold=0.15,                   # Phase 2 触发更保守
            dead_fraction_threshold=0.90,
            grad_min_threshold=0.0,
            cool_down=cfg['revival_interval'],
            swap_kill=cfg['revival_swap_kill'],
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg['phase2_lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    earlystop_budget = phase1_ebops * 2.0  # 宽松: 不以 eBOPs 为硬约束
    early_stop_cb = BudgetAwareEarlyStopping(
        ebops_budget         = earlystop_budget,
        patience             = cfg['earlystop_patience'],
        min_delta            = 5e-5,
        min_epoch            = phase1_epochs + 1000,
        restore_best_weights = True,
    )

    phase2_callbacks = [
        ebops_cb,
        phase2_budget_ctrl,
        mask_enforcer,
        act_bits_fixer,
        pareto_cb,
        make_lr_scheduler(cfg['phase2_lr'], cfg['phase2_lr_cycle'], cfg['phase2_lr_mmul']),
        trace_cb,
        early_stop_cb,
    ]
    if revival_cb_p2 is not None:
        phase2_callbacks.insert(3, revival_cb_p2)

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        initial_epoch=phase1_epochs,
        epochs=total_epochs,
        callbacks=phase2_callbacks,
        verbose=1,
    )

    # ── 最终统计 ──────────────────────────────────────────────────────────────
    final_ebops = compute_model_ebops(model, _sample)
    final_res = model.evaluate(dataset_val, verbose=0)
    elapsed = time.time() - t0
    print_bk_stats(model, 'final')
    print_ramanujan_topology_stats(model)

    result = {
        'init_bw':        init_bw,
        'init_ebops':     raw_ebops,
        'final_ebops':    final_ebops,
        'final_val_acc':  float(final_res[1]),
        'final_val_loss': float(final_res[0]),
        'phase1_val_acc': float(phase1_res[1]),
        'phase1_ebops':   phase1_ebops,
        'elapsed_sec':    elapsed,
        'revival_enabled': cfg['revival_enabled'],
        'config':         {k: v for k, v in cfg.items()},
    }

    # 从 trace 读取最佳 accuracy
    import h5py
    trace_path = os.path.join(output_folder, 'training_trace.h5')
    if os.path.exists(trace_path):
        with h5py.File(trace_path, 'r') as f:
            va = f['val_accuracy'][:]
            eb = f['ebops'][:]
            result['best_val_acc'] = float(va.max())
            result['best_epoch']   = int(va.argmax())
            result['best_ebops']   = float(eb[va.argmax()])

    print(f'\n  {"=" * 60}')
    print(f'  init_bw={init_bw}  init_ebops={raw_ebops:.0f}  final_ebops={final_ebops:.0f}  '
          f'val_acc={final_res[1]:.4f}  best_acc={result.get("best_val_acc", 0):.4f}  '
          f'time={elapsed:.0f}s')
    print(f'  {"=" * 60}')

    # 保存结果摘要
    summary_path = os.path.join(output_folder, 'result_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Summary saved: {summary_path}')

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # 构建配置 dict
    cfg = dict(DEFAULT_CONFIG)
    cfg['seed'] = args.seed
    cfg['init_bw']        = args.init_bw
    cfg['init_bw_a']      = args.init_bw_a  # None = same as init_bw
    cfg['phase1_epochs']  = args.phase1_epochs
    cfg['phase2_epochs']  = args.phase2_epochs
    cfg['phase1_lr']      = args.phase1_lr
    cfg['phase2_lr']      = args.phase2_lr
    cfg['ram_multiplier'] = args.ram_multiplier
    cfg['ram_min_degree'] = args.ram_min_degree
    cfg['mask_hold']      = args.mask_hold if args.mask_hold is not None else args.phase1_epochs
    cfg['mask_fade']      = args.mask_fade
    cfg['init_only']      = args.init_only
    cfg['output_dir']     = args.output_dir
    cfg['revival_enabled']       = args.revival
    cfg['revival_interval']      = args.revival_interval
    cfg['revival_max_per_layer'] = args.revival_max_per_layer
    cfg['revival_b_val']         = args.revival_b_val
    cfg['revival_swap_kill']     = args.swap_kill

    if args.sweep_bw:
        # ── Sweep 模式：批量测试不同位宽 ─────────────────────────────────────
        bws = [int(x.strip()) for x in args.sweep_bw.split(',')]
        print(f'\n{"#" * 72}')
        print(f'  Ramanujan v4 — Bitwidth SWEEP:  {bws}')
        print(f'  Revival: enabled={cfg["revival_enabled"]}  swap_kill={cfg["revival_swap_kill"]}')
        print(f'{"#" * 72}\n')

        all_results = []
        for bw in bws:
            cfg_copy = dict(cfg)
            cfg_copy['output_dir'] = ''  # 让每次运行自动命名
            result = run_single_bw(bw, cfg_copy)
            all_results.append(result)

            # 打印累积进度
            print(f'\n  ── Sweep progress ──')
            for r in sorted(all_results, key=lambda x: x['init_bw']):
                print(f'    {r["init_bw"]}-bit:  init_ebops={r["init_ebops"]:.0f}  '
                      f'final_ebops={r.get("final_ebops", 0):.0f}  '
                      f'best_acc={r.get("best_val_acc", 0):.4f}')
            print()

        # 保存 sweep 汇总
        sweep_summary_path = 'results/ramanujan_v4_bw_sweep_summary.json'
        os.makedirs('results', exist_ok=True)
        with open(sweep_summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nSweep summary saved: {sweep_summary_path}')

        # 最终表格
        print(f'\n{"=" * 72}')
        print(f'  Ramanujan v4 Bitwidth Sweep — Final Results')
        print(f'  {"bw":>4s}  {"init_eb":>8s}  {"final_eb":>8s}  {"best_acc":>8s}  {"time":>6s}')
        print(f'  {"-" * 44}')
        for r in sorted(all_results, key=lambda x: x['init_bw']):
            print(f'  {r["init_bw"]:4d}  '
                  f'{r["init_ebops"]:8.0f}  '
                  f'{r.get("final_ebops", 0):8.0f}  '
                  f'{r.get("best_val_acc", 0):8.4f}  '
                  f'{r.get("elapsed_sec", 0):5.0f}s')
        print(f'{"=" * 72}')

    else:
        # ── 单次运行 ─────────────────────────────────────────────────────────
        run_single_bw(cfg['init_bw'], cfg)
