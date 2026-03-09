"""
Ramanujan 图初始化 v3 — 三阶段训练：先探索再压缩
================================================
v2 → v3 的核心改进：前期不压缩 eBOPs，充分在稀疏拓扑下探索精度空间，
再逐步压缩到目标 eBOPs，最后精调。

三阶段策略
----------
Phase 0 (拓扑探索):
  - 无 budget 压力（beta ≈ 0），mask 固定（Ramanujan 拓扑完全保护）
  - 高 LR (2e-3)，让模型在固定稀疏拓扑下充分收敛，找到精度上界
  - 目的：在剪枝后的拓扑上充分训练，避免后续压缩从未收敛的点出发

Phase 1 (eBOPs 压缩):
  - 开启 BudgetController，逐步将 eBOPs 压缩至 target
  - mask 从 hold 开始, fade 阶段逐渐放开
  - 中等 LR (1e-3)，确保精度不在压缩过程中崩溃

Phase 2 (精调):
  - 低 LR (5e-4)，窄 margin (±5%)，mask 完全释放，Adam 重置
  - 精确维持 target eBOPs，最大化精度

与 v2 的关键区别
----------------
v2: Phase1 从 calibrated start 立即开始压缩 → 模型权重尚未适应稀疏拓扑
v3: Phase0 先在稀疏拓扑下充分训练 → Phase1 从已收敛的好初始点开始压缩

理论依据
--------
稀疏 Ramanujan 拓扑的表达能力需要足够的训练时间才能被充分挖掘。
在量化位宽尚未压缩时（b_k 较高）参数充足，模型可以学到更好的权重分布，
这些权重分布在后续 eBOPs 压缩时提供了更好的起点，类似于渐进式剪枝中
"先训好再剪"的策略。
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
    BetaOnlyBudgetController,
    KQBStabilizer,
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
    target_ebops    = 3162,
    init_bw_k       = 3,

    # Phase 0: 拓扑探索（新增，无 budget 压力）
    phase0_epochs   = 3000,
    phase0_lr       = 2e-3,
    phase0_lr_cycle = 1000,
    phase0_lr_mmul  = 0.9,

    # Phase 1: eBOPs 压缩（对应 v2 Phase 1，但起点已充分训练）
    phase1_epochs   = 4000,
    phase1_lr       = 1e-3,
    phase1_lr_cycle = 1500,
    phase1_lr_mmul  = 0.92,
    phase1_beta_init = 5e-7,
    phase1_beta_min  = 1e-8,
    phase1_beta_max  = 2e-4,
    phase1_margin    = 0.20,

    # Phase 2: 精调（对齐 v2 Phase 2）
    phase2_epochs   = 8000,
    phase2_lr       = 5e-4,
    phase2_lr_cycle = 800,
    phase2_lr_mmul  = 0.95,
    phase2_beta_min  = 1e-7,
    phase2_beta_max  = 5e-4,
    phase2_margin    = 0.05,

    # Ramanujan
    ram_multiplier  = 1.5,
    ram_min_degree  = 4,
    ram_budget_weight = 'capacity',

    # KQB Stabilizer（Phase 1 用，防止压缩过程中 b_k 暴跌）
    kqb_hold_epochs    = 50,
    kqb_release_epochs = 200,
    kqb_hold_strength  = 0.8,

    # Early stopping
    earlystop_patience = 5000,
)

# ── 命令行参数 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Ramanujan-init v3: Explore → Compress → Finetune'
)
parser.add_argument('--target_ebops',    type=float, default=DEFAULT_CONFIG['target_ebops'])
parser.add_argument('--phase0_epochs',   type=int,   default=DEFAULT_CONFIG['phase0_epochs'])
parser.add_argument('--phase1_epochs',   type=int,   default=DEFAULT_CONFIG['phase1_epochs'])
parser.add_argument('--phase2_epochs',   type=int,   default=DEFAULT_CONFIG['phase2_epochs'])
parser.add_argument('--phase0_lr',       type=float, default=DEFAULT_CONFIG['phase0_lr'])
parser.add_argument('--phase1_lr',       type=float, default=DEFAULT_CONFIG['phase1_lr'])
parser.add_argument('--phase2_lr',       type=float, default=DEFAULT_CONFIG['phase2_lr'])
parser.add_argument('--init_bw_k',       type=int,   default=DEFAULT_CONFIG['init_bw_k'])
parser.add_argument('--ram_multiplier',  type=float, default=DEFAULT_CONFIG['ram_multiplier'])
parser.add_argument('--ram_min_degree',  type=int,   default=DEFAULT_CONFIG['ram_min_degree'])
parser.add_argument('--mask_hold',       type=int,   default=None,
                    help='Epochs (absolute) to hold mask rigid; default = phase0_epochs + phase1_epochs')
parser.add_argument('--mask_fade',       type=int,   default=2000,
                    help='Fade epochs for mask release (default: 2000, spanning Phase 1 → Phase 2)')
parser.add_argument('--sweep',           type=str,   default=None,
                    help='Comma-separated eBOPs targets for sweep')
parser.add_argument('--seed',            type=int,   default=42)
parser.add_argument('--init_only',       action='store_true',
                    help='Only do Ramanujan init + save weights + plot graph, skip training')
parser.add_argument('--skip_calib',      action='store_true',
                    help='Skip calibration: start from high b_k')
parser.add_argument('--calib_multiplier', type=float, default=1.0,
                    help='Calibrate to target*multiplier eBOPs')
parser.add_argument('--output_dir',      type=str, default='',
                    help='Optional output directory for a single-target run')
args, _ = parser.parse_known_args()

input_folder = 'data/dataset.h5'
batch_size   = 33200


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数（与 v2 相同）
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


def make_lr_scheduler(lr_init, cycle, mmul, offset=0):
    """offset: starting epoch offset so the schedule is phase-local."""
    fn = cosine_decay_restarts_schedule(lr_init, cycle, t_mul=1.0, m_mul=mmul,
                                        alpha=1e-6, alpha_steps=50)
    return LearningRateScheduler(lambda epoch: fn(epoch - offset))


def calibrate_kqb_to_target(model, sample_input, target_ebops, snapshots=None):
    """校准 kq.b 使实际 eBOPs ≈ target_ebops。"""
    if snapshots is None:
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

        rng = np.random.RandomState(42)
        for b_var, b_snap in snapshots:
            active = b_snap > 0.05
            noise = rng.uniform(0.8, 1.2, size=b_snap.shape).astype(np.float32)
            b_noisy = np.where(active, b_snap * noise, b_snap)
            b_var.assign(b_noisy.astype(np.float32))
        snapshots = [(bv, bv.numpy().copy()) for bv, _ in snapshots]

    def _apply_scale(s):
        for b_var, b_snap in snapshots:
            active = b_snap > 0.05
            b_new = np.where(active, (b_snap * s).clip(0.1, 8.0), b_snap)
            b_var.assign(b_new.astype(np.float32))

    def _measure(s):
        _apply_scale(s)
        return compute_model_ebops(model, sample_input)

    lo, hi = 0.005, 5.0
    noisy_ebops = _measure(1.0)
    best_scale, best_ebops, best_err = 1.0, noisy_ebops, abs(noisy_ebops - target_ebops)

    for i in range(50):
        mid = (lo + hi) / 2.0
        e = _measure(mid)
        err = abs(e - target_ebops)
        if err < best_err:
            best_scale, best_ebops, best_err = mid, e, err
        if err / max(target_ebops, 1) < 0.03:
            best_scale, best_ebops = mid, e
            break
        if e > target_ebops:
            hi = mid
        else:
            lo = mid

    _apply_scale(best_scale)
    actual = compute_model_ebops(model, sample_input)
    return actual, snapshots


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
# 单次训练
# ══════════════════════════════════════════════════════════════════════════════

def run_single_target(target_ebops: float, cfg: dict):
    """三阶段 Ramanujan 训练: Phase0 探索 → Phase1 压缩 → Phase2 精调。"""

    np.random.seed(cfg.get('seed', 42))
    random.seed(cfg.get('seed', 42))

    phase0_epochs = cfg['phase0_epochs']
    phase1_epochs = cfg['phase1_epochs']
    phase2_epochs = cfg['phase2_epochs']
    total_epochs  = phase0_epochs + phase1_epochs + phase2_epochs

    output_folder = cfg.get('output_dir', '') or f'results/ramanujan_v3_{int(target_ebops)}/'
    if not cfg.get('output_dir', ''):
        if cfg.get('skip_calib', False):
            output_folder = f'results/ramanujan_v3_{int(target_ebops)}_skipcalib/'
        elif cfg.get('calib_multiplier', 1.0) > 1.01:
            cm = cfg['calib_multiplier']
            output_folder = f'results/ramanujan_v3_{int(target_ebops)}_cm{cm:.1f}/'
    os.makedirs(output_folder, exist_ok=True)
    device = get_tf_device()

    skip_calib = cfg.get('skip_calib', False)

    print('=' * 72)
    print(f'  Ramanujan-Init v3  |  target_ebops = {target_ebops:.0f}')
    print(f'  ── v3 三阶段策略（探索 → 压缩 → 精调）──')
    print(f'  Phase 0 (探索): {phase0_epochs} ep, lr={cfg["phase0_lr"]:.1e}, NO budget pressure')
    print(f'  Phase 1 (压缩): {phase1_epochs} ep, lr={cfg["phase1_lr"]:.1e}, margin=±{cfg["phase1_margin"]*100:.0f}%')
    print(f'  Phase 2 (精调): {phase2_epochs} ep, lr={cfg["phase2_lr"]:.1e}, margin=±{cfg["phase2_margin"]*100:.0f}%')
    print(f'  Ramanujan: mult={cfg["ram_multiplier"]}, min_degree={cfg["ram_min_degree"]}')
    print(f'  Output: {output_folder}')
    print('=' * 72)

    t0 = time.time()

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/6] Loading dataset...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src='openml')
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val   = Dataset(X_val,   y_val,   batch_size, device)
    _sample = tf.constant(X_train[:min(2048, len(X_train))], dtype=tf.float32)

    # ── 2. 构建模型 ──────────────────────────────────────────────────────────
    print(f'\n[2/6] Building HGQ model (init_bw_k={cfg["init_bw_k"]})...')
    model = get_model_hgq(init_bw_k=cfg['init_bw_k'], init_bw_a=3)

    init_ebops = compute_model_ebops(model, _sample)
    print(f'  Random-init eBOPs: {init_ebops:.0f}')

    # ── 3. Ramanujan 初始化 + 校准 ──────────────────────────────────────────
    print(f'\n[3/6] Ramanujan BW-aware init → calibrate to {target_ebops:.0f} eBOPs...')

    per_layer_degree, per_layer_bk = compute_bw_aware_degree(
        model,
        target_ebops=target_ebops,
        b_a_init=3.0,
        b_k_min=0.5,
        b_k_max=8.0,
        multiplier=cfg['ram_multiplier'],
        min_degree=cfg['ram_min_degree'],
        budget_weight=cfg['ram_budget_weight'],
        verbose=True,
    )

    # 4-bit 初始化（高起点，给 Phase 0 探索更多容量）
    INIT_BK = 4.0
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
    print(f'  After topology (b_k={INIT_BK}): eBOPs = {raw_ebops:.0f}')
    print_ramanujan_topology_stats(model)

    # 保存拓扑图
    pre_calib_path = os.path.join(output_folder, 'ramanujan_init_pre_calib.keras')
    model.save(pre_calib_path)
    print(f'  Pre-calibration weights saved: {pre_calib_path}')

    topo_plot_dir = os.path.join(output_folder, 'init_graph')
    plotter = TopologyGraphPlotter(symmetric_topology_plot=False, mirror_edges=False)
    topo_outputs = plotter.plot_from_keras(
        keras_path=pre_calib_path,
        output_dir=topo_plot_dir,
    )
    print(f'  Ramanujan topology matrix: {topo_outputs["matrix_path"]}')
    print(f'  Ramanujan topology circle: {topo_outputs["circle_path"]}')

    # 校准 b_k（Phase 0 探索时从 calibrated level 出发，而不是从高位宽压）
    if skip_calib:
        print(f'  [skip_calib] Starting from b_k={INIT_BK}, eBOPs={raw_ebops:.0f}.')
        start_ebops = raw_ebops
    else:
        calib_mult = cfg.get('calib_multiplier', 1.0)
        calib_target = target_ebops * calib_mult
        calib_ebops, _ = calibrate_kqb_to_target(model, _sample, calib_target)
        print(f'  After calibration: eBOPs = {calib_ebops:.0f}  '
              f'(calib_target = {calib_target:.0f}, final_target = {target_ebops:.0f})')
        print_bk_stats(model, 'calibrated')
        start_ebops = calib_ebops

    # 保存初始化权重
    init_model_path = os.path.join(output_folder, 'ramanujan_init.keras')
    model.save(init_model_path)
    print(f'  Init weights saved: {init_model_path}')

    if cfg.get('init_only', False):
        print(f'\n  [init_only] Done. Skipping training.')
        return {
            'target_ebops':    target_ebops,
            'start_ebops':     start_ebops,
            'init_model':      init_model_path,
            'pre_calib_model': pre_calib_path,
            'matrix_plot':     str(topo_outputs['matrix_path']),
            'circle_plot':     str(topo_outputs['circle_path']),
        }

    # ── mask enforcer 贯穿三阶段 ──────────────────────────────────────────
    # mask_hold 默认涵盖 Phase 0 全程 + Phase 1 全程，Phase 2 开始后 fade
    mask_hold = cfg.get('mask_hold', phase0_epochs + phase1_epochs)
    mask_fade = cfg.get('mask_fade', 2000)
    use_floor = skip_calib or cfg.get('calib_multiplier', 1.0) > 1.01
    mask_enforcer = RamanujanMaskEnforcer(
        release_epoch=mask_hold,
        fade_epochs=mask_fade,
        min_active_frac_bits=1.0 if use_floor else None,
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

    # ── 4. Phase 0: 拓扑探索（无 budget 压力）──────────────────────────────
    print(f'\n[4/6] Phase 0: Topology Exploration  ({phase0_epochs} epochs)')
    print(f'  ▸ beta ≈ 0, mask固定, 让模型在稀疏拓扑下充分收敛')
    print(f'  ▸ 当前 eBOPs={start_ebops:.0f}, 不压缩到 {target_ebops:.0f}')

    # beta 设为近零：无位宽压力
    set_all_beta(model, 1e-12)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg['phase0_lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    phase0_callbacks = [
        ebops_cb,
        # ★ 无 BudgetController —— Phase 0 不压缩 eBOPs ★
        mask_enforcer,        # mask 完全固定（release_epoch 远未到）
        pareto_cb,
        make_lr_scheduler(cfg['phase0_lr'], cfg['phase0_lr_cycle'], cfg['phase0_lr_mmul'],
                          offset=0),
        trace_cb,
    ]

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=phase0_epochs,
        callbacks=phase0_callbacks,
        verbose=1,
    )

    phase0_ebops = compute_model_ebops(model, _sample)
    phase0_res = model.evaluate(dataset_val, verbose=0)
    print(f'\n  Phase 0 done:  val_acc={phase0_res[1]:.4f}  eBOPs={phase0_ebops:.0f}  (target={target_ebops:.0f})')
    print_bk_stats(model, 'end of Phase 0')

    # ── 5. Phase 1: eBOPs 压缩 ──────────────────────────────────────────────
    print(f'\n[5/6] Phase 1: eBOPs Compression  ({phase1_epochs} epochs)')
    print(f'  ▸ 从 {phase0_ebops:.0f} 逐步压缩至 {target_ebops:.0f} eBOPs')

    # KQB Stabilizer：防止压缩初期 b_k 暴跌
    if not skip_calib:
        kqb_stab = KQBStabilizer(
            hold_epochs=cfg['kqb_hold_epochs'],
            release_epochs=cfg['kqb_release_epochs'],
            hold_strength=cfg['kqb_hold_strength'],
        )
        kqb_stab.capture_snapshot(model)
    else:
        kqb_stab = None

    phase1_budget_ctrl = BetaOnlyBudgetController(
        target_ebops     = target_ebops,
        margin           = cfg['phase1_margin'],
        beta_init        = cfg['phase1_beta_init'],
        beta_min         = cfg['phase1_beta_min'],
        beta_max         = cfg['phase1_beta_max'],
        adjust_factor    = 1.15,
        ema_alpha        = 0.25,
        warmup_epochs    = 100,        # 短暂 warmup，避免突然施压
        max_change_ratio = 1.5,
        init_ebops       = phase0_ebops,   # 从 Phase 0 结束时的 eBOPs 出发
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg['phase1_lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    phase1_callbacks = [
        ebops_cb,
        phase1_budget_ctrl,   # ★ 开始压缩 ★
        mask_enforcer,        # mask 仍然固定（release_epoch 未到）
        pareto_cb,
        make_lr_scheduler(cfg['phase1_lr'], cfg['phase1_lr_cycle'], cfg['phase1_lr_mmul'],
                          offset=phase0_epochs),
        trace_cb,
    ]
    if kqb_stab is not None:
        phase1_callbacks.insert(2, kqb_stab)

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        initial_epoch=phase0_epochs,
        epochs=phase0_epochs + phase1_epochs,
        callbacks=phase1_callbacks,
        verbose=1,
    )

    phase1_ebops = compute_model_ebops(model, _sample)
    phase1_res = model.evaluate(dataset_val, verbose=0)
    phase1_beta = phase1_budget_ctrl.beta_current
    print(f'\n  Phase 1 done:  val_acc={phase1_res[1]:.4f}  eBOPs={phase1_ebops:.0f}  '
          f'beta={phase1_beta:.2e}')
    print_bk_stats(model, 'end of Phase 1')

    # ── 6. Phase 2: 精调 ─────────────────────────────────────────────────────
    print(f'\n[6/6] Phase 2: Accuracy Fine-tuning  ({phase2_epochs} epochs)')
    print(f'  ▸ 低 LR, 窄 margin(±{cfg["phase2_margin"]*100:.0f}%), mask 逐渐释放, Adam 重置')

    phase2_budget_ctrl = BetaOnlyBudgetController(
        target_ebops     = target_ebops,
        margin           = cfg['phase2_margin'],
        beta_init        = phase1_beta,
        beta_min         = cfg['phase2_beta_min'],
        beta_max         = cfg['phase2_beta_max'],
        adjust_factor    = 1.15,
        ema_alpha        = 0.15,
        warmup_epochs    = 0,
        max_change_ratio = 1.5,
        init_ebops       = target_ebops,
    )

    # 重置 Adam momentum
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg['phase2_lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    earlystop_budget = target_ebops * 1.5
    early_stop_cb = BudgetAwareEarlyStopping(
        ebops_budget         = earlystop_budget,
        patience             = cfg['earlystop_patience'],
        min_delta            = 5e-5,
        min_epoch            = phase0_epochs + phase1_epochs + 1000,
        restore_best_weights = True,
    )

    # mask_enforcer 在 Phase 2 开始时进入 fade 阶段（因 release_epoch 已到）
    phase2_callbacks = [
        ebops_cb,
        phase2_budget_ctrl,
        mask_enforcer,        # Phase 2 期间 fade 释放
        pareto_cb,
        make_lr_scheduler(cfg['phase2_lr'], cfg['phase2_lr_cycle'], cfg['phase2_lr_mmul'],
                          offset=phase0_epochs + phase1_epochs),
        trace_cb,
        early_stop_cb,
    ]

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        initial_epoch=phase0_epochs + phase1_epochs,
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
        'target_ebops':   target_ebops,
        'final_ebops':    final_ebops,
        'final_val_acc':  float(final_res[1]),
        'final_val_loss': float(final_res[0]),
        'phase0_val_acc': float(phase0_res[1]),
        'phase0_ebops':   phase0_ebops,
        'phase1_val_acc': float(phase1_res[1]),
        'phase1_ebops':   phase1_ebops,
        'elapsed_sec':    elapsed,
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
            mask = eb <= target_ebops * 1.1
            if mask.any():
                filtered = np.where(mask, va, 0)
                best_filtered = filtered.argmax()
                result['best_acc_in_budget'] = float(va[best_filtered])
                result['best_epoch_in_budget'] = int(best_filtered)

    print(f'\n  {"=" * 60}')
    print(f'  v3 Result:')
    print(f'    phase0_acc={phase0_res[1]:.4f} (at eBOPs={phase0_ebops:.0f}, no compression)')
    print(f'    phase1_acc={phase1_res[1]:.4f} (after compression to eBOPs={phase1_ebops:.0f})')
    print(f'    final_acc={final_res[1]:.4f}   best_acc={result.get("best_val_acc", 0):.4f}')
    print(f'    best_in_budget={result.get("best_acc_in_budget", 0):.4f}')
    print(f'    target={target_ebops:.0f}  final_ebops={final_ebops:.0f}  time={elapsed:.0f}s')
    print(f'  {"=" * 60}')

    summary_path = os.path.join(output_folder, 'result_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Summary saved: {summary_path}')

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    cfg = dict(DEFAULT_CONFIG)
    cfg['seed']            = args.seed
    cfg['target_ebops']    = args.target_ebops
    cfg['phase0_epochs']   = args.phase0_epochs
    cfg['phase1_epochs']   = args.phase1_epochs
    cfg['phase2_epochs']   = args.phase2_epochs
    cfg['phase0_lr']       = args.phase0_lr
    cfg['phase1_lr']       = args.phase1_lr
    cfg['phase2_lr']       = args.phase2_lr
    cfg['init_bw_k']       = args.init_bw_k
    cfg['ram_multiplier']  = args.ram_multiplier
    cfg['ram_min_degree']  = args.ram_min_degree
    cfg['mask_hold']       = (args.mask_hold if args.mask_hold is not None
                              else args.phase0_epochs + args.phase1_epochs)
    cfg['mask_fade']       = args.mask_fade
    cfg['init_only']       = args.init_only
    cfg['skip_calib']      = args.skip_calib
    cfg['calib_multiplier'] = args.calib_multiplier
    cfg['output_dir']      = args.output_dir

    if args.sweep:
        # ── Sweep 模式 ───────────────────────────────────────────────────────
        targets = [float(x.strip()) for x in args.sweep.split(',')]
        print(f'\n{"#" * 72}')
        print(f'  Ramanujan v3 — eBOPs SWEEP:  {targets}')
        print(f'{"#" * 72}\n')

        all_results = []
        for t in targets:
            cfg_copy = dict(cfg)
            result = run_single_target(t, cfg_copy)
            all_results.append(result)

            print(f'\n  ── Sweep progress (Pareto so far) ──')
            for r in sorted(all_results, key=lambda x: x['target_ebops']):
                print(f'    target={r["target_ebops"]:6.0f}  '
                      f'best_acc={r.get("best_val_acc", 0):.4f}  '
                      f'best_in_budget={r.get("best_acc_in_budget", 0):.4f}  '
                      f'final_ebops={r["final_ebops"]:.0f}')
            print()

        sweep_summary_path = 'results/ramanujan_v3_sweep_summary.json'
        with open(sweep_summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nSweep summary saved: {sweep_summary_path}')

        print(f'\n{"=" * 72}')
        print(f'  Ramanujan v3 Sweep — Final Pareto')
        print(f'  {"target":>8s}  {"best_acc":>8s}  {"in_budget":>9s}  {"final_eb":>8s}  {"time":>6s}')
        print(f'  {"-" * 50}')
        for r in sorted(all_results, key=lambda x: x['target_ebops']):
            print(f'  {r["target_ebops"]:8.0f}  '
                  f'{r.get("best_val_acc", 0):8.4f}  '
                  f'{r.get("best_acc_in_budget", 0):9.4f}  '
                  f'{r["final_ebops"]:8.0f}  '
                  f'{r["elapsed_sec"]:5.0f}s')
        print(f'{"=" * 72}')

    else:
        # ── 单次运行 ─────────────────────────────────────────────────────────
        run_single_target(cfg['target_ebops'], cfg)
