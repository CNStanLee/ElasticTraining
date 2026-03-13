#!/usr/bin/env python3
"""
run_experiment_F_elasticity.py — Topology Elasticity 消融实验 (400 eBOPs, v3)
================================================================================

验证 Topology Elasticity 三机制组合在 400 eBOPs 极低预算下解决 Beta 死锁的效果。

─── Beta 死锁根因 ───
kq.b < 0.5 → round_conv=0 → 前向=0 → 任务梯度=0 → 仅 beta 压力 → b 继续↓ → 不可逆死亡
拓扑在渐进压缩早期被锁定, 优化器只能在已有拓扑上优化, 无法探索更优拓扑。

─── 失败的方法 ───
消融 D (SoftDeathFloor): b_floor=0.05 < 0.5 → 前向仍为 0, 与 beta 对抗循环 → 越激进越差
消融 E (TopologyRescue): 事后干预 → 拓扑已提交 → 4 swap/layer 太小 → 无法超越 baseline

─── 新方法: Topology Elasticity ───
三个互补机制, 从根源解决死锁:

  WarmTopologyFloor (WTF):
    衰减 b 下限 (b0 ≥ 0.5 → 0), 使预算压缩期间连接保持活跃。
    关键: b0 = 0.55 ≥ 0.5 → 前向有信号 → 梯度有效 (SDF 的 b0=0.05 < 0.5 → 无效!)
    Floor 衰减到 0 后连接可自然死亡 → 不与 beta 持续对抗。

  ImportanceRebalance (IR):
    周期性重分配 b 预算: 重要连接 (|w|·b 大) 获得更多 b, 弱连接加速淘汰。
    打破 uniform beta 的拓扑决策随机性 → 基于任务相关性做拓扑选择。
    总 b 守恒 → 预算中性。

  StochasticTopologyExplore (STE):
    模拟退火式拓扑扰动: 检查点 → swap → 短期评估 → 接受/回滚。
    梯度下降无法跨越死连接障碍, 但 stochastic swap 可以 → 拓扑空间探索。

消融维度:
  ── 对照组 ──
  F0: Baseline (无任何弹性机制, = E0/D0)

  ── 单机制有效性 ──
  F1: WarmFloor only (衰减 b 下限, b0=0.55, anneal=budget_decay_epochs)
  F2: ImportanceRebalance only (周期 50, fraction=0.2, delta=0.1)
  F3: StochasticExplore only (interval=200, eval_window=50, n_swap=4)

  ── 组合 ──
  F4: WarmFloor + ImportanceRebalance (最主要的组合)
  F5: WarmFloor + ImportanceRebalance + StochasticExplore (全部)

  ── WarmFloor 参数消融 ──
  F6: b_floor_init=0.7 (更高的初始下限, 更强保护)
  F7: anneal_power=2.5 (前慢后快衰减, 更长的探索期)

  ── ImportanceRebalance 参数消融 ──
  F8: rebalance_interval=25, delta=0.15 (更频繁更激进的再分配)

  ── StochasticExplore 参数消融 ──
  F9: explore_interval=100, n_swap=8 (更频繁更大的扰动)

用法:
  python run_experiment_F_elasticity.py --list          # 列出所有配置
  python run_experiment_F_elasticity.py --run F0 F1     # 运行指定实验
  python run_experiment_F_elasticity.py --run F          # 运行所有 F 实验
  python run_experiment_F_elasticity.py --run-all        # 运行全部
  python run_experiment_F_elasticity.py --dry-run        # 仅打印配置
  python run_experiment_F_elasticity.py --skip-existing  # 跳过已有结果
  python run_experiment_F_elasticity.py --compare-only   # 仅打印对比表

输出: results/experiment_F_elasticity/<实验名>/
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import sys
import time
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# 实验定义
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_ROOT = 'results/experiment_F_elasticity'
TARGET_EBOPS = 400


def _wtf(b0=0.55, anneal_epochs=None, anneal_power=1.5, **kw):
    """WarmTopologyFloor 参数。anneal_epochs 默认 = budget_decay_epochs (由 build_config 填充)。"""
    base = dict(
        wtf_enabled        = True,
        wtf_b_floor_init   = b0,
        wtf_anneal_epochs  = anneal_epochs,   # None → 由 build_config 设为 budget_decay_epochs
        wtf_anneal_power   = anneal_power,
        wtf_apply_every    = 1,
        wtf_protect_kernel = True,
    )
    base.update(kw)
    return base


def _ir(interval=50, fraction=0.2, delta=0.1, **kw):
    """ImportanceRebalance 参数。"""
    base = dict(
        ir_enabled             = True,
        ir_rebalance_interval  = interval,
        ir_rebalance_fraction  = fraction,
        ir_rebalance_delta     = delta,
        ir_alive_threshold     = 0.3,
        ir_decay_factor        = 0.98,
    )
    base.update(kw)
    return base


def _ste(interval=200, eval_window=50, n_swap=4, **kw):
    """StochasticTopologyExplore 参数。"""
    base = dict(
        ste_enabled            = True,
        ste_explore_interval   = interval,
        ste_eval_window        = eval_window,
        ste_n_swap_per_layer   = n_swap,
        ste_accept_tolerance   = 0.005,
        ste_initial_temperature = 0.01,
        ste_temp_decay         = 0.85,
        ste_start_epoch        = 500,
        ste_max_explorations   = 15,
        ste_revival_b_val      = 1.0,
    )
    base.update(kw)
    return base


def _disabled():
    """所有弹性机制禁用。"""
    return dict(
        wtf_enabled = False,
        ir_enabled  = False,
        ste_enabled = False,
        topo_rescue_enabled = False,
    )


def _merge(*dicts):
    """合并多个 dict。"""
    out = {}
    for d in dicts:
        out.update(d)
    return out


EXPERIMENTS = {
    # ── F0: Baseline (无弹性机制) ────────────────────────────────────────
    'F0': dict(
        desc='Baseline (no elasticity)',
        overrides=_disabled(),
    ),

    # ── F1: WarmFloor only ─────────────────────────────────────────────
    'F1': dict(
        desc='WarmFloor only (b0=0.55)',
        overrides=_merge(_disabled(), _wtf()),
    ),

    # ── F2: ImportanceRebalance only ─────────────────────────────────────
    'F2': dict(
        desc='ImportRebalance only',
        overrides=_merge(_disabled(), _ir()),
    ),

    # ── F3: StochasticExplore only ───────────────────────────────────────
    'F3': dict(
        desc='StochExplore only',
        overrides=_merge(_disabled(), _ste()),
    ),

    # ── F4: WarmFloor + ImportanceRebalance ──────────────────────────────
    'F4': dict(
        desc='WarmFloor + ImportRebalance',
        overrides=_merge(_disabled(), _wtf(), _ir()),
    ),

    # ── F5: All three ────────────────────────────────────────────────────
    'F5': dict(
        desc='WTF + IR + STE (all three)',
        overrides=_merge(_disabled(), _wtf(), _ir(), _ste()),
    ),

    # ── F6: WarmFloor b0=0.7 ────────────────────────────────────────────
    'F6': dict(
        desc='WarmFloor b0=0.7 (stronger)',
        overrides=_merge(_disabled(), _wtf(b0=0.7)),
    ),

    # ── F7: WarmFloor anneal_power=2.5 ──────────────────────────────────
    'F7': dict(
        desc='WarmFloor power=2.5 (slow start)',
        overrides=_merge(_disabled(), _wtf(anneal_power=2.5)),
    ),

    # ── F8: ImportanceRebalance frequent ─────────────────────────────────
    'F8': dict(
        desc='ImportRebal freq (25, δ=0.15)',
        overrides=_merge(_disabled(), _ir(interval=25, delta=0.15)),
    ),

    # ── F9: StochasticExplore aggressive ─────────────────────────────────
    'F9': dict(
        desc='StochExplore aggr (100, swap=8)',
        overrides=_merge(_disabled(), _ste(interval=100, n_swap=8)),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 运行逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def build_config(exp_name: str) -> dict:
    """合并 run_all_v3 DEFAULT_CONFIG + target overrides + 实验覆盖 → 完整配置。"""
    from FPL2.jsc.bk.run_all_v3 import DEFAULT_CONFIG, get_target_overrides_v3

    exp_def = EXPERIMENTS[exp_name]
    cfg = dict(DEFAULT_CONFIG)
    cfg['pretrained'] = 'pretrained_weight/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras'
    cfg['target_ebops'] = TARGET_EBOPS

    # 应用 target overrides (v3)
    target_ov = get_target_overrides_v3(TARGET_EBOPS)
    cfg.update(target_ov)

    # 禁用 v3 默认的 TopologyRescue (由实验覆盖控制)
    cfg['topo_rescue_enabled'] = False

    # 应用实验覆盖 (最高优先级)
    cfg.update(exp_def['overrides'])

    # WarmFloor anneal_epochs 默认 = budget_decay_epochs
    if cfg.get('wtf_enabled', False) and cfg.get('wtf_anneal_epochs') is None:
        cfg['wtf_anneal_epochs'] = cfg.get('budget_decay_epochs', 2000)

    # 输出目录
    cfg['output_dir'] = os.path.join(OUTPUT_ROOT, exp_name)

    return cfg


def _format_mechanisms(cfg: dict) -> str:
    """格式化当前启用的机制。"""
    parts = []
    if cfg.get('wtf_enabled', False):
        parts.append(f'WTF(b0={cfg.get("wtf_b_floor_init", 0.55):.2f},'
                     f'T={cfg.get("wtf_anneal_epochs", "?")},p={cfg.get("wtf_anneal_power", 1.5):.1f})')
    if cfg.get('ir_enabled', False):
        parts.append(f'IR(i={cfg.get("ir_rebalance_interval", 50)},'
                     f'f={cfg.get("ir_rebalance_fraction", 0.2):.1f},'
                     f'δ={cfg.get("ir_rebalance_delta", 0.1):.2f})')
    if cfg.get('ste_enabled', False):
        parts.append(f'STE(i={cfg.get("ste_explore_interval", 200)},'
                     f'w={cfg.get("ste_eval_window", 50)},'
                     f's={cfg.get("ste_n_swap_per_layer", 4)})')
    if cfg.get('topo_rescue_enabled', False):
        parts.append('TopoRescue')
    return ' + '.join(parts) if parts else 'DISABLED'


def run_single(exp_name: str, dry_run: bool = False) -> dict | None:
    """运行单个消融实验。"""
    exp_def = EXPERIMENTS[exp_name]
    cfg = build_config(exp_name)

    total = cfg['phase1_epochs'] + cfg['phase2_epochs']
    mech_str = _format_mechanisms(cfg)

    print(f'\n{"#" * 72}')
    print(f'  Experiment {exp_name}: {exp_def["desc"]}')
    print(f'  Target: {TARGET_EBOPS} eBOPs  |  Total: {total} epochs')
    print(f'  Mechanisms: {mech_str}')
    print(f'  Output: {cfg["output_dir"]}')
    print(f'{"#" * 72}')

    if dry_run:
        print('  [DRY RUN] Skipping...')
        relevant_keys = [
            'wtf_enabled', 'wtf_b_floor_init', 'wtf_anneal_epochs', 'wtf_anneal_power',
            'ir_enabled', 'ir_rebalance_interval', 'ir_rebalance_fraction', 'ir_rebalance_delta',
            'ste_enabled', 'ste_explore_interval', 'ste_eval_window', 'ste_n_swap_per_layer',
            'ste_accept_tolerance', 'ste_initial_temperature',
            'topo_rescue_enabled',
            'phase1_epochs', 'phase2_epochs', 'budget_decay_epochs',
            'warmup_ebops_mul', 'target_ebops',
        ]
        for k in relevant_keys:
            if k in cfg:
                print(f'    {k}: {cfg.get(k)}')
        return None

    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 保存元数据
    meta = {
        'experiment': exp_name,
        'desc': exp_def['desc'],
        'target_ebops': TARGET_EBOPS,
        'total_epochs': total,
        'mechanisms': mech_str,
        'wtf_enabled': cfg.get('wtf_enabled', False),
        'ir_enabled': cfg.get('ir_enabled', False),
        'ste_enabled': cfg.get('ste_enabled', False),
    }
    if cfg.get('wtf_enabled', False):
        meta.update({
            'wtf_b_floor_init': cfg.get('wtf_b_floor_init'),
            'wtf_anneal_epochs': cfg.get('wtf_anneal_epochs'),
            'wtf_anneal_power': cfg.get('wtf_anneal_power'),
        })
    if cfg.get('ir_enabled', False):
        meta.update({
            'ir_rebalance_interval': cfg.get('ir_rebalance_interval'),
            'ir_rebalance_fraction': cfg.get('ir_rebalance_fraction'),
            'ir_rebalance_delta': cfg.get('ir_rebalance_delta'),
        })
    if cfg.get('ste_enabled', False):
        meta.update({
            'ste_explore_interval': cfg.get('ste_explore_interval'),
            'ste_eval_window': cfg.get('ste_eval_window'),
            'ste_n_swap_per_layer': cfg.get('ste_n_swap_per_layer'),
            'ste_accept_tolerance': cfg.get('ste_accept_tolerance'),
        })
    with open(os.path.join(cfg['output_dir'], 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 使用 v3 管线 (带弹性扩展)
    result = _run_with_elasticity(cfg, exp_name, exp_def)
    return result


def _run_with_elasticity(cfg: dict, exp_name: str, exp_def: dict) -> dict:
    """带 Topology Elasticity 扩展的 v3 管线。"""
    from FPL2.jsc.bk.run_all_v3 import run as run_v3_pipeline

    # 注入弹性回调到 cfg 中 (v3 管线会读取这些)
    # 由于 v3 管线不直接支持弹性回调, 我们需要 monkey-patch

    # 策略: 用 run_all_v3_elastic() 替代, 它在 run_all_v3.run() 基础上注入回调。
    # 但这需要修改 v3 — 更简单的做法: 直接在此复制 v3 管线的核心逻辑并加入弹性回调。
    # 为避免大量代码重复, 我们采用另一种策略: 在 cfg 中传递回调工厂, v3 如果发现就使用。

    # 最简方案: 直接调用 v3 管线的各个步骤, 在创建回调列表时注入弹性回调。
    # 这需要我们重构 run() 为可扩展的, 但那会破坏已有代码。
    # 折中方案: 自己实现一个完整管线 (基于 v3 的代码)。

    t0 = time.time()
    result = _run_elastic_pipeline(cfg)
    elapsed = time.time() - t0
    result['experiment'] = exp_name
    result['desc'] = exp_def['desc']
    result['elapsed_total'] = elapsed
    return result


def _run_elastic_pipeline(cfg: dict) -> dict:
    """完整的弹性管线 — 基于 run_all_v3.run() 并注入弹性回调。"""
    import random

    import keras
    import numpy as np
    import tensorflow as tf
    from hgq.utils.sugar import Dataset, FreeEBOPs, ParetoFront
    from keras.callbacks import LearningRateScheduler

    from data.data import get_data
    from utils import (
        get_tf_device,
        spectral_quant_prune_to_ebops,
        SensitivityAwarePruner,
        bisect_ebops_to_target,
        compute_model_ebops,
        BetaOnlyBudgetController,
        ActivationBitsFixer,
        ProgressiveBudgetController,
        BetaCurriculumController,
        AdaptiveLRBiwidthScaler,
        _set_all_beta,
        _get_active_bk_mean,
        cosine_decay_restarts_schedule,
        TrainingTraceToH5,
        BudgetAwareEarlyStopping,
        TopologyPlotCallback,
        plot_topology,
        _flatten_layers,
        _get_kq_var,
        WarmTopologyFloor,
        ImportanceRebalance,
        StochasticTopologyExplore,
    )

    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    init_bw       = cfg['init_bw']
    init_bw_a     = cfg.get('init_bw_a') or init_bw
    target_ebops  = cfg['target_ebops']
    phase1_epochs = cfg['phase1_epochs']
    phase2_epochs = cfg['phase2_epochs']
    total_epochs  = phase1_epochs + phase2_epochs

    warmup_ebops_mul = cfg.get('warmup_ebops_mul', 5.0)
    warmup_ebops = target_ebops * warmup_ebops_mul
    budget_decay_epochs = cfg.get('budget_decay_epochs', 2000)

    output_folder = cfg.get('output_dir', '') or f'results/v3e_ebops{int(target_ebops)}/'
    os.makedirs(output_folder, exist_ok=True)
    device = get_tf_device()
    batch_size = 33200

    mech_str = _format_mechanisms(cfg)

    print('=' * 72)
    print(f'  FPL2-JSC Elastic Pipeline  |  target_eBOPs={target_ebops:.0f}  init_bw={init_bw}-bit')
    print(f'  Total epochs: {total_epochs} (P1={phase1_epochs}, P2={phase2_epochs})')
    print(f'  Mechanisms: {mech_str}')
    print(f'  Progressive budget: {warmup_ebops:.0f} → {target_ebops:.0f} over {budget_decay_epochs} ep')
    print(f'  Output: {output_folder}')
    print('=' * 72)

    t0 = time.time()

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/5] Loading dataset...')
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset.h5')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(data_path, src='openml')
    _sample = tf.constant(X_train[:min(2048, len(X_train))], dtype=tf.float32)

    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val   = Dataset(X_val, y_val, batch_size, device)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # ── 2. 加载预训练模型 ────────────────────────────────────────────────────
    print(f'\n[2/5] Loading pretrained model from {cfg["pretrained"]}...')
    import model.model as _model_mod  # noqa: F401
    model = keras.models.load_model(cfg['pretrained'], compile=False)
    model.summary()

    from FPL2.jsc.bk.run_all_v3 import print_bk_stats
    print_bk_stats(model, 'before pruning')

    pretrained_ebops = compute_model_ebops(model, _sample)
    print(f'  Pretrained eBOPs: {pretrained_ebops:.0f}')

    # ── 3. 剪枝到 warmup_ebops ──────────────────────────────────────────────
    use_sensitivity = cfg.get('use_sensitivity_pruner', False)
    if use_sensitivity:
        print(f'\n[3/5] SensitivityAwarePruner → warmup_eBOPs={warmup_ebops:.0f}')
        pruner = SensitivityAwarePruner(target_ebops=warmup_ebops, pruned_threshold=0.1, b_k_min=0.3)
        pruner.prune_to_ebops(model, current_ebops=pretrained_ebops, verbose=True)
    else:
        print(f'\n[3/5] Spectral-quant pruning → warmup_eBOPs={warmup_ebops:.0f}')
        spectral_quant_prune_to_ebops(
            model, target_ebops=warmup_ebops, sample_input=_sample,
            min_degree=cfg['min_degree'], b_floor=0.35, b_ceiling=6.0, verbose=True,
        )

    _bisect_bk_min = 0.35 if not use_sensitivity else 0.01
    calibrated_ebops = bisect_ebops_to_target(
        model, target_ebops=warmup_ebops, sample_input=_sample,
        tolerance=0.04, max_iter=24, b_k_min=_bisect_bk_min,
        allow_connection_kill=not use_sensitivity,
    )

    pruned_ebops = compute_model_ebops(model, _sample)
    print(f'  After pruning + calibration: eBOPs={pruned_ebops:.0f}')
    print_bk_stats(model, 'post-pruning')

    plot_topology(model, output_folder, 'init_after_pruning', ebops=pruned_ebops, plot_matrix=False)

    init_path = os.path.join(output_folder, f'pruned_init_{init_bw}bit.keras')
    model.save(init_path)

    # ── Helper: LR scheduler ─────────────────────────────────────────────────
    def make_lr_scheduler(lr_init, cycle, mmul, warmup_epochs=0, offset=0):
        fn = cosine_decay_restarts_schedule(lr_init, cycle, t_mul=1.0, m_mul=mmul,
                                            alpha=1e-6, alpha_steps=50)
        def schedule(epoch):
            effective_epoch = max(0, epoch - offset)
            lr = fn(effective_epoch)
            if warmup_epochs > 0 and effective_epoch < warmup_epochs:
                lr *= (effective_epoch + 1) / warmup_epochs
            return lr
        return LearningRateScheduler(lambda epoch: schedule(epoch))

    # ── Helper: 创建弹性回调 ─────────────────────────────────────────────────
    def create_elasticity_callbacks(cfg, phase='phase1'):
        """根据 cfg 创建弹性回调列表。"""
        cbs = []

        if cfg.get('wtf_enabled', False):
            cbs.append(WarmTopologyFloor(
                b_floor_init=cfg.get('wtf_b_floor_init', 0.55),
                anneal_epochs=cfg.get('wtf_anneal_epochs', budget_decay_epochs),
                anneal_power=cfg.get('wtf_anneal_power', 1.5),
                apply_every=cfg.get('wtf_apply_every', 1),
                protect_kernel=cfg.get('wtf_protect_kernel', True),
                log_interval=100,
            ))

        if cfg.get('ir_enabled', False):
            # Phase 2: 更保守的参数
            delta_mul = 1.0 if phase == 'phase1' else 0.5
            cbs.append(ImportanceRebalance(
                rebalance_interval=cfg.get('ir_rebalance_interval', 50),
                rebalance_fraction=cfg.get('ir_rebalance_fraction', 0.2),
                rebalance_delta=cfg.get('ir_rebalance_delta', 0.1) * delta_mul,
                alive_threshold=cfg.get('ir_alive_threshold', 0.3),
                decay_factor=cfg.get('ir_decay_factor', 0.98),
                start_epoch=0,
                log_interval=100,
            ))

        if cfg.get('ste_enabled', False):
            # Phase 2: 更保守 (更少探索, 更少 swap)
            n_swap = cfg.get('ste_n_swap_per_layer', 4)
            interval = cfg.get('ste_explore_interval', 200)
            if phase == 'phase2':
                n_swap = max(1, n_swap // 2)
                interval = int(interval * 1.5)
            cbs.append(StochasticTopologyExplore(
                explore_interval=interval,
                eval_window=cfg.get('ste_eval_window', 50),
                n_swap_per_layer=n_swap,
                accept_tolerance=cfg.get('ste_accept_tolerance', 0.005),
                initial_temperature=cfg.get('ste_initial_temperature', 0.01),
                temp_decay=cfg.get('ste_temp_decay', 0.85),
                start_epoch=cfg.get('ste_start_epoch', 500) if phase == 'phase1' else 0,
                max_explorations=cfg.get('ste_max_explorations', 15),
                min_degree=cfg.get('min_degree', 2),
                revival_b_val=cfg.get('ste_revival_b_val', 1.0),
            ))

        return cbs

    # ── 4. Phase 1: 恢复训练 + 渐进压缩 + 弹性 ──────────────────────────────
    print(f'\n[4/5] Phase 1: Recovery + Progressive Compression + Elasticity ({phase1_epochs} epochs)')

    _set_all_beta(model, cfg['phase1_beta_init'])

    phase1_budget = BetaOnlyBudgetController(
        target_ebops=warmup_ebops,
        margin=cfg['phase1_margin'],
        beta_init=cfg['phase1_beta_init'],
        beta_min=cfg['phase1_beta_min'],
        beta_max=cfg['phase1_beta_max'],
        adjust_factor=1.3,
        ema_alpha=0.3,
        warmup_epochs=200,
        max_change_ratio=2.0,
        init_ebops=pruned_ebops,
    )

    prog_budget = ProgressiveBudgetController(
        budget_ctrl=phase1_budget,
        warmup_ebops=warmup_ebops,
        final_ebops=target_ebops,
        decay_epochs=budget_decay_epochs,
        start_epoch=0,
    )
    print(f'  [Progressive] {warmup_ebops:.0f} → {target_ebops:.0f} over {budget_decay_epochs} ep')

    act_fixer = ActivationBitsFixer(b_a_fixed=float(init_bw_a), start_epoch=0)
    ebops_cb  = FreeEBOPs()
    pareto_cb = ParetoFront(
        output_folder,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )
    trace_cb = TrainingTraceToH5(output_dir=output_folder, filename='training_trace.h5', max_bits=8)

    plot_interval = cfg.get('plot_interval', 500)
    topo_plot_cb = TopologyPlotCallback(
        output_dir=output_folder, plot_interval=plot_interval,
        sample_input=_sample, plot_matrix=True,
    )

    clipnorm = cfg.get('clipnorm', 1.0)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg['phase1_lr'], clipnorm=clipnorm),
        loss=loss_fn, metrics=['accuracy'], steps_per_execution=32,
    )

    warmup = cfg.get('lr_warmup', 100)
    phase1_callbacks = [
        ebops_cb,
        phase1_budget,
        prog_budget,
        act_fixer,
        pareto_cb,
        make_lr_scheduler(cfg['phase1_lr'], cfg['phase1_lr_cycle'], cfg['phase1_lr_mmul'],
                          warmup_epochs=warmup),
        trace_cb,
        topo_plot_cb,
    ]

    # Beta 课程重启
    if cfg.get('beta_curriculum_enabled', True):
        p1_curriculum = BetaCurriculumController(
            budget_ctrl=phase1_budget,
            stall_patience=cfg.get('beta_stall_patience', 600),
            recover_epochs=cfg.get('beta_recover_epochs', 300),
            min_delta=5e-5,
            restart_decay=cfg.get('beta_restart_decay', 0.25),
            max_restarts=cfg.get('beta_max_restarts', 8),
            recover_beta_floor=cfg.get('beta_recover_floor', None),
        )
        phase1_callbacks.append(p1_curriculum)

    # 自适应 LR
    if cfg.get('adaptive_lr_enabled', True):
        p1_lr_scaler = AdaptiveLRBiwidthScaler(
            bk_threshold=cfg.get('adaptive_lr_bk_threshold', 2.0),
            scale_power=cfg.get('adaptive_lr_scale_power', 0.5),
            lr_max_factor=cfg.get('adaptive_lr_max_factor', 4.0),
            log=False,
        )
        phase1_callbacks.append(p1_lr_scaler)

    # ★★★ 弹性回调注入 ★★★
    elasticity_cbs_p1 = create_elasticity_callbacks(cfg, phase='phase1')
    phase1_callbacks.extend(elasticity_cbs_p1)
    if elasticity_cbs_p1:
        print(f'  [Elasticity] Phase 1 callbacks: {[type(cb).__name__ for cb in elasticity_cbs_p1]}')

    print('  Starting Phase 1...')
    model.fit(
        dataset_train, validation_data=dataset_val,
        epochs=phase1_epochs, callbacks=phase1_callbacks, verbose=1,
    )

    phase1_ebops = compute_model_ebops(model, _sample)
    phase1_res = model.evaluate(dataset_val, verbose=0)
    phase1_beta = phase1_budget.beta_current
    print(f'\n  Phase 1 done: val_acc={phase1_res[1]:.4f}  eBOPs={phase1_ebops:.0f}  beta={phase1_beta:.2e}')
    print_bk_stats(model, 'end-of-Phase1')

    # ── 5. Phase 2: 精调 + 弹性 ─────────────────────────────────────────────
    if phase2_epochs <= 0:
        print(f'\n[5/5] Phase 2: SKIPPED')
    else:
        print(f'\n[5/5] Phase 2: Fine-tuning + Elasticity ({phase2_epochs} epochs)')

        p2_adjust = cfg.get('phase2_adjust_factor', 1.15)
        phase2_budget = BetaOnlyBudgetController(
            target_ebops=target_ebops,
            margin=cfg['phase2_margin'],
            beta_init=phase1_beta,
            beta_min=cfg['phase2_beta_min'],
            beta_max=cfg['phase2_beta_max'],
            adjust_factor=p2_adjust,
            ema_alpha=0.15,
            warmup_epochs=0,
            max_change_ratio=1.5,
            init_ebops=phase1_ebops,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg['phase2_lr'], clipnorm=clipnorm),
            loss=loss_fn, metrics=['accuracy'], steps_per_execution=32,
        )

        early_stop = BudgetAwareEarlyStopping(
            ebops_budget=target_ebops * 2.0,
            patience=cfg['earlystop_patience'],
            min_delta=5e-5,
            min_epoch=phase1_epochs + min(500, phase2_epochs // 2),
            restore_best_weights=True,
        )

        phase2_callbacks = [
            ebops_cb,
            phase2_budget,
            act_fixer,
            pareto_cb,
            make_lr_scheduler(cfg['phase2_lr'], cfg['phase2_lr_cycle'], cfg['phase2_lr_mmul'],
                              offset=phase1_epochs),
            trace_cb,
            early_stop,
            topo_plot_cb,
        ]

        p2_curriculum_enabled = cfg.get('beta_curriculum_p2', cfg.get('beta_curriculum_enabled', True))
        if p2_curriculum_enabled:
            p2_curriculum = BetaCurriculumController(
                budget_ctrl=phase2_budget,
                stall_patience=cfg.get('beta_stall_patience', 600) * 2,
                recover_epochs=cfg.get('beta_recover_epochs', 300),
                min_delta=2e-5,
                restart_decay=cfg.get('beta_restart_decay', 0.25),
                max_restarts=cfg.get('beta_max_restarts', 8),
                recover_beta_floor=cfg.get('beta_recover_floor', None),
            )
            phase2_callbacks.append(p2_curriculum)

        if cfg.get('adaptive_lr_enabled', True):
            p2_lr_scaler = AdaptiveLRBiwidthScaler(
                bk_threshold=cfg.get('adaptive_lr_bk_threshold', 2.0),
                scale_power=cfg.get('adaptive_lr_scale_power', 0.5) * 0.5,
                lr_max_factor=2.0,
                log=False,
            )
            phase2_callbacks.append(p2_lr_scaler)

        # ★★★ 弹性回调注入 (Phase 2) ★★★
        elasticity_cbs_p2 = create_elasticity_callbacks(cfg, phase='phase2')
        phase2_callbacks.extend(elasticity_cbs_p2)
        if elasticity_cbs_p2:
            print(f'  [Elasticity] Phase 2 callbacks: {[type(cb).__name__ for cb in elasticity_cbs_p2]}')

        model.fit(
            dataset_train, validation_data=dataset_val,
            initial_epoch=phase1_epochs, epochs=total_epochs,
            callbacks=phase2_callbacks, verbose=1,
        )

    # ── 最终统计 ──────────────────────────────────────────────────────────────
    final_ebops = compute_model_ebops(model, _sample)
    final_res = model.evaluate(dataset_val, verbose=0)
    elapsed = time.time() - t0
    print_bk_stats(model, 'final')

    plot_topology(model, output_folder, 'final', ebops=final_ebops)

    result = {
        'target_ebops':   target_ebops,
        'init_bw':        init_bw,
        'pretrained_ebops': pretrained_ebops,
        'pruned_ebops':   pruned_ebops,
        'final_ebops':    final_ebops,
        'final_val_acc':  float(final_res[1]),
        'final_val_loss': float(final_res[0]),
        'phase1_val_acc': float(phase1_res[1]),
        'phase1_ebops':   phase1_ebops,
        'elapsed_sec':    elapsed,
        'revival_enabled': False,
        'topo_rescue_enabled': False,
        'wtf_enabled':    cfg.get('wtf_enabled', False),
        'ir_enabled':     cfg.get('ir_enabled', False),
        'ste_enabled':    cfg.get('ste_enabled', False),
        'total_epochs':   total_epochs,
        'phase1_epochs':  phase1_epochs,
        'phase2_epochs':  phase2_epochs,
    }

    import h5py
    trace_path = os.path.join(output_folder, 'training_trace.h5')
    if os.path.exists(trace_path):
        with h5py.File(trace_path, 'r') as f:
            va = f['val_accuracy'][:]
            eb = f['ebops'][:]
            result['best_val_acc'] = float(va.max())
            result['best_epoch'] = int(va.argmax())
            result['best_ebops'] = float(eb[va.argmax()])
            mask = (eb >= target_ebops * 0.875) & (eb <= target_ebops * 1.125)
            if mask.any():
                import numpy as _np
                idx = _np.where(mask)[0][_np.argmax(va[mask])]
                result['best_acc_at_target'] = float(va[idx])
                result['best_epoch_at_target'] = int(idx)
                result['best_ebops_at_target'] = float(eb[idx])

    print(f'\n  {"=" * 60}')
    print(f'  target={target_ebops:.0f}  pruned={pruned_ebops:.0f}  final={final_ebops:.0f}  '
          f'val_acc={final_res[1]:.4f}  best_acc={result.get("best_val_acc", 0):.4f}  '
          f'best@target={result.get("best_acc_at_target", 0):.4f}  '
          f'time={elapsed:.0f}s')
    print(f'  {"=" * 60}')

    summary_path = os.path.join(output_folder, 'result_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Summary: {summary_path}')

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 对比表
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(results: list[dict]):
    """打印消融实验对比表。"""
    import h5py
    import numpy as np

    print(f'\n{"=" * 125}')
    print(f'  Topology Elasticity ABLATION — target={TARGET_EBOPS} eBOPs')
    print(f'{"=" * 125}')
    print(f'  {"Exp":>4s}  {"Description":<30s}  {"Mechanisms":<28s}  '
          f'{"Best@Tgt":>8s}  {"@Epoch":>6s}  {"Final":>7s}  {"eBOPs":>6s}  {"Time":>5s}  {"Δ F0":>7s}')
    print(f'  {"-" * 121}')

    # 找 F0 baseline
    f0_acc = None
    for r in results:
        if r.get('experiment') == 'F0':
            f0_acc = _get_best_at_target(r)
            break

    for r in sorted(results, key=lambda x: x.get('experiment', '')):
        exp_name = r.get('experiment', '?')
        desc = r.get('desc', '?')[:30]

        output_dir = os.path.join(OUTPUT_ROOT, exp_name)
        best_at_target, best_ep = _read_trace_best(output_dir)

        meta = _read_meta(output_dir)
        mechanisms = meta.get('mechanisms', _format_mechanisms_from_result(r))[:28]

        acc_str = f'{best_at_target:.4f}' if best_at_target is not None else f'{r.get("best_val_acc", 0):.4f}'
        ep_str = f'{best_ep}' if best_ep is not None else f'{r.get("best_epoch", "?")}'
        final_acc = r.get('final_val_acc', 0)
        final_eb = r.get('final_ebops', 0)
        elapsed = r.get('elapsed_total', r.get('elapsed_sec', 0))

        acc_val = best_at_target if best_at_target is not None else r.get('best_val_acc', 0)
        delta = f'{acc_val - f0_acc:+.4f}' if f0_acc is not None and acc_val is not None else '—'

        print(f'  {exp_name:>4s}  {desc:<30s}  {mechanisms:<28s}  '
              f'{acc_str:>8s}  {ep_str:>6s}  {final_acc:7.4f}  {final_eb:6.0f}  {elapsed:5.0f}s  {delta:>7s}')

    print(f'{"=" * 125}')

    # 结论
    if f0_acc is not None:
        best_name = None
        best_acc = f0_acc
        for r in results:
            exp = r.get('experiment', '')
            if exp == 'F0':
                continue
            output_dir = os.path.join(OUTPUT_ROOT, exp)
            acc, _ = _read_trace_best(output_dir)
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_name = exp

        if best_name:
            meta = _read_meta(os.path.join(OUTPUT_ROOT, best_name))
            print(f'\n  ★ Best config: {best_name} ({best_acc:.4f}) '
                  f'vs F0 baseline ({f0_acc:.4f}) → Δ = {best_acc - f0_acc:+.4f}')
            print(f'    Mechanisms: {meta.get("mechanisms", "?")}')
        else:
            print(f'\n  ★ No config beats the F0 baseline ({f0_acc:.4f})')


def _format_mechanisms_from_result(r: dict) -> str:
    parts = []
    if r.get('wtf_enabled', False):
        parts.append('WTF')
    if r.get('ir_enabled', False):
        parts.append('IR')
    if r.get('ste_enabled', False):
        parts.append('STE')
    return '+'.join(parts) if parts else 'DISABLED'


def _get_best_at_target(r: dict):
    output_dir = os.path.join(OUTPUT_ROOT, r.get('experiment', ''))
    acc, _ = _read_trace_best(output_dir)
    if acc is not None:
        return acc
    return r.get('best_acc_at_target', r.get('best_val_acc', 0))


def _read_trace_best(output_dir: str):
    """从 training_trace.h5 读取 target 范围内 best accuracy。"""
    import h5py
    import numpy as np
    trace_path = os.path.join(output_dir, 'training_trace.h5')
    if not os.path.isfile(trace_path):
        return None, None
    try:
        with h5py.File(trace_path, 'r') as f:
            va = f['val_accuracy'][:]
            eb = f['ebops'][:]
            mask = (eb >= TARGET_EBOPS * 0.875) & (eb <= TARGET_EBOPS * 1.125)
            if mask.any():
                idx = np.where(mask)[0][np.argmax(va[mask])]
                return float(va[idx]), int(idx)
    except Exception:
        pass
    return None, None


def _read_meta(output_dir: str) -> dict:
    meta_path = os.path.join(output_dir, 'experiment_meta.json')
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Experiment F: Topology Elasticity ablation at 400 eBOPs')
    parser.add_argument('--run', nargs='+', default=None,
                        help='Experiment names to run (e.g. F0 F1, or "F" for all, or "all")')
    parser.add_argument('--list', action='store_true',
                        help='List all experiment configs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configs without running')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all experiments')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip experiments with existing results')
    parser.add_argument('--compare-only', action='store_true',
                        help='Only print comparison from existing results')
    args = parser.parse_args()

    if args.list:
        print(f'\n  Topology Elasticity Experiments (target={TARGET_EBOPS} eBOPs):')
        print(f'  {"Name":>4s}  {"Description":<30s}  {"WTF":>3s}  {"IR":>2s}  {"STE":>3s}  '
              f'{"b0":>5s}  {"pow":>3s}  {"δ":>5s}  {"swap":>4s}')
        print(f'  {"-" * 85}')
        for name, exp in sorted(EXPERIMENTS.items()):
            ov = exp['overrides']
            wtf = '✓' if ov.get('wtf_enabled', False) else '—'
            ir = '✓' if ov.get('ir_enabled', False) else '—'
            ste = '✓' if ov.get('ste_enabled', False) else '—'
            b0_s = f'{ov["wtf_b_floor_init"]:.2f}' if ov.get('wtf_enabled', False) else '—'
            pw_s = f'{ov["wtf_anneal_power"]:.1f}' if ov.get('wtf_enabled', False) else '—'
            dl_s = f'{ov["ir_rebalance_delta"]:.2f}' if ov.get('ir_enabled', False) else '—'
            sw_s = f'{ov["ste_n_swap_per_layer"]}' if ov.get('ste_enabled', False) else '—'

            print(f'  {name:>4s}  {exp["desc"]:<30s}  {wtf:>3s}  {ir:>2s}  {ste:>3s}  '
                  f'{b0_s:>5s}  {pw_s:>3s}  {dl_s:>5s}  {sw_s:>4s}')
        return

    if args.compare_only:
        results = []
        for name in sorted(EXPERIMENTS.keys()):
            result_path = os.path.join(OUTPUT_ROOT, name, 'result_summary.json')
            if os.path.isfile(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                r['experiment'] = name
                r['desc'] = EXPERIMENTS[name]['desc']
                results.append(r)
        if results:
            print_comparison(results)
        else:
            print('  No results found. Run experiments first.')
        return

    # 确定要运行的实验
    if args.run_all:
        exp_names = sorted(EXPERIMENTS.keys())
    elif args.run:
        exp_names = []
        for token in args.run:
            if token.lower() == 'all':
                exp_names = sorted(EXPERIMENTS.keys())
                break
            elif token.upper() == 'F':
                exp_names = sorted(EXPERIMENTS.keys())
                break
            else:
                name = token.upper()
                if name not in EXPERIMENTS:
                    print(f'[ERROR] Unknown experiment: {name}')
                    print(f'  Available: {sorted(EXPERIMENTS.keys())}')
                    sys.exit(1)
                exp_names.append(name)
    else:
        parser.print_help()
        print(f'\n  Use --list to see experiments, --run F0 F1 to run, --dry-run to preview.')
        return

    print(f'\n{"#" * 72}')
    print(f'  Experiment F: Topology Elasticity Ablation')
    print(f'  Target: {TARGET_EBOPS} eBOPs  |  Pipeline: elastic (v3-based)')
    print(f'  Running: {exp_names}')
    print(f'{"#" * 72}')

    all_results = []
    for name in exp_names:
        output_dir = os.path.join(OUTPUT_ROOT, name)
        result_path = os.path.join(output_dir, 'result_summary.json')

        if args.skip_existing and os.path.isfile(result_path):
            print(f'\n  [Skip] {name}: result exists')
            with open(result_path) as f:
                result = json.load(f)
            result['experiment'] = name
            result['desc'] = EXPERIMENTS[name]['desc']
            all_results.append(result)
            continue

        if args.dry_run:
            run_single(name, dry_run=True)
            continue

        result = run_single(name)
        if result:
            all_results.append(result)

            # 实时进度
            print(f'\n  ── Progress ({len(all_results)}/{len(exp_names)}) ──')
            for r in all_results:
                print(f'    {r["experiment"]}: best_acc={r.get("best_val_acc", 0):.4f}  '
                      f'@target={r.get("best_acc_at_target", 0):.4f}  '
                      f'time={r.get("elapsed_total", 0):.0f}s')

    if all_results and not args.dry_run:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        summary_path = os.path.join(OUTPUT_ROOT, 'all_results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Summary saved: {summary_path}')

        print_comparison(all_results)


if __name__ == '__main__':
    main()
