"""
FPL2/jsc — run_all_v3: 拓扑修复版谱约束剪枝 + Beta 调度 Pareto 搜索
=====================================================================

基于 run_all_v2 (fast decay)，核心改进：
  用 TopologyRescueCallback 替换 SoftDeathFloor。

Experiment D 证明 SoftDeathFloor 无效甚至有害：
  - SDF 无条件地将所有濒死连接钳位到 b_floor=0.05
  - b=0.05 < 0.5 → round_conv=0 → 前向输出仍为 0，梯度只靠 STE
  - SDF 与 beta 优化器形成对抗循环: beta 压低 b ↔ SDF 抬高 b
  - 越频繁干预精度越差 (D0 baseline=0.7428, D6 aggressive=0.2044)

TopologyRescue 的设计哲学完全不同:
  1. 停滞触发: 只在 val_accuracy 停滞时干预 (不与正常训练冲突)
  2. 结构感知: 只复活谱条件候选 (入度/出度不足的节点的死连接)
  3. 有效复活: revival_b_val=1.0 → round_conv>0 → 前向有信号
  4. 预算中性: swap-kill 最弱活跃连接 → 不增加 eBOPs → 不扰乱 beta
  5. 衰减冷却: 干预后长冷却期 + 强度衰减 → 收敛期不干预

关键改动 (vs run_all_v2.py):
  1. 移除 SoftDeathFloor 相关配置
  2. 新增 TopologyRescue 配置 (topo_rescue_*)
  3. run() 中用 TopologyRescueCallback 替换 SoftDeathFloor

Pipeline (不变):
  1. 加载预训练权重 (pretrained_weight/)
  2. 谱约束一次性剪枝 (保留 1-bit 下的必要谱连接)
  3. 渐进式预算从 warmup → target，同时用 Beta 调度器训练
  4. ParetoFront 自动保存 Pareto 最优模型

用法:
  python run_all_v3.py                               # 默认 target_ebops=400
  python run_all_v3.py --target_ebops 1500
  python run_all_v3.py --sweep_ebops 400,1500,2500,6800,12000
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import random
import time

import keras
import numpy as np
import tensorflow as tf
from hgq.layers import QLayerBase
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
    TopologyRescueCallback,
    ProgressiveBudgetController,
    BetaCurriculumController,
    AdaptiveLRBiwidthScaler,
    EBOPsConstantProjector,
    _set_all_beta,
    _get_active_bk_mean,
    SpectralGradientRevivalCallback,
    cosine_decay_restarts_schedule,
    TrainingTraceToH5,
    BudgetAwareEarlyStopping,
    TopologyPlotCallback,
    plot_topology,
    _flatten_layers,
    _get_kq_var,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 默认配置 (v3: TopologyRescue 替代 SoftDeathFloor)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    # 剪枝
    target_ebops     = 400,
    min_degree       = 2,

    # 渐进式预算
    warmup_ebops_mul = 5.0,
    budget_decay_epochs = 2000,

    # 位宽
    init_bw          = 1,
    init_bw_a        = 3,

    # Phase 1: 恢复 + 渐进压缩
    phase1_epochs    = 2500,
    phase1_lr        = 2e-3,
    phase1_lr_cycle  = 1200,
    phase1_lr_mmul   = 0.9,
    phase1_beta_init = 1e-5,
    phase1_beta_min  = 1e-8,
    phase1_beta_max  = 5e-4,
    phase1_margin    = 0.15,

    # Phase 2: 精调
    phase2_epochs    = 1000,
    phase2_lr        = 5e-4,
    phase2_lr_cycle  = 500,
    phase2_lr_mmul   = 0.95,
    phase2_beta_min  = 1e-8,
    phase2_beta_max  = 5e-4,
    phase2_margin    = 0.05,

    # 梯度裁剪 + LR 预热
    clipnorm           = 1.0,
    lr_warmup          = 100,

    # ★★★ TopologyRescue (替代 SoftDeathFloor) ★★★
    topo_rescue_enabled       = True,       # ★ 默认启用
    topo_rescue_revival_b     = 1.0,        # 复活位宽 (有效 1-bit)
    topo_rescue_check_interval = 50,        # 每 50 epoch 检查
    topo_rescue_stall_patience = 300,       # 停滞 300 epoch 后触发
    topo_rescue_min_delta     = 5e-5,       # 改善判定阈值
    topo_rescue_max_swap      = 4,          # 每层最多 swap 4 连接
    topo_rescue_swap_kill     = True,       # 预算中性 swap-kill
    topo_rescue_min_degree    = 2,          # 谱条件最小度
    topo_rescue_cool_down     = 200,        # 干预间隔
    topo_rescue_max_interventions = 10,     # 总干预上限
    topo_rescue_alive_threshold = 0.5,      # 死亡阈值
    topo_rescue_decay_factor  = 0.85,       # swap 强度衰减

    # Beta 课程重启
    beta_curriculum_enabled   = True,
    beta_stall_patience       = 600,
    beta_recover_epochs       = 300,
    beta_restart_decay        = 0.25,
    beta_max_restarts         = 8,
    beta_recover_floor        = None,

    # 自适应 LR 缩放
    adaptive_lr_enabled       = True,
    adaptive_lr_bk_threshold  = 2.0,
    adaptive_lr_scale_power   = 0.5,
    adaptive_lr_max_factor    = 4.0,

    # 连接复活 (RigL 风格, 保留但默认关闭)
    revival_enabled        = False,
    revival_interval       = 200,
    revival_max_per_layer  = 8,
    revival_b_val          = None,
    revival_swap_kill      = False,

    # 早停
    earlystop_patience = 2000,

    # 知识蒸馏
    kd_enabled         = False,
    kd_alpha           = 0.3,
    kd_temperature     = 4.0,

    # 拓扑绘图
    plot_interval      = 500,

    seed = 42,
)


# ── 不同 eBOPs 目标的专属配置 ────────────────────────────────────────────────

def get_target_overrides_v3(target_ebops: float) -> dict:
    """返回针对特定 eBOPs 目标的 v3 配置覆盖项。
    训练时长与 v2 完全一致，仅 SDF → TopologyRescue。
    """
    if target_ebops <= 500:
        return dict(
            use_sensitivity_pruner = True,
            warmup_ebops_mul    = 5.0,
            budget_decay_epochs = 2000,
            min_degree          = 2,

            phase1_epochs       = 2500,
            phase1_lr           = 2e-3,
            phase1_lr_cycle     = 1200,
            phase1_lr_mmul      = 0.9,
            phase1_beta_init    = 1e-5,
            phase1_beta_min     = 1e-8,
            phase1_beta_max     = 5e-4,
            phase1_margin       = 0.15,

            phase2_epochs       = 1000,
            phase2_lr           = 5e-4,
            phase2_lr_cycle     = 500,
            phase2_lr_mmul      = 0.95,
            phase2_beta_min     = 1e-8,
            phase2_beta_max     = 5e-4,
            phase2_margin       = 0.05,

            beta_curriculum_enabled = True,
            beta_stall_patience = 600,
            beta_recover_epochs = 300,
            beta_restart_decay  = 0.25,
            beta_max_restarts   = 8,

            adaptive_lr_enabled = True,

            # TopologyRescue 配置
            topo_rescue_enabled        = True,
            topo_rescue_revival_b      = 1.0,
            topo_rescue_stall_patience = 300,
            topo_rescue_max_swap       = 4,
            topo_rescue_swap_kill      = True,
            topo_rescue_cool_down      = 200,

            revival_enabled     = False,
            earlystop_patience  = 2000,
            clipnorm            = 1.0,
        )
    elif target_ebops <= 2000:
        return dict(
            warmup_ebops_mul    = 2.0,
            budget_decay_epochs = 1500,
            phase1_epochs       = 2000,
            phase2_epochs       = 1000,
            phase1_lr_cycle     = 1000,
            phase2_lr_cycle     = 500,
            earlystop_patience  = 2000,
            topo_rescue_enabled        = True,
            topo_rescue_stall_patience = 300,
            topo_rescue_max_swap       = 4,
            topo_rescue_cool_down      = 200,
        )
    elif target_ebops <= 4000:
        return dict(
            warmup_ebops_mul    = 2.0,
            budget_decay_epochs = 1500,
            phase1_epochs       = 2000,
            phase2_epochs       = 1000,
            phase1_lr_cycle     = 1000,
            phase2_lr_cycle     = 500,
            topo_rescue_enabled        = True,
            topo_rescue_stall_patience = 400,
            topo_rescue_max_swap       = 3,
            topo_rescue_cool_down      = 250,
        )
    elif target_ebops <= 9000:
        return dict(
            warmup_ebops_mul    = 1.5,
            budget_decay_epochs = 1200,
            min_degree          = 3,
            phase1_epochs       = 2000,
            phase2_epochs       = 1500,
            phase1_lr           = 2e-3,
            phase2_lr           = 4e-4,
            phase1_lr_cycle     = 1000,
            phase2_lr_cycle     = 500,
            beta_stall_patience = 800,
            beta_recover_epochs = 200,
            beta_recover_floor  = 5e-6,
            beta_max_restarts   = 6,
            earlystop_patience  = 2000,
            phase1_margin       = 0.12,
            phase2_margin       = 0.05,
            topo_rescue_enabled        = True,
            topo_rescue_stall_patience = 500,
            topo_rescue_max_swap       = 3,
            topo_rescue_cool_down      = 300,
            topo_rescue_min_degree     = 3,
        )
    else:
        return dict(
            warmup_ebops_mul    = 1.2,
            budget_decay_epochs = 1000,
            min_degree          = 4,
            phase1_epochs       = 1500,
            phase2_epochs       = 1000,
            phase1_lr           = 1.5e-3,
            phase2_lr           = 3e-4,
            phase1_lr_cycle     = 750,
            phase2_lr_cycle     = 500,
            beta_stall_patience = 800,
            earlystop_patience  = 1500,
            phase1_margin       = 0.10,
            phase2_margin       = 0.03,
            topo_rescue_enabled        = True,
            topo_rescue_stall_patience = 500,
            topo_rescue_max_swap       = 2,
            topo_rescue_cool_down      = 300,
            topo_rescue_min_degree     = 4,
        )


get_target_overrides = get_target_overrides_v3


# ═══════════════════════════════════════════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description='FPL2-JSC v3 (TopologyRescue): Spectral pruning + Beta schedule Pareto search')

parser.add_argument('--pretrained', type=str,
                    default='pretrained_weight/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras',
                    help='Pretrained weight path')
parser.add_argument('--target_ebops', type=float, default=DEFAULT_CONFIG['target_ebops'],
                    help='Target eBOPs (default: 400)')
parser.add_argument('--init_bw', type=int, default=DEFAULT_CONFIG['init_bw'])
parser.add_argument('--init_bw_a', type=int, default=DEFAULT_CONFIG['init_bw_a'])
parser.add_argument('--phase1_epochs', type=int, default=None,
                    help='Phase 1 epochs (default: auto from target overrides)')
parser.add_argument('--phase2_epochs', type=int, default=None,
                    help='Phase 2 epochs (default: auto from target overrides)')
parser.add_argument('--phase1_lr', type=float, default=None)
parser.add_argument('--phase2_lr', type=float, default=None)
parser.add_argument('--min_degree', type=int, default=None)
parser.add_argument('--plot_interval', type=int, default=None)
parser.add_argument('--revival', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_CONFIG['revival_enabled'])
parser.add_argument('--revival_interval', type=int, default=DEFAULT_CONFIG['revival_interval'])
parser.add_argument('--revival_max_per_layer', type=int, default=DEFAULT_CONFIG['revival_max_per_layer'])
parser.add_argument('--swap_kill', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_CONFIG['revival_swap_kill'])
parser.add_argument('--topo_rescue', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_CONFIG['topo_rescue_enabled'],
                    help='Enable/disable TopologyRescue (default: enabled)')
parser.add_argument('--sweep_ebops', type=str, default=None,
                    help='Comma-separated eBOPs targets for sweep')
parser.add_argument('--output_dir', type=str, default='',
                    help='Output directory (auto-named if empty)')
parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'])

args, _ = parser.parse_known_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def make_lr_scheduler(lr_init, cycle, mmul, warmup_epochs=0, offset=0):
    """Cosine decay with warm restarts + optional linear warmup + epoch offset."""
    fn = cosine_decay_restarts_schedule(lr_init, cycle, t_mul=1.0, m_mul=mmul,
                                        alpha=1e-6, alpha_steps=50)
    def schedule(epoch):
        effective_epoch = max(0, epoch - offset)
        lr = fn(effective_epoch)
        if warmup_epochs > 0 and effective_epoch < warmup_epochs:
            lr *= (effective_epoch + 1) / warmup_epochs
        return lr
    return LearningRateScheduler(lambda epoch: schedule(epoch))


def print_bk_stats(mdl, label=''):
    """打印模型 kq.b 统计信息。"""
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
        dead = arr[arr <= 0.1]
        if len(active) > 0:
            print(f'  [bk {label}]  mean={arr.mean():.3f}  '
                  f'dead={len(dead)}/{len(arr)} ({100*len(dead)/len(arr):.1f}%)  '
                  f'active_mean={active.mean():.3f}')
        else:
            print(f'  [bk {label}]  ALL DEAD!')


# ═══════════════════════════════════════════════════════════════════════════════
# 核心管线
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict):
    """完整管线：加载预训练权重 → 谱剪枝 → 两阶段训练 → Pareto 搜索。"""

    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    init_bw      = cfg['init_bw']
    init_bw_a    = cfg.get('init_bw_a') or init_bw
    target_ebops = cfg['target_ebops']
    phase1_epochs = cfg['phase1_epochs']
    phase2_epochs = cfg['phase2_epochs']
    total_epochs  = phase1_epochs + phase2_epochs
    revival_b_val = cfg.get('revival_b_val') or float(init_bw)

    # 渐进式预算参数
    warmup_ebops_mul = cfg.get('warmup_ebops_mul', 5.0)
    warmup_ebops = target_ebops * warmup_ebops_mul
    budget_decay_epochs = cfg.get('budget_decay_epochs', 2000)

    # 投影器模式
    use_projector = cfg.get('use_projector', False)

    output_folder = cfg.get('output_dir', '') or f'results/v3_ebops{int(target_ebops)}_{init_bw}bit/'
    os.makedirs(output_folder, exist_ok=True)
    device = get_tf_device()
    batch_size = 33200

    # TopologyRescue 状态
    topo_rescue_enabled = cfg.get('topo_rescue_enabled', True)

    print('=' * 72)
    print(f'  FPL2-JSC v3 (TopologyRescue)  |  target_eBOPs={target_ebops:.0f}  init_bw={init_bw}-bit')
    print(f'  Total epochs: {total_epochs} (P1={phase1_epochs}, P2={phase2_epochs})')
    print(f'  Pretrained: {cfg["pretrained"]}')
    print(f'  Progressive budget: {warmup_ebops:.0f} → {target_ebops:.0f} over {budget_decay_epochs} ep')
    print(f'  Phase 1: {phase1_epochs} ep, lr={cfg["phase1_lr"]:.1e}, '
          f'beta=[{cfg["phase1_beta_min"]:.0e},{cfg["phase1_beta_max"]:.0e}] init={cfg["phase1_beta_init"]:.0e}, '
          f'margin=±{cfg["phase1_margin"]*100:.0f}%')
    print(f'  Phase 2: {phase2_epochs} ep, lr={cfg["phase2_lr"]:.1e}, margin=±{cfg["phase2_margin"]*100:.0f}%')
    print(f'  Gradient clip: clipnorm={cfg.get("clipnorm", 1.0):.1f}  '
          f'LR warmup: {cfg.get("lr_warmup", 0)} ep')
    print(f'  Pruning: spectral_quant  min_deg={cfg["min_degree"]}')
    print(f'  Beta curriculum: {cfg.get("beta_curriculum_enabled", True)}  '
          f'Adaptive LR: {cfg.get("adaptive_lr_enabled", True)}')
    print(f'  TopologyRescue: {"enabled" if topo_rescue_enabled else "disabled"}'
          + (f'  stall={cfg["topo_rescue_stall_patience"]}, '
             f'max_swap={cfg["topo_rescue_max_swap"]}, '
             f'swap_kill={cfg["topo_rescue_swap_kill"]}, '
             f'cool_down={cfg["topo_rescue_cool_down"]}'
             if topo_rescue_enabled else ''))
    if use_projector:
        print(f'  ★ Projector mode: beta=0, EBOPsConstantProjector controls eBOPs')
    kd_enabled = cfg.get('kd_enabled', False)
    kd_alpha = cfg.get('kd_alpha', 0.3)
    kd_temperature = cfg.get('kd_temperature', 4.0)
    if kd_enabled:
        print(f'  KD: enabled, α={kd_alpha}, T={kd_temperature}')
    print(f'  Revival: enabled={cfg["revival_enabled"]}')
    print(f'  Output: {output_folder}')
    print('=' * 72)

    t0 = time.time()

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/5] Loading dataset...')
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset.h5')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(data_path, src='openml')
    _sample = tf.constant(X_train[:min(2048, len(X_train))], dtype=tf.float32)

    # ── 2. 加载预训练模型 ────────────────────────────────────────────────────
    print(f'\n[2/5] Loading pretrained model from {cfg["pretrained"]}...')
    import model.model as _model_mod  # noqa: F401
    model = keras.models.load_model(cfg['pretrained'], compile=False)

    if kd_enabled:
        print(f'  [KD] Computing teacher soft labels (T={kd_temperature}, α={kd_alpha})...')
        teacher_model = keras.models.load_model(cfg['pretrained'], compile=False)
        teacher_logits = teacher_model.predict(X_train, batch_size=batch_size, verbose=0)
        teacher_probs = tf.nn.softmax(teacher_logits / kd_temperature).numpy().astype(np.float32)
        num_classes = teacher_probs.shape[-1]
        y_onehot = tf.one_hot(y_train, num_classes).numpy().astype(np.float32)
        y_soft = ((1.0 - kd_alpha) * y_onehot + kd_alpha * teacher_probs).astype(np.float32)
        y_val_oh = tf.one_hot(y_val, num_classes).numpy().astype(np.float32)
        dataset_train = Dataset(X_train, y_soft, batch_size, device, shuffle=True)
        dataset_val = Dataset(X_val, y_val_oh, batch_size, device)
        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        del teacher_model
    else:
        dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
        dataset_val = Dataset(X_val, y_val, batch_size, device)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.summary()
    print_bk_stats(model, 'before pruning')

    pretrained_ebops = compute_model_ebops(model, _sample)
    print(f'  Pretrained eBOPs: {pretrained_ebops:.0f}')

    # ── 3. 剪枝到 warmup_ebops ──────────────────────────────────────────────
    pruning_method = cfg.get('pruning_method', 'auto')
    use_sensitivity = cfg.get('use_sensitivity_pruner', False)

    if pruning_method in ('random', 'magnitude'):
        from utils.pruning import random_prune_to_ebops, magnitude_prune_to_ebops
        print(f'\n[3/5] {pruning_method} pruning → warmup_eBOPs={warmup_ebops:.0f} '
              f'(final target={target_ebops:.0f})')
        if pruning_method == 'random':
            random_prune_to_ebops(model, warmup_ebops, _sample, verbose=True)
        else:
            magnitude_prune_to_ebops(model, warmup_ebops, _sample, verbose=True)
        print_bk_stats(model, f'after {pruning_method}')
        used_structured = False
    elif use_sensitivity or pruning_method == 'sensitivity':
        print(f'\n[3/5] SensitivityAwarePruner → warmup_eBOPs={warmup_ebops:.0f} '
              f'(final target={target_ebops:.0f})')
        pruner = SensitivityAwarePruner(
            target_ebops=warmup_ebops,
            pruned_threshold=0.1,
            b_k_min=0.3,
        )
        pruner.prune_to_ebops(model, current_ebops=pretrained_ebops, verbose=True)
        post_prune_e = compute_model_ebops(model, _sample)
        print(f'  Post SensitivityAwarePruner: eBOPs={post_prune_e:.0f}')
        print_bk_stats(model, 'after SAP')
        used_structured = False
    else:
        print(f'\n[3/5] Spectral-quant pruning → warmup_eBOPs={warmup_ebops:.0f} '
              f'(final target={target_ebops:.0f})')
        _, used_structured = spectral_quant_prune_to_ebops(
            model,
            target_ebops=warmup_ebops,
            sample_input=_sample,
            min_degree=cfg['min_degree'],
            b_floor=0.35,
            b_ceiling=6.0,
            verbose=True,
        )
        print_bk_stats(model, 'after spectral_quant')

    _bisect_bk_min = 0.35 if not use_sensitivity else 0.01
    calibrated_ebops = bisect_ebops_to_target(
        model,
        target_ebops=warmup_ebops,
        sample_input=_sample,
        tolerance=0.04,
        max_iter=24,
        b_k_min=_bisect_bk_min,
        allow_connection_kill=not used_structured and not use_sensitivity,
    )

    pruned_ebops = compute_model_ebops(model, _sample)
    print(f'  After pruning + calibration: eBOPs={pruned_ebops:.0f} '
          f'(warmup={warmup_ebops:.0f}, final_target={target_ebops:.0f})')
    print_bk_stats(model, 'post-pruning')

    plot_topology(model, output_folder, 'init_after_pruning', ebops=pruned_ebops, plot_matrix=False)

    init_path = os.path.join(output_folder, f'pruned_init_{init_bw}bit.keras')
    model.save(init_path)
    print(f'  Pruned init saved: {init_path}')

    # ── 4. Phase 1: 恢复训练 + 渐进压缩 ─────────────────────────────────────
    print(f'\n[4/5] Phase 1: Recovery + {"Projector" if use_projector else "Progressive Compression"} ({phase1_epochs} epochs)')

    _set_all_beta(model, cfg['phase1_beta_init'])

    if use_projector:
        _set_all_beta(model, 0.0)
        phase1_projector = EBOPsConstantProjector(
            target_ebops=target_ebops,
            b_k_min=cfg.get('proj_b_k_min', 0.5),
            b_k_max=8.0,
            pruned_threshold=0.1,
            start_epoch=0,
            alpha_gamma=cfg.get('proj_alpha_gamma', 0.5),
            alpha_min=cfg.get('proj_alpha_min', 0.80),
            alpha_max=cfg.get('proj_alpha_max', 1.25),
            ema_alpha=0.3,
            project_activation=False,
            log_scale=False,
        )
        class _FakeBudget:
            beta_current = 0.0
        phase1_budget = _FakeBudget()
        prog_budget = None
    else:
        phase1_projector = None
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

    _probe_n = min(512, len(X_train))
    _probe_x = tf.constant(X_train[:_probe_n], dtype=tf.float32)
    _probe_y = tf.constant(y_train[:_probe_n], dtype=tf.int64)

    revival_cb = None
    if cfg['revival_enabled']:
        revival_cb = SpectralGradientRevivalCallback(
            target_ebops=target_ebops,
            probe_x=_probe_x,
            probe_y=_probe_y,
            min_degree=cfg['min_degree'],
            revival_b_val=revival_b_val,
            max_revival_per_layer=cfg['revival_max_per_layer'],
            revival_interval=cfg['revival_interval'],
            ebops_deficit_threshold=0.20,
            dead_fraction_threshold=0.85,
            grad_min_threshold=0.0,
            cool_down=cfg['revival_interval'] // 2,
            swap_kill=cfg['revival_swap_kill'],
        )

    # ★★★ TopologyRescue 替代 SoftDeathFloor ★★★
    topo_rescue_cb = None
    if topo_rescue_enabled:
        topo_rescue_cb = TopologyRescueCallback(
            revival_b_val=cfg.get('topo_rescue_revival_b', 1.0),
            check_interval=cfg.get('topo_rescue_check_interval', 50),
            stall_patience=cfg.get('topo_rescue_stall_patience', 300),
            min_delta=cfg.get('topo_rescue_min_delta', 5e-5),
            max_swap_per_layer=cfg.get('topo_rescue_max_swap', 4),
            swap_budget_neutral=cfg.get('topo_rescue_swap_kill', True),
            min_degree=cfg.get('topo_rescue_min_degree', cfg['min_degree']),
            cool_down=cfg.get('topo_rescue_cool_down', 200),
            max_interventions=cfg.get('topo_rescue_max_interventions', 10),
            alive_threshold=cfg.get('topo_rescue_alive_threshold', 0.5),
            decay_factor=cfg.get('topo_rescue_decay_factor', 0.85),
            kernel_init_scale=0.01,
        )

    act_fixer = ActivationBitsFixer(b_a_fixed=float(init_bw_a), start_epoch=0)

    ebops_cb = FreeEBOPs()
    pareto_cb = ParetoFront(
        output_folder,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )
    trace_cb = TrainingTraceToH5(output_dir=output_folder, filename='training_trace.h5', max_bits=8)

    plot_interval = cfg.get('plot_interval', 500)
    topo_plot_cb = TopologyPlotCallback(
        output_dir=output_folder,
        plot_interval=plot_interval,
        sample_input=_sample,
        plot_matrix=True,
    )

    clipnorm = cfg.get('clipnorm', 1.0)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg['phase1_lr'], clipnorm=clipnorm),
        loss=loss_fn,
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    warmup = cfg.get('lr_warmup', 100)
    phase1_callbacks = [
        ebops_cb,
        act_fixer,
        pareto_cb,
        make_lr_scheduler(cfg['phase1_lr'], cfg['phase1_lr_cycle'], cfg['phase1_lr_mmul'],
                          warmup_epochs=warmup),
        trace_cb,
        topo_plot_cb,
    ]
    if use_projector:
        phase1_callbacks.insert(1, phase1_projector)
    else:
        phase1_callbacks.insert(1, phase1_budget)
        phase1_callbacks.insert(2, prog_budget)
    if revival_cb is not None:
        phase1_callbacks.insert(4, revival_cb)
    if topo_rescue_cb is not None:
        phase1_callbacks.append(topo_rescue_cb)

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

    if cfg.get('adaptive_lr_enabled', True):
        p1_lr_scaler = AdaptiveLRBiwidthScaler(
            bk_threshold=cfg.get('adaptive_lr_bk_threshold', 2.0),
            scale_power=cfg.get('adaptive_lr_scale_power', 0.5),
            lr_max_factor=cfg.get('adaptive_lr_max_factor', 4.0),
            log=False,
        )
        phase1_callbacks.append(p1_lr_scaler)

    print('  Starting Phase 1...')
    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=phase1_epochs,
        callbacks=phase1_callbacks,
        verbose=1,
    )

    phase1_ebops = compute_model_ebops(model, _sample)
    phase1_res = model.evaluate(dataset_val, verbose=0)
    phase1_beta = getattr(phase1_budget, 'beta_current', 0.0)
    print(f'\n  Phase 1 done: val_acc={phase1_res[1]:.4f}  eBOPs={phase1_ebops:.0f}  beta={phase1_beta:.2e}')
    print_bk_stats(model, 'end-of-Phase1')

    # ── 5. Phase 2: 精调 + Pareto 搜索 ───────────────────────────────────────
    if phase2_epochs <= 0:
        print(f'\n[5/5] Phase 2: SKIPPED (phase2_epochs=0)')
    else:
        print(f'\n[5/5] Phase 2: Fine-tuning + Pareto search ({phase2_epochs} epochs)')

        if use_projector:
            _set_all_beta(model, 0.0)
            phase2_projector = EBOPsConstantProjector(
                target_ebops=target_ebops,
                b_k_min=cfg.get('proj_b_k_min', 0.5),
                b_k_max=8.0,
                pruned_threshold=0.1,
                start_epoch=phase1_epochs,
                alpha_gamma=cfg.get('proj_alpha_gamma', 0.5),
                alpha_min=cfg.get('proj_alpha_min', 0.80),
                alpha_max=cfg.get('proj_alpha_max', 1.25),
                ema_alpha=0.3,
                project_activation=False,
                log_scale=False,
            )
            phase2_budget = None
        else:
            phase2_projector = None
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

        revival_cb_p2 = None
        if cfg['revival_enabled']:
            revival_cb_p2 = SpectralGradientRevivalCallback(
                target_ebops=target_ebops,
                probe_x=_probe_x,
                probe_y=_probe_y,
                min_degree=cfg['min_degree'],
                revival_b_val=revival_b_val,
                max_revival_per_layer=cfg['revival_max_per_layer'],
                revival_interval=cfg['revival_interval'] * 2,
                ebops_deficit_threshold=0.15,
                dead_fraction_threshold=0.90,
                grad_min_threshold=0.0,
                cool_down=cfg['revival_interval'],
                swap_kill=cfg['revival_swap_kill'],
            )

        # Phase2 TopologyRescue: 更保守参数
        topo_rescue_cb_p2 = None
        if topo_rescue_enabled:
            p2_stall = int(cfg.get('topo_rescue_stall_patience', 300) * 1.5)
            p2_max_swap = max(1, cfg.get('topo_rescue_max_swap', 4) - 1)
            p2_cool_down = int(cfg.get('topo_rescue_cool_down', 200) * 1.5)
            topo_rescue_cb_p2 = TopologyRescueCallback(
                revival_b_val=cfg.get('topo_rescue_revival_b', 1.0),
                check_interval=cfg.get('topo_rescue_check_interval', 50),
                stall_patience=p2_stall,
                min_delta=cfg.get('topo_rescue_min_delta', 5e-5),
                max_swap_per_layer=p2_max_swap,
                swap_budget_neutral=cfg.get('topo_rescue_swap_kill', True),
                min_degree=cfg.get('topo_rescue_min_degree', cfg['min_degree']),
                cool_down=p2_cool_down,
                max_interventions=cfg.get('topo_rescue_max_interventions', 10) // 2,
                alive_threshold=cfg.get('topo_rescue_alive_threshold', 0.5),
                decay_factor=cfg.get('topo_rescue_decay_factor', 0.85),
                kernel_init_scale=0.01,
            )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg['phase2_lr'], clipnorm=clipnorm),
            loss=loss_fn,
            metrics=['accuracy'],
            steps_per_execution=32,
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
            act_fixer,
            pareto_cb,
            make_lr_scheduler(cfg['phase2_lr'], cfg['phase2_lr_cycle'], cfg['phase2_lr_mmul'],
                              offset=phase1_epochs),
            trace_cb,
            early_stop,
            topo_plot_cb,
        ]
        if use_projector:
            phase2_callbacks.insert(1, phase2_projector)
        else:
            phase2_callbacks.insert(1, phase2_budget)
        if revival_cb_p2 is not None:
            phase2_callbacks.insert(3, revival_cb_p2)
        if topo_rescue_cb_p2 is not None:
            phase2_callbacks.append(topo_rescue_cb_p2)

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
        'revival_enabled': cfg['revival_enabled'],
        'topo_rescue_enabled': topo_rescue_enabled,
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
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    cfg = dict(DEFAULT_CONFIG)
    cfg['pretrained']      = args.pretrained
    cfg['target_ebops']    = args.target_ebops
    cfg['init_bw']         = args.init_bw
    cfg['init_bw_a']       = args.init_bw_a
    cfg['output_dir']      = args.output_dir
    cfg['seed']            = args.seed
    cfg['revival_enabled']       = args.revival
    cfg['revival_interval']      = args.revival_interval
    cfg['revival_max_per_layer'] = args.revival_max_per_layer
    cfg['revival_swap_kill']     = args.swap_kill
    cfg['topo_rescue_enabled']   = args.topo_rescue

    # 先应用 target overrides
    overrides = get_target_overrides_v3(cfg['target_ebops'])
    cfg.update(overrides)

    # 命令行显式参数优先于 target overrides
    if args.phase1_epochs is not None:
        cfg['phase1_epochs'] = args.phase1_epochs
    if args.phase2_epochs is not None:
        cfg['phase2_epochs'] = args.phase2_epochs
    if args.phase1_lr is not None:
        cfg['phase1_lr'] = args.phase1_lr
    if args.phase2_lr is not None:
        cfg['phase2_lr'] = args.phase2_lr
    if args.min_degree is not None:
        cfg['min_degree'] = args.min_degree
    if args.plot_interval is not None:
        cfg['plot_interval'] = args.plot_interval

    if args.sweep_ebops:
        targets = [float(x.strip()) for x in args.sweep_ebops.split(',')]
        print(f'\n{"#" * 72}')
        print(f'  FPL2-JSC v3 — eBOPs SWEEP: {targets}')
        print(f'{"#" * 72}\n')

        all_results = []
        for tgt in targets:
            cfg_copy = dict(cfg)
            cfg_copy['target_ebops'] = tgt
            cfg_copy['output_dir'] = ''
            ov = get_target_overrides_v3(tgt)
            if ov:
                cfg_copy.update(ov)
            result = run(cfg_copy)
            all_results.append(result)

        sweep_path = 'results/v3_sweep_summary.json'
        os.makedirs('results', exist_ok=True)
        with open(sweep_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nSweep summary: {sweep_path}')

        print(f'\n{"=" * 72}')
        print(f'  {"target":>8s}  {"epochs":>7s}  {"pruned":>8s}  {"final":>8s}  '
              f'{"best_acc":>8s}  {"@target":>8s}  {"time":>6s}')
        print(f'  {"-" * 60}')
        for r in sorted(all_results, key=lambda x: x['target_ebops']):
            print(f'  {r["target_ebops"]:8.0f}  {r.get("total_epochs", 0):7d}  '
                  f'{r.get("pruned_ebops", 0):8.0f}  {r.get("final_ebops", 0):8.0f}  '
                  f'{r.get("best_val_acc", 0):8.4f}  '
                  f'{r.get("best_acc_at_target", 0):8.4f}  '
                  f'{r.get("elapsed_sec", 0):5.0f}s')
        print(f'{"=" * 72}')

    else:
        run(cfg)
