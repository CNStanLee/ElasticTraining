"""
FPL2/jsc — 谱约束一次性剪枝 + RigL 复活 + Beta 调度 Pareto 搜索
================================================================

Pipeline:
  1. 加载预训练权重 (pretrained_weight/)
  2. 谱约束一次性剪枝 (保留 1-bit 下的必要谱连接)
  3. 使用 RigL 风格连接复活 + Beta 调度器在目标 eBOPs 附近训练优化
  4. ParetoFront 自动保存 Pareto 最优模型

用法:
  python run_all.py                          # 默认 1-bit, target_ebops=3000
  python run_all.py --target_ebops 5000      # 指定目标 eBOPs
  python run_all.py --init_bw 2              # 2-bit 剪枝
  python run_all.py --sweep_ebops 2000,3000,5000  # 扫描多个 eBOPs 目标
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
    bisect_ebops_to_target,
    compute_model_ebops,
    BetaOnlyBudgetController,
    ActivationBitsFixer,
    SoftDeathFloor,
    ProgressiveBudgetController,
    BetaCurriculumController,
    AdaptiveLRBiwidthScaler,
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
# 默认配置
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    # 剪枝
    target_ebops     = 1500,    # 目标 eBOPs (剪枝 + 训练的约束目标)
    min_degree       = 3,       # spectral_quant 列最小度 (谱连通保证)

    # 渐进式预算 (避免一次性剪枝冲击)
    warmup_ebops_mul = 2.0,     # 初始剪枝目标 = target * mul (留更多初始容量)
    budget_decay_epochs = 3000, # 从 warmup 指数衰减到 target 的 epoch 数

    # 位宽 (仅影响 activation / revival，剪枝位宽由 spectral_quant 自动决定)
    init_bw          = 1,       # 活跃连接默认位宽 (revival 新连接用)
    init_bw_a        = 3,       # Activation bitwidth (None = 和 init_bw 相同)

    # Phase 1: 恢复训练 + 连接复活
    phase1_epochs    = 6000,    # 增加: 需要更多时间完成渐进压缩
    phase1_lr        = 2e-3,    # 对齐 reference
    phase1_lr_cycle  = 2000,
    phase1_lr_mmul   = 0.9,
    phase1_beta_init = 1e-5,    # 对齐 reference (从更低 beta 开始)
    phase1_beta_min  = 1e-8,
    phase1_beta_max  = 5e-4,    # 对齐 reference (提高上限)
    phase1_margin    = 0.15,    # 对齐 reference

    # Phase 2: 精调
    phase2_epochs    = 12000,   # 增加: 更充分的精调时间
    phase2_lr        = 5e-4,
    phase2_lr_cycle  = 800,
    phase2_lr_mmul   = 0.95,
    phase2_beta_min  = 1e-8,
    phase2_beta_max  = 5e-4,
    phase2_margin    = 0.05,    # 紧 margin: 对齐 reference

    # 梯度裁剪 + LR 预热
    clipnorm           = 1.0,   # 梯度范数裁剪, 防止单步灾变
    lr_warmup          = 100,   # LR 线性预热 epoch 数

    # 软死亡下限（防止连接永久死亡）
    soft_floor_b       = 0.05,   # 死连接的 kq.b 下限
    soft_floor_alive   = 0.4,    # b >= 此值视为存活
    soft_floor_every   = 50,     # 每 N epochs 执行一次 (降低干扰)

    # Beta 课程重启 (打破 acc 停滞死锁)
    beta_curriculum_enabled   = True,
    beta_stall_patience       = 800,   # acc 停滞多少 ep 后触发
    beta_recover_epochs       = 300,   # RECOVER 降 beta 多少 ep
    beta_restart_decay        = 0.25,  # 重启 beta 缩放因子
    beta_max_restarts         = 8,
    beta_recover_floor        = None,  # RECOVER 期间 beta 下限 (None=auto=beta_min)

    # 自适应 LR 缩放 (补偿低位宽 STE 噪声)
    adaptive_lr_enabled       = True,
    adaptive_lr_bk_threshold  = 2.0,
    adaptive_lr_scale_power   = 0.5,
    adaptive_lr_max_factor    = 4.0,

    # 连接复活 (默认关闭: reference 不用 revival 也达 0.748)
    revival_enabled        = False,
    revival_interval       = 200,
    revival_max_per_layer  = 8,
    revival_b_val          = None,  # None = 和 init_bw 相同
    revival_swap_kill      = False,

    # 早停
    earlystop_patience = 5000,

    # 知识蒸馏 (teacher soft label) — 关闭: 在极低预算下效果不明显且增加复杂度
    kd_enabled         = False,
    kd_alpha           = 0.3,
    kd_temperature     = 4.0,

    # 拓扑绘图
    plot_interval      = 1000,  # 每 N epochs 绘制一次拓扑图

    seed = 42,
)


# ── 不同 eBOPs 目标的专属配置 ─────────────────────────────────────────────────
# 根据目标 eBOPs 范围, 覆盖 DEFAULT_CONFIG 中的关键超参数
# 设计原则:
#   - 越低的 eBOPs → 需要越多的 warmup 缓冲、越长的训练、越松的剪枝约束
#   - 越高的 eBOPs → 压缩压力小, 更短训练即可收敛
# 参考基准:
#   400 → 0.706, 1500 → 0.749, 2500 → 0.757, 6800 → 0.767, 12000 → ~0.770

def get_target_overrides(target_ebops: float) -> dict:
    """返回针对特定 eBOPs 目标的配置覆盖项。"""
    if target_ebops <= 500:
        # 极端低预算: 仅保留 ~2% 连接
        # 关键: Phase 2 必须极度保守 — 没有 SpectralReg/EdgeRewiring 保护
        # 稍有不慎 beta 就会杀死仅存的连接 → acc 崩塌
        return dict(
            warmup_ebops_mul    = 3.0,       # 从 1200 渐降到 400
            budget_decay_epochs = 4000,      # 更长衰减
            min_degree          = 2,         # 谱保证放松
            phase1_epochs       = 8000,      # 更多恢复时间
            phase2_epochs       = 8000,      # 保守: 不需要太长
            phase1_lr           = 2e-3,
            phase2_lr           = 3e-4,      # Phase2 低 LR: 保护脆弱连接
            phase1_lr_cycle     = 2000,
            phase2_lr_cycle     = 600,
            phase1_beta_max     = 2e-4,      # Phase1 限制 beta 上限
            phase2_beta_max     = 5e-6,      # ★ Phase2 极低 beta 上限: 仅轻微约束
            beta_stall_patience = 600,       # 更快检测停滞
            beta_recover_epochs = 200,       # 短恢复
            beta_recover_floor  = 1e-7,      # RECOVER 期间保留微弱压力
            beta_max_restarts   = 10,
            beta_curriculum_p2  = False,     # ★ Phase2 禁用课程重启 (会摧毁模型)
            earlystop_patience  = 6000,
            phase1_margin       = 0.25,      # 宽 margin 
            phase2_margin       = 0.15,      # Phase2 也宽: 容忍波动
            phase2_adjust_factor = 1.05,     # ★ Phase2 极弱 beta 调节
        )
    elif target_ebops <= 2000:
        # 低预算 (1000-2000): 当前默认配置已针对 1500 优化
        return dict()  # 使用默认
    elif target_ebops <= 4000:
        # 中等预算 (2000-4000)
        return dict(
            warmup_ebops_mul    = 2.0,
            budget_decay_epochs = 2500,
            phase1_epochs       = 5000,
            phase2_epochs       = 10000,
            phase1_lr_cycle     = 1800,
            phase2_lr_cycle     = 700,
        )
    elif target_ebops <= 9000:
        # 高预算 (4000-9000): 压缩压力较小
        # 关键修复: RECOVER 期间必须保留 beta 压力, 否则 eBOPs 爆炸到 17k+
        return dict(
            warmup_ebops_mul    = 1.5,       # 不需要太大缓冲
            budget_decay_epochs = 2000,
            min_degree          = 3,
            phase1_epochs       = 5000,      # 增加: 给更多时间压缩到目标
            phase2_epochs       = 12000,     # 增加: 在目标附近充分精调
            phase1_lr           = 2e-3,
            phase2_lr           = 4e-4,
            phase1_lr_cycle     = 1500,
            phase2_lr_cycle     = 600,
            beta_stall_patience = 1200,      # 稍长 patience, 避免过早触发
            beta_recover_epochs = 200,       # 短 RECOVER: 长了 eBOPs 会drift
            beta_recover_floor  = 5e-6,      # ★ RECOVER 期间保持压力!
            beta_max_restarts   = 6,
            earlystop_patience  = 5000,
            phase1_margin       = 0.12,      # 稍紧 margin
            phase2_margin       = 0.05,
        )
    else:
        # 极高预算 (>9000): 接近原始模型, 轻微压缩
        return dict(
            warmup_ebops_mul    = 1.2,       # 12000 * 1.2 = 14400, 远低于 pretrained ~20k
            budget_decay_epochs = 1500,
            min_degree          = 4,
            phase1_epochs       = 3000,
            phase2_epochs       = 6000,
            phase1_lr           = 1.5e-3,
            phase2_lr           = 3e-4,
            phase1_lr_cycle     = 1000,
            phase2_lr_cycle     = 500,
            beta_stall_patience = 1200,
            earlystop_patience  = 3000,
            phase1_margin       = 0.10,
            phase2_margin       = 0.03,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description='FPL2-JSC: Spectral pruning + RigL revival Pareto search')

parser.add_argument('--pretrained', type=str,
                    default='pretrained_weight/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras',
                    help='Pretrained weight path')
parser.add_argument('--target_ebops', type=float, default=DEFAULT_CONFIG['target_ebops'],
                    help='Target eBOPs for pruning + training (default: 1500)')
parser.add_argument('--init_bw', type=int, default=DEFAULT_CONFIG['init_bw'],
                    help='Revival connection bitwidth (default: 1)')
parser.add_argument('--init_bw_a', type=int, default=DEFAULT_CONFIG['init_bw_a'],
                    help='Activation bitwidth (default: same as init_bw)')
parser.add_argument('--phase1_epochs', type=int, default=DEFAULT_CONFIG['phase1_epochs'])
parser.add_argument('--phase2_epochs', type=int, default=DEFAULT_CONFIG['phase2_epochs'])
parser.add_argument('--phase1_lr', type=float, default=DEFAULT_CONFIG['phase1_lr'])
parser.add_argument('--phase2_lr', type=float, default=DEFAULT_CONFIG['phase2_lr'])
parser.add_argument('--min_degree', type=int, default=DEFAULT_CONFIG['min_degree'],
                    help='Minimum column degree for spectral_quant (default: 2)')
parser.add_argument('--plot_interval', type=int, default=DEFAULT_CONFIG['plot_interval'],
                    help='Plot topology every N epochs (default: 1000)')
parser.add_argument('--revival', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_CONFIG['revival_enabled'])
parser.add_argument('--revival_interval', type=int, default=DEFAULT_CONFIG['revival_interval'])
parser.add_argument('--revival_max_per_layer', type=int, default=DEFAULT_CONFIG['revival_max_per_layer'])
parser.add_argument('--swap_kill', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_CONFIG['revival_swap_kill'])
parser.add_argument('--sweep_ebops', type=str, default=None,
                    help='Comma-separated eBOPs targets for sweep, e.g. "2000,3000,5000"')
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
    warmup_ebops_mul = cfg.get('warmup_ebops_mul', 2.0)
    warmup_ebops = target_ebops * warmup_ebops_mul
    budget_decay_epochs = cfg.get('budget_decay_epochs', 3000)

    output_folder = cfg.get('output_dir', '') or f'results/ebops{int(target_ebops)}_{init_bw}bit/'
    os.makedirs(output_folder, exist_ok=True)
    device = get_tf_device()
    batch_size = 33200

    print('=' * 72)
    print(f'  FPL2-JSC  |  target_eBOPs={target_ebops:.0f}  init_bw={init_bw}-bit')
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

    # ── 2. 加载预训练模型 + 知识蒸馏 ─────────────────────────────────────────
    print(f'\n[2/5] Loading pretrained model from {cfg["pretrained"]}...')
    import model.model as _model_mod  # noqa: F401  注册自定义层
    model = keras.models.load_model(cfg['pretrained'], compile=False)

    # 知识蒸馏: 用 teacher 的 soft label 替代 hard one-hot
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
        print(f'  [KD] Soft labels ready: shape={y_soft.shape}, '
              f'teacher avg confidence={teacher_probs.max(axis=-1).mean():.3f}')
        del teacher_model  # 释放内存
    else:
        dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
        dataset_val = Dataset(X_val, y_val, batch_size, device)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.summary()
    print_bk_stats(model, 'before pruning')

    pretrained_ebops = compute_model_ebops(model, _sample)
    print(f'  Pretrained eBOPs: {pretrained_ebops:.0f}')

    # ── 3. 谱约束剪枝 (到 warmup_ebops，而非直接到 target) ─────────────────
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

    # 二分校准位宽精确命中 warmup 目标
    calibrated_ebops = bisect_ebops_to_target(
        model,
        target_ebops=warmup_ebops,
        sample_input=_sample,
        tolerance=0.03,
        max_iter=24,
        b_k_min=0.35,
        allow_connection_kill=not used_structured,
    )

    pruned_ebops = compute_model_ebops(model, _sample)
    print(f'  After pruning + calibration: eBOPs={pruned_ebops:.0f} '
          f'(warmup={warmup_ebops:.0f}, final_target={target_ebops:.0f})')
    print_bk_stats(model, f'post-pruning')

    # 绘制初始化拓扑图
    plot_topology(model, output_folder, 'init_after_pruning', ebops=pruned_ebops, plot_matrix = False)

    # 保存剪枝后初始权重
    init_path = os.path.join(output_folder, f'pruned_init_{init_bw}bit.keras')
    model.save(init_path)
    print(f'  Pruned init saved: {init_path}')

    # ── 4. Phase 1: 恢复训练 + 渐进压缩 ─────────────────────────
    print(f'\n[4/5] Phase 1: Recovery + Progressive Compression ({phase1_epochs} epochs)')

    _set_all_beta(model, cfg['phase1_beta_init'])

    # Beta 预算控制器 (初始 target = warmup_ebops，由 ProgressiveBudgetController 渐进降低)
    phase1_budget = BetaOnlyBudgetController(
        target_ebops=warmup_ebops,   # 从 warmup 开始，而非直接 target
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

    # 渐进式预算: warmup_ebops → target_ebops 指数衰减
    prog_budget = ProgressiveBudgetController(
        budget_ctrl=phase1_budget,
        warmup_ebops=warmup_ebops,
        final_ebops=target_ebops,
        decay_epochs=budget_decay_epochs,
        start_epoch=0,
    )
    print(f'  [Progressive] {warmup_ebops:.0f} \u2192 {target_ebops:.0f} over {budget_decay_epochs} ep')

    # 连接复活回调
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
        print(f'  [Revival] enabled: interval={cfg["revival_interval"]}  '
              f'max/layer={cfg["revival_max_per_layer"]}  swap_kill={cfg["revival_swap_kill"]}')

    # 软死亡下限：阻止连接永久死亡 (降低频率减少干扰)
    soft_floor = SoftDeathFloor(
        b_floor=cfg.get('soft_floor_b', 0.05),
        alive_threshold=cfg.get('soft_floor_alive', 0.4),
        apply_every=cfg.get('soft_floor_every', 50),
        protect_kernel=True,
        kernel_init_scale=0.01,
    )

    # 激活位宽固定
    act_fixer = ActivationBitsFixer(b_a_fixed=float(init_bw_a), start_epoch=0)

    # 通用 callbacks
    ebops_cb = FreeEBOPs()
    pareto_cb = ParetoFront(
        output_folder,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )
    trace_cb = TrainingTraceToH5(output_dir=output_folder, filename='training_trace.h5', max_bits=8)

    # 拓扑绘图回调 (每 plot_interval epochs 绘制一次)
    plot_interval = cfg.get('plot_interval', 1000)
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
        phase1_budget,
        prog_budget,        # 渐进式预算衰减
        soft_floor,
        act_fixer,
        pareto_cb,
        make_lr_scheduler(cfg['phase1_lr'], cfg['phase1_lr_cycle'], cfg['phase1_lr_mmul'],
                          warmup_epochs=warmup),
        trace_cb,
        topo_plot_cb,
    ]
    if revival_cb is not None:
        phase1_callbacks.insert(4, revival_cb)  # 复活放 soft_floor 之后

    # Beta 课程重启 (打破 acc 停滞)
    if cfg.get('beta_curriculum_enabled', True):
        p1_curriculum = BetaCurriculumController(
            budget_ctrl=phase1_budget,
            stall_patience=cfg.get('beta_stall_patience', 800),
            recover_epochs=cfg.get('beta_recover_epochs', 300),
            min_delta=5e-5,
            restart_decay=cfg.get('beta_restart_decay', 0.25),
            max_restarts=cfg.get('beta_max_restarts', 8),
            recover_beta_floor=cfg.get('beta_recover_floor', None),
        )
        phase1_callbacks.append(p1_curriculum)

    # 自适应 LR 缩放 (补偿低位宽 STE 噪声)
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
    phase1_beta = phase1_budget.beta_current
    print(f'\n  Phase 1 done: val_acc={phase1_res[1]:.4f}  eBOPs={phase1_ebops:.0f}  beta={phase1_beta:.2e}')
    print_bk_stats(model, 'end-of-Phase1')

    # ── 5. Phase 2: 精调 + Pareto 搜索 ───────────────────────────────────────
    print(f'\n[5/5] Phase 2: Fine-tuning + Pareto search ({phase2_epochs} epochs)')

    # 继承 Phase 1 均衡 beta（对齐 reference: 平滑过渡）
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

    # 重新编译（重置 Adam 动量，保持梯度裁剪）
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
        min_epoch=phase1_epochs + 1000,
        restore_best_weights=True,
    )

    phase2_callbacks = [
        ebops_cb,
        phase2_budget,
        soft_floor,
        act_fixer,
        pareto_cb,
        make_lr_scheduler(cfg['phase2_lr'], cfg['phase2_lr_cycle'], cfg['phase2_lr_mmul'],
                          offset=phase1_epochs),
        trace_cb,
        early_stop,
        topo_plot_cb,
    ]
    if revival_cb_p2 is not None:
        phase2_callbacks.insert(3, revival_cb_p2)

    # Beta 课程重启 (Phase 2: 更长 patience; 极低预算可关闭)
    p2_curriculum_enabled = cfg.get('beta_curriculum_p2', cfg.get('beta_curriculum_enabled', True))
    if p2_curriculum_enabled:
        p2_curriculum = BetaCurriculumController(
            budget_ctrl=phase2_budget,
            stall_patience=cfg.get('beta_stall_patience', 800) * 2,
            recover_epochs=cfg.get('beta_recover_epochs', 300),
            min_delta=2e-5,
            restart_decay=cfg.get('beta_restart_decay', 0.25),
            max_restarts=cfg.get('beta_max_restarts', 8),
            recover_beta_floor=cfg.get('beta_recover_floor', None),
        )
        phase2_callbacks.append(p2_curriculum)

    # 自适应 LR 缩放 (Phase 2: 更保守)
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

    # 绘制最终拓扑图
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
    }

    # 从 trace 读取最佳 accuracy
    import h5py
    trace_path = os.path.join(output_folder, 'training_trace.h5')
    if os.path.exists(trace_path):
        with h5py.File(trace_path, 'r') as f:
            va = f['val_accuracy'][:]
            eb = f['ebops'][:]
            result['best_val_acc'] = float(va.max())
            result['best_epoch'] = int(va.argmax())
            result['best_ebops'] = float(eb[va.argmax()])

    print(f'\n  {"=" * 60}')
    print(f'  target={target_ebops:.0f}  pruned={pruned_ebops:.0f}  final={final_ebops:.0f}  '
          f'val_acc={final_res[1]:.4f}  best_acc={result.get("best_val_acc", 0):.4f}  '
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
    cfg['phase1_epochs']   = args.phase1_epochs
    cfg['phase2_epochs']   = args.phase2_epochs
    cfg['phase1_lr']       = args.phase1_lr
    cfg['phase2_lr']       = args.phase2_lr
    cfg['min_degree']      = args.min_degree
    cfg['plot_interval']   = args.plot_interval
    cfg['output_dir']      = args.output_dir
    cfg['seed']            = args.seed
    cfg['revival_enabled']       = args.revival
    cfg['revival_interval']      = args.revival_interval
    cfg['revival_max_per_layer'] = args.revival_max_per_layer
    cfg['revival_swap_kill']     = args.swap_kill

    if args.sweep_ebops:
        # ── Sweep 模式：批量测试不同 eBOPs 目标 ───────────────────────────
        targets = [float(x.strip()) for x in args.sweep_ebops.split(',')]
        print(f'\n{"#" * 72}')
        print(f'  FPL2-JSC — eBOPs SWEEP: {targets}')
        print(f'  init_bw={cfg["init_bw"]}  revival={cfg["revival_enabled"]}')
        print(f'{"#" * 72}\n')

        all_results = []
        for tgt in targets:
            cfg_copy = dict(cfg)
            cfg_copy['target_ebops'] = tgt
            cfg_copy['output_dir'] = ''  # 自动命名
            # 应用该 eBOPs 目标的专属覆盖配置
            overrides = get_target_overrides(tgt)
            if overrides:
                print(f'  [Sweep] Applying {len(overrides)} overrides for target={tgt:.0f}: '
                      f'{list(overrides.keys())}')
            cfg_copy.update(overrides)
            result = run(cfg_copy)
            all_results.append(result)

            print(f'\n  ── Sweep progress ──')
            for r in sorted(all_results, key=lambda x: x['target_ebops']):
                print(f'    target={r["target_ebops"]:.0f}: '
                      f'final={r.get("final_ebops", 0):.0f}  '
                      f'best_acc={r.get("best_val_acc", 0):.4f}')
            print()

        sweep_path = 'results/sweep_summary.json'
        os.makedirs('results', exist_ok=True)
        with open(sweep_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nSweep summary: {sweep_path}')

        print(f'\n{"=" * 72}')
        print(f'  {"target":>8s}  {"pruned":>8s}  {"final":>8s}  {"best_acc":>8s}  {"time":>6s}')
        print(f'  {"-" * 46}')
        for r in sorted(all_results, key=lambda x: x['target_ebops']):
            print(f'  {r["target_ebops"]:8.0f}  {r.get("pruned_ebops", 0):8.0f}  '
                  f'{r.get("final_ebops", 0):8.0f}  '
                  f'{r.get("best_val_acc", 0):8.4f}  {r.get("elapsed_sec", 0):5.0f}s')
        print(f'{"=" * 72}')

    else:
        # ── 单次运行 ─────────────────────────────────────────────────────────
        # 应用该 eBOPs 目标的专属覆盖配置 (与 sweep 一致)
        overrides = get_target_overrides(cfg['target_ebops'])
        if overrides:
            # 只覆盖未被命令行显式修改的参数
            for k, v in overrides.items():
                if k not in cfg or cfg[k] == DEFAULT_CONFIG.get(k):
                    cfg[k] = v
            print(f'  [Auto] Applied {len(overrides)} target overrides for '
                  f'ebops={cfg["target_ebops"]:.0f}')
        run(cfg)
