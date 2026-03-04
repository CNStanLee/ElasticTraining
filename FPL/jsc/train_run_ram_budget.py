"""
Ramanujan BW-aware 初始化 + 恒定 EBOPs 预算训练。

目的
----
验证拉马努金初始化能否在极稀疏 + 极低量化（EBOPs ≈ 300）条件下保证网络
可训练，并达到与 baseline 长训练扫描相当的精度。

对比基线
--------
- baseline 扫描（200k epoch）在 EBOPs≈300 处达到 val_acc≈0.736
- prune-finetune 方案在 EBOPs=300 处完全无法训练（精度 ~0.20）

方法（3D Ramanujan）
-------------------
1. 构建全新 HGQ 模型（不加载预训练权重）
2. 用 compute_3d_ramanujan_allocation 联合优化拓扑和逐连接位宽：
   a. 生成宽松的 Ramanujan d-regular 候选拓扑
   b. 按 |kernel| 为每条候选连接打分
   c. 将 EBOPS 预算按容量权重分配到各层（每层保底 min_per_layer 条连接）
   d. 层内按分数排序剪枝到预算以内，存活连接按重要性分配 b_k ∈ [b_k_viable, b_k_max]
3. 用 apply_3d_ramanujan_init 写入逐连接位宽 + 剪枝
4. 训练时用 BetaOnlyBudgetController（维持预算），无 mask 约束
   - Phase 1（warmup）: beta=0，模型自由学习特征，可重新激活连接
   - Phase 2（compression）: beta 从 0 开始 log ramp，自然压缩 EBOPS
   - Phase 3（budget）: BudgetController 微调 beta 维持 EBOPS ≈ target

与旧版均匀位宽 Ramanujan init 的关键区别
---------------------------------------
- 旧版：d-regular 拓扑所有活跃连接获得相同极低 b_k（~0.1）→ STE 梯度全是噪声
- 本版：少量连接获得 b_k ≥ 1.0（少而精）→ 类似 baseline 自然压缩的分布 → 可训练
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
from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PieceWiseSchedule
from keras.callbacks import LearningRateScheduler
from model.model import get_model_hgq
from utils.train_utils import cosine_decay_restarts_schedule, TrainingTraceToH5
from utils.tf_device import get_tf_device
from utils.ramanujan_budget_utils import (
    compute_3d_ramanujan_allocation,
    apply_3d_ramanujan_init,
    BetaOnlyBudgetController,
    _flatten_layers,
    _get_kq_var,
)

np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

TARGET_EBOPS = 300

input_folder = 'data/dataset.h5'
batch_size   = 33200

# ── 模型初始化位宽 ─────────────────────────────────────────────────────────
INIT_BW_K = 3
INIT_BW_A = 3

# ── 3D Ramanujan 初始化参数 ──────────────────────────────────────────────────
# 三维拉马努金图初始化：联合优化拓扑（哪些连接存活）和逐连接位宽（精度分配）
# 与旧版均匀位宽方案不同，本方案将预算集中到少量高重要性连接上（少而精）
RAM_BASE_MULTIPLIER = 2.0    # 基础 Ramanujan 度 = sqrt(in_dim) * mult
                             # 较高→候选更多→重要性选择粒度更细
RAM_BK_VIABLE = 1.0          # 拓扑选择的计算位宽（决定存活数量）
RAM_BK_MAX    = 6.0          # 最高位宽上限
RAM_INIT_BK   = 3.0          # 存活连接的实际初始 b_k（= 模型原始 f0，保证 Glorot 权重可训练）
                             # 初始 EBOPS > target，由 BetaOnlyBudgetController 自然压缩
RAM_MIN_PER_LAYER = 5        # 每层至少保留的连接数（保证梯度流贯通）
RAM_BUDGET_WEIGHT = 'capacity'  # 按层容量分配预算（大层多分）

# ── 训练参数 ──────────────────────────────────────────────────────────────────
EPOCHS       = 30000
LR           = 5e-3
LR_CYCLE     = 2000
LR_MMUL      = 0.94

# Beta schedule: warmup(beta=0) → log ramp → 恒定
# warmup 阶段 beta=0：让模型自由学习特征，避免 EBOPS 梯度过早压缩 b_k
# （98 条活跃连接的 per-connection EBOPS 远高于 baseline，即使 5e-7 的 beta
#   也会因 Adam momentum 快速将 b_k 推低到不可训练水平）
# !! warmup 用固定值（~300），不与总 epoch 成比例 !!
# 模型在 ~20 epoch 内就能学到基本特征（vacc 0.20→0.34），
# 过长 warmup 会让 EBOPS 膨胀到 10000+，之后难以压缩回 300。
BETA_WARMUP  = 300              # 固定 300 epoch（足够学特征，不过度膨胀）
BETA_MAX     = 3e-3             # 比 baseline 的 1e-3 更大，加速初始压缩阶段

# Budget controller（在 beta schedule 之上叠加 EBOPS 约束）
BUDGET_CTRL_MARGIN = 0.15       # ±15% 容忍带


# ── 命令行参数覆盖 ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='3D Ramanujan init + budget-constrained training')
parser.add_argument('--target_ebops',  type=float, default=TARGET_EBOPS)
parser.add_argument('--epochs',        type=int,   default=EPOCHS)
parser.add_argument('--multiplier',    type=float, default=RAM_BASE_MULTIPLIER)
parser.add_argument('--bk_viable',     type=float, default=RAM_BK_VIABLE)
parser.add_argument('--bk_max',        type=float, default=RAM_BK_MAX)
args, _ = parser.parse_known_args()

TARGET_EBOPS        = args.target_ebops
EPOCHS              = args.epochs
RAM_BASE_MULTIPLIER = args.multiplier
RAM_BK_VIABLE       = args.bk_viable
RAM_BK_MAX          = args.bk_max
BETA_WARMUP    = 300   # 固定值，不随 EPOCHS 变

output_folder = f'results/ram_budget_{int(TARGET_EBOPS)}/'
device = get_tf_device()
os.makedirs(output_folder, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

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
              f'n_zero(<=0.01)={int((arr<=0.01).sum())}/{len(arr)}')


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('  3D Ramanujan Init + Budget-constrained Training')
    print(f'  Target EBOPs   : {TARGET_EBOPS}')
    print(f'  Epochs         : {EPOCHS}')
    print(f'  Base multiplier: {RAM_BASE_MULTIPLIER}  (degree = sqrt(in_dim) * mult)')
    print(f'  b_k viable     : {RAM_BK_VIABLE}  (min bitwidth for surviving conn.)')
    print(f'  b_k max        : {RAM_BK_MAX}')
    print(f'  Min per layer  : {RAM_MIN_PER_LAYER}')
    print(f'  Beta schedule  : warmup={BETA_WARMUP} (beta=0), max={BETA_MAX:.1e}')
    print(f'  Budget ctrl    : margin=±{BUDGET_CTRL_MARGIN:.0%}')
    print(f'  Output         : {output_folder}')
    print('=' * 70)

    # ── 1. 数据 ──────────────────────────────────────────────────────────────
    print('\n[1/5] Loading dataset...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src='openml')
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val   = Dataset(X_val,   y_val,   batch_size, device)

    # ── 2. 构建全新模型（不加载预训练权重）──────────────────────────────────
    print(f'\n[2/5] Building fresh HGQ model (init_bw_k={INIT_BW_K}, init_bw_a={INIT_BW_A})...')
    model = get_model_hgq(INIT_BW_K, INIT_BW_A)
    model.summary()
    print_bk_stats(model, 'after model creation (before Ramanujan init)')

    # ── 3. 3D Ramanujan 初始化 ────────────────────────────────────────────────
    # 联合优化拓扑 + 逐连接位宽：少量高位宽存活连接 + 大量剪除连接
    print(f'\n[3/5] Computing 3D Ramanujan allocation for target_ebops={TARGET_EBOPS}...')
    per_layer_bk_map = compute_3d_ramanujan_allocation(
        model,
        target_ebops=TARGET_EBOPS,
        b_a_init=float(INIT_BW_A),
        b_k_viable=RAM_BK_VIABLE,
        b_k_max=RAM_BK_MAX,
        init_bk=RAM_INIT_BK,
        base_multiplier=RAM_BASE_MULTIPLIER,
        min_per_layer=RAM_MIN_PER_LAYER,
        budget_weight=RAM_BUDGET_WEIGHT,
        seed=42,
        verbose=True,
    )

    # 应用 3D Ramanujan 稀疏初始化
    print('\nApplying 3D Ramanujan initialization...')
    apply_3d_ramanujan_init(
        model,
        per_layer_bk_map,
        active_int_bits=1.0,      # 存活连接的整数位宽
        pruned_int_bits=0.0,      # 被剪连接 kq.i=0
        pruned_frac_bits=0.0,     # 被剪连接 kq.b=0 → 量化输出恒为 0
        also_zero_kernel=True,    # 清零被剪连接的浮点 kernel
        verbose=True,
    )
    print_bk_stats(model, 'after Ramanujan BW init')

    # 初始化后快速评估（未训练，期望接近随机猜测）
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )
    res = model.evaluate(dataset_val, verbose=0)
    print(f'  Post-init  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}')

    # ── 4. 设置 Callbacks ────────────────────────────────────────────────────
    print(f'\n[4/5] Setting up training callbacks...')

    ebops_cb = FreeEBOPs()
    pareto_cb = ParetoFront(
        output_folder,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )

    # Beta schedule: warmup 阶段 beta=0（让模型自由学习），之后 log ramp 到 beta_max
    beta_sched = BetaScheduler(
        PieceWiseSchedule([
            (0,            0, 'constant'),        # warmup: zero EBOPS 压力
            (BETA_WARMUP,  1e-7, 'log'),           # post-warmup: gentle ramp
            (EPOCHS,       BETA_MAX, 'constant'),
        ])
    )

    # LR schedule: cosine decay with restarts
    lr_sched = LearningRateScheduler(
        cosine_decay_restarts_schedule(
            LR, LR_CYCLE, t_mul=1.0, m_mul=LR_MMUL,
            alpha=1e-6, alpha_steps=50,
        )
    )

    # 注意：不使用 RamanujanMaskEnforcer（消融实验表明移除 mask 效果更好）
    # Ramanujan init 提供好的起点，但模型需要自由重新激活连接以学习特征
    # 原 98 条活跃连接不足以在初始阶段学到有用的表示

    # EBOPS 预算控制器：warmup 期间不介入，之后叠加在 beta schedule 之上
    # 排在 BetaScheduler 之后，warmup 结束后从 BetaScheduler 值接管
    budget_ctrl = BetaOnlyBudgetController(
        target_ebops    = TARGET_EBOPS,
        warmup_epochs   = BETA_WARMUP,
        margin          = BUDGET_CTRL_MARGIN,
        beta_min        = 1e-8,
        beta_max        = BETA_MAX,     # 3e-3：允许比 baseline 更强的压缩力
        adjust_factor   = 1.2,          # 稍快的调整速度
        max_step_factor = 2.0,          # 允许初始阶段快速提升 beta
        ema_alpha       = 0.3,
    )

    trace_cb = TrainingTraceToH5(
        output_dir=output_folder,
        filename='training_trace.h5',
        max_bits=8,
        beta_callback=beta_sched,
    )

    # Callback 顺序很重要：
    # 1. ebops_cb     : 计算 EBOPs（供后续 callback 读取）
    # 2. pareto_cb    : 保存 Pareto 前沿检查点
    # 3. beta_sched   : 基础 beta schedule（warmup: beta=0）
    # 4. budget_ctrl  : 叠加 EBOPS 预算约束（warmup 后从 BetaScheduler 值接管）
    # 5. lr_sched     : 学习率调度
    # 6. trace_cb     : 记录训练轨迹
    callbacks = [
        ebops_cb,
        pareto_cb,
        beta_sched,
        budget_ctrl,
        lr_sched,
        trace_cb,
    ]

    # ── 5. 训练 ──────────────────────────────────────────────────────────────
    print(f'\n[5/5] Training for {EPOCHS} epochs...')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        steps_per_execution=32,
    )

    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 完成 ──────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('Training complete.')
    print(f'Pareto checkpoints : {output_folder}')
    print(f'Training trace     : {os.path.join(output_folder, "training_trace.h5")}')

    # 最终评估
    res = model.evaluate(dataset_val, verbose=0)
    print(f'Final  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}')
    print_bk_stats(model, 'final')
    print('=' * 70)
