"""
train_run_low_budget.py
=======================
极低预算（ebops ≤ 500）专用训练脚本。

解决 train_run_prune_finetune.py 在 target=500 时 acc 卡在随机水平的三类问题：

1. Ramanujan 稀疏角度
   - 拓扑热身（TopologyWarmupCallback）：从高 degree 出发，随训练按梯度剪边，
     保证早期梯度流完整。
   - 死边重连（EdgeRewiringCallback）：周期性将死亡神经元的最弱边换为新边，
     维持 d-regular 谱间隙。

2. 谱约束角度
   - 最小奇异值惩罚（SpectralRegularizationCallback）：就地对权重做梯度步，
     防止活跃子矩阵秩塌缩（σ_min → 0）。

3. STE / β 优化角度
   - β 课程重启（BetaCurriculumController）：acc 停滞 → β 归零恢复 → 小 β 重启，
     打破 β↑ → b_k↓ → STE 失效 → acc 停 的正反馈死循环。
   - 渐进式预算（ProgressiveBudgetController）：target 从 warmup_ebops
     指数衰减到 final_ebops，避免一次性剪枝造成的量化冷启动冲击。
   - 自适应 LR（AdaptiveLRBiwidthScaler）：mean b_k 低时自动升高 LR，
     补偿 STE 低位宽信噪比下降。
   - β 梯度截断（BetaGradClipCallback）：单 epoch β 变化率上限，
     防止 EBOPs 梯度突变导致 b_k 雪崩。
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Always run from the script's own directory so relative paths work correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import random
import argparse

import keras
import numpy as np
import tensorflow as tf

from data.data import get_data
from hgq.layers import QLayerBase
from hgq.utils.sugar import Dataset, FreeEBOPs, ParetoFront
from keras.callbacks import LearningRateScheduler

from utils.train_utils import (
    cosine_decay_restarts_schedule,
    TrainingTraceToH5,
    BudgetAwareEarlyStopping,
)
from utils.tf_device import get_tf_device
from utils.ramanujan_budget_utils import (
    BetaOnlyBudgetController,
    SensitivityAwarePruner,
    HighBitPruner,
    compute_bw_aware_degree,
    _flatten_layers,
    _get_kq_var,
)
from utils.low_budget_utils import (
    TopologyWarmupCallback,
    EdgeRewiringCallback,
    SpectralRegularizationCallback,
    BetaCurriculumController,
    AdaptiveLRBiwidthScaler,
    ProgressiveBudgetController,
    BetaGradClipCallback,
    _set_all_beta,
    _get_active_bk_mean,
)

np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

BASELINE_CKPT = "results/baseline/epoch=2236-val_acc=0.770-ebops=23589-val_loss=0.641.keras"
BASELINE_EBOPS = 23589

TARGET_EBOPS = 500  # 最终目标（可通过命令行覆盖）

# ── 输入 ─────────────────────────────────────────────────────────────────────
input_folder = "data/dataset.h5"
batch_size = 33200

# ── 渐进式预算 ────────────────────────────────────────────────────────────────
# 不用一次性剪枝冲击，而是从 warmup_ebops 指数衰减到 TARGET_EBOPS
WARMUP_EBOPS = 2000  # 起始目标（约 baseline 的 1/10，留足学习空间）
BUDGET_DECAY_EP = 3000  # 多少 epoch 内从 warmup 降到 final

# ── Phase 1：恢复 + 渐进压缩 ──────────────────────────────────────────────────
PHASE1_EPOCHS = 6000
PHASE1_LR = 2e-3
PHASE1_LR_CYCLE = 2000
PHASE1_LR_MMUL = 0.9
PHASE1_BETA_INIT = 1e-5
PHASE1_BETA_MIN = 1e-8
PHASE1_BETA_MAX = 5e-4

# ── Phase 2：精度最大化 ────────────────────────────────────────────────────────
PHASE2_EPOCHS = 12000
PHASE2_LR = 5e-4
PHASE2_LR_CYCLE = 800
PHASE2_LR_MMUL = 0.95
PHASE2_BETA_INIT = 1e-5
PHASE2_BETA_MIN = 1e-8
PHASE2_BETA_MAX = 5e-4

# ── 拓扑热身（Ramanujan）────────────────────────────────────────────────────
TOPO_WARMUP_MUL = 3.0  # 初始 degree = d_final × 3（高度保证早期梯度流）
TOPO_STEP_EPOCH = 300  # 每 300 epoch 降一度
TOPO_MIN_DEGREE = 2  # 绝对下限

# ── 谱正则化 ────────────────────────────────────────────────────────────────
SIGMA_MIN_TARGET = 0.02  # 最小奇异值阈值
LR_SPEC = 5e-5  # 谱惩罚梯度步长
SPEC_CHECK_EVERY = 200  # 每 200 epoch 检查一次

# ── 边重连 ──────────────────────────────────────────────────────────────────
REWIRE_INTERVAL = 600
DEAD_THRESHOLD = 0.01

# ── β 课程 ──────────────────────────────────────────────────────────────────
STALL_PATIENCE = 600  # acc 停滞 epoch 数阈值（500 较小模型用 600）
RECOVER_EPOCHS = 300
RESTART_DECAY = 0.25
MAX_RESTARTS = 8

# ── 自适应 LR ────────────────────────────────────────────────────────────────
BK_THRESHOLD = 2.0
LR_SCALE_POWER = 0.5
LR_MAX_FACTOR = 4.0

# ── β 截断 ──────────────────────────────────────────────────────────────────
BETA_MAX_RATIO = 1.5

# ── Early stopping ──────────────────────────────────────────────────────────
EARLYSTOP_PATIENCE = 5000
EARLYSTOP_BUDGET = TARGET_EBOPS * 1.5

# ── 命令行覆盖 ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Low-budget HGQ training (ebops<=500)")
parser.add_argument("--target_ebops", type=float, default=TARGET_EBOPS)
parser.add_argument("--warmup_ebops", type=float, default=WARMUP_EBOPS)
parser.add_argument("--phase1_epochs", type=int, default=PHASE1_EPOCHS)
parser.add_argument("--phase2_epochs", type=int, default=PHASE2_EPOCHS)
parser.add_argument("--checkpoint", type=str, default=BASELINE_CKPT)
parser.add_argument("--no_topo_warmup", action="store_true", help="Disable topology warmup")
parser.add_argument("--no_spectral_reg", action="store_true", help="Disable spectral regularization")
parser.add_argument("--no_rewiring", action="store_true", help="Disable edge rewiring")
parser.add_argument("--no_adaptive_lr", action="store_true", help="Disable adaptive LR scaling")
parser.add_argument(
    "--no_progressive",
    action="store_true",
    help="Disable progressive budget (use one-shot prune)",
)
args, _ = parser.parse_known_args()

TARGET_EBOPS = args.target_ebops
WARMUP_EBOPS = args.warmup_ebops
PHASE1_EPOCHS = args.phase1_epochs
PHASE2_EPOCHS = args.phase2_epochs
BASELINE_CKPT = args.checkpoint
EARLYSTOP_BUDGET = TARGET_EBOPS * 1.5

USE_TOPO_WARMUP = not args.no_topo_warmup
USE_SPECTRAL_REG = not args.no_spectral_reg
USE_REWIRING = not args.no_rewiring
USE_ADAPTIVE_LR = not args.no_adaptive_lr
USE_PROGRESSIVE = not args.no_progressive

output_folder = f"results/low_budget_{int(TARGET_EBOPS)}/"
device = get_tf_device()
os.makedirs(output_folder, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def compute_model_ebops(model, sample_input) -> float:
    from keras import ops
    model(sample_input, training=True)
    total = 0
    for layer in model._flatten_layers():
        if isinstance(layer, QLayerBase) and layer.enable_ebops and layer._ebops is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def make_lr_scheduler(lr_init, cycle, mmul, offset=0):
    fn = cosine_decay_restarts_schedule(
        lr_init, cycle, t_mul=1.0, m_mul=mmul, alpha=1e-6, alpha_steps=50
    )

    def schedule(epoch):
        return fn(max(0, epoch - offset))

    return LearningRateScheduler(schedule)


def print_bk_stats(model, label=""):
    all_b = []
    for layer in _flatten_layers(model):
        kq = getattr(layer, "kq", None)
        if kq is None:
            continue
        b_var = _get_kq_var(kq, "b")
        if b_var is None:
            b_var = _get_kq_var(kq, "f")
        if b_var is not None:
            all_b.extend(b_var.numpy().ravel().tolist())
    if all_b:
        arr = np.array(all_b)
        print(
            f"  [bk_stats {label}]  "
            f"mean={arr.mean():.3f}  std={arr.std():.3f}  "
            f"min={arr.min():.3f}  max={arr.max():.3f}  "
            f"p5={np.percentile(arr,5):.3f}  p95={np.percentile(arr,95):.3f}  "
            f"n_dead(<=0.1)={int((arr<=0.1).sum())}/{len(arr)}"
        )


def bisect_ebops_to_target(
    model,
    target_ebops,
    sample_input,
    tolerance=0.05,
    max_iter=20,
    b_k_min=0.01,
    b_k_max=8.0,
):
    """用二分搜索将全模型 kq.b 缩放至目标 EBOPs。"""
    from keras import ops as kops

    snapshots = {}
    for layer in _flatten_layers(model):
        kq = getattr(layer, "kq", None)
        if kq is None:
            continue
        b_var = _get_kq_var(kq, "b")
        if b_var is None:
            b_var = _get_kq_var(kq, "f")
        if b_var is None:
            continue
        snapshots[id(layer)] = (b_var, b_var.numpy().copy())

    def apply_scale(s):
        for b_var, b_snap in snapshots.values():
            b_new = np.where(
                b_snap > 0.1,
                np.clip(b_snap * s, b_k_min, b_k_max),
                0.0,
            )
            b_var.assign(b_new.astype(np.float32))

    def measure(s):
        apply_scale(s)
        model(sample_input, training=True)
        return float(
            sum(
                int(kops.convert_to_numpy(l._ebops))
                for l in model._flatten_layers()
                if isinstance(l, QLayerBase) and l.enable_ebops and l._ebops is not None
            )
        )

    e_1 = measure(1.0)
    if abs(e_1 - target_ebops) / max(target_ebops, 1) <= tolerance:
        return e_1

    if e_1 < target_ebops:
        lo, hi = 1.0, 2.0
        while measure(hi) < target_ebops and hi < 1e4:
            hi *= 2.0
        lo_e, hi_e = measure(lo), measure(hi)
    else:
        lo, hi = 0.5, 1.0
        while measure(lo) > target_ebops and lo > 1e-6:
            lo /= 2.0
        lo_e, hi_e = measure(lo), measure(hi)

    best_s, best_e = lo, lo_e
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        mid_e = measure(mid)
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
    final_e = measure(best_s)
    print(
        f"  [BisectEBOPs] scale={best_s:.5f}  final_ebops={final_e:.1f}  "
        f"target={target_ebops:.1f}  err={abs(final_e-target_ebops)/target_ebops*100:.1f}%"
    )
    return final_e


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

    print("=" * 70)
    print("  Low-Budget HGQ Training  (target ebops = %g)" % TARGET_EBOPS)
    print(f"  Checkpoint    : {BASELINE_CKPT}")
    print(
        f"  Warmup EBOPs  : {WARMUP_EBOPS}  →  {TARGET_EBOPS}  "
        f"over {BUDGET_DECAY_EP} epochs"
    )
    print(
        f"  Topology warmup : {USE_TOPO_WARMUP}  |  "
        f"Spectral reg: {USE_SPECTRAL_REG}  |  "
        f"Edge rewiring: {USE_REWIRING}"
    )
    print(
        f"  Adaptive LR   : {USE_ADAPTIVE_LR}  |  "
        f"Progressive budget: {USE_PROGRESSIVE}"
    )
    print(f"  Phase1: {PHASE1_EPOCHS} ep  |  Phase2: {PHASE2_EPOCHS} ep")
    print(f"  Output: {output_folder}")
    print("=" * 70)

    # ── 1. 数据 ───────────────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src="openml")
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val = Dataset(X_val, y_val, batch_size, device)
    _sample = tf.constant(X_val[:512], dtype=tf.float32)

    # ── 2. 加载 baseline ──────────────────────────────────────────────────────
    print(f"\n[2/6] Loading checkpoint: {BASELINE_CKPT}")
    model = keras.models.load_model(BASELINE_CKPT)
    model.summary()
    print_bk_stats(model, "loaded")

    actual_baseline_ebops = compute_model_ebops(model, _sample)
    print(f"  Baseline EBOPs (measured): {actual_baseline_ebops:.1f}")

    # ── 3. 初始预算校准 ─────────────────────────────────────────────────────
    print(
        f"\n[3/6] Initial budget calibration  "
        f'({"progressive" if USE_PROGRESSIVE else "one-shot"})...'
    )

    init_target = WARMUP_EBOPS if USE_PROGRESSIVE else TARGET_EBOPS
    pruner = SensitivityAwarePruner(target_ebops=init_target, pruned_threshold=0.1, b_k_min=0.3)
    pruner.prune_to_ebops(model, current_ebops=actual_baseline_ebops, verbose=True)
    post_prune_e = compute_model_ebops(model, _sample)
    print(f"  Post-prune EBOPs: {post_prune_e:.1f}  target: {init_target:.1f}")

    post_prune_e = bisect_ebops_to_target(model, init_target, _sample, tolerance=0.04, max_iter=20)
    print_bk_stats(model, "after initial calibration")

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        steps_per_execution=32,
    )
    res = model.evaluate(dataset_val, verbose=0)
    print(f"  Post-calibration  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}")

    # ── 4. 计算 Ramanujan 拓扑参数（用于 warmup 和 rewiring）──────────────────
    print("\n[4/6] Computing Ramanujan topology...")
    per_layer_degree, per_layer_bk = compute_bw_aware_degree(
        model,
        target_ebops=TARGET_EBOPS,
        b_a_init=3.0,
        b_k_min=0.5,
        b_k_max=8.0,
        multiplier=1.5,
        min_degree=TOPO_MIN_DEGREE,
        budget_weight="capacity",
        verbose=True,
    )

    # ── 5. 共享 Callbacks ─────────────────────────────────────────────────────
    ebops_cb = FreeEBOPs()
    pareto_cb = ParetoFront(
        output_folder,
        ["val_accuracy", "ebops"],
        [1, -1],
        fname_format="epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras",
    )
    trace_cb = TrainingTraceToH5(
        output_dir=output_folder,
        filename="training_trace.h5",
        max_bits=8,
        beta_callback=None,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        steps_per_execution=32,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1：恢复 + 渐进压缩
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[5/6] PHASE 1  Recovery + Progressive Compression  ({PHASE1_EPOCHS} ep)")

    _set_all_beta(model, PHASE1_BETA_INIT)

    p1_budget_ctrl = BetaOnlyBudgetController(
        target_ebops=WARMUP_EBOPS if USE_PROGRESSIVE else TARGET_EBOPS,
        margin=0.15,
        beta_init=PHASE1_BETA_INIT,
        beta_min=PHASE1_BETA_MIN,
        beta_max=PHASE1_BETA_MAX,
        adjust_factor=1.3,
        ema_alpha=0.3,
    )

    p1_callbacks = [ebops_cb, pareto_cb, p1_budget_ctrl]

    if USE_PROGRESSIVE:
        prog_budget = ProgressiveBudgetController(
            budget_ctrl=p1_budget_ctrl,
            warmup_ebops=WARMUP_EBOPS,
            final_ebops=TARGET_EBOPS,
            decay_epochs=BUDGET_DECAY_EP,
            start_epoch=0,
        )
        p1_callbacks.append(prog_budget)

    if USE_TOPO_WARMUP:
        topo_warmup = TopologyWarmupCallback(
            per_layer_degree_final=per_layer_degree,
            degree_warmup_mul=TOPO_WARMUP_MUL,
            step_interval=TOPO_STEP_EPOCH,
            min_degree=TOPO_MIN_DEGREE,
            seed=42,
        )
        p1_callbacks.append(topo_warmup)

    if USE_SPECTRAL_REG:
        spec_reg = SpectralRegularizationCallback(
            sigma_min_target=SIGMA_MIN_TARGET,
            lr_spec=LR_SPEC,
            check_interval=SPEC_CHECK_EVERY,
        )
        p1_callbacks.append(spec_reg)

    p1_curriculum = BetaCurriculumController(
        budget_ctrl=p1_budget_ctrl,
        stall_patience=STALL_PATIENCE,
        recover_epochs=RECOVER_EPOCHS,
        min_delta=5e-5,
        restart_decay=RESTART_DECAY,
        max_restarts=MAX_RESTARTS,
    )
    p1_callbacks.append(p1_curriculum)

    if USE_ADAPTIVE_LR:
        p1_lr_scaler = AdaptiveLRBiwidthScaler(
            bk_threshold=BK_THRESHOLD,
            scale_power=LR_SCALE_POWER,
            lr_max_factor=LR_MAX_FACTOR,
            log=False,
        )
        p1_callbacks.append(p1_lr_scaler)

    p1_clip = BetaGradClipCallback(budget_ctrl=p1_budget_ctrl, max_ratio=BETA_MAX_RATIO)
    p1_callbacks.append(p1_clip)

    p1_callbacks.append(make_lr_scheduler(PHASE1_LR, PHASE1_LR_CYCLE, PHASE1_LR_MMUL, offset=0))
    p1_callbacks.append(trace_cb)

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=PHASE1_EPOCHS,
        callbacks=p1_callbacks,
        verbose=1,
    )

    print_bk_stats(model, "end of phase1")
    res = model.evaluate(dataset_val, verbose=0)
    print(
        f"  Phase1 end  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}  "
        f"mean_bk={_get_active_bk_mean(model):.3f}"
    )

    phase1_final_beta = p1_budget_ctrl.beta_current
    print(f"  Phase1 equilibrium β: {phase1_final_beta:.2e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2：精度最大化
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[6/6] PHASE 2  Accuracy Maximization  ({PHASE2_EPOCHS} ep)")

    _set_all_beta(model, phase1_final_beta)

    p2_budget_ctrl = BetaOnlyBudgetController(
        target_ebops=TARGET_EBOPS,
        margin=0.05,
        beta_init=phase1_final_beta,
        beta_min=PHASE2_BETA_MIN,
        beta_max=PHASE2_BETA_MAX,
        adjust_factor=1.15,
        ema_alpha=0.15,
    )

    p2_callbacks = [ebops_cb, pareto_cb, p2_budget_ctrl]

    if USE_REWIRING:
        current_degree = dict(per_layer_degree)  # 保守：使用 final degree
        if USE_TOPO_WARMUP:
            current_degree.update(topo_warmup._current_degree)
        edge_rewire = EdgeRewiringCallback(
            per_layer_degree=current_degree,
            rewire_interval=REWIRE_INTERVAL,
            dead_threshold=DEAD_THRESHOLD,
            seed=1234,
        )
        p2_callbacks.append(edge_rewire)

    if USE_SPECTRAL_REG:
        p2_callbacks.append(
            SpectralRegularizationCallback(
                sigma_min_target=SIGMA_MIN_TARGET,
                lr_spec=LR_SPEC * 0.5,  # Phase 2 谱步长减半，更温和
                check_interval=SPEC_CHECK_EVERY,
            )
        )

    p2_curriculum = BetaCurriculumController(
        budget_ctrl=p2_budget_ctrl,
        stall_patience=STALL_PATIENCE * 2,
        recover_epochs=RECOVER_EPOCHS,
        min_delta=2e-5,
        restart_decay=RESTART_DECAY,
        max_restarts=MAX_RESTARTS,
    )
    p2_callbacks.append(p2_curriculum)

    if USE_ADAPTIVE_LR:
        p2_callbacks.append(
            AdaptiveLRBiwidthScaler(
                bk_threshold=BK_THRESHOLD,
                scale_power=LR_SCALE_POWER * 0.5,
                lr_max_factor=2.0,
                log=False,
            )
        )

    p2_callbacks.append(BetaGradClipCallback(budget_ctrl=p2_budget_ctrl, max_ratio=BETA_MAX_RATIO))

    early_stop = BudgetAwareEarlyStopping(
        ebops_budget=EARLYSTOP_BUDGET,
        patience=EARLYSTOP_PATIENCE,
        min_delta=5e-5,
        min_epoch=PHASE1_EPOCHS + 1000,
        restore_best_weights=True,
    )
    p2_callbacks.append(early_stop)

    p2_callbacks.append(make_lr_scheduler(PHASE2_LR, PHASE2_LR_CYCLE, PHASE2_LR_MMUL, offset=PHASE1_EPOCHS))
    p2_callbacks.append(trace_cb)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        steps_per_execution=32,
    )

    model.fit(
        dataset_train,
        validation_data=dataset_val,
        initial_epoch=PHASE1_EPOCHS,
        epochs=TOTAL_EPOCHS,
        callbacks=p2_callbacks,
        verbose=1,
    )

    print("\n" + "=" * 70)
    print("Training complete.")
    res = model.evaluate(dataset_val, verbose=0)
    print(
        f"Final  val_loss={res[0]:.4f}  val_accuracy={res[1]:.4f}  "
        f"mean_bk={_get_active_bk_mean(model):.3f}"
    )
    print(f"Pareto checkpoints : {output_folder}")
    print(f"Training trace     : {os.path.join(output_folder, 'training_trace.h5')}")
    print("=" * 70)