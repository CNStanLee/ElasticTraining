import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # jax, torch, tensorflow
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random

import keras
import numpy as np

from data.data import get_data
from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PieceWiseSchedule
from keras.callbacks import LearningRateScheduler
from model.model import get_model_hgq, get_model_hgq_deep
from utils.train_utils import cosine_decay_restarts_schedule, EBOPsAdaptiveBeta
from utils.tf_device import get_tf_device
from utils.train_utils import TrainingTraceToH5

from utils.ramaujian_utils import apply_ramanujan_init, RamanujanMaskEnforcer, compute_per_layer_degree

np.random.seed(42)
random.seed(42)

input_folder = 'data/dataset.h5'
output_folder = 'results/ram_init/' 
batch_size = 33200
learning_rate = 5e-3

# Ramanujan init degree — 由 compute_per_layer_degree() 在模型构建后自动计算，
# 公式：degree = clamp(round(sqrt(in_dim) * multiplier), min_degree, in_dim)
# multiplier=1.5 保证谱间隙余量，min_degree=4 防止极端稀疏。
ram_degree_multiplier = 1.5
ram_degree_min = 4

# ── Epochs & beta schedule ────────────────────────────────────────────────────
# 在 epoch 固定 12000 的约束下，核心改进是 warmup 结束后渐进放开 mask 约束：
#   [0,    1200] warmup：mask 完全固定，Ramanujan 拓扑稳定初期收敛方向
#   [1200, 4200] fade：mask 线性放开（3000 epoch 渐进窗口），HGQ 梯度逐步修复次优连接
#   [4200,12000] 完全放开：等价于从好初始点出发的 baseline，HGQ 自由决定稀疏结构
# beta_hard_max 设为 2e-2：在有限 epoch 内仍需足够压缩力，但不再像 1e-1 那样过激
# ─────────────────────────────────────────────────────────────────────────────
epochs = 12000

beta_sch_0 = 0
beta_sch_1 = epochs // 10   # 1200 warmup
beta_sch_2 = epochs
beta_max = 5e-4

# Mask 放开时间窗口：warmup 结束后开始渐进释放，3000 epoch 内完全放开
mask_release_epoch = beta_sch_1          # 1200
mask_fade_epochs   = 3000                # 完全放开于 epoch 4200

device = get_tf_device()
os.makedirs(output_folder, exist_ok=True)


if __name__ == '__main__':
    print('Starting training (Ramanujan init)...')

    print('get dataset...')
    src = 'openml'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(input_folder, src=src)
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val = Dataset(X_val, y_val, batch_size, device)

    print('init EBOPs and pareto...')
    ebops_cb = FreeEBOPs()
    pareto_cb = ParetoFront(
        output_folder,
        ['val_accuracy', 'ebops'],
        [1, -1],
        fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
    )

    print("get model...")
    # model = get_model_hgq_deep(3, 3)
    model = get_model_hgq(3, 3)
    model.summary()

    # 根据各层输入维度自动计算 Ramanujan 度
    ram_per_layer_degree = compute_per_layer_degree(
        model,
        multiplier=ram_degree_multiplier,
        min_degree=ram_degree_min,
    )

    # ==========================================================
    # Ramanujan-like sparse initialization
    # ==========================================================
    print('apply Ramanujan-like sparse initialization...')
    apply_ramanujan_init(
        model,
        default_degree=ram_degree_min,   # fallback，HGQ 层已由 per_layer_degree 全覆盖
        per_layer_degree=ram_per_layer_degree,
        seed=42,
        pruned_frac_bits=0.0,
        pruned_int_bits=0.0,
        also_zero_kernel=True,
    )
    print('Ramanujan initialization done.')
    # ==========================================================

    print('set hyperparameters...')
    beta_sched = BetaScheduler(
        PieceWiseSchedule([
            (beta_sch_0, 5e-7, 'constant'),
            (beta_sch_1, 5e-7, 'log'),
            (beta_sch_2, beta_max, 'constant')
        ])
    )
    # lr_scheduler = LearningRateScheduler(cosine_decay_restarts_schedule(learning_rate, epochs))
    lr_sched = LearningRateScheduler(
        cosine_decay_restarts_schedule(learning_rate, 4000, t_mul=1.0, m_mul=0.94, alpha=1e-6, alpha_steps=50)
    )
    trace_cb = TrainingTraceToH5(
        output_dir=output_folder,
        filename="training_trace.h5",
        max_bits=8,
        beta_callback=beta_sched,
    )

    # release_epoch=mask_release_epoch: warmup 后逐步放开拓扑约束
    # fade_epochs=mask_fade_epochs: 渐进窗口，避免突然切换
    ram_enforcer = RamanujanMaskEnforcer(
        release_epoch=mask_release_epoch,
        fade_epochs=mask_fade_epochs,
    )
    # EBOPsAdaptiveBeta 须排在 BetaScheduler 之后，覆盖写入 layer._beta
    # beta_hard_max=2e-2：12k epoch 有限时间内需要足够压缩力，但不如 1e-1 那样过激
    adaptive_beta = EBOPsAdaptiveBeta(
        beta_scheduler=beta_sched,
        warmup_end_epoch=beta_sch_1,
        total_epochs=epochs,
        beta_hard_max=2e-2,   # 12k epoch 适中压缩力
        boost_power=0.6,
        target_ebops=0.0,
    )
    callbacks = [ebops_cb, pareto_cb, beta_sched, lr_sched, trace_cb, ram_enforcer, adaptive_beta]

    opt = keras.optimizers.Adam()
    metrics = ['accuracy']
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, steps_per_execution=32)

    print('start training...')
    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print('training completed.')
    print(f"Trace saved to: {os.path.join(output_folder, 'training_trace.h5')}")