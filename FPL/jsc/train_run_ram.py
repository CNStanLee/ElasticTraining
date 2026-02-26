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
from model.model import get_model_hgq
from utils.train_utils import cosine_decay_restarts_schedule
from utils.tf_device import get_tf_device
from utils.train_utils import TrainingTraceToH5

from utils.ramaujian_utils import apply_ramanujan_init, RamanujanMaskEnforcer

np.random.seed(42)
random.seed(42)

input_folder = 'data/dataset.h5'
output_folder = 'results/ram_init/' 
batch_size = 33200
learning_rate = 5e-3

# Ramanujan init degree
# 理论依据：有效度下限 ≈ sqrt(in_dim)（保证谱间隙 / 扩展性）。
# degree 过小（如 4）会导致容量瓶颈，精度上限被锁死。
# 用 per_layer_degree 按各层输入宽度单独设置：
#   t1 (in=16): sqrt(16)=4, 设 8 留余量
#   t2 (in=64): sqrt(64)=8, 设 12 保证扩展性同时保持稀疏
#   t3 (in=64): 同 t2
#   out(in=32): sqrt(32)≈6, 设 8
ram_degree = 8  # 全局默认（兜底，实际被 per_layer_degree 覆盖）
ram_per_layer_degree = {
    't1':  8,   # in=16,  sparsity=50.0%
    't2': 12,   # in=64,  sparsity=81.3%
    't3': 12,   # in=64,  sparsity=81.3%
    'out': 8,   # in=32,  sparsity=75.0%
}

# ── Epochs & beta schedule ────────────────────────────────────────────────────
# Ramanujan init 以高稀疏度（隐藏层 ~94%）为起点，与 baseline（全连接）相比：
#   1. warmup（beta_sch_1）须更长：让稀疏拓扑先学出合理精度，再施加位宽压缩压力。
#      baseline 用 epochs//50（4000），这里用 epochs//10（15000），约 5x 更长。
#   2. 总 epochs 适当缩短（150000 vs 200000）：模型从已压缩状态出发，不需要
#      像 baseline 那样花大量 epoch 慢慢「学会压缩」。
# ─────────────────────────────────────────────────────────────────────────────
epochs = 12000

beta_sch_0 = 0
beta_sch_1 = epochs // 10   # 1200  (baseline: epochs//50 = 4000)
beta_sch_2 = epochs
beta_max = min(1e-3, 5e-7 * (epochs / 100))

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
    model = get_model_hgq(3, 3)
    model.summary()

    # ==========================================================
    # Ramanujan-like sparse initialization
    # ==========================================================
    print('apply Ramanujan-like sparse initialization...')
    apply_ramanujan_init(
        model,
        default_degree=ram_degree,
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

    ram_enforcer = RamanujanMaskEnforcer()
    callbacks = [ebops_cb, pareto_cb, beta_sched, lr_sched, trace_cb, ram_enforcer]

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