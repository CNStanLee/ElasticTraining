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


np.random.seed(42)
random.seed(42)

input_folder = 'data/dataset.h5'
output_folder = 'results/baseline/'
batch_size = 33200
learning_rate = 5e-3
epochs = 200000

beta_sch_0 = 0
beta_sch_1 = epochs // 50
beta_sch_2 = epochs
beta_max = min(1e-3, 5e-7 * (epochs / 100))

device = get_tf_device()
os.makedirs(output_folder, exist_ok=True)


if __name__ == '__main__':
    print('Starting training...')

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

    print('set hyperparameters...')
    beta_sched = BetaScheduler(
        PieceWiseSchedule([
            (beta_sch_0, 5e-7, 'constant'),
            (beta_sch_1, 5e-7, 'log'),
            (beta_sch_2, beta_max, 'constant')
        ])
    )
    #lr_scheduler = LearningRateScheduler(cosine_decay_restarts_schedule(learning_rate, epochs))
    lr_sched = LearningRateScheduler(
        cosine_decay_restarts_schedule(learning_rate, 4000, t_mul=1.0, m_mul=0.94, alpha=1e-6, alpha_steps=50)
    )
    trace_cb = TrainingTraceToH5(
        output_dir=output_folder,
        filename="training_trace.h5",
        max_bits=8,
        beta_callback=beta_sched,
    )

    callbacks = [ebops_cb, pareto_cb, beta_sched, lr_sched, trace_cb]

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