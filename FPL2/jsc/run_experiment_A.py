"""
run_experiment_A.py — Experiment A: Baseline with Beta Decrease Schedule
=========================================================================

Faithful reproduction of FPL/jsc/train_run_baseline.py with configurable
dataset, epochs, and beta direction.

Config 4 (OpenML, 200k) is the EXACT original baseline (beta increase)
for verification. Configs 1–3 apply the beta-decrease modification.

Experiment configs:
  1. Cernbox dataset, 20000 epochs,  beta DECREASE
  2. OpenML dataset,  20000 epochs,  beta DECREASE
  3. Cernbox dataset, 200000 epochs, beta DECREASE
  4. OpenML dataset,  200000 epochs, beta INCREASE (= original baseline)

Usage:
  python run_experiment_A.py --exp 1
  python run_experiment_A.py --exp 4          # reproduce original baseline
  python run_experiment_A.py --exp 1 2 3 4
  python run_experiment_A.py --exp all
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

from data.data import get_data
from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PieceWiseSchedule
from keras.callbacks import LearningRateScheduler
from model.model import get_model_hgq
from utils import (
    get_tf_device,
    cosine_decay_restarts_schedule,
    TrainingTraceToH5,
    compute_model_ebops,
)

import tensorflow as tf


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Configurations
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_CONFIGS = {
    #                                                beta_direction:
    #                                                  'increase' = original baseline (5e-7 → beta_max)
    #                                                  'decrease' = reversed (beta_max → 5e-7)
    1: dict(src='cernbox', epochs=6000,  label='cernbox_6k',  beta_direction='increase'),
    2: dict(src='openml',  epochs=6000,  label='openml_6k',   beta_direction='increase'),
    3: dict(src='cernbox', epochs=200000, label='cernbox_200k', beta_direction='increase'),
    4: dict(src='openml',  epochs=200000, label='openml_200k',  beta_direction='increase'),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Training Pipeline — 1:1 faithful to FPL/jsc/train_run_baseline.py
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(exp_id: int, output_root: str = 'results/experiment_A', seed: int = 42):
    """Run a single experiment.

    Everything is identical to FPL/jsc/train_run_baseline.py except:
      - dataset / epochs are configurable
      - beta direction can be 'increase' (original) or 'decrease' (reversed)
    """

    cfg = EXPERIMENT_CONFIGS[exp_id]
    src = cfg['src']
    epochs = cfg['epochs']
    label = cfg['label']
    beta_direction = cfg['beta_direction']

    np.random.seed(seed)
    random.seed(seed)

    # ── Hyperparameters — IDENTICAL to baseline ──────────────────────────
    batch_size = 33200
    learning_rate = 5e-3

    beta_sch_0 = 0
    beta_sch_1 = epochs // 50
    beta_sch_2 = epochs
    # Fixed beta_max = 1e-3 for ALL epoch counts.
    # The original baseline formula min(1e-3, 5e-7*(epochs/100)) scales down
    # beta_max for short runs (e.g. 1000ep → 5e-6), which is too weak to
    # compress eBOPs to near 0. Using a fixed 1e-3 ensures sufficient
    # regularization pressure regardless of training length.
    beta_max = 1e-3

    output_folder = os.path.join(output_root, label)
    os.makedirs(output_folder, exist_ok=True)
    device = get_tf_device()

    t0 = time.time()

    print('=' * 72)
    print(f'  Experiment A{exp_id}: {"Original Baseline" if beta_direction == "increase" else "Beta Decrease"}')
    print(f'  Dataset: {src}   Epochs: {epochs}')
    if beta_direction == 'increase':
        print(f'  Beta: 5e-7 → {beta_max:.2e} (increase, original baseline)')
    else:
        print(f'  Beta: {beta_max:.2e} → 5e-7 (decrease, reversed)')
    print(f'  Beta schedule: constant [0,{beta_sch_1}], log-ramp [{beta_sch_1},{beta_sch_2}]')
    print(f'  LR: {learning_rate:.1e}, cycle=4000, t_mul=1.0, m_mul=0.94')
    print(f'  Output: {output_folder}')
    print('=' * 72)

    # ── 1. Load Data ─────────────────────────────────────────────────────
    print('\n[1/4] Loading dataset...')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if src == 'cernbox':
        data_path = os.path.join(data_dir, 'cernbox.h5')
    else:
        data_path = os.path.join(data_dir, 'dataset.h5')

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(data_path, src=src)
    dataset_train = Dataset(X_train, y_train, batch_size, device, shuffle=True)
    dataset_val = Dataset(X_val, y_val, batch_size, device)
    _sample = tf.constant(X_train[:min(2048, len(X_train))], dtype=tf.float32)
    print(f'  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

    # ── 2. Build Model — same as baseline: get_model_hgq(3, 3) ──────────
    print('\n[2/4] Building model...')
    model = get_model_hgq(3, 3)
    model.summary()

    # ── 3. Setup Callbacks — IDENTICAL to baseline ───────────────────────
    print('\n[3/4] Setting up training...')

    # Beta schedule:
    #   baseline (increase): (0, 5e-7, constant) → (sch1, 5e-7, log) → (sch2, beta_max, constant)
    #   reversed (decrease): (0, beta_max, constant) → (sch1, beta_max, log) → (sch2, 5e-7, constant)
    if beta_direction == 'increase':
        # EXACT original baseline
        beta_sched = BetaScheduler(
            PieceWiseSchedule([
                (beta_sch_0, 5e-7,     'constant'),
                (beta_sch_1, 5e-7,     'log'),
                (beta_sch_2, beta_max, 'constant'),
            ])
        )
    else:
        # Reversed: decrease from beta_max to 5e-7
        beta_sched = BetaScheduler(
            PieceWiseSchedule([
                (beta_sch_0, beta_max, 'constant'),
                (beta_sch_1, beta_max, 'log'),
                (beta_sch_2, 5e-7,     'constant'),
            ])
        )

    # LR schedule — IDENTICAL to baseline
    lr_sched = LearningRateScheduler(
        cosine_decay_restarts_schedule(
            learning_rate, 4000, t_mul=1.0, m_mul=0.94, alpha=1e-6, alpha_steps=50
        )
    )

    ebops_cb = FreeEBOPs()
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
        beta_callback=beta_sched,
    )

    # Callback order — IDENTICAL to baseline
    callbacks = [ebops_cb, pareto_cb, beta_sched, lr_sched, trace_cb]

    # Compile — IDENTICAL to baseline
    opt = keras.optimizers.Adam()
    metrics = ['accuracy']
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, steps_per_execution=32)

    # ── 4. Train — IDENTICAL to baseline ─────────────────────────────────
    print('\n[4/4] Training...')
    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Results ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    final_ebops = compute_model_ebops(model, _sample)
    final_res = model.evaluate(dataset_val, verbose=0)

    result = {
        'experiment': f'A{exp_id}',
        'dataset': src,
        'epochs': epochs,
        'beta_direction': beta_direction,
        'beta_max': beta_max,
        'final_ebops': float(final_ebops),
        'final_val_acc': float(final_res[1]),
        'final_val_loss': float(final_res[0]),
        'elapsed_sec': elapsed,
    }

    # Read best accuracy from trace
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
    print(f'  Experiment A{exp_id} ({src}, {epochs} epochs, beta {beta_direction})')
    print(f'  final_ebops={final_ebops:.0f}  val_acc={final_res[1]:.4f}  '
          f'best_acc={result.get("best_val_acc", 0):.4f}  time={elapsed:.0f}s')
    print(f'  {"=" * 60}')

    summary_path = os.path.join(output_folder, 'result_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Summary saved: {summary_path}')
    print(f'  Trace saved: {trace_path}')

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment A: Baseline with configurable beta direction')
    parser.add_argument(
        '--exp', nargs='+', default=['all'],
        help='Experiment IDs to run: 1, 2, 3, 4, or "all" (default: all)')
    parser.add_argument(
        '--output_root', type=str, default='results/experiment_A',
        help='Root output directory (default: results/experiment_A)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)')
    args = parser.parse_args()

    # Parse experiment IDs
    if 'all' in args.exp:
        exp_ids = [1, 2, 3, 4]
    else:
        exp_ids = [int(x) for x in args.exp]
        for eid in exp_ids:
            if eid not in EXPERIMENT_CONFIGS:
                parser.error(f'Invalid experiment ID: {eid}. Must be 1, 2, 3, or 4.')

    print(f'\n{"#" * 72}')
    print(f'  Experiment A')
    print(f'  Experiments to run: {exp_ids}')
    for eid in exp_ids:
        c = EXPERIMENT_CONFIGS[eid]
        print(f'    {eid}: {c["src"]}, {c["epochs"]}ep, beta {c["beta_direction"]}')
    print(f'  Output: {args.output_root}')
    print(f'{"#" * 72}\n')

    all_results = []
    for eid in exp_ids:
        result = run_experiment(eid, output_root=args.output_root, seed=args.seed)
        all_results.append(result)

        # Progress summary
        print(f'\n  ── Progress ──')
        for r in all_results:
            print(f'    A{r["experiment"][-1]} ({r["dataset"]}, {r["epochs"]}ep, '
                  f'beta {r["beta_direction"]}): '
                  f'best_acc={r.get("best_val_acc", 0):.4f}  '
                  f'final_ebops={r.get("final_ebops", 0):.0f}  '
                  f'time={r.get("elapsed_sec", 0):.0f}s')
        print()

    # Save combined summary
    os.makedirs(args.output_root, exist_ok=True)
    combined_path = os.path.join(args.output_root, 'all_results_summary.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nCombined summary: {combined_path}')

    # Final table
    print(f'\n{"=" * 72}')
    print(f'  {"Exp":>4s}  {"Dataset":>8s}  {"Epochs":>8s}  {"Beta":>8s}  '
          f'{"best_acc":>8s}  {"final_ebops":>11s}  {"time":>6s}')
    print(f'  {"-" * 66}')
    for r in all_results:
        print(f'  A{r["experiment"][-1]:>3s}  {r["dataset"]:>8s}  {r["epochs"]:>8d}  '
              f'{r["beta_direction"]:>8s}  '
              f'{r.get("best_val_acc", 0):>8.4f}  '
              f'{r.get("final_ebops", 0):>11.0f}  '
              f'{r.get("elapsed_sec", 0):>5.0f}s')
    print(f'{"=" * 72}')
