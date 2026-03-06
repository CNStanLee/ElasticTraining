#!/usr/bin/env python3
"""
Evaluate gradient/trainability for pruned models at selected EBOPs.

Outputs:
- model_topo/analysis/trainability_metrics.csv
- model_topo/analysis/trainability_summary.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.prune_methods import DEFAULT_TOPOLOGY_METHODS, normalize_methods
from utils.trainability_analysis_utils import evaluate_trainability, load_model_for_method


def parse_targets(s: str) -> list[int]:
    out = []
    for x in s.split(','):
        x = x.strip()
        if x:
            out.append(int(float(x)))
    if not out:
        raise ValueError('No valid targets in --targets')
    return out


def _take(x, y, n: int):
    n = int(min(len(x), max(1, n)))
    return x[:n], y[:n]


def main() -> None:
    here = Path(__file__).resolve().parent
    jsc_root = here.parent
    if str(jsc_root) not in sys.path:
        sys.path.insert(0, str(jsc_root))
    import model.model  # noqa: F401
    from data.data import get_data

    parser = argparse.ArgumentParser(description='Gradient/trainability analysis for pruned models')
    parser.add_argument('--targets', type=str, default='400,1500,2500')
    parser.add_argument('--methods', type=str, default=','.join(DEFAULT_TOPOLOGY_METHODS))
    parser.add_argument(
        '--base_model',
        type=str,
        default='baseline/epoch=2236-val_acc=0.770-ebops=23589-val_loss=0.641.keras',
    )
    parser.add_argument('--baseline_dir', type=str, default='baseline')
    parser.add_argument('--pruned_root', type=str, default='pruned_models')
    parser.add_argument('--data_h5', type=str, default='../data/dataset.h5')
    parser.add_argument('--eval_batch', type=int, default=4096)
    parser.add_argument('--grad_batch', type=int, default=33200)
    parser.add_argument('--one_step_lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='analysis')
    args = parser.parse_args()

    tf.random.set_seed(int(args.seed))
    np.random.seed(int(args.seed))

    targets = parse_targets(args.targets)
    methods = normalize_methods(args.methods, DEFAULT_TOPOLOGY_METHODS)

    base_model = here / args.base_model
    baseline_dir = here / args.baseline_dir
    pruned_root = here / args.pruned_root
    data_h5 = (here / args.data_h5).resolve()
    out_dir = here / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (_, _), (x_val, y_val), _ = get_data(data_h5)
    x_eval, y_eval = _take(x_val, y_val, args.eval_batch)
    x_grad, y_grad = _take(x_val, y_val, args.grad_batch)
    x_eval = tf.constant(x_eval, dtype=tf.float32)
    y_eval = tf.constant(y_eval, dtype=tf.int32)
    x_grad = tf.constant(x_grad, dtype=tf.float32)
    y_grad = tf.constant(y_grad, dtype=tf.int32)

    rows = []
    for t in targets:
        for m in methods:
            model, measured_hint, source_path = load_model_for_method(
                method=m,
                target=t,
                base_model_path=base_model,
                baseline_dir=baseline_dir,
                pruned_root=pruned_root,
            )
            rec = evaluate_trainability(
                model=model,
                target=t,
                method=m,
                measured_ebops_hint=measured_hint,
                source_path=source_path,
                x_eval=x_eval,
                y_eval=y_eval,
                x_grad=x_grad,
                y_grad=y_grad,
                one_step_lr=args.one_step_lr,
            )
            rows.append(rec)
            print(
                f'[EVAL] target={t} method={m} ebops={rec.measured_ebops:.1f} '
                f'grad_norm={rec.grad_global_norm:.3e} drop={rec.one_step_loss_drop:.4e} '
                f'trainable={rec.trainable}'
            )

    csv_path = out_dir / 'trainability_metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'target_ebops', 'method', 'measured_ebops', 'val_loss', 'val_acc',
            'grad_global_norm', 'grad_near_zero_ratio', 'grad_first_last_ratio', 'grad_log_std',
            'grad_batch_loss', 'one_step_loss_drop', 'trainable', 'verdict_reason', 'source_path',
        ])
        for r in rows:
            w.writerow([
                r.target_ebops, r.method, f'{r.measured_ebops:.6f}', f'{r.val_loss:.6f}', f'{r.val_acc:.6f}',
                f'{r.grad_global_norm:.6e}', f'{r.grad_near_zero_ratio:.6f}', f'{r.grad_first_last_ratio:.6e}',
                f'{r.grad_log_std:.6f}', f'{r.grad_batch_loss:.6f}', f'{r.one_step_loss_drop:.6e}',
                int(r.trainable), r.verdict_reason, r.source_path,
            ])

    md_path = out_dir / 'trainability_summary.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Trainability Summary\n\n')
        f.write('| target | method | ebops | grad_norm | near_zero | first/last | one_step_drop | trainable | reason |\n')
        f.write('|---:|---|---:|---:|---:|---:|---:|:---:|---|\n')
        for r in rows:
            f.write(
                f'| {r.target_ebops} | {r.method} | {r.measured_ebops:.0f} | {r.grad_global_norm:.3e} | '
                f'{r.grad_near_zero_ratio:.3f} | {r.grad_first_last_ratio:.3e} | {r.one_step_loss_drop:.3e} | '
                f'{"yes" if r.trainable else "no"} | {r.verdict_reason} |\n'
            )

    print('\nDone.')
    print(f'csv: {csv_path}')
    print(f'md : {md_path}')


if __name__ == '__main__':
    main()
