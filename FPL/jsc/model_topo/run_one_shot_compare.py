#!/usr/bin/env python3
"""
Run one-shot pruning comparison under model_topo:
- spectral_quant (opt)
- sensitivity (baseline)
- uniform (baseline)

Writes summary CSV into model_topo/pruned_models/compare_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.one_shot_compare_utils import (
    default_python_from_env,
    parse_targets,
    run_compare,
    write_summary_csv,
)
from utils.prune_methods import DEFAULT_COMPARE_METHODS, normalize_methods


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]

    parser = argparse.ArgumentParser(description='One-shot prune comparison (opt vs baselines)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='model_topo/baseline/epoch=2236-val_acc=0.770-ebops=23589-val_loss=0.641.keras',
    )
    parser.add_argument('--targets', type=str, default='400,1500,2500,6800,11700')
    parser.add_argument('--methods', type=str, default=','.join(DEFAULT_COMPARE_METHODS))
    parser.add_argument('--input_h5', type=str, default='data/dataset.h5')
    parser.add_argument('--sample_size', type=int, default=512)
    parser.add_argument('--high_budget_ratio', type=float, default=0.45)
    parser.add_argument('--output_root', type=str, default='model_topo/pruned_models')
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    methods = normalize_methods(args.methods, DEFAULT_COMPARE_METHODS)
    if not methods:
        raise ValueError('No valid methods parsed from --methods')

    out_root = repo_root / 'FPL' / 'jsc' / args.output_root

    rows = run_compare(
        repo_root=repo_root,
        checkpoint=args.checkpoint,
        targets=targets,
        methods=methods,
        input_h5=args.input_h5,
        sample_size=args.sample_size,
        out_root=out_root,
        python_exec=default_python_from_env(),
        high_budget_ratio=args.high_budget_ratio,
    )

    summary_csv = out_root / 'compare_summary.csv'
    write_summary_csv(rows, summary_csv)

    print('\n=== Done ===')
    print(f'checkpoint: {args.checkpoint}')
    print(f'targets   : {targets}')
    print(f'methods   : {methods}')
    print(f'summary   : {summary_csv}')


if __name__ == '__main__':
    main()
