#!/usr/bin/env python3
"""
Analyze topology of existing pruned models and generate white-background,
black-frame comparison plots with darkness driven by bitwidth and weight value.

Default targets: 400,1500,2500
Output: model_topo/plots/topology_compare_target{target}.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.topology_analysis_utils import (
    load_layer_maps,
    pick_gradual_ckpt_for_target,
    pick_meta_for_target,
    plot_topology_compare,
)
from utils.prune_methods import DEFAULT_TOPOLOGY_METHODS, normalize_methods


def parse_targets(s: str) -> list[int]:
    out = []
    for x in s.split(','):
        x = x.strip()
        if x:
            out.append(int(float(x)))
    if not out:
        raise ValueError('No valid targets in --targets')
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here
    jsc_root = here.parent
    if str(jsc_root) not in sys.path:
        sys.path.insert(0, str(jsc_root))
    import model.model  # noqa: F401  # register custom layers before load_model

    parser = argparse.ArgumentParser(description='Topology analysis for pruned models')
    parser.add_argument('--targets', type=str, default='400,1500,2500,6800,11700')
    parser.add_argument('--methods', type=str, default=','.join(DEFAULT_TOPOLOGY_METHODS))
    parser.add_argument(
        '--base_model',
        type=str,
        default='baseline/epoch=2236-val_acc=0.770-ebops=23589-val_loss=0.641.keras',
    )
    parser.add_argument('--baseline_dir', type=str, default='baseline')
    parser.add_argument('--pruned_root', type=str, default='pruned_models')
    parser.add_argument('--plots_dir', type=str, default='plots')
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    methods = normalize_methods(args.methods, DEFAULT_TOPOLOGY_METHODS)

    base_model = root / args.base_model
    baseline_dir = root / args.baseline_dir
    pruned_root = root / args.pruned_root
    plots_dir = root / args.plots_dir

    if not base_model.exists():
        raise FileNotFoundError(f'Base model not found: {base_model}')

    for t in targets:
        method_maps = {}
        for m in methods:
            if m == 'gradual':
                info = pick_gradual_ckpt_for_target(baseline_dir=baseline_dir, target=t)
                layer_names, maps = load_layer_maps(
                    base_model_path=base_model,
                    weights_path=info.weights_path,
                    direct_model=True,
                )
            else:
                info = pick_meta_for_target(pruned_root=pruned_root, method=m, target=t)
                layer_names, maps = load_layer_maps(
                    base_model_path=base_model,
                    weights_path=info.weights_path,
                    direct_model=False,
                )
            method_maps[m] = (layer_names, maps, info.measured_ebops)
            print(f'[LOAD] target={t} method={m} measured_ebops={info.measured_ebops:.1f}')

        out_path = plots_dir / f'topology_compare_target{t}.png'
        plot_topology_compare(target=t, method_maps=method_maps, out_path=out_path)
        print(f'[PLOT] {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
