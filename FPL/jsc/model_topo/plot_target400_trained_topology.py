#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.topology_analysis_utils import (
    load_layer_maps,
    pick_gradual_ckpt_for_target,
    pick_meta_for_target,
)


def plot_rows(
    target: int,
    rows: list[tuple[str, list[str], list[np.ndarray], float]],
    out_path: Path,
) -> None:
    if not rows:
        raise RuntimeError('No rows to plot')
    layer_names = rows[0][1]
    nrows = len(rows)
    ncols = len(layer_names)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows), squeeze=False)
    min_nonzero_darkness = 0.22
    dead_zone_rgb = np.array([0.82, 0.33, 0.33], dtype=np.float32)

    all_pos = []
    for _, _, maps, _ in rows:
        for mp in maps:
            pos = mp[mp > 0]
            if pos.size:
                all_pos.append(pos)
    global_vmax = float(np.percentile(np.concatenate(all_pos), 92)) if all_pos else 1.0
    global_vmax = max(global_vmax, 1e-8)

    for r, (label, _, maps, measured) in enumerate(rows):
        for c, lname in enumerate(layer_names):
            ax = axes[r, c]
            mp = maps[c]
            disp = np.clip(np.maximum(mp, 0.0) / global_vmax, 0.0, 1.0)
            disp = np.power(disp, 0.55, dtype=np.float32)
            nz = mp > 0
            disp = np.where(nz, min_nonzero_darkness + (1.0 - min_nonzero_darkness) * disp, 0.0).astype(np.float32)

            rgb = np.ones((disp.shape[0], disp.shape[1], 3), dtype=np.float32)
            rgb[nz] = 1.0 - disp[nz, None]
            rgb[mp < 0] = dead_zone_rgb
            ax.imshow(rgb, aspect='auto', vmin=0.0, vmax=1.0)
            ax.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.2)
            if r == 0:
                ax.set_title(lname)
            if c == 0:
                ax.set_ylabel(f'{label}\nEBOPs={measured:.0f}')
            ax.set_xticks([])
            ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap='Greys', norm=plt.Normalize(vmin=0.0, vmax=global_vmax))
    sm.set_array([])
    cax = fig.add_axes([0.945, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Global intensity (bitwidth * |weight|)')
    fig.text(0.92, 0.98, 'Red: active & |w|>0 but Q(w)=0', ha='right', va='top', fontsize=9, color='black')

    fig.suptitle(f'Target {target} Topology After Training')
    fig.subplots_adjust(left=0.05, right=0.92, top=0.90, bottom=0.06, wspace=0.12, hspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    jsc_root = here.parent
    if str(jsc_root) not in sys.path:
        sys.path.insert(0, str(jsc_root))
    import model.model  # noqa: F401

    parser = argparse.ArgumentParser(description='Plot topology after training for a target budget')
    parser.add_argument('--target', type=int, default=400)
    parser.add_argument(
        '--base_model',
        type=str,
        default='baseline/epoch=2236-val_acc=0.770-ebops=23589-val_loss=0.641.keras',
    )
    parser.add_argument('--baseline_dir', type=str, default='baseline')
    parser.add_argument('--pruned_root', type=str, default='pruned_models')
    parser.add_argument('--method', type=str, default='spectral_quant')
    parser.add_argument('--trained_model', type=str, default='opt_train500/target400/best.keras')
    parser.add_argument('--plots_dir', type=str, default='plots')
    args = parser.parse_args()

    base_model = here / args.base_model
    baseline_dir = here / args.baseline_dir
    pruned_root = here / args.pruned_root
    trained_model = here / args.trained_model
    plots_dir = here / args.plots_dir
    target = int(args.target)

    gradual = pick_gradual_ckpt_for_target(baseline_dir=baseline_dir, target=target)
    ln_g, mp_g = load_layer_maps(base_model_path=base_model, weights_path=gradual.weights_path, direct_model=True)

    init_info = pick_meta_for_target(pruned_root=pruned_root, method=args.method, target=target)
    ln_i, mp_i = load_layer_maps(base_model_path=base_model, weights_path=init_info.weights_path, direct_model=False)

    ln_t, mp_t = load_layer_maps(base_model_path=base_model, weights_path=trained_model, direct_model=True)

    if ln_g != ln_i or ln_g != ln_t:
        raise RuntimeError('Layer mismatch among gradual/init/trained models')

    rows = [
        ('Gradual baseline', ln_g, mp_g, gradual.measured_ebops),
        (f'{args.method} one-shot init', ln_i, mp_i, init_info.measured_ebops),
        (f'Trained (target{target})', ln_t, mp_t, float(target)),
    ]
    out_path = plots_dir / f'topology_target{target}_trained_compare.png'
    plot_rows(target=target, rows=rows, out_path=out_path)
    print(f'[PLOT] {out_path}')


if __name__ == '__main__':
    main()
