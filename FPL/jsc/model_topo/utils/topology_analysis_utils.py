from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np

from .prune_methods import METHOD_STYLE


@dataclass
class MethodModel:
    method: str
    target_ebops: int
    measured_ebops: float
    meta_path: Path
    weights_path: Path


def pick_meta_for_target(pruned_root: Path, method: str, target: int) -> MethodModel:
    d = pruned_root / method
    if not d.exists():
        raise FileNotFoundError(f'Method dir not found: {d}')

    cands = sorted(d.glob(f'*-oneshot-{method}-target{int(target)}-ebops*.weights.meta.json'))
    if not cands:
        raise FileNotFoundError(f'No meta found for method={method}, target={target} under {d}')

    # Prefer the latest generated one.
    meta_path = sorted(cands, key=lambda p: p.stat().st_mtime)[-1]
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    return MethodModel(
        method=method,
        target_ebops=int(target),
        measured_ebops=float(meta.get('post_prune_ebops_measured', np.nan)),
        meta_path=meta_path,
        weights_path=Path(meta['weights_path']),
    )


def pick_gradual_ckpt_for_target(baseline_dir: Path, target: int) -> MethodModel:
    cands = sorted(baseline_dir.glob('*.keras'))
    if not cands:
        raise FileNotFoundError(f'No baseline checkpoints found under {baseline_dir}')

    best = None
    for p in cands:
        name = p.name
        if 'ebops=' not in name:
            continue
        try:
            eb = float(name.split('ebops=')[1].split('-')[0])
        except Exception:
            continue
        d = abs(eb - float(target))
        rec = (d, eb, p)
        if best is None or rec[0] < best[0]:
            best = rec
    if best is None:
        raise FileNotFoundError(f'No parseable baseline checkpoint with ebops in {baseline_dir}')

    eb, p = best[1], best[2]
    return MethodModel(
        method='gradual',
        target_ebops=int(target),
        measured_ebops=float(eb),
        meta_path=p,
        weights_path=p,
    )


def _layer_score_map(layer) -> np.ndarray | None:
    if not hasattr(layer, 'kernel') or getattr(layer, 'kq', None) is None:
        return None

    kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
    if kernel.ndim < 2:
        return None

    try:
        from keras import ops

        bits = layer.kq.bits_(ops.shape(layer.kernel))
        bits = np.array(ops.convert_to_numpy(bits), dtype=np.float32)
    except Exception:
        return None

    # Flatten high-rank kernels to 2D for visualization
    if kernel.ndim > 2:
        k2 = kernel.reshape(-1, kernel.shape[-1])
        b2 = bits.reshape(-1, bits.shape[-1])
    else:
        k2 = kernel
        b2 = bits

    # Score uses both bitwidth and weight magnitude.
    # Positive values: active edges, darker => larger (bitwidth * |weight|).
    # Negative values: quantization dead zone defined by quantizer behavior:
    # active quantization + nonzero weight but quantized value == 0.
    bw = np.clip(b2, 0.0, None)
    wa = np.abs(k2)

    bw_pos = bw[bw > 0]
    wa_pos = wa[bw > 0]
    active = bw > 1e-6
    dead_zone = np.zeros_like(active, dtype=bool)
    try:
        from keras import ops

        qker = layer.kq(layer.kernel, training=False)
        qnp = np.array(ops.convert_to_numpy(qker), dtype=np.float32)
        if qnp.ndim > 2:
            q2 = qnp.reshape(-1, qnp.shape[-1])
        else:
            q2 = qnp
        dead_zone = active & (wa > 1e-8) & (np.abs(q2) <= 1e-12)
    except Exception:
        dead_zone = np.zeros_like(active, dtype=bool)
    if bw_pos.size == 0 or wa_pos.size == 0:
        out = np.zeros_like(bw, dtype=np.float32)
        out[dead_zone] = -1.0
        return out

    bw_scale = float(np.percentile(bw_pos, 95)) if bw_pos.size else 1.0
    wa_scale = float(np.percentile(wa_pos, 95)) if wa_pos.size else 1.0
    bw_scale = max(bw_scale, 1e-8)
    wa_scale = max(wa_scale, 1e-8)

    bw_n = np.clip(bw / bw_scale, 0.0, 1.0)
    wa_n = np.clip(wa / wa_scale, 0.0, 1.0)

    score = bw_n * np.sqrt(wa_n)
    out = np.where(active, score, 0.0).astype(np.float32)
    out[dead_zone] = -1.0
    return out


def load_layer_maps(base_model_path: Path, weights_path: Path, direct_model: bool = False) -> tuple[list[str], list[np.ndarray]]:
    if direct_model:
        model = keras.models.load_model(weights_path, compile=False)
    else:
        model = keras.models.load_model(base_model_path, compile=False)
        model.load_weights(weights_path)

    names: list[str] = []
    maps: list[np.ndarray] = []
    for layer in model._flatten_layers():
        sm = _layer_score_map(layer)
        if sm is None:
            continue
        names.append(layer.name)
        maps.append(sm)

    return names, maps


def plot_topology_compare(
    target: int,
    method_maps: dict[str, tuple[list[str], list[np.ndarray], float]],
    out_path: Path,
) -> None:
    methods = [m for m in ['gradual', 'spectral_quant', 'sensitivity', 'uniform'] if m in method_maps]
    if not methods:
        raise RuntimeError('No methods to plot')

    layer_names = method_maps[methods[0]][0]
    nrows = len(methods)
    ncols = len(layer_names)

    min_nonzero_darkness = 0.22
    dead_zone_rgb = np.array([0.82, 0.33, 0.33], dtype=np.float32)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows), squeeze=False)

    # Global normalization for cross-method comparability.
    all_pos = []
    for m in methods:
        _, maps, _ = method_maps[m]
        for mp in maps:
            pos = mp[mp > 0]
            if pos.size:
                all_pos.append(pos)
    if all_pos:
        all_pos = np.concatenate(all_pos)
        global_vmax = float(np.percentile(all_pos, 92))
    else:
        global_vmax = 1.0
    global_vmax = max(global_vmax, 1e-8)

    for r, m in enumerate(methods):
        names, maps, measured = method_maps[m]
        for c, lname in enumerate(layer_names):
            ax = axes[r, c]
            mp = maps[c]
            disp = np.clip(np.maximum(mp, 0.0) / global_vmax, 0.0, 1.0)
            disp = np.power(disp, 0.55, dtype=np.float32)  # stronger low-mid contrast
            nz = mp > 0
            disp = np.where(nz, min_nonzero_darkness + (1.0 - min_nonzero_darkness) * disp, 0.0).astype(np.float32)

            rgb = np.ones((disp.shape[0], disp.shape[1], 3), dtype=np.float32)
            rgb[nz] = (1.0 - disp[nz, None])  # grayscale on white background
            dz = mp < 0
            rgb[dz] = dead_zone_rgb  # red for quantization dead-zone points
            ax.imshow(rgb, aspect='auto', vmin=0.0, vmax=1.0)
            ax.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.2)
            if r == 0:
                ax.set_title(lname)
            if c == 0:
                label = METHOD_STYLE.get(m, m)
                ax.set_ylabel(f'{label}\nEBOPs={measured:.0f}')
            ax.set_xticks([])
            ax.set_yticks([])
    # Global grayscale colorbar for active edges.
    sm = plt.cm.ScalarMappable(cmap='Greys', norm=plt.Normalize(vmin=0.0, vmax=global_vmax))
    sm.set_array([])
    cax = fig.add_axes([0.945, 0.15, 0.015, 0.70])  # outside right
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Global intensity (bitwidth * |weight|)')

    # Minimal legend note for dead-zone color.
    fig.text(0.92, 0.98, 'Red: active & |w|>0 but Q(w)=0', ha='right', va='top', fontsize=9, color='black')

    fig.suptitle(f'Topology Contrast (target={target})\nwhite bg + black frame + global scale')
    fig.subplots_adjust(left=0.05, right=0.92, top=0.90, bottom=0.06, wspace=0.12, hspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
