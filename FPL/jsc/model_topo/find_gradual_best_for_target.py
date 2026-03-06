#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np


def parse_ckpt_info(p: Path):
    n = p.name
    m_e = re.search(r'ebops=(\d+)', n)
    m_a = re.search(r'val_acc=([0-9.]+)', n)
    m_ep = re.search(r'epoch=(\d+)', n)
    if not (m_e and m_a and m_ep):
        return None
    return {
        'path': p,
        'ebops': float(m_e.group(1)),
        'val_acc': float(m_a.group(1)),
        'epoch': int(m_ep.group(1)),
        'name': n,
    }


def best_ckpt_in_band(ckpts: list[dict], target: float, tol: float):
    lo = target * (1.0 - tol)
    hi = target * (1.0 + tol)
    cands = [c for c in ckpts if lo <= c['ebops'] <= hi]
    if not cands:
        return None
    cands.sort(key=lambda x: (-x['val_acc'], abs(x['ebops'] - target), x['epoch']))
    return cands[0]


def best_ckpt_under_budget(ckpts: list[dict], target: float):
    cands = [c for c in ckpts if c['ebops'] <= target]
    if not cands:
        return None
    cands.sort(key=lambda x: (-x['val_acc'], -x['ebops'], x['epoch']))
    return cands[0]


def best_trace_in_band(trace_h5: Path, target: float, tol: float):
    with h5py.File(trace_h5, 'r') as f:
        ep = np.array(f['epochs'][:], dtype=np.int64)
        eb = np.array(f['ebops'][:], dtype=np.float64)
        va = np.array(f['val_accuracy'][:], dtype=np.float64)
        vl = np.array(f['val_loss'][:], dtype=np.float64) if 'val_loss' in f else None
    lo = target * (1.0 - tol)
    hi = target * (1.0 + tol)
    m = (eb >= lo) & (eb <= hi)
    if not np.any(m):
        return None
    idxs = np.where(m)[0]
    i = idxs[np.argmax(va[idxs])]
    out = {
        'epoch': int(ep[i]),
        'ebops': float(eb[i]),
        'val_acc': float(va[i]),
    }
    if vl is not None:
        out['val_loss'] = float(vl[i])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Find best gradual baseline point near target EBOPs')
    parser.add_argument('--target', type=float, default=400.0)
    parser.add_argument('--tol', type=float, default=0.08, help='relative band, e.g. 0.08 => +-8%')
    parser.add_argument('--baseline_dir', type=str, default='results/baseline')
    parser.add_argument('--trace_h5', type=str, default='results/baseline/training_trace.h5')
    parser.add_argument('--trained_summary', type=str, default='model_topo/opt_train500/target400/summary.txt')
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    jsc_root = here.parent
    baseline_dir = (jsc_root / args.baseline_dir).resolve()
    trace_h5 = (jsc_root / args.trace_h5).resolve()
    trained_summary = (jsc_root / args.trained_summary).resolve()

    ckpts = []
    for p in baseline_dir.glob('*.keras'):
        info = parse_ckpt_info(p)
        if info is not None:
            ckpts.append(info)
    if not ckpts:
        raise FileNotFoundError(f'No parseable checkpoints under {baseline_dir}')

    best_band = best_ckpt_in_band(ckpts, target=float(args.target), tol=float(args.tol))
    best_le = best_ckpt_under_budget(ckpts, target=float(args.target))
    best_trace = best_trace_in_band(trace_h5=trace_h5, target=float(args.target), tol=float(args.tol))

    trained_best = None
    trained_eb = None
    if trained_summary.exists():
        txt = trained_summary.read_text(encoding='utf-8')
        ma = re.search(r'best_val_acc=([0-9.]+)', txt)
        me = re.search(r'final_ebops_measured=([0-9.]+)', txt)
        if ma:
            trained_best = float(ma.group(1))
        if me:
            trained_eb = float(me.group(1))

    print(f'[TARGET] {args.target:.1f} tol=+-{args.tol*100:.1f}%')
    print(f'[BASELINE DIR] {baseline_dir}')
    if best_band is not None:
        print(
            f"[CKPT best in band] epoch={best_band['epoch']} ebops={best_band['ebops']:.1f} "
            f"val_acc={best_band['val_acc']:.6f} file={best_band['name']}"
        )
    else:
        print('[CKPT best in band] None')

    if best_le is not None:
        print(
            f"[CKPT best <= target] epoch={best_le['epoch']} ebops={best_le['ebops']:.1f} "
            f"val_acc={best_le['val_acc']:.6f} file={best_le['name']}"
        )
    else:
        print('[CKPT best <= target] None')

    if best_trace is not None:
        print(
            f"[TRACE best in band] epoch={best_trace['epoch']} ebops={best_trace['ebops']:.1f} "
            f"val_acc={best_trace['val_acc']:.6f}"
        )
    else:
        print('[TRACE best in band] None')

    if trained_best is not None:
        print(f'[TRAINED] best_val_acc={trained_best:.6f} final_ebops={trained_eb}')
        if best_band is not None:
            print(f"[GAP vs gradual(ckpt-band)] {best_band['val_acc'] - trained_best:+.6f}")
        if best_trace is not None:
            print(f"[GAP vs gradual(trace-band)] {best_trace['val_acc'] - trained_best:+.6f}")


if __name__ == '__main__':
    main()

