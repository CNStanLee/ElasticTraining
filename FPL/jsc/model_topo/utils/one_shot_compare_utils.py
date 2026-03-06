from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import keras

from .prune_algorithms import build_sample_input, prune_once, extract_layer_profile


@dataclass
class RunResult:
    method: str
    target_ebops: float
    meta_path: str
    post_prune_ebops: float
    baseline_ebops: float


def parse_targets(targets: str) -> list[float]:
    vals = []
    for x in targets.split(','):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError('No valid targets parsed from --targets')
    return vals


def _run_one_shot_programmatic(
    checkpoint: Path,
    gradual_ref_ckpt: Path | None,
    target: float,
    method: str,
    sample_size: int,
    output_dir: Path,
    high_budget_ratio: float,
) -> tuple[Path, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(str(checkpoint), compile=False)
    sample_input = build_sample_input(model, sample_size=int(sample_size))
    target_profile = None
    if method == 'spectral_quant' and gradual_ref_ckpt is not None and gradual_ref_ckpt.exists():
        gm = keras.models.load_model(str(gradual_ref_ckpt), compile=False)
        target_profile = extract_layer_profile(gm)

    pr = prune_once(
        model=model,
        method=method,
        target_ebops=float(target),
        sample_input=sample_input,
        high_budget_ratio=float(high_budget_ratio),
        target_profile=target_profile,
    )

    # If profile-guided topology is infeasible at very low budget, fallback to
    # feasibility-first spectral pruning.
    if method == 'spectral_quant' and pr.post_prune_ebops > float(target) * 1.5:
        model = keras.models.load_model(str(checkpoint), compile=False)
        sample_input = build_sample_input(model, sample_size=int(sample_size))
        pr = prune_once(
            model=model,
            method=method,
            target_ebops=float(target),
            sample_input=sample_input,
            high_budget_ratio=float(high_budget_ratio),
            target_profile=None,
        )

    ckpt_name = checkpoint.stem
    weights_path = output_dir / (
        f'{ckpt_name}-oneshot-{method}-target{int(target)}-ebops{int(round(pr.post_prune_ebops))}.weights.h5'
    )
    model.save_weights(str(weights_path))

    meta = {
        'checkpoint': str(checkpoint),
        'target_ebops': float(target),
        'prune_method': method,
        'high_budget_ratio': float(high_budget_ratio),
        'sample_input': f'synthetic:{int(sample_size)}',
        'baseline_ebops_measured': float(pr.baseline_ebops),
        'post_prune_ebops_measured': float(pr.post_prune_ebops),
        'used_structured_low_budget': bool(pr.used_structured_low_budget),
        'calibrated': True,
        'weights_path': str(weights_path),
    }

    meta_path = weights_path.with_suffix('.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta_path, meta


def run_one_shot_once(
    repo_root: Path,
    checkpoint: str,
    target: float,
    method: str,
    input_h5: str,
    sample_size: int,
    output_dir: Path,
    python_exec: str | None = None,
    high_budget_ratio: float = 0.45,
) -> RunResult:
    del input_h5, python_exec
    jsc_root = (repo_root / 'FPL' / 'jsc').resolve()
    if str(jsc_root) not in sys.path:
        sys.path.insert(0, str(jsc_root))
    import model.model  # noqa: F401  # register custom layers

    ckpt_path = (repo_root / 'FPL' / 'jsc' / checkpoint).resolve()
    baseline_dir = (repo_root / 'FPL' / 'jsc' / 'model_topo' / 'baseline').resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')

    gradual_ref = None
    if method == 'spectral_quant' and baseline_dir.exists():
        cands = sorted(baseline_dir.glob('*.keras'))
        best = None
        for p in cands:
            if 'ebops=' not in p.name:
                continue
            try:
                eb = float(p.name.split('ebops=')[1].split('-')[0])
            except Exception:
                continue
            d = abs(eb - float(target))
            rec = (d, p)
            if best is None or rec[0] < best[0]:
                best = rec
        if best is not None:
            gradual_ref = best[1]

    print(f'[RUN] method={method} target={target} checkpoint={ckpt_path.name}', flush=True)
    meta_path, meta = _run_one_shot_programmatic(
        checkpoint=ckpt_path,
        gradual_ref_ckpt=gradual_ref,
        target=float(target),
        method=method,
        sample_size=int(sample_size),
        output_dir=output_dir,
        high_budget_ratio=float(high_budget_ratio),
    )

    return RunResult(
        method=method,
        target_ebops=float(target),
        meta_path=str(meta_path),
        post_prune_ebops=float(meta.get('post_prune_ebops_measured', float('nan'))),
        baseline_ebops=float(meta.get('baseline_ebops_measured', float('nan'))),
    )


def run_compare(
    repo_root: Path,
    checkpoint: str,
    targets: Iterable[float],
    methods: Iterable[str],
    input_h5: str,
    sample_size: int,
    out_root: Path,
    python_exec: str | None = None,
    high_budget_ratio: float = 0.45,
) -> list[RunResult]:
    rows: list[RunResult] = []
    for t in targets:
        for m in methods:
            out_dir = out_root / m
            rows.append(
                run_one_shot_once(
                    repo_root=repo_root,
                    checkpoint=checkpoint,
                    target=t,
                    method=m,
                    input_h5=input_h5,
                    sample_size=sample_size,
                    output_dir=out_dir,
                    python_exec=python_exec,
                    high_budget_ratio=high_budget_ratio,
                )
            )
    return rows


def write_summary_csv(rows: list[RunResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method', 'target_ebops', 'post_prune_ebops', 'baseline_ebops', 'meta_path'])
        for r in rows:
            w.writerow([r.method, r.target_ebops, r.post_prune_ebops, r.baseline_ebops, r.meta_path])


def default_python_from_env() -> str:
    return os.environ.get('PYTHON', sys.executable)
