#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import argparse

import keras
import numpy as np
import tensorflow as tf

import model.model  # noqa: F401
from data.data import get_data
from run_one_shot_prune_only import build_sample_input, compute_model_ebops
from utils.ramanujan_budget_utils import _flatten_layers
from utils.topology_graph_plot_utils import TopologyGraphPlotter

DEFAULT_CHECKPOINT = "results/baseline/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras"
DEFAULT_METHODS = ["baseline", "uniform", "sensitivity", "snip", "grasp", "synflow", "spectral_quant", "snows"]
DEFAULT_TARGETS = [400, 1500, 2500]


def _to_float(x) -> float:
    try:
        return float(x.numpy())
    except Exception:
        return float(x)


def _labels_to_sparse(y_np: np.ndarray) -> np.ndarray:
    y_np = np.array(y_np)
    if y_np.ndim >= 2 and y_np.shape[-1] > 1:
        return np.argmax(y_np, axis=-1).astype(np.int32)
    return y_np.reshape(-1).astype(np.int32)


def _bits_np(q, shape_like):
    if q is None:
        return None
    try:
        from keras import ops

        bits = q.bits_(ops.shape(shape_like))
        return np.array(ops.convert_to_numpy(bits), dtype=np.float32)
    except Exception:
        return None


def _flatten_kernel_and_mask(kernel_np: np.ndarray, mask_np: np.ndarray | None):
    if kernel_np.ndim > 2:
        k2 = kernel_np.reshape(-1, kernel_np.shape[-1])
        m2 = mask_np.reshape(-1, mask_np.shape[-1]) if mask_np is not None else None
    else:
        k2 = kernel_np
        m2 = mask_np
    return k2, m2


def _grad_stats(model, x_grad, y_grad) -> dict[str, float]:
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        logits = model(x_grad, training=True)
        loss = loss_fn(y_grad, logits)
    grads = tape.gradient(loss, model.trainable_variables)

    g_sumsq = 0.0
    g_total = 0
    g_near_zero = 0
    kernel_grad_norms = []

    for v, g in zip(model.trainable_variables, grads):
        if g is None:
            continue
        g_np = np.array(g.numpy(), dtype=np.float32)
        g_sumsq += float(np.sum(g_np * g_np))
        g_total += int(g_np.size)
        g_near_zero += int(np.sum(np.abs(g_np) < 1e-8))
        if "kernel" in v.name:
            kernel_grad_norms.append(float(np.linalg.norm(g_np)))

    first_last = 0.0
    log_std = 0.0
    if kernel_grad_norms:
        first_last = float(kernel_grad_norms[-1] / (kernel_grad_norms[0] + 1e-12))
        log_std = float(np.std(np.log(np.array(kernel_grad_norms) + 1e-12)))

    return {
        "loss": float(_to_float(loss)),
        "grad_global_norm": float(np.sqrt(g_sumsq)),
        "grad_near_zero_ratio": float(g_near_zero / max(g_total, 1)),
        "grad_first_last_ratio": float(first_last),
        "grad_log_std": float(log_std),
    }


def _one_step_loss_drop(model, x_grad, y_grad, lr: float = 1e-5) -> float:
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    backups = [v.numpy().copy() for v in model.trainable_variables]
    try:
        with tf.GradientTape() as tape:
            logits = model(x_grad, training=True)
            pre_loss = loss_fn(y_grad, logits)
        grads = tape.gradient(pre_loss, model.trainable_variables)
        opt = keras.optimizers.SGD(learning_rate=float(lr), momentum=0.0)
        opt.apply_gradients([(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
        post_logits = model(x_grad, training=True)
        post_loss = loss_fn(y_grad, post_logits)
        return float(_to_float(pre_loss) - _to_float(post_loss))
    finally:
        for v, b in zip(model.trainable_variables, backups):
            v.assign(b)


def _training_probe(
    model,
    x_train_probe,
    y_train_probe,
    x_holdout,
    y_holdout,
    steps: int = 12,
    batch_size: int = 256,
    lr_candidates: tuple[float, ...] = (1e-4, 3e-4, 1e-3),
) -> dict[str, float | bool]:
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    backups = [v.numpy().copy() for v in model.trainable_variables]
    x_train_probe = tf.convert_to_tensor(x_train_probe, dtype=tf.float32)
    y_train_probe = tf.convert_to_tensor(y_train_probe, dtype=tf.int32)
    x_holdout = tf.convert_to_tensor(x_holdout, dtype=tf.float32)
    y_holdout = tf.convert_to_tensor(y_holdout, dtype=tf.int32)
    n = int(x_train_probe.shape[0])
    bs = max(1, min(int(batch_size), n))
    best = None
    try:
        for lr in lr_candidates:
            for v, b in zip(model.trainable_variables, backups):
                v.assign(b)

            opt = keras.optimizers.Adam(learning_rate=float(lr))
            train_losses = []
            finite_steps = 0
            nonzero_grad_steps = 0
            best_train_loss = float("inf")
            start_train_loss = None
            holdout_start = float(_to_float(loss_fn(y_holdout, model(x_holdout, training=False))))

            for step in range(max(1, int(steps))):
                start = (step * bs) % n
                end = start + bs
                if end <= n:
                    xb = x_train_probe[start:end]
                    yb = y_train_probe[start:end]
                else:
                    overflow = end - n
                    xb = tf.concat([x_train_probe[start:n], x_train_probe[0:overflow]], axis=0)
                    yb = tf.concat([y_train_probe[start:n], y_train_probe[0:overflow]], axis=0)

                with tf.GradientTape() as tape:
                    logits = model(xb, training=True)
                    loss = loss_fn(yb, logits)
                grads = tape.gradient(loss, model.trainable_variables)
                grad_norm_sq = 0.0
                pairs = []
                for g, v in zip(grads, model.trainable_variables):
                    if g is None:
                        continue
                    g_np = np.array(g.numpy(), dtype=np.float32)
                    grad_norm_sq += float(np.sum(g_np * g_np))
                    pairs.append((g, v))

                loss_val = float(_to_float(loss))
                if not np.isfinite(loss_val):
                    break
                finite_steps += 1
                if np.isfinite(grad_norm_sq) and grad_norm_sq > 1e-16:
                    nonzero_grad_steps += 1
                if start_train_loss is None:
                    start_train_loss = loss_val
                train_losses.append(loss_val)
                best_train_loss = min(best_train_loss, loss_val)
                if pairs:
                    opt.apply_gradients(pairs)

            holdout_end = float(_to_float(loss_fn(y_holdout, model(x_holdout, training=False))))
            if start_train_loss is None:
                start_train_loss = float("inf")
            end_train_loss = float(train_losses[-1]) if train_losses else float("inf")
            train_best_drop = (
                float(start_train_loss - best_train_loss)
                if np.isfinite(start_train_loss) and np.isfinite(best_train_loss)
                else float("-inf")
            )
            train_rel_drop = (
                float(train_best_drop / max(abs(start_train_loss), 1e-8))
                if np.isfinite(train_best_drop)
                else float("-inf")
            )
            holdout_drop = (
                float(holdout_start - holdout_end)
                if np.isfinite(holdout_start) and np.isfinite(holdout_end)
                else float("-inf")
            )
            holdout_rel_drop = (
                float(holdout_drop / max(abs(holdout_start), 1e-8))
                if np.isfinite(holdout_drop)
                else float("-inf")
            )
            stable = (
                finite_steps >= max(3, int(steps) // 2)
                and nonzero_grad_steps >= max(2, int(steps) // 3)
                and np.isfinite(holdout_end)
                and holdout_end <= holdout_start * 1.05
            )
            improved = (
                train_rel_drop > 2e-3
                or holdout_rel_drop > 5e-4
                or (stable and train_rel_drop > 5e-4 and holdout_rel_drop > -5e-3)
            )
            trainable = bool(stable and improved)
            if trainable:
                verdict = "ok"
            elif finite_steps < max(3, int(steps) // 2):
                verdict = "nonfinite"
            elif nonzero_grad_steps < max(2, int(steps) // 3):
                verdict = "zero_grad"
            elif not np.isfinite(holdout_end) or holdout_end > holdout_start * 1.2:
                verdict = "unstable"
            else:
                verdict = "no_progress"

            rec = {
                "probe_trainable": bool(trainable),
                "probe_start_loss": float(start_train_loss),
                "probe_end_loss": float(end_train_loss),
                "probe_best_loss": float(best_train_loss),
                "probe_loss_drop": float(start_train_loss - end_train_loss) if np.isfinite(end_train_loss) else float("-inf"),
                "probe_best_drop": float(train_best_drop),
                "probe_rel_best_drop": float(train_rel_drop),
                "probe_holdout_start": float(holdout_start),
                "probe_holdout_end": float(holdout_end),
                "probe_holdout_drop": float(holdout_drop),
                "probe_holdout_rel_drop": float(holdout_rel_drop),
                "probe_finite_steps": float(finite_steps),
                "probe_nonzero_grad_steps": float(nonzero_grad_steps),
                "probe_lr": float(lr),
                "probe_verdict": verdict,
            }

            if best is None:
                best = rec
            else:
                best_key = (
                    int(bool(best["probe_trainable"])),
                    float(best["probe_holdout_rel_drop"]),
                    float(best["probe_rel_best_drop"]),
                )
                rec_key = (
                    int(bool(rec["probe_trainable"])),
                    float(rec["probe_holdout_rel_drop"]),
                    float(rec["probe_rel_best_drop"]),
                )
                if rec_key > best_key:
                    best = rec

        assert best is not None
        return best
    finally:
        for v, b in zip(model.trainable_variables, backups):
            v.assign(b)


def _judge_trainable(
    grad_norm: float,
    near_zero: float,
    first_last: float,
    one_step_drop: float,
    probe: dict[str, float | bool],
) -> tuple[bool, str]:
    if bool(probe["probe_trainable"]):
        return True, "ok"
    reasons = [str(probe["probe_verdict"])]
    if not (np.isfinite(grad_norm) and grad_norm > 1e-10):
        reasons.append("bad_grad")
    if near_zero >= 0.999:
        reasons.append("zero_grads")
    if np.isfinite(first_last) and first_last == 0.0:
        reasons.append("last_grad_zero")
    if one_step_drop < -5e-2:
        reasons.append("step_bad")
    return False, ",".join(sorted(set(reasons)))


def _judge_repr_trainable(
    mean_output_degree: float,
    disconnected_output_ratio: float,
    last_layer_mean_degree: float,
    fully_disconnected_layers: int,
    active_total: int,
) -> tuple[bool, str]:
    reasons = []
    if active_total <= 0:
        reasons.append("no_active_edges")
    if mean_output_degree <= 5e-2:
        reasons.append("deg_zero")
    if disconnected_output_ratio >= 0.995:
        reasons.append("all_outputs_disconnected")
    if last_layer_mean_degree <= 0.0:
        reasons.append("last_layer_empty")
    if fully_disconnected_layers > 0:
        reasons.append("empty_layer")
    if reasons:
        return False, ",".join(sorted(set(reasons)))
    return True, "ok"


def _spectral_layer_stats(mask_2d: np.ndarray) -> dict[str, float]:
    a = np.array(mask_2d, dtype=np.float32)
    if a.size == 0 or int(np.sum(a > 0.5)) == 0:
        return {
            "score": 0.0,
            "effective_rank_ratio": 0.0,
            "stable_rank_ratio": 0.0,
            "condition_ratio": 0.0,
            "rank_ratio": 0.0,
        }

    s = np.linalg.svd(a, compute_uv=False)
    s = s[np.isfinite(s)]
    s = s[s > 1e-8]
    if s.size == 0:
        return {
            "score": 0.0,
            "effective_rank_ratio": 0.0,
            "stable_rank_ratio": 0.0,
            "condition_ratio": 0.0,
            "rank_ratio": 0.0,
        }

    rmax = float(max(1, min(a.shape)))
    power = s * s
    p = power / max(float(np.sum(power)), 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    effective_rank_ratio = float(np.exp(entropy) / rmax)
    stable_rank_ratio = float(np.sum(power) / max(float(s[0] * s[0]), 1e-12) / rmax)
    condition_ratio = float(s[-1] / max(float(s[0]), 1e-12))
    rank_ratio = float(len(s) / rmax)
    score = float(
        np.clip(
            0.35 * effective_rank_ratio
            + 0.35 * stable_rank_ratio
            + 0.20 * condition_ratio
            + 0.10 * rank_ratio,
            0.0,
            1.0,
        )
    )
    return {
        "score": score,
        "effective_rank_ratio": effective_rank_ratio,
        "stable_rank_ratio": stable_rank_ratio,
        "condition_ratio": condition_ratio,
        "rank_ratio": rank_ratio,
    }


def _spectral_regime(
    network_score: float,
    bottleneck_score: float,
    last_layer_score: float,
    fully_disconnected_layers: int,
) -> tuple[str, bool]:
    if (
        fully_disconnected_layers > 0
        or bottleneck_score < 0.03
        or last_layer_score < 0.03
        or network_score < 15.0
    ):
        return "cannot_train", False
    if network_score < 35.0 or bottleneck_score < 0.12 or last_layer_score < 0.10:
        return "suboptimal", True
    return "healthy", True


def _compute_plot_metrics(model, x_grad, y_grad, x_probe_train, y_probe_train) -> dict[str, float | bool | str]:
    n_w_total = 0
    n_w_dead = 0
    out_total = 0
    out_disconnected = 0
    layer_deg_means = []
    active_total = 0
    dead_zone_total = 0
    masks = []
    spectral_layer_scores = []
    spectral_effective_ranks = []
    spectral_stable_ranks = []
    spectral_condition_ratios = []
    spectral_rank_ratios = []

    for layer in _flatten_layers(model):
        if not hasattr(layer, "kernel") or getattr(layer, "kq", None) is None:
            continue

        kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
        bits = _bits_np(layer.kq, layer.kernel)
        if bits is None:
            continue

        active = bits > 1e-6
        k2, m2 = _flatten_kernel_and_mask(kernel, active.astype(np.float32))
        if m2 is None:
            continue
        active2 = m2 > 0.5
        masks.append(active2.astype(bool))
        spec = _spectral_layer_stats(active2.astype(np.float32))
        spectral_layer_scores.append(float(spec["score"]))
        spectral_effective_ranks.append(float(spec["effective_rank_ratio"]))
        spectral_stable_ranks.append(float(spec["stable_rank_ratio"]))
        spectral_condition_ratios.append(float(spec["condition_ratio"]))
        spectral_rank_ratios.append(float(spec["rank_ratio"]))

        n_w_total += int(active2.size)
        n_w_dead += int(np.sum(~active2))

        deg = np.sum(active2, axis=0)
        out_total += int(deg.size)
        out_disconnected += int(np.sum(deg == 0))
        layer_deg_means.append(float(np.mean(deg)))

        try:
            from keras import ops

            qker = layer.kq(layer.kernel, training=False)
            qnp = np.array(ops.convert_to_numpy(qker), dtype=np.float32)
            q2, _ = _flatten_kernel_and_mask(qnp, active2.astype(np.float32))
            dead_zone = active2 & (np.abs(k2) > 1e-8) & (np.abs(q2) <= 1e-12)
            dead_zone_total += int(np.sum(dead_zone))
        except Exception:
            pass
        active_total += int(np.sum(active2))

    bridge_in_only = 0
    bridge_out_only = 0
    for i in range(len(masks) - 1):
        prev_cols = np.sum(masks[i], axis=0) > 0
        next_rows = np.sum(masks[i + 1], axis=1) > 0
        w = min(len(prev_cols), len(next_rows))
        if w <= 0:
            continue
        prev_cols = prev_cols[:w]
        next_rows = next_rows[:w]
        bridge_in_only += int(np.sum(prev_cols & (~next_rows)))
        bridge_out_only += int(np.sum((~prev_cols) & next_rows))

    gs = _grad_stats(model, x_grad, y_grad)
    one_drop = _one_step_loss_drop(model, x_grad, y_grad, lr=1e-5)
    probe = _training_probe(
        model,
        x_train_probe=x_probe_train,
        y_train_probe=y_probe_train,
        x_holdout=x_grad,
        y_holdout=y_grad,
        steps=12,
        batch_size=min(256, int(x_probe_train.shape[0])),
        lr_candidates=(1e-4, 3e-4, 1e-3),
    )
    opt_trainable, opt_verdict = _judge_trainable(
        grad_norm=float(gs["grad_global_norm"]),
        near_zero=float(gs["grad_near_zero_ratio"]),
        first_last=float(gs["grad_first_last_ratio"]),
        one_step_drop=float(one_drop),
        probe=probe,
    )
    mean_output_degree = float(np.mean(layer_deg_means)) if layer_deg_means else 0.0
    disconnected_output_ratio = float(out_disconnected / max(out_total, 1))
    last_layer_mean_degree = float(layer_deg_means[-1]) if layer_deg_means else 0.0
    fully_disconnected_layers = int(sum(1 for x in layer_deg_means if x <= 1e-12))
    repr_trainable, repr_verdict = _judge_repr_trainable(
        mean_output_degree=mean_output_degree,
        disconnected_output_ratio=disconnected_output_ratio,
        last_layer_mean_degree=last_layer_mean_degree,
        fully_disconnected_layers=fully_disconnected_layers,
        active_total=int(active_total),
    )
    spectral_bottleneck_score = 100.0 * float(min(spectral_layer_scores)) if spectral_layer_scores else 0.0
    spectral_mean_score = 100.0 * float(np.mean(spectral_layer_scores)) if spectral_layer_scores else 0.0
    spectral_last_score = 100.0 * float(spectral_layer_scores[-1]) if spectral_layer_scores else 0.0
    spectral_score = float(
        np.clip(
            0.50 * spectral_bottleneck_score + 0.30 * spectral_mean_score + 0.20 * spectral_last_score,
            0.0,
            100.0,
        )
    )
    spectral_regime, spectral_repr_ok = _spectral_regime(
        network_score=spectral_score,
        bottleneck_score=spectral_bottleneck_score,
        last_layer_score=spectral_last_score,
        fully_disconnected_layers=fully_disconnected_layers,
    )
    repr_trainable = bool(repr_trainable and spectral_repr_ok)
    if not repr_trainable and repr_verdict == "ok":
        repr_verdict = f"spectral:{spectral_regime}"
    trainable = bool(opt_trainable and repr_trainable)
    if trainable:
        verdict = "ok"
    else:
        reasons = []
        if not opt_trainable:
            reasons.append(f"opt:{opt_verdict}")
        if not repr_trainable:
            reasons.append(f"repr:{repr_verdict}")
        verdict = "|".join(reasons)

    return {
        "dead_ratio_kernel": float(n_w_dead / max(n_w_total, 1)),
        "disconnected_output_ratio": disconnected_output_ratio,
        "mean_output_degree": mean_output_degree,
        "bridge_in_only_total": float(bridge_in_only),
        "bridge_out_only_total": float(bridge_out_only),
        "qdead_active_ratio": float(dead_zone_total / max(active_total, 1)),
        "grad_global_norm": float(gs["grad_global_norm"]),
        "grad_near_zero_ratio": float(gs["grad_near_zero_ratio"]),
        "grad_first_last_ratio": float(gs["grad_first_last_ratio"]),
        "one_step_loss_drop": float(one_drop),
        "probe_start_loss": float(probe["probe_start_loss"]),
        "probe_end_loss": float(probe["probe_end_loss"]),
        "probe_best_drop": float(probe["probe_best_drop"]),
        "probe_rel_best_drop": float(probe["probe_rel_best_drop"]),
        "probe_holdout_start": float(probe["probe_holdout_start"]),
        "probe_holdout_end": float(probe["probe_holdout_end"]),
        "probe_holdout_drop": float(probe["probe_holdout_drop"]),
        "probe_holdout_rel_drop": float(probe["probe_holdout_rel_drop"]),
        "probe_finite_steps": float(probe["probe_finite_steps"]),
        "probe_nonzero_grad_steps": float(probe["probe_nonzero_grad_steps"]),
        "probe_lr": float(probe["probe_lr"]),
        "last_layer_mean_degree": float(last_layer_mean_degree),
        "fully_disconnected_layers": float(fully_disconnected_layers),
        "spectral_score": float(spectral_score),
        "spectral_bottleneck_score": float(spectral_bottleneck_score),
        "spectral_mean_score": float(spectral_mean_score),
        "spectral_last_score": float(spectral_last_score),
        "spectral_effective_rank_mean": float(np.mean(spectral_effective_ranks)) if spectral_effective_ranks else 0.0,
        "spectral_stable_rank_mean": float(np.mean(spectral_stable_ranks)) if spectral_stable_ranks else 0.0,
        "spectral_condition_ratio_mean": float(np.mean(spectral_condition_ratios)) if spectral_condition_ratios else 0.0,
        "spectral_rank_ratio_mean": float(np.mean(spectral_rank_ratios)) if spectral_rank_ratios else 0.0,
        "spectral_regime": spectral_regime,
        "opt_trainable": bool(opt_trainable),
        "repr_trainable": bool(repr_trainable),
        "opt_verdict": opt_verdict,
        "repr_verdict": repr_verdict,
        "trainable": bool(trainable),
        "verdict": verdict,
    }


def _find_latest_weights(search_roots: list[Path], method: str, target: int) -> Path | None:
    pattern = f"**/*-oneshot-{method}-target{int(target)}-ebops*.weights.h5"
    cands: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        cands.extend(p for p in root.glob(pattern) if p.is_file())
    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime)[-1]


def _extract_ebops_from_name(path: Path) -> float | None:
    name = path.name
    if "ebops=" not in name:
        return None
    try:
        return float(name.split("ebops=")[1].split("-")[0])
    except Exception:
        return None


def _pick_baseline_checkpoint(baseline_dir: Path, target: int) -> Path:
    cands = sorted(baseline_dir.glob("*.keras"))
    if not cands:
        raise FileNotFoundError(f"No baseline checkpoints found under {baseline_dir}")

    best: tuple[float, float, Path] | None = None
    for p in cands:
        eb = _extract_ebops_from_name(p)
        if eb is None:
            continue
        rec = (abs(float(target) - eb), eb, p)
        if best is None or rec[0] < best[0]:
            best = rec
    if best is None:
        raise FileNotFoundError(f"No parseable baseline checkpoint in {baseline_dir}")
    return best[2]


def _load_meta(weights_path: Path) -> dict:
    meta_path = weights_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _generate_missing_weights(
    python_exe: str,
    checkpoint: Path,
    input_h5: Path,
    output_root: Path,
    method: str,
    target: int,
    sample_size: int,
) -> Path:
    out_dir = output_root / method
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        "run_one_shot_prune_only.py",
        "--checkpoint",
        str(checkpoint),
        "--target_ebops",
        str(int(target)),
        "--prune_method",
        method,
        "--input_h5",
        str(input_h5),
        "--sample_size",
        str(int(sample_size)),
        "--output_dir",
        str(out_dir),
    ]
    if method == "snows":
        cmd.extend(
            [
                "--snows_init_method",
                "sensitivity",
                "--snows_k_step",
                "2",
                "--snows_newton_steps",
                "2",
                "--snows_cg_iters",
                "20",
            ]
        )

    print(f"[generate] method={method:14s} target={target:4d}")
    subprocess.run(cmd, check=True, cwd=HERE)

    weights = _find_latest_weights([out_dir], method=method, target=target)
    if weights is None:
        raise FileNotFoundError(f"Generated weights not found for method={method}, target={target}")
    return weights


def _load_pruned_model(weights_path: Path, checkpoint: Path) -> tuple[keras.Model, float]:
    meta = _load_meta(weights_path)
    ckpt = Path(meta.get("checkpoint", checkpoint))
    if not ckpt.is_absolute():
        ckpt = (HERE / ckpt).resolve()

    model = keras.models.load_model(ckpt, compile=False)
    model.load_weights(weights_path)

    measured_ebops = meta.get("post_prune_ebops_measured")
    if measured_ebops is None:
        sample_input, _ = build_sample_input(model, sample_size=512, input_h5="data/dataset.h5")
        measured_ebops = compute_model_ebops(model, sample_input)

    return model, float(measured_ebops)


def _load_baseline_model(keras_path: Path) -> tuple[keras.Model, float]:
    model = keras.models.load_model(keras_path, compile=False)
    measured_ebops = _extract_ebops_from_name(keras_path)
    if measured_ebops is None:
        sample_input, _ = build_sample_input(model, sample_size=512, input_h5="data/dataset.h5")
        measured_ebops = compute_model_ebops(model, sample_input)
    return model, float(measured_ebops)


def _overall_score(metrics: dict[str, float | bool | str]) -> dict[str, float]:
    conn_disc = max(0.0, 1.0 - float(metrics["disconnected_output_ratio"]))
    conn_deg = np.tanh(max(float(metrics["mean_output_degree"]), 0.0) / 2.0)
    bridge_pen = np.exp(-0.10 * (float(metrics["bridge_in_only_total"]) + float(metrics["bridge_out_only_total"])))
    conn_score = float(np.clip(0.45 * conn_disc + 0.35 * conn_deg + 0.20 * bridge_pen, 0.0, 1.0))

    grad_dense = max(0.0, 1.0 - float(metrics["grad_near_zero_ratio"]))
    fl = abs(float(metrics["grad_first_last_ratio"]))
    depth_bal = float(np.exp(-abs(np.log(max(fl, 1e-12)))))
    step = float(metrics["one_step_loss_drop"])
    step_score = 0.0 if (not np.isfinite(step) or step < 0.0) else float(np.tanh(step / 1e-3))
    probe_step = float(metrics["probe_best_drop"])
    probe_score = 0.0 if (not np.isfinite(probe_step) or probe_step < 0.0) else float(np.tanh(probe_step / 1e-3))
    holdout_probe = float(metrics["probe_holdout_rel_drop"])
    holdout_probe_score = 0.0 if (not np.isfinite(holdout_probe)) else float(np.clip(0.5 + 25.0 * holdout_probe, 0.0, 1.0))
    train_flag = 1.0 if bool(metrics["trainable"]) else 0.0
    repr_flag = 1.0 if bool(metrics["repr_trainable"]) else 0.0
    train_score = float(
        np.clip(
            0.25 * train_flag + 0.15 * repr_flag + 0.10 * grad_dense + 0.10 * depth_bal + 0.10 * step_score + 0.15 * probe_score + 0.15 * holdout_probe_score,
            0.0,
            1.0,
        )
    )
    spectral_repr_score = float(np.clip(float(metrics["spectral_score"]) / 100.0, 0.0, 1.0))
    train_score = float(np.clip(0.75 * train_score + 0.25 * spectral_repr_score, 0.0, 1.0))

    qdead_score = float(np.clip(1.0 - float(metrics["qdead_active_ratio"]), 0.0, 1.0))
    dead_kernel_score = float(np.clip(1.0 - float(metrics["dead_ratio_kernel"]), 0.0, 1.0))
    quant_score = float(np.clip(0.75 * qdead_score + 0.25 * dead_kernel_score, 0.0, 1.0))

    total = float(np.clip(100.0 * (0.40 * conn_score + 0.40 * train_score + 0.20 * quant_score), 0.0, 100.0))
    return {
        "overall_score": total,
        "connectivity_score": 100.0 * conn_score,
        "trainability_score": 100.0 * train_score,
        "quant_deadzone_score": 100.0 * quant_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one-shot pruning topologies for multiple methods/targets."
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--baseline_dir", type=str, default="results/baseline")
    parser.add_argument("--input_h5", type=str, default="data/dataset.h5")
    parser.add_argument("--sample_size", type=int, default=512)
    parser.add_argument("--eval_batch", type=int, default=2048)
    parser.add_argument("--grad_batch", type=int, default=4096)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--targets", nargs="+", type=int, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--python_exe",
        type=str,
        default="/home/changhong/anaconda3/envs/py12tf/bin/python",
        help="Python executable used to generate missing one-shot models.",
    )
    parser.add_argument(
        "--generated_root",
        type=str,
        default="results/one_shot_topology_models",
        help="Where to store newly generated one-shot weights.",
    )
    parser.add_argument(
        "--plots_root",
        type=str,
        default="results/one_shot_topology_plots",
        help="Where to store topology figures.",
    )
    parser.add_argument(
        "--no_generate_missing",
        action="store_true",
        help="Do not generate missing one-shot weights; fail instead.",
    )
    args = parser.parse_args()

    checkpoint = (HERE / args.checkpoint).resolve()
    baseline_dir = (HERE / args.baseline_dir).resolve()
    input_h5 = (HERE / args.input_h5).resolve()
    generated_root = (HERE / args.generated_root).resolve()
    plots_root = (HERE / args.plots_root).resolve()
    plots_root.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_val, y_val), _ = get_data(input_h5, src="openml")
    n_eval = min(len(x_val), max(1, int(args.eval_batch)))
    n_grad = min(len(x_val), max(1, int(args.grad_batch)))
    n_probe_train = min(len(x_train), max(1024, int(args.grad_batch)))
    x_eval = tf.constant(x_val[:n_eval], dtype=tf.float32)
    y_eval = tf.constant(_labels_to_sparse(y_val[:n_eval]), dtype=tf.int32)
    x_grad = tf.constant(x_val[:n_grad], dtype=tf.float32)
    y_grad = tf.constant(_labels_to_sparse(y_val[:n_grad]), dtype=tf.int32)
    x_probe_train = tf.constant(x_train[:n_probe_train], dtype=tf.float32)
    y_probe_train = tf.constant(_labels_to_sparse(y_train[:n_probe_train]), dtype=tf.int32)

    generated_search_roots = [generated_root]
    fallback_search_roots = [
        HERE / "results" / "one_shot_prune_only",
        HERE / "model_to_analyse" / "oneshot",
        HERE / "model_to_analyse" / "oneshot_opt",
        HERE / "results",
    ]

    plotter = TopologyGraphPlotter(
        symmetric_topology_plot=False,
        mirror_edges=False,
        plot_matrix=True,
        strict_original_connections=True,
    )

    summary_rows: list[dict[str, str | int | float]] = []

    for target in args.targets:
        for method in args.methods:
            if method == "baseline":
                weights_path = _pick_baseline_checkpoint(baseline_dir=baseline_dir, target=int(target))
                model, measured_ebops = _load_baseline_model(weights_path)
            else:
                weights_path = _find_latest_weights(generated_search_roots, method=method, target=target)
                if weights_path is None and args.no_generate_missing:
                    weights_path = _find_latest_weights(fallback_search_roots, method=method, target=target)
                    if weights_path is None:
                        raise FileNotFoundError(f"No weights found for method={method}, target={target}")
                if weights_path is None:
                    weights_path = _generate_missing_weights(
                        python_exe=args.python_exe,
                        checkpoint=checkpoint,
                        input_h5=input_h5,
                        output_root=generated_root,
                        method=method,
                        target=int(target),
                        sample_size=int(args.sample_size),
                    )

                model, measured_ebops = _load_pruned_model(weights_path, checkpoint=checkpoint)
            layers = plotter.extract_layer_graph_data(model)
            plot_metrics = _compute_plot_metrics(
                model,
                x_grad=x_grad,
                y_grad=y_grad,
                x_probe_train=x_probe_train,
                y_probe_train=y_probe_train,
            )
            score_metrics = _overall_score(plot_metrics)

            method_dir = plots_root / f"target{int(target)}" / method
            method_dir.mkdir(parents=True, exist_ok=True)
            stem = weights_path.stem
            subtitle = "\n".join(
                [
                    f"target_ebops={int(target)}   measured_ebops={measured_ebops:.0f}",
                    (
                        f"score: total={float(score_metrics['overall_score']):.1f}/100   "
                        f"conn={float(score_metrics['connectivity_score']):.1f}   "
                        f"train={float(score_metrics['trainability_score']):.1f}   "
                        f"q={float(score_metrics['quant_deadzone_score']):.1f}"
                    ),
                    (
                        f"conn: disc_out={100.0*float(plot_metrics['disconnected_output_ratio']):.1f}%   "
                        f"mean_deg={float(plot_metrics['mean_output_degree']):.2f}   "
                        f"in_only={float(plot_metrics['bridge_in_only_total']):.0f}   "
                        f"out_only={float(plot_metrics['bridge_out_only_total']):.0f}"
                    ),
                    (
                        f"train: {'yes' if bool(plot_metrics['trainable']) else 'no'}({plot_metrics['verdict']})   "
                        f"opt={'yes' if bool(plot_metrics['opt_trainable']) else 'no'}   "
                        f"repr={'yes' if bool(plot_metrics['repr_trainable']) else 'no'}   "
                        f"g={float(plot_metrics['grad_global_norm']):.2e}   "
                        f"zero_g={100.0*float(plot_metrics['grad_near_zero_ratio']):.1f}%   "
                        f"f/l={float(plot_metrics['grad_first_last_ratio']):.2e}   "
                        f"d1={float(plot_metrics['one_step_loss_drop']):.2e}   "
                        f"probe={float(plot_metrics['probe_rel_best_drop']):.2e}   "
                        f"h={float(plot_metrics['probe_holdout_rel_drop']):.2e}   "
                        f"last_deg={float(plot_metrics['last_layer_mean_degree']):.2f}   "
                        f"lr={float(plot_metrics['probe_lr']):.0e}"
                    ),
                    (
                        f"spec: {plot_metrics['spectral_regime']}   "
                        f"S={float(plot_metrics['spectral_score']):.1f}   "
                        f"bottleneck={float(plot_metrics['spectral_bottleneck_score']):.1f}   "
                        f"last={float(plot_metrics['spectral_last_score']):.1f}"
                    ),
                    (
                        f"qdead: active_zero={100.0*float(plot_metrics['qdead_active_ratio']):.1f}%   "
                        f"dead_kernel={100.0*float(plot_metrics['dead_ratio_kernel']):.1f}%"
                    ),
                ]
            )
            title_prefix = "Baseline" if method == "baseline" else f"One-shot {method}"

            matrix_path = method_dir / f"{stem}_weighted_topology_matrix.png"
            circle_path = method_dir / f"{stem}_weighted_circle_graph.png"

            plotter.plot_weighted_topology_matrices(
                layers,
                matrix_path,
                title=f"{title_prefix} Weighted Topology Matrices",
                subtitle=subtitle,
            )
            plotter.plot_circle_graph(
                layers,
                circle_path,
                title=f"{title_prefix} Circle Topology Graph",
                subtitle=subtitle,
            )

            print(
                f"[plot] method={method:14s} target={target:4d}  "
                f"ebops={measured_ebops:7.1f}  circle={circle_path}"
            )
            summary_rows.append(
                {
                    "target_ebops": int(target),
                    "method": method,
                    "measured_ebops": float(measured_ebops),
                    "overall_score": float(score_metrics["overall_score"]),
                    "connectivity_score": float(score_metrics["connectivity_score"]),
                    "trainability_score": float(score_metrics["trainability_score"]),
                    "quant_deadzone_score": float(score_metrics["quant_deadzone_score"]),
                    "disconnected_output_ratio": float(plot_metrics["disconnected_output_ratio"]),
                    "mean_output_degree": float(plot_metrics["mean_output_degree"]),
                    "bridge_in_only_total": float(plot_metrics["bridge_in_only_total"]),
                    "bridge_out_only_total": float(plot_metrics["bridge_out_only_total"]),
                    "qdead_active_ratio": float(plot_metrics["qdead_active_ratio"]),
                    "dead_ratio_kernel": float(plot_metrics["dead_ratio_kernel"]),
                    "trainable": int(bool(plot_metrics["trainable"])),
                    "trainability_verdict": str(plot_metrics["verdict"]),
                    "grad_global_norm": float(plot_metrics["grad_global_norm"]),
                    "grad_near_zero_ratio": float(plot_metrics["grad_near_zero_ratio"]),
                    "grad_first_last_ratio": float(plot_metrics["grad_first_last_ratio"]),
                    "one_step_loss_drop": float(plot_metrics["one_step_loss_drop"]),
                    "probe_start_loss": float(plot_metrics["probe_start_loss"]),
                    "probe_end_loss": float(plot_metrics["probe_end_loss"]),
                    "probe_best_drop": float(plot_metrics["probe_best_drop"]),
                    "probe_rel_best_drop": float(plot_metrics["probe_rel_best_drop"]),
                    "probe_holdout_start": float(plot_metrics["probe_holdout_start"]),
                    "probe_holdout_end": float(plot_metrics["probe_holdout_end"]),
                    "probe_holdout_drop": float(plot_metrics["probe_holdout_drop"]),
                    "probe_holdout_rel_drop": float(plot_metrics["probe_holdout_rel_drop"]),
                    "probe_finite_steps": float(plot_metrics["probe_finite_steps"]),
                    "probe_nonzero_grad_steps": float(plot_metrics["probe_nonzero_grad_steps"]),
                    "probe_lr": float(plot_metrics["probe_lr"]),
                    "last_layer_mean_degree": float(plot_metrics["last_layer_mean_degree"]),
                    "fully_disconnected_layers": float(plot_metrics["fully_disconnected_layers"]),
                    "spectral_score": float(plot_metrics["spectral_score"]),
                    "spectral_bottleneck_score": float(plot_metrics["spectral_bottleneck_score"]),
                    "spectral_mean_score": float(plot_metrics["spectral_mean_score"]),
                    "spectral_last_score": float(plot_metrics["spectral_last_score"]),
                    "spectral_effective_rank_mean": float(plot_metrics["spectral_effective_rank_mean"]),
                    "spectral_stable_rank_mean": float(plot_metrics["spectral_stable_rank_mean"]),
                    "spectral_condition_ratio_mean": float(plot_metrics["spectral_condition_ratio_mean"]),
                    "spectral_rank_ratio_mean": float(plot_metrics["spectral_rank_ratio_mean"]),
                    "spectral_regime": str(plot_metrics["spectral_regime"]),
                    "opt_trainable": int(bool(plot_metrics["opt_trainable"])),
                    "repr_trainable": int(bool(plot_metrics["repr_trainable"])),
                    "opt_verdict": str(plot_metrics["opt_verdict"]),
                    "repr_verdict": str(plot_metrics["repr_verdict"]),
                    "weights_path": str(weights_path),
                    "matrix_path": str(matrix_path),
                    "circle_path": str(circle_path),
                }
            )

    summary_path = plots_root / "summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_ebops",
                "method",
                "measured_ebops",
                "overall_score",
                "connectivity_score",
                "trainability_score",
                "quant_deadzone_score",
                "disconnected_output_ratio",
                "mean_output_degree",
                "bridge_in_only_total",
                "bridge_out_only_total",
                "qdead_active_ratio",
                "dead_ratio_kernel",
                "trainable",
                "trainability_verdict",
                "grad_global_norm",
                "grad_near_zero_ratio",
                "grad_first_last_ratio",
                "one_step_loss_drop",
                "probe_start_loss",
                "probe_end_loss",
                "probe_best_drop",
                "probe_rel_best_drop",
                "probe_holdout_start",
                "probe_holdout_end",
                "probe_holdout_drop",
                "probe_holdout_rel_drop",
                "probe_finite_steps",
                "probe_nonzero_grad_steps",
                "probe_lr",
                "last_layer_mean_degree",
                "fully_disconnected_layers",
                "spectral_score",
                "spectral_bottleneck_score",
                "spectral_mean_score",
                "spectral_last_score",
                "spectral_effective_rank_mean",
                "spectral_stable_rank_mean",
                "spectral_condition_ratio_mean",
                "spectral_rank_ratio_mean",
                "spectral_regime",
                "opt_trainable",
                "repr_trainable",
                "opt_verdict",
                "repr_verdict",
                "weights_path",
                "matrix_path",
                "circle_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
