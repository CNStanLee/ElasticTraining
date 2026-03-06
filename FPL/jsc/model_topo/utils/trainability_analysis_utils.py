from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .topology_analysis_utils import pick_gradual_ckpt_for_target, pick_meta_for_target


@dataclass
class TrainabilityRecord:
    target_ebops: int
    method: str
    measured_ebops: float
    val_loss: float
    val_acc: float
    grad_global_norm: float
    grad_near_zero_ratio: float
    grad_first_last_ratio: float
    grad_log_std: float
    grad_batch_loss: float
    one_step_loss_drop: float
    trainable: bool
    verdict_reason: str
    source_path: str


def _to_float(x) -> float:
    try:
        return float(x.numpy())
    except Exception:
        return float(x)


def _compute_model_ebops(model, sample_input) -> float:
    from keras import ops

    model(sample_input, training=True)
    total = 0.0
    for layer in model._flatten_layers():
        if getattr(layer, 'enable_ebops', False) and getattr(layer, '_ebops', None) is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


def _quick_eval(model, x, y) -> tuple[float, float]:
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    logits = model(x, training=False)
    loss = _to_float(loss_fn(y, logits))
    pred = tf.argmax(logits, axis=-1)
    acc = _to_float(tf.reduce_mean(tf.cast(tf.equal(pred, tf.cast(y, pred.dtype)), tf.float32)))
    return float(loss), float(acc)


def _grad_stats(model, x_grad, y_grad) -> dict:
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
        if 'kernel' in v.name:
            kernel_grad_norms.append(float(np.linalg.norm(g_np)))

    first_last = 0.0
    log_std = 0.0
    if kernel_grad_norms:
        first_last = float(kernel_grad_norms[-1] / (kernel_grad_norms[0] + 1e-12))
        log_std = float(np.std(np.log(np.array(kernel_grad_norms) + 1e-12)))

    return {
        'loss': float(_to_float(loss)),
        'grads': grads,
        'grad_global_norm': float(np.sqrt(g_sumsq)),
        'grad_near_zero_ratio': float(g_near_zero / max(g_total, 1)),
        'grad_first_last_ratio': float(first_last),
        'grad_log_std': float(log_std),
    }


def _one_step_loss_drop(model, x_grad, y_grad, lr: float) -> float:
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = keras.optimizers.SGD(learning_rate=float(lr), momentum=0.0)

    with tf.GradientTape() as tape:
        logits = model(x_grad, training=True)
        pre_loss = loss_fn(y_grad, logits)
    grads = tape.gradient(pre_loss, model.trainable_variables)
    grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    if grads_vars:
        opt.apply_gradients(grads_vars)

    post_logits = model(x_grad, training=True)
    post_loss = loss_fn(y_grad, post_logits)
    return float(_to_float(pre_loss) - _to_float(post_loss))


def _judge_trainable(grad_norm: float, near_zero: float, first_last: float, one_step_drop: float) -> tuple[bool, str]:
    finite = np.isfinite(grad_norm) and (grad_norm > 1e-10)
    dense_enough = near_zero < 0.995
    balanced = (first_last > 1e-5) and (first_last < 1e5) if np.isfinite(first_last) else False
    one_step_ok = one_step_drop > -5e-2

    if finite and dense_enough and balanced and one_step_ok:
        return True, 'ok: finite gradients + non-flat + depth-balanced'
    reasons = []
    if not finite:
        reasons.append('bad_grad_norm')
    if not dense_enough:
        reasons.append('too_many_zero_grads')
    if not balanced:
        reasons.append('depth_imbalance')
    if not one_step_ok:
        reasons.append('one_step_degrades_too_much')
    return False, ','.join(reasons)


def load_model_for_method(
    method: str,
    target: int,
    base_model_path: Path,
    baseline_dir: Path,
    pruned_root: Path,
):
    if method == 'gradual':
        info = pick_gradual_ckpt_for_target(baseline_dir=baseline_dir, target=target)
        model = keras.models.load_model(info.weights_path, compile=False)
        return model, info.measured_ebops, str(info.weights_path)

    info = pick_meta_for_target(pruned_root=pruned_root, method=method, target=target)
    model = keras.models.load_model(base_model_path, compile=False)
    model.load_weights(info.weights_path)
    return model, info.measured_ebops, str(info.weights_path)


def evaluate_trainability(
    model,
    target: int,
    method: str,
    measured_ebops_hint: float,
    source_path: str,
    x_eval,
    y_eval,
    x_grad,
    y_grad,
    one_step_lr: float,
) -> TrainabilityRecord:
    measured = _compute_model_ebops(model, x_eval)
    if measured_ebops_hint > 0:
        measured = float(measured)
    val_loss, val_acc = _quick_eval(model, x_eval, y_eval)
    gs = _grad_stats(model, x_grad, y_grad)
    one_drop = _one_step_loss_drop(model, x_grad, y_grad, lr=one_step_lr)
    trainable, reason = _judge_trainable(
        grad_norm=gs['grad_global_norm'],
        near_zero=gs['grad_near_zero_ratio'],
        first_last=gs['grad_first_last_ratio'],
        one_step_drop=one_drop,
    )

    return TrainabilityRecord(
        target_ebops=int(target),
        method=method,
        measured_ebops=float(measured),
        val_loss=float(val_loss),
        val_acc=float(val_acc),
        grad_global_norm=float(gs['grad_global_norm']),
        grad_near_zero_ratio=float(gs['grad_near_zero_ratio']),
        grad_first_last_ratio=float(gs['grad_first_last_ratio']),
        grad_log_std=float(gs['grad_log_std']),
        grad_batch_loss=float(gs['loss']),
        one_step_loss_drop=float(one_drop),
        trainable=bool(trainable),
        verdict_reason=reason,
        source_path=source_path,
    )
