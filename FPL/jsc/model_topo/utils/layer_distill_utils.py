from __future__ import annotations

from typing import Dict, Iterable, List

import keras
import numpy as np


def dense_prunable_layers(model) -> List:
    layers = []
    for layer in model._flatten_layers() if hasattr(model, "_flatten_layers") else model.layers:
        if not hasattr(layer, "kernel") or getattr(layer, "kq", None) is None:
            continue
        if len(layer.kernel.shape) != 2:
            continue
        layers.append(layer)
    return layers


def layer_active_mask(layer, get_q_var) -> np.ndarray:
    from keras import ops

    try:
        bits = layer.kq.bits_(ops.shape(layer.kernel))
        bits_np = np.array(ops.convert_to_numpy(bits), dtype=np.float32)
        return bits_np > 1e-6
    except Exception:
        b_var = get_q_var(layer.kq, "b")
        if b_var is None:
            b_var = get_q_var(layer.kq, "f")
        if b_var is None:
            return np.zeros(tuple(layer.kernel.shape), dtype=bool)
        b_np = np.array(b_var.numpy(), dtype=np.float32)
        k_var = get_q_var(layer.kq, "k")
        if k_var is None:
            return b_np > 0.0
        k_np = np.array(k_var.numpy(), dtype=np.float32)
        return (b_np > 0.0) | (k_np > 0.0)


def collect_named_outputs(model, layer_names: Iterable[str], x, training: bool = False) -> Dict[str, np.ndarray]:
    names = list(layer_names)
    outputs = [model.get_layer(n).output for n in names]
    probe = keras.Model(model.inputs, outputs)
    vals = probe(x, training=training)
    if not isinstance(vals, (list, tuple)):
        vals = [vals]
    ret = {}
    for n, v in zip(names, vals):
        ret[n] = np.array(v.numpy(), dtype=np.float32)
    return ret


def reestimate_quantizer_ranges(
    model,
    get_q_var,
    b_floor: float = 0.35,
    b_ceiling: float = 6.0,
    i_min: float = -2.0,
    i_max: float = 6.0,
) -> None:
    for layer in dense_prunable_layers(model):
        kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
        active = layer_active_mask(layer, get_q_var)

        b_var = get_q_var(layer.kq, "b")
        if b_var is None:
            b_var = get_q_var(layer.kq, "f")
        i_var = get_q_var(layer.kq, "i")
        k_var = get_q_var(layer.kq, "k")

        if b_var is not None:
            b_old = np.array(b_var.numpy(), dtype=np.float32)
            b_new = np.where(active, np.clip(np.maximum(b_old, b_floor), b_floor, b_ceiling), 0.0)
            b_var.assign(b_new.astype(np.float32))

        if i_var is not None:
            i_old = np.array(i_var.numpy(), dtype=np.float32)
            i_req = np.ceil(np.log2(np.maximum(np.abs(kernel), 1e-8))).astype(np.float32) + 1.0
            i_new = np.where(active, np.clip(np.maximum(i_old, i_req), i_min, i_max), -16.0)
            i_var.assign(i_new.astype(np.float32))

        if k_var is not None:
            k_var.assign(np.where(active, 1.0, 0.0).astype(np.float32))

        cols_active = np.sum(active, axis=0) > 0
        bq = getattr(layer, "bq", None)
        if bq is None:
            continue

        bb_var = get_q_var(bq, "b")
        if bb_var is None:
            bb_var = get_q_var(bq, "f")
        bi_var = get_q_var(bq, "i")
        bk_var = get_q_var(bq, "k")

        if bb_var is not None:
            bb_old = np.array(bb_var.numpy(), dtype=np.float32)
            bb_new = np.where(cols_active, np.clip(np.maximum(bb_old, b_floor), b_floor, b_ceiling), 0.0)
            bb_var.assign(bb_new.astype(np.float32))

        if bi_var is not None:
            bi_old = np.array(bi_var.numpy(), dtype=np.float32)
            bi_new = np.where(cols_active, np.clip(bi_old, i_min, i_max), -16.0)
            bi_var.assign(bi_new.astype(np.float32))

        if bk_var is not None:
            bk_var.assign(np.where(cols_active, 1.0, 0.0).astype(np.float32))


def teacher_guided_layer_distill(
    student_model,
    teacher_model,
    sample_input,
    get_q_var,
    passes: int = 2,
    b_floor: float = 0.35,
    b_ceiling: float = 6.0,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
    shift_clip: float = 8.0,
) -> float:
    s_layers = dense_prunable_layers(student_model)
    if not s_layers:
        return 0.0

    layer_names = []
    for layer in s_layers:
        try:
            teacher_model.get_layer(layer.name)
            layer_names.append(layer.name)
        except Exception:
            continue
    if not layer_names:
        return 0.0

    mean_change = 0.0
    for _ in range(max(1, int(passes))):
        y_teacher = collect_named_outputs(teacher_model, layer_names, sample_input, training=False)
        y_student = collect_named_outputs(student_model, layer_names, sample_input, training=False)
        changes = []
        for name in layer_names:
            layer = student_model.get_layer(name)
            ys = y_student[name]
            yt = y_teacher[name]
            if ys.shape != yt.shape:
                continue

            c = ys.shape[-1]
            ys2 = ys.reshape(-1, c)
            yt2 = yt.reshape(-1, c)
            if ys2.shape[0] < 2:
                continue

            active = layer_active_mask(layer, get_q_var)
            active_cols = np.sum(active, axis=0) > 0
            if not np.any(active_cols):
                continue

            mu_s = np.mean(ys2, axis=0)
            mu_t = np.mean(yt2, axis=0)
            zs = ys2 - mu_s
            zt = yt2 - mu_t
            var_s = np.mean(zs * zs, axis=0)
            cov_st = np.mean(zs * zt, axis=0)

            scale = cov_st / (var_s + 1e-8)
            scale = np.where(np.isfinite(scale), scale, 1.0).astype(np.float32)
            scale = np.clip(scale, scale_min, scale_max)
            shift = np.clip(mu_t - scale * mu_s, -shift_clip, shift_clip).astype(np.float32)

            k = np.array(layer.kernel.numpy(), dtype=np.float32)
            k_new = k.copy()
            k_new[:, active_cols] = k[:, active_cols] * scale[active_cols][None, :]
            k_new[:, ~active_cols] = 0.0
            layer.kernel.assign(k_new.astype(np.float32))

            b = getattr(layer, "bias", None)
            if b is not None:
                b_old = np.array(b.numpy(), dtype=np.float32)
                b_new = b_old.copy()
                b_new[active_cols] = b_old[active_cols] * scale[active_cols] + shift[active_cols]
                b_new[~active_cols] = 0.0
                b.assign(b_new.astype(np.float32))
            changes.append(float(np.mean(np.abs(scale[active_cols] - 1.0))))

        reestimate_quantizer_ranges(
            student_model,
            get_q_var=get_q_var,
            b_floor=b_floor,
            b_ceiling=b_ceiling,
        )
        if changes:
            mean_change = float(np.mean(changes))
    return mean_change
