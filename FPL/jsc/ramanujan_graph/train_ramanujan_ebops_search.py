#!/usr/bin/env python3
"""
Train a Ramanujan-initialized JSC model and search best val_acc near a target EBOPs.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


THIS_DIR = Path(__file__).resolve().parent
JSC_ROOT = THIS_DIR.parent
if str(JSC_ROOT) not in sys.path:
    sys.path.insert(0, str(JSC_ROOT))

# Ensure custom layers are registered for model loading.
import model.model  # noqa: F401
from data.data import get_data
from hgq.layers import QLayerBase
from hgq.utils.sugar import FreeEBOPs
from utils.ramanujan_budget_utils import _flatten_layers, BetaOnlyBudgetController
from utils.train_utils import BudgetAwareEarlyStopping


def parse_list(s: str, cast=float):
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(cast(x))
    return vals


def _forward_update_ebops_no_bn_drift(model: keras.Model, sample_input: tf.Tensor):
    bn_layers = []
    old_momentum = []
    for layer in _flatten_layers(model):
        if isinstance(layer, keras.layers.BatchNormalization):
            bn_layers.append(layer)
            old_momentum.append(float(layer.momentum))
            layer.momentum = 1.0
    try:
        model(sample_input, training=True)
    finally:
        for layer, m in zip(bn_layers, old_momentum):
            layer.momentum = m


def compute_model_ebops(model: keras.Model, sample_input: tf.Tensor) -> float:
    from keras import ops

    _forward_update_ebops_no_bn_drift(model, sample_input)
    total = 0.0
    for layer in _flatten_layers(model):
        if isinstance(layer, QLayerBase) and getattr(layer, "enable_ebops", False) and layer._ebops is not None:
            total += int(ops.convert_to_numpy(layer._ebops))
    return float(total)


@dataclass
class EpochRec:
    epoch: int
    acc: float
    val_acc: float
    loss: float
    val_loss: float
    ebops_measured: float
    ebops_log: float


class WindowTracker(keras.callbacks.Callback):
    def __init__(
        self,
        sample_input: tf.Tensor,
        window_lo: float,
        window_hi: float,
        best_window_ckpt: Path,
        print_every: int = 25,
    ):
        super().__init__()
        self.sample_input = sample_input
        self.window_lo = float(window_lo)
        self.window_hi = float(window_hi)
        self.best_window_ckpt = best_window_ckpt
        self.print_every = int(max(1, print_every))
        self.records: list[EpochRec] = []
        self.best_window = None
        self.best_overall = None

    def _get_metric(self, logs, k1: str, k2: str):
        v = logs.get(k1, logs.get(k2, np.nan))
        try:
            return float(v)
        except Exception:
            return float("nan")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ebops = compute_model_ebops(self.model, self.sample_input)
        logs["ebops_measured"] = ebops

        rec = EpochRec(
            epoch=int(epoch + 1),
            acc=self._get_metric(logs, "acc", "accuracy"),
            val_acc=self._get_metric(logs, "val_acc", "val_accuracy"),
            loss=self._get_metric(logs, "loss", "loss"),
            val_loss=self._get_metric(logs, "val_loss", "val_loss"),
            ebops_measured=float(ebops),
            ebops_log=float(logs.get("ebops", np.nan)),
        )
        self.records.append(rec)

        if self.best_overall is None or rec.val_acc > self.best_overall.val_acc:
            self.best_overall = rec

        in_window = self.window_lo <= rec.ebops_measured <= self.window_hi
        if in_window and (self.best_window is None or rec.val_acc > self.best_window.val_acc):
            self.best_window = rec
            self.best_window_ckpt.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_window_ckpt)

        if rec.epoch % self.print_every == 0 or rec.epoch == 1:
            print(
                f"[epoch {rec.epoch:4d}] val_acc={rec.val_acc:.4f} "
                f"ebops={rec.ebops_measured:.1f} (window={self.window_lo:.0f}~{self.window_hi:.0f})"
            )


def set_all_beta(model: keras.Model, beta_value: float):
    bv = tf.constant(beta_value, dtype=tf.float32)
    for layer in _flatten_layers(model):
        if hasattr(layer, "_beta"):
            try:
                layer._beta.assign(tf.ones_like(layer._beta) * bv)
            except Exception:
                pass


class FixedBetaCallback(keras.callbacks.Callback):
    def __init__(self, beta_value: float):
        super().__init__()
        self.beta_value = float(beta_value)

    def on_train_begin(self, logs=None):
        set_all_beta(self.model, self.beta_value)

    def on_epoch_begin(self, epoch, logs=None):
        set_all_beta(self.model, self.beta_value)


def set_quantizers_trainable(model: keras.Model, trainable: bool):
    """Freeze/unfreeze quantizer variables while keeping kernels/BN trainable."""
    q_names = {"f", "b", "i", "k"}
    for v in model.variables:
        leaf = str(v.name).split(":")[0]
        if leaf in q_names:
            # Keras/TF variables expose `trainable` as read-only in some builds;
            # `_trainable` is the reliable internal switch before compile().
            v._trainable = bool(trainable)


def run_one(
    model_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    sample_input: tf.Tensor,
    out_dir: Path,
    run_name: str,
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
    target_ebops: float,
    window_ratio: float,
    beta_init: float,
    beta_min: float,
    beta_max: float,
    margin: float,
    adjust_factor: float,
    ema_alpha: float,
    earlystop_patience: int,
    earlystop_min_epoch: int,
    steps_per_execution: int,
    budget_mode: str,
    freeze_quantizers: bool,
    fixed_beta: float,
):
    tf.keras.utils.set_random_seed(int(seed))
    np.random.seed(int(seed))

    model = keras.models.load_model(model_path, compile=False)
    if freeze_quantizers:
        set_quantizers_trainable(model, trainable=False)
    set_all_beta(model, float(fixed_beta))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(lr)),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
        steps_per_execution=int(max(1, steps_per_execution)),
    )

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    window_lo = float(target_ebops) * (1.0 - float(window_ratio))
    window_hi = float(target_ebops) * (1.0 + float(window_ratio))

    tracker = WindowTracker(
        sample_input=sample_input,
        window_lo=window_lo,
        window_hi=window_hi,
        best_window_ckpt=run_dir / "best_window.keras",
        print_every=25,
    )

    callbacks = [FreeEBOPs(), FixedBetaCallback(beta_value=float(fixed_beta))]
    if budget_mode == "beta":
        callbacks.append(
            BetaOnlyBudgetController(
                target_ebops=float(target_ebops),
                margin=float(margin),
                beta_init=float(beta_init),
                beta_min=float(beta_min),
                beta_max=float(beta_max),
                adjust_factor=float(adjust_factor),
                ema_alpha=float(ema_alpha),
            )
        )
    callbacks.extend(
        [
            BudgetAwareEarlyStopping(
            ebops_budget=window_hi,
            patience=int(earlystop_patience),
            min_delta=1e-4,
            min_epoch=int(earlystop_min_epoch),
            restore_best_weights=True,
        ),
        tracker,
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        ]
    )

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=callbacks,
        verbose=0,
    )

    model.save(run_dir / "last.keras")

    recs = tracker.records
    final_rec = recs[-1] if recs else None
    best_overall = tracker.best_overall
    best_window = tracker.best_window

    summary = {
        "run_name": run_name,
        "seed": int(seed),
        "lr": float(lr),
        "epochs_ran": int(len(hist.history.get("loss", []))),
        "best_overall": None
        if best_overall is None
        else {
            "epoch": best_overall.epoch,
            "val_acc": best_overall.val_acc,
            "ebops_measured": best_overall.ebops_measured,
        },
        "best_in_window": None
        if best_window is None
        else {
            "epoch": best_window.epoch,
            "val_acc": best_window.val_acc,
            "ebops_measured": best_window.ebops_measured,
        },
        "final": None
        if final_rec is None
        else {
            "epoch": final_rec.epoch,
            "val_acc": final_rec.val_acc,
            "ebops_measured": final_rec.ebops_measured,
        },
        "history_csv": str((run_dir / "history.csv").resolve()),
        "best_window_ckpt": str((run_dir / "best_window.keras").resolve()),
        "last_ckpt": str((run_dir / "last.keras").resolve()),
        "budget_mode": budget_mode,
        "freeze_quantizers": bool(freeze_quantizers),
        "fixed_beta": float(fixed_beta),
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(run_dir / "epoch_records.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "acc", "val_acc", "loss", "val_loss", "ebops_measured", "ebops_log"])
        for r in recs:
            w.writerow([r.epoch, r.acc, r.val_acc, r.loss, r.val_loss, r.ebops_measured, r.ebops_log])

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train Ramanujan-init model and search best val_acc around target EBOPs."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ramanujan_graph/ramanujan_graph/output_bit02/jsc_ramanujan_importance_init.keras",
    )
    parser.add_argument("--input_h5", type=str, default="data/dataset.h5")
    parser.add_argument("--out_dir", type=str, default="ramanujan_graph/train_search_2800")
    parser.add_argument("--target_ebops", type=float, default=2800.0)
    parser.add_argument("--ebops_window_ratio", type=float, default=0.08)
    parser.add_argument("--seeds", type=str, default="42,123,2026")
    parser.add_argument("--lrs", type=str, default="0.001,0.0015,0.0007")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=33200)
    parser.add_argument("--sample_size", type=int, default=512)
    parser.add_argument("--beta_init", type=float, default=1e-5)
    parser.add_argument("--beta_min", type=float, default=1e-7)
    parser.add_argument("--beta_max", type=float, default=8e-4)
    parser.add_argument("--budget_margin", type=float, default=0.05)
    parser.add_argument("--beta_adjust_factor", type=float, default=1.2)
    parser.add_argument("--beta_ema_alpha", type=float, default=0.25)
    parser.add_argument("--earlystop_patience", type=int, default=160)
    parser.add_argument("--earlystop_min_epoch", type=int, default=120)
    parser.add_argument("--steps_per_execution", type=int, default=32)
    parser.add_argument("--budget_mode", type=str, default="none", choices=["none", "beta"])
    parser.add_argument("--fixed_beta", type=float, default=0.0)
    parser.add_argument("--freeze_quantizers", action="store_true", default=True)
    parser.add_argument("--no_freeze_quantizers", action="store_true")
    parser.add_argument("--max_runs", type=int, default=0, help="0 means run all combinations")
    args = parser.parse_args()
    if args.no_freeze_quantizers:
        args.freeze_quantizers = False

    os.chdir(JSC_ROOT)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (JSC_ROOT / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (JSC_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[DATA] loading dataset...")
    (x_train, y_train), (x_val, y_val), _ = get_data(args.input_h5, src="openml")
    sample_n = int(min(max(1, args.sample_size), x_val.shape[0]))
    sample_input = tf.constant(x_val[:sample_n], dtype=tf.float32)
    print(
        f"[DATA] train={x_train.shape} val={x_val.shape} "
        f"sample_for_ebops={tuple(sample_input.shape)}"
    )

    seeds = parse_list(args.seeds, cast=int)
    lrs = parse_list(args.lrs, cast=float)
    runs = [(s, lr) for s in seeds for lr in lrs]
    if args.max_runs > 0:
        runs = runs[: int(args.max_runs)]

    print(
        f"[SEARCH] target_ebops={args.target_ebops:.1f} "
        f"window=±{args.ebops_window_ratio*100:.1f}% runs={len(runs)}"
    )
    print(f"[SEARCH] seeds={seeds} lrs={lrs}")

    run_summaries = []
    for i, (seed, lr) in enumerate(runs, start=1):
        run_name = f"run{i:02d}_seed{seed}_lr{lr:g}"
        print("=" * 72)
        print(f"[RUN {i}/{len(runs)}] {run_name}")
        summ = run_one(
            model_path=model_path,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            sample_input=sample_input,
            out_dir=out_dir,
            run_name=run_name,
            seed=seed,
            lr=lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            target_ebops=args.target_ebops,
            window_ratio=args.ebops_window_ratio,
            beta_init=args.beta_init,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            margin=args.budget_margin,
            adjust_factor=args.beta_adjust_factor,
            ema_alpha=args.beta_ema_alpha,
            earlystop_patience=args.earlystop_patience,
            earlystop_min_epoch=args.earlystop_min_epoch,
            steps_per_execution=args.steps_per_execution,
            budget_mode=args.budget_mode,
            freeze_quantizers=args.freeze_quantizers,
            fixed_beta=args.fixed_beta,
        )
        run_summaries.append(summ)

        b = summ.get("best_in_window")
        if b is None:
            print("[RESULT] no epoch reached the EBOPs window.")
        else:
            print(
                f"[RESULT] best_in_window val_acc={b['val_acc']:.4f} "
                f"ebops={b['ebops_measured']:.1f} epoch={b['epoch']}"
            )

    def key_fn(x):
        bw = x.get("best_in_window")
        return -1.0 if bw is None else float(bw["val_acc"])

    best = None
    if run_summaries:
        best = sorted(run_summaries, key=key_fn, reverse=True)[0]

    global_summary = {
        "target_ebops": float(args.target_ebops),
        "ebops_window_ratio": float(args.ebops_window_ratio),
        "model_path": str(model_path),
        "runs": run_summaries,
        "best_run_near_target": best,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_name",
                "seed",
                "lr",
                "best_window_val_acc",
                "best_window_ebops",
                "best_window_epoch",
                "best_overall_val_acc",
                "best_overall_ebops",
                "final_val_acc",
                "final_ebops",
            ]
        )
        for r in run_summaries:
            bw = r.get("best_in_window") or {}
            bo = r.get("best_overall") or {}
            ff = r.get("final") or {}
            w.writerow(
                [
                    r.get("run_name"),
                    r.get("seed"),
                    r.get("lr"),
                    bw.get("val_acc"),
                    bw.get("ebops_measured"),
                    bw.get("epoch"),
                    bo.get("val_acc"),
                    bo.get("ebops_measured"),
                    ff.get("val_acc"),
                    ff.get("ebops_measured"),
                ]
            )

    print("=" * 72)
    print(f"[DONE] summary json: {(out_dir / 'summary.json').resolve()}")
    print(f"[DONE] summary csv : {(out_dir / 'summary.csv').resolve()}")
    if best is not None and best.get("best_in_window") is not None:
        b = best["best_in_window"]
        print(
            f"[BEST near target] run={best['run_name']} "
            f"val_acc={b['val_acc']:.4f} ebops={b['ebops_measured']:.1f} epoch={b['epoch']}"
        )
    else:
        print("[BEST near target] no run found in the requested EBOPs window.")


if __name__ == "__main__":
    main()
