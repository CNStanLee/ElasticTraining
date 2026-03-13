#!/usr/bin/env python3
"""Aggregate experiment C and G results into plotting-friendly tables."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

BASE_DIR = Path("FPL2/jsc/results")
G_FILE = BASE_DIR / "experiment_G_restart_decay" / "all_results_by_target.json"
C_FILE = BASE_DIR / "experiment_C_pruning_methods" / "all_results.json"
OUT_DIR = BASE_DIR / "experiment_GC_merged"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: str


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def _derived_fields(row: dict[str, Any]) -> dict[str, Any]:
    target = _to_float(row.get("target_ebops"))
    final_ebops = _to_float(row.get("final_ebops"))
    best_ebops_target = _to_float(row.get("best_ebops_at_target"))
    pretrained = _to_float(row.get("pretrained_ebops"))
    pruned = _to_float(row.get("pruned_ebops"))
    phase1_ebops = _to_float(row.get("phase1_ebops"))
    elapsed_sec = _to_float(row.get("elapsed_sec"))

    final_val_acc = _to_float(row.get("final_val_acc"))
    phase1_val_acc = _to_float(row.get("phase1_val_acc"))
    best_val_acc = _to_float(row.get("best_val_acc"))

    final_minus_target = None if final_ebops is None or target is None else final_ebops - target
    best_minus_target = (
        None if best_ebops_target is None or target is None else best_ebops_target - target
    )
    final_acc_gain_phase1 = (
        None if final_val_acc is None or phase1_val_acc is None else final_val_acc - phase1_val_acc
    )
    best_acc_gain_phase1 = (
        None if best_val_acc is None or phase1_val_acc is None else best_val_acc - phase1_val_acc
    )

    hours = _safe_div(elapsed_sec, 3600.0)

    return {
        "family": row.get("family"),
        "variant_key": row.get("variant_key"),
        "variant_value": row.get("variant_value"),
        "is_target_met": final_ebops is not None and target is not None and final_ebops <= target,
        "final_minus_target": final_minus_target,
        "final_minus_target_abs": None if final_minus_target is None else abs(final_minus_target),
        "final_target_error_pct": _safe_div(final_minus_target, target),
        "final_target_error_pct_abs": None
        if final_minus_target is None or target in (None, 0)
        else abs(final_minus_target) / target,
        "best_minus_target": best_minus_target,
        "best_target_error_pct": _safe_div(best_minus_target, target),
        "final_over_target": _safe_div(final_ebops, target),
        "best_over_target": _safe_div(best_ebops_target, target),
        "pretrained_to_final_reduction_pct": (
            None if pretrained in (None, 0) or final_ebops is None else 1.0 - final_ebops / pretrained
        ),
        "pruned_to_final_reduction_pct": (
            None if pruned in (None, 0) or final_ebops is None else 1.0 - final_ebops / pruned
        ),
        "pretrained_to_pruned_reduction_pct": (
            None if pretrained in (None, 0) or pruned is None else 1.0 - pruned / pretrained
        ),
        "phase1_to_final_ebops_delta": (
            None if final_ebops is None or phase1_ebops is None else final_ebops - phase1_ebops
        ),
        "final_acc_gain_vs_phase1": final_acc_gain_phase1,
        "best_acc_gain_vs_phase1": best_acc_gain_phase1,
        "best_minus_final_acc": (
            None if best_val_acc is None or final_val_acc is None else best_val_acc - final_val_acc
        ),
        "final_acc_gain_per_hour": _safe_div(final_acc_gain_phase1, hours),
        "best_acc_gain_per_hour": _safe_div(best_acc_gain_phase1, hours),
        "best_acc_per_hour": _safe_div(best_val_acc, hours),
    }


def _rank(rows: list[dict[str, Any]], key_name: str, new_col: str, reverse: bool) -> None:
    sortable: list[tuple[int, float]] = []
    for i, row in enumerate(rows):
        val = _to_float(row.get(key_name))
        if val is not None:
            sortable.append((i, val))

    sortable.sort(key=lambda x: x[1], reverse=reverse)
    for rank, (i, _) in enumerate(sortable, start=1):
        rows[i][new_col] = rank


def _load_data() -> list[dict[str, Any]]:
    with G_FILE.open("r", encoding="utf-8") as f:
        g_data = json.load(f)
    with C_FILE.open("r", encoding="utf-8") as f:
        c_data = json.load(f)

    out: list[dict[str, Any]] = []

    for target, runs in g_data.items():
        for run in runs:
            row = dict(run)
            row["family"] = "G"
            row["method_key"] = row.get("method_key") or "restart_decay"
            row["variant_key"] = "restart_decay"
            row["variant_value"] = row.get("restart_decay")
            row["source_file"] = str(G_FILE)
            row["source_target_key"] = target
            out.append(row)

    for run in c_data:
        row = dict(run)
        row["family"] = "C"
        row["method_key"] = row.get("method_key") or "unknown"
        row["variant_key"] = "method_key"
        row["variant_value"] = row.get("method_key")
        row["source_file"] = str(C_FILE)
        row["source_target_key"] = str(row.get("target_ebops"))
        out.append(row)

    for row in out:
        row.update(_derived_fields(row))

    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_long_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        MetricSpec("final_val_acc", "final_val_acc"),
        MetricSpec("best_val_acc", "best_val_acc"),
        MetricSpec("final_val_loss", "final_val_loss"),
        MetricSpec("final_ebops", "final_ebops"),
        MetricSpec("target_ebops", "target_ebops"),
        MetricSpec("best_ebops", "best_ebops"),
        MetricSpec("best_ebops_at_target", "best_ebops_at_target"),
        MetricSpec("final_target_error_pct", "final_target_error_pct"),
        MetricSpec("pretrained_to_final_reduction_pct", "pretrained_to_final_reduction_pct"),
        MetricSpec("pretrained_to_pruned_reduction_pct", "pretrained_to_pruned_reduction_pct"),
        MetricSpec("final_acc_gain_vs_phase1", "final_acc_gain_vs_phase1"),
        MetricSpec("best_minus_final_acc", "best_minus_final_acc"),
        MetricSpec("elapsed_sec", "elapsed_sec"),
        MetricSpec("total_epochs", "total_epochs"),
    ]

    long_rows: list[dict[str, Any]] = []
    for row in rows:
        run_id = f"{row.get('family')}_{row.get('experiment')}"
        for spec in specs:
            val = row.get(spec.source)
            if val is None:
                continue
            long_rows.append(
                {
                    "run_id": run_id,
                    "family": row.get("family"),
                    "target_ebops": row.get("target_ebops"),
                    "experiment": row.get("experiment"),
                    "method_key": row.get("method_key"),
                    "restart_decay": row.get("restart_decay"),
                    "variant_key": row.get("variant_key"),
                    "variant_value": row.get("variant_value"),
                    "metric": spec.name,
                    "value": val,
                    "is_target_met": row.get("is_target_met"),
                }
            )
    return long_rows


def _build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("family"), row.get("target_ebops"), row.get("variant_value"))
        buckets.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (family, target, variant), group in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]))):
        best_vals = [_to_float(r.get("best_val_acc")) for r in group]
        best_vals = [v for v in best_vals if v is not None]
        final_vals = [_to_float(r.get("final_val_acc")) for r in group]
        final_vals = [v for v in final_vals if v is not None]
        err_vals = [_to_float(r.get("final_target_error_pct")) for r in group]
        err_vals = [v for v in err_vals if v is not None]

        elapsed_vals = [_to_float(r.get("elapsed_sec")) for r in group]
        elapsed_vals = [v for v in elapsed_vals if v is not None]

        summary_rows.append(
            {
                "family": family,
                "target_ebops": target,
                "variant_value": variant,
                "runs": len(group),
                "target_met_rate": mean(1.0 if r.get("is_target_met") else 0.0 for r in group),
                "best_val_acc_mean": mean(best_vals) if best_vals else None,
                "best_val_acc_median": median(best_vals) if best_vals else None,
                "final_val_acc_mean": mean(final_vals) if final_vals else None,
                "final_val_acc_median": median(final_vals) if final_vals else None,
                "final_target_error_pct_mean": mean(err_vals) if err_vals else None,
                "final_target_error_pct_median": median(err_vals) if err_vals else None,
                "elapsed_sec_mean": mean(elapsed_vals) if elapsed_vals else None,
            }
        )
    return summary_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = _load_data()

    rows.sort(
        key=lambda r: (
            r.get("family"),
            _to_float(r.get("target_ebops")) or 0,
            str(r.get("variant_value")),
            str(r.get("experiment")),
        )
    )

    # Ranking fields (overall and within each target).
    _rank(rows, "best_val_acc", "rank_best_val_acc_overall", reverse=True)
    _rank(rows, "final_val_acc", "rank_final_val_acc_overall", reverse=True)
    _rank(rows, "final_target_error_pct_abs", "rank_target_error_abs_overall", reverse=False)

    by_target: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(row.get("target_ebops"), []).append(row)
    for target_rows in by_target.values():
        _rank(target_rows, "best_val_acc", "rank_best_val_acc_within_target", reverse=True)
        _rank(target_rows, "final_val_acc", "rank_final_val_acc_within_target", reverse=True)
        _rank(target_rows, "final_target_error_pct_abs", "rank_target_error_within_target", reverse=False)

    flat_fields = sorted({k for row in rows for k in row.keys()})
    _write_csv(OUT_DIR / "gc_all_runs_flat.csv", rows, flat_fields)

    core_fields = [
        "family",
        "target_ebops",
        "experiment",
        "method_key",
        "restart_decay",
        "variant_key",
        "variant_value",
        "best_val_acc",
        "final_val_acc",
        "phase1_val_acc",
        "best_minus_final_acc",
        "final_acc_gain_vs_phase1",
        "best_acc_gain_vs_phase1",
        "final_val_loss",
        "pretrained_ebops",
        "pruned_ebops",
        "phase1_ebops",
        "final_ebops",
        "best_ebops_at_target",
        "final_minus_target",
        "final_minus_target_abs",
        "final_target_error_pct",
        "final_target_error_pct_abs",
        "final_over_target",
        "is_target_met",
        "elapsed_sec",
        "total_epochs",
        "rank_best_val_acc_overall",
        "rank_best_val_acc_within_target",
        "rank_target_error_abs_overall",
        "rank_target_error_within_target",
    ]
    _write_csv(OUT_DIR / "gc_plot_core.csv", rows, core_fields)

    long_rows = _build_long_metrics(rows)
    long_fields = [
        "run_id",
        "family",
        "target_ebops",
        "experiment",
        "method_key",
        "restart_decay",
        "variant_key",
        "variant_value",
        "metric",
        "value",
        "is_target_met",
    ]
    _write_csv(OUT_DIR / "gc_metrics_long.csv", long_rows, long_fields)

    summary_rows = _build_summary(rows)
    summary_fields = [
        "family",
        "target_ebops",
        "variant_value",
        "runs",
        "target_met_rate",
        "best_val_acc_mean",
        "best_val_acc_median",
        "final_val_acc_mean",
        "final_val_acc_median",
        "final_target_error_pct_mean",
        "final_target_error_pct_median",
        "elapsed_sec_mean",
    ]
    _write_csv(OUT_DIR / "gc_summary_by_group.csv", summary_rows, summary_fields)

    metadata = {
        "source_files": [str(G_FILE), str(C_FILE)],
        "total_runs": len(rows),
        "family_counts": {
            "G": sum(1 for r in rows if r.get("family") == "G"),
            "C": sum(1 for r in rows if r.get("family") == "C"),
        },
        "targets": sorted({r.get("target_ebops") for r in rows}),
        "outputs": [
            str(OUT_DIR / "gc_all_runs_flat.csv"),
            str(OUT_DIR / "gc_plot_core.csv"),
            str(OUT_DIR / "gc_metrics_long.csv"),
            str(OUT_DIR / "gc_summary_by_group.csv"),
        ],
    }
    with (OUT_DIR / "README_gc_tables.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
