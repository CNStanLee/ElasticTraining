"""
run_hls_impl.py  –  HLS (Vitis HLS C++) implementation of the JSC model.

Contrast with run_test.py (RTLModel / Verilog):
  ┌───────────────┬────────────────────────────────┬──────────────────────────┐
  │               │ run_test.py                    │ run_hls_impl.py          │
  ├───────────────┼────────────────────────────────┼──────────────────────────┤
  │ Backend       │ RTLModel  → Verilog (.v)       │ HLSModel  → C++ (.cc/.hh)│
  │ Synth tool    │ Vivado (synth + P&R)            │ Vitis HLS (C-synth)      │
  │ SW emulator   │ Verilator (RTL sim)             │ g++ / OpenMP            │
  │ Clock default │ 1.0 ns  (1 GHz)                │ 5.0 ns  (200 MHz)        │
  └───────────────┴────────────────────────────────┴──────────────────────────┘

Usage
-----
  python run_hls_impl.py                  # default model & output dir
  python run_hls_impl.py --no-synth       # skip Vitis HLS synthesis
  python run_hls_impl.py --force-synth    # re-run even if reports exist
  python run_hls_impl.py --cosim          # also run C/RTL co-simulation
  python run_hls_impl.py --flavor vitis   # Vitis HLS (default)
  python run_hls_impl.py --flavor hlslib  # HLSlib (FPGA cross-platform)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

# ─── Vivado / Vitis toolchain paths ──────────────────────────────────────────
_XILINX_ROOT  = '/tools/Xilinx'
_VITIS_HLS_BIN = os.path.join(_XILINX_ROOT, 'Vitis_HLS/2022.2/bin')
_VIVADO_BIN    = os.path.join(_XILINX_ROOT, 'Vivado/2022.2/bin')
for _bin in (_VITIS_HLS_BIN, _VIVADO_BIN):
    if _bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = _bin + os.pathsep + os.environ.get('PATH', '')

os.environ['KERAS_BACKEND']        = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path

import keras
import numpy as np
from da4ml.codegen import HLSModel
from da4ml.converter import trace_model
from da4ml.trace import HWConfig, comb_trace
from hgq.utils import trace_minmax  # registers HGQ custom Keras classes

from data.data import get_data


# ─── Defaults ─────────────────────────────────────────────────────────────────
MODEL_PATH = 'results/baseline/epoch=165789-val_acc=0.712-ebops=304-val_loss=0.888.keras'
DATA_PATH  = 'data/dataset.h5'
OUTPUT_DIR = 'results/hls_impl/epoch=165789'
MODULE_NAME = 'jsc'

# da4ml trace settings (keep identical to run_test.py for reproducibility)
HW_CONFIG      = HWConfig(1, -1, -1)
SOLVER_OPTIONS = {'hard_dc': 2}

# HLS synthesis settings
CLOCK_PERIOD    = 5       # ns  →  200 MHz target (Vitis HLS default)
CLOCK_UNCERTAINTY = 0.5   # ns
FPGA_PART       = 'xc7k160tffg676-1'
HLS_FLAVOR      = 'vitis'  # 'vitis' | 'hlslib' | 'oneapi'


def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.argmax(y_pred, axis=-1) == y_true))


def _load_metadata(path: Path) -> dict:
    p = path / 'metadata.json'
    return json.loads(p.read_text()) if p.exists() else {}


def _save_metadata(path: Path, misc: dict) -> None:
    with open(path / 'metadata.json', 'w') as f:
        json.dump(misc, f, indent=2)


# ─── Step 1 : trace & save ────────────────────────────────────────────────────

def trace_and_save(model: keras.Model, path: Path, *datasets: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds0, rest = datasets[0], datasets[1:]
    trace_minmax(model, ds0, batch_size=25600, reset=True)
    for ds in rest:
        trace_minmax(model, ds, batch_size=25600, reset=False)
    model.save(path)


# ─── Step 2 : convert to HLS C++ ─────────────────────────────────────────────

def build_hls_model(model: keras.Model, out_dir: Path,
                    flavor: str = HLS_FLAVOR) -> tuple['HLSModel', object]:
    """Trace → combinational logic → HLS C++ source."""
    inp, out = trace_model(model, hwconf=HW_CONFIG, solver_options=SOLVER_OPTIONS)
    comb = comb_trace(inp, out)

    hls = HLSModel(
        comb,
        MODULE_NAME,
        out_dir / 'src',
        flavor=flavor,
        part_name=FPGA_PART,
        clock_period=CLOCK_PERIOD,
        clock_uncertainty=CLOCK_UNCERTAINTY,
    )
    hls.write()
    print(f'HLS C++ source written to: {out_dir / "src"}')
    print()
    print(repr(hls))
    print()
    return hls, comb


# ─── Step 3 : SW evaluation ──────────────────────────────────────────────────

def sw_test(model: keras.Model, comb, hls: HLSModel,
            X_test: np.ndarray, y_test: np.ndarray,
            misc: dict) -> tuple[dict, np.ndarray | None]:
    print('\n── SW accuracy ──────────────────────────────────')

    # Keras float prediction
    y_keras   = model.predict(X_test, batch_size=25600, verbose=0)
    keras_acc = _accuracy(y_test, y_keras)
    print(f'  Keras (float) accuracy : {keras_acc:.4f}')

    # Combinational fixed-point (pure Python / C)
    c_pred   = comb.predict(X_test, n_threads=4)
    comb_acc = _accuracy(y_test, c_pred)
    print(f'  Comb  (fixed) accuracy : {comb_acc:.4f}')

    # HLS C++ emulator (compile once, predict)
    hls_pred = None
    print('  Compiling HLS C++ emulator …')
    try:
        for attempt in range(8):
            try:
                hls._compile(openmp=True, o3=True)
                break
            except RuntimeError:
                if attempt == 7:
                    raise
        hls_pred    = hls.predict(np.array(X_test, dtype=np.float32))
        hls_sw_acc  = _accuracy(y_test, hls_pred)
        ndiff       = int(np.sum(hls_pred != c_pred))
        print(f'  HLS emulator accuracy  : {hls_sw_acc:.4f}')
        if ndiff:
            print(f'  HLS/comb mismatch      : {ndiff} / {hls_pred.size}')
        else:
            print('  HLS/comb match         : perfect')
        misc['hls_sw_metric'] = hls_sw_acc
        misc['hls_comb_diff'] = ndiff / hls_pred.size
    except Exception as e:
        print(f'  [warn] HLS C++ emulator failed: {e}')

    misc['keras_metric'] = keras_acc
    misc['comb_metric']  = comb_acc
    return misc, hls_pred


# ─── Step 4 : Vitis HLS synthesis ────────────────────────────────────────────

def _write_vitis_tcl(src_dir: Path, out_dir: Path) -> Path:
    """Generate a Vitis HLS batch TCL script."""
    tcl = f"""# Auto-generated by run_hls_impl.py
open_project -reset prj_{MODULE_NAME}
set_top {MODULE_NAME}_fn
add_files "{src_dir}/{MODULE_NAME}.cc" -cflags "-std=c++14 -I{src_dir} -I{src_dir}/ap_types"
add_files -tb "{src_dir}/{MODULE_NAME}.cc" -cflags "-std=c++14 -I{src_dir} -I{src_dir}/ap_types"

open_solution -reset sol1 -flow_target vivado
set_part {{{FPGA_PART}}}
create_clock -period {CLOCK_PERIOD}ns -name default

csynth_design
export_design -flow impl -rtl verilog -format ip_catalog
"""
    tcl_path = out_dir / '_vitis_hls.tcl'
    tcl_path.write_text(tcl)
    return tcl_path


def run_vitis_synth(out_dir: Path, misc: dict, force: bool = False) -> dict:
    src_dir   = out_dir / 'src'
    rpt_root  = out_dir / f'prj_{MODULE_NAME}' / 'sol1'
    util_rpt  = rpt_root / 'syn' / 'report' / f'{MODULE_NAME}_fn_csynth.rpt'

    if util_rpt.exists() and not force:
        print('\n── Vitis HLS synthesis ──────────────────────────')
        print('  Reports already exist — skipping re-synthesis.')
        print('  (use --force-synth to re-run)')
    else:
        tcl_path = _write_vitis_tcl(src_dir, out_dir)
        log_path = out_dir / 'vitis_hls.log'
        print('\n── Vitis HLS C-synthesis ────────────────────────')
        print(f'  Part   : {FPGA_PART}')
        print(f'  Clock  : {CLOCK_PERIOD} ns  ({1000//CLOCK_PERIOD} MHz)')
        print(f'  Source : {src_dir}')
        print('  Running csynth_design …')
        try:
            with open(log_path, 'w') as logf:
                subprocess.run(
                    ['vitis_hls', '-f', tcl_path.name],
                    cwd=out_dir,
                    check=True,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError:
            errors = [l for l in open(log_path) if 'ERROR' in l or 'error' in l]
            for e in errors[-5:]:
                print(f'  {e.strip()}')
            print(f'  [warn] Vitis HLS failed — see {log_path}')
            return misc

    # ── Parse csynth report ────────────────────────────────────────────────
    if not util_rpt.exists():
        print(f'  [warn] Synthesis report not found: {util_rpt}')
        return misc

    text = util_rpt.read_text()

    # Timing from "Performance Estimates"
    m = re.search(r'Target\s*\|?\s*([\d\.]+)\s*\|\s*([\d\.]+)', text)
    if m:
        try:
            misc['hls_target_ns'] = float(m.group(1))
            misc['hls_worst_ns']  = float(m.group(2))
        except ValueError:
            pass

    # Latency
    m = re.search(r'Latency\s*\(clock cycles\)\s*\|[^|]*\|[^|]*\|\s*(\d+)\s*\|', text)
    if m:
        try:
            misc['hls_latency_cycles'] = int(m.group(1))
        except ValueError:
            pass

    # Resources from "Utilization Estimates"
    for key, pat in [
        ('hls_bram',  r'BRAM_18K\s*\|\s*(\d+)'),
        ('hls_dsp',   r'DSP48E\s*\|\s*(\d+)'),
        ('hls_ff',    r'FF\s*\|\s*(\d+)'),
        ('hls_lut',   r'LUT\s*\|\s*(\d+)'),
        ('hls_uram',  r'URAM\s*\|\s*(\d+)'),
    ]:
        m = re.search(pat, text)
        if m:
            try:
                misc[key] = int(m.group(1))
            except ValueError:
                pass

    return misc


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(hls: HLSModel, misc: dict, model: keras.Model, sw: bool):
    sep = '=' * 55
    print(f'\n{sep}')
    print('  PERFORMANCE SUMMARY  (HLS / Vitis HLS)')
    print(sep)

    sol = hls._solution

    # Accuracy
    print('\n[Accuracy]')
    if sw:
        print(f'  Keras (float)  : {misc.get("keras_metric", "N/A"):.4f}')
        print(f'  Comb  (fixed)  : {misc.get("comb_metric",  "N/A"):.4f}')
        if 'hls_sw_metric' in misc:
            print(f'  HLS C++ emu    : {misc["hls_sw_metric"]:.4f}')
        if 'hls_comb_diff' in misc:
            print(f'  HLS/comb diff  : {misc["hls_comb_diff"]:.4%}')

    # EBOPs
    print('\n[Effective BOPs]')
    print(f'  EBOPs/inference : {misc.get("ebops", "N/A")}')

    # Timing
    print('\n[Timing & Throughput]')
    print(f'  Clock target    : {CLOCK_PERIOD} ns  ({1000//CLOCK_PERIOD} MHz)')
    if 'hls_worst_ns' in misc:
        wns   = CLOCK_PERIOD - misc['hls_worst_ns']
        slack = 'MET' if wns >= 0 else 'VIOLATED'
        fmax  = 1e3 / misc['hls_worst_ns']
        print(f'  Worst-case path : {misc["hls_worst_ns"]:.3f} ns')
        print(f'  Timing slack    : {wns:.3f} ns  [{slack}]')
        print(f'  Fmax estimate   : {fmax:.1f} MHz')
    if 'hls_latency_cycles' in misc:
        lat = misc['hls_latency_cycles']
        print(f'  Latency         : {lat} clock(s) = {lat * CLOCK_PERIOD:.1f} ns')
        print(f'  Throughput (II=1): {1000//CLOCK_PERIOD} MSa/s')
    else:
        print(f'  (Timing requires vitis_hls)')

    # Resources
    print('\n[Resource Utilization]')
    print(f'  Estimated LUTs (da4ml) : {round(sol.cost)}')
    if any(k in misc for k in ('hls_lut', 'hls_ff')):
        print('  --- Vitis HLS post-csynth estimates ---')
        for key, label in [
            ('hls_lut',  '  LUT   '),
            ('hls_ff',   '  FF    '),
            ('hls_dsp',  '  DSP   '),
            ('hls_bram', '  BRAM  '),
            ('hls_uram', '  URAM  '),
        ]:
            if key in misc:
                print(f'{label}: {misc[key]}')
    else:
        print('  (Resource counts require vitis_hls)')

    # Per-layer bitwidths
    print('\n[Per-layer bitwidths]')
    found = False
    for layer in model.layers:
        parts = []
        v = getattr(layer, 'kif', None)
        if v is not None:
            try:
                k, i, f = [int(x) for x in np.array(v).flatten()[:3]]
                parts.append(f'bw={k+i+f} (k={k},i={i},f={f})')
            except Exception:
                pass
        if hasattr(layer, 'enable_ebops'):
            parts.append(f'EBOPs={int(getattr(layer, "ebops", 0))}')
        if parts:
            print(f'  {layer.name:30s}  {"  ".join(parts)}')
            found = True
    if not found:
        print('  (no HGQ per-layer bitwidth info)')

    print(f'\n{sep}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Generate Vitis HLS C++ implementation for the JSC model.')
    p.add_argument('--model',       default=MODEL_PATH,
                   help='Source .keras model path (default: %(default)s)')
    p.add_argument('--out-dir',     default=OUTPUT_DIR,
                   help='Output directory (default: %(default)s)')
    p.add_argument('--data',        default=DATA_PATH,
                   help='Dataset HDF5 (default: %(default)s)')
    p.add_argument('--flavor',      default=HLS_FLAVOR,
                   choices=['vitis', 'hlslib', 'oneapi'],
                   help='HLS C++ flavor (default: %(default)s)')
    p.add_argument('--clock',       type=int, default=CLOCK_PERIOD,
                   help='Clock period in ns (default: %(default)s)')
    p.add_argument('--part',        default=FPGA_PART,
                   help='FPGA part (default: %(default)s)')
    p.add_argument('--no-sw',       action='store_true',
                   help='Skip SW (Keras + comb + HLS emulator) test')
    p.add_argument('--no-synth',    action='store_true',
                   help='Skip Vitis HLS synthesis even if available')
    p.add_argument('--force-synth', action='store_true',
                   help='Re-run Vitis HLS even if reports exist')
    return p.parse_args()


def main():
    args = parse_args()

    # apply CLI overrides to module globals used by helper functions
    global CLOCK_PERIOD, FPGA_PART, HLS_FLAVOR
    CLOCK_PERIOD = args.clock
    FPGA_PART    = args.part
    HLS_FLAVOR   = args.flavor

    out_dir   = Path(args.out_dir)
    do_sw     = not args.no_sw
    do_synth  = _has_tool('vitis_hls') and not args.no_synth

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Output dir    : {out_dir.resolve()}')
    print(f'HLS flavor    : {HLS_FLAVOR}')
    print(f'Clock period  : {CLOCK_PERIOD} ns  ({1000//CLOCK_PERIOD} MHz)')
    print(f'FPGA part     : {FPGA_PART}')
    print(f'SW test       : {do_sw}')
    print(f'Vitis synth   : {do_synth}  (vitis_hls: {_has_tool("vitis_hls")})')
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    print('Loading data …')
    (X_train, _), (X_val, _), (X_test, y_test) = get_data(args.data, src='openml')

    # ── Load / trace model ────────────────────────────────────────────────
    traced_path = out_dir / 'model.keras'
    if not traced_path.exists():
        print(f'Loading model from {args.model} …')
        model: keras.Model = keras.models.load_model(args.model, compile=False)  # type: ignore
        print('Tracing model (calibrating quantization ranges) …')
        trace_and_save(model, traced_path, X_train, X_val)
        print(f'Traced model saved to {traced_path}')
    else:
        print(f'Loading traced model from {traced_path} …')
        model: keras.Model = keras.models.load_model(traced_path, compile=False)  # type: ignore

    model.summary()

    # Collect EBOPs
    misc = _load_metadata(out_dir)
    misc['ebops'] = sum(
        int(l.ebops) for l in model.layers if getattr(l, 'enable_ebops', False)
    )

    # ── Generate HLS C++ source ───────────────────────────────────────────
    print('\nGenerating HLS C++ source …')
    hls, comb = build_hls_model(model, out_dir, flavor=HLS_FLAVOR)
    _save_metadata(out_dir, misc)

    # ── SW evaluation ─────────────────────────────────────────────────────
    if do_sw:
        misc, _ = sw_test(model, comb, hls, X_test, y_test, misc)
        _save_metadata(out_dir, misc)

    # ── Vitis HLS synthesis ───────────────────────────────────────────────
    if do_synth:
        misc = run_vitis_synth(out_dir, misc, force=args.force_synth)
        _save_metadata(out_dir, misc)

    print_summary(hls, misc, model, do_sw)
    print(f'Results saved to {out_dir / "metadata.json"}')


if __name__ == '__main__':
    main()
