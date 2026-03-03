"""
eval_hls.py  –  Evaluate an already-generated HLS implementation.

Re-uses the traced model and pre-written Verilog that were created by
run_test.py, and runs only the evaluation phases:

  • SW accuracy  : Keras (float) and combinational C model
  • HW sim       : Verilator RTL emulation (requires verilator)
  • Vivado impl  : synth + place + route utilisation/timing (requires vivado)

Usage
-----
  # evaluate the default HLS directory
  python eval_hls.py

  # evaluate a specific directory
  python eval_hls.py --hls-dir results/hls/epoch=165789

  # skip Vivado even if it is installed
  python eval_hls.py --no-synth

  # force Vivado re-synthesis even if reports already exist
  python eval_hls.py --force-synth
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

# ─── Vivado toolchain path ────────────────────────────────────────────────────
_VIVADO_ROOT = '/tools/Xilinx/Vivado/2022.2'
_VIVADO_BIN  = os.path.join(_VIVADO_ROOT, 'bin')
if _VIVADO_BIN not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _VIVADO_BIN + os.pathsep + os.environ.get('PATH', '')

os.environ['KERAS_BACKEND']       = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path

import keras
import numpy as np
from da4ml.codegen import RTLModel
from da4ml.converter import trace_model
from da4ml.trace import HWConfig, comb_trace
from hgq.utils import trace_minmax  # registers HGQ custom Keras classes

from data.data import get_data


# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_HLS_DIR  = 'results/hls/epoch=165789'
DATA_PATH        = 'data/dataset.h5'
MODULE_NAME      = 'jsc'

# Must match the settings used when the RTL was originally generated
LATENCY_CUTOFF     = 2
CLOCK_PERIOD       = 1.0
CLOCK_UNCERTAINTY  = 0.0
HW_CONFIG          = HWConfig(1, -1, -1)
SOLVER_OPTIONS     = {'hard_dc': 2}
FPGA_PART          = 'xc7k160tffg676-1'


def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.argmax(y_pred, axis=-1) == y_true))


def _load_metadata(path: Path) -> dict:
    p = path / 'metadata.json'
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _save_metadata(path: Path, misc: dict) -> None:
    p = path / 'metadata.json'
    with open(p, 'w') as f:
        json.dump(misc, f, indent=2)


def load_model_and_comb(hls_dir: Path):
    """Load traced model and rebuild combinational trace from the saved .keras."""
    traced = hls_dir / 'model.keras'
    if not traced.exists():
        sys.exit(f'[error] Traced model not found: {traced}\n'
                 '        Run run_test.py first to generate the HLS artefacts.')

    print(f'Loading traced model from {traced} …')
    model: keras.Model = keras.models.load_model(traced, compile=False)  # type: ignore
    model.summary()

    print('Rebuilding combinational trace …')
    inp, out = trace_model(model, hwconf=HW_CONFIG, solver_options=SOLVER_OPTIONS)
    comb = comb_trace(inp, out)

    rtl = RTLModel(
        comb,
        MODULE_NAME,
        hls_dir,
        latency_cutoff=LATENCY_CUTOFF,
        clock_period=CLOCK_PERIOD,
        clock_uncertainty=CLOCK_UNCERTAINTY,
    )
    print(repr(rtl))
    return model, comb, rtl


# ─── Evaluation phases ────────────────────────────────────────────────────────

def run_sw_test(model, comb, X_test, y_test, misc: dict) -> dict:
    print('\n── SW accuracy ──────────────────────────────────')
    y_keras = model.predict(X_test, batch_size=25600, verbose=0)
    keras_acc = _accuracy(y_test, y_keras)
    print(f'  Keras (float) accuracy : {keras_acc:.4f}')

    c_pred = comb.predict(X_test, n_threads=4)
    comb_acc = _accuracy(y_test, c_pred)
    print(f'  Comb  (fixed) accuracy : {comb_acc:.4f}')

    misc['keras_metric'] = keras_acc
    misc['comb_metric']  = comb_acc
    return misc, c_pred


def run_hw_test(rtl: RTLModel, X_test, y_test, misc: dict,
                c_pred=None) -> dict:
    print('\n── HW simulation (Verilator) ────────────────────')
    print('  Compiling RTL emulator …')
    for attempt in range(8):
        try:
            rtl._compile(openmp=True, nproc=4)
            break
        except RuntimeError:
            if attempt == 7:
                print('  [warn] Verilator compilation failed after 8 attempts')
                return misc

    y_hw = rtl.predict(np.array(X_test))
    hw_acc = _accuracy(y_test, y_hw)
    print(f'  HW simulation accuracy : {hw_acc:.4f}')
    misc['hw_metric'] = hw_acc

    if c_pred is not None:
        ndiff = int(np.sum(y_hw != c_pred))
        ratio = ndiff / y_hw.size
        if ndiff:
            print(f'  HW/comb mismatch : {ndiff} / {y_hw.size}  ({ratio:.4%})')
        else:
            print('  HW/comb match    : perfect')
        misc['hw_sw_diff'] = ratio

    return misc


def run_vivado_synth(hls_dir: Path, misc: dict, force: bool = False) -> dict:
    """Run full Vivado implementation (synth+P&R) on the existing Verilog."""
    src_dir  = hls_dir / 'src'
    top_v    = src_dir / f'{MODULE_NAME}.v'
    if not top_v.exists():
        print(f'  [warn] {top_v} not found — skipping Vivado implementation.')
        return misc

    out_dir      = hls_dir / f'output_{MODULE_NAME}'
    rpt_dir      = out_dir / 'reports'
    util_rpt_path = rpt_dir / f'{MODULE_NAME}_util.rpt'

    if util_rpt_path.exists() and not force:
        print('\n── Vivado implementation ────────────────────────')
        print('  Reports already exist — skipping re-synthesis.')
        print(f'  (use --force-synth to re-run)')
    else:
        rpt_dir.mkdir(parents=True, exist_ok=True)
        verilog_files = list(src_dir.glob('*.v')) + list((src_dir / 'static').glob('*.v'))
        verilog_list  = ' '.join(f'[file normalize "{f}"]' for f in verilog_files)
        xdc_file  = src_dir / f'{MODULE_NAME}.xdc'
        xdc_block = f'read_xdc -mode out_of_context "{xdc_file}"' if xdc_file.exists() else ''

        tcl = f"""# Auto-generated full-implementation script (eval_hls.py)
set prj  {MODULE_NAME}
set part {FPGA_PART}
set rpt  "{rpt_dir.resolve()}"
set out  "{out_dir.resolve()}"

read_verilog {verilog_list}
{xdc_block}

synth_design \\
    -top $prj \\
    -part $part \\
    -mode out_of_context \\
    -flatten_hierarchy full \\
    -resource_sharing auto \\
    -directive AreaOptimized_High

write_checkpoint -force "$out/${{prj}}_post_synth.dcp"
report_utilization     -file "$rpt/${{prj}}_post_synth_util.rpt"
report_timing_summary  -file "$rpt/${{prj}}_post_synth_timing.rpt"

opt_design   -directive ExploreWithRemap
place_design -directive Default
phys_opt_design -directive AggressiveExplore
write_checkpoint -force "$out/${{prj}}_post_place.dcp"
file delete -force "$out/${{prj}}_post_synth.dcp"

route_design -directive NoTimingRelaxation
write_checkpoint -force "$out/${{prj}}_post_route.dcp"
file delete -force "$out/${{prj}}_post_place.dcp"

report_utilization    -file "$rpt/${{prj}}_util.rpt"
report_timing_summary -file "$rpt/${{prj}}_timing.rpt"
report_power          -file "$rpt/${{prj}}_power.rpt"
report_drc            -file "$rpt/${{prj}}_drc.rpt"
puts "Implementation complete."
"""
        tcl_path = hls_dir / '_eval_impl.tcl'
        tcl_path.write_text(tcl)
        log_path = hls_dir / 'vivado_impl.log'

        print('\n── Vivado implementation (synth+place+route) ────')
        print(f'  Part   : {FPGA_PART}')
        print(f'  Top    : {MODULE_NAME}')
        print(f'  (~5-15 min) …')
        try:
            with open(log_path, 'w') as logf:
                subprocess.run(
                    ['vivado', '-mode', 'batch', '-source', tcl_path.name],
                    cwd=hls_dir,
                    check=True,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError:
            errors = [l for l in open(log_path) if 'ERROR' in l]
            for e in errors[-3:]:
                print(f'  {e.strip()}')
            print(f'  [warn] Vivado failed — see {log_path}')
            return misc

    # ── Parse reports ──────────────────────────────────────────────────────
    report: dict = {}

    util_rpt = rpt_dir / f'{MODULE_NAME}_util.rpt'
    if util_rpt.exists():
        text = util_rpt.read_text()
        for key, pat in [
            ('synth_lut',     r'\|\s*LUT as Logic\s*\|\s*(\d+)'),
            ('synth_ff',      r'\|\s*Register as Flip Flop\s*\|\s*(\d+)'),
            ('synth_lut_ram', r'\|\s*LUT as Memory\s*\|\s*(\d+)'),
            ('synth_dsp',     r'\|\s*DSPs\s*\|\s*(\d+)'),
            ('synth_bram',    r'\|\s*Block RAM Tile\s*\|\s*(\d+)'),
        ]:
            m = re.search(pat, text)
            if m:
                try:
                    report[key] = int(m.group(1))
                except ValueError:
                    pass

    timing_rpt = rpt_dir / f'{MODULE_NAME}_timing.rpt'
    if timing_rpt.exists():
        text = timing_rpt.read_text()
        m = re.search(
            r'WNS\(ns\)\s+TNS\(ns\)[^\n]*\n\s*-+\s+-+[^\n]*\n\s*([\-\d\.]+)\s+([\-\d\.]+)',
            text,
        )
        if m:
            try:
                report['synth_wns_ns'] = float(m.group(1))
                report['synth_tns_ns'] = float(m.group(2))
            except ValueError:
                pass

    power_rpt = rpt_dir / f'{MODULE_NAME}_power.rpt'
    if power_rpt.exists():
        text = power_rpt.read_text()
        for key, pat in [
            ('synth_total_power_mw',   r'\|\s*Total On-Chip Power \(W\)\s*\|\s*([\d\.]+)'),
            ('synth_dynamic_power_mw', r'\|\s*Dynamic \(W\)\s*\|\s*([\d\.]+)'),
            ('synth_static_power_mw',  r'\|\s*Device Static \(W\)\s*\|\s*([\d\.]+)'),
        ]:
            m = re.search(pat, text)
            if m:
                try:
                    report[key] = round(float(m.group(1)) * 1000, 3)
                except ValueError:
                    pass

    misc.update(report)
    return misc


# ─── Summary printer ──────────────────────────────────────────────────────────

def print_summary(rtl: RTLModel, misc: dict, model: keras.Model,
                  sw: bool, hw: bool):
    sep = '=' * 55
    print(f'\n{sep}')
    print('  PERFORMANCE SUMMARY  (eval_hls.py)')
    print(sep)

    sol  = rtl._solution
    pipe = rtl._pipe
    cp   = rtl._clock_period

    # Accuracy
    print('\n[Accuracy]')
    if sw:
        print(f'  Keras (float) : {misc.get("keras_metric", "N/A"):.4f}')
        print(f'  Comb  (fixed) : {misc.get("comb_metric",  "N/A"):.4f}')
    if hw:
        print(f'  HW simulation : {misc.get("hw_metric",    "N/A"):.4f}')
        if 'hw_sw_diff' in misc:
            print(f'  HW/SW mismatch: {misc["hw_sw_diff"]:.4%}')

    # EBOPs
    print('\n[Effective BOPs]')
    print(f'  EBOPs/inference : {misc.get("ebops", "N/A")}')

    # Timing & throughput
    print('\n[Timing & Throughput]')
    print(f'  Clock period    : {cp:.3f} ns  ({1e3/cp:.1f} MHz target)')
    if pipe is not None:
        n_stages = misc.get('latency', len(pipe[0]))
        lat_ns   = n_stages * cp
        tp_gsps  = 1.0 / cp
        print(f'  Pipeline stages : {n_stages}')
        print(f'  Latency         : {lat_ns:.3f} ns  ({n_stages} clocks)')
        print(f'  Initiation interval : 1 clock = {cp:.3f} ns')
        print(f'  Throughput      : {tp_gsps:.3f} GSa/s  ({tp_gsps*1e3:.1f} MSa/s)')
    else:
        print(f'  Combinational delay : {getattr(sol, "latency", "N/A")}')

    wns = misc.get('synth_wns_ns')
    if wns is not None:
        tns   = misc.get('synth_tns_ns', 0.0)
        slack = 'MET' if wns >= 0 else 'VIOLATED'
        fmax  = 1e3 / (cp - wns)
        print(f'  WNS (post-impl) : {wns:.3f} ns  [{slack}]')
        print(f'  TNS (post-impl) : {tns:.3f} ns')
        print(f'  Fmax            : {fmax:.1f} MHz  '
              f'(critical path = {cp - wns:.3f} ns)')

    # Resources
    print('\n[Resource Utilization]')
    print(f'  Estimated LUTs  : {round(sol.cost)}')
    if pipe is not None:
        print(f'  Estimated FFs   : {misc.get("reg_bits", "N/A")}')
    if any(k in misc for k in ('synth_lut', 'synth_ff')):
        print('  --- Vivado post-implementation ---')
        for key, label in [
            ('synth_lut',     '  LUT as Logic  '),
            ('synth_lut_ram', '  LUT as RAM    '),
            ('synth_ff',      '  Flip-Flops    '),
            ('synth_dsp',     '  DSPs          '),
            ('synth_bram',    '  BRAMs         '),
        ]:
            if key in misc:
                print(f'{label}: {int(misc[key])}')
    else:
        print('  (Run with --synth for Vivado utilization)')

    # Power
    print('\n[Power]')
    if 'synth_total_power_mw' in misc:
        print(f'  Total power     : {misc["synth_total_power_mw"]:.2f} mW')
        if 'synth_dynamic_power_mw' in misc:
            print(f'    Dynamic       : {misc["synth_dynamic_power_mw"]:.2f} mW')
        if 'synth_static_power_mw' in misc:
            print(f'    Static        : {misc["synth_static_power_mw"]:.2f} mW')
    else:
        print('  (Run with --synth for Vivado power estimate)')

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
        print('  (no per-layer HGQ bitwidth info)')

    print(f'\n{sep}')


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Evaluate a pre-generated HLS implementation.')
    p.add_argument('--hls-dir',     default=DEFAULT_HLS_DIR,
                   help='Path to HLS output directory (default: %(default)s)')
    p.add_argument('--data',        default=DATA_PATH,
                   help='Path to dataset HDF5 file (default: %(default)s)')
    p.add_argument('--no-sw',       action='store_true',
                   help='Skip SW (Keras + comb C model) accuracy test')
    p.add_argument('--hw',          action='store_true',
                   help='Run HW simulation via Verilator (auto if available)')
    p.add_argument('--synth',       action='store_true',
                   help='Run Vivado implementation (auto if available)')
    p.add_argument('--no-synth',    action='store_true',
                   help='Disable Vivado even if it is installed')
    p.add_argument('--force-synth', action='store_true',
                   help='Re-run Vivado even if reports already exist')
    return p.parse_args()


def main():
    args = parse_args()

    hls_dir  = Path(args.hls_dir)
    do_sw    = not args.no_sw
    do_hw    = args.hw or _has_tool('verilator')
    do_synth = (args.synth or _has_tool('vivado')) and not args.no_synth

    if not hls_dir.exists():
        sys.exit(f'[error] HLS directory not found: {hls_dir}')

    print(f'HLS directory : {hls_dir.resolve()}')
    print(f'SW test       : {do_sw}')
    print(f'HW sim        : {do_hw}  (verilator: {_has_tool("verilator")})')
    print(f'Vivado synth  : {do_synth}  (vivado: {_has_tool("vivado")})')
    print()

    # Load data
    print('Loading data …')
    (X_train, _), (X_val, _), (X_test, y_test) = get_data(args.data, src='openml')

    # Load model + rebuild RTL
    model, comb, rtl = load_model_and_comb(hls_dir)

    misc   = _load_metadata(hls_dir)
    c_pred = None

    if do_sw:
        misc, c_pred = run_sw_test(model, comb, X_test, y_test, misc)
        _save_metadata(hls_dir, misc)

    if do_hw:
        misc = run_hw_test(rtl, X_test, y_test, misc, c_pred)
        _save_metadata(hls_dir, misc)

    if do_synth:
        misc = run_vivado_synth(hls_dir, misc, force=args.force_synth)
        _save_metadata(hls_dir, misc)

    print_summary(rtl, misc, model, do_sw, do_hw)
    print(f'Results saved to {hls_dir / "metadata.json"}')


if __name__ == '__main__':
    main()
