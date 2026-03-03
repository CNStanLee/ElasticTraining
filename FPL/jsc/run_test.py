import os
import shutil
import subprocess

# ─── Vivado toolchain path ────────────────────────────────────────────────────
_VIVADO_ROOT = '/tools/Xilinx/Vivado/2022.2'
_VIVADO_BIN  = os.path.join(_VIVADO_ROOT, 'bin')
if _VIVADO_BIN not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _VIVADO_BIN + os.pathsep + os.environ.get('PATH', '')

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from pathlib import Path

import keras
import numpy as np
from da4ml.codegen import RTLModel
from da4ml.converter import trace_model
from da4ml.trace import HWConfig, comb_trace
from hgq.utils import trace_minmax

from data.data import get_data


def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = 'results/baseline/epoch=165789-val_acc=0.712-ebops=304-val_loss=0.888.keras'
DATA_PATH = 'data/dataset.h5'
OUTPUT_DIR = 'results/hls/epoch=165789'

# ─── HLS settings ─────────────────────────────────────────────────────────────
LATENCY_CUTOFF = 2
CLOCK_PERIOD = 1.0
CLOCK_UNCERTAINTY = 0.0
HW_CONFIG = HWConfig(1, -1, -1)
SOLVER_OPTIONS = {'hard_dc': 2}
# Xilinx device for synthesis (WebPack-compatible default; change to your target part)
FPGA_PART = 'xc7k160tffg676-1'

SW_TEST = True
HW_TEST = _has_tool('verilator')  # auto-enable when Verilator is installed
SYNTH = _has_tool('vivado')        # auto-enable when Vivado is installed


def trace_and_save(model: keras.Model, path: str | Path, *datasets: np.ndarray):
    """Run trace_minmax over the calibration sets then re-save the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ds, _dss = datasets[0], datasets[1:]
    trace_minmax(model, _ds, batch_size=25600, reset=True)
    for ds in _dss:
        trace_minmax(model, ds, batch_size=25600, reset=False)
    model.save(path)


def convert_and_test(
    model: keras.Model,
    name: str,
    path: str | Path,
    ds_test: tuple,
    metric,
    sw_test: bool = True,
    hw_test: bool = False,
):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    X_test, y_test = ds_test

    # ── trace → combinational circuit ──────────────────────────────────────
    inp, out = trace_model(
        model,
        hwconf=HW_CONFIG,
        solver_options=SOLVER_OPTIONS,
    )
    comb = comb_trace(inp, out)

    # ── generate RTL / HLS ─────────────────────────────────────────────────
    rtl = RTLModel(
        comb,
        name,
        path,
        latency_cutoff=LATENCY_CUTOFF,
        clock_period=CLOCK_PERIOD,
        clock_uncertainty=CLOCK_UNCERTAINTY,
    )
    rtl.write()
    print(f'HLS code written to: {path}')
    print()
    print(repr(rtl))   # prints estimated LUTs, FFs, pipeline stages
    print()

    # ── collect ebops ──────────────────────────────────────────────────────
    metadata_path = path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            misc = json.load(f)
    else:
        misc = {}

    ebops = 0
    for layer in model.layers:
        if getattr(layer, 'enable_ebops', False):
            ebops += int(layer.ebops)  # type: ignore
    misc['ebops'] = ebops

    with open(metadata_path, 'w') as f:
        json.dump(misc, f, indent=2)

    # ── SW evaluation ──────────────────────────────────────────────────────
    if sw_test:
        y_pred_keras = model.predict(X_test, batch_size=25600, verbose=0)
        keras_acc = metric(y_test, y_pred_keras)
        print(f'Keras accuracy:  {keras_acc:.4f}')

        c_pred = comb.predict(X_test, n_threads=4)
        comb_acc = metric(y_test, c_pred)
        print(f'Comb accuracy:   {comb_acc:.4f}')

        misc['keras_metric'] = float(keras_acc)
        misc['comb_metric'] = float(comb_acc)
        with open(metadata_path, 'w') as f:
            json.dump(misc, f, indent=2)

    # ── HW simulation (requires Verilator) ───────────────────────────────
    if hw_test:
        print('Compiling RTL emulator (Verilator)...')
        for _ in range(8):
            try:
                rtl._compile(openmp=True, nproc=4)
                break
            except RuntimeError:
                pass
        y_pred_hw = rtl.predict(np.array(X_test))
        hw_acc = metric(y_test, y_pred_hw)
        print(f'HW (sim) accuracy: {hw_acc:.4f}')
        misc['hw_metric'] = float(hw_acc)
        with open(metadata_path, 'w') as f:
            json.dump(misc, f, indent=2)

        if sw_test:
            ndiff = int(np.sum(y_pred_hw != c_pred))
            if ndiff > 0:
                print(f'HW/comb diff: {ndiff} / {y_pred_hw.size}')
            misc['hw_sw_diff'] = ndiff / y_pred_hw.size
            with open(metadata_path, 'w') as f:
                json.dump(misc, f, indent=2)

    # ── Vivado synthesis (requires Vivado in PATH) ─────────────────────────
    vivado_report = {}
    if SYNTH:
        print('\nRunning Vivado synthesis for exact utilization and timing...')
        vivado_report = _run_vivado_synth(path)
        misc.update(vivado_report)
        with open(metadata_path, 'w') as f:
            json.dump(misc, f, indent=2)

    # ── full performance summary ───────────────────────────────────────────
    _print_performance_summary(rtl, misc, vivado_report, model, sw_test, hw_test)

    print(f'\nResults saved to {metadata_path}')
    return misc


def _run_vivado_synth(path: Path) -> dict:
    """Generate a synthesis-only TCL and run Vivado OOC synthesis."""
    import re
    path    = path.resolve()          # make absolute to avoid doubled segments in TCL
    src_dir = path / 'src'
    top_v   = src_dir / 'jsc.v'
    if not top_v.exists():
        print('  [warn] jsc.v not found — skipping Vivado synthesis.')
        return {}

    out_dir       = path / 'output_jsc'
    rpt_dir       = out_dir / 'reports'
    util_rpt_path = rpt_dir / 'jsc_util.rpt'

    if not util_rpt_path.exists():
        rpt_dir.mkdir(parents=True, exist_ok=True)

        # Collect all Verilog source files
        verilog_files = list(src_dir.glob('*.v')) + list((src_dir / 'static').glob('*.v'))
        verilog_list  = ' '.join(f'[file normalize "{f}"]' for f in verilog_files)

        # Optional XDC timing constraint
        xdc_file  = src_dir / 'jsc.xdc'
        xdc_block = f'read_xdc -mode out_of_context "{xdc_file}"' if xdc_file.exists() else ''

        synth_tcl = f"""# Auto-generated full implementation script
set prj  jsc
set part {FPGA_PART}
set rpt  "{rpt_dir}"
set out  "{out_dir}"

read_verilog {verilog_list}
{xdc_block}

# ── Synthesis ────────────────────────────────────────────────────────────
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

# ── Opt ──────────────────────────────────────────────────────────────────
opt_design -directive ExploreWithRemap

# ── Place ────────────────────────────────────────────────────────────────
place_design -directive Default
phys_opt_design -directive AggressiveExplore
write_checkpoint -force "$out/${{prj}}_post_place.dcp"
file delete -force "$out/${{prj}}_post_synth.dcp"

# ── Route ────────────────────────────────────────────────────────────────
route_design -directive NoTimingRelaxation
write_checkpoint -force "$out/${{prj}}_post_route.dcp"
file delete -force "$out/${{prj}}_post_place.dcp"

# ── Reports (post-route = actual implementation results) ─────────────────
report_utilization    -file "$rpt/${{prj}}_util.rpt"
report_timing_summary -file "$rpt/${{prj}}_timing.rpt"
report_power          -file "$rpt/${{prj}}_power.rpt"
report_drc            -file "$rpt/${{prj}}_drc.rpt"
puts "Implementation complete."
"""
        synth_tcl_path = path / '_synth_only.tcl'
        synth_tcl_path.write_text(synth_tcl)

        log_path = path / 'vivado_impl.log'
        print(f'  Running full implementation (synth+place+route) on {FPGA_PART} (~5-15 min)...')
        try:
            with open(log_path, 'w') as logf:
                subprocess.run(
                    ['vivado', '-mode', 'batch', '-source', synth_tcl_path.name],
                    cwd=path,
                    check=True,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError:
            errors = [l for l in open(log_path) if 'ERROR' in l]
            for err in errors[-3:]:
                print(f'  {err.strip()}')
            print(f'  [warn] Vivado implementation failed (see {log_path})')
            return {}
    else:
        print('  Reports already exist — skipping re-synthesis.')

    report = {}

    # Utilization
    util_rpt = rpt_dir / 'jsc_util.rpt'
    if util_rpt.exists():
        text = util_rpt.read_text()
        for key, pattern in [
            ('synth_lut',     r'\|\s*LUT as Logic\s*\|\s*(\d+)'),
            ('synth_ff',      r'\|\s*Register as Flip Flop\s*\|\s*(\d+)'),
            ('synth_lut_ram', r'\|\s*LUT as Memory\s*\|\s*(\d+)'),
            ('synth_dsp',     r'\|\s*DSPs\s*\|\s*(\d+)'),
            ('synth_bram',    r'\|\s*Block RAM Tile\s*\|\s*(\d+)'),
        ]:
            m = re.search(pattern, text)
            if m:
                try:
                    report[key] = int(m.group(1))
                except ValueError:
                    pass

    # Timing
    timing_rpt = rpt_dir / 'jsc_timing.rpt'
    if timing_rpt.exists():
        text = timing_rpt.read_text()
        m = re.search(
            r'WNS\(ns\)\s+TNS\(ns\)[^\n]*\n'
            r'\s*-+\s+-+[^\n]*\n'
            r'\s*([\-\d\.]+)\s+([\-\d\.]+)',
            text,
        )
        if m:
            try:
                report['synth_wns_ns'] = float(m.group(1))
                report['synth_tns_ns'] = float(m.group(2))
            except ValueError:
                pass

    # Power (Vivado reports in Watts; convert to mW)
    power_rpt = rpt_dir / 'jsc_power.rpt'
    if power_rpt.exists():
        text = power_rpt.read_text()
        for key, pattern in [
            ('synth_total_power_mw',   r'\|\s*Total On-Chip Power \(W\)\s*\|\s*([\d\.]+)'),
            ('synth_dynamic_power_mw', r'\|\s*Dynamic \(W\)\s*\|\s*([\d\.]+)'),
            ('synth_static_power_mw',  r'\|\s*Device Static \(W\)\s*\|\s*([\d\.]+)'),
        ]:
            m = re.search(pattern, text)
            if m:
                try:
                    report[key] = round(float(m.group(1)) * 1000, 3)  # W → mW
                except ValueError:
                    pass

    return report


def _print_performance_summary(rtl: RTLModel, misc: dict, vivado: dict, model: keras.Model, sw_test: bool, hw_test: bool):
    sep = '=' * 55
    print(f'\n{sep}')
    print('  PERFORMANCE SUMMARY')
    print(sep)

    sol = rtl._solution
    pipe = rtl._pipe
    cp = rtl._clock_period  # ns

    # ── Accuracy ──────────────────────────────────────────────────────────
    print('\n[Accuracy]')
    if sw_test:
        print(f'  Keras (floating-point) : {misc.get("keras_metric", "N/A"):.4f}')
        print(f'  Combinational (fixed)  : {misc.get("comb_metric",  "N/A"):.4f}')
    if hw_test:
        print(f'  HW simulation          : {misc.get("hw_metric",    "N/A"):.4f}')

    # ── EBOPs ─────────────────────────────────────────────────────────────
    print('\n[Effective BOPs]')
    print(f'  EBOPs/inference : {misc.get("ebops", "N/A")}')

    # ── Timing & Throughput ───────────────────────────────────────────────
    print('\n[Timing & Throughput]')
    print(f'  Clock period    : {cp:.3f} ns  ({1e3/cp:.1f} MHz)')
    if pipe is not None:
        n_stages = misc.get('latency', len(pipe[0]))
        lat_ns = n_stages * cp
        ii_ns = cp           # II = 1 clock
        throughput_gsps = 1.0 / ii_ns   # samples per ns = GSa/s
        print(f'  Pipeline stages : {n_stages}')
        print(f'  Latency         : {lat_ns:.3f} ns  ({n_stages} clocks)')
        print(f'  Initiation interval (II) : 1 clock = {ii_ns:.3f} ns')
        print(f'  Throughput      : {throughput_gsps:.3f} GSa/s  '
              f'({throughput_gsps*1e3:.1f} MSa/s)')
    else:
        comb_delay = getattr(sol, 'latency', 'N/A')
        print(f'  Combinational delay : {comb_delay} (relative units)')
        print(f'  Throughput          : limited by combinational path')

    if 'synth_wns_ns' in vivado:
        wns = vivado['synth_wns_ns']
        tns = vivado.get('synth_tns_ns', 0.0)
        slack = 'MET' if wns >= 0 else 'VIOLATED'
        fmax_mhz = 1e3 / (cp - wns)   # cp - wns = actual critical path (ns)
        print(f'  Timing slack WNS    : {wns:.3f} ns  [{slack}]')
        print(f'  Timing slack TNS    : {tns:.3f} ns')
        print(f'  Max frequency (Fmax): {fmax_mhz:.1f} MHz  '
              f'(critical path = {cp - wns:.3f} ns)  [post-implementation]')

    # ── Resource estimates ────────────────────────────────────────────────
    print('\n[Resource Utilization]')
    est_lut = round(sol.cost)
    print(f'  Estimated LUTs  : {est_lut}')
    if pipe is not None:
        reg_bits = misc.get('reg_bits', getattr(pipe, 'reg_bits', 'N/A'))
        print(f'  Estimated FFs   : {reg_bits}  (pipeline register bits)')

    if vivado:
        print('  --- Vivado post-implementation (place & route) ---')
        for key, label in [
            ('synth_lut',     '  LUT as Logic    '),
            ('synth_lut_ram', '  LUT as RAM      '),
            ('synth_ff',      '  Flip-Flops      '),
            ('synth_dsp',     '  DSPs            '),
            ('synth_bram',    '  BRAMs           '),
        ]:
            if key in vivado:
                print(f'{label}: {int(vivado[key])}')
    else:
        print('  (Actual LUT/FF/DSP/BRAM counts require Vivado synthesis)')

    # ── Power ─────────────────────────────────────────────────────────────
    print('\n[Power]')
    if 'synth_total_power_mw' in vivado:
        print(f'  Total on-chip power : {vivado["synth_total_power_mw"]:.2f} mW')
        if 'synth_dynamic_power_mw' in vivado:
            print(f'    Dynamic           : {vivado["synth_dynamic_power_mw"]:.2f} mW')
        if 'synth_static_power_mw' in vivado:
            print(f'    Static            : {vivado["synth_static_power_mw"]:.2f} mW')
    else:
        print('  Power estimate requires Vivado (not available)')

    # ── Per-layer bitwidths ────────────────────────────────────────────────────────
    print('\n[Per-layer bitwidths]')
    bw_found = False
    for layer in model.layers:
        # HGQ2 layers expose .kif (keep, int, frac) for weights / activations
        layer_info = []
        for attr in ('kif',):
            v = getattr(layer, attr, None)
            if v is not None:
                try:
                    k, i, f = [int(x) for x in np.array(v).flatten()[:3]]
                    bw = k + i + f
                    layer_info.append(f'bw={bw} (k={k},i={i},f={f})')
                except Exception:
                    pass
        if hasattr(layer, 'enable_ebops'):
            ebops_l = int(getattr(layer, 'ebops', 0))
            layer_info.append(f'EBOPs={ebops_l}')
        if layer_info:
            print(f'  {layer.name:30s}  {"  ".join(layer_info)}')
            bw_found = True
    if not bw_found:
        print('  (no per-layer bitwidth info found on HGQ layers)')

    print(f'\n{sep}')


if __name__ == '__main__':
    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── load data ──────────────────────────────────────────────────────────
    print('Loading data...')
    (X_train, _), (X_val, _), (X_test, y_test) = get_data(DATA_PATH, src='openml')
    ds_test = (X_test, y_test)

    # ── load / trace model ─────────────────────────────────────────────────
    traced_path = out_path / 'model.keras'
    if not traced_path.exists():
        print(f'Loading model from {MODEL_PATH}...')
        model: keras.Model = keras.models.load_model(MODEL_PATH, compile=False)  # type: ignore
        print('Tracing model (calibrating quantization ranges)...')
        trace_and_save(model, traced_path, X_train, X_val)
        print(f'Traced model saved to {traced_path}')
    else:
        print(f'Loading traced model from {traced_path}...')
        model: keras.Model = keras.models.load_model(traced_path, compile=False)  # type: ignore

    model.summary()

    # ── convert & evaluate ─────────────────────────────────────────────────
    print('\nConverting to HLS and evaluating...')
    convert_and_test(
        model,
        'jsc',
        out_path,
        ds_test,
        lambda y_true, y_pred: float(np.mean(np.argmax(y_pred, axis=-1) == y_true)),
        sw_test=SW_TEST,
        hw_test=HW_TEST,
    )
