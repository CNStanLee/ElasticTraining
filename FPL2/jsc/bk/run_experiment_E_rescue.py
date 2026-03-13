#!/usr/bin/env python3
"""
run_experiment_E_rescue.py — TopologyRescue 消融实验 (400 eBOPs, v3)
================================================================================

验证 TopologyRescueCallback 在 400 eBOPs 极低预算下的有效性，
与 SoftDeathFloor (Experiment D 已证明无效) 做对比。

TopologyRescue 核心机制:
  - 停滞触发: 只在 val_accuracy 停滞 stall_patience epoch 后干预
  - 结构感知: 只复活谱条件候选 (入度/出度不足节点的死连接)
  - 有效复活: revival_b_val = 1.0 (前向有信号, 梯度有效)
  - 预算中性: swap-kill 最弱活跃连接, 不增加 eBOPs
  - 衰减冷却: 干预后长 cool_down + 强度衰减

消融维度:
  ── 对照组 ──
  E0: TopologyRescue disabled (= v2 baseline, 与 D0 相同管线)
  E9: SDF D4 baseline (SDF alive_threshold=0.4, 作为旧机制对比)

  ── 核心有效性 ──
  E1: TopologyRescue defaults (stall=300, swap_kill=True, max_swap=4)
  E2: TopologyRescue NO swap-kill (只复活不杀死, 测试预算中性的重要性)

  ── stall_patience 消融 ──
  E3: stall_patience=150 (更频繁触发)
  E4: stall_patience=600 (更稀少触发)

  ── max_swap_per_layer 消融 ──
  E5: max_swap=2 (保守: 每次只换 2 个连接)
  E6: max_swap=8 (激进: 每次换 8 个连接)

  ── revival_b_val 消融 ──
  E7: revival_b_val=0.5 (较低复活强度, 接近 1-bit 下限)

  ── cool_down 消融 ──
  E8: cool_down=100 (更短冷却期, 允许更频繁干预)

用法:
  python run_experiment_E_rescue.py --list            # 列出所有配置
  python run_experiment_E_rescue.py --run E0 E1       # 运行指定实验
  python run_experiment_E_rescue.py --run E            # 运行所有 E 实验
  python run_experiment_E_rescue.py --run-all          # 运行全部
  python run_experiment_E_rescue.py --dry-run          # 仅打印配置
  python run_experiment_E_rescue.py --skip-existing    # 跳过已有结果
  python run_experiment_E_rescue.py --compare-only     # 仅打印对比表

输出: results/experiment_E_rescue/<实验名>/
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import sys
import time
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# 实验定义
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_ROOT = 'results/experiment_E_rescue'
TARGET_EBOPS = 400

# TopologyRescue 默认参数
def _rescue(**kwargs):
    base = dict(
        topo_rescue_enabled       = True,
        topo_rescue_revival_b     = 1.0,
        topo_rescue_check_interval = 50,
        topo_rescue_stall_patience = 300,
        topo_rescue_min_delta     = 5e-5,
        topo_rescue_max_swap      = 4,
        topo_rescue_swap_kill     = True,
        topo_rescue_min_degree    = 2,
        topo_rescue_cool_down     = 200,
        topo_rescue_max_interventions = 10,
        topo_rescue_alive_threshold = 0.5,
        topo_rescue_decay_factor  = 0.85,
    )
    base.update(kwargs)
    return base

# SDF 参数 (用于 E9 对照)
def _sdf(alive=0.4, every=50, b_floor=0.05):
    return dict(
        topo_rescue_enabled = False,
        soft_floor_b=b_floor,
        soft_floor_every=every,
        soft_floor_alive=alive,
    )


EXPERIMENTS = {
    # ── E0: Baseline (无任何拓扑修复) ────────────────────────────────────
    'E0': dict(
        desc='No rescue (v2/v3 baseline)',
        overrides=dict(
            topo_rescue_enabled = False,
        ),
    ),

    # ── E1: TopologyRescue 默认参数 ─────────────────────────────────────
    'E1': dict(
        desc='TopologyRescue defaults',
        overrides=_rescue(),
    ),

    # ── E2: 无 swap-kill (测试预算中性的重要性) ──────────────────────────
    'E2': dict(
        desc='TopologyRescue NO swap-kill',
        overrides=_rescue(topo_rescue_swap_kill=False),
    ),

    # ── E3-E4: stall_patience 消融 ─────────────────────────────────────
    'E3': dict(
        desc='stall_patience=150 (frequent)',
        overrides=_rescue(topo_rescue_stall_patience=150),
    ),
    'E4': dict(
        desc='stall_patience=600 (rare)',
        overrides=_rescue(topo_rescue_stall_patience=600),
    ),

    # ── E5-E6: max_swap 消融 ───────────────────────────────────────────
    'E5': dict(
        desc='max_swap=2 (conservative)',
        overrides=_rescue(topo_rescue_max_swap=2),
    ),
    'E6': dict(
        desc='max_swap=8 (aggressive)',
        overrides=_rescue(topo_rescue_max_swap=8),
    ),

    # ── E7: revival_b_val 消融 ──────────────────────────────────────────
    'E7': dict(
        desc='revival_b=0.5 (lower strength)',
        overrides=_rescue(topo_rescue_revival_b=0.5),
    ),

    # ── E8: cool_down 消融 ──────────────────────────────────────────────
    'E8': dict(
        desc='cool_down=100 (shorter)',
        overrides=_rescue(topo_rescue_cool_down=100),
    ),

    # ── E9: SDF D4 baseline (旧机制对照) ─────────────────────────────────
    'E9': dict(
        desc='SDF alive=0.4 (old mechanism)',
        overrides=_sdf(alive=0.4, every=50, b_floor=0.05),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 运行逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def build_config(exp_name: str) -> dict:
    """合并 run_all_v3 DEFAULT_CONFIG + target overrides + 实验覆盖 → 完整配置。"""
    from FPL2.jsc.bk.run_all_v3 import DEFAULT_CONFIG, get_target_overrides_v3

    exp_def = EXPERIMENTS[exp_name]
    cfg = dict(DEFAULT_CONFIG)
    cfg['pretrained'] = 'pretrained_weight/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras'
    cfg['target_ebops'] = TARGET_EBOPS

    # 应用 target overrides (v3)
    target_ov = get_target_overrides_v3(TARGET_EBOPS)
    cfg.update(target_ov)

    # 应用实验覆盖 (最高优先级)
    cfg.update(exp_def['overrides'])

    # 输出目录
    cfg['output_dir'] = os.path.join(OUTPUT_ROOT, exp_name)

    return cfg


def _is_sdf_experiment(cfg: dict) -> bool:
    """判断是否是 SDF 对照实验。"""
    return cfg.get('soft_floor_b', 0) > 0 and cfg.get('soft_floor_every', 999999) < 999999


def run_single(exp_name: str, dry_run: bool = False) -> dict | None:
    """运行单个消融实验。"""
    exp_def = EXPERIMENTS[exp_name]
    cfg = build_config(exp_name)

    total = cfg['phase1_epochs'] + cfg['phase2_epochs']
    is_sdf = _is_sdf_experiment(cfg)
    topo_enabled = cfg.get('topo_rescue_enabled', False)

    mechanism = 'SDF' if is_sdf else ('TopologyRescue' if topo_enabled else 'DISABLED')

    print(f'\n{"#" * 72}')
    print(f'  Experiment {exp_name}: {exp_def["desc"]}')
    print(f'  Target: {TARGET_EBOPS} eBOPs  |  Total: {total} epochs (v3)')
    print(f'  Mechanism: {mechanism}')
    if topo_enabled:
        print(f'    stall_patience={cfg["topo_rescue_stall_patience"]}, '
              f'max_swap={cfg["topo_rescue_max_swap"]}, '
              f'swap_kill={cfg["topo_rescue_swap_kill"]}, '
              f'revival_b={cfg["topo_rescue_revival_b"]}, '
              f'cool_down={cfg["topo_rescue_cool_down"]}')
    if is_sdf:
        print(f'    b_floor={cfg["soft_floor_b"]}, alive={cfg["soft_floor_alive"]}, '
              f'every={cfg["soft_floor_every"]}')
    print(f'  Output: {cfg["output_dir"]}')
    print(f'{"#" * 72}')

    if dry_run:
        print('  [DRY RUN] Skipping...')
        relevant_keys = [
            'topo_rescue_enabled', 'topo_rescue_stall_patience',
            'topo_rescue_max_swap', 'topo_rescue_swap_kill',
            'topo_rescue_revival_b', 'topo_rescue_cool_down',
            'soft_floor_b', 'soft_floor_alive', 'soft_floor_every',
            'phase1_epochs', 'phase2_epochs', 'budget_decay_epochs',
            'warmup_ebops_mul', 'target_ebops',
        ]
        for k in relevant_keys:
            if k in cfg:
                print(f'    {k}: {cfg.get(k)}')
        return None

    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 保存元数据
    meta = {
        'experiment': exp_name,
        'desc': exp_def['desc'],
        'target_ebops': TARGET_EBOPS,
        'total_epochs': total,
        'mechanism': mechanism,
        'topo_rescue_enabled': topo_enabled,
    }
    if topo_enabled:
        meta.update({
            'topo_rescue_stall_patience': cfg.get('topo_rescue_stall_patience'),
            'topo_rescue_max_swap': cfg.get('topo_rescue_max_swap'),
            'topo_rescue_swap_kill': cfg.get('topo_rescue_swap_kill'),
            'topo_rescue_revival_b': cfg.get('topo_rescue_revival_b'),
            'topo_rescue_cool_down': cfg.get('topo_rescue_cool_down'),
            'topo_rescue_decay_factor': cfg.get('topo_rescue_decay_factor'),
        })
    if is_sdf:
        meta.update({
            'sdf_enabled': True,
            'soft_floor_b': cfg.get('soft_floor_b'),
            'soft_floor_alive': cfg.get('soft_floor_alive'),
            'soft_floor_every': cfg.get('soft_floor_every'),
        })
    with open(os.path.join(cfg['output_dir'], 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 选择运行管线
    if is_sdf:
        # SDF 对照实验: 用 v2 管线
        from run_all_v2 import run as run_v2_pipeline
        result = _run_timed(run_v2_pipeline, cfg, exp_name, exp_def)
    else:
        # TopologyRescue / baseline: 用 v3 管线
        from FPL2.jsc.bk.run_all_v3 import run as run_v3_pipeline
        result = _run_timed(run_v3_pipeline, cfg, exp_name, exp_def)

    return result


def _run_timed(pipeline_fn, cfg, exp_name, exp_def):
    """带计时的管线运行。"""
    t0 = time.time()
    result = pipeline_fn(cfg)
    elapsed = time.time() - t0
    result['experiment'] = exp_name
    result['desc'] = exp_def['desc']
    result['elapsed_total'] = elapsed
    return result


def print_comparison(results: list[dict]):
    """打印消融实验对比表。"""
    import h5py
    import numpy as np

    print(f'\n{"=" * 120}')
    print(f'  TopologyRescue ABLATION — target={TARGET_EBOPS} eBOPs (v3, {results[0].get("total_epochs", "?")} ep)')
    print(f'{"=" * 120}')
    print(f'  {"Exp":>4s}  {"Description":<32s}  {"Mechanism":>11s}  '
          f'{"Key Param":<18s}  '
          f'{"Best@Tgt":>8s}  {"@Epoch":>6s}  {"Final":>7s}  {"eBOPs":>6s}  {"Time":>5s}  {"Δ E0":>7s}')
    print(f'  {"-" * 116}')

    # 找 E0 baseline
    e0_acc = None
    for r in results:
        if r.get('experiment') == 'E0':
            e0_acc = _get_best_at_target(r)
            break

    for r in sorted(results, key=lambda x: x.get('experiment', '')):
        exp_name = r.get('experiment', '?')
        desc = r.get('desc', '?')[:32]

        # 读取 trace 获取 best@target
        output_dir = os.path.join(OUTPUT_ROOT, exp_name)
        best_at_target, best_ep = _read_trace_best(output_dir)

        # 确定 mechanism 和 key param
        meta = _read_meta(output_dir)
        mechanism = meta.get('mechanism', '?')[:11]
        key_param = _format_key_param(meta)

        acc_str = f'{best_at_target:.4f}' if best_at_target is not None else f'{r.get("best_val_acc", 0):.4f}'
        ep_str = f'{best_ep}' if best_ep is not None else f'{r.get("best_epoch", "?")}'
        final_acc = r.get('final_val_acc', 0)
        final_eb = r.get('final_ebops', 0)
        elapsed = r.get('elapsed_total', r.get('elapsed_sec', 0))

        acc_val = best_at_target if best_at_target is not None else r.get('best_val_acc', 0)
        delta = f'{acc_val - e0_acc:+.4f}' if e0_acc is not None and acc_val is not None else '—'

        print(f'  {exp_name:>4s}  {desc:<32s}  {mechanism:>11s}  '
              f'{key_param:<18s}  '
              f'{acc_str:>8s}  {ep_str:>6s}  {final_acc:7.4f}  {final_eb:6.0f}  {elapsed:5.0f}s  {delta:>7s}')

    print(f'{"=" * 120}')

    # 简要结论
    if e0_acc is not None:
        best_rescue = None
        best_rescue_acc = e0_acc
        for r in results:
            exp = r.get('experiment', '')
            if exp == 'E0':
                continue
            output_dir = os.path.join(OUTPUT_ROOT, exp)
            acc, _ = _read_trace_best(output_dir)
            if acc is not None and acc > best_rescue_acc:
                best_rescue_acc = acc
                best_rescue = exp

        if best_rescue:
            meta = _read_meta(os.path.join(OUTPUT_ROOT, best_rescue))
            print(f'\n  ★ Best config: {best_rescue} ({best_rescue_acc:.4f}) '
                  f'vs E0 baseline ({e0_acc:.4f}) → Δ = {best_rescue_acc - e0_acc:+.4f}')
            print(f'    Mechanism: {meta.get("mechanism", "?")}')
        else:
            print(f'\n  ★ No config beats the E0 baseline ({e0_acc:.4f})')

        # 与 SDF 对比
        e9_acc = None
        for r in results:
            if r.get('experiment') == 'E9':
                output_dir = os.path.join(OUTPUT_ROOT, 'E9')
                e9_acc, _ = _read_trace_best(output_dir)
                break
        if e9_acc is not None:
            print(f'\n  ★ SDF comparison: E9 (SDF)={e9_acc:.4f} vs E0 (none)={e0_acc:.4f} → Δ={e9_acc-e0_acc:+.4f}')
            if best_rescue:
                print(f'    TopologyRescue ({best_rescue})={best_rescue_acc:.4f} vs SDF (E9)={e9_acc:.4f} '
                      f'→ Δ={best_rescue_acc-e9_acc:+.4f}')


def _get_best_at_target(r: dict):
    output_dir = os.path.join(OUTPUT_ROOT, r.get('experiment', ''))
    acc, _ = _read_trace_best(output_dir)
    if acc is not None:
        return acc
    return r.get('best_acc_at_target', r.get('best_val_acc', 0))


def _read_trace_best(output_dir: str):
    """从 training_trace.h5 读取 target 范围内 best accuracy。"""
    import h5py
    import numpy as np
    trace_path = os.path.join(output_dir, 'training_trace.h5')
    if not os.path.isfile(trace_path):
        return None, None
    try:
        with h5py.File(trace_path, 'r') as f:
            va = f['val_accuracy'][:]
            eb = f['ebops'][:]
            mask = (eb >= TARGET_EBOPS * 0.875) & (eb <= TARGET_EBOPS * 1.125)
            if mask.any():
                idx = np.where(mask)[0][np.argmax(va[mask])]
                return float(va[idx]), int(idx)
    except Exception:
        pass
    return None, None


def _read_meta(output_dir: str) -> dict:
    meta_path = os.path.join(output_dir, 'experiment_meta.json')
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _format_key_param(meta: dict) -> str:
    """格式化关键参数字段。"""
    mechanism = meta.get('mechanism', '')
    if mechanism == 'TopologyRescue':
        stall = meta.get('topo_rescue_stall_patience', '?')
        swap = meta.get('topo_rescue_max_swap', '?')
        kill = meta.get('topo_rescue_swap_kill', '?')
        b_val = meta.get('topo_rescue_revival_b', '?')
        cd = meta.get('topo_rescue_cool_down', '?')
        return f's={stall},m={swap},k={kill}'
    elif mechanism == 'SDF':
        b = meta.get('soft_floor_b', '?')
        a = meta.get('soft_floor_alive', '?')
        e = meta.get('soft_floor_every', '?')
        return f'b={b},a={a},e={e}'
    else:
        return '—'


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Experiment E: TopologyRescue ablation at 400 eBOPs (v3)')
    parser.add_argument('--run', nargs='+', default=None,
                        help='Experiment names to run (e.g. E0 E1, or "E" for all E*, or "all")')
    parser.add_argument('--list', action='store_true',
                        help='List all experiment configs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configs without running')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all experiments')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip experiments with existing results')
    parser.add_argument('--compare-only', action='store_true',
                        help='Only print comparison from existing results')
    args = parser.parse_args()

    if args.list:
        print(f'\n  TopologyRescue Ablation Experiments (target={TARGET_EBOPS} eBOPs):')
        print(f'  {"Name":>4s}  {"Description":<34s}  {"Mechanism":<13s}  '
              f'{"stall":>5s}  {"swap":>4s}  {"kill":>4s}  {"b_val":>5s}  {"cd":>3s}')
        print(f'  {"-" * 95}')
        for name, exp in sorted(EXPERIMENTS.items()):
            ov = exp['overrides']
            if ov.get('topo_rescue_enabled', False):
                mech = 'TopoRescue'
                stall = str(ov.get('topo_rescue_stall_patience', 300))
                swap = str(ov.get('topo_rescue_max_swap', 4))
                kill = str(ov.get('topo_rescue_swap_kill', True))
                b_val = f'{ov.get("topo_rescue_revival_b", 1.0):.1f}'
                cd = str(ov.get('topo_rescue_cool_down', 200))
            elif ov.get('soft_floor_b', 0) > 0:
                mech = 'SDF'
                stall = '—'
                swap = '—'
                kill = '—'
                b_val = f'{ov.get("soft_floor_b", 0):.2f}'
                cd = str(ov.get('soft_floor_every', 50))
            else:
                mech = 'DISABLED'
                stall = swap = kill = b_val = cd = '—'
            print(f'  {name:>4s}  {exp["desc"]:<34s}  {mech:<13s}  '
                  f'{stall:>5s}  {swap:>4s}  {kill:>4s}  {b_val:>5s}  {cd:>3s}')
        return

    if args.compare_only:
        results = []
        for name in sorted(EXPERIMENTS.keys()):
            result_path = os.path.join(OUTPUT_ROOT, name, 'result_summary.json')
            if os.path.isfile(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                r['experiment'] = name
                r['desc'] = EXPERIMENTS[name]['desc']
                results.append(r)
        if results:
            print_comparison(results)
        else:
            print('  No results found. Run experiments first.')
        return

    # 确定要运行的实验
    if args.run_all:
        exp_names = sorted(EXPERIMENTS.keys())
    elif args.run:
        exp_names = []
        for token in args.run:
            if token.lower() == 'all':
                exp_names = sorted(EXPERIMENTS.keys())
                break
            elif token.upper() == 'E':
                exp_names = sorted(EXPERIMENTS.keys())
                break
            else:
                name = token.upper()
                if name not in EXPERIMENTS:
                    print(f'[ERROR] Unknown experiment: {name}')
                    print(f'  Available: {sorted(EXPERIMENTS.keys())}')
                    sys.exit(1)
                exp_names.append(name)
    else:
        parser.print_help()
        print(f'\n  Use --list to see available experiments, --run E0 E1 to run specific ones.')
        return

    print(f'\n{"#" * 72}')
    print(f'  Experiment E: TopologyRescue Ablation')
    print(f'  Target: {TARGET_EBOPS} eBOPs  |  Pipeline: run_all_v3 (TopologyRescue)')
    print(f'  Running: {exp_names}')
    print(f'{"#" * 72}')

    all_results = []
    for name in exp_names:
        output_dir = os.path.join(OUTPUT_ROOT, name)
        result_path = os.path.join(output_dir, 'result_summary.json')

        if args.skip_existing and os.path.isfile(result_path):
            print(f'\n  [Skip] {name}: result exists')
            with open(result_path) as f:
                result = json.load(f)
            result['experiment'] = name
            result['desc'] = EXPERIMENTS[name]['desc']
            all_results.append(result)
            continue

        if args.dry_run:
            run_single(name, dry_run=True)
            continue

        result = run_single(name)
        if result:
            all_results.append(result)

            # 实时进度
            print(f'\n  ── Progress ({len(all_results)}/{len(exp_names)}) ──')
            for r in all_results:
                print(f'    {r["experiment"]}: best_acc={r.get("best_val_acc", 0):.4f}  '
                      f'@target={r.get("best_acc_at_target", 0):.4f}  '
                      f'time={r.get("elapsed_total", 0):.0f}s')

    if all_results and not args.dry_run:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        summary_path = os.path.join(OUTPUT_ROOT, 'all_results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Summary saved: {summary_path}')

        print_comparison(all_results)


if __name__ == '__main__':
    main()
