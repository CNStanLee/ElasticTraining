#!/usr/bin/env python3
"""
run_experiment_D_floor.py — SoftDeathFloor 消融实验 (400 eBOPs, fast decay v2)
================================================================================

验证 SoftDeathFloor 机制在 400 eBOPs 极低预算下的有效性。

SoftDeathFloor 作用: 定期将 "死连接" (kq.b ≤ alive_threshold) 的位宽
提升到 b_floor，防止连接永久死亡，给优化器机会重新激活有用连接。

消融维度:
  D0: SDF disabled (baseline, 与 fast_decay 默认一致)
  D1: SDF enabled, alive_threshold=0.1 (极宽松: 几乎不干预)
  D2: SDF enabled, alive_threshold=0.2
  D3: SDF enabled, alive_threshold=0.3
  D4: SDF enabled, alive_threshold=0.4 (默认值)
  D5: SDF enabled, alive_threshold=0.5
  D6: SDF enabled, alive_threshold=0.6 (激进: 频繁干预)
  D7: SDF enabled, alive=0.4, every=25 (高频率)
  D8: SDF enabled, alive=0.4, every=100 (低频率)
  D9: SDF enabled, alive=0.4, b_floor=0.1 (更高下限)

用法:
  python run_experiment_D_floor.py --list            # 列出所有配置
  python run_experiment_D_floor.py --run D0 D4       # 运行指定实验
  python run_experiment_D_floor.py --run D            # 运行所有 D 实验
  python run_experiment_D_floor.py --run-all          # 运行全部
  python run_experiment_D_floor.py --dry-run          # 仅打印配置
  python run_experiment_D_floor.py --skip-existing    # 跳过已有结果

输出: results/experiment_D_floor/<实验名>/
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

OUTPUT_ROOT = 'results/experiment_D_floor'
TARGET_EBOPS = 400

# SDF 公共基础: 启用 SoftDeathFloor
def _sdf(alive=0.4, every=50, b_floor=0.05):
    return dict(soft_floor_b=b_floor, soft_floor_every=every, soft_floor_alive=alive)

EXPERIMENTS = {
    # ── D0: Baseline (SDF 关闭) ──────────────────────────────────────────
    'D0': dict(
        desc='SDF disabled (baseline)',
        overrides=dict(
            soft_floor_b=0.0,
            soft_floor_every=999999,
        ),
    ),

    # ── D1-D6: alive_threshold 消融 ─────────────────────────────────────
    'D1': dict(desc='SDF alive_threshold=0.1',  overrides=_sdf(alive=0.1)),
    'D2': dict(desc='SDF alive_threshold=0.2',  overrides=_sdf(alive=0.2)),
    'D3': dict(desc='SDF alive_threshold=0.3',  overrides=_sdf(alive=0.3)),
    'D4': dict(desc='SDF alive_threshold=0.4 (default)', overrides=_sdf(alive=0.4)),
    'D5': dict(desc='SDF alive_threshold=0.5',  overrides=_sdf(alive=0.5)),
    'D6': dict(desc='SDF alive_threshold=0.6',  overrides=_sdf(alive=0.6)),

    # ── D7-D8: 执行频率消融 ─────────────────────────────────────────────
    'D7': dict(desc='SDF every=25 (high freq)',  overrides=_sdf(every=25)),
    'D8': dict(desc='SDF every=100 (low freq)',  overrides=_sdf(every=100)),

    # ── D9: b_floor 值消融 ──────────────────────────────────────────────
    'D9': dict(desc='SDF b_floor=0.1 (higher floor)', overrides=_sdf(b_floor=0.1)),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 运行逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def build_config(exp_name: str) -> dict:
    """合并 run_all_v2 DEFAULT_CONFIG + target overrides + 实验覆盖 → 完整配置。"""
    from run_all_v2 import DEFAULT_CONFIG, get_target_overrides_v2

    exp_def = EXPERIMENTS[exp_name]
    cfg = dict(DEFAULT_CONFIG)
    cfg['pretrained'] = 'pretrained_weight/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras'
    cfg['target_ebops'] = TARGET_EBOPS

    # 应用 target overrides (fast decay v2)
    target_ov = get_target_overrides_v2(TARGET_EBOPS)
    cfg.update(target_ov)

    # 应用实验覆盖 (最高优先级)
    cfg.update(exp_def['overrides'])

    # 输出目录
    cfg['output_dir'] = os.path.join(OUTPUT_ROOT, exp_name)

    return cfg


def run_single(exp_name: str, dry_run: bool = False) -> dict | None:
    """运行单个消融实验。"""
    from run_all_v2 import run as run_pipeline

    exp_def = EXPERIMENTS[exp_name]
    cfg = build_config(exp_name)

    total = cfg['phase1_epochs'] + cfg['phase2_epochs']
    sdf_status = 'ENABLED' if cfg.get('soft_floor_b', 0) > 0 and cfg.get('soft_floor_every', 999999) < 999999 else 'DISABLED'

    print(f'\n{"#" * 72}')
    print(f'  Experiment {exp_name}: {exp_def["desc"]}')
    print(f'  Target: {TARGET_EBOPS} eBOPs  |  Total: {total} epochs (fast decay v2)')
    print(f'  SoftDeathFloor: {sdf_status}')
    if sdf_status == 'ENABLED':
        print(f'    b_floor={cfg["soft_floor_b"]}, alive={cfg["soft_floor_alive"]}, '
              f'every={cfg["soft_floor_every"]}')
    print(f'  Output: {cfg["output_dir"]}')
    print(f'{"#" * 72}')

    if dry_run:
        print('  [DRY RUN] Skipping...')
        relevant_keys = ['soft_floor_b', 'soft_floor_alive', 'soft_floor_every',
                         'phase1_epochs', 'phase2_epochs', 'budget_decay_epochs',
                         'warmup_ebops_mul', 'target_ebops']
        for k in relevant_keys:
            print(f'    {k}: {cfg.get(k)}')
        return None

    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 保存元数据
    meta = {
        'experiment': exp_name,
        'desc': exp_def['desc'],
        'target_ebops': TARGET_EBOPS,
        'total_epochs': total,
        'sdf_enabled': sdf_status == 'ENABLED',
        'soft_floor_b': cfg.get('soft_floor_b', 0),
        'soft_floor_alive': cfg.get('soft_floor_alive', 0.4),
        'soft_floor_every': cfg.get('soft_floor_every', 999999),
    }
    with open(os.path.join(cfg['output_dir'], 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    t0 = time.time()
    result = run_pipeline(cfg)
    elapsed = time.time() - t0

    result['experiment'] = exp_name
    result['desc'] = exp_def['desc']
    result['elapsed_total'] = elapsed

    return result


def print_comparison(results: list[dict]):
    """打印消融实验对比表。"""
    import h5py
    import numpy as np

    print(f'\n{"=" * 110}')
    print(f'  SoftDeathFloor ABLATION — target={TARGET_EBOPS} eBOPs (fast decay v2, {results[0].get("total_epochs", "?")} ep)')
    print(f'{"=" * 110}')
    print(f'  {"Exp":>4s}  {"Description":<32s}  {"b_floor":>7s}  {"alive":>6s}  {"every":>5s}  '
          f'{"Best@Tgt":>8s}  {"@Epoch":>6s}  {"Final":>7s}  {"eBOPs":>6s}  {"Time":>5s}  {"Δ D0":>7s}')
    print(f'  {"-" * 106}')

    # 找 D0 baseline
    d0_acc = None
    for r in results:
        if r.get('experiment') == 'D0':
            d0_acc = r.get('best_acc_at_target', r.get('best_val_acc', 0))
            break

    for r in sorted(results, key=lambda x: x.get('experiment', '')):
        exp_name = r.get('experiment', '?')
        desc = r.get('desc', '?')[:32]

        # 读取 trace 获取 best@target
        output_dir = os.path.join(OUTPUT_ROOT, exp_name)
        trace_path = os.path.join(output_dir, 'training_trace.h5')
        best_at_target = None
        best_ep = None
        if os.path.isfile(trace_path):
            with h5py.File(trace_path, 'r') as f:
                va = f['val_accuracy'][:]
                eb = f['ebops'][:]
                mask = (eb >= TARGET_EBOPS * 0.875) & (eb <= TARGET_EBOPS * 1.125)
                if mask.any():
                    idx = np.where(mask)[0][np.argmax(va[mask])]
                    best_at_target = va[idx]
                    best_ep = idx

        # 从 meta 读取 SDF 参数
        meta_path = os.path.join(output_dir, 'experiment_meta.json')
        b_floor = '—'
        alive = '—'
        every = '—'
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get('sdf_enabled'):
                b_floor = f'{meta["soft_floor_b"]:.2f}'
                alive = f'{meta["soft_floor_alive"]:.1f}'
                every = f'{meta["soft_floor_every"]}'
            else:
                b_floor = 'off'
                alive = '—'
                every = '—'

        acc_str = f'{best_at_target:.4f}' if best_at_target is not None else f'{r.get("best_val_acc", 0):.4f}'
        ep_str = f'{best_ep}' if best_ep is not None else f'{r.get("best_epoch", "?")}'
        final_acc = r.get('final_val_acc', 0)
        final_eb = r.get('final_ebops', 0)
        elapsed = r.get('elapsed_total', r.get('elapsed_sec', 0))

        acc_val = best_at_target if best_at_target is not None else r.get('best_val_acc', 0)
        delta = f'{acc_val - d0_acc:+.4f}' if d0_acc is not None else '—'

        print(f'  {exp_name:>4s}  {desc:<32s}  {b_floor:>7s}  {alive:>6s}  {every:>5s}  '
              f'{acc_str:>8s}  {ep_str:>6s}  {final_acc:7.4f}  {final_eb:6.0f}  {elapsed:5.0f}s  {delta:>7s}')

    print(f'{"=" * 110}')

    # 简要结论
    if d0_acc is not None:
        best_sdf = None
        best_sdf_acc = d0_acc
        for r in results:
            if r.get('experiment', '') == 'D0':
                continue
            output_dir = os.path.join(OUTPUT_ROOT, r.get('experiment', ''))
            trace_path = os.path.join(output_dir, 'training_trace.h5')
            if os.path.isfile(trace_path):
                with h5py.File(trace_path, 'r') as f:
                    va = f['val_accuracy'][:]
                    eb = f['ebops'][:]
                    mask = (eb >= TARGET_EBOPS * 0.875) & (eb <= TARGET_EBOPS * 1.125)
                    if mask.any():
                        acc = va[mask].max()
                        if acc > best_sdf_acc:
                            best_sdf_acc = acc
                            best_sdf = r.get('experiment')

        if best_sdf:
            print(f'\n  ★ Best SDF config: {best_sdf} ({best_sdf_acc:.4f}) vs D0 baseline ({d0_acc:.4f})'
                  f' → Δ = {best_sdf_acc - d0_acc:+.4f}')
        else:
            print(f'\n  ★ No SDF config beats the D0 baseline ({d0_acc:.4f})')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Experiment D: SoftDeathFloor ablation at 400 eBOPs (fast decay v2)')
    parser.add_argument('--run', nargs='+', default=None,
                        help='Experiment names to run (e.g. D0 D4, or "D" for all D*, or "all")')
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
        print(f'\n  SoftDeathFloor Ablation Experiments (target={TARGET_EBOPS} eBOPs):')
        print(f'  {"Name":>4s}  {"Description":<40s}  {"SDF":>4s}  {"b_floor":>7s}  {"alive":>6s}  {"every":>5s}')
        print(f'  {"-" * 75}')
        for name, exp in sorted(EXPERIMENTS.items()):
            ov = exp['overrides']
            enabled = ov.get('soft_floor_b', 0) > 0 and ov.get('soft_floor_every', 999999) < 999999
            b_floor = f'{ov["soft_floor_b"]:.2f}' if enabled else 'off'
            alive = f'{ov.get("soft_floor_alive", 0.4):.1f}' if enabled else '—'
            every = f'{ov.get("soft_floor_every", 50)}' if enabled else '—'
            print(f'  {name:>4s}  {exp["desc"]:<40s}  {"on" if enabled else "off":>4s}  '
                  f'{b_floor:>7s}  {alive:>6s}  {every:>5s}')
        return

    if args.compare_only:
        # 从现有结果加载
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
            elif token.upper() == 'D':
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
        print(f'\n  Use --list to see available experiments, --run D0 D4 to run specific ones.')
        return

    print(f'\n{"#" * 72}')
    print(f'  Experiment D: SoftDeathFloor Ablation')
    print(f'  Target: {TARGET_EBOPS} eBOPs  |  Pipeline: run_all_v2 (fast decay)')
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
        # 保存汇总
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        summary_path = os.path.join(OUTPUT_ROOT, 'all_results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Summary saved: {summary_path}')

        print_comparison(all_results)


if __name__ == '__main__':
    main()
