#!/usr/bin/env python3
"""
run_experiment_C_pruning_methods.py — 剪枝方法消融实验
================================================================================

Experiment C: 对比 8 种剪枝方法 × 5 个 eBOPs 目标的训练后精度。

基于 run_all_v2 管线，使用 v2 的默认 decay 配置。
所有方法使用相同的训练配置 (Phase 1 + Phase 2)，仅剪枝方法不同。

─── 剪枝方法 (均来自 utils/pruning.py) ───
  C_random_init : Random Init — 随机重初始化权重后按比例剪枝 (最弱 baseline)
  C_random      : Random      — 随机剪枝 (无谱约束, 无连通保证)
  C_magnitude   : Magnitude   — 全局权重幅值 top-k 剪枝 (无谱约束)
  C_sensitivity : Sensitivity — 按层敏感度分配剪枝预算
  C_synflow     : SynFlow     — 数据无关迭代剪枝 (Tanaka et al., NeurIPS 2020)
  C_grasp       : GraSP       — 梯度信号保持剪枝 (Wang et al., ICLR 2020)
  C_snip        : SNIP        — 单次连接敏感度 (Lee et al., ICLR 2019)
  C_spectral    : Spectral ★  — 谱约束剪枝 (本项目, 作为参考)

─── eBOPs 目标 ───
  400, 1551, 2585, 6839, 11718
python run_experiment_C_pruning_methods.py --run C_spectral --ebops 400
─── 评估指标 ───
  Max accuracy @ target eBOPs ±12.5% 范围 (训练后)
  python run_experiment_C_pruning_methods.py --run C_spectral --ebops 400
  python run_experiment_C_pruning_methods.py --skip-existing --run C
  python run_experiment_C_pruning_methods.py --skip-existing --run C --ebops 400
用法:
  python run_experiment_C_pruning_methods.py --list
  python run_experiment_C_pruning_methods.py --run C
  python run_experiment_C_pruning_methods.py --run C_magnitude C_sensitivity
  python run_experiment_C_pruning_methods.py --ebops 400
  python run_experiment_C_pruning_methods.py --dry-run --run C
  python run_experiment_C_pruning_methods.py --skip-existing --run C
  python run_experiment_C_pruning_methods.py --compare-only
  python run_experiment_C_pruning_methods.py --run C_spectral --ebops 400
输出: results/experiment_C_pruning_methods/<method>_ebops<target>/
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 从 run_all_v2 继承核心配置和运行逻辑
from run_all_v2 import DEFAULT_CONFIG, get_target_overrides_v2, run as run_v2


# ═══════════════════════════════════════════════════════════════════════════════
# 实验定义
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_ROOT = 'results/experiment_C_pruning_methods'

TARGET_EBOPS_LIST = [400, 1551, 2585, 6839, 11718]

# 所有剪枝方法均来自 utils/pruning.py 的实现
PRUNING_METHODS = {
    # 'random_init': dict(
    #     desc='Random Init (reinit + proportional prune)',
    #     pruning_method='random_init',
    # ),
    'random': dict(
        desc='Random (random prune, no spectral)',
        pruning_method='random',
    ),
    'magnitude': dict(
        desc='Magnitude (global top-k by |w|)',
        pruning_method='magnitude',
    ),
    'sensitivity': dict(
        desc='Sensitivity (per-layer budget allocation)',
        pruning_method='sensitivity',
    ),
    'synflow': dict(
        desc='SynFlow (data-free iterative)',
        pruning_method='synflow',
    ),
    'grasp': dict(
        desc='GraSP (gradient signal preservation)',
        pruning_method='grasp',
    ),
    'snip': dict(
        desc='SNIP (connection sensitivity)',
        pruning_method='snip',
    ),
    # 'spectral': dict(
    #     desc='Spectral-Quant (proposed, reference)',
    #     pruning_method='auto',  # auto → spectral in run_all_v2
    # ),
    # 'spectral_sensitivity': dict(
    #     desc='Spectral + Sensitivity (hybrid)',
    #     pruning_method='spectral_sensitivity',
    # ),
    # 'spectral_sensitivity_nokill': dict(
    #     desc='Spectral + Sensitivity (no bisect kill)',
    #     pruning_method='spectral_sensitivity',
    #     cfg_overrides=dict(
    #         spectral_sens_allow_connection_kill=False,
    #     ),
    # ),
    # 'spectral_sensitivity_nokill_pp03': dict(
    #     desc='Spectral + Sensitivity (no kill, pp=0.3)',
    #     pruning_method='spectral_sensitivity',
    #     cfg_overrides=dict(
    #         spectral_sens_allow_connection_kill=False,
    #         spectral_sens_protect_power=0.3,
    #     ),
    # ),
    # 'spectral_sensitivity_nokill_pp07': dict(
    #     desc='Spectral + Sensitivity (no kill, pp=0.7)',
    #     pruning_method='spectral_sensitivity',
    #     cfg_overrides=dict(
    #         spectral_sens_allow_connection_kill=False,
    #         spectral_sens_protect_power=0.7,
    #     ),
    # ),
}


def _build_experiments():
    """生成完整实验名 → (method, target_ebops) 映射。"""
    exps = {}
    for method_key, method_def in PRUNING_METHODS.items():
        for target in TARGET_EBOPS_LIST:
            exp_name = f'C_{method_key}_ebops{target}'
            exps[exp_name] = dict(
                method_key=method_key,
                desc=f'{method_def["desc"]} @ {target} eBOPs',
                target_ebops=target,
                pruning_method=method_def['pruning_method'],
                cfg_overrides=method_def.get('cfg_overrides', {}),
            )
    return exps

EXPERIMENTS = _build_experiments()


# ═══════════════════════════════════════════════════════════════════════════════
# 配置构建
# ═══════════════════════════════════════════════════════════════════════════════

def build_config(exp_name: str) -> dict:
    """合并 run_all_v2 DEFAULT_CONFIG + target overrides + 实验覆盖 → 完整配置。"""
    exp_def = EXPERIMENTS[exp_name]
    cfg = dict(DEFAULT_CONFIG)
    cfg['pretrained'] = 'pretrained_weight/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras'
    cfg['target_ebops'] = exp_def['target_ebops']

    # 应用 target overrides (使用 v2 默认 decay，不做额外修改)
    target_ov = get_target_overrides_v2(exp_def['target_ebops'])
    cfg.update(target_ov)

    # 设置剪枝方法
    cfg['pruning_method'] = exp_def['pruning_method']
    if exp_def.get('cfg_overrides'):
        cfg.update(exp_def['cfg_overrides'])

    # ★ Explicitly control pruning method for each case
    # sensitivity: uses SensitivityAwarePruner
    # auto (spectral): uses spectral_quant_prune_to_ebops (default)
    # others (random, magnitude, etc.): use their dedicated methods
    if exp_def['pruning_method'] == 'sensitivity':
        cfg['use_sensitivity_pruner'] = True
    else:
        # Force spectral-quant for 'auto' and explicit pruning methods
        cfg['use_sensitivity_pruner'] = False

    cfg['output_dir'] = os.path.join(OUTPUT_ROOT, exp_name)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 运行逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def run_single(exp_name: str, dry_run: bool = False) -> dict | None:
    exp_def = EXPERIMENTS[exp_name]
    cfg = build_config(exp_name)
    total = cfg['phase1_epochs'] + cfg['phase2_epochs']

    print(f'\n{"#" * 72}')
    print(f'  {exp_name}: {exp_def["desc"]}')
    print(f'  Target: {exp_def["target_ebops"]} eBOPs  |  {total} epochs')
    print(f'  Pruning: {exp_def["pruning_method"]}  |  decay={cfg.get("beta_restart_decay", "default")}')
    print(f'  Output: {cfg["output_dir"]}')
    print(f'{"#" * 72}')

    if dry_run:
        print('  [DRY RUN]')
        for k in ['pruning_method', 'target_ebops', 'beta_restart_decay',
                   'phase1_epochs', 'phase2_epochs', 'warmup_ebops_mul',
                   'budget_decay_epochs', 'use_sensitivity_pruner']:
            if k in cfg:
                print(f'    {k}: {cfg[k]}')
        return None

    os.makedirs(cfg['output_dir'], exist_ok=True)

    meta = {
        'experiment': exp_name,
        'desc': exp_def['desc'],
        'method_key': exp_def['method_key'],
        'pruning_method': exp_def['pruning_method'],
        'target_ebops': exp_def['target_ebops'],
        'total_epochs': total,
        'beta_restart_decay': cfg.get('beta_restart_decay'),
    }
    with open(os.path.join(cfg['output_dir'], 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    # 调用 run_all_v2.run()
    t0 = time.time()
    result = run_v2(cfg)
    elapsed = time.time() - t0
    result['experiment'] = exp_name
    result['desc'] = exp_def['desc']
    result['method_key'] = exp_def['method_key']
    result['elapsed_total'] = elapsed
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 对比表
# ═══════════════════════════════════════════════════════════════════════════════

def _read_trace_best(output_dir: str, target_ebops: float):
    """从 training_trace.h5 读取 target eBOPs ±12.5% 范围内的最佳精度。"""
    import h5py, numpy as np
    trace = os.path.join(output_dir, 'training_trace.h5')
    if not os.path.isfile(trace):
        return None, None, None
    try:
        with h5py.File(trace, 'r') as f:
            va, eb = f['val_accuracy'][:], f['ebops'][:]
            mask = (eb >= target_ebops * 0.875) & (eb <= target_ebops * 1.125)
            if mask.any():
                idx = np.where(mask)[0][np.argmax(va[mask])]
                return float(va[idx]), int(idx), float(eb[idx])
    except Exception:
        pass
    return None, None, None


def _read_meta(output_dir: str) -> dict:
    p = os.path.join(output_dir, 'experiment_meta.json')
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}


def print_comparison(results: list[dict]):
    import numpy as np

    # ── 按 eBOPs 分组显示 ──
    for target in TARGET_EBOPS_LIST:
        target_results = [r for r in results if r.get('target_ebops') == target
                          or (isinstance(r.get('experiment', ''), str) and f'ebops{target}' in r['experiment'])]
        if not target_results:
            continue

        print(f'\n{"=" * 100}')
        print(f'  Experiment C — Pruning Method Comparison @ {target} eBOPs')
        print(f'{"=" * 100}')
        print(f'  {"Method":<15s}  {"Pruning":<14s}  '
              f'{"Best@Tgt":>8s}  {"@Ep":>5s}  {"@eBOPs":>7s}  '
              f'{"Final":>7s}  {"FinalEB":>7s}  {"Time":>5s}')
        print(f'  {"-" * 96}')

        # 读取 spectral 作为参考
        spectral_acc = None
        for r in target_results:
            mk = r.get('method_key', '')
            if mk == 'spectral':
                d = os.path.join(OUTPUT_ROOT, r.get('experiment', ''))
                acc, _, _ = _read_trace_best(d, target)
                if acc is None:
                    acc = r.get('best_acc_at_target', r.get('best_val_acc', 0))
                spectral_acc = acc

        for r in sorted(target_results, key=lambda x: x.get('method_key', '')):
            exp = r.get('experiment', '?')
            mk = r.get('method_key', '?')
            desc_short = PRUNING_METHODS.get(mk, {}).get('desc', '?')[:14]
            d = os.path.join(OUTPUT_ROOT, exp)

            best, ep, best_eb = _read_trace_best(d, target)
            if best is None:
                best = r.get('best_acc_at_target', r.get('best_val_acc', 0))
                ep = r.get('best_epoch_at_target', r.get('best_epoch', 0))
                best_eb = r.get('best_ebops_at_target', 0)

            acc_s = f'{best:.4f}' if best else '—'
            ep_s = f'{ep}' if ep is not None else '?'
            eb_s = f'{best_eb:.0f}' if best_eb else '?'

            final_acc = r.get('final_val_acc', 0)
            final_eb = r.get('final_ebops', 0)
            elapsed = r.get('elapsed_total', r.get('elapsed_sec', 0))

            delta_s = ''
            if spectral_acc is not None and best:
                delta = best - spectral_acc
                delta_s = f'  Δspec={delta:+.4f}'

            print(f'  {mk:<15s}  {desc_short:<14s}  '
                  f'{acc_s:>8s}  {ep_s:>5s}  {eb_s:>7s}  '
                  f'{final_acc:7.4f}  {final_eb:7.0f}  {elapsed:5.0f}s{delta_s}')

    # ── 汇总表 ──
    print(f'\n\n{"=" * 120}')
    print(f'  SUMMARY: Max Accuracy @ Target eBOPs (±12.5%)')
    print(f'{"=" * 120}')
    header = f'  {"Method":<15s}'
    for target in TARGET_EBOPS_LIST:
        header += f'  {target:>7d}'
    print(header)
    print(f'  {"-" * (15 + 9 * len(TARGET_EBOPS_LIST))}')

    for mk in PRUNING_METHODS.keys():
        row = f'  {mk:<15s}'
        for target in TARGET_EBOPS_LIST:
            exp_name = f'C_{mk}_ebops{target}'
            d = os.path.join(OUTPUT_ROOT, exp_name)
            acc, _, _ = _read_trace_best(d, target)
            if acc is None:
                # Try from results list
                for r in results:
                    if r.get('experiment') == exp_name:
                        acc = r.get('best_acc_at_target', r.get('best_val_acc'))
                        break
            row += f'  {acc:7.4f}' if acc else f'  {"—":>7s}'
        print(row)
    print(f'{"=" * 120}')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Experiment C: Pruning method comparison across eBOPs targets')
    parser.add_argument('--run', nargs='+', default=None,
                        help='Experiments to run. "C" for all, or specific names like '
                             '"C_magnitude_ebops400", or method names like "magnitude sensitivity"')
    parser.add_argument('--ebops', type=int, nargs='+', default=None,
                        help='Only run specified eBOPs targets (default: all)')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Only run specified methods (default: all)')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--compare-only', action='store_true')
    args = parser.parse_args()

    if args.list:
        print(f'\n  Experiment C: Pruning Methods × eBOPs Targets')
        print(f'  {"=" * 80}')
        print(f'\n  Methods (all from utils/pruning.py):')
        for mk, md in PRUNING_METHODS.items():
            print(f'    {mk:<15s}  {md["desc"]}')
        print(f'\n  eBOPs targets: {TARGET_EBOPS_LIST}')
        print(f'\n  Total combinations: {len(EXPERIMENTS)}')
        print(f'\n  Experiment list:')
        for name in sorted(EXPERIMENTS.keys()):
            ed = EXPERIMENTS[name]
            print(f'    {name:<35s}  {ed["desc"]}')
        return

    if args.compare_only:
        results = []
        for name in sorted(EXPERIMENTS.keys()):
            rp = os.path.join(OUTPUT_ROOT, name, 'result_summary.json')
            if os.path.isfile(rp):
                with open(rp) as f:
                    r = json.load(f)
                r['experiment'] = name
                r['method_key'] = EXPERIMENTS[name]['method_key']
                r['desc'] = EXPERIMENTS[name]['desc']
                results.append(r)
        if results:
            print_comparison(results)
        else:
            print('  No results found.')
        return

    # ── 确定要运行的实验 ──
    ebops_filter = set(args.ebops) if args.ebops else set(TARGET_EBOPS_LIST)
    methods_filter = set(args.methods) if args.methods else None

    if args.run:
        exp_names = []
        for token in args.run:
            token_lower = token.lower()
            if token_lower in ('all', 'c'):
                exp_names = sorted(EXPERIMENTS.keys())
                break
            elif token_lower in PRUNING_METHODS:
                for target in TARGET_EBOPS_LIST:
                    if target in ebops_filter:
                        exp_names.append(f'C_{token_lower}_ebops{target}')
            elif token in EXPERIMENTS:
                exp_names.append(token)
            else:
                matched = [n for n in EXPERIMENTS if token_lower in n.lower()]
                if matched:
                    exp_names.extend(matched)
                else:
                    print(f'[ERROR] Unknown: {token}. Use --list to see available.')
                    sys.exit(1)
    elif args.ebops or args.methods:
        # --ebops / --methods 单独使用时，隐含 --run C (运行所有匹配的实验)
        exp_names = sorted(EXPERIMENTS.keys())
    else:
        parser.print_help()
        print(f'\n  --list to see configs, --run C to run all, --compare-only for results.')
        return

    # 应用过滤器
    if methods_filter:
        exp_names = [n for n in exp_names
                     if EXPERIMENTS[n]['method_key'] in methods_filter]
    exp_names = [n for n in exp_names
                 if EXPERIMENTS[n]['target_ebops'] in ebops_filter]

    # 去重保序
    seen = set()
    unique_names = []
    for n in exp_names:
        if n not in seen:
            seen.add(n)
            unique_names.append(n)
    exp_names = unique_names

    if not exp_names:
        print('[ERROR] No experiments to run after filtering.')
        return

    print(f'\n{"#" * 72}')
    print(f'  Experiment C: Pruning Method Comparison (based on run_all_v2)')
    print(f'  Methods: {sorted(set(EXPERIMENTS[n]["method_key"] for n in exp_names))}')
    print(f'  eBOPs: {sorted(set(EXPERIMENTS[n]["target_ebops"] for n in exp_names))}')
    print(f'  Total runs: {len(exp_names)}')
    print(f'  Decay: using run_all_v2 defaults')
    print(f'{"#" * 72}')

    all_results = []
    for i, name in enumerate(exp_names):
        rp = os.path.join(OUTPUT_ROOT, name, 'result_summary.json')
        if args.skip_existing and os.path.isfile(rp):
            print(f'\n  [Skip] {name}: result exists')
            with open(rp) as f:
                r = json.load(f)
            r['experiment'] = name
            r['method_key'] = EXPERIMENTS[name]['method_key']
            r['desc'] = EXPERIMENTS[name]['desc']
            all_results.append(r)
            continue

        if args.dry_run:
            run_single(name, dry_run=True)
            continue

        result = run_single(name)
        if result:
            all_results.append(result)
            print(f'\n  ── Progress ({len(all_results)}/{len(exp_names)}) ──')
            for r in all_results[-5:]:
                print(f'    {r["experiment"]}: best={r.get("best_val_acc", 0):.4f}  '
                      f'@target={r.get("best_acc_at_target", 0):.4f}  '
                      f'time={r.get("elapsed_total", 0):.0f}s')

    if all_results and not args.dry_run:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        with open(os.path.join(OUTPUT_ROOT, 'all_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print_comparison(all_results)


if __name__ == '__main__':
    main()
