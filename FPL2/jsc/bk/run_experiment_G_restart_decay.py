#!/usr/bin/env python3
"""
run_experiment_G_restart_decay.py — restart_decay 精细消融 (400 eBOPs)
================================================================================

聚焦消融 BetaCurriculumController.restart_decay 参数。

─── restart_decay 语义 ───
每次 RECOVER 结束后:  beta_new = beta_at_stall × restart_decay
                      beta_new = max(beta_new, beta_min)

  decay=0.0  → beta 几乎重置为 beta_min (1e-8)，最激进重启
  decay=0.25 → v2 默认
  decay=1.0  → 重启后 beta 不变 (RECOVER 仍发生，但无压缩衰减)
  disabled   → 完全关闭 curriculum (无 RECOVER，无 RESTART)

注意: decay=0 ≠ disabled!
  - decay=0: 状态机仍工作 (检测停滞→RECOVER phase→RESTART)，
             只是 RESTART 后 beta 回到 beta_min
  - disabled: 完全没有停滞检测/RECOVER/RESTART

消融设计 (7 configs):
  ── 对照 ──
  D0: curriculum DISABLED (无停滞检测，纯 BetaOnlyBudgetController)
  D1: decay=0.00  (最激进: 每次重启 beta → beta_min)

  ── 梯度扫描 ──
  D2: decay=0.10  (激进)
  D3: decay=0.25  (v2 默认 ★)
  D4: decay=0.50  (温和)
  D5: decay=0.75  (保守)

  ── 上界 ──
  D6: decay=1.00  (重启不减 beta, 仅有 RECOVER 阶段的临时降压)

用法:
  python run_experiment_G_restart_decay.py --list
  python run_experiment_G_restart_decay.py --run D
  python run_experiment_G_restart_decay.py --run D0 D3
  python run_experiment_G_restart_decay.py --dry-run --run D
  python run_experiment_G_restart_decay.py --skip-existing --run D
  python run_experiment_G_restart_decay.py --compare-only

输出: results/experiment_G_restart_decay/<实验名>/
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


# ═══════════════════════════════════════════════════════════════════════════════
# 实验定义
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_ROOT = 'results/experiment_G_restart_decay'
TARGET_EBOPS = 400


def _curriculum(decay: float) -> dict:
    """Curriculum ON with specified restart_decay, other params = v2 default."""
    return dict(
        beta_curriculum_enabled   = True,
        beta_stall_patience       = 600,
        beta_recover_epochs       = 300,
        beta_restart_decay        = decay,
        beta_max_restarts         = 8,
        beta_recover_floor        = None,   # → beta_min
        beta_curriculum_p2        = True,
    )


EXPERIMENTS = {
    # ── D0: Curriculum disabled (真正的 baseline) ────────────────────────
    'D0': dict(
        desc='Curriculum DISABLED',
        overrides=dict(
            beta_curriculum_enabled = False,
            beta_curriculum_p2      = False,
        ),
    ),

    # ── D1: decay=0.00 (最激进: beta → beta_min) ────────────────────────
    'D1': dict(
        desc='decay=0.00 (most aggressive)',
        overrides=_curriculum(0.00),
    ),

    # ── D2: decay=0.10 ──────────────────────────────────────────────────
    'D2': dict(
        desc='decay=0.10 (aggressive)',
        overrides=_curriculum(0.10),
    ),

    # ── D3: decay=0.25 (v2 default) ────────────────────────────────────
    'D3': dict(
        desc='decay=0.25 (v2 default)',
        overrides=_curriculum(0.25),
    ),

    # ── D4: decay=0.50 ──────────────────────────────────────────────────
    'D4': dict(
        desc='decay=0.50 (moderate)',
        overrides=_curriculum(0.50),
    ),

    # ── D5: decay=0.75 ──────────────────────────────────────────────────
    'D5': dict(
        desc='decay=0.75 (conservative)',
        overrides=_curriculum(0.75),
    ),

    # ── D6: decay=1.00 (重启不减 beta) ──────────────────────────────────
    'D6': dict(
        desc='decay=1.00 (restart noop)',
        overrides=_curriculum(1.00),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 配置构建
# ═══════════════════════════════════════════════════════════════════════════════

def build_config(exp_name: str) -> dict:
    """合并 run_all_v2 DEFAULT_CONFIG + target overrides + 实验覆盖。"""
    from run_all_v2 import DEFAULT_CONFIG, get_target_overrides_v2

    exp_def = EXPERIMENTS[exp_name]
    cfg = dict(DEFAULT_CONFIG)
    cfg['pretrained'] = 'pretrained_weight/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras'
    cfg['target_ebops'] = TARGET_EBOPS

    target_ov = get_target_overrides_v2(TARGET_EBOPS)
    cfg.update(target_ov)

    cfg.update(exp_def['overrides'])

    cfg['output_dir'] = os.path.join(OUTPUT_ROOT, exp_name)
    return cfg


def _format_config(cfg: dict) -> str:
    """单行格式化 curriculum 配置。"""
    if not cfg.get('beta_curriculum_enabled', False):
        return 'DISABLED'
    decay = cfg.get('beta_restart_decay', '?')
    return (f'decay={decay}, pat={cfg.get("beta_stall_patience", 600)}, '
            f'rec={cfg.get("beta_recover_epochs", 300)}, max={cfg.get("beta_max_restarts", 8)}')


# ═══════════════════════════════════════════════════════════════════════════════
# 运行逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def run_single(exp_name: str, dry_run: bool = False) -> dict | None:
    exp_def = EXPERIMENTS[exp_name]
    cfg = build_config(exp_name)
    total = cfg['phase1_epochs'] + cfg['phase2_epochs']
    cfg_str = _format_config(cfg)

    print(f'\n{"#" * 72}')
    print(f'  {exp_name}: {exp_def["desc"]}')
    print(f'  Target: {TARGET_EBOPS} eBOPs  |  {total} epochs  |  {cfg_str}')
    print(f'  Output: {cfg["output_dir"]}')
    print(f'{"#" * 72}')

    if dry_run:
        print('  [DRY RUN]')
        for k in ['beta_curriculum_enabled', 'beta_restart_decay',
                   'beta_stall_patience', 'beta_recover_epochs',
                   'beta_max_restarts', 'beta_recover_floor',
                   'beta_curriculum_p2',
                   'phase1_epochs', 'phase2_epochs', 'target_ebops',
                   'phase1_beta_min', 'phase1_beta_max']:
            if k in cfg:
                print(f'    {k}: {cfg[k]}')
        return None

    os.makedirs(cfg['output_dir'], exist_ok=True)

    meta = {
        'experiment': exp_name,
        'desc': exp_def['desc'],
        'target_ebops': TARGET_EBOPS,
        'total_epochs': total,
        'beta_curriculum_enabled': cfg.get('beta_curriculum_enabled', False),
        'beta_restart_decay': cfg.get('beta_restart_decay'),
    }
    with open(os.path.join(cfg['output_dir'], 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    from run_all_v2 import run as run_v2
    t0 = time.time()
    result = run_v2(cfg)
    elapsed = time.time() - t0
    result['experiment'] = exp_name
    result['desc'] = exp_def['desc']
    result['elapsed_total'] = elapsed
    result['restart_decay'] = cfg.get('beta_restart_decay')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 对比表
# ═══════════════════════════════════════════════════════════════════════════════

def _read_trace_best(output_dir: str):
    import h5py, numpy as np
    trace = os.path.join(output_dir, 'training_trace.h5')
    if not os.path.isfile(trace):
        return None, None
    try:
        with h5py.File(trace, 'r') as f:
            va, eb = f['val_accuracy'][:], f['ebops'][:]
            mask = (eb >= TARGET_EBOPS * 0.875) & (eb <= TARGET_EBOPS * 1.125)
            if mask.any():
                idx = np.where(mask)[0][np.argmax(va[mask])]
                return float(va[idx]), int(idx)
    except Exception:
        pass
    return None, None


def _read_restart_count(output_dir: str) -> int | None:
    import h5py
    trace = os.path.join(output_dir, 'training_trace.h5')
    if not os.path.isfile(trace):
        return None
    try:
        with h5py.File(trace, 'r') as f:
            if 'beta_curriculum_restarts' in f:
                arr = f['beta_curriculum_restarts'][:]
                return int(arr[-1]) if len(arr) > 0 else None
    except Exception:
        pass
    return None


def _read_meta(output_dir: str) -> dict:
    p = os.path.join(output_dir, 'experiment_meta.json')
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}


def print_comparison(results: list[dict]):
    import numpy as np

    print(f'\n{"=" * 115}')
    print(f'  restart_decay ABLATION — target={TARGET_EBOPS} eBOPs')
    print(f'  beta_new = beta_at_stall × decay  (clamped to beta_min=1e-8)')
    print(f'{"=" * 115}')
    print(f'  {"Exp":>3s}  {"decay":>5s}  {"Description":<28s}  '
          f'{"Best@Tgt":>8s}  {"@Ep":>5s}  {"Final":>7s}  {"eBOPs":>6s}  '
          f'{"#Rst":>4s}  {"Time":>5s}  {"Δ D0":>7s}  {"Δ D3":>7s}')
    print(f'  {"-" * 111}')

    # 找 D0 (disabled) 和 D3 (default) 的 acc
    ref = {}
    for r in results:
        exp = r.get('experiment', '')
        d = os.path.join(OUTPUT_ROOT, exp)
        acc, _ = _read_trace_best(d)
        if acc is None:
            acc = r.get('best_acc_at_target', r.get('best_val_acc', 0))
        ref[exp] = acc
    d0_acc = ref.get('D0')
    d3_acc = ref.get('D3')

    for r in sorted(results, key=lambda x: x.get('experiment', '')):
        exp = r.get('experiment', '?')
        desc = r.get('desc', '?')[:28]
        d = os.path.join(OUTPUT_ROOT, exp)

        best, ep = _read_trace_best(d)
        acc = best if best is not None else r.get('best_acc_at_target', r.get('best_val_acc', 0))
        acc_s = f'{acc:.4f}' if acc else '—'
        ep_s = f'{ep}' if ep is not None else '?'

        ov = EXPERIMENTS.get(exp, {}).get('overrides', {})
        decay_val = ov.get('beta_restart_decay')
        decay_s = f'{decay_val:.2f}' if decay_val is not None else 'OFF'

        n_rst = _read_restart_count(d)
        rst_s = f'{n_rst}' if n_rst is not None else '—'

        final_acc = r.get('final_val_acc', 0)
        final_eb = r.get('final_ebops', 0)
        elapsed = r.get('elapsed_total', r.get('elapsed_sec', 0))

        dd0 = f'{acc - d0_acc:+.4f}' if d0_acc is not None and acc else '—'
        dd3 = f'{acc - d3_acc:+.4f}' if d3_acc is not None and acc else '—'

        print(f'  {exp:>3s}  {decay_s:>5s}  {desc:<28s}  '
              f'{acc_s:>8s}  {ep_s:>5s}  {final_acc:7.4f}  {final_eb:6.0f}  '
              f'{rst_s:>4s}  {elapsed:5.0f}s  {dd0:>7s}  {dd3:>7s}')

    print(f'{"=" * 115}')

    if d0_acc is not None and d3_acc is not None:
        print(f'\n  D0 (no curriculum) = {d0_acc:.4f}  |  D3 (default, decay=0.25) = {d3_acc:.4f}  |  Δ = {d3_acc - d0_acc:+.4f}')

    # 找最佳
    best_exp, best_acc = None, -1.0
    for exp, acc in ref.items():
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_exp = exp
    if best_exp:
        meta = _read_meta(os.path.join(OUTPUT_ROOT, best_exp))
        print(f'  ★ Best: {best_exp} ({best_acc:.4f}) — {meta.get("desc", EXPERIMENTS.get(best_exp, {}).get("desc", "?"))}')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='restart_decay ablation at 400 eBOPs')
    parser.add_argument('--run', nargs='+', default=None,
                        help='Experiments to run (D0..D6, "D" or "all" for all)')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--compare-only', action='store_true')
    args = parser.parse_args()

    if args.list:
        print(f'\n  restart_decay Ablation (target={TARGET_EBOPS} eBOPs):')
        print(f'  {"Name":>4s}  {"decay":>5s}  {"Description":<28s}  {"Curriculum":>10s}')
        print(f'  {"-" * 55}')
        for name, exp in sorted(EXPERIMENTS.items()):
            ov = exp['overrides']
            en = ov.get('beta_curriculum_enabled', True)
            dec = ov.get('beta_restart_decay')
            dec_s = f'{dec:.2f}' if dec is not None else '—'
            en_s = '✓' if en else '✗'
            print(f'  {name:>4s}  {dec_s:>5s}  {exp["desc"]:<28s}  {en_s:>10s}')
        return

    if args.compare_only:
        results = []
        for name in sorted(EXPERIMENTS.keys()):
            rp = os.path.join(OUTPUT_ROOT, name, 'result_summary.json')
            if os.path.isfile(rp):
                with open(rp) as f:
                    r = json.load(f)
                r['experiment'] = name
                r['desc'] = EXPERIMENTS[name]['desc']
                results.append(r)
        if results:
            print_comparison(results)
        else:
            print('  No results found.')
        return

    if args.run_all:
        exp_names = sorted(EXPERIMENTS.keys())
    elif args.run:
        exp_names = []
        for token in args.run:
            if token.lower() in ('all', 'd'):
                exp_names = sorted(EXPERIMENTS.keys())
                break
            name = token.upper()
            if name not in EXPERIMENTS:
                print(f'[ERROR] Unknown: {name}. Available: {sorted(EXPERIMENTS.keys())}')
                sys.exit(1)
            exp_names.append(name)
    else:
        parser.print_help()
        print(f'\n  --list to see configs, --run D to run all, --compare-only for results.')
        return

    print(f'\n{"#" * 72}')
    print(f'  restart_decay Ablation @ {TARGET_EBOPS} eBOPs')
    print(f'  decay values: OFF, 0.00, 0.10, 0.25★, 0.50, 0.75, 1.00')
    print(f'  Running: {exp_names}')
    print(f'{"#" * 72}')

    all_results = []
    for name in exp_names:
        rp = os.path.join(OUTPUT_ROOT, name, 'result_summary.json')
        if args.skip_existing and os.path.isfile(rp):
            print(f'\n  [Skip] {name}: result exists')
            with open(rp) as f:
                r = json.load(f)
            r['experiment'] = name
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
            for r in all_results:
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
