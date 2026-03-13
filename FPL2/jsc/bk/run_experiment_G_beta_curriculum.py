#!/usr/bin/env python3
"""
run_experiment_G_beta_curriculum.py — Beta 课程重启消融实验 (400 eBOPs)
================================================================================

在 run_all_v2.py 管线上，消融 BetaCurriculumController 的各个维度。

─── Beta 课程重启机制 ───
BetaCurriculumController 是一个状态机，解决 beta↑ → b_k↓ → STE 失效 → acc 停 的死锁:
  COMPRESS → 正常运行 (BetaOnlyBudgetController 控制 beta)
  RECOVER  → val_acc 停滞 stall_patience 个 epoch → 将 beta 降至 recover_beta_floor
  RESTART  → recover_epochs 后 beta = beta_at_stall × restart_decay，重新开始压缩

关键参数:
  stall_patience   : val_acc 停滞多久触发 RECOVER (默认 600)
  recover_epochs   : RECOVER 阶段持续多少 epoch (默认 300)
  restart_decay    : 重启 beta 相对停滞时的缩放 (默认 0.25)
  max_restarts     : 最大重启次数 (默认 8)
  recover_beta_floor : RECOVER 期间 beta 底值 (默认 None → beta_min)

消融维度:
  ── 对照组 ──
  G0: Baseline (curriculum enabled, v2 default @ 400 eBOPs)
  G1: Curriculum DISABLED (纯 BetaOnlyBudgetController)

  ── stall_patience ──
  G2: patience=300  (更敏感，更早触发重启)
  G3: patience=1000 (更耐心，减少假阳性重启)

  ── recover_epochs ──
  G4: recover=100  (更短恢复期，快速重启)
  G5: recover=600  (更长恢复期，acc 充分恢复)

  ── restart_decay ──
  G6: decay=0.10  (更激进衰减，每次重启 beta 降更多)
  G7: decay=0.50  (更保守衰减，beta 下降幅度小)

  ── max_restarts ──
  G8:  max_restarts=2  (极少重启机会)
  G9:  max_restarts=16 (更多重启余量)

  ── recover_beta_floor ──
  G10: floor=0       (完全归零，acc 最大恢复但 eBOPs 可能爆炸)
  G11: floor=1e-5    (维持微弱压力，平衡恢复与预算控制)

  ── Phase 2 curriculum ──
  G12: P2 curriculum disabled (仅 Phase 1 使用 curriculum)

用法:
  python run_experiment_G_beta_curriculum.py --list
  python run_experiment_G_beta_curriculum.py --run G0 G1
  python run_experiment_G_beta_curriculum.py --run G
  python run_experiment_G_beta_curriculum.py --run-all
  python run_experiment_G_beta_curriculum.py --dry-run
  python run_experiment_G_beta_curriculum.py --skip-existing
  python run_experiment_G_beta_curriculum.py --compare-only

输出: results/experiment_G_beta_curriculum/<实验名>/
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

OUTPUT_ROOT = 'results/experiment_G_beta_curriculum'
TARGET_EBOPS = 400


def _default_curriculum():
    """v2 @ 400 eBOPs 默认 curriculum 参数。"""
    return dict(
        beta_curriculum_enabled   = True,
        beta_stall_patience       = 600,
        beta_recover_epochs       = 300,
        beta_restart_decay        = 0.25,
        beta_max_restarts         = 8,
        beta_recover_floor        = None,       # → beta_min
        beta_curriculum_p2        = True,       # Phase 2 也启用
    )


def _no_curriculum():
    """完全禁用 curriculum。"""
    return dict(
        beta_curriculum_enabled   = False,
        beta_curriculum_p2        = False,
    )


EXPERIMENTS = {
    # ── G0: Baseline (curriculum enabled, v2 default) ─────────────────────
    'G0': dict(
        desc='Baseline (curriculum ON, v2 default)',
        overrides=_default_curriculum(),
    ),

    # ── G1: Curriculum disabled ───────────────────────────────────────────
    'G1': dict(
        desc='Curriculum DISABLED',
        overrides=_no_curriculum(),
    ),

    # ── G2-G3: stall_patience ─────────────────────────────────────────────
    'G2': dict(
        desc='patience=300 (sensitive)',
        overrides={**_default_curriculum(), 'beta_stall_patience': 300},
    ),
    'G3': dict(
        desc='patience=1000 (patient)',
        overrides={**_default_curriculum(), 'beta_stall_patience': 1000},
    ),

    # ── G4-G5: recover_epochs ─────────────────────────────────────────────
    'G4': dict(
        desc='recover=100 (short)',
        overrides={**_default_curriculum(), 'beta_recover_epochs': 100},
    ),
    'G5': dict(
        desc='recover=600 (long)',
        overrides={**_default_curriculum(), 'beta_recover_epochs': 600},
    ),

    # ── G6-G7: restart_decay ─────────────────────────────────────────────
    'G6': dict(
        desc='decay=0.10 (aggressive)',
        overrides={**_default_curriculum(), 'beta_restart_decay': 0.10},
    ),
    'G7': dict(
        desc='decay=0.50 (conservative)',
        overrides={**_default_curriculum(), 'beta_restart_decay': 0.50},
    ),

    # ── G8-G9: max_restarts ──────────────────────────────────────────────
    'G8': dict(
        desc='max_restarts=2 (few)',
        overrides={**_default_curriculum(), 'beta_max_restarts': 2},
    ),
    'G9': dict(
        desc='max_restarts=16 (many)',
        overrides={**_default_curriculum(), 'beta_max_restarts': 16},
    ),

    # ── G10-G11: recover_beta_floor ──────────────────────────────────────
    'G10': dict(
        desc='floor=0 (full zero)',
        overrides={**_default_curriculum(), 'beta_recover_floor': 0.0},
    ),
    'G11': dict(
        desc='floor=1e-5 (mild pressure)',
        overrides={**_default_curriculum(), 'beta_recover_floor': 1e-5},
    ),

    # ── G12: Phase 2 curriculum disabled ─────────────────────────────────
    'G12': dict(
        desc='P2 curriculum disabled',
        overrides={**_default_curriculum(), 'beta_curriculum_p2': False},
    ),
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

    # 应用 target overrides (v2)
    target_ov = get_target_overrides_v2(TARGET_EBOPS)
    cfg.update(target_ov)

    # 应用实验覆盖 (最高优先级)
    cfg.update(exp_def['overrides'])

    # 输出目录
    cfg['output_dir'] = os.path.join(OUTPUT_ROOT, exp_name)

    return cfg


def _format_curriculum(cfg: dict) -> str:
    """格式化当前 curriculum 设置。"""
    if not cfg.get('beta_curriculum_enabled', False):
        return 'DISABLED'
    parts = [
        f'pat={cfg.get("beta_stall_patience", 600)}',
        f'rec={cfg.get("beta_recover_epochs", 300)}',
        f'dec={cfg.get("beta_restart_decay", 0.25):.2f}',
        f'max={cfg.get("beta_max_restarts", 8)}',
    ]
    floor = cfg.get('beta_recover_floor')
    if floor is not None:
        parts.append(f'floor={floor:.0e}')
    if not cfg.get('beta_curriculum_p2', True):
        parts.append('P2=OFF')
    return ', '.join(parts)


def run_single(exp_name: str, dry_run: bool = False) -> dict | None:
    """运行单个消融实验。"""
    exp_def = EXPERIMENTS[exp_name]
    cfg = build_config(exp_name)

    total = cfg['phase1_epochs'] + cfg['phase2_epochs']
    cur_str = _format_curriculum(cfg)

    print(f'\n{"#" * 72}')
    print(f'  Experiment {exp_name}: {exp_def["desc"]}')
    print(f'  Target: {TARGET_EBOPS} eBOPs  |  Total: {total} epochs')
    print(f'  Curriculum: {cur_str}')
    print(f'  Output: {cfg["output_dir"]}')
    print(f'{"#" * 72}')

    if dry_run:
        print('  [DRY RUN] Skipping...')
        relevant_keys = [
            'beta_curriculum_enabled', 'beta_stall_patience',
            'beta_recover_epochs', 'beta_restart_decay',
            'beta_max_restarts', 'beta_recover_floor',
            'beta_curriculum_p2',
            'phase1_epochs', 'phase2_epochs', 'budget_decay_epochs',
            'warmup_ebops_mul', 'target_ebops',
            'phase1_beta_init', 'phase1_beta_min', 'phase1_beta_max',
            'phase2_beta_min', 'phase2_beta_max',
            'adaptive_lr_enabled',
        ]
        for k in relevant_keys:
            if k in cfg:
                print(f'    {k}: {cfg[k]}')
        return None

    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 保存元数据
    meta = {
        'experiment': exp_name,
        'desc': exp_def['desc'],
        'target_ebops': TARGET_EBOPS,
        'total_epochs': total,
        'curriculum_str': cur_str,
        'beta_curriculum_enabled': cfg.get('beta_curriculum_enabled', False),
        'beta_stall_patience': cfg.get('beta_stall_patience'),
        'beta_recover_epochs': cfg.get('beta_recover_epochs'),
        'beta_restart_decay': cfg.get('beta_restart_decay'),
        'beta_max_restarts': cfg.get('beta_max_restarts'),
        'beta_recover_floor': cfg.get('beta_recover_floor'),
        'beta_curriculum_p2': cfg.get('beta_curriculum_p2', True),
    }
    with open(os.path.join(cfg['output_dir'], 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    # 使用 v2 管线
    from run_all_v2 import run as run_v2_pipeline
    t0 = time.time()
    result = run_v2_pipeline(cfg)
    elapsed = time.time() - t0
    result['experiment'] = exp_name
    result['desc'] = exp_def['desc']
    result['elapsed_total'] = elapsed
    result['curriculum_str'] = cur_str

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 对比表
# ═══════════════════════════════════════════════════════════════════════════════

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


def _read_restart_count(output_dir: str) -> int | None:
    """尝试从 training_trace.h5 读取最终 restart 次数。"""
    import h5py
    trace_path = os.path.join(output_dir, 'training_trace.h5')
    if not os.path.isfile(trace_path):
        return None
    try:
        with h5py.File(trace_path, 'r') as f:
            if 'beta_curriculum_restarts' in f:
                arr = f['beta_curriculum_restarts'][:]
                return int(arr[-1]) if len(arr) > 0 else None
    except Exception:
        pass
    return None


def print_comparison(results: list[dict]):
    """打印 Beta Curriculum 消融实验对比表。"""
    import numpy as np

    print(f'\n{"=" * 140}')
    print(f'  Beta Curriculum Restart ABLATION — target={TARGET_EBOPS} eBOPs')
    print(f'{"=" * 140}')
    print(f'  {"Exp":>4s}  {"Description":<32s}  {"Curriculum Config":<38s}  '
          f'{"Best@Tgt":>8s}  {"@Ep":>5s}  {"Final":>7s}  {"eBOPs":>6s}  '
          f'{"#Rst":>4s}  {"Time":>5s}  {"Δ G0":>7s}  {"Δ G1":>7s}')
    print(f'  {"-" * 136}')

    # 找 G0 (baseline with curriculum) 和 G1 (no curriculum)
    g0_acc, g1_acc = None, None
    for r in results:
        exp = r.get('experiment', '')
        output_dir = os.path.join(OUTPUT_ROOT, exp)
        acc, _ = _read_trace_best(output_dir)
        if acc is None:
            acc = r.get('best_acc_at_target', r.get('best_val_acc', 0))
        if exp == 'G0':
            g0_acc = acc
        elif exp == 'G1':
            g1_acc = acc

    for r in sorted(results, key=lambda x: x.get('experiment', '')):
        exp_name = r.get('experiment', '?')
        desc = r.get('desc', '?')[:32]

        output_dir = os.path.join(OUTPUT_ROOT, exp_name)
        best_at_target, best_ep = _read_trace_best(output_dir)

        meta = _read_meta(output_dir)
        cur_str = meta.get('curriculum_str', _format_curriculum(r))
        if len(cur_str) > 38:
            cur_str = cur_str[:35] + '...'

        acc_val = best_at_target if best_at_target is not None else r.get('best_acc_at_target', r.get('best_val_acc', 0))
        acc_str = f'{acc_val:.4f}' if acc_val else '—'
        ep_str = f'{best_ep}' if best_ep is not None else f'{r.get("best_epoch_at_target", "?")}'
        final_acc = r.get('final_val_acc', 0)
        final_eb = r.get('final_ebops', 0)
        elapsed = r.get('elapsed_total', r.get('elapsed_sec', 0))

        n_rst = _read_restart_count(output_dir)
        rst_str = f'{n_rst}' if n_rst is not None else '—'

        d_g0 = f'{acc_val - g0_acc:+.4f}' if g0_acc is not None and acc_val else '—'
        d_g1 = f'{acc_val - g1_acc:+.4f}' if g1_acc is not None and acc_val else '—'

        print(f'  {exp_name:>4s}  {desc:<32s}  {cur_str:<38s}  '
              f'{acc_str:>8s}  {ep_str:>5s}  {final_acc:7.4f}  {final_eb:6.0f}  '
              f'{rst_str:>4s}  {elapsed:5.0f}s  {d_g0:>7s}  {d_g1:>7s}')

    print(f'{"=" * 140}')

    # 结论
    if g0_acc is not None and g1_acc is not None:
        print(f'\n  Reference: G0 (curriculum ON) = {g0_acc:.4f}  |  G1 (curriculum OFF) = {g1_acc:.4f}  |  Δ(G0-G1) = {g0_acc - g1_acc:+.4f}')

    if g0_acc is not None:
        best_name, best_acc = None, g0_acc
        for r in results:
            exp = r.get('experiment', '')
            output_dir = os.path.join(OUTPUT_ROOT, exp)
            acc, _ = _read_trace_best(output_dir)
            if acc is None:
                acc = r.get('best_acc_at_target', r.get('best_val_acc', 0))
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_name = exp

        if best_name and best_name != 'G0':
            meta = _read_meta(os.path.join(OUTPUT_ROOT, best_name))
            print(f'  ★ Best config: {best_name} ({best_acc:.4f}) — {meta.get("desc", EXPERIMENTS.get(best_name, {}).get("desc", "?"))}')
            print(f'    Curriculum: {meta.get("curriculum_str", "?")}')
        else:
            print(f'  ★ G0 default is already the best or tied ({g0_acc:.4f})')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Experiment G: Beta Curriculum Restart ablation at 400 eBOPs')
    parser.add_argument('--run', nargs='+', default=None,
                        help='Experiment names to run (e.g. G0 G1, or "G" for all, or "all")')
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
        print(f'\n  Beta Curriculum Restart Ablation (target={TARGET_EBOPS} eBOPs):')
        print(f'  {"Name":>4s}  {"Description":<32s}  {"Enabled":>7s}  {"Patience":>8s}  '
              f'{"Recover":>7s}  {"Decay":>5s}  {"Max":>3s}  {"Floor":>7s}  {"P2":>3s}')
        print(f'  {"-" * 100}')
        for name, exp in sorted(EXPERIMENTS.items()):
            ov = exp['overrides']
            en = '✓' if ov.get('beta_curriculum_enabled', False) else '✗'
            pat = str(ov.get('beta_stall_patience', '—')) if ov.get('beta_curriculum_enabled', False) else '—'
            rec = str(ov.get('beta_recover_epochs', '—')) if ov.get('beta_curriculum_enabled', False) else '—'
            dec = f'{ov["beta_restart_decay"]:.2f}' if ov.get('beta_curriculum_enabled', False) else '—'
            mx = str(ov.get('beta_max_restarts', '—')) if ov.get('beta_curriculum_enabled', False) else '—'
            fl = ov.get('beta_recover_floor')
            fl_s = f'{fl:.0e}' if fl is not None and ov.get('beta_curriculum_enabled', False) else 'β_min'
            if not ov.get('beta_curriculum_enabled', False):
                fl_s = '—'
            p2 = '✓' if ov.get('beta_curriculum_p2', True) and ov.get('beta_curriculum_enabled', False) else '✗'

            print(f'  {name:>4s}  {exp["desc"]:<32s}  {en:>7s}  {pat:>8s}  '
                  f'{rec:>7s}  {dec:>5s}  {mx:>3s}  {fl_s:>7s}  {p2:>3s}')
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
            if token.lower() in ('all', 'g'):
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
        print(f'\n  Use --list to see experiments, --run G0 G1 to run, --dry-run to preview.')
        return

    print(f'\n{"#" * 72}')
    print(f'  Experiment G: Beta Curriculum Restart Ablation')
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
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        summary_path = os.path.join(OUTPUT_ROOT, 'all_results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Summary saved: {summary_path}')

        print_comparison(all_results)


if __name__ == '__main__':
    main()
