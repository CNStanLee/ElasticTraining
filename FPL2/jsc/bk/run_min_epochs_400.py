#!/usr/bin/env python3
"""
run_min_epochs_400.py — 寻找 400 eBOPs 目标下的最少 epoch 数
===============================================================

动机:
  现有 A1_sweep_400 实验使用 18000 epochs (Phase1=6000 + Phase2=12000),
  但训练轨迹分析表明:
    - 最佳 acc@[350-450] = 0.7305 出现在 epoch 2826
    - Phase 2 (epoch 6000-18000) 从未超越 Phase 1 的最佳结果
    - 实际有效训练仅在前 ~3500 epochs 内完成

  本脚本系统地测试不同 epoch 配置,寻找能匹配 baseline 精度 (≥0.730)
  的最少 epoch 数,关键旋钮:
    1. budget_decay_epochs  — 渐进压缩速度 (从 warmup→target)
    2. warmup_ebops_mul     — 初始缓冲倍率
    3. phase1/phase2_epochs — 训练阶段长度
    4. LR cycle length      — 匹配更短训练的 LR 周期

  此外还包括从已剪枝预训练模型直接启动的 "warm start" 方案。

用法:
  python run_min_epochs_400.py                        # 运行所有配置
  python run_min_epochs_400.py --exp 1 2 3            # 运行指定配置
  python run_min_epochs_400.py --exp all               # 运行所有
  python run_min_epochs_400.py --list                  # 列出所有配置
  python run_min_epochs_400.py --dry-run               # 仅打印配置

Baseline reference (A1_sweep_400):
  Total: 18000 epochs, best_acc@[350-450] = 0.7305 (epoch 2826), time ~2556s
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

# 确保工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# 实验配置
# ═══════════════════════════════════════════════════════════════════════════════

# 基准预训练权重 (与 run_paper_experiments.py 一致)
PRETRAINED_FULL = 'pretrained_weight/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras'
# 已在 ~419 eBOPs 附近的预训练模型 (适用于 warm start)
PRETRAINED_419  = 'pretrained_weight/epoch=159151-val_acc=0.720-ebops=419-val_loss=0.844.keras'

TARGET_EBOPS = 400

# 所有实验配置:
# 参数说明:
#   phase1_epochs     — Phase 1 长度 (恢复+渐进压缩)
#   phase2_epochs     — Phase 2 长度 (精调)
#   budget_decay_epochs — 渐进压缩从 warmup→target 的 epoch 数
#   warmup_ebops_mul  — warmup = target × mul
#   phase1_lr_cycle   — Phase 1 cosine restarts 周期
#   phase2_lr_cycle   — Phase 2 cosine restarts 周期

CONFIGS = {
    # ── 参考: 当前 baseline (与 A1_sweep_400 完全一致) ─────────────────
    0: dict(
        label='baseline_18000',
        desc='Current baseline (18000 ep, budget_decay=3000)',
        phase1_epochs=6000,
        phase2_epochs=12000,
        budget_decay_epochs=3000,
        warmup_ebops_mul=3000.0 / TARGET_EBOPS,  # 7.5x
        phase1_lr_cycle=2000,
        phase2_lr_cycle=800,
    ),

    # ── 实验 1: 砍掉 Phase 2,其余不变 ────────────────────────────────
    # 假设: Phase 2 完全无用 (数据已证实)
    1: dict(
        label='no_phase2_4000',
        desc='No Phase 2: phase1=4000 (budget_decay=3000)',
        phase1_epochs=4000,
        phase2_epochs=0,
        budget_decay_epochs=3000,
        warmup_ebops_mul=3000.0 / TARGET_EBOPS,
        phase1_lr_cycle=2000,
        phase2_lr_cycle=800,
    ),

    # ── 实验 2: 缩短 Phase 1 + 短 Phase 2 ────────────────────────────
    # 假设: decay 完成后仅需少量稳定 epochs
    2: dict(
        label='short_4500',
        desc='Shortened: phase1=3000, phase2=1500 (decay=2500)',
        phase1_epochs=3000,
        phase2_epochs=1500,
        budget_decay_epochs=2500,
        warmup_ebops_mul=3000.0 / TARGET_EBOPS,
        phase1_lr_cycle=1500,
        phase2_lr_cycle=500,
    ),

    # ── 实验 3: 加速 decay ────────────────────────────────────────────
    # 假设: 可以在 2000 epochs 内完成 warmup→target 压缩
    3: dict(
        label='fast_decay_3500',
        desc='Fast decay: phase1=2500, phase2=1000 (decay=2000)',
        phase1_epochs=2500,
        phase2_epochs=1000,
        budget_decay_epochs=2000,
        warmup_ebops_mul=5.0,   # warmup=2000
        phase1_lr_cycle=1200,
        phase2_lr_cycle=500,
    ),

    # ── 实验 4: 更快 decay + 低 warmup ───────────────────────────────
    4: dict(
        label='faster_3000',
        desc='Faster: phase1=2000, phase2=1000 (decay=1500, warmup=4x)',
        phase1_epochs=2000,
        phase2_epochs=1000,
        budget_decay_epochs=1500,
        warmup_ebops_mul=4.0,   # warmup=1600
        phase1_lr_cycle=1000,
        phase2_lr_cycle=500,
    ),

    # ── 实验 5: 激进配置 ─────────────────────────────────────────────
    5: dict(
        label='aggressive_2000',
        desc='Aggressive: phase1=1500, phase2=500 (decay=1000, warmup=3x)',
        phase1_epochs=1500,
        phase2_epochs=500,
        budget_decay_epochs=1000,
        warmup_ebops_mul=3.0,   # warmup=1200
        phase1_lr_cycle=750,
        phase2_lr_cycle=250,
    ),

    # ── 实验 6: 极端激进 ─────────────────────────────────────────────
    6: dict(
        label='extreme_1500',
        desc='Extreme: phase1=1000, phase2=500 (decay=800, warmup=2.5x)',
        phase1_epochs=1000,
        phase2_epochs=500,
        budget_decay_epochs=800,
        warmup_ebops_mul=2.5,   # warmup=1000
        phase1_lr_cycle=500,
        phase2_lr_cycle=250,
    ),

    # ── 实验 7: Warm start (从已剪枝到 ~419 eBOPs 的模型启动) ───────
    # 跳过剪枝和渐进压缩,直接在目标附近精调
    7: dict(
        label='warm_start_2000',
        desc='Warm start from 419-eBOPs pretrained (skip pruning)',
        pretrained=PRETRAINED_419,
        phase1_epochs=1500,
        phase2_epochs=500,
        budget_decay_epochs=1,     # 无需渐进压缩 (已在目标附近)
        warmup_ebops_mul=1.05,     # 极小 warmup (几乎不留缓冲)
        phase1_lr=1e-3,            # 更低 LR (已收敛,避免破坏)
        phase2_lr=2e-4,
        phase1_beta_init=1e-4,     # 中等 beta (维持 eBOPs 约束)
        phase1_lr_cycle=750,
        phase2_lr_cycle=250,
        beta_stall_patience=300,
        beta_recover_epochs=100,
    ),

    # ── 实验 8: Warm start 极短 ──────────────────────────────────────
    8: dict(
        label='warm_start_1000',
        desc='Warm start from 419-eBOPs pretrained, minimal fine-tune',
        pretrained=PRETRAINED_419,
        phase1_epochs=800,
        phase2_epochs=200,
        budget_decay_epochs=1,
        warmup_ebops_mul=1.05,
        phase1_lr=5e-4,
        phase2_lr=1e-4,
        phase1_beta_init=1e-4,
        phase1_lr_cycle=400,
        phase2_lr_cycle=200,
        beta_stall_patience=200,
        beta_recover_epochs=80,
    ),

    # ── 实验 9: 单阶段训练 (无 Phase 2) + 中度加速 ──────────────────
    9: dict(
        label='single_phase_3000',
        desc='Single phase: 3000 ep only (no Phase 2, decay=2500)',
        phase1_epochs=3000,
        phase2_epochs=0,
        budget_decay_epochs=2500,
        warmup_ebops_mul=3000.0 / TARGET_EBOPS,
        phase1_lr_cycle=1500,
        phase2_lr_cycle=500,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 运行逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def build_full_config(exp_id: int) -> dict:
    """合并 DEFAULT_CONFIG + target overrides + 实验覆盖 → 完整配置。"""
    from FPL2.jsc.bk.run_all import DEFAULT_CONFIG, get_target_overrides

    exp_cfg = CONFIGS[exp_id]
    cfg = dict(DEFAULT_CONFIG)

    # 预训练路径
    cfg['pretrained'] = exp_cfg.get('pretrained', PRETRAINED_FULL)
    cfg['target_ebops'] = TARGET_EBOPS

    # 应用 400 eBOPs 的 target overrides (与 run_all.py 一致)
    target_ov = get_target_overrides(TARGET_EBOPS)
    cfg.update(target_ov)

    # 应用实验自身覆盖 (最高优先级)
    for key in [
        'phase1_epochs', 'phase2_epochs', 'budget_decay_epochs',
        'warmup_ebops_mul', 'phase1_lr_cycle', 'phase2_lr_cycle',
        'phase1_lr', 'phase2_lr', 'phase1_beta_init',
        'beta_stall_patience', 'beta_recover_epochs',
    ]:
        if key in exp_cfg:
            cfg[key] = exp_cfg[key]

    # 输出目录
    label = exp_cfg['label']
    cfg['output_dir'] = f'results/min_epochs_400/{label}/'

    # 减少拓扑绘图频率 (加速)
    cfg['plot_interval'] = max(500, cfg.get('phase1_epochs', 3000) // 3)

    return cfg


def run_single(exp_id: int, dry_run: bool = False) -> dict | None:
    """运行单个实验配置。"""
    from FPL2.jsc.bk.run_all import run as run_pipeline

    exp_cfg = CONFIGS[exp_id]
    cfg = build_full_config(exp_id)

    total = cfg['phase1_epochs'] + cfg['phase2_epochs']
    warmup = cfg['target_ebops'] * cfg['warmup_ebops_mul']

    print(f'\n{"#" * 72}')
    print(f'  Config {exp_id}: {exp_cfg["desc"]}')
    print(f'  Total epochs: {total}  (P1={cfg["phase1_epochs"]}, P2={cfg["phase2_epochs"]})')
    print(f'  Budget decay: {cfg["budget_decay_epochs"]} ep  '
          f'(warmup={warmup:.0f} → target={cfg["target_ebops"]:.0f})')
    print(f'  LR cycles: P1={cfg["phase1_lr_cycle"]}, P2={cfg["phase2_lr_cycle"]}')
    print(f'  Output: {cfg["output_dir"]}')
    print(f'{"#" * 72}')

    if dry_run:
        print('  [DRY RUN] Skipping...')
        print(f'  Full config: {json.dumps({k: v for k, v in cfg.items() if not isinstance(v, type)}, indent=2, default=str)}')
        return None

    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 保存实验元数据
    meta = {
        'exp_id': exp_id,
        'label': exp_cfg['label'],
        'desc': exp_cfg['desc'],
        'total_epochs': total,
        'phase1_epochs': cfg['phase1_epochs'],
        'phase2_epochs': cfg['phase2_epochs'],
        'budget_decay_epochs': cfg['budget_decay_epochs'],
        'warmup_ebops_mul': cfg['warmup_ebops_mul'],
        'pretrained': cfg['pretrained'],
    }
    with open(os.path.join(cfg['output_dir'], 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    t0 = time.time()
    result = run_pipeline(cfg)
    elapsed = time.time() - t0

    result['exp_id'] = exp_id
    result['label'] = exp_cfg['label']
    result['desc'] = exp_cfg['desc']
    result['total_epochs'] = total
    result['elapsed_total'] = elapsed

    return result


def print_comparison(results: list[dict]):
    """打印所有实验结果的对比表。"""
    baseline_ref_acc = 0.7305  # A1_sweep_400 最佳 acc@[350-450]

    print(f'\n{"=" * 100}')
    print(f'  RESULTS COMPARISON — target=400 eBOPs')
    print(f'  Baseline reference: best_acc@[350-450] = {baseline_ref_acc:.4f} (A1, 18000 ep)')
    print(f'{"=" * 100}')
    print(f'  {"ID":>3s}  {"Label":<25s}  {"Total":>6s}  {"P1":>5s}  {"P2":>6s}  '
          f'{"Decay":>5s}  {"Best Acc":>8s}  {"eBOPs":>6s}  {"Ep@Best":>7s}  '
          f'{"Final Acc":>9s}  {"Time":>6s}  {"Δ Ref":>7s}')
    print(f'  {"-" * 96}')

    for r in sorted(results, key=lambda x: x.get('total_epochs', 0)):
        exp_id = r.get('exp_id', '?')
        label = r.get('label', '?')[:25]
        total = r.get('total_epochs', 0)
        p1 = r.get('phase1_epochs_cfg', r.get('total_epochs', 0))
        p2 = total - p1 if isinstance(p1, int) else '?'
        best_acc = r.get('best_val_acc', 0)
        best_ebops = r.get('best_ebops', 0)
        best_ep = r.get('best_epoch', 0)
        final_acc = r.get('final_val_acc', 0)
        elapsed = r.get('elapsed_total', r.get('elapsed_sec', 0))
        delta = best_acc - baseline_ref_acc

        # 从 training trace 找 best acc @ [350-450] eBOPs
        trace_path = os.path.join(r.get('output_dir', ''), 'training_trace.h5')
        best_at_target = None
        best_at_target_ep = None
        if os.path.isfile(trace_path):
            import h5py
            import numpy as np
            with h5py.File(trace_path, 'r') as f:
                va = f['val_accuracy'][:]
                eb = f['ebops'][:]
                mask = (eb >= 350) & (eb <= 450)
                if mask.any():
                    idx = np.where(mask)[0][np.argmax(va[mask])]
                    best_at_target = va[idx]
                    best_at_target_ep = idx

        acc_str = f'{best_at_target:.4f}' if best_at_target else f'{best_acc:.4f}'
        ep_str = f'{best_at_target_ep}' if best_at_target_ep else f'{best_ep}'
        delta_val = (best_at_target - baseline_ref_acc) if best_at_target else (best_acc - baseline_ref_acc)
        delta_str = f'{delta_val:+.4f}'
        color = '' if delta_val >= -0.005 else '  ⚠'

        print(f'  {exp_id:>3}  {label:<25s}  {total:6d}  '
              f'{r.get("phase1_epochs_cfg", "?"):>5}  {p2:>6}  '
              f'{r.get("budget_decay_epochs_cfg", "?"):>5}  '
              f'{acc_str:>8s}  {best_ebops:6.0f}  {ep_str:>7s}  '
              f'{final_acc:9.4f}  {elapsed:5.0f}s  {delta_str:>7s}{color}')

    print(f'{"=" * 100}')

    # 找到 epochs 最少但 delta >= -0.005 的配置
    valid = [r for r in results if r.get('total_epochs', 99999) < 18000]
    if valid:
        print(f'\n  Recommendation:')
        for r in sorted(valid, key=lambda x: x.get('total_epochs', 99999)):
            # 检查是否匹配 baseline
            trace_path = os.path.join(r.get('output_dir', ''), 'training_trace.h5')
            if os.path.isfile(trace_path):
                import h5py
                import numpy as np
                with h5py.File(trace_path, 'r') as f:
                    va = f['val_accuracy'][:]
                    eb = f['ebops'][:]
                    mask = (eb >= 350) & (eb <= 450)
                    if mask.any():
                        best_at = va[mask].max()
                        if best_at >= baseline_ref_acc - 0.005:
                            ratio = 18000 / r['total_epochs']
                            print(f'    ★ Config {r["exp_id"]} ({r["label"]}): '
                                  f'{r["total_epochs"]} epochs → {ratio:.1f}x speedup, '
                                  f'acc={best_at:.4f} (Δ={best_at - baseline_ref_acc:+.4f})')
                            break
        else:
            print(f'    No config matches baseline within 0.005 tolerance.')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Find minimum epochs for 400 eBOPs target')
    parser.add_argument('--exp', nargs='+', default=None,
                        help='Experiment IDs to run (e.g. 1 2 3, or "all")')
    parser.add_argument('--list', action='store_true',
                        help='List all experiment configs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configs without running')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip experiments with existing results')
    args = parser.parse_args()

    if args.list:
        print(f'\n  Available experiment configs:')
        print(f'  {"ID":>3s}  {"Label":<25s}  {"Total":>6s}  {"P1":>5s}  {"P2":>5s}  '
              f'{"Decay":>5s}  {"Warmup":>7s}  Description')
        print(f'  {"-" * 90}')
        for eid, ecfg in sorted(CONFIGS.items()):
            total = ecfg.get('phase1_epochs', 0) + ecfg.get('phase2_epochs', 0)
            warmup = TARGET_EBOPS * ecfg.get('warmup_ebops_mul', 7.5)
            print(f'  {eid:>3d}  {ecfg["label"]:<25s}  {total:6d}  '
                  f'{ecfg["phase1_epochs"]:5d}  {ecfg["phase2_epochs"]:5d}  '
                  f'{ecfg["budget_decay_epochs"]:5d}  {warmup:7.0f}  {ecfg["desc"]}')
        return

    # 确定要运行的实验
    if args.exp is None or 'all' in args.exp:
        exp_ids = sorted(CONFIGS.keys())
    else:
        exp_ids = [int(x) for x in args.exp]

    # 验证
    for eid in exp_ids:
        if eid not in CONFIGS:
            print(f'[ERROR] Unknown experiment ID: {eid}')
            print(f'  Available: {sorted(CONFIGS.keys())}')
            sys.exit(1)

    print(f'\n{"#" * 72}')
    print(f'  Minimum Epochs Experiment — target={TARGET_EBOPS} eBOPs')
    print(f'  Running configs: {exp_ids}')
    print(f'  Baseline reference: 18000 ep → best_acc@[350-450] = 0.7305')
    print(f'{"#" * 72}')

    all_results = []
    for eid in exp_ids:
        ecfg = CONFIGS[eid]
        output_dir = f'results/min_epochs_400/{ecfg["label"]}/'

        # 跳过已有结果
        result_path = os.path.join(output_dir, 'result_summary.json')
        if args.skip_existing and os.path.isfile(result_path):
            print(f'\n  [Skip] Config {eid} ({ecfg["label"]}): result exists at {result_path}')
            with open(result_path) as f:
                result = json.load(f)
            result['exp_id'] = eid
            result['label'] = ecfg['label']
            result['desc'] = ecfg['desc']
            result['total_epochs'] = ecfg['phase1_epochs'] + ecfg['phase2_epochs']
            result['output_dir'] = output_dir
            result['phase1_epochs_cfg'] = ecfg['phase1_epochs']
            result['budget_decay_epochs_cfg'] = ecfg['budget_decay_epochs']
            all_results.append(result)
            continue

        if args.dry_run:
            run_single(eid, dry_run=True)
            continue

        result = run_single(eid)
        if result:
            result['output_dir'] = output_dir
            result['phase1_epochs_cfg'] = ecfg['phase1_epochs']
            result['budget_decay_epochs_cfg'] = ecfg['budget_decay_epochs']
            all_results.append(result)

            # 实时进度
            print(f'\n  ── Progress ({len(all_results)}/{len(exp_ids)}) ──')
            for r in sorted(all_results, key=lambda x: x.get('total_epochs', 0)):
                print(f'    Config {r["exp_id"]} ({r["label"]}): '
                      f'{r["total_epochs"]} ep, '
                      f'best_acc={r.get("best_val_acc", 0):.4f}, '
                      f'time={r.get("elapsed_total", 0):.0f}s')

    if all_results and not args.dry_run:
        # 保存汇总
        summary_dir = 'results/min_epochs_400/'
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, 'all_results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Summary saved: {summary_path}')

        print_comparison(all_results)


if __name__ == '__main__':
    main()
