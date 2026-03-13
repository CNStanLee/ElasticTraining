#!/usr/bin/env python3
"""
run_paper_experiments.py — FPL 论文所需全部实验的统一入口
=========================================================

实验列表:
  A. 主 Pareto 扫描 (5 个 eBOPs 目标)
  B. 剪枝方法 Ablation (random / magnitude / spectral)
  C. 渐进预算 Ablation (不同 warmup 倍率 μ)
  D. Beta 课程 Ablation (开/关 + 不同 patience)
  E. 自适应 LR Ablation (开/关)
  F. SoftDeathFloor Ablation (开/关 + 不同 soft_floor_alive 阈值)

用法:
  python run_paper_experiments.py --list              # 列出所有实验
  python run_paper_experiments.py --run A1            # 运行实验 A1
  python run_paper_experiments.py --run A             # 运行 A 类所有实验
  python run_paper_experiments.py --run-all           # 运行全部实验 (极耗时!)
  python run_paper_experiments.py --run-all --skip-existing  # 跳过已有结果
  python run_paper_experiments.py --dry-run           # 仅打印配置不运行
  python run_paper_experiments.py --workers 3         # 同时运行 3 个实验 (默认 3)

输出目录: results/paper/<实验名>/
原始绘图数据: results/paper/plot_data/  (JSON + CSV + .mat, 可用 MATLAB 重绘)

估计总时间 (单 GPU, JSC 模型, workers=1):
  A 类: 5 × ~3h = ~15h
  B 类: 3 × ~3h = ~9h
  C 类: 4 × ~3h = ~12h
  D 类: 3 × ~3h = ~9h
  E 类: 1 × ~3h = ~3h
  F 类: 7 × ~3h = ~21h  (F1 可复用 A1 结果)
  总计: ~69h  (workers=3 时 ~23h)
"""

import os
import sys
import json
import argparse
import copy
import time
import subprocess
import signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 确保工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# 实验定义
# ═══════════════════════════════════════════════════════════════════════════════

# 基础配置 (从 run_all.py 的 DEFAULT_CONFIG 加载, 这里只列覆盖项)
BASE_PRETRAINED = 'pretrained_weight/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras'

EXPERIMENTS = {}


def exp(name, desc, overrides):
    """注册一个实验。"""
    EXPERIMENTS[name] = {
        'name': name,
        'desc': desc,
        'overrides': overrides,
        'output_dir': f'results/paper/{name}/',
    }


# ─── A. 主 Pareto 扫描 ────────────────────────────────────────────────────
# Table 1, Figure 1: 不同 eBOPs 目标下的 best accuracy
exp('A1_sweep_400',   'Main: target=400 eBOPs',   dict(target_ebops=400))
exp('A2_sweep_1500',  'Main: target=1500 eBOPs',  dict(target_ebops=1500))
exp('A3_sweep_2500',  'Main: target=2500 eBOPs',  dict(target_ebops=2500))
exp('A4_sweep_6800',  'Main: target=6800 eBOPs',  dict(target_ebops=6800))
exp('A5_sweep_12000', 'Main: target=12000 eBOPs', dict(target_ebops=12000))

# ─── B. 剪枝方法 Ablation (Table 2, Figure 3) ────────────────────────────
# 固定 target=400, 对比 spectral vs random vs magnitude
exp('B1_prune_spectral',  'Prune ablation: spectral (default)',
    dict(target_ebops=400, pruning_method='auto'))
exp('B2_prune_random',    'Prune ablation: random',
    dict(target_ebops=400, pruning_method='random'))
exp('B3_prune_magnitude', 'Prune ablation: magnitude',
    dict(target_ebops=400, pruning_method='magnitude'))

# ─── C. 渐进预算 Ablation (Table 3) ──────────────────────────────────────
# 固定 target=400, 对比不同 warmup 倍率
exp('C1_budget_direct', 'Budget ablation: direct (μ=1.0)',
    dict(target_ebops=400, warmup_ebops_mul=1.0, budget_decay_epochs=1,
         use_sensitivity_pruner=True))
exp('C2_budget_mu2',    'Budget ablation: μ=2.0',
    dict(target_ebops=400, warmup_ebops_mul=2.0, budget_decay_epochs=2000,
         use_sensitivity_pruner=True))
exp('C3_budget_mu7.5',  'Budget ablation: μ=7.5 (default for 400)',
    dict(target_ebops=400))  # 使用 get_target_overrides(400) 默认值
exp('C4_budget_mu15',   'Budget ablation: μ=15.0',
    dict(target_ebops=400, warmup_ebops_mul=15.0, budget_decay_epochs=5000,
         use_sensitivity_pruner=True))

# ─── D. Beta 课程 Ablation (Table 4) ─────────────────────────────────────
# 固定 target=400, 对比 beta curriculum 开关和 patience
exp('D1_beta_disabled',      'Beta curriculum: disabled',
    dict(target_ebops=400, beta_curriculum_enabled=False))
exp('D2_beta_patience300',   'Beta curriculum: patience=300',
    dict(target_ebops=400, beta_stall_patience=300))
exp('D3_beta_patience600',   'Beta curriculum: patience=600 (default)',
    dict(target_ebops=400))  # 使用默认
exp('D4_beta_patience1200',  'Beta curriculum: patience=1200',
    dict(target_ebops=400, beta_stall_patience=1200))

# ─── E. 自适应 LR Ablation (Table 5) ─────────────────────────────────────
exp('E1_lr_disabled', 'Adaptive LR: disabled',
    dict(target_ebops=400, adaptive_lr_enabled=False))
# E2 = A1 (enabled by default), 不需要单独跑

# ─── F. SoftDeathFloor Ablation (Table 6) ────────────────────────────
# 固定 target=400, 测试 SoftDeathFloor 效果 + soft_floor_alive 阈值消融
# 注意: 400 eBOPs 的 get_target_overrides 默认禁用 SoftDeathFloor,
#   因此需要在 overrides 中显式重新启用 (soft_floor_b, soft_floor_every)
_SDF_ENABLE_BASE = dict(
    target_ebops=400,
    soft_floor_b=0.05,       # 重新启用: 死连接下限
    soft_floor_every=50,     # 重新启用: 每 50 epoch 执行
    plot_interval=3000,      # 减少拓扑绘图频率 (加速)
)

exp('F1_sdf_disabled',    'SDF ablation: disabled (baseline)',
    dict(target_ebops=400))  # 使用 400 默认 (SDF 关闭)
exp('F2_sdf_alive0.1',    'SDF ablation: alive_threshold=0.1',
    dict(**_SDF_ENABLE_BASE, soft_floor_alive=0.1))
exp('F3_sdf_alive0.2',    'SDF ablation: alive_threshold=0.2',
    dict(**_SDF_ENABLE_BASE, soft_floor_alive=0.2))
exp('F4_sdf_alive0.3',    'SDF ablation: alive_threshold=0.3',
    dict(**_SDF_ENABLE_BASE, soft_floor_alive=0.3))
exp('F5_sdf_alive0.4',    'SDF ablation: alive_threshold=0.4 (default)',
    dict(**_SDF_ENABLE_BASE, soft_floor_alive=0.4))
exp('F6_sdf_alive0.5',    'SDF ablation: alive_threshold=0.5',
    dict(**_SDF_ENABLE_BASE, soft_floor_alive=0.5))
exp('F7_sdf_alive0.6',    'SDF ablation: alive_threshold=0.6',
    dict(**_SDF_ENABLE_BASE, soft_floor_alive=0.6))

# ─── Note: F1 等价于 A1 (400 eBOPs default, SDF 关闭)
# ─── Note: B1, C3, D3 均等价于 A1 (400 eBOPs default)
# 如果 A1 已完成, B1/C3/D3/F1 可直接链接其结果

# 等价实验映射: {实验名: 可复用的源实验名}
# 当源实验已完成时, 直接 symlink 结果, 跳过训练
ALIASES = {
    'B1_prune_spectral': 'A1_sweep_400',
    'C3_budget_mu7.5':   'A1_sweep_400',
    'D3_beta_patience600': 'A1_sweep_400',
    'F1_sdf_disabled':   'A1_sweep_400',
}


# ═══════════════════════════════════════════════════════════════════════════════
# 实验运行器
# ═══════════════════════════════════════════════════════════════════════════════

def build_config(exp_def):
    """构建完整实验配置 (合并 DEFAULT_CONFIG + target overrides + 实验覆盖)。"""
    from FPL2.jsc.bk.run_all import DEFAULT_CONFIG, get_target_overrides

    cfg = dict(DEFAULT_CONFIG)
    cfg['pretrained'] = BASE_PRETRAINED

    # 找到 target_ebops (可能在 overrides 里)
    target = exp_def['overrides'].get('target_ebops', cfg['target_ebops'])
    cfg['target_ebops'] = target

    # 应用 target-specific overrides (来自 get_target_overrides)
    target_ov = get_target_overrides(target)
    cfg.update(target_ov)

    # 应用实验自身的 overrides (最高优先级)
    cfg.update(exp_def['overrides'])

    # 输出目录
    cfg['output_dir'] = exp_def['output_dir']

    return cfg


def run_experiment(exp_name, dry_run=False):
    """运行单个实验 (in-process, 用于 --workers 1 或 --_worker 子进程)。"""
    if exp_name not in EXPERIMENTS:
        print(f'[ERROR] Unknown experiment: {exp_name}')
        print(f'  Available: {", ".join(sorted(EXPERIMENTS.keys()))}')
        return None

    # ── 检查是否可复用等价实验的结果 ────────────────────────────────────
    alias_src = ALIASES.get(exp_name)
    if alias_src and not dry_run:
        src_dir = f'results/paper/{alias_src}/'
        src_result = os.path.join(src_dir, 'result_summary.json')
        if os.path.isfile(src_result):
            import shutil
            dst_dir = EXPERIMENTS[exp_name]['output_dir']
            print(f'\n  [{exp_name}] ≡ {alias_src} (等价实验, 复制结果)')
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            # 更新 meta 中的实验名
            meta_path = os.path.join(dst_dir, 'experiment_meta.json')
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                meta['experiment'] = exp_name
                meta['aliased_from'] = alias_src
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
            with open(src_result) as f:
                result = json.load(f)
            result['experiment'] = exp_name
            result['aliased_from'] = alias_src
            result_path = os.path.join(dst_dir, 'result_summary.json')
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f'  [{exp_name}] Done (copied from {alias_src})')
            return result

    exp_def = EXPERIMENTS[exp_name]
    cfg = build_config(exp_def)

    print(f'\n{"#" * 72}')
    print(f'  Experiment: {exp_name}')
    print(f'  Description: {exp_def["desc"]}')
    print(f'  Output: {cfg["output_dir"]}')
    print(f'  target_ebops={cfg["target_ebops"]:.0f}  '
          f'warmup_mul={cfg.get("warmup_ebops_mul", 2.0):.1f}  '
          f'beta_curriculum={cfg.get("beta_curriculum_enabled", True)}  '
          f'adaptive_lr={cfg.get("adaptive_lr_enabled", True)}  '
          f'pruning={cfg.get("pruning_method", "auto")}  '
          f'sdf_alive={cfg.get("soft_floor_alive", 0.4)}  '
          f'sdf_b={cfg.get("soft_floor_b", 0.05)}')
    print(f'{"#" * 72}')

    if dry_run:
        print('  [dry-run] Config:')
        for k in sorted(cfg):
            print(f'    {k}: {cfg[k]}')
        return None

    # 保存实验配置
    os.makedirs(cfg['output_dir'], exist_ok=True)
    meta = {
        'experiment': exp_name,
        'description': exp_def['desc'],
        'overrides': exp_def['overrides'],
        'full_config': {k: v for k, v in cfg.items()
                        if isinstance(v, (int, float, str, bool, type(None)))},
    }
    with open(os.path.join(cfg['output_dir'], 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 运行
    from FPL2.jsc.bk.run_all import run
    t0 = time.time()
    try:
        result = run(cfg)
    except Exception as e:
        print(f'\n[ERROR] Experiment {exp_name} failed: {e}')
        import traceback
        traceback.print_exc()
        result = {'error': str(e)}
    elapsed = time.time() - t0

    if result:
        result['experiment'] = exp_name
        result['elapsed_total'] = elapsed

    # 保存结果
    result_path = os.path.join(cfg['output_dir'], 'result_summary.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  [{exp_name}] Done in {elapsed:.0f}s  → {result_path}')

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 并行运行 (subprocess, 对 TF/GPU 最安全)
# ═══════════════════════════════════════════════════════════════════════════════

def _launch_worker(exp_name, gpu_id=None):
    """启动一个子进程运行单个实验, 返回 (exp_name, subprocess.Popen)。"""
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cmd = [sys.executable, __file__, '--_worker', exp_name]
    log_dir = f'results/paper/{exp_name}/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'worker.log')
    log_fp = open(log_path, 'w')
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return exp_name, proc, log_fp, log_path


def run_parallel(to_run, max_workers=3, gpu_ids=None):
    """并行运行多个实验, 同时最多 max_workers 个。

    Args:
        to_run: 要运行的实验名列表
        max_workers: 最大并行数 (默认 3)
        gpu_ids: GPU ID 列表, 如 [0,1,2]. 为 None 则不限定.
    Returns:
        完成的实验结果列表
    """
    print(f'\n{"=" * 72}')
    print(f'  并行模式: {len(to_run)} experiments, max_workers={max_workers}')
    if gpu_ids:
        print(f'  GPU IDs: {gpu_ids}')
    print(f'{"=" * 72}\n')

    active = {}       # exp_name → (proc, log_fp, log_path)
    pending = list(to_run)
    finished = []
    failed = []
    gpu_pool = list(gpu_ids) if gpu_ids else [None] * max_workers

    def _poll_active():
        """检查已完成的进程。"""
        done_names = []
        for name, (proc, log_fp, log_path) in active.items():
            ret = proc.poll()
            if ret is not None:
                log_fp.close()
                done_names.append((name, ret, log_path))
        return done_names

    t0 = time.time()

    try:
        while pending or active:
            # 回收完成的进程
            for name, retcode, log_path in _poll_active():
                del active[name]
                if retcode == 0 and is_experiment_done(name):
                    finished.append(name)
                    print(f'  ✓ {name} 完成 (log: {log_path})')
                else:
                    failed.append(name)
                    print(f'  ✗ {name} 失败 (retcode={retcode}, log: {log_path})')
                # 回收 GPU
                if gpu_ids:
                    gpu_pool.append(gpu_ids[len(gpu_pool) % len(gpu_ids)])

                # 进度
                total = len(to_run)
                elapsed_min = (time.time() - t0) / 60
                print(f'  进度: {len(finished)}/{total} 完成, '
                      f'{len(failed)} 失败, {len(active)} 运行中, '
                      f'{len(pending)} 等待  ({elapsed_min:.0f} min)')

            # 启动新进程
            while pending and len(active) < max_workers:
                exp_name = pending.pop(0)
                gpu_id = gpu_pool.pop(0) if gpu_ids else None
                print(f'  ▶ 启动: {exp_name}' +
                      (f' (GPU {gpu_id})' if gpu_id is not None else ''))
                name, proc, log_fp, log_path = _launch_worker(exp_name, gpu_id)
                active[name] = (proc, log_fp, log_path)

            if active:
                time.sleep(5)  # 每 5 秒检查一次

    except KeyboardInterrupt:
        print(f'\n  [Ctrl+C] 正在终止所有子进程...')
        for name, (proc, log_fp, log_path) in active.items():
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log_fp.close()
            print(f'    终止: {name}')
        print(f'  已终止 {len(active)} 个进程')

    elapsed = time.time() - t0
    print(f'\n{"=" * 72}')
    print(f'  并行运行完成: {len(finished)} 成功, {len(failed)} 失败, '
          f'耗时 {elapsed/60:.1f} min')
    print(f'{"=" * 72}')

    # 收集结果
    all_results = []
    for name in finished:
        rp = os.path.join('results/paper', name, 'result_summary.json')
        if os.path.exists(rp):
            with open(rp) as f:
                r = json.load(f)
                r['experiment'] = name
                all_results.append(r)
    return all_results


def is_experiment_done(exp_name):
    """检查实验是否已完成 (result_summary.json 存在且含 best_val_acc)。"""
    exp_def = EXPERIMENTS[exp_name]
    result_path = os.path.join(exp_def['output_dir'], 'result_summary.json')
    if not os.path.exists(result_path):
        return False
    try:
        with open(result_path) as f:
            r = json.load(f)
        return 'best_val_acc' in r or 'final_val_acc' in r
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 原始绘图数据导出 (JSON + CSV + .mat)
# ═══════════════════════════════════════════════════════════════════════════════

def export_plot_data(all_results=None):
    """收集所有实验的训练轨迹和结果, 导出为 MATLAB 可读的格式。

    导出:
      - plot_data/summary_table.json / .csv       全部实验的汇总指标
      - plot_data/traces/<exp>.json               每个实验的完整训练轨迹
      - plot_data/pareto_points.json / .csv       所有 Pareto 前沿点
      - plot_data/all_plot_data.mat               MATLAB .mat 文件 (如果 scipy 可用)
    """
    import csv
    import glob
    import numpy as np

    out_dir = 'results/paper/plot_data'
    trace_dir = os.path.join(out_dir, 'traces')
    os.makedirs(trace_dir, exist_ok=True)

    print(f'\n{"=" * 72}')
    print(f'  导出原始绘图数据 → {out_dir}/')
    print(f'{"=" * 72}')

    # ── 1. 汇总表 ────────────────────────────────────────────────────────────
    summaries = []
    paper_dir = 'results/paper'
    if os.path.isdir(paper_dir):
        for d in sorted(os.listdir(paper_dir)):
            rp = os.path.join(paper_dir, d, 'result_summary.json')
            if os.path.isfile(rp):
                try:
                    with open(rp) as f:
                        r = json.load(f)
                    r.setdefault('experiment', d)
                    summaries.append(r)
                except Exception:
                    pass

    # 也包含 legacy results
    results_dir = 'results'
    if os.path.isdir(results_dir):
        for d in sorted(os.listdir(results_dir)):
            if d == 'paper':
                continue
            rp = os.path.join(results_dir, d, 'result_summary.json')
            if os.path.isfile(rp):
                try:
                    with open(rp) as f:
                        r = json.load(f)
                    r.setdefault('experiment', f'legacy_{d}')
                    summaries.append(r)
                except Exception:
                    pass

    # JSON
    with open(os.path.join(out_dir, 'summary_table.json'), 'w') as f:
        json.dump(summaries, f, indent=2)

    # CSV
    if summaries:
        all_keys = []
        for s in summaries:
            for k in s:
                if k not in all_keys:
                    all_keys.append(k)
        csv_path = os.path.join(out_dir, 'summary_table.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(summaries)
        print(f'  汇总表: {csv_path} ({len(summaries)} experiments)')

    # ── 2. 训练轨迹 (每个实验 → JSON + CSV) ─────────────────────────────────
    try:
        import h5py
        _has_h5py = True
    except ImportError:
        _has_h5py = False
        print('  [WARN] h5py not available, skipping trace export')

    mat_data = {}  # 用于 .mat 导出
    trace_count = 0

    dirs_to_scan = []
    if os.path.isdir(paper_dir):
        for d in sorted(os.listdir(paper_dir)):
            fp = os.path.join(paper_dir, d)
            if os.path.isdir(fp) and d != 'plot_data':
                dirs_to_scan.append((d, fp))
    if os.path.isdir(results_dir):
        for d in sorted(os.listdir(results_dir)):
            fp = os.path.join(results_dir, d)
            if os.path.isdir(fp) and d != 'paper':
                dirs_to_scan.append((f'legacy_{d}', fp))

    for exp_name, exp_dir in dirs_to_scan:
        h5_path = os.path.join(exp_dir, 'training_trace.h5')
        if not _has_h5py or not os.path.isfile(h5_path):
            continue

        trace = {}
        try:
            with h5py.File(h5_path, 'r') as hf:
                for key in hf.keys():
                    arr = hf[key][:]
                    trace[key] = arr
        except Exception as e:
            print(f'  [WARN] Failed to read {h5_path}: {e}')
            continue

        if not trace:
            continue

        trace_count += 1
        safe_name = exp_name.replace('/', '_').replace(' ', '_')

        # JSON (arrays → lists)
        trace_json = {k: v.tolist() for k, v in trace.items()}
        with open(os.path.join(trace_dir, f'{safe_name}.json'), 'w') as f:
            json.dump(trace_json, f)

        # CSV (每列一个指标, 行 = epoch)
        max_len = max(len(v) for v in trace.values())
        csv_path = os.path.join(trace_dir, f'{safe_name}.csv')
        with open(csv_path, 'w', newline='') as f:
            keys = sorted(trace.keys())
            w = csv.writer(f)
            w.writerow(keys)
            for i in range(max_len):
                row = []
                for k in keys:
                    arr = trace[k]
                    if arr.ndim == 1 and i < len(arr):
                        row.append(float(arr[i]))
                    elif arr.ndim == 2 and i < arr.shape[0]:
                        # 多列 (如 bw_distribution): 展成 "v1;v2;v3"
                        row.append(';'.join(f'{x:.4f}' for x in arr[i]))
                    else:
                        row.append('')
                w.writerow(row)

        # 为 .mat 累积 (使用 MATLAB 合法变量名)
        mat_key = safe_name.replace('-', '_').replace('.', '_')
        # MATLAB struct-like: 每个 trace 的每个 key 都是一个 field
        for k, v in trace.items():
            mat_data[f'{mat_key}__{k}'] = v.astype(np.float64) if v.dtype.kind == 'f' else v

    print(f'  训练轨迹: {trace_count} experiments → {trace_dir}/')

    # ── 3. Pareto 前沿点 ─────────────────────────────────────────────────────
    pareto_points = []
    for exp_name, exp_dir in dirs_to_scan:
        for f in glob.glob(os.path.join(exp_dir, 'epoch=*.keras')):
            name = os.path.basename(f)
            try:
                parts = name.replace('.keras', '').split('-')
                info = {'experiment': exp_name}
                for p in parts:
                    if '=' in p:
                        k, v = p.split('=', 1)
                        info[k] = float(v) if k != 'epoch' else int(v)
                if 'val_acc' in info and 'ebops' in info:
                    pareto_points.append(info)
            except Exception:
                pass

    with open(os.path.join(out_dir, 'pareto_points.json'), 'w') as f:
        json.dump(pareto_points, f, indent=2)

    if pareto_points:
        pp_keys = list(pareto_points[0].keys())
        csv_path = os.path.join(out_dir, 'pareto_points.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=pp_keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(pareto_points)
        print(f'  Pareto 点: {csv_path} ({len(pareto_points)} points)')

        # 加入 .mat
        mat_data['pareto_ebops'] = np.array([p.get('ebops', 0) for p in pareto_points])
        mat_data['pareto_val_acc'] = np.array([p.get('val_acc', 0) for p in pareto_points])

    # ── 4. 汇总表加入 .mat ───────────────────────────────────────────────────
    if summaries:
        numeric_keys = ['target_ebops', 'final_ebops', 'final_val_acc', 'final_val_loss',
                        'best_val_acc', 'best_ebops', 'pruned_ebops', 'phase1_val_acc',
                        'phase1_ebops', 'elapsed_sec']
        for k in numeric_keys:
            vals = [float(s.get(k, np.nan)) for s in summaries]
            mat_data[f'summary_{k}'] = np.array(vals)
        # experiment names as a list
        mat_data['summary_experiment_names'] = np.array(
            [s.get('experiment', '') for s in summaries], dtype=object)

    # ── 5. 导出 .mat (需要 scipy) ────────────────────────────────────────────
    try:
        from scipy.io import savemat
        mat_path = os.path.join(out_dir, 'all_plot_data.mat')
        savemat(mat_path, mat_data, do_compression=True)
        print(f'  MATLAB: {mat_path} ({len(mat_data)} variables)')
    except ImportError:
        print('  [INFO] scipy 未安装, 跳过 .mat 导出. '
              '可用 pip install scipy 安装后重新运行 --export-data')
    except Exception as e:
        print(f'  [WARN] .mat 导出失败: {e}')

    print(f'\n  绘图数据导出完成! 目录: {out_dir}/')
    print(f'  MATLAB 用法:')
    print(f'    data = load("all_plot_data.mat");')
    print(f'    plot(data.A1_sweep_400__epochs, data.A1_sweep_400__val_accuracy)')
    print(f'  或直接读 CSV:')
    print(f'    T = readtable("traces/A1_sweep_400.csv");')
    print(f'    plot(T.epochs, T.val_accuracy)')

    return out_dir

def print_experiment_list():
    """打印所有实验列表及状态。"""
    print(f'\n{"=" * 80}')
    print(f'  FPL 论文实验列表 ({len(EXPERIMENTS)} experiments)')
    print(f'{"=" * 80}')
    prev_group = ''
    for name in sorted(EXPERIMENTS):
        group = name[0]
        if group != prev_group:
            labels = {
                'A': 'A. 主 Pareto 扫描 (Table 1, Figure 1)',
                'B': 'B. 剪枝方法 Ablation (Table 2)',
                'C': 'C. 渐进预算 Ablation (Table 3)',
                'D': 'D. Beta 课程 Ablation (Table 4)',
                'E': 'E. 自适应 LR Ablation (Table 5)',
                'F': 'F. SoftDeathFloor Ablation (Table 6)',
            }
            print(f'\n  ── {labels.get(group, group)} ──')
            prev_group = group

        done = is_experiment_done(name)
        status = '✓ DONE' if done else '  TODO'
        desc = EXPERIMENTS[name]['desc']
        print(f'    [{status}]  {name:25s}  {desc}')

    # 统计
    n_done = sum(1 for n in EXPERIMENTS if is_experiment_done(n))
    print(f'\n  Progress: {n_done}/{len(EXPERIMENTS)} done')
    print(f'{"=" * 80}\n')


def main():
    parser = argparse.ArgumentParser(description='FPL Paper Experiments Runner')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    parser.add_argument('--run', type=str, default=None,
                        help='Run experiment(s) by name or prefix (e.g. "A1" or "A")')
    parser.add_argument('--run-all', action='store_true', help='Run all experiments')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip experiments with existing results')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print config without running')
    parser.add_argument('--workers', type=int, default=3,
                        help='Max parallel workers (default: 3). Use 1 for sequential.')
    parser.add_argument('--gpu-ids', type=str, default=None,
                        help='Comma-separated GPU IDs for workers, e.g. "0,1,2"')
    parser.add_argument('--export-data', action='store_true',
                        help='Export raw plotting data (JSON/CSV/MAT) without running')
    parser.add_argument('--_worker', type=str, default=None,
                        help=argparse.SUPPRESS)  # 内部: 子进程运行单个实验
    args = parser.parse_args()

    # ── 子进程 worker 模式 (内部使用) ─────────────────────────────────────────
    if args._worker:
        exp_name = args._worker
        result = run_experiment(exp_name, dry_run=False)
        sys.exit(0 if (result and 'error' not in result) else 1)

    # ── 仅导出绘图数据 ───────────────────────────────────────────────────────
    if args.export_data:
        export_plot_data()
        return

    if args.list or (not args.run and not args.run_all):
        print_experiment_list()
        return

    # 确定要运行的实验
    if args.run_all:
        to_run = sorted(EXPERIMENTS.keys())
    elif args.run:
        prefix = args.run
        # 精确匹配
        if prefix in EXPERIMENTS:
            to_run = [prefix]
        else:
            # 前缀匹配
            to_run = sorted(n for n in EXPERIMENTS if n.startswith(prefix))
        if not to_run:
            print(f'[ERROR] No experiments matching: {prefix}')
            print_experiment_list()
            return
    else:
        print_experiment_list()
        return

    if args.skip_existing:
        before = len(to_run)
        to_run = [n for n in to_run if not is_experiment_done(n)]
        print(f'  Skipped {before - len(to_run)} existing experiments')

    print(f'\n  Experiments to run: {to_run}')
    print(f'  Total: {len(to_run)}')
    print(f'  Workers: {args.workers}')

    # 解析 GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]

    if args.dry_run:
        # dry-run 模式: 串行打印配置
        for name in to_run:
            run_experiment(name, dry_run=True)
        return

    # ── 运行 (并行 or 串行) ───────────────────────────────────────────────
    if args.workers > 1 and len(to_run) > 1:
        # 并行: 通过子进程运行
        all_results = run_parallel(to_run, max_workers=args.workers, gpu_ids=gpu_ids)
    else:
        # 串行: 直接 in-process 运行
        all_results = []
        for i, name in enumerate(to_run):
            print(f'\n{"=" * 72}')
            print(f'  [{i+1}/{len(to_run)}] Running: {name}')
            print(f'{"=" * 72}')
            result = run_experiment(name, dry_run=False)
            if result:
                all_results.append(result)

    # 保存汇总
    if all_results:
        summary_path = 'results/paper/all_results_summary.json'
        os.makedirs('results/paper', exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\n  All results summary: {summary_path}')

    # ── 自动导出原始绘图数据 ─────────────────────────────────────────────────
    if all_results:
        export_plot_data(all_results)

    print(f'\n  Done! {len(all_results)} experiments completed.')
    print(f'  Run `python plot_paper.py` to generate figures and tables.')
    print(f'  Raw data for MATLAB: results/paper/plot_data/')


if __name__ == '__main__':
    main()
