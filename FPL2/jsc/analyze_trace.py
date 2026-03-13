"""
分析 training_trace.h5 文件
============================
1. 是否所有 epoch 都被记录？
2. 是否能恢复所有 Pareto-optimal (Acc, eBOPs) 点及其 epoch？
3. ACC 精度到小数点后几位？
"""

import os, sys, glob, re
import h5py
import numpy as np

TRACE_PATH = "results/experiment_A/openml_200k/training_trace.h5"
MODEL_DIR  = "results/experiment_A/openml_200k"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 读取 HDF5 内容
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print(f"  分析文件: {TRACE_PATH}")
print("=" * 72)

with h5py.File(TRACE_PATH, "r") as f:
    print("\n[1] HDF5 datasets:")
    for key in sorted(f.keys()):
        ds = f[key]
        print(f"    {key:20s}  shape={ds.shape}  dtype={ds.dtype}")
    
    epochs      = f["epochs"][:]
    val_acc     = f["val_accuracy"][:]
    ebops       = f["ebops"][:]
    accuracy    = f["accuracy"][:]
    loss_arr    = f["loss"][:]
    val_loss    = f["val_loss"][:]
    lr          = f["lr"][:]
    beta        = f["beta"][:]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Epoch 完整性检查
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  [2] Epoch 完整性")
print("=" * 72)

n = len(epochs)
print(f"  记录总数: {n}")
print(f"  epoch 范围: [{epochs.min()}, {epochs.max()}]")
print(f"  期望总 epoch 数 (0..{epochs.max()}): {epochs.max() + 1}")

# 检查是否连续
diffs = np.diff(epochs)
if np.all(diffs == 1):
    print(f"  ✓ 所有 epoch 连续记录 (0 到 {epochs.max()})，无缺失")
else:
    gaps = np.where(diffs != 1)[0]
    print(f"  ✗ 发现 {len(gaps)} 处间隙:")
    for g in gaps[:10]:
        print(f"    epoch {epochs[g]} → {epochs[g+1]}  (gap={diffs[g]})")
    if len(gaps) > 10:
        print(f"    ... 省略 {len(gaps)-10} 处")
    missing = int(epochs.max() + 1 - n)
    print(f"  总共缺少 {missing} 个 epoch (记录 {n}/{epochs.max()+1})")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ACC 精度分析
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  [3] ACC 精度分析")
print("=" * 72)

print(f"  val_accuracy dtype: {val_acc.dtype}")
print(f"  accuracy     dtype: {accuracy.dtype}")

# 检查小数位数
def analyze_precision(arr, name):
    """分析数组中数值的有效小数位数"""
    unique_vals = np.unique(arr[~np.isnan(arr)])
    # 用字符串表示查看精度
    max_decimals = 0
    sample_strs = []
    for v in unique_vals[:20]:
        # 完整精度字符串
        s = f"{v:.20f}".rstrip("0")
        dec = len(s.split(".")[-1]) if "." in s else 0
        max_decimals = max(max_decimals, dec)
        sample_strs.append(f"{v:.15g}")
    
    # 也用 repr 观察
    print(f"\n  --- {name} ---")
    print(f"  唯一值数量: {len(unique_vals)}")
    print(f"  范围: [{arr[~np.isnan(arr)].min():.15g}, {arr[~np.isnan(arr)].max():.15g}]")
    
    # 统计不同小数位数的有效位
    decimals_count = {}
    for v in unique_vals:
        s = f"{v:.20f}".rstrip("0")
        dec = len(s.split(".")[-1]) if "." in s else 0
        decimals_count[dec] = decimals_count.get(dec, 0) + 1
    
    print(f"  小数位分布:")
    for d in sorted(decimals_count.keys()):
        print(f"    {d:2d} 位: {decimals_count[d]:6d} 个值")
    
    # 观察精度是否受限于 float32 / float64
    print(f"  前10个唯一值 (full precision):")
    for v in unique_vals[:10]:
        print(f"    {v!r}  →  {v:.20f}")
    
    # 检查是否存在 3 位小数的离散化 (如 0.xxx)
    rounded3 = np.round(unique_vals, 3)
    diff_from_3dec = np.abs(unique_vals - rounded3)
    if np.all(diff_from_3dec < 1e-10):
        print(f"  ⚠ 所有值都恰好是 3 位小数，表明精度被限制在 3 位！")
    
    rounded4 = np.round(unique_vals, 4)
    diff_from_4dec = np.abs(unique_vals - rounded4)
    if np.all(diff_from_4dec < 1e-10):
        print(f"  → 所有值在 4 位小数内精确")
    
    for nd in range(1, 16):
        rounded_nd = np.round(unique_vals, nd)
        diff_nd = np.abs(unique_vals - rounded_nd)
        if np.all(diff_nd < 1e-15):
            print(f"  → 最小完整精度: {nd} 位小数")
            return nd
    return -1

prec_val = analyze_precision(val_acc, "val_accuracy")
prec_train = analyze_precision(accuracy, "accuracy")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Pareto Front 还原检查
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  [4] Pareto Front 检查")
print("=" * 72)

# 从保存的 .keras 文件名中提取 Pareto 点
pareto_files = glob.glob(os.path.join(MODEL_DIR, "epoch=*.keras"))
print(f"  保存的 Pareto 模型数: {len(pareto_files)}")

# 解析文件名
pat = re.compile(r"epoch=(\d+)-val_acc=([\d.]+)-ebops=(\d+)-val_loss=([\d.]+)\.keras")
pareto_points = []
for fp in pareto_files:
    m = pat.search(os.path.basename(fp))
    if m:
        pareto_points.append({
            "epoch": int(m.group(1)),
            "val_acc": float(m.group(2)),
            "ebops": int(m.group(3)),
            "val_loss": float(m.group(4)),
        })

pareto_points.sort(key=lambda x: x["epoch"])
print(f"  成功解析: {len(pareto_points)}")

# 检查这些 Pareto 点是否在 trace 中都能找到
print(f"\n  在 trace 中查找 Pareto 点:")
found = 0
not_found = 0
mismatch = 0
for pp in pareto_points:
    ep = pp["epoch"]
    # 查找这个 epoch 在 trace 中的 index
    idx = np.where(epochs == ep)[0]
    if len(idx) == 0:
        not_found += 1
        if not_found <= 5:
            print(f"    ✗ epoch={ep} 不在 trace 中！(Pareto acc={pp['val_acc']:.3f}, ebops={pp['ebops']})")
        continue
    idx = idx[0]
    trace_acc = val_acc[idx]
    trace_ebops = ebops[idx]
    
    # 比较精度 — 文件名中 val_acc 是 3 位小数
    acc_match = abs(trace_acc - pp["val_acc"]) < 0.001
    ebops_match = abs(trace_ebops - pp["ebops"]) < 1.0
    
    if acc_match and ebops_match:
        found += 1
    else:
        mismatch += 1
        if mismatch <= 5:
            print(f"    ⚠ epoch={ep}: trace(acc={trace_acc:.6f}, ebops={trace_ebops:.0f}) "
                  f"vs file(acc={pp['val_acc']:.3f}, ebops={pp['ebops']})")

print(f"\n  汇总:")
print(f"    匹配: {found}/{len(pareto_points)}")
print(f"    未找到 epoch: {not_found}")
print(f"    数值不匹配: {mismatch}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 重新计算 Pareto Front 并与文件名比较
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  [5] 从 trace 重算 Pareto Front 并比较")
print("=" * 72)

# Pareto: maximize val_accuracy, minimize ebops
# 即：一个点 (acc_i, ebops_i) 是 Pareto 最优当且仅当不存在 j 使得 acc_j >= acc_i 且 ebops_j <= ebops_i（至少一个严格）
def compute_pareto(acc_arr, ebops_arr):
    """返回 Pareto 最优点的索引 (maximize acc, minimize ebops)"""
    n = len(acc_arr)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        # 被 i 支配的点
        dominated = (acc_arr[i] >= acc_arr) & (ebops_arr[i] <= ebops_arr) & \
                    ((acc_arr[i] > acc_arr) | (ebops_arr[i] < ebops_arr))
        dominated[i] = False
        is_pareto[dominated] = False
    return np.where(is_pareto)[0]

valid = ~np.isnan(val_acc) & ~np.isnan(ebops)
pareto_idx = compute_pareto(val_acc[valid], ebops[valid])
# Map back to original indexing
valid_indices = np.where(valid)[0]
pareto_orig_idx = valid_indices[pareto_idx]

print(f"  从 trace 重算的 Pareto 点数: {len(pareto_orig_idx)}")
print(f"  从 .keras 文件名解析的 Pareto 点数: {len(pareto_points)}")

# 从 trace 中的 Pareto 点
trace_pareto_epochs = set(epochs[pareto_orig_idx])
file_pareto_epochs = set(pp["epoch"] for pp in pareto_points)

only_in_trace = trace_pareto_epochs - file_pareto_epochs
only_in_files = file_pareto_epochs - trace_pareto_epochs

print(f"\n  仅在 trace 中的 Pareto 点 (未存模型): {len(only_in_trace)}")
if only_in_trace:
    for ep in sorted(only_in_trace)[:10]:
        idx = np.where(epochs == ep)[0][0]
        print(f"    epoch={ep}  acc={val_acc[idx]:.6f}  ebops={ebops[idx]:.0f}")
    if len(only_in_trace) > 10:
        print(f"    ... 共 {len(only_in_trace)} 个")

print(f"  仅在文件中而 trace 中不存在: {len(only_in_files)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. 最终总结
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  [6] 总结")
print("=" * 72)
print(f"  Trace 记录了 {n} 个 epoch (共 {epochs.max()+1} 个)")
if n == epochs.max() + 1:
    print(f"  ✓ 每个 epoch 都有记录，所有 Acc-eBOPs 点均被保存")
else:
    print(f"  ✗ 缺少 {epochs.max()+1 - n} 个 epoch 数据")
print(f"  val_accuracy 精度: {prec_val} 位小数 (dtype={val_acc.dtype})")
print(f"  accuracy 精度: {prec_train} 位小数 (dtype={accuracy.dtype})")
print(f"  eBOPs 数据类型: {ebops.dtype}")

# 检查 ebops 精度
ebops_unique = np.unique(ebops[~np.isnan(ebops)])
ebops_all_int = np.all(np.abs(ebops_unique - np.round(ebops_unique)) < 1e-6)
print(f"  eBOPs 是否整数: {ebops_all_int}")
if not ebops_all_int:
    print(f"  eBOPs 示例值:")
    for v in ebops_unique[:10]:
        print(f"    {v!r}")
