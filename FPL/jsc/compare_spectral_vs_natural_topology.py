#!/usr/bin/env python3
"""
compare_spectral_vs_natural_topology.py
=======================================
对比 spectral_quant 剪枝和自然训练（gradual baseline）模型在多个 eBOPs 目标下
的拓扑结构，并从图谱理论和量化器信息论的角度，定量评估：
  1. 可训练性（Trainability）
  2. 是否可达最优解（Optimality Reachability）

理论依据
--------
### 判据 1：可训练性 (Trainability Index, TI)

来源：稀疏二部图的谱间隙理论 (Alon 1986, Hoory-Linial-Widgerson 2006)
      + 量化网络的 STE 有效信噪比。

  对于每层 l 的权重矩阵 W_l (shape: n_{l-1} × n_l)，定义其
  **有效邻接矩阵** A_l = B_l ⊙ |W_l|，其中 B_l 是 bit-width 矩阵。

  定义二部图的对称化 Laplacian:
      L_l = D_l - A_l_sym,   其中 A_l_sym = [0, A_l; A_l^T, 0],
      D_l = diag(A_l_sym · 1)

  **谱间隙** (spectral gap):
      λ_gap^(l) = λ_1(L_l) / λ_max(L_l)    (归一化第一非零特征值)

  Ramanujan 条件要求:  λ_2(A_l_sym) ≤ 2√(d-1)  (d = degree)
  → 谱间隙越大 → 随机游走混合越快 → 梯度在稀疏图上传播越均匀

  对于量化网络，STE 的有效信噪比 ∝ 2^{mean(b_k)}。定义：
      Q_eff^(l) = mean(b_k^(l)[active]) · log(2)     (有效量化精度)

  **单层可训练性**:
      TI_l = λ_gap^(l) · Q_eff^(l) · σ_min(W_l[active])

  **全网络可训练性** (链式梯度流):
      TI = Π_l TI_l   →  log(TI) = Σ_l log(TI_l)

  判定：log(TI) > τ_train  ⟹  可训练
        （τ_train 由满秩无量化基线标定）

### 判据 2：最优解可达性 (Optimality Reachability Index, ORI)

来源：有效表示容量理论 (Arora et al. 2018, Lottery Ticket Hypothesis)
      + Fisher 信息矩阵的迹范数界。

  稀疏量化网络的 **有效容量**:
      C_eff = Σ_l Σ_{(i,j) active} b_k^(l)[i,j]     (= eBOPs / b_a)

  **连通性约束**：从输入到每个输出类的 **有效路径数**:
      P_eff = 从 input 到 output 的 edge-disjoint 路径数

  **Cheeger 常数** (discrete isoperimetric constant):
      h(G_l) = min_{|S|≤n/2} |∂S| / |S|
  其中 ∂S = 从 S 到 V\\S 的边集。Cheeger 不等式：
      λ_gap / 2 ≤ h ≤ √(2 · λ_gap)
  → h 越大 → 图的 "扩张性" 越好 → 信息不被瓶颈阻断

  **层间秩保持**:  rank_ratio_l = rank(W_l[active]) / min(n_{l-1}, n_l)

  **最优解可达索引**:
      ORI = (C_eff / C_baseline) · (P_eff / P_baseline) · Π_l rank_ratio_l · Π_l h(G_l)

  判定：log(ORI) > τ_opt ⟹ 最优解可达
        （τ_opt 由全精度 baseline 标定）

使用方式
--------
  cd FPL/jsc
  python compare_spectral_vs_natural_topology.py

输出
----
  results/spectral_vs_natural/
    ├── topology_target{ebops}_{method}.png            (circle graph)
    ├── topology_matrix_target{ebops}_{method}.png     (matrix plot)
    ├── comparison_summary.log                         (文本日志)
    └── metrics_table.csv
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 保证相对路径基于脚本目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
import re
import csv
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

import keras
import numpy as np
import tensorflow as tf

from data.data import get_data
import model.model as _   # noqa: F401  # 注册自定义层
from hgq.layers import QLayerBase
from utils.ramanujan_budget_utils import (
    _flatten_layers,
    _get_kq_var,
    compute_bw_aware_degree,
    SensitivityAwarePruner,
)
from utils.topology_graph_plot_utils import TopologyGraphPlotter
from run_one_shot_prune_only import (
    spectral_quant_prune_to_ebops,
    bisect_ebops_to_target,
    compute_model_ebops,
    _forward_update_ebops_no_bn_drift,
    _dense_prunable_layers,
    _layer_active_mask,
    build_sample_input,
)

np.random.seed(42)
tf.random.set_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_CKPT = "results/baseline/epoch=7789-val_acc=0.770-ebops=19899-val_loss=0.641.keras"
BASELINE_DIR  = Path("results/baseline")
INPUT_H5      = "data/dataset.h5"
OUTPUT_DIR    = Path("results/spectral_vs_natural")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EBOPS_LIST = [400, 1000, 1500, 2000, 2500, 6800, 12000, 24000]

# ══════════════════════════════════════════════════════════════════════════════
# 日志设置
# ══════════════════════════════════════════════════════════════════════════════

log_path = OUTPUT_DIR / "comparison_summary.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayerMetrics:
    name: str
    shape: tuple
    # 拓扑
    n_active: int = 0
    n_total: int = 0
    sparsity: float = 0.0
    col_degree_mean: float = 0.0
    col_degree_std: float = 0.0
    row_degree_mean: float = 0.0
    row_orphan_inputs: int = 0
    # 谱性质
    spectral_gap: float = 0.0        # λ_gap = λ_1 / λ_max  (归一化谱间隙)
    lambda_2_adj: float = 0.0        # 邻接矩阵第二大特征值
    ramanujan_bound: float = 0.0     # 2*sqrt(d-1)
    is_ramanujan: bool = False       # λ_2 ≤ 2*sqrt(d-1)
    cheeger_lower: float = 0.0       # Cheeger 常数下界 = λ_gap / 2
    cheeger_upper: float = 0.0       # Cheeger 常数上界 = sqrt(2*λ_gap)
    sigma_min: float = 0.0           # 最小奇异值 (活跃子矩阵)
    sigma_max: float = 0.0           # 最大奇异值
    condition_number: float = 0.0    # σ_max / σ_min
    effective_rank: float = 0.0      # exp(H(σ)) Shannon entropy of singular values
    rank_ratio: float = 0.0          # rank / min(m,n)
    # 量化性质
    mean_bk_active: float = 0.0
    std_bk_active: float = 0.0
    min_bk_active: float = 0.0
    max_bk_active: float = 0.0
    q_eff: float = 0.0              # mean(b_k[active]) * log(2)
    n_dead_quant: int = 0           # b_k ≤ 0.1 的连接数
    # 可训练性
    TI_l: float = 0.0               # 单层可训练性 = λ_gap · Q_eff · σ_min
    log_TI_l: float = 0.0


@dataclass
class ModelMetrics:
    method: str
    target_ebops: int
    measured_ebops: float = 0.0
    val_acc: float = 0.0
    val_loss: float = 0.0
    source_path: str = ""
    # 全网络指标
    log_TI: float = 0.0             # Σ log(TI_l)  可训练性
    C_eff: float = 0.0              # 有效容量
    P_eff: float = 0.0              # 有效路径数
    rank_product: float = 0.0       # Π rank_ratio_l
    cheeger_product: float = 0.0    # Π h(G_l)
    log_ORI: float = 0.0            # log(最优解可达索引)
    trainable_verdict: str = ""
    optimal_verdict: str = ""
    layers: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# 谱分析核心函数
# ══════════════════════════════════════════════════════════════════════════════

def _bipartite_spectral_gap(adj_2d: np.ndarray) -> tuple[float, float, float]:
    """计算二部图的归一化 Laplacian 谱间隙。

    输入: adj_2d (m, n) — 非负邻接矩阵 (bit × |weight|)
    输出: (spectral_gap, lambda_2_adj, lambda_max_adj)

    方法: 使用归一化 Laplacian L_norm = I - D^{-1/2} A_sym D^{-1/2}
    其中 A_sym = [[0, A], [A^T, 0]]。

    归一化 Laplacian 的特征值 ∈ [0, 2]。
    对连通图: λ_0 = 0, λ_1 > 0。
    谱间隙 = λ_1(L_norm) = 第一个非零特征值。

    对二部图: λ_max(L_norm) = 2（对称谱性质）。
    因此 λ_1/λ_max = λ_1/2 直接度量混合速率。

    同时返回 A_sym 的奇异值信息用于 Ramanujan 判定。

    理论意义:
      - Alon-Boppana 界: 对于 d-正则二部图, λ_1(L_norm) ≤ 1 - 2√(d-1)/d
        当 λ_1 达到此界时为 Ramanujan 图
      - 混合时间 ∝ 1/λ_1
      - Cheeger 不等式: λ_1/2 ≤ h(G) ≤ √(2·λ_1)
    """
    m, n = adj_2d.shape
    N = m + n

    # 构造对称化邻接矩阵
    A_sym = np.zeros((N, N), dtype=np.float64)
    A_sym[:m, m:] = adj_2d
    A_sym[m:, :m] = adj_2d.T

    # 度矩阵
    deg = np.sum(A_sym, axis=1)
    if np.all(deg < 1e-12):
        return 0.0, 0.0, 0.0

    # 归一化 Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
    # 避免除零
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 1e-12
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    L_norm = np.eye(N, dtype=np.float64) - D_inv_sqrt @ A_sym @ D_inv_sqrt

    # 计算特征值
    eigs_L = np.sort(np.linalg.eigvalsh(L_norm))  # 升序

    # 找第一个显著非零特征值 (λ_1)
    # 对于连通图: 只有一个 0 特征值
    # 对于多连通分量: 有多个 0 特征值
    tol = 1e-8
    nonzero_eigs = eigs_L[eigs_L > tol]
    lambda_1 = float(nonzero_eigs[0]) if len(nonzero_eigs) > 0 else 0.0

    # A_sym 的特征值用于 Ramanujan 判定
    eigs_A = np.sort(np.linalg.eigvalsh(A_sym))
    # 对二部图，特征值关于 0 对称，只看非负部分
    eigs_A_pos = np.sort(eigs_A[eigs_A > tol])[::-1]
    lambda_max_adj = float(eigs_A_pos[0]) if len(eigs_A_pos) > 0 else 0.0
    # 第二大特征值 = 第二个不同的正特征值
    lambda_2_adj = float(eigs_A_pos[1]) if len(eigs_A_pos) > 1 else 0.0

    # 谱间隙 = λ_1(L_norm)  (∈ [0, 2])
    spectral_gap = lambda_1
    return spectral_gap, lambda_2_adj, lambda_max_adj


def _compute_sigma_stats(W_active_2d: np.ndarray) -> dict:
    """计算活跃子矩阵的奇异值统计。"""
    if W_active_2d.size == 0 or W_active_2d.shape[0] == 0 or W_active_2d.shape[1] == 0:
        return {
            'sigma_min': 0.0, 'sigma_max': 0.0,
            'condition_number': float('inf'),
            'effective_rank': 0.0, 'rank_ratio': 0.0,
        }

    try:
        s = np.linalg.svd(W_active_2d.astype(np.float64), compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            'sigma_min': 0.0, 'sigma_max': 0.0,
            'condition_number': float('inf'),
            'effective_rank': 0.0, 'rank_ratio': 0.0,
        }

    s = s[s > 1e-12]
    if len(s) == 0:
        return {
            'sigma_min': 0.0, 'sigma_max': 0.0,
            'condition_number': float('inf'),
            'effective_rank': 0.0, 'rank_ratio': 0.0,
        }

    sigma_min = float(s[-1])
    sigma_max = float(s[0])
    cond = sigma_max / max(sigma_min, 1e-12)

    # Effective rank via Shannon entropy of normalized singular values
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p + 1e-30))
    eff_rank = math.exp(entropy)

    min_dim = min(W_active_2d.shape)
    numerical_rank = int(np.sum(s > sigma_max * 1e-5))
    rank_ratio = numerical_rank / max(min_dim, 1)

    return {
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'condition_number': cond,
        'effective_rank': eff_rank,
        'rank_ratio': min(rank_ratio, 1.0),
    }


def _effective_path_count(masks: list[np.ndarray]) -> float:
    """计算从输入层到输出层的有效路径数。

    通过布尔矩阵乘法逐层传播路径数（截断以防溢出）。
    """
    if not masks:
        return 0.0
    path_counts = np.ones(masks[0].shape[0], dtype=np.float64)
    for mask in masks:
        bool_mask = (mask > 0.5).astype(np.float64)
        path_counts = np.clip(path_counts @ bool_mask, 0.0, 1e30)
    return float(np.sum(path_counts))


def _compute_cheeger_estimate(spectral_gap: float) -> tuple[float, float]:
    """利用离散 Cheeger 不等式估计扩张常数。

    Cheeger 不等式 (对归一化 Laplacian):
        λ_1 / 2  ≤  h(G)  ≤  √(2 · λ_1)

    其中 λ_1 = spectral_gap (归一化 Laplacian 的第一个非零特征值)
    """
    lower = spectral_gap / 2.0
    upper = math.sqrt(2.0 * max(spectral_gap, 0.0))
    return lower, upper


def analyze_layer(layer, kernel: np.ndarray) -> LayerMetrics:
    """分析单层的拓扑、谱和量化性质。"""
    kq = layer.kq
    shape = kernel.shape
    assert kernel.ndim == 2
    m, n = shape

    # 提取 bit-width
    b_var = _get_kq_var(kq, 'b')
    if b_var is None:
        b_var = _get_kq_var(kq, 'f')
    bits = b_var.numpy().astype(np.float32) if b_var is not None else np.zeros_like(kernel)

    # 活跃连接掩码
    active_mask = _layer_active_mask(layer)
    n_active = int(np.sum(active_mask))
    n_total = int(np.prod(shape))
    sparsity = 1.0 - n_active / max(n_total, 1)

    # 度统计
    col_deg = np.sum(active_mask, axis=0)    # 每个输出的输入数
    row_deg = np.sum(active_mask, axis=1)    # 每个输入的输出数
    col_degree_mean = float(np.mean(col_deg))
    col_degree_std = float(np.std(col_deg))
    row_degree_mean = float(np.mean(row_deg))
    row_orphan = int(np.sum(row_deg == 0))

    # 构造有效邻接矩阵: A = bits ⊙ |W|
    adj = np.clip(bits, 0, None) * np.abs(kernel)
    adj = adj * active_mask  # 确保仅活跃连接

    # 谱间隙
    spectral_gap, lambda_2, lambda_max = _bipartite_spectral_gap(adj)

    # Ramanujan 判定: λ_2 ≤ 2*sqrt(d-1)
    d = max(col_degree_mean, 1.0)
    ram_bound = 2.0 * math.sqrt(max(d - 1.0, 0.0))
    is_ram = lambda_2 <= ram_bound + 1e-6

    # Cheeger 常数
    cheeger_lo, cheeger_hi = _compute_cheeger_estimate(spectral_gap)

    # 奇异值分析（仅活跃连接子矩阵）
    # 提取活跃子矩阵: 保留有活跃连接的行和列
    active_rows = np.any(active_mask, axis=1)
    active_cols = np.any(active_mask, axis=0)
    W_sub = kernel[np.ix_(active_rows, active_cols)]
    mask_sub = active_mask[np.ix_(active_rows, active_cols)]
    W_active = W_sub * mask_sub
    svd_stats = _compute_sigma_stats(W_active)

    # 量化统计 (仅活跃连接)
    bits_active = bits[active_mask]
    if bits_active.size > 0:
        mean_bk = float(np.mean(bits_active))
        std_bk = float(np.std(bits_active))
        min_bk = float(np.min(bits_active))
        max_bk = float(np.max(bits_active))
    else:
        mean_bk = std_bk = min_bk = max_bk = 0.0
    q_eff = mean_bk * math.log(2.0)
    n_dead_q = int(np.sum(bits_active <= 0.1)) if bits_active.size > 0 else 0

    # 单层可训练性: TI_l = λ_gap · Q_eff · σ_min
    TI_l = spectral_gap * q_eff * svd_stats['sigma_min']
    log_TI_l = math.log(max(TI_l, 1e-30))

    return LayerMetrics(
        name=layer.name,
        shape=shape,
        n_active=n_active,
        n_total=n_total,
        sparsity=sparsity,
        col_degree_mean=col_degree_mean,
        col_degree_std=col_degree_std,
        row_degree_mean=row_degree_mean,
        row_orphan_inputs=row_orphan,
        spectral_gap=spectral_gap,
        lambda_2_adj=lambda_2,
        ramanujan_bound=ram_bound,
        is_ramanujan=is_ram,
        cheeger_lower=cheeger_lo,
        cheeger_upper=cheeger_hi,
        sigma_min=svd_stats['sigma_min'],
        sigma_max=svd_stats['sigma_max'],
        condition_number=svd_stats['condition_number'],
        effective_rank=svd_stats['effective_rank'],
        rank_ratio=svd_stats['rank_ratio'],
        mean_bk_active=mean_bk,
        std_bk_active=std_bk,
        min_bk_active=min_bk,
        max_bk_active=max_bk,
        q_eff=q_eff,
        n_dead_quant=n_dead_q,
        TI_l=TI_l,
        log_TI_l=log_TI_l,
    )


def analyze_model(
    model,
    sample_input,
    method: str,
    target_ebops: int,
    source_path: str = "",
    x_val=None,
    y_val=None,
) -> ModelMetrics:
    """分析整个模型的拓扑、谱和量化性质。"""
    measured = compute_model_ebops(model, sample_input)

    # 评估精度
    val_acc = 0.0
    val_loss = 0.0
    if x_val is not None and y_val is not None:
        logits = model(x_val, training=False)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        val_loss = float(loss_fn(y_val, logits).numpy())
        preds = np.argmax(logits.numpy(), axis=-1)
        val_acc = float(np.mean(preds == np.array(y_val).ravel()))

    layers = _dense_prunable_layers(model)
    layer_metrics = []
    masks = []
    for layer in layers:
        kernel = layer.kernel.numpy().astype(np.float32)
        lm = analyze_layer(layer, kernel)
        layer_metrics.append(lm)
        masks.append((_layer_active_mask(layer)).astype(np.float32))

    # 有效路径数
    P_eff = _effective_path_count(masks)

    # 有效容量 C_eff = Σ Σ b_k[active]
    C_eff = 0.0
    for layer in layers:
        kq = layer.kq
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            bits = b_var.numpy()
            C_eff += float(np.sum(np.clip(bits, 0, None)))

    # 全网络可训练性: log(TI) = Σ log(TI_l)
    log_TI = sum(lm.log_TI_l for lm in layer_metrics)

    # Rank product
    rank_prod = 1.0
    for lm in layer_metrics:
        rank_prod *= max(lm.rank_ratio, 1e-10)

    # Cheeger product
    cheeger_prod = 1.0
    for lm in layer_metrics:
        cheeger_prod *= max(lm.cheeger_lower, 1e-10)

    # ORI = (C_eff / C_baseline) · (P_eff / P_baseline) · Π rank_ratio · Π h(G_l)
    # 这里 C_baseline 和 P_baseline 在后续标定
    log_ORI = (
        math.log(max(C_eff, 1e-10))
        + math.log(max(P_eff, 1e-10))
        + math.log(max(rank_prod, 1e-30))
        + math.log(max(cheeger_prod, 1e-30))
    )

    return ModelMetrics(
        method=method,
        target_ebops=target_ebops,
        measured_ebops=measured,
        val_acc=val_acc,
        val_loss=val_loss,
        source_path=source_path,
        log_TI=log_TI,
        C_eff=C_eff,
        P_eff=P_eff,
        rank_product=rank_prod,
        cheeger_product=cheeger_prod,
        log_ORI=log_ORI,
        trainable_verdict="",
        optimal_verdict="",
        layers=layer_metrics,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 查找最近的 baseline checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def _extract_ebops_from_name(name: str) -> int | None:
    m = re.search(r'ebops=(\d+)', name)
    return int(m.group(1)) if m else None


def _extract_val_acc_from_name(name: str) -> float | None:
    m = re.search(r'val_acc=([0-9.]+)', name)
    return float(m.group(1)) if m else None


def find_best_baseline_ckpt(baseline_dir: Path, target_ebops: int, max_ratio: float = 1.5) -> tuple[Path, int, float]:
    """从 baseline 目录找最接近 target_ebops 且 ebops ≤ target * max_ratio 的检查点。

    优先选 ebops ≥ target 中最接近的；如果没有，取 ebops < target 中最接近的。
    """
    candidates = []
    for p in baseline_dir.glob("*.keras"):
        e = _extract_ebops_from_name(p.name)
        a = _extract_val_acc_from_name(p.name)
        if e is None:
            continue
        candidates.append((p, e, a or 0.0))

    if not candidates:
        raise FileNotFoundError(f"No baseline checkpoints found in {baseline_dir}")

    # 优先选 ebops >= target, 距离最近, 且在合理范围内
    above = [(p, e, a) for p, e, a in candidates if e >= target_ebops and e <= target_ebops * max_ratio]
    below = [(p, e, a) for p, e, a in candidates if e < target_ebops]

    if above:
        above.sort(key=lambda x: (abs(x[1] - target_ebops), -x[2]))
        return above[0]
    elif below:
        below.sort(key=lambda x: (abs(x[1] - target_ebops), -x[2]))
        return below[0]
    else:
        # 所有都太大，取最小的
        candidates.sort(key=lambda x: x[1])
        return candidates[0]


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 80)
    logger.info("对比 spectral_quant 剪枝 vs 自然训练（gradual）模型拓扑")
    logger.info("=" * 80)

    # ── 理论公式说明 ──────────────────────────────────────────────────────
    logger.info("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         理论框架：定量评估指标                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ▎判据 1: 可训练性指标 (Trainability Index, TI)                             ║
║                                                                              ║
║  基础理论:                                                                   ║
║  ① 谱间隙理论 (Alon 1986): 对于 d-正则二部图 G, 其归一化 Laplacian          ║
║     的最小非零特征值 λ₁(L) 控制了随机游走的混合速率:                         ║
║       mixing time ∝ 1/λ₁                                                    ║
║     对应到神经网络: 梯度在稀疏图上的传播均匀性 ∝ λ_gap                      ║
║                                                                              ║
║  ② STE 信噪比 (Hubara et al. 2016): 量化器的直通估计器 (STE) 的             ║
║     有效信噪比:                                                              ║
║       SNR_STE ∝ 2^{b_k}                                                     ║
║     定义 Q_eff = mean(b_k[active]) · ln(2)                                  ║
║                                                                              ║
║  ③ 权重矩阵的秩条件: σ_min(W) > 0 保证梯度不在该层消失                     ║
║     (Saxe et al. 2014, "Exact solutions to nonlinear dynamics")              ║
║                                                                              ║
║  单层可训练性:                                                               ║
║       TI_l = λ_gap^(l) · Q_eff^(l) · σ_min(W_l)                            ║
║                                                                              ║
║  全网络可训练性 (链式法则, 梯度流):                                          ║
║       log(TI) = Σ_l log(TI_l)                                               ║
║                                                                              ║
║  TI > 0 且 log(TI) > τ_train ⟹ 网络可训练                                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ▎判据 2: 最优解可达性 (Optimality Reachability Index, ORI)                 ║
║                                                                              ║
║  基础理论:                                                                   ║
║  ① 有效容量 (Arora et al. 2018):                                            ║
║       C_eff = Σ_l Σ_{(i,j)∈active} b_k[i,j]                                ║
║     表示网络能编码的有效信息位数                                             ║
║                                                                              ║
║  ② Lottery Ticket Hypothesis (Frankle & Carlin 2019):                       ║
║     稀疏子网络能否达到稠密网络精度取决于:                                    ║
║     a) 有效路径数 P_eff — 从 input 到 output 的独立路径总数                  ║
║     b) 路径多样性 — 通过不同中间节点的路径                                   ║
║                                                                              ║
║  ③ 图的扩张性 (Cheeger 不等式, Alon-Milman 1985):                           ║
║       λ_gap/2  ≤  h(G)  ≤  √(2·λ_gap)                                      ║
║     h(G) = min_{|S|≤n/2} |∂S|/|S|  (discrete isoperimetric constant)        ║
║     h 越大 → 信息瓶颈越少 → 表示能力损失越小                                ║
║                                                                              ║
║  ④ 层间秩保持 (Pennington & Worah 2017):                                    ║
║       rank_ratio_l = rank(W_l[active]) / min(n_{l-1}, n_l)                  ║
║     秩塌缩 → 输出空间维度不足 → 无法拟合最优解                              ║
║                                                                              ║
║  最优解可达索引:                                                             ║
║       ORI = (C_eff/C_ref)·(P_eff/P_ref)·Π_l rank_ratio_l · Π_l h(G_l)      ║
║                                                                              ║
║  log(ORI) > τ_opt ⟹ 最优解可达                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # ── 加载数据 ──────────────────────────────────────────────────────────
    logger.info("加载数据...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(INPUT_H5, src='openml')
    sample_input = tf.constant(X_val[:512], dtype=tf.float32)
    x_eval = tf.constant(X_val[:4096], dtype=tf.float32)
    y_eval = tf.constant(y_val[:4096], dtype=tf.int32)

    # ── 生成拓扑绘图器 ───────────────────────────────────────────────────
    plotter = TopologyGraphPlotter(
        symmetric_topology_plot=False,
        mirror_edges=False,
        plot_matrix=True,
        strict_original_connections=True,
    )

    # ── 分析 baseline full model 作为参考 ────────────────────────────────
    logger.info(f"\n加载 baseline 完整模型: {BASELINE_CKPT}")
    baseline_model = keras.models.load_model(BASELINE_CKPT, compile=False)
    baseline_metrics = analyze_model(
        baseline_model, sample_input, "baseline_full",
        target_ebops=24000,
        source_path=BASELINE_CKPT,
        x_val=x_eval, y_val=y_eval,
    )
    logger.info(f"  Baseline: ebops={baseline_metrics.measured_ebops:.0f}, "
                f"acc={baseline_metrics.val_acc:.4f}, "
                f"C_eff={baseline_metrics.C_eff:.1f}, "
                f"P_eff={baseline_metrics.P_eff:.1e}, "
                f"log_TI={baseline_metrics.log_TI:.3f}")
    C_baseline = baseline_metrics.C_eff
    P_baseline = baseline_metrics.P_eff
    log_TI_baseline = baseline_metrics.log_TI
    del baseline_model

    # ── 标定阈值 ─────────────────────────────────────────────────────────
    # τ_train: 基线的 80% (对数域)
    tau_train = log_TI_baseline - abs(log_TI_baseline) * 0.5
    # τ_opt: 在对数域，基线ORI减去合理margin
    tau_opt = 0.0  # 将在首次计算ORI后标定

    logger.info(f"\n  可训练性阈值 τ_train = {tau_train:.3f}")
    logger.info(f"  (基线 log_TI = {log_TI_baseline:.3f})")

    # ── 逐目标分析 ───────────────────────────────────────────────────────
    all_metrics: list[ModelMetrics] = []

    for target_ebops in TARGET_EBOPS_LIST:
        logger.info(f"\n{'='*80}")
        logger.info(f"  TARGET eBOPs = {target_ebops}")
        logger.info(f"{'='*80}")

        for method in ['gradual', 'spectral_quant']:
            logger.info(f"\n  ── 方法: {method} ──")

            if method == 'gradual':
                # 从 baseline 目录找最接近的自然训练检查点
                try:
                    ckpt_path, ckpt_ebops, ckpt_acc = find_best_baseline_ckpt(
                        BASELINE_DIR, target_ebops, max_ratio=2.5
                    )
                except FileNotFoundError as exc:
                    logger.warning(f"  跳过 gradual@{target_ebops}: {exc}")
                    continue

                logger.info(f"  加载自然训练检查点: {ckpt_path.name}")
                logger.info(f"    (ebops={ckpt_ebops}, val_acc={ckpt_acc:.3f})")
                model = keras.models.load_model(str(ckpt_path), compile=False)
                source = str(ckpt_path)
            else:
                # spectral_quant: 从基线剪枝
                logger.info(f"  从基线剪枝到 target={target_ebops}...")
                model = keras.models.load_model(BASELINE_CKPT, compile=False)
                measured, _ = spectral_quant_prune_to_ebops(
                    model, float(target_ebops), sample_input,
                    min_degree=2, b_floor=0.25, b_ceiling=6.0,
                    verbose=True,
                )
                # 二分校准到精确 eBOPs
                calibrated = bisect_ebops_to_target(
                    model, float(target_ebops), sample_input,
                    tolerance=0.05, max_iter=30,
                )
                source = f"pruned_from:{BASELINE_CKPT}"
                logger.info(f"  校准后 eBOPs = {calibrated:.0f}")

            # 分析
            metrics = analyze_model(
                model, sample_input, method, target_ebops,
                source_path=source,
                x_val=x_eval, y_val=y_eval,
            )

            # 计算归一化 ORI
            metrics.log_ORI = (
                math.log(max(metrics.C_eff / max(C_baseline, 1.0), 1e-30))
                + math.log(max(metrics.P_eff / max(P_baseline, 1.0), 1e-30))
                + math.log(max(metrics.rank_product, 1e-30))
                + math.log(max(metrics.cheeger_product, 1e-30))
            )

            # 判定
            metrics.trainable_verdict = (
                "TRAINABLE" if metrics.log_TI > tau_train else "NOT_TRAINABLE"
            )
            # τ_opt: 对数ORI > -20 (经验上很宽松的界,实际由理论确定)
            if tau_opt == 0.0:
                tau_opt = metrics.log_ORI - abs(metrics.log_ORI) * 0.5
            metrics.optimal_verdict = (
                "REACHABLE" if metrics.log_ORI > tau_opt else "UNREACHABLE"
            )

            all_metrics.append(metrics)

            # ── 打印拓扑图 ───────────────────────────────────────────────
            layer_data = plotter.extract_layer_graph_data(model)
            subtitle = (
                f"eBOPs={metrics.measured_ebops:.0f}  "
                f"acc={metrics.val_acc:.4f}  "
                f"log(TI)={metrics.log_TI:.2f}  "
                f"log(ORI)={metrics.log_ORI:.2f}"
            )
            # Circle graph
            circle_path = OUTPUT_DIR / f"topology_target{target_ebops}_{method}_circle.png"
            plotter.plot_circle_graph(
                layer_data, circle_path,
                title=f"{method} @ target={target_ebops}",
                subtitle=subtitle,
            )
            # Matrix plot
            matrix_path = OUTPUT_DIR / f"topology_target{target_ebops}_{method}_matrix.png"
            plotter.plot_weighted_topology_matrices(
                layer_data, matrix_path,
                title=f"{method} @ target={target_ebops}",
                subtitle=subtitle,
            )
            logger.info(f"  拓扑图已保存: {circle_path.name}, {matrix_path.name}")

            # ── 打印详细指标到日志 ───────────────────────────────────────
            _print_metrics(metrics, tau_train, tau_opt)

            del model
            keras.backend.clear_session()

    # ── 设置最终 τ_opt 并重新判定 ────────────────────────────────────────
    # 用所有 gradual 方法的最大 log_ORI 作为参考
    gradual_oris = [m.log_ORI for m in all_metrics if m.method == 'gradual']
    if gradual_oris:
        ori_ref = max(gradual_oris)
        tau_opt_final = ori_ref - abs(ori_ref) * 0.3
        logger.info(f"\n最终 τ_opt 标定 = {tau_opt_final:.3f} (基于 gradual 最佳 ORI = {ori_ref:.3f})")
        for m in all_metrics:
            m.optimal_verdict = "REACHABLE" if m.log_ORI > tau_opt_final else "UNREACHABLE"
    else:
        tau_opt_final = tau_opt

    # ── 汇总表 ───────────────────────────────────────────────────────────
    _print_summary_table(all_metrics, tau_train, tau_opt_final)

    # ── 保存 CSV ─────────────────────────────────────────────────────────
    _save_csv(all_metrics, OUTPUT_DIR / "metrics_table.csv")

    logger.info(f"\n所有结果已保存到: {OUTPUT_DIR}")
    logger.info("完成。")


def _print_metrics(m: ModelMetrics, tau_train: float, tau_opt: float):
    """打印单个模型的详细指标。"""
    logger.info(f"\n  ┌─────────── {m.method} @ target={m.target_ebops} ───────────┐")
    logger.info(f"  │  measured eBOPs:  {m.measured_ebops:.0f}")
    logger.info(f"  │  val_acc:         {m.val_acc:.4f}")
    logger.info(f"  │  val_loss:        {m.val_loss:.4f}")
    logger.info(f"  │")
    logger.info(f"  │  ── 全网络指标 ──")
    logger.info(f"  │  有效容量 C_eff:                {m.C_eff:.1f}")
    logger.info(f"  │  有效路径数 P_eff:              {m.P_eff:.3e}")
    logger.info(f"  │  秩乘积 Π(rank_ratio):         {m.rank_product:.6f}")
    logger.info(f"  │  Cheeger乘积 Π(h_lower):       {m.cheeger_product:.6e}")
    logger.info(f"  │")
    logger.info(f"  │  ★ log(TI) = {m.log_TI:.4f}    (τ_train={tau_train:.4f})  → {m.trainable_verdict}")
    logger.info(f"  │  ★ log(ORI) = {m.log_ORI:.4f}   (τ_opt={tau_opt:.4f})  → {m.optimal_verdict}")
    logger.info(f"  │")

    for lm in m.layers:
        logger.info(f"  │  ── 层: {lm.name} ({lm.shape[0]}×{lm.shape[1]}) ──")
        logger.info(f"  │    活跃连接: {lm.n_active}/{lm.n_total}  "
                     f"(稀疏度={lm.sparsity:.1%})")
        logger.info(f"  │    列度: mean={lm.col_degree_mean:.2f} ± {lm.col_degree_std:.2f}")
        logger.info(f"  │    行度: mean={lm.row_degree_mean:.2f}  孤立输入={lm.row_orphan_inputs}")
        logger.info(f"  │    [谱] λ_gap={lm.spectral_gap:.4f}  "
                     f"λ₂={lm.lambda_2_adj:.4f}  "
                     f"Ramanujan界={lm.ramanujan_bound:.2f}  "
                     f"满足={'✓' if lm.is_ramanujan else '✗'}")
        logger.info(f"  │    [谱] Cheeger: [{lm.cheeger_lower:.4f}, {lm.cheeger_upper:.4f}]")
        logger.info(f"  │    [SVD] σ_min={lm.sigma_min:.4e}  σ_max={lm.sigma_max:.4e}  "
                     f"κ={lm.condition_number:.1f}  "
                     f"eff_rank={lm.effective_rank:.2f}  "
                     f"rank_ratio={lm.rank_ratio:.3f}")
        logger.info(f"  │    [量化] mean_bk={lm.mean_bk_active:.3f} ± {lm.std_bk_active:.3f}  "
                     f"range=[{lm.min_bk_active:.2f}, {lm.max_bk_active:.2f}]  "
                     f"dead={lm.n_dead_quant}")
        logger.info(f"  │    [量化] Q_eff={lm.q_eff:.4f}")
        logger.info(f"  │    ★ TI_l = λ_gap·Q_eff·σ_min = "
                     f"{lm.spectral_gap:.4f} × {lm.q_eff:.4f} × {lm.sigma_min:.4e} = {lm.TI_l:.4e}")
        logger.info(f"  │    ★ log(TI_l) = {lm.log_TI_l:.4f}")

    logger.info(f"  └{'─'*50}┘")


def _print_summary_table(all_metrics: list[ModelMetrics], tau_train: float, tau_opt: float):
    """打印汇总对比表。"""
    logger.info(f"\n{'='*120}")
    logger.info("                                      汇   总   对   比   表")
    logger.info(f"{'='*120}")
    logger.info(f"  τ_train = {tau_train:.4f}    τ_opt = {tau_opt:.4f}")
    logger.info(f"{'─'*120}")

    header = (
        f"  {'target':>7}  {'method':<16}  {'meas_ebops':>10}  {'val_acc':>8}  "
        f"{'C_eff':>8}  {'P_eff':>12}  {'rank_Π':>10}  {'cheeger_Π':>12}  "
        f"{'log(TI)':>9}  {'log(ORI)':>10}  {'train?':<14}  {'optimal?':<12}"
    )
    logger.info(header)
    logger.info(f"{'─'*120}")

    # 按 target_ebops 分组
    targets_seen = sorted(set(m.target_ebops for m in all_metrics))
    for t in targets_seen:
        for m in all_metrics:
            if m.target_ebops != t:
                continue
            row = (
                f"  {m.target_ebops:>7}  {m.method:<16}  {m.measured_ebops:>10.0f}  {m.val_acc:>8.4f}  "
                f"{m.C_eff:>8.1f}  {m.P_eff:>12.3e}  {m.rank_product:>10.6f}  {m.cheeger_product:>12.6e}  "
                f"{m.log_TI:>9.3f}  {m.log_ORI:>10.3f}  {m.trainable_verdict:<14}  {m.optimal_verdict:<12}"
            )
            logger.info(row)
        logger.info(f"{'─'*120}")

    logger.info("")

    # 同一目标下对比
    logger.info("逐目标对比分析:")
    for t in targets_seen:
        group = [m for m in all_metrics if m.target_ebops == t]
        gradual = [m for m in group if m.method == 'gradual']
        spectral = [m for m in group if m.method == 'spectral_quant']
        if gradual and spectral:
            g, s = gradual[0], spectral[0]
            logger.info(f"\n  target={t}:")
            logger.info(f"    精度:   gradual={g.val_acc:.4f}  spectral={s.val_acc:.4f}  "
                         f"Δ={s.val_acc - g.val_acc:+.4f}")
            logger.info(f"    log(TI): gradual={g.log_TI:.3f}  spectral={s.log_TI:.3f}  "
                         f"Δ={s.log_TI - g.log_TI:+.3f}")
            logger.info(f"    log(ORI): gradual={g.log_ORI:.3f}  spectral={s.log_ORI:.3f}  "
                         f"Δ={s.log_ORI - g.log_ORI:+.3f}")
            logger.info(f"    容量:   gradual={g.C_eff:.1f}  spectral={s.C_eff:.1f}")
            logger.info(f"    路径数: gradual={g.P_eff:.3e}  spectral={s.P_eff:.3e}")

            # 逐层对比谱间隙
            if len(g.layers) == len(s.layers):
                for gl, sl in zip(g.layers, s.layers):
                    logger.info(f"    {gl.name}:")
                    logger.info(f"      谱间隙:  g={gl.spectral_gap:.4f}  s={sl.spectral_gap:.4f}")
                    logger.info(f"      σ_min:   g={gl.sigma_min:.4e}  s={sl.sigma_min:.4e}")
                    logger.info(f"      mean_bk: g={gl.mean_bk_active:.3f}  s={sl.mean_bk_active:.3f}")
                    logger.info(f"      Ramanujan: g={'✓' if gl.is_ramanujan else '✗'}  "
                                 f"s={'✓' if sl.is_ramanujan else '✗'}")


def _save_csv(all_metrics: list[ModelMetrics], csv_path: Path):
    """保存结果到 CSV。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        # 模型级 header
        model_fields = [
            'target_ebops', 'method', 'measured_ebops', 'val_acc', 'val_loss',
            'C_eff', 'P_eff', 'rank_product', 'cheeger_product',
            'log_TI', 'log_ORI', 'trainable_verdict', 'optimal_verdict',
        ]
        # 层级 header (最多4层)
        layer_fields = [
            'n_active', 'sparsity', 'col_degree_mean', 'spectral_gap',
            'lambda_2_adj', 'ramanujan_bound', 'is_ramanujan',
            'cheeger_lower', 'sigma_min', 'sigma_max', 'condition_number',
            'effective_rank', 'rank_ratio',
            'mean_bk_active', 'q_eff', 'n_dead_quant', 'TI_l', 'log_TI_l',
        ]
        max_layers = max(len(m.layers) for m in all_metrics) if all_metrics else 0
        header = model_fields[:]
        for i in range(max_layers):
            header.extend([f"L{i}_{f}" for f in layer_fields])
        w.writerow(header)

        for m in all_metrics:
            row = [
                m.target_ebops, m.method, f'{m.measured_ebops:.2f}',
                f'{m.val_acc:.6f}', f'{m.val_loss:.6f}',
                f'{m.C_eff:.2f}', f'{m.P_eff:.6e}',
                f'{m.rank_product:.8f}', f'{m.cheeger_product:.8e}',
                f'{m.log_TI:.6f}', f'{m.log_ORI:.6f}',
                m.trainable_verdict, m.optimal_verdict,
            ]
            for lm in m.layers:
                row.extend([
                    lm.n_active, f'{lm.sparsity:.6f}',
                    f'{lm.col_degree_mean:.4f}', f'{lm.spectral_gap:.6f}',
                    f'{lm.lambda_2_adj:.6f}', f'{lm.ramanujan_bound:.4f}',
                    int(lm.is_ramanujan),
                    f'{lm.cheeger_lower:.6f}',
                    f'{lm.sigma_min:.6e}', f'{lm.sigma_max:.6e}',
                    f'{lm.condition_number:.4f}', f'{lm.effective_rank:.4f}',
                    f'{lm.rank_ratio:.6f}',
                    f'{lm.mean_bk_active:.6f}', f'{lm.q_eff:.6f}',
                    lm.n_dead_quant,
                    f'{lm.TI_l:.8e}', f'{lm.log_TI_l:.6f}',
                ])
            # 填充缺失层
            for _ in range(max_layers - len(m.layers)):
                row.extend([''] * len(layer_fields))
            w.writerow(row)

    logger.info(f"  CSV 已保存: {csv_path}")


if __name__ == '__main__':
    main()
