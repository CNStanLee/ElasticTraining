#!/usr/bin/env python3
"""
compare_spectral_vs_natural_topology_v2.py
==========================================
对比 spectral_quant 剪枝和自然训练（gradual baseline）模型在多个 eBOPs 目标下
的拓扑结构，并从图谱理论和信息论的角度，定量评估：
  1. 可训练性 (Trainability)
  2. 是否可达 Pareto 前沿精度 (Pareto-frontier Reachability)

然后基于这两个判据，给出**预热膨胀决策**：
当 target eBOPs 过低，需要膨胀到多少再进行 warm-up 训练。

理论框架
--------

### 问题背景

对 HGQ 量化 MLP 进行 one-shot spectral_quant 剪枝时，低 eBOPs 目标可能导致
某些层的活跃连接数极少（如仅 1-2 个），形成严重的 **信息瓶颈**。

模型结构：Input(16) → Dense(64) → Dense(64) → Dense(32) → Dense(5)
每层的权重矩阵 W_l ∈ R^{n_l × n_{l+1}} 伴随：
  - 二值掩码 M_l ∈ {0,1}^{n_l × n_{l+1}} (哪些连接活跃)
  - 比特宽度 b_l ∈ R_+^{n_l × n_{l+1}}    (每个连接的量化精度)

═══════════════════════════════════════════════════════════════════════════════

### 判据 1: 可训练性 — 信息瓶颈秩比 (Information Bottleneck Ratio, IBR)

**理论依据**:

① **复合映射秩不等式** (Composition Rank Inequality):
   对 L 层网络 f = φ_L ∘ σ ∘ φ_{L-1} ∘ ⋯ ∘ σ ∘ φ_0，有：
       rank(f) ≤ min_l rank(φ_l)

② **梯度 Jacobian 秩条件** (Gradient Rank Condition):
   反向传播中，层 l 的梯度 Jacobian J_l = ∂L/∂W_l 的秩 ≤ r_l^s。
   - 当 r_l^s = 1 时：J_l = u·v^T (秩1外积)，所有参数获得相同方向的
     更新信号 → SGD 无法区分不同参数 → 无法训练
   - 当 r_l^s ≥ 2 时：J_l 有 ≥2 个独立方向 → 不同参数获得差异化的
     更新信号 → SGD 可以有效优化

③ **结构秩** (Structural Rank, König 最大匹配定理):
   层 l 的结构秩 r_l^s = rank(M_l) — 二值掩码矩阵的秩。
   这等于二部图 (U_l, V_l, E_l) 的最大匹配数，代表通过该层的
   **最大独立信息通道数**。

   结构秩是上界: 无论权重如何赋值，该层的秩不会超过 r_l^s。

④ **Beta scheduling 可修复拓扑**:
   HGQ 的 kq.k 为可训练变量。训练过程中可重新激活被剪枝的连接，
   从而提升结构秩。但初始秩 ≥ 2 是优化器获得足够梯度信号的前提。

**判据公式**:
    τ_train = 2  (梯度分化最低要求)
    IBR = min_l(r_l^s) / τ_train

    IBR ≥ 1.0 ⟹ 可训练 (每层至少 2 个独立梯度方向)
    IBR < 1.0 ⟹ 不可训练 (存在标量瓶颈层，梯度信号无法分化)

**直觉**: 若某层结构秩 = 1（如仅 1 个活跃连接），则该层将所有输入压缩为
标量——梯度只有 1 个方向，优化器无法区分 64 个神经元中哪个该加强。

═══════════════════════════════════════════════════════════════════════════════

### 判据 2: Pareto 前沿可达性 — 量化信息吞吐比 (Pareto Reachability Index, PRI)

**理论依据**:

① **量化信息吞吐量** (Quantized Information Throughput, QIT):
   层 l 的有效信息吞吐量：
       QIT_l = r_l^s × mean(b_l[active])
   含义：r_l^s 个独立通道 × 每通道平均 b 位精度 = 总有效比特数

② **K 类判别的最低信息需求** (Minimum Information for K-class Comparison):
   K 类 softmax 分类器需要进行 K-1 次成对比较来确定 argmax。
   每次比较需要至少 0.5 有效比特的精度 (足以分辨两个 logit 的大小关系)。
   因此总的最低信息吞吐量:
       τ_reach = (K-1) × 0.5 = (K-1)/2

   STE 理论 (Hubara et al. 2016):
     量化梯度的信噪比 SNR ∝ 2^b。当 QIT < (K-1)/2 时,
     STE 梯度噪声过大, 优化器无法稳定地改善所有 K 类的分类精度。

③ **Beta scheduling 的比特重分配**:
   训练过程中, beta scheduling 可以重新分配比特宽度 b_k, 但不能超过
   结构秩所限制的通道数。因此 PRI 同时依赖拓扑(秩)和量化(比特)。

**判据公式**:
    QIT_l = r_l^s × mean(b_l[active])
    τ_reach = (K-1) / 2
    PRI = min_l(QIT_l) / τ_reach

    PRI ≥ 1.0 ⟹ 可达 Pareto 前沿 (信息吞吐充足)
    PRI < 1.0 ⟹ 可能无法达到 Pareto 前沿

**两判据关系**:
    IBR 是纯拓扑条件 (仅看掩码的秩, 阈值=2)
    PRI 同时考虑拓扑和量化精度 (秩 × 比特, 阈值=(K-1)/2)
    存在 IBR ≥ 1 但 PRI < 1 的情况: 拓扑足够但量化精度太低

═══════════════════════════════════════════════════════════════════════════════

### 预热膨胀决策 (Warm-up Budget Inflation)

当 target eBOPs 过低导致 IBR < 1 或 PRI < 1 时，需要膨胀 eBOPs 预算。
通过二分搜索找到两个边界：

  eBOPs_train = min{E : IBR(E) ≥ 1}    — 最小可训练预算
  eBOPs_reach = min{E : PRI(E) ≥ 1}    — 最小可达前沿预算

推荐预热预算 = max(eBOPs_train, eBOPs_reach)

训练流程：
  1. 以推荐预热预算进行 one-shot 剪枝
  2. Beta scheduling 预热训练 (阶段 1)
  3. 逐步减小 eBOPs 到目标值 (阶段 2)

═══════════════════════════════════════════════════════════════════════════════

### 实验验证 (对本 JSC 数据集, K=5, τ_train=2, τ_reach=2.0)

  | target | method   | min(r_l^s) | IBR  | min(QIT) | PRI  | 实际       |
  |--------|----------|-----------|------|----------|------|------------|
  |   400  | spectral |     1     | 0.50 |   0.78   | 0.39 | 不可训练   |
  |  1000  | spectral |     3     | 1.50 |   2.06   | 1.03 | 可达前沿   |
  |   400  | gradual  |     3     | 1.50 |   4.05   | 2.03 | 可训练     |

使用方式
--------
  cd FPL/jsc
  python compare_spectral_vs_natural_topology_v2.py

输出
----
  results/spectral_vs_natural_v2/
    ├── topology_target{ebops}_{method}_circle.png
    ├── topology_target{ebops}_{method}_matrix.png
    ├── comparison_summary.log
    ├── metrics_table.csv
    └── warmup_decision.txt
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
from dataclasses import dataclass, field

import keras
import numpy as np
import tensorflow as tf

from data.data import get_data
import model.model as _   # noqa: F401  # 注册自定义层
from hgq.layers import QLayerBase
from utils.ramanujan_budget_utils import (
    _flatten_layers,
    _get_kq_var,
)
from utils.topology_graph_plot_utils import TopologyGraphPlotter
from run_one_shot_prune_only import (
    spectral_quant_prune_to_ebops,
    bisect_ebops_to_target,
    compute_model_ebops,
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
OUTPUT_DIR    = Path("results/spectral_vs_natural_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EBOPS_LIST = [400, 1000, 1500, 2000, 2500, 6800, 12000, 24000]
K_CLASSES = 5                # JSC 数据集的类别数
RUN_WARMUP_SEARCH = True     # 是否运行预热膨胀边界搜索
WARMUP_SEARCH_TOL = 50       # 二分搜索精度 (eBOPs)

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
    n_active_outputs: int = 0        # 有 ≥1 活跃输入的输出神经元数
    n_active_inputs: int = 0         # 有 ≥1 活跃输出的输入神经元数
    # 结构秩 (核心指标)
    structural_rank: int = 0         # rank(M_l) — 二值掩码矩阵的秩
    # 量化信息吞吐量
    QIT: float = 0.0                 # r_l^s × mean(b_k[active])
    # 谱性质
    spectral_gap: float = 0.0
    lambda_2_adj: float = 0.0
    ramanujan_bound: float = 0.0
    is_ramanujan: bool = False
    cheeger_lower: float = 0.0
    # 奇异值
    sigma_min: float = 0.0
    sigma_max: float = 0.0
    condition_number: float = 0.0
    effective_rank: float = 0.0
    numerical_rank: int = 0
    # 量化性质
    mean_bk_active: float = 0.0
    std_bk_active: float = 0.0
    min_bk_active: float = 0.0
    max_bk_active: float = 0.0
    n_dead_quant: int = 0


@dataclass
class ModelMetrics:
    method: str
    target_ebops: int
    measured_ebops: float = 0.0
    val_acc: float = 0.0
    val_loss: float = 0.0
    pareto_acc: float = 0.0          # 该 eBOPs 下 Pareto 前沿精度
    source_path: str = ""
    # 核心判据
    min_structural_rank: int = 0     # min_l(r_l^s)
    IBR: float = 0.0                 # min_l(r_l^s) / (K-1)
    bottleneck_layer_ibr: str = ""   # IBR 瓶颈层名称
    min_QIT: float = 0.0             # min_l(QIT_l)
    PRI: float = 0.0                 # min_l(QIT_l) / log2(K)
    bottleneck_layer_pri: str = ""   # PRI 瓶颈层名称
    trainable_verdict: str = ""      # TRAINABLE / NOT_TRAINABLE
    reachable_verdict: str = ""      # REACHABLE / UNREACHABLE
    # 辅助指标
    C_eff: float = 0.0               # 有效容量 Σ b_k[active]
    P_eff: float = 0.0               # 有效路径数
    rank_product: float = 0.0        # Π (r_l^s / min_dim_l)
    cheeger_product: float = 0.0     # Π h(G_l)
    layers: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# 核心分析函数
# ══════════════════════════════════════════════════════════════════════════════

def _structural_rank(mask_2d: np.ndarray) -> int:
    """计算二值掩码矩阵的秩 (structural rank)。

    等价于二部图 (U, V, E) 的最大匹配数 (König 定理)。
    代表：通过该层最多能传输的独立信息通道数。
    """
    if mask_2d.size == 0:
        return 0
    return int(np.linalg.matrix_rank(mask_2d.astype(np.float64)))


def _bipartite_spectral_gap(adj_2d: np.ndarray) -> tuple[float, float, float]:
    """计算二部图的归一化 Laplacian 谱间隙。

    输入: adj_2d (m, n) — 非负邻接矩阵 (bit × |weight|)
    输出: (spectral_gap, lambda_2_adj, lambda_max_adj)

    方法: 使用归一化 Laplacian L_norm = I - D^{-1/2} A_sym D^{-1/2}
    """
    m, n = adj_2d.shape
    N = m + n

    A_sym = np.zeros((N, N), dtype=np.float64)
    A_sym[:m, m:] = adj_2d
    A_sym[m:, :m] = adj_2d.T

    deg = np.sum(A_sym, axis=1)
    if np.all(deg < 1e-12):
        return 0.0, 0.0, 0.0

    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 1e-12
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    L_norm = np.eye(N, dtype=np.float64) - D_inv_sqrt @ A_sym @ D_inv_sqrt

    eigs_L = np.sort(np.linalg.eigvalsh(L_norm))
    tol = 1e-8
    nonzero_eigs = eigs_L[eigs_L > tol]
    lambda_1 = float(nonzero_eigs[0]) if len(nonzero_eigs) > 0 else 0.0

    eigs_A = np.sort(np.linalg.eigvalsh(A_sym))
    eigs_A_pos = np.sort(eigs_A[eigs_A > tol])[::-1]
    lambda_max_adj = float(eigs_A_pos[0]) if len(eigs_A_pos) > 0 else 0.0
    lambda_2_adj = float(eigs_A_pos[1]) if len(eigs_A_pos) > 1 else 0.0

    return lambda_1, lambda_2_adj, lambda_max_adj


def _compute_sigma_stats(W_active_2d: np.ndarray) -> dict:
    """计算活跃子矩阵的奇异值统计。"""
    null = {
        'sigma_min': 0.0, 'sigma_max': 0.0,
        'condition_number': float('inf'),
        'effective_rank': 0.0, 'numerical_rank': 0,
    }
    if W_active_2d.size == 0 or W_active_2d.shape[0] == 0 or W_active_2d.shape[1] == 0:
        return null

    try:
        s = np.linalg.svd(W_active_2d.astype(np.float64), compute_uv=False)
    except np.linalg.LinAlgError:
        return null

    s = s[s > 1e-12]
    if len(s) == 0:
        return null

    sigma_min = float(s[-1])
    sigma_max = float(s[0])
    cond = sigma_max / max(sigma_min, 1e-12)

    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p + 1e-30))
    eff_rank = math.exp(entropy)

    num_rank = int(np.sum(s > sigma_max * 1e-5))

    return {
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'condition_number': cond,
        'effective_rank': eff_rank,
        'numerical_rank': num_rank,
    }


def _effective_path_count(masks: list[np.ndarray]) -> float:
    """计算从输入层到输出层的有效路径数（布尔矩阵乘法）。"""
    if not masks:
        return 0.0
    path_counts = np.ones(masks[0].shape[0], dtype=np.float64)
    for mask in masks:
        bool_mask = (mask > 0.5).astype(np.float64)
        path_counts = np.clip(path_counts @ bool_mask, 0.0, 1e30)
    return float(np.sum(path_counts))


def _compute_cheeger_estimate(spectral_gap: float) -> float:
    """Cheeger 常数下界 = λ_1 / 2  (归一化 Laplacian 的 Cheeger 不等式)。"""
    return spectral_gap / 2.0


def analyze_layer(layer, kernel: np.ndarray) -> LayerMetrics:
    """分析单层的拓扑、结构秩、谱和量化性质。"""
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
    col_deg = np.sum(active_mask, axis=0)
    row_deg = np.sum(active_mask, axis=1)
    col_degree_mean = float(np.mean(col_deg))
    col_degree_std = float(np.std(col_deg))
    row_degree_mean = float(np.mean(row_deg))
    row_orphan = int(np.sum(row_deg == 0))

    # 有活跃连接的输入/输出数
    n_active_outputs = int(np.sum(col_deg > 0))   # 有 ≥1 输入的输出
    n_active_inputs = int(np.sum(row_deg > 0))     # 有 ≥1 输出的输入

    # ═══ 核心指标: 结构秩 ═══
    struct_rank = _structural_rank(active_mask)

    # 量化统计 (仅活跃连接)
    bits_active = bits[active_mask.astype(bool)]
    if bits_active.size > 0:
        mean_bk = float(np.mean(bits_active))
        std_bk = float(np.std(bits_active))
        min_bk = float(np.min(bits_active))
        max_bk = float(np.max(bits_active))
    else:
        mean_bk = std_bk = min_bk = max_bk = 0.0
    n_dead_q = int(np.sum(bits_active <= 0.1)) if bits_active.size > 0 else 0

    # ═══ 量化信息吞吐量 ═══
    QIT = struct_rank * mean_bk

    # 构造有效邻接矩阵: A = bits ⊙ |W|
    adj = np.clip(bits, 0, None) * np.abs(kernel)
    adj = adj * active_mask

    # 谱间隙
    spectral_gap, lambda_2, lambda_max = _bipartite_spectral_gap(adj)

    # Ramanujan 判定
    d = max(col_degree_mean, 1.0)
    ram_bound = 2.0 * math.sqrt(max(d - 1.0, 0.0))
    is_ram = lambda_2 <= ram_bound + 1e-6

    # Cheeger 常数下界
    cheeger_lo = _compute_cheeger_estimate(spectral_gap)

    # 奇异值分析
    active_rows = np.any(active_mask, axis=1)
    active_cols = np.any(active_mask, axis=0)
    W_sub = kernel[np.ix_(active_rows, active_cols)]
    mask_sub = active_mask[np.ix_(active_rows, active_cols)]
    W_active = W_sub * mask_sub
    svd_stats = _compute_sigma_stats(W_active)

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
        n_active_outputs=n_active_outputs,
        n_active_inputs=n_active_inputs,
        structural_rank=struct_rank,
        QIT=QIT,
        spectral_gap=spectral_gap,
        lambda_2_adj=lambda_2,
        ramanujan_bound=ram_bound,
        is_ramanujan=is_ram,
        cheeger_lower=cheeger_lo,
        sigma_min=svd_stats['sigma_min'],
        sigma_max=svd_stats['sigma_max'],
        condition_number=svd_stats['condition_number'],
        effective_rank=svd_stats['effective_rank'],
        numerical_rank=svd_stats['numerical_rank'],
        mean_bk_active=mean_bk,
        std_bk_active=std_bk,
        min_bk_active=min_bk,
        max_bk_active=max_bk,
        n_dead_quant=n_dead_q,
    )


def analyze_model(
    model,
    sample_input,
    method: str,
    target_ebops: int,
    pareto_acc: float = 0.0,
    source_path: str = "",
    x_val=None,
    y_val=None,
) -> ModelMetrics:
    """分析整个模型，计算 IBR, PRI 等判据。"""
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
        masks.append(_layer_active_mask(layer).astype(np.float32))

    # ═══ 核心判据 ═══

    # IBR = min_l(r_l^s) / τ_train,  τ_train = 2 (梯度分化最低要求)
    TAU_TRAIN = 2
    struct_ranks = [lm.structural_rank for lm in layer_metrics]
    min_struct_rank = min(struct_ranks) if struct_ranks else 0
    ibr = min_struct_rank / TAU_TRAIN
    ibr_bottleneck = layer_metrics[struct_ranks.index(min_struct_rank)].name if layer_metrics else ""

    # PRI = min_l(QIT_l) / τ_reach,  τ_reach = (K-1)/2 (K类判别最低信息需求)
    TAU_REACH = (K_CLASSES - 1) / 2.0
    qit_values = [lm.QIT for lm in layer_metrics]
    min_qit = min(qit_values) if qit_values else 0.0
    pri = min_qit / TAU_REACH if TAU_REACH > 0 else 0.0
    pri_bottleneck = layer_metrics[qit_values.index(min_qit)].name if layer_metrics else ""

    # 判定
    trainable = ibr >= 1.0
    reachable = trainable and pri >= 1.0

    # 辅助指标
    P_eff = _effective_path_count(masks)
    C_eff = 0.0
    for layer in layers:
        kq = layer.kq
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            bits = b_var.numpy()
            C_eff += float(np.sum(np.clip(bits, 0, None)))

    rank_prod = 1.0
    for lm in layer_metrics:
        min_dim = min(lm.shape)
        rank_prod *= max(lm.structural_rank / max(min_dim, 1), 1e-10)

    cheeger_prod = 1.0
    for lm in layer_metrics:
        cheeger_prod *= max(lm.cheeger_lower, 1e-10)

    return ModelMetrics(
        method=method,
        target_ebops=target_ebops,
        measured_ebops=measured,
        val_acc=val_acc,
        val_loss=val_loss,
        pareto_acc=pareto_acc,
        source_path=source_path,
        min_structural_rank=min_struct_rank,
        IBR=ibr,
        bottleneck_layer_ibr=ibr_bottleneck,
        min_QIT=min_qit,
        PRI=pri,
        bottleneck_layer_pri=pri_bottleneck,
        trainable_verdict="TRAINABLE" if trainable else "NOT_TRAINABLE",
        reachable_verdict="REACHABLE" if reachable else "UNREACHABLE",
        C_eff=C_eff,
        P_eff=P_eff,
        rank_product=rank_prod,
        cheeger_product=cheeger_prod,
        layers=layer_metrics,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pareto 前沿查询
# ══════════════════════════════════════════════════════════════════════════════

def _extract_ebops_from_name(name: str) -> int | None:
    m = re.search(r'ebops=(\d+)', name)
    return int(m.group(1)) if m else None


def _extract_val_acc_from_name(name: str) -> float | None:
    m = re.search(r'val_acc=([0-9.]+)', name)
    return float(m.group(1)) if m else None


def pareto_accuracy_at(baseline_dir: Path, target_ebops: int) -> float:
    """查找 Pareto 前沿在 target_ebops 预算下的最大精度。

    即所有 eBOPs ≤ target_ebops 的 baseline 检查点中最高的 val_acc。
    """
    best_acc = 0.0
    for p in baseline_dir.glob("*.keras"):
        e = _extract_ebops_from_name(p.name)
        a = _extract_val_acc_from_name(p.name)
        if e is not None and a is not None and e <= target_ebops:
            if a > best_acc:
                best_acc = a
    return best_acc


def find_best_baseline_ckpt(
    baseline_dir: Path, target_ebops: int, max_ratio: float = 1.5,
) -> tuple[Path, int, float]:
    """从 baseline 目录找最接近 target_ebops 的检查点。"""
    candidates = []
    for p in baseline_dir.glob("*.keras"):
        e = _extract_ebops_from_name(p.name)
        a = _extract_val_acc_from_name(p.name)
        if e is None:
            continue
        candidates.append((p, e, a or 0.0))

    if not candidates:
        raise FileNotFoundError(f"No baseline checkpoints in {baseline_dir}")

    above = [(p, e, a) for p, e, a in candidates if e >= target_ebops and e <= target_ebops * max_ratio]
    below = [(p, e, a) for p, e, a in candidates if e < target_ebops]

    if above:
        above.sort(key=lambda x: (abs(x[1] - target_ebops), -x[2]))
        return above[0]
    elif below:
        below.sort(key=lambda x: (abs(x[1] - target_ebops), -x[2]))
        return below[0]
    else:
        candidates.sort(key=lambda x: x[1])
        return candidates[0]


# ══════════════════════════════════════════════════════════════════════════════
# 快速判据检查 (不做完整分析，仅算结构秩)
# ══════════════════════════════════════════════════════════════════════════════

def quick_check_criteria(model, sample_input) -> tuple[float, float, list[int], list[float]]:
    """快速计算 IBR 和 PRI，不做完整谱分析。

    Returns: (IBR, PRI, structural_ranks_per_layer, QIT_per_layer)
    """
    layers = _dense_prunable_layers(model)
    struct_ranks = []
    qit_values = []

    for layer in layers:
        mask = _layer_active_mask(layer)
        sr = _structural_rank(mask)
        struct_ranks.append(sr)

        kq = layer.kq
        b_var = _get_kq_var(kq, 'b')
        if b_var is None:
            b_var = _get_kq_var(kq, 'f')
        if b_var is not None:
            bits = b_var.numpy().astype(np.float32)
            bits_active = bits[mask.astype(bool)]
            mean_bk = float(np.mean(bits_active)) if bits_active.size > 0 else 0.0
        else:
            mean_bk = 0.0
        qit_values.append(sr * mean_bk)

    min_sr = min(struct_ranks) if struct_ranks else 0
    TAU_TRAIN = 2
    ibr = min_sr / TAU_TRAIN

    min_qit = min(qit_values) if qit_values else 0.0
    TAU_REACH = (K_CLASSES - 1) / 2.0
    pri = min_qit / TAU_REACH if TAU_REACH > 0 else 0.0

    return ibr, pri, struct_ranks, qit_values


# ══════════════════════════════════════════════════════════════════════════════
# 预热膨胀边界搜索
# ══════════════════════════════════════════════════════════════════════════════

def find_criterion_boundary(
    baseline_ckpt: str,
    sample_input,
    low_ebops: float,
    high_ebops: float,
    criterion: str = 'IBR',   # 'IBR' or 'PRI'
    tol: float = 50.0,
    max_iter: int = 15,
) -> tuple[float, list[dict]]:
    """二分搜索：找到 spectral_quant 剪枝后满足给定判据的最小 eBOPs。

    Args:
        baseline_ckpt: 基线模型路径
        sample_input: 校准用输入
        low_ebops: 搜索下界 (已知不满足判据)
        high_ebops: 搜索上界 (已知满足判据)
        criterion: 'IBR' (IBR ≥ 1) 或 'PRI' (PRI ≥ 1)
        tol: 搜索精度 (eBOPs)
        max_iter: 最大迭代次数

    Returns:
        (boundary_ebops, search_history)
    """
    history = []

    for i in range(max_iter):
        if high_ebops - low_ebops <= tol:
            break

        mid = (low_ebops + high_ebops) / 2.0
        logger.info(f"    [搜索 iter={i}] 尝试 eBOPs={mid:.0f} "
                     f"(范围 [{low_ebops:.0f}, {high_ebops:.0f}])...")

        model = keras.models.load_model(baseline_ckpt, compile=False)
        try:
            spectral_quant_prune_to_ebops(
                model, mid, sample_input,
                min_degree=2, b_floor=0.25, b_ceiling=6.0,
                verbose=False,
            )
            bisect_ebops_to_target(
                model, mid, sample_input,
                tolerance=0.05, max_iter=30,
            )
            ibr, pri, ranks, qits = quick_check_criteria(model, sample_input)
            actual_ebops = compute_model_ebops(model, sample_input)
        except Exception as exc:
            logger.warning(f"    [搜索] eBOPs={mid:.0f} 失败: {exc}")
            ibr, pri, ranks, qits = 0.0, 0.0, [], []
            actual_ebops = mid
        finally:
            del model
            keras.backend.clear_session()

        entry = {
            'target': mid, 'actual': actual_ebops,
            'IBR': ibr, 'PRI': pri,
            'ranks': ranks, 'QITs': qits,
        }
        history.append(entry)

        check_val = ibr if criterion == 'IBR' else pri
        if check_val >= 1.0:
            high_ebops = mid
            logger.info(f"    [搜索]   → 满足 (IBR={ibr:.3f}, PRI={pri:.3f}), "
                         f"ranks={ranks}")
        else:
            low_ebops = mid
            logger.info(f"    [搜索]   → 不满足 (IBR={ibr:.3f}, PRI={pri:.3f}), "
                         f"ranks={ranks}")

    return high_ebops, history


# ══════════════════════════════════════════════════════════════════════════════
# 输出函数
# ══════════════════════════════════════════════════════════════════════════════

def _print_metrics(m: ModelMetrics):
    """打印单个模型的详细指标。"""
    logger.info(f"\n  ┌─────────── {m.method} @ target={m.target_ebops} ───────────┐")
    logger.info(f"  │  measured eBOPs:  {m.measured_ebops:.0f}")
    logger.info(f"  │  val_acc:         {m.val_acc:.4f}")
    logger.info(f"  │  val_loss:        {m.val_loss:.4f}")
    logger.info(f"  │  Pareto前沿精度:  {m.pareto_acc:.4f}")
    logger.info(f"  │")
    logger.info(f"  │  ══ 核心判据 ══")
    logger.info(f"  │  ★ IBR = min(r_l^s) / τ_train = {m.min_structural_rank} / 2 "
                f"= {m.IBR:.3f}  → {m.trainable_verdict}")
    logger.info(f"  │    瓶颈层: {m.bottleneck_layer_ibr}")
    tau_reach = (K_CLASSES - 1) / 2.0
    logger.info(f"  │  ★ PRI = min(QIT_l) / τ_reach = {m.min_QIT:.3f} / {tau_reach:.1f} "
                f"= {m.PRI:.3f}  → {m.reachable_verdict}")
    logger.info(f"  │    瓶颈层: {m.bottleneck_layer_pri}")
    logger.info(f"  │")
    logger.info(f"  │  ── 辅助指标 ──")
    logger.info(f"  │  有效容量 C_eff:          {m.C_eff:.1f}")
    logger.info(f"  │  有效路径 P_eff:          {m.P_eff:.3e}")
    logger.info(f"  │  秩比乘积 Π(r_l/d_l):    {m.rank_product:.6f}")
    logger.info(f"  │  Cheeger乘积 Π(h_lower):  {m.cheeger_product:.6e}")
    logger.info(f"  │")

    for lm in m.layers:
        logger.info(f"  │  ── 层: {lm.name} ({lm.shape[0]}→{lm.shape[1]}) ──")
        logger.info(f"  │    活跃: {lm.n_active}/{lm.n_total} "
                     f"(稀疏度={lm.sparsity:.1%})")
        logger.info(f"  │    活跃输出={lm.n_active_outputs}/{lm.shape[1]}  "
                     f"活跃输入={lm.n_active_inputs}/{lm.shape[0]}")
        logger.info(f"  │    ★ 结构秩 r_l^s = {lm.structural_rank}  "
                     f"(需≥2 for trainability)")
        tau_reach = (K_CLASSES - 1) / 2.0
        logger.info(f"  │    ★ QIT_l = r_l^s × mean_bk = "
                     f"{lm.structural_rank} × {lm.mean_bk_active:.3f} = {lm.QIT:.3f}  "
                     f"(需≥{tau_reach:.1f} for reachability)")
        logger.info(f"  │    列度: mean={lm.col_degree_mean:.2f}±{lm.col_degree_std:.2f}")
        logger.info(f"  │    [谱] λ_gap={lm.spectral_gap:.4f}  "
                     f"Ramanujan={'✓' if lm.is_ramanujan else '✗'}  "
                     f"Cheeger≥{lm.cheeger_lower:.4f}")
        logger.info(f"  │    [SVD] σ_min={lm.sigma_min:.4e}  σ_max={lm.sigma_max:.4e}  "
                     f"κ={lm.condition_number:.1f}  "
                     f"eff_rank={lm.effective_rank:.2f}  "
                     f"num_rank={lm.numerical_rank}")
        logger.info(f"  │    [量化] mean_bk={lm.mean_bk_active:.3f}±{lm.std_bk_active:.3f}  "
                     f"range=[{lm.min_bk_active:.2f}, {lm.max_bk_active:.2f}]  "
                     f"dead={lm.n_dead_quant}")

    logger.info(f"  └{'─'*55}┘")


def _print_summary_table(all_metrics: list[ModelMetrics]):
    """打印汇总对比表。"""
    logger.info(f"\n{'='*140}")
    logger.info("                                           汇   总   对   比   表")
    logger.info(f"{'='*140}")
    tau_reach = (K_CLASSES - 1) / 2.0
    logger.info(f"  判据: IBR = min(r_l^s)/2 ≥ 1.0 → 可训练;  "
                f"PRI = min(QIT_l)/{tau_reach:.1f} ≥ 1.0 → 可达前沿;  K={K_CLASSES}")
    logger.info(f"{'─'*140}")

    header = (
        f"  {'target':>7}  {'method':<16}  {'eBOPs':>7}  {'val_acc':>8}  "
        f"{'Pareto':>7}  {'min_r':>5}  {'IBR':>6}  {'min_QIT':>8}  {'PRI':>6}  "
        f"{'C_eff':>7}  {'P_eff':>12}  "
        f"{'层结构秩':^24}  {'train?':<14}  {'reach?':<12}"
    )
    logger.info(header)
    logger.info(f"{'─'*140}")

    targets_seen = sorted(set(m.target_ebops for m in all_metrics))
    for t in targets_seen:
        for m in all_metrics:
            if m.target_ebops != t:
                continue
            # 各层结构秩
            layer_ranks = [f"{lm.structural_rank}" for lm in m.layers]
            ranks_str = "[" + ",".join(layer_ranks) + "]"
            row = (
                f"  {m.target_ebops:>7}  {m.method:<16}  {m.measured_ebops:>7.0f}  "
                f"{m.val_acc:>8.4f}  {m.pareto_acc:>7.4f}  "
                f"{m.min_structural_rank:>5}  {m.IBR:>6.3f}  "
                f"{m.min_QIT:>8.3f}  {m.PRI:>6.3f}  "
                f"{m.C_eff:>7.1f}  {m.P_eff:>12.3e}  "
                f"{ranks_str:<24}  {m.trainable_verdict:<14}  {m.reachable_verdict:<12}"
            )
            logger.info(row)
        logger.info(f"{'─'*140}")

    logger.info("")

    # 逐目标对比
    logger.info("逐目标对比分析:")
    for t in targets_seen:
        group = [m for m in all_metrics if m.target_ebops == t]
        gradual = [m for m in group if m.method == 'gradual']
        spectral = [m for m in group if m.method == 'spectral_quant']
        if gradual and spectral:
            g, s = gradual[0], spectral[0]
            logger.info(f"\n  target={t} (Pareto前沿精度={g.pareto_acc:.4f}):")
            logger.info(f"    精度:  gradual={g.val_acc:.4f}  spectral={s.val_acc:.4f}  "
                         f"Δ={s.val_acc - g.val_acc:+.4f}")
            logger.info(f"    IBR:   gradual={g.IBR:.3f}  spectral={s.IBR:.3f}")
            logger.info(f"    PRI:   gradual={g.PRI:.3f}  spectral={s.PRI:.3f}")
            logger.info(f"    结构秩: gradual={[lm.structural_rank for lm in g.layers]}  "
                         f"spectral={[lm.structural_rank for lm in s.layers]}")
            logger.info(f"    QIT:   gradual={[f'{lm.QIT:.2f}' for lm in g.layers]}  "
                         f"spectral={[f'{lm.QIT:.2f}' for lm in s.layers]}")


def _save_csv(all_metrics: list[ModelMetrics], csv_path: Path):
    """保存结果到 CSV。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    model_fields = [
        'target_ebops', 'method', 'measured_ebops', 'val_acc', 'val_loss',
        'pareto_acc', 'min_structural_rank', 'IBR',
        'bottleneck_layer_ibr', 'min_QIT', 'PRI', 'bottleneck_layer_pri',
        'trainable_verdict', 'reachable_verdict',
        'C_eff', 'P_eff', 'rank_product', 'cheeger_product',
    ]
    layer_fields = [
        'n_active', 'sparsity', 'structural_rank', 'QIT',
        'n_active_outputs', 'n_active_inputs',
        'col_degree_mean', 'spectral_gap', 'lambda_2_adj',
        'ramanujan_bound', 'is_ramanujan', 'cheeger_lower',
        'sigma_min', 'sigma_max', 'condition_number',
        'effective_rank', 'numerical_rank',
        'mean_bk_active', 'std_bk_active', 'n_dead_quant',
    ]

    max_layers = max(len(m.layers) for m in all_metrics) if all_metrics else 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = model_fields[:]
        for i in range(max_layers):
            header.extend([f"L{i}_{field}" for field in layer_fields])
        w.writerow(header)

        for m in all_metrics:
            row = [
                m.target_ebops, m.method, f'{m.measured_ebops:.2f}',
                f'{m.val_acc:.6f}', f'{m.val_loss:.6f}',
                f'{m.pareto_acc:.4f}',
                m.min_structural_rank, f'{m.IBR:.4f}',
                m.bottleneck_layer_ibr, f'{m.min_QIT:.4f}',
                f'{m.PRI:.4f}', m.bottleneck_layer_pri,
                m.trainable_verdict, m.reachable_verdict,
                f'{m.C_eff:.2f}', f'{m.P_eff:.6e}',
                f'{m.rank_product:.8f}', f'{m.cheeger_product:.8e}',
            ]
            for lm in m.layers:
                row.extend([
                    lm.n_active, f'{lm.sparsity:.6f}',
                    lm.structural_rank, f'{lm.QIT:.4f}',
                    lm.n_active_outputs, lm.n_active_inputs,
                    f'{lm.col_degree_mean:.4f}', f'{lm.spectral_gap:.6f}',
                    f'{lm.lambda_2_adj:.6f}', f'{lm.ramanujan_bound:.4f}',
                    int(lm.is_ramanujan), f'{lm.cheeger_lower:.6f}',
                    f'{lm.sigma_min:.6e}', f'{lm.sigma_max:.6e}',
                    f'{lm.condition_number:.4f}',
                    f'{lm.effective_rank:.4f}', lm.numerical_rank,
                    f'{lm.mean_bk_active:.6f}', f'{lm.std_bk_active:.6f}',
                    lm.n_dead_quant,
                ])
            for _ in range(max_layers - len(m.layers)):
                row.extend([''] * len(layer_fields))
            w.writerow(row)

    logger.info(f"  CSV 已保存: {csv_path}")


def _save_warmup_decision(
    warmup_results: dict,
    all_metrics: list[ModelMetrics],
    save_path: Path,
):
    """保存预热膨胀决策到文本文件。"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("═" * 80 + "\n")
        f.write("          预热膨胀决策报告 (Warm-up Budget Inflation Decision)\n")
        f.write("═" * 80 + "\n\n")

        f.write("理论判据:\n")
        tau_reach = (K_CLASSES - 1) / 2.0
        f.write(f"  可训练性  IBR = min_l(structural_rank_l) / 2 ≥ 1.0  (τ_train=2)\n")
        f.write(f"  可达前沿  PRI = min_l(QIT_l) / {tau_reach:.1f} ≥ 1.0  "
                f"(τ_reach=(K-1)/2={tau_reach:.1f})\n\n")

        if 'ibr_boundary' in warmup_results:
            f.write(f"搜索结果:\n")
            f.write(f"  IBR 边界 (最小可训练 eBOPs): {warmup_results['ibr_boundary']:.0f}\n")
            if 'pri_boundary' in warmup_results:
                f.write(f"  PRI 边界 (最小可达前沿 eBOPs): {warmup_results['pri_boundary']:.0f}\n")
            warmup_budget = warmup_results.get('recommended_warmup', 0)
            f.write(f"  推荐预热预算: {warmup_budget:.0f} eBOPs\n\n")

        f.write("─" * 80 + "\n")
        f.write("各目标决策:\n\n")
        f.write(f"  {'target':>8}  {'method':<16}  {'IBR':>6}  {'PRI':>6}  "
                f"{'train?':<14}  {'reach?':<12}  {'决策':<30}\n")
        f.write(f"  {'─'*8}  {'─'*16}  {'─'*6}  {'─'*6}  {'─'*14}  {'─'*12}  {'─'*30}\n")

        for m in all_metrics:
            if m.method != 'spectral_quant':
                continue
            if m.trainable_verdict == "NOT_TRAINABLE":
                warmup = warmup_results.get('recommended_warmup', '?')
                decision = f"膨胀到 {warmup:.0f} 预热" if isinstance(warmup, (int, float)) else "需要膨胀"
            elif m.reachable_verdict == "UNREACHABLE":
                warmup = warmup_results.get('pri_boundary', '?')
                decision = f"膨胀到 {warmup:.0f} 预热" if isinstance(warmup, (int, float)) else "可能需膨胀"
            else:
                decision = "直接剪枝训练"

            f.write(f"  {m.target_ebops:>8}  {m.method:<16}  {m.IBR:>6.3f}  "
                    f"{m.PRI:>6.3f}  {m.trainable_verdict:<14}  "
                    f"{m.reachable_verdict:<12}  {decision:<30}\n")

        f.write(f"\n{'═' * 80}\n")
        f.write("决策规则:\n")
        f.write("  1. IBR < 1.0 → 模型不可训练，必须膨胀 eBOPs 到 IBR ≥ 1.0\n")
        f.write("  2. IBR ≥ 1.0 但 PRI < 1.0 → 模型可训练但可能无法达到 Pareto 前沿\n")
        f.write("     → 膨胀 eBOPs 到 PRI ≥ 1.0\n")
        f.write("  3. IBR ≥ 1.0 且 PRI ≥ 1.0 → 直接进行 spectral_quant 剪枝 + beta scheduling\n")
        f.write(f"\n{'═' * 80}\n")

    logger.info(f"  预热决策已保存: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 80)
    logger.info("对比 spectral_quant 剪枝 vs 自然训练（gradual）模型拓扑  [V2]")
    logger.info("=" * 80)

    # ── 理论说明 ──────────────────────────────────────────────────────────
    tau_reach_val = (K_CLASSES - 1) / 2.0
    logger.info(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   理论框架 V2：信息瓶颈与量化吞吐判据                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ▎判据 1: 可训练性 — 信息瓶颈秩比 (IBR)                                    ║
║                                                                              ║
║  理论: 梯度 Jacobian 秩条件                                                 ║
║    反向传播中层 l 的 Jacobian 秩 ≤ r_l^s = rank(M_l)                        ║
║    当 r_l^s = 1 → 所有参数获得相同梯度方向 → SGD 无法分化                   ║
║    当 r_l^s ≥ 2 → 参数获得差异化梯度信号 → SGD 可有效优化                   ║
║                                                                              ║
║  结构秩: r_l^s = rank(M_l)  — 二值掩码矩阵的秩                             ║
║         = 二部图最大匹配 (König 定理)                                        ║
║         = 层 l 最多能传输的独立信息通道数                                    ║
║                                                                              ║
║  ★ IBR = min_l(r_l^s) / τ_train,  τ_train = 2                               ║
║    IBR ≥ 1.0 ⟹ 每层至少 2 个独立梯度方向 → 可训练                          ║
║    IBR < 1.0 ⟹ 存在标量瓶颈层 → 不可训练                                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ▎判据 2: 可达 Pareto 前沿 — 量化信息吞吐比 (PRI)                          ║
║                                                                              ║
║  量化信息吞吐: QIT_l = r_l^s × mean(b_l[active])                            ║
║    含义: (独立通道数) × (每通道平均精度) = 总有效比特                        ║
║                                                                              ║
║  K 类判别需 K-1 次成对比较, 每次 ≥ 0.5 有效比特:                            ║
║    τ_reach = (K-1)/2 = {tau_reach_val:.1f}                                   ║
║                                                                              ║
║  ★ PRI = min_l(QIT_l) / τ_reach                                             ║
║    PRI ≥ 1.0 ⟹ 信息吞吐充足 → 可达 Pareto 前沿                            ║
║    PRI < 1.0 ⟹ 信息吞吐不足 → 可能无法达到前沿                             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ▎预热决策: 当 IBR < 1 或 PRI < 1 时膨胀 eBOPs                              ║
║                                                                              ║
║  τ_train = 2 (梯度质量条件)                                                 ║
║  τ_reach = (K-1)/2 = {tau_reach_val:.1f} (量化信息条件)                      ║
║  二分搜索:                                                                    ║
║    eBOPs_train = min{{E : IBR(E) ≥ 1}}  — 最小可训练 eBOPs                   ║
║    eBOPs_reach = min{{E : PRI(E) ≥ 1}}  — 最小可达前沿 eBOPs                 ║
║    推荐预热 = max(eBOPs_train, eBOPs_reach)                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # ── 加载数据 ──────────────────────────────────────────────────────────
    logger.info("加载数据...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data(INPUT_H5, src='openml')
    sample_input = tf.constant(X_val[:512], dtype=tf.float32)
    x_eval = tf.constant(X_val[:4096], dtype=tf.float32)
    y_eval = tf.constant(y_val[:4096], dtype=tf.int32)

    # ── 拓扑绘图器 ───────────────────────────────────────────────────────
    plotter = TopologyGraphPlotter(
        symmetric_topology_plot=False,
        mirror_edges=False,
        plot_matrix=True,
        strict_original_connections=True,
    )

    # ── Baseline 全模型参考 ──────────────────────────────────────────────
    logger.info(f"\n加载 baseline 完整模型: {BASELINE_CKPT}")
    baseline_model = keras.models.load_model(BASELINE_CKPT, compile=False)
    baseline_metrics = analyze_model(
        baseline_model, sample_input, "baseline_full",
        target_ebops=24000,
        pareto_acc=0.770,
        source_path=BASELINE_CKPT,
        x_val=x_eval, y_val=y_eval,
    )
    logger.info(f"  Baseline: ebops={baseline_metrics.measured_ebops:.0f}, "
                f"acc={baseline_metrics.val_acc:.4f}, "
                f"IBR={baseline_metrics.IBR:.3f}, PRI={baseline_metrics.PRI:.3f}")
    logger.info(f"  层结构秩: {[lm.structural_rank for lm in baseline_metrics.layers]}")
    logger.info(f"  层 QIT:   {[f'{lm.QIT:.2f}' for lm in baseline_metrics.layers]}")
    del baseline_model

    # ── 逐目标分析 ───────────────────────────────────────────────────────
    all_metrics: list[ModelMetrics] = []

    for target_ebops in TARGET_EBOPS_LIST:
        logger.info(f"\n{'='*80}")
        logger.info(f"  TARGET eBOPs = {target_ebops}")
        logger.info(f"{'='*80}")

        # Pareto 前沿精度
        pareto_acc = pareto_accuracy_at(BASELINE_DIR, target_ebops)
        logger.info(f"  Pareto 前沿精度 @ ≤{target_ebops} eBOPs: {pareto_acc:.4f}")

        for method in ['gradual', 'spectral_quant']:
            logger.info(f"\n  ── 方法: {method} ──")

            if method == 'gradual':
                try:
                    ckpt_path, ckpt_ebops, ckpt_acc = find_best_baseline_ckpt(
                        BASELINE_DIR, target_ebops, max_ratio=2.5
                    )
                except FileNotFoundError as exc:
                    logger.warning(f"  跳过 gradual@{target_ebops}: {exc}")
                    continue

                logger.info(f"  加载检查点: {ckpt_path.name}")
                logger.info(f"    (ebops={ckpt_ebops}, val_acc={ckpt_acc:.3f})")
                model = keras.models.load_model(str(ckpt_path), compile=False)
                source = str(ckpt_path)
            else:
                logger.info(f"  从基线剪枝到 target={target_ebops}...")
                model = keras.models.load_model(BASELINE_CKPT, compile=False)
                measured, _ = spectral_quant_prune_to_ebops(
                    model, float(target_ebops), sample_input,
                    min_degree=2, b_floor=0.25, b_ceiling=6.0,
                    verbose=True,
                )
                calibrated = bisect_ebops_to_target(
                    model, float(target_ebops), sample_input,
                    tolerance=0.05, max_iter=30,
                )
                source = f"pruned_from:{BASELINE_CKPT}"
                logger.info(f"  校准后 eBOPs = {calibrated:.0f}")

            # 分析
            metrics = analyze_model(
                model, sample_input, method, target_ebops,
                pareto_acc=pareto_acc,
                source_path=source,
                x_val=x_eval, y_val=y_eval,
            )
            all_metrics.append(metrics)

            # 拓扑图
            layer_data = plotter.extract_layer_graph_data(model)
            subtitle = (
                f"eBOPs={metrics.measured_ebops:.0f}  "
                f"acc={metrics.val_acc:.4f}  "
                f"IBR={metrics.IBR:.2f}  PRI={metrics.PRI:.2f}  "
                f"ranks={[lm.structural_rank for lm in metrics.layers]}"
            )
            circle_path = OUTPUT_DIR / f"topology_target{target_ebops}_{method}_circle.png"
            plotter.plot_circle_graph(
                layer_data, circle_path,
                title=f"{method} @ target={target_ebops}",
                subtitle=subtitle,
            )
            matrix_path = OUTPUT_DIR / f"topology_target{target_ebops}_{method}_matrix.png"
            plotter.plot_weighted_topology_matrices(
                layer_data, matrix_path,
                title=f"{method} @ target={target_ebops}",
                subtitle=subtitle,
            )
            logger.info(f"  拓扑图: {circle_path.name}, {matrix_path.name}")

            # 详细指标
            _print_metrics(metrics)

            del model
            keras.backend.clear_session()

    # ── 汇总表 ───────────────────────────────────────────────────────────
    _print_summary_table(all_metrics)

    # ── 保存 CSV ─────────────────────────────────────────────────────────
    _save_csv(all_metrics, OUTPUT_DIR / "metrics_table.csv")

    # ══════════════════════════════════════════════════════════════════════
    # 预热膨胀边界搜索
    # ══════════════════════════════════════════════════════════════════════
    warmup_results = {}

    if RUN_WARMUP_SEARCH:
        logger.info(f"\n{'='*80}")
        logger.info("  预热膨胀边界搜索 (Warm-up Budget Boundary Search)")
        logger.info(f"{'='*80}")

        # 找到需要搜索的范围
        spectral_metrics = [m for m in all_metrics if m.method == 'spectral_quant']
        spectral_metrics.sort(key=lambda m: m.target_ebops)

        # 找到最小的可训练 spectral target
        trainable_targets = [m.target_ebops for m in spectral_metrics
                             if m.trainable_verdict == "TRAINABLE"]
        non_trainable_targets = [m.target_ebops for m in spectral_metrics
                                 if m.trainable_verdict == "NOT_TRAINABLE"]

        if non_trainable_targets and trainable_targets:
            low_bound = max(non_trainable_targets)
            high_bound = min(trainable_targets)

            logger.info(f"\n  IBR 边界搜索:")
            logger.info(f"    已知不可训练: {non_trainable_targets}")
            logger.info(f"    已知可训练: {trainable_targets}")
            logger.info(f"    搜索范围: [{low_bound}, {high_bound}]")

            ibr_boundary, ibr_history = find_criterion_boundary(
                BASELINE_CKPT, sample_input,
                low_ebops=float(low_bound),
                high_ebops=float(high_bound),
                criterion='IBR',
                tol=WARMUP_SEARCH_TOL,
            )
            warmup_results['ibr_boundary'] = ibr_boundary
            warmup_results['ibr_history'] = ibr_history
            logger.info(f"\n  ★ IBR 边界 (最小可训练 eBOPs) ≈ {ibr_boundary:.0f}")

            # PRI 边界搜索
            reachable_targets = [m.target_ebops for m in spectral_metrics
                                 if m.reachable_verdict == "REACHABLE"]
            unreachable_targets = [m.target_ebops for m in spectral_metrics
                                   if m.reachable_verdict == "UNREACHABLE"]

            if unreachable_targets and reachable_targets:
                low_pri = max(unreachable_targets)
                high_pri = min(reachable_targets)

                if low_pri < high_pri:
                    logger.info(f"\n  PRI 边界搜索:")
                    logger.info(f"    搜索范围: [{low_pri}, {high_pri}]")

                    pri_boundary, pri_history = find_criterion_boundary(
                        BASELINE_CKPT, sample_input,
                        low_ebops=float(low_pri),
                        high_ebops=float(high_pri),
                        criterion='PRI',
                        tol=WARMUP_SEARCH_TOL,
                    )
                    warmup_results['pri_boundary'] = pri_boundary
                    warmup_results['pri_history'] = pri_history
                    logger.info(f"\n  ★ PRI 边界 (最小可达前沿 eBOPs) ≈ {pri_boundary:.0f}")
                else:
                    warmup_results['pri_boundary'] = ibr_boundary
                    logger.info(f"\n  PRI 边界与 IBR 边界重合: {ibr_boundary:.0f}")
            else:
                warmup_results['pri_boundary'] = ibr_boundary

            recommended = max(
                warmup_results.get('ibr_boundary', 0),
                warmup_results.get('pri_boundary', 0),
            )
            # 加 10% safety margin
            recommended = recommended * 1.1
            warmup_results['recommended_warmup'] = recommended

            logger.info(f"\n  ★★★ 推荐预热预算 (含10%余量): {recommended:.0f} eBOPs ★★★")
            logger.info(f"  决策: 当 target_ebops < {recommended:.0f} 时，"
                         f"膨胀到 {recommended:.0f} eBOPs 进行预热训练")

        elif non_trainable_targets:
            logger.info("  所有测试目标都不可训练，需要更大的搜索上界")
        else:
            logger.info("  所有测试目标都可训练，无需预热膨胀")
            warmup_results['recommended_warmup'] = 0

    # ── 保存决策报告 ─────────────────────────────────────────────────────
    _save_warmup_decision(warmup_results, all_metrics, OUTPUT_DIR / "warmup_decision.txt")

    # ── 最终总结 ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*80}")
    logger.info("                           最 终 总 结")
    logger.info(f"{'='*80}")
    logger.info(f"""
  可训练性判据 (IBR):
    IBR = min_l(structural_rank_l) / τ_train,  τ_train = 2
    IBR ≥ 1.0 → 可训练;  IBR < 1.0 → 不可训练

  Pareto可达判据 (PRI):
    PRI = min_l(structural_rank_l × mean_bk_l) / τ_reach,  τ_reach = (K-1)/2 = {tau_reach_val:.1f}
    PRI ≥ 1.0 → 可达前沿;  PRI < 1.0 → 可能无法达到

  关键发现:
""")
    # 列出所有 spectral 结果
    for m in all_metrics:
        if m.method == 'spectral_quant':
            status = f"{m.trainable_verdict}, {m.reachable_verdict}"
            logger.info(f"    target={m.target_ebops:>6}: IBR={m.IBR:.3f}, PRI={m.PRI:.3f}  "
                         f"→ {status}")

    if 'recommended_warmup' in warmup_results:
        w = warmup_results['recommended_warmup']
        if w > 0:
            logger.info(f"\n  ★ 预热决策: 当 target < {w:.0f} eBOPs 时，")
            logger.info(f"    先膨胀到 {w:.0f} eBOPs 剪枝 → warm-up 训练 → "
                         f"beta scheduling 缩减到目标 eBOPs")
        else:
            logger.info(f"\n  ★ 所有目标均满足判据，无需预热膨胀")

    logger.info(f"\n所有结果已保存到: {OUTPUT_DIR}")
    logger.info("完成。")


if __name__ == '__main__':
    main()
