"""
FPL2/jsc/utils — 谱约束剪枝 + RigL 复活 + Beta 调度 工具包
==========================================================

共享的 HGQ 底层工具函数放在此文件中，其余按功能拆分到子模块：
  - tf_device.py   : GPU 检测
  - pruning.py     : 谱约束一次性剪枝 + Mask 强制 + spectral_quant + bisect + snap
  - budget.py      : Beta 预算控制器 + KQB 稳定器 + 激活位宽固定
  - revival.py     : RigL 风格谱梯度连接复活
  - training.py    : LR 调度、训练轨迹记录、智能早停
  - plotting.py    : 拓扑图绘制 + 周期绘图回调
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf
import keras


# ═══════════════════════════════════════════════════════════════════════════════
# 共享 HGQ 底层工具
# ═══════════════════════════════════════════════════════════════════════════════

def _get_kq_var(kq, name: str):
    """从 HGQ 量化器中按逻辑名 (b/i/f) 稳健地获取变量。"""
    # 1) 直接属性
    if hasattr(kq, name):
        v = getattr(kq, name)
        if isinstance(v, tf.Variable):
            return v

    # 2) 后缀匹配
    cand = []
    for v in getattr(kq, "variables", []):
        leaf = v.name.split("/")[-1].split(":")[0]
        if leaf == name:
            cand.append(v)
    if len(cand) == 1:
        return cand[0]
    if len(cand) > 1:
        cand.sort(key=lambda x: len(x.name))
        return cand[0]

    # 3) 包含匹配
    for v in getattr(kq, "variables", []):
        if ("/" + name + ":") in v.name:
            return v
    return None


def _flatten_layers(model: keras.Model) -> Iterable[keras.layers.Layer]:
    """遍历模型所有层（包括嵌套），兼容 Keras 2/3。"""
    if hasattr(model, "_flatten_layers"):
        try:
            return list(model._flatten_layers(include_self=False, recursive=True))
        except TypeError:
            return list(model._flatten_layers())
    out = []
    stack = list(getattr(model, "layers", []))
    while stack:
        layer = stack.pop(0)
        out.append(layer)
        stack.extend(getattr(layer, "layers", []))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷 re-export
# ═══════════════════════════════════════════════════════════════════════════════

from .tf_device import get_tf_device
from .pruning import (
    compute_bw_aware_degree,
    apply_ramanujan_bw_init,
    RamanujanMaskEnforcer,
    spectral_quant_prune_to_ebops,
    bisect_ebops_to_target,
    snap_active_bk,
    compute_model_ebops,
)
from .budget import (
    BetaOnlyBudgetController,
    KQBStabilizer,
    ActivationBitsFixer,
    SoftDeathFloor,
    ProgressiveBudgetController,
    BetaCurriculumController,
    AdaptiveLRBiwidthScaler,
    _set_all_beta,
    _get_active_bk_mean,
)
from .revival import SpectralGradientRevivalCallback
from .training import (
    cosine_decay_restarts_schedule,
    TrainingTraceToH5,
    BudgetAwareEarlyStopping,
)
from .plotting import (
    TopologyGraphPlotter,
    TopologyPlotCallback,
    plot_topology,
)
