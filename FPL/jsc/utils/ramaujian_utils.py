# ramanujan_utils.py
import numpy as np
import tensorflow as tf
import keras


def _ramanujan_like_mask_for_kernel(shape, degree, rng):
    """
    给定 kernel 的 shape 生成一个 'd-regular' 式的稀疏掩膜。
    - Dense: 每个输出结点随机选 degree 个输入连接，其余为 0
    - Conv2D: 在输入通道维度上做 degree 连接，并在空间维度上广播
    """
    if len(shape) == 2:
        # Dense / QDense / QEinsumDenseBatchnorm 的 kernel: (in_dim, out_dim)
        in_dim, out_dim = shape
        degree = min(degree, in_dim)
        mask = np.zeros((in_dim, out_dim), dtype=np.float32)
        all_inputs = np.arange(in_dim)
        for o in range(out_dim):
            idx = rng.choice(all_inputs, size=degree, replace=False)
            mask[idx, o] = 1.0
        return mask

    elif len(shape) == 4:
        # Conv2D: (kh, kw, in_ch, out_ch)
        kh, kw, in_ch, out_ch = shape
        degree = min(degree, in_ch)
        base = np.zeros((in_ch, out_ch), dtype=np.float32)
        all_inputs = np.arange(in_ch)
        for o in range(out_ch):
            idx = rng.choice(all_inputs, size=degree, replace=False)
            base[idx, o] = 1.0
        mask = np.broadcast_to(base[None, None, :, :], (kh, kw, in_ch, out_ch))
        return mask.astype(np.float32)

    else:
        raise ValueError(f"Unsupported kernel shape for Ramanujan init: {shape}")


def _get_kq_var(kq, name):
    """从 HGQ Quantizer 的 variables 列表里按 short name 取变量。"""
    for v in kq.variables:
        if v.name == name:
            return v
    return None


def compute_per_layer_degree(
    model: keras.Model,
    multiplier: float = 1.5,
    min_degree: int = 4,
) -> dict:
    """根据每层 kernel 的输入维度自动计算 Ramanujan 度。

    公式：degree = clamp(round(sqrt(in_dim) * multiplier), min_degree, in_dim)

    参数:
        model      : Keras / HGQ 模型
        multiplier : sqrt(in_dim) 的放大系数，保证谱间隙余量（默认 1.5）
        min_degree : 度的下限（默认 4）

    返回:
        {layer.name: degree}，仅包含同时具备 kernel 和 kq 的 HGQ 层
    """
    import math
    result = {}
    for layer in model.layers:
        if getattr(layer, 'kq', None) is None or not hasattr(layer, 'kernel'):
            continue
        shape = layer.kernel.shape
        # Dense: (in_dim, out_dim);  Conv2D: (kh, kw, in_ch, out_ch)
        in_dim = shape[0] if len(shape) == 2 else shape[2]
        degree = int(round(math.sqrt(in_dim) * multiplier))
        degree = max(degree, min_degree)
        degree = min(degree, int(in_dim))
        result[layer.name] = degree
        print(
            f'[compute_per_layer_degree] layer={layer.name}, '
            f'in_dim={in_dim}, degree={degree}, '
            f'sparsity={1.0 - degree / in_dim:.1%}'
        )
    return result


def apply_ramanujan_init(
    model: keras.Model,
    default_degree: int = 8,
    per_layer_degree: dict | None = None,
    seed: int = 42,
    pruned_frac_bits: float = 0.0,
    pruned_int_bits: float = 0.0,
    also_zero_kernel: bool = True,
):
    """
    对 HGQ 模型的 kernel 量化器做 Ramanujan-like 稀疏初始化。

    核心原理：
        旧方法只把 kernel * mask（浮点权重值清零），训练时 optimizer 会把它们更新回来，
        稀疏结构完全无法保持。

        新方法把稀疏掩膜应用到 kq.b（小数位宽）和 kq.i（整数位宽）上：
        被剪掉的连接位宽设为 0 → HGQ 量化器输出恒为 0 → 训练不可逆转地抑制这些连接。
        这才是在 HGQ 框架下"真正生效"的稀疏初始化。

    参数:
        model            : Keras / HGQ 模型
        default_degree   : 每个输出结点保留的输入连接数（Ramanujan 度）
        per_layer_degree : {layer.name: degree}，覆盖默认度
        seed             : 随机种子
        pruned_frac_bits : 被剪掉连接的小数位宽（默认 0 → 量化输出为 0）
        pruned_int_bits  : 被剪掉连接的整数位宽（默认 0）
        also_zero_kernel : 同时把浮点 kernel 对应位置清零，保持初始一致性
    """
    rng = np.random.RandomState(seed)

    for layer in model.layers:
        # HGQ 层须同时具备 kernel 和 kq（kernel quantizer）
        kq = getattr(layer, "kq", None)
        if kq is None or not hasattr(layer, "kernel"):
            continue

        kernel_var: tf.Variable = layer.kernel
        shape = kernel_var.shape

        # 选择该层的度
        degree = default_degree
        if per_layer_degree is not None and layer.name in per_layer_degree:
            degree = per_layer_degree[layer.name]

        # 生成 Ramanujan-like 稀疏掩膜（1=保留, 0=剪掉）
        mask = _ramanujan_like_mask_for_kernel(shape, degree, rng)
        pruned = 1.0 - mask  # 被剪掉的位置

        # ---- 核心：将掩膜应用到 kq.b（小数位宽）-----------------------
        b_var = _get_kq_var(kq, "b")
        if b_var is not None:
            b_new = b_var.numpy() * mask + pruned_frac_bits * pruned
            b_var.assign(b_new)

        # ---- 同步应用到 kq.i（整数位宽）--------------------------------
        i_var = _get_kq_var(kq, "i")
        if i_var is not None:
            i_new = i_var.numpy() * mask + pruned_int_bits * pruned
            i_var.assign(i_new)

        # ---- 可选：同步清零浮点 kernel，保持初始一致性 ------------------
        if also_zero_kernel:
            kernel_var.assign(kernel_var.numpy() * mask)

        # 把 mask 挂在 layer 上，供 RamanujanMaskEnforcer 和调试使用
        layer.ramanujan_mask = tf.constant(mask, dtype=tf.float32)

        active = int(mask.sum())
        total = int(mask.size)
        print(
            f"[RamanujanInit] layer={layer.name}, shape={shape}, "
            f"degree={degree}, active={active}/{total}, "
            f"sparsity={1.0 - mask.mean():.4f}"
        )


class RamanujanMaskEnforcer(keras.callbacks.Callback):
    """
    在训练过程中将 kq.b / kq.i 中被剪掉连接的位宽钳制为指定值（默认 0），
    防止 optimizer 将其从 0 更新回来，从而保持 Ramanujan 稀疏拓扑。

    支持两种模式：
    - 强制模式（release_epoch=None）：始终强制，拓扑完全固定（原始行为）。
    - 渐进放开模式（release_epoch 指定）：
        * [0, release_epoch)：每 batch 强制，拓扑完全固定（warmup 阶段）。
        * [release_epoch, release_epoch + fade_epochs)：按线性衰减概率施加约束，
          允许梯度逐渐"修复"初始随机拓扑中次优的连接。
        * [release_epoch + fade_epochs, ∞)：完全放开，HGQ 自由学习稀疏结构。

    推荐设置：
        release_epoch = warmup_end_epoch（= beta_sch_1）
        fade_epochs   = total_epochs // 5  （渐进窗口）

    用法（在 train_run_ram.py 里加入 callbacks）：
        enforcer = RamanujanMaskEnforcer(release_epoch=1200, fade_epochs=2400)
        callbacks = [..., enforcer]
    """

    def __init__(
        self,
        layer_names=None,
        enforce_frac_bits: float = 0.0,
        enforce_int_bits: float = 0.0,
        release_epoch: int | None = None,
        fade_epochs: int = 0,
    ):
        super().__init__()
        self.layer_names = set(layer_names) if layer_names is not None else None
        self.enforce_frac_bits = enforce_frac_bits
        self.enforce_int_bits = enforce_int_bits
        self.release_epoch = release_epoch
        self.fade_epochs = max(fade_epochs, 1)
        self._current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def _enforce_strength(self) -> float:
        """返回 [0, 1] 的约束强度：1=完全固定，0=完全放开。"""
        if self.release_epoch is None:
            return 1.0
        epoch = self._current_epoch
        if epoch < self.release_epoch:
            return 1.0
        progress = (epoch - self.release_epoch) / self.fade_epochs
        return max(0.0, 1.0 - progress)

    def on_train_batch_end(self, batch, logs=None):
        strength = self._enforce_strength()
        if strength <= 0.0:
            return  # 完全放开，不干预

        for layer in self.model.layers:
            mask = getattr(layer, "ramanujan_mask", None)
            if mask is None:
                continue
            if self.layer_names is not None and layer.name not in self.layer_names:
                continue

            kq = getattr(layer, "kq", None)
            if kq is None:
                continue

            pruned = 1.0 - mask.numpy()

            b_var = _get_kq_var(kq, "b")
            if b_var is not None:
                b_arr = b_var.numpy()
                if strength >= 1.0:
                    b_arr = np.where(pruned > 0, self.enforce_frac_bits, b_arr)
                else:
                    # 线性插值：渐进放开
                    target = np.where(pruned > 0, self.enforce_frac_bits, b_arr)
                    b_arr = strength * target + (1.0 - strength) * b_arr
                b_var.assign(b_arr)

            i_var = _get_kq_var(kq, "i")
            if i_var is not None:
                i_arr = i_var.numpy()
                if strength >= 1.0:
                    i_arr = np.where(pruned > 0, self.enforce_int_bits, i_arr)
                else:
                    target = np.where(pruned > 0, self.enforce_int_bits, i_arr)
                    i_arr = strength * target + (1.0 - strength) * i_arr
                i_var.assign(i_arr)
