# ramanujan_utils.py
import numpy as np
import tensorflow as tf
import keras


def _ramanujan_like_mask_for_kernel(shape, degree, rng):
    """
    给定 kernel 的 shape 生成一个 'd-regular' 式的稀疏掩膜。
    注意这里是工程上的 Ramanujan-like 稀疏拓扑：
    - Dense: 每个输出结点连接 degree 个输入
    - Conv2D: 在通道维度上做 degree 连接，并在空间维度上广播
    """
    if len(shape) == 2:
        # Dense: (in_dim, out_dim)
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
        # 在空间维度上广播
        mask = np.broadcast_to(base[None, None, :, :], (kh, kw, in_ch, out_ch))
        return mask.astype(np.float32)

    else:
        # 其他情况先不处理
        raise ValueError(f"Unsupported kernel shape for Ramanujan init: {shape}")


def apply_ramanujan_init(
    model: keras.Model,
    default_degree: int = 8,
    per_layer_degree: dict | None = None,
    seed: int = 42,
    include_dense: bool = True,
    include_conv: bool = True,
):
    """
    对模型中具有 kernel 的层做拉马努金式稀疏初始化（只改变初始化，不固定拓扑）。

    参数:
        model: Keras 模型
        default_degree: 默认每个输出结点的度
        per_layer_degree: 可选字典 {layer.name: degree}，覆盖默认度
        seed: 随机种子
        include_dense: 是否作用于 Dense / QuantDense 之类的层
        include_conv: 是否作用于 Conv2D / 量化 Conv 层
    """
    rng = np.random.RandomState(seed)

    for layer in model.layers:
        # 只处理有 kernel 的层
        if not hasattr(layer, "kernel"):
            continue

        # 类型筛选（大部分量化层也会继承 Dense/Conv2D）
        if isinstance(layer, keras.layers.Dense):
            if not include_dense:
                continue
        elif isinstance(layer, keras.layers.Conv2D):
            if not include_conv:
                continue
        else:
            # 对一些自定义 HGQ 层：如果它们也有 kernel，可以根据需要放开
            # 这里先保守一点，只处理 Dense/Conv2D
            continue

        kernel_var: tf.Variable = layer.kernel
        kernel = kernel_var.numpy()
        shape = kernel.shape

        # 选择该层的度
        degree = default_degree
        if per_layer_degree is not None and layer.name in per_layer_degree:
            degree = per_layer_degree[layer.name]

        # 生成掩膜并应用
        mask = _ramanujan_like_mask_for_kernel(shape, degree, rng)
        kernel_new = kernel * mask

        # 回写到变量
        kernel_var.assign(kernel_new)

        # 把 mask 挂在 layer 上，为后续“固定拓扑/训练中重置”做准备
        layer.ramanujan_mask = tf.constant(mask, dtype=kernel_var.dtype)

        print(
            f"[RamanujanInit] layer={layer.name}, shape={shape}, "
            f"degree={degree}, sparsity={1.0 - mask.mean():.4f}"
        )


class RamanujanMaskEnforcer(keras.callbacks.Callback):
    """
    （后面消融时用）在训练过程中保持 kernel 与 Ramanujan 掩膜相乘。
    现在只做初始化实验，可以先不使用这个 callback。
    """

    def __init__(self, layer_names=None):
        super().__init__()
        self.layer_names = set(layer_names) if layer_names is not None else None

    def on_train_batch_end(self, batch, logs=None):
        for layer in self.model.layers:
            mask = getattr(layer, "ramanujan_mask", None)
            if mask is None:
                continue
            if self.layer_names is not None and layer.name not in self.layer_names:
                continue
            kernel = layer.kernel
            kernel.assign(kernel * mask)