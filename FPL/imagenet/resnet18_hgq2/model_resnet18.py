import keras
from hgq.config import QuantizerConfigScope
from hgq.layers import QAdd, QBatchNormalization, QConv2D, QDense


def _basic_block_fp(x, filters: int, stride: int, name: str):
    shortcut = x

    x = keras.layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding='same',
        use_bias=False,
        name=f'{name}_conv1',
    )(x)
    x = keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = keras.layers.ReLU(name=f'{name}_relu1')(x)

    x = keras.layers.Conv2D(
        filters,
        3,
        strides=1,
        padding='same',
        use_bias=False,
        name=f'{name}_conv2',
    )(x)
    x = keras.layers.BatchNormalization(name=f'{name}_bn2')(x)

    in_channels = shortcut.shape[-1]
    in_channels = int(in_channels) if in_channels is not None else None
    if stride != 1 or in_channels != filters:
        shortcut = keras.layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding='same',
            use_bias=False,
            name=f'{name}_proj_conv',
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(name=f'{name}_proj_bn')(shortcut)

    x = keras.layers.Add(name=f'{name}_add')([x, shortcut])
    x = keras.layers.ReLU(name=f'{name}_out')(x)
    return x


def _basic_block_hgq(x, filters: int, stride: int, name: str):
    shortcut = x

    x = QConv2D(
        filters,
        3,
        strides=stride,
        padding='same',
        use_bias=False,
        name=f'{name}_conv1',
    )(x)
    x = QBatchNormalization(name=f'{name}_bn1')(x)
    x = keras.layers.ReLU(name=f'{name}_relu1')(x)

    x = QConv2D(
        filters,
        3,
        strides=1,
        padding='same',
        use_bias=False,
        name=f'{name}_conv2',
    )(x)
    x = QBatchNormalization(name=f'{name}_bn2')(x)

    in_channels = shortcut.shape[-1]
    in_channels = int(in_channels) if in_channels is not None else None
    if stride != 1 or in_channels != filters:
        shortcut = QConv2D(
            filters,
            1,
            strides=stride,
            padding='same',
            use_bias=False,
            name=f'{name}_proj_conv',
        )(shortcut)
        shortcut = QBatchNormalization(name=f'{name}_proj_bn')(shortcut)

    x = QAdd(name=f'{name}_add')([x, shortcut])
    x = keras.layers.ReLU(name=f'{name}_out')(x)
    return x


def _resnet18_body(inputs, block_fn, quantized: bool = False):
    if quantized:
        x = QConv2D(
            64,
            7,
            strides=2,
            padding='same',
            use_bias=False,
            name='qstem_conv',
        )(inputs)
        x = QBatchNormalization(name='qstem_bn')(x)
        x = keras.layers.ReLU(name='qstem_relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='qstem_pool')(x)
    else:
        x = keras.layers.Conv2D(
            64,
            7,
            strides=2,
            padding='same',
            use_bias=False,
            name='stem_conv',
        )(inputs)
        x = keras.layers.BatchNormalization(name='stem_bn')(x)
        x = keras.layers.ReLU(name='stem_relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='stem_pool')(x)

    stage_filters = [64, 128, 256, 512]
    for stage_idx, filters in enumerate(stage_filters, start=1):
        first_stride = 1 if stage_idx == 1 else 2
        x = block_fn(x, filters, first_stride, name=f's{stage_idx}_b1')
        x = block_fn(x, filters, 1, name=f's{stage_idx}_b2')

    return x


def build_resnet18_fp32(input_shape=(224, 224, 3), num_classes: int = 1000):
    inputs = keras.Input(shape=input_shape, name='image')
    x = _resnet18_body(inputs, _basic_block_fp, quantized=False)
    x = keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    logits = keras.layers.Dense(num_classes, name='fc')(x)
    return keras.Model(inputs, logits, name='resnet18_fp32')


def build_resnet18_hgq2(
    input_shape=(224, 224, 3),
    num_classes: int = 1000,
    init_bw_k: int = 2,
    init_bw_a: int = 2,
):
    # IMPORTANT: The default QuantizerConfig uses KBI format (b=total bits, i=integer bits).
    # The `f0` parameter only works with the KIF quantizer type, NOT the default KBI.
    # We use b0/i0 (KBI) directly so the init_bw parameters actually take effect.
    #
    # For weights: b0 = init_bw_k + 2 (adding 2 integer bits), i0 = 2
    #   → fractional bits = b0 - i0 = init_bw_k, step = 2^(-init_bw_k)
    # For datalane: b0 = init_bw_a + 1 (adding 1 integer bit), i0 = 1
    #   → fractional bits = b0 - i0 = init_bw_a, step = 2^(-init_bw_a)
    #
    # With init_bw_k=2: step=0.25, which is TOO COARSE for deep networks
    #   (glorot init ≈ 0.04 for ResNet-18 stem, all weights → 0).
    # Use init_bw_k >= 6 for ResNet-18 to avoid dead initialization.
    with (
        QuantizerConfigScope(place='weight', overflow_mode='SAT_SYM', b0=init_bw_k + 2, i0=2, trainable=True),
        QuantizerConfigScope(place='bias', overflow_mode='WRAP', b0=init_bw_k + 2, i0=2, trainable=True),
        QuantizerConfigScope(place='datalane', i0=1, b0=init_bw_a + 1),
    ):
        inputs = keras.Input(shape=input_shape, name='image')
        x = _resnet18_body(inputs, _basic_block_hgq, quantized=True)
        x = keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
        logits = QDense(num_classes, name='fc')(x)
        model = keras.Model(inputs, logits, name='resnet18_hgq2')
    return model
