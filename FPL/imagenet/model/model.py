import keras
from keras import layers


def _basic_block(x, filters, stride=1, name='block'):
    """
    ResNet-18/34 BasicBlock: two 3×3 conv layers with a skip connection.

    If the spatial size or channel depth changes, a 1×1 conv projects the
    shortcut to match the residual branch.
    """
    shortcut = x

    # --- residual branch ---
    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False, name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)

    x = layers.Conv2D(filters, 3, strides=1, padding='same',
                      use_bias=False, name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)

    # --- projection shortcut when dimensions change ---
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 use_bias=False, name=f'{name}_proj')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_proj_bn')(shortcut)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name}_relu2')(x)
    return x


def _make_layer(x, filters, num_blocks, stride, name):
    """Stack ``num_blocks`` BasicBlocks; the first may down-sample."""
    x = _basic_block(x, filters, stride=stride, name=f'{name}_0')
    for i in range(1, num_blocks):
        x = _basic_block(x, filters, stride=1, name=f'{name}_{i}')
    return x


def get_resnet18(num_classes: int = 1000, input_shape=(224, 224, 3)):
    """
    Build a standard float32 ResNet-18 Keras functional model.

    Args:
        num_classes (int): Number of output classes.
        input_shape (tuple): Input image shape (H, W, C).

    Returns:
        keras.Model
    """
    inputs = keras.Input(shape=input_shape, name='input')

    # Stem
    x = layers.Conv2D(64, 7, strides=2, padding='same',
                      use_bias=False, name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same',
                            name='stem_pool')(x)

    # Residual stages  [2, 2, 2, 2]
    x = _make_layer(x, 64,  num_blocks=2, stride=1, name='layer1')
    x = _make_layer(x, 128, num_blocks=2, stride=2, name='layer2')
    x = _make_layer(x, 256, num_blocks=2, stride=2, name='layer3')
    x = _make_layer(x, 512, num_blocks=2, stride=2, name='layer4')

    # Head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    outputs = layers.Dense(num_classes, name='fc')(x)

    model = keras.Model(inputs, outputs, name='resnet18')
    return model
