"""GPU 检测与内存增长配置。"""

import tensorflow as tf


def get_tf_device() -> str:
    """检测 GPU 并启用内存增长，返回设备字符串。"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            device = 'GPU:0'
            print(f'GPU(s) detected ({len(gpus)}), using {device}')
        else:
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            device = 'cpu:0'
            print('No GPUs detected, using CPU')
    except Exception:
        device = 'cpu:0'
        print('GPU detection error, falling back to CPU')
    return device
