import tensorflow as tf

def get_tf_device():
    try:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    # Best-effort: if memory growth setting fails, continue.
                    pass
                device = 'GPU:0'
                print(f'GPU(s) detected ({len(gpus)}), using {device}')
            else:
                # No GPUs available: explicitly hide GPUs to ensure CPU-only execution
                try:
                    tf.config.set_visible_devices([], 'GPU')
                except Exception:
                    pass
                device = 'cpu:0'
                print('No GPUs detected, using CPU')
        except Exception:
            # Fallback to CPU if any error occurs during GPU detection
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            device = 'cpu:0'
            print('GPU detection error, falling back to CPU')
    except Exception:
        # If TensorFlow import/config fails, set a safe default device
        device = 'cpu:0'

    return device