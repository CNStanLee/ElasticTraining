import math
import keras


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warm-up for ``warmup_steps`` steps, then cosine decay to ``eta_min``.

    Pass to a Keras optimizer as the ``learning_rate`` argument.
    """

    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int,
                 eta_min: float = 1e-5):
        super().__init__()
        self.base_lr      = float(base_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)
        self.eta_min      = float(eta_min)

    def __call__(self, step):
        import tensorflow as tf
        step = tf.cast(step, tf.float32)

        # --- warm-up phase ---
        warmup_lr = self.base_lr * (step + 1.0) / (self.warmup_steps + 1.0)

        # --- cosine decay phase ---
        decay_steps   = self.total_steps - self.warmup_steps
        progress      = (step - self.warmup_steps) / tf.maximum(decay_steps, 1.0)
        cosine_scale  = 0.5 * (1.0 + tf.cos(math.pi * tf.minimum(progress, 1.0)))
        cosine_lr     = self.eta_min + (self.base_lr - self.eta_min) * cosine_scale

        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'base_lr':      self.base_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps':  self.total_steps,
            'eta_min':      self.eta_min,
        }
