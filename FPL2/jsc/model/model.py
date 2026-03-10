import keras
from hgq.config import QuantizerConfigScope
from hgq.layers import QDense, QDenseT, QEinsumDenseBatchnorm


def get_model_hgq(init_bw_k=3, init_bw_a=3):
    with (
        QuantizerConfigScope(place=('weight'), overflow_mode='SAT_SYM', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place=('bias'), overflow_mode='WRAP', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place='datalane', i0=1, f0=init_bw_a),
    ):
        inp = keras.Input(shape=(16,))
        out = QEinsumDenseBatchnorm('bc,cC->bC', 64, name='t1', bias_axes='C', activation='relu')(inp)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 64, name='t2', bias_axes='C', activation='relu')(out)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 32, name='t3', bias_axes='C', activation='relu')(out)
        out = QDense(5, name='out')(out)

    return keras.Model(inp, out)


def get_model_hgq_deep(init_bw_k=3, init_bw_a=3, widths=None):
    """可配置深度的 HGQ 模型，用于研究梯度随层数的变化。

    参数:
        widths : 隐藏层宽度列表，例如：
            [64, 64, 32]          -> 3 层（等价于原始 baseline）
            [64, 64, 64, 32]      -> 4 层
            [64]*6 + [32]         -> 7 层（开始出现梯度问题的典型区域）
            [64]*10 + [32]        -> 11 层（梯度消失/爆炸验证）
        默认 [64, 64, 32]（与 get_model_hgq 等价，方便对照）。

    层命名规则: t0, t1, ..., t{n-1}, out
    每层均使用 QEinsumDenseBatchnorm + ReLU（BN 有助于缓解梯度问题，
    可对比去掉 BN 时的差异）。
    """
    if widths is None:
        widths = [64, 64, 32]  # 默认等价于原始 baseline

    with (
        QuantizerConfigScope(place=('weight'), overflow_mode='SAT_SYM', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place=('bias'), overflow_mode='WRAP', f0=init_bw_k, trainable=True),
        QuantizerConfigScope(place='datalane', i0=1, f0=init_bw_a),
    ):
        inp = keras.Input(shape=(16,))
        out = inp
        for i, w in enumerate(widths):
            out = QEinsumDenseBatchnorm(
                'bc,cC->bC', w, name=f't{i}', bias_axes='C', activation='relu'
            )(out)
        out = QDense(5, name='out')(out)

    return keras.Model(inp, out)


def get_model_hgqt(init_bw=10, init_int=2):
    with QuantizerConfigScope(k0=1, b0=init_bw, i0=init_int):
        with QuantizerConfigScope(place='table', homogeneous_axis=(0,)):
            inp = keras.layers.Input((16,))
            out = QDenseT(20, batch_norm=True)(inp)
            out = QDenseT(5, batch_norm=False)(inp)
    return keras.Model(inp, out)
