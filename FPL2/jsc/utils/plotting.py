"""
拓扑图绘制 — 位宽矩阵 + 圈连接图 + 周期性绘图回调
==============================================

TopologyGraphPlotter
    从 keras 模型提取层间连接信息，绘制位宽矩阵图和圈连接图。

TopologyPlotCallback
    Keras Callback，在初始化和每 N epochs 自动绘制拓扑图。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import keras
import matplotlib
matplotlib.use('Agg')  # 非交互后端，服务器安全
import matplotlib.pyplot as plt
import numpy as np

from . import _flatten_layers, _get_kq_var


@dataclass
class LayerGraphData:
    name: str
    bit_matrix: np.ndarray
    weighted_matrix: np.ndarray


class TopologyGraphPlotter:
    """Plot weighted topology matrix and circle graph from model layer data."""

    def __init__(
        self,
        symmetric_topology_plot: bool = False,
        mirror_edges: bool = False,
        plot_matrix: bool = True,
        strict_original_connections: bool = True,
    ) -> None:
        self.strict_original_connections = bool(strict_original_connections)
        self.plot_matrix = bool(plot_matrix)
        self.symmetric_topology_plot = bool(symmetric_topology_plot)
        self.mirror_edges = bool(mirror_edges)
        if self.strict_original_connections:
            self.symmetric_topology_plot = False
            self.mirror_edges = False

    @staticmethod
    def _to_2d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr.astype(np.float32)
        if arr.ndim < 2:
            return arr.reshape(-1, 1).astype(np.float32)
        return arr.reshape(-1, arr.shape[-1]).astype(np.float32)

    @staticmethod
    def _broadcast_like(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        arr = np.array(arr, dtype=np.float32)
        if arr.shape == target_shape:
            return arr
        try:
            return np.broadcast_to(arr, target_shape).astype(np.float32)
        except Exception:
            fill = float(np.mean(arr)) if arr.size else 0.0
            return np.full(target_shape, fill, dtype=np.float32)

    def _extract_raw_b(self, layer, kernel_shape: tuple[int, ...]) -> np.ndarray:
        """Return raw (pre-round_conv) _b parameter values as float32."""
        b_var = _get_kq_var(layer.kq, "b")
        if b_var is None:
            b_var = _get_kq_var(layer.kq, "f")
        if b_var is not None:
            raw = np.array(b_var.numpy(), dtype=np.float32)
            return self._broadcast_like(raw, kernel_shape)
        # fallback: try bits_()
        try:
            from keras import ops
            bits = layer.kq.bits_(ops.shape(layer.kernel))
            bits = np.array(ops.convert_to_numpy(bits), dtype=np.float32)
            return self._broadcast_like(bits, kernel_shape)
        except Exception:
            return np.zeros(kernel_shape, dtype=np.float32)

    @staticmethod
    def _to_symmetric_matrix(m: np.ndarray) -> np.ndarray:
        m_lr = np.concatenate([np.fliplr(m), m], axis=1)
        return np.concatenate([np.flipud(m_lr), m_lr], axis=0)

    @staticmethod
    def _symmetric_y_positions(n: int, span: float = 0.9) -> np.ndarray:
        if n <= 1:
            return np.array([0.0], dtype=np.float32)
        return np.linspace(-span, span, n, dtype=np.float32)

    def extract_layer_graph_data(self, model: keras.Model) -> list[LayerGraphData]:
        out: list[LayerGraphData] = []
        for layer in _flatten_layers(model):
            if not hasattr(layer, "kernel") or getattr(layer, "kq", None) is None:
                continue
            kernel = np.array(layer.kernel.numpy(), dtype=np.float32)
            if kernel.ndim < 2:
                continue
            bits = self._extract_raw_b(layer, kernel.shape)
            k2 = self._to_2d(kernel)
            b2 = self._to_2d(bits)
            if b2.shape != k2.shape:
                b2 = self._broadcast_like(b2, k2.shape)
            bit_matrix = np.clip(b2, 0.0, None).astype(np.float32)
            weighted_matrix = (bit_matrix * np.abs(k2)).astype(np.float32)
            out.append(LayerGraphData(
                name=layer.name,
                bit_matrix=bit_matrix,
                weighted_matrix=weighted_matrix,
            ))
        return out

    def plot_weighted_topology_matrices(
        self,
        layers: list[LayerGraphData],
        out_png: Path,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> None:
        if not layers:
            return
        vmax = 8.0
        n = len(layers)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
        axes = axes[0]
        im = None
        for i, item in enumerate(layers):
            ax = axes[i]
            m = item.bit_matrix
            if self.symmetric_topology_plot:
                m = self._to_symmetric_matrix(m)
            im = ax.imshow(m, aspect="auto", vmin=0.0, vmax=vmax, cmap="viridis")
            ax.set_title(item.name)
            ax.set_xlabel("out")
            ax.set_ylabel("in")
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
        cbar.set_label("_b (raw, pre-round)")
        title_text = title or "Weighted Topology Connectivity Matrices"
        if subtitle:
            fig.suptitle(f"{title_text}\n{subtitle}")
        else:
            fig.suptitle(title_text)
        fig.subplots_adjust(left=0.05, right=0.94, top=0.84, bottom=0.12, wspace=0.22)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=180)
        plt.close(fig)

    def plot_circle_graph(
        self,
        layers: list[LayerGraphData],
        out_png: Path,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> None:
        if not layers:
            return
        layer_sizes = [layers[0].bit_matrix.shape[0]] + [x.bit_matrix.shape[1] for x in layers]
        x_pos = np.arange(len(layer_sizes), dtype=np.float32)
        y_pos = [self._symmetric_y_positions(n) for n in layer_sizes]
        vmax = 8.0
        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=0.0, vmax=vmax)

        total_edges = 0
        for li, item in enumerate(layers):
            mat = item.bit_matrix
            ys = y_pos[li]
            yt = y_pos[li + 1]
            # 只绘制 b > 0.4 的连接（soft_floor ≤ 0.05 不可见）
            src, dst = np.where(mat > 0.4)
            total_edges += int(src.size)
            for i, j in zip(src, dst):
                bits_val = float(mat[i, j])
                intensity = min(bits_val / vmax, 1.0)
                linewidth = 0.5 + 1.6 * intensity
                alpha = 0.35 + 0.45 * intensity
                color = cmap(norm(bits_val))
                linestyle = "--" if bits_val < 0.4 else "-"
                ax.plot(
                    [x_pos[li], x_pos[li + 1]], [ys[i], yt[j]],
                    color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                )

        for li, n in enumerate(layer_sizes):
            y = y_pos[li]
            ax.scatter(
                np.full(n, x_pos[li]), y,
                s=28 if n <= 32 else 18,
                facecolors="white", edgecolors="black", linewidths=0.8, zorder=3,
            )

        labels = ["input"] + [x.name for x in layers]
        for li, name in enumerate(labels):
            ax.text(x_pos[li], 1.05, f"{name}\n(n={layer_sizes[li]})",
                    ha="center", va="bottom", fontsize=10)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("_b (raw, pre-round)")

        ax.set_xlim(-0.4, x_pos[-1] + 0.4)
        ax.set_ylim(-1.08, 1.18)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        title_text = title or f"Circle Connection Graph (active edges={total_edges})"
        if subtitle:
            ax.set_title(f"{title_text}\n{subtitle}")
        else:
            ax.set_title(title_text)
        ax.set_frame_on(False)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)


def plot_topology(model, output_dir: str | Path, tag: str,
                  ebops: float | None = None, plot_matrix: bool = True):
    """绘制模型拓扑的快捷函数。"""
    out = Path(output_dir) / 'topology_plots'
    out.mkdir(parents=True, exist_ok=True)

    plotter = TopologyGraphPlotter(
        symmetric_topology_plot=False,
        mirror_edges=False,
        plot_matrix=plot_matrix,
    )
    layers = plotter.extract_layer_graph_data(model)
    if not layers:
        return

    subtitle = f'ebops={ebops:.0f}' if ebops is not None else None

    if plot_matrix:
        plotter.plot_weighted_topology_matrices(
            layers, out / f'{tag}_matrix.png',
            title=f'Topology Matrix — {tag}', subtitle=subtitle,
        )
    plotter.plot_circle_graph(
        layers, out / f'{tag}_circle.png',
        title=f'Circle Graph — {tag}', subtitle=subtitle,
    )
    print(f'  [TopologyPlot] saved: {tag}  (→ {out})')


class TopologyPlotCallback(keras.callbacks.Callback):
    """每 plot_interval epochs 自动绘制一次拓扑图。

    Parameters
    ----------
    output_dir : str | Path
        输出目录
    plot_interval : int
        绘图间隔 (epochs), 默认 1000
    sample_input : tf.Tensor | None
        用于 eBOPs 计算的输入样本
    plot_matrix : bool
        是否同时绘制位宽矩阵图
    """

    def __init__(
        self,
        output_dir: str | Path,
        plot_interval: int = 1000,
        sample_input=None,
        plot_matrix: bool = True,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.plot_interval = max(1, plot_interval)
        self.sample_input = sample_input
        self.plot_matrix = plot_matrix

    def _get_ebops(self) -> float | None:
        if self.sample_input is None:
            return None
        try:
            from .pruning import compute_model_ebops
            return compute_model_ebops(self.model, self.sample_input)
        except Exception:
            return None

    def on_epoch_end(self, epoch, logs=None):
        # epoch 是 0-indexed，第一次在 epoch=plot_interval-1 时触发
        # 即第 1000, 2000, ... epoch 结束后
        real_epoch = epoch + 1
        if real_epoch % self.plot_interval != 0:
            return

        ebops = self._get_ebops()
        tag = f'epoch_{real_epoch:06d}'
        try:
            plot_topology(
                self.model, self.output_dir, tag,
                ebops=ebops, plot_matrix=self.plot_matrix,
            )
        except Exception as e:
            print(f'  [TopologyPlotCallback] plot failed at epoch {real_epoch}: {e}')
