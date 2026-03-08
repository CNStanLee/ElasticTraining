from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np

from .ramanujan_budget_utils import _flatten_layers, _get_kq_var


@dataclass
class LayerGraphData:
    name: str
    bit_matrix: np.ndarray
    weighted_matrix: np.ndarray


class TopologyGraphPlotter:
    """Plot weighted topology matrix and circle graph from a .keras model."""

    def __init__(
        self,
        symmetric_topology_plot: bool = False,
        mirror_edges: bool = False,
        plot_matrix: bool = False,
        strict_original_connections: bool = True,
    ) -> None:
        self.strict_original_connections = bool(strict_original_connections)
        self.plot_matrix = bool(plot_matrix)
        self.symmetric_topology_plot = bool(symmetric_topology_plot)
        self.mirror_edges = bool(mirror_edges)
        if self.strict_original_connections:
            # Enforce strict original graph rendering:
            # no symmetric matrix expansion, no mirrored synthetic edges.
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

    def _extract_bits(self, layer, kernel_shape: tuple[int, ...]) -> np.ndarray:
        bits = None
        try:
            from keras import ops

            bits = layer.kq.bits_(ops.shape(layer.kernel))
            bits = np.array(ops.convert_to_numpy(bits), dtype=np.float32)
        except Exception:
            bits = None

        if bits is None:
            b_var = _get_kq_var(layer.kq, "b")
            if b_var is None:
                b_var = _get_kq_var(layer.kq, "f")
            if b_var is not None:
                bits = np.array(b_var.numpy(), dtype=np.float32)

        if bits is None:
            bits = np.zeros(kernel_shape, dtype=np.float32)

        return self._broadcast_like(bits, kernel_shape)

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

            bits = self._extract_bits(layer, kernel.shape)
            k2 = self._to_2d(kernel)
            b2 = self._to_2d(bits)
            if b2.shape != k2.shape:
                b2 = self._broadcast_like(b2, k2.shape)

            bit_matrix = np.clip(b2, 0.0, None).astype(np.float32)
            weighted_matrix = (bit_matrix * np.abs(k2)).astype(np.float32)
            out.append(
                LayerGraphData(
                    name=layer.name,
                    bit_matrix=bit_matrix,
                    weighted_matrix=weighted_matrix,
                )
            )
        return out

    def plot_weighted_topology_matrices(
        self,
        layers: list[LayerGraphData],
        out_png: Path,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> None:
        if not layers:
            raise RuntimeError("No quantized kernel layers found in model.")

        values = [x.weighted_matrix[x.weighted_matrix > 0] for x in layers]
        values = [v for v in values if v.size > 0]
        vmax = float(np.percentile(np.concatenate(values), 99.0)) if values else 1.0
        vmax = max(vmax, 1e-8)

        n = len(layers)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
        axes = axes[0]
        im = None
        for i, item in enumerate(layers):
            ax = axes[i]
            m = item.weighted_matrix
            if self.symmetric_topology_plot:
                m = self._to_symmetric_matrix(m)
            im = ax.imshow(m, aspect="auto", vmin=0.0, vmax=vmax, cmap="viridis")
            ax.set_title(item.name)
            ax.set_xlabel("out")
            ax.set_ylabel("in")

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
        cbar.set_label("Edge score (bit * |weight|)")
        title_text = title or (
            "Weighted Topology Connectivity Matrices"
            + (" (symmetric view)" if self.symmetric_topology_plot else "")
        )
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
            raise RuntimeError("No quantized kernel layers found in model.")

        layer_sizes = [layers[0].weighted_matrix.shape[0]] + [x.weighted_matrix.shape[1] for x in layers]
        x_pos = np.arange(len(layer_sizes), dtype=np.float32)
        y_pos = [self._symmetric_y_positions(n) for n in layer_sizes]

        values = [x.weighted_matrix[x.weighted_matrix > 0] for x in layers]
        values = [v for v in values if v.size > 0]
        vmax = float(np.percentile(np.concatenate(values), 99.0)) if values else 1.0
        vmax = max(vmax, 1e-8)

        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=0.0, vmax=vmax)

        total_edges = 0
        for li, item in enumerate(layers):
            mat = item.weighted_matrix
            ys = y_pos[li]
            yt = y_pos[li + 1]
            src, dst = np.where(mat > 1e-8)
            total_edges += int(src.size)

            for i, j in zip(src, dst):
                score = float(mat[i, j])
                intensity = min(score / vmax, 1.0)
                linewidth = 0.5 + 1.6 * intensity
                alpha = 0.35 + 0.45 * intensity
                color = cmap(norm(score))
                ax.plot(
                    [x_pos[li], x_pos[li + 1]],
                    [ys[i], yt[j]],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )
                if self.mirror_edges and (abs(float(ys[i])) > 1e-9 or abs(float(yt[j])) > 1e-9):
                    ax.plot(
                        [x_pos[li], x_pos[li + 1]],
                        [-ys[i], -yt[j]],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                    )

        for li, n in enumerate(layer_sizes):
            y = y_pos[li]
            ax.scatter(
                np.full(n, x_pos[li]),
                y,
                s=28 if n <= 32 else 18,
                facecolors="white",
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )

        labels = ["input"] + [x.name for x in layers]
        for li, name in enumerate(labels):
            ax.text(x_pos[li], 1.05, f"{name}\n(n={layer_sizes[li]})", ha="center", va="bottom", fontsize=10)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Edge score (bit * |weight|)")

        ax.set_xlim(-0.4, x_pos[-1] + 0.4)
        ax.set_ylim(-1.08, 1.18)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        title_text = title or (
            f"Weighted Circle Connection Graph (active edges={total_edges})"
            + (" (mirror view)" if self.mirror_edges else "")
        )
        if subtitle:
            ax.set_title(f"{title_text}\n{subtitle}")
        else:
            ax.set_title(title_text)
        ax.set_frame_on(False)

        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

    def plot_from_keras(
        self,
        keras_path: str | Path,
        output_dir: str | Path = "results",
    ) -> dict[str, Path | None]:
        model_path = Path(keras_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = keras.models.load_model(model_path, compile=False)
        layers = self.extract_layer_graph_data(model)
        if not layers:
            raise RuntimeError(f"No graphable quantized layers found in model: {model_path}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = model_path.stem
        matrix_path = out_dir / f"{stem}_weighted_topology_matrix.png"
        circle_path = out_dir / f"{stem}_weighted_circle_graph.png"

        if self.plot_matrix:
            self.plot_weighted_topology_matrices(layers, matrix_path)
        else:
            matrix_path = None
        self.plot_circle_graph(layers, circle_path)
        return {
            "matrix_path": matrix_path,
            "circle_path": circle_path,
        }
