#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    import model.model  # noqa: F401  # register custom layers before load_model
    from utils.topology_graph_plot_utils import TopologyGraphPlotter

    parser = argparse.ArgumentParser(
        description="Plot weighted topology matrix + circle graph from a .keras model"
    )
    parser.add_argument("keras_path", type=str, help="Path to .keras model file")
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Use mirrored symmetric view for topology matrix (default: off)",
    )
    parser.add_argument(
        "--mirror_edges",
        action="store_true",
        help="Draw mirrored edges in circle graph (default: off)",
    )
    parser.add_argument(
        "--plot_matrix",
        action="store_true",
        help="Also generate topology matrix plot (default: off)",
    )
    parser.add_argument(
        "--allow_synthetic_edges",
        action="store_true",
        help="Allow symmetric/mirrored synthetic rendering (default: strict original connections)",
    )
    args = parser.parse_args()

    plotter = TopologyGraphPlotter(
        symmetric_topology_plot=bool(args.symmetric),
        mirror_edges=bool(args.mirror_edges),
        plot_matrix=bool(args.plot_matrix),
        strict_original_connections=(not bool(args.allow_synthetic_edges)),
    )
    model_stem = Path(args.keras_path).stem
    outputs = plotter.plot_from_keras(
        keras_path=args.keras_path,
        output_dir=here / "results" / "plot_graph" / model_stem,
    )

    if outputs["matrix_path"] is not None:
        print(f"Matrix plot: {outputs['matrix_path']}")
    print(f"Circle plot: {outputs['circle_path']}")


if __name__ == "__main__":
    main()
