# Ramanujan Graph Generation (JSC)

This folder provides scripts to:
- generate a Ramanujan-style initialized JSC HGQ model,
- assign importance-aware bitwidths,
- plot symmetric topology bit matrices,
- plot circle-node connection graphs.

## Prerequisites

Use the `py12tf` conda environment (as used in this project):

```bash
conda activate py12tf
```

Or run commands with:

```bash
conda run -n py12tf <command>
```

## Files

- `generate_ramanujan_jsc_model.py`
  - all-in-one script, saves:
    - `.keras` model
    - `topology_bit_matrices.png`
    - `circle_connection_graph.png`
    - `report.json` (includes EBOPs and per-layer stats)
- `plot_circle_connection_graph.py` (optional)
  - standalone circle-graph plotting for existing models.

## Run: One Script (Model + Symmetric Topology + Circle Graph)

From `FPL/jsc`:

```bash
python ramanujan_graph/generate_ramanujan_jsc_model.py \
  --output_dir ramanujan_graph/output_bit03 \
  --min_degree 2 \
  --bit_low 0 \
  --bit_high 3 \
  --sample_size 512 \
  --symmetric_topology_plot \
  --mirror_edges
```

Equivalent command with explicit env:

```bash
conda run -n py12tf bash -lc 'cd /home/changhong/prj/ElasticTraining/FPL/jsc && \
python ramanujan_graph/generate_ramanujan_jsc_model.py \
  --output_dir ramanujan_graph/output_bit03 \
  --min_degree 2 \
  --bit_low 0 \
  --bit_high 3 \
  --sample_size 512 \
  --symmetric_topology_plot \
  --mirror_edges'
```

### Key options

- `--min_degree`: minimum trainable degree per output node (e.g. `2`)
- `--bit_low`: minimum bit in requested range (can be `0`)
- `--bit_high`: maximum bit in requested range (must be `>=1`)
- `--symmetric_topology_plot` / `--no_symmetric_topology_plot`
- `--mirror_edges` / `--no_mirror_edges` (circle graph mirror view)

The topology matrix plot is symmetric by default, and circle graph uses mirrored edges by default.

## Optional: Standalone Circle Plot Script

```bash
python ramanujan_graph/plot_circle_connection_graph.py \
  --model_path ramanujan_graph/output_bit03/jsc_ramanujan_importance_init.keras \
  --out_png ramanujan_graph/output_bit03/circle_connection_graph.png
```

For strictly symmetric mirrored edge view (default):

```bash
python ramanujan_graph/plot_circle_connection_graph.py \
  --model_path ramanujan_graph/output_bit03/jsc_ramanujan_importance_init.keras \
  --out_png ramanujan_graph/output_bit03/circle_connection_graph_symmetric.png \
  --mirror_edges
```

Disable mirrored edges:

```bash
python ramanujan_graph/plot_circle_connection_graph.py \
  --model_path ramanujan_graph/output_bit03/jsc_ramanujan_importance_init.keras \
  --out_png ramanujan_graph/output_bit03/circle_connection_graph_raw.png \
  --no_mirror_edges
```

## Output Structure

After generation, `output_dir` typically contains:

- `jsc_ramanujan_importance_init.keras`
- `topology_bit_matrices.png`
- `circle_connection_graph.png`
- `report.json`

## Notes

- Run commands from `FPL/jsc` directory for simplest relative paths.
- `report.json` contains:
  - `bit_range`
  - `total_ebops`
  - `per_layer_ebops`
  - per-layer topology/bit histograms
- If no dataset file is found at `data/dataset.h5`, generation falls back to synthetic input for EBOPs measurement.




python generate_ramanujan_jsc_model.py \
  --output_dir ramanujan_graph/output_bit02 \
  --min_degree 2 \
  --bit_low 0 \
  --bit_high 2 \
  --sample_size 512 \
  --symmetric_topology_plot \
  --mirror_edges