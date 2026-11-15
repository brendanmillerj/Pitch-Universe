# scripts

Flat on purpose: many scripts import each other through `sys.path`.

## Maps, comps and changes

| | |
|---|---|
| `atlas_core.py` | Pitcher-season profiles → t-SNE maps → Triple-A placement → comps, uniqueness, biggest changes. Run it to rebuild `data/meta/atlas_type_*.parquet` |
| `profile_embed.py` | Profile embedding; defines `IDENTITY_BLOCKS`, the 15 map features |
| `comps.py` | Lookalikes, uniqueness and where success concentrates |
| `pitch_subclusters.py` | Clusters within each pitch type (labels in `pitcher_types.json`) |
| `midseason_changes.py` | League-wide changepoint scan within each season |

## Benchmark and paper

| | |
|---|---|
| `umap_benchmark.py` | t-SNE against UMAP, PaCMAP, PHATE, PCA and the movement plot |
| `umap_benchmark_figures.py` | Tables with pitcher-clustered bootstrap intervals |
| `umap_benchmark_plots.py` | Figures for `analysis/umap_benchmark.md` |
| `stratification_scan.py` | Ancestry control for pitch–outcome associations |
| `design_forecast.py` | Does moving a pitch on the map forecast its results? |
| `figure_style.py`, `cell_typography.py` | Shared figure style (Cell print rules). Uses Arial if installed, with a warning otherwise |

## archive/

The earlier UMAP pipeline, superseded by the t-SNE maps and kept for the record.
Run with `PYTHONPATH=scripts` since some import `profile_embed`.

## Dependencies

Beyond the package's own: `scikit-learn openTSNE leidenalg python-igraph
umap-learn pynndescent scipy matplotlib adjustText Pillow`. The benchmark also
needs `pacmap trimap phate harmonypy`.
