# scripts

| | |
|---|---|
| `atlas_core.py` | Pitcher-season profiles → t-SNE maps → Triple-A placement → comps, uniqueness, biggest changes. Run it to rebuild `data/meta/atlas_type_*.parquet` |
| `profile_embed.py` | Profile embedding; defines `IDENTITY_BLOCKS`, the 15 map features |
| `midseason_changes.py` | League-wide changepoint scan within each season |

Dependencies beyond the package's own: `scikit-learn openTSNE leidenalg
python-igraph umap-learn scipy`.
