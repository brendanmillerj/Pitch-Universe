# Pitch-omics

Single-cell methods applied to pitching. Each pitch is a cell, its release and
flight measurements are the features, and pitchers are the samples. The result
is a map of every pitch type thrown in MLB and Triple-A, 2024–2026, on which
pitches that move and release alike sit together.

Live atlas: **[pitchomics.com](https://pitchomics.com)**

## What's here

| | |
|---|---|
| [`src/mlbpitch/`](src/mlbpitch/README.md) | Installable package. Fetches Statcast pitches into a verified parquet store and derives release/approach angles and dead-zone deltas |
| [`scripts/`](scripts/README.md) | Profiles, t-SNE maps, Triple-A placement and comps |
| `data/meta/` | Small per-season tables. Bulk pitch data is not tracked |

## Quickstart

```bash
pip install -e .
mlbpitch --season 2026 backfill               # fetch every completed game-date
mlbpitch --season 2026 verify                 # reconcile against the MLB schedule
mlbpitch --season 2026 --level aaa backfill   # same for Triple-A
```

```python
from mlbpitch import store
df = store.pitcher(645261)                    # feature columns
full = store.pitcher(645261, columns="*")     # every column
store.query("SELECT pitch_name, count(*) n FROM pitches GROUP BY 1 ORDER BY n DESC")
```

The store lives at `<repo>/data` unless `MLBPITCH_DATA` is set. The analysis
scripts need further packages, listed in [`scripts/README.md`](scripts/README.md).

## Method

A pitch type is described by 15 measurements of its shape, movement and
delivery. Location, command and usage are shown on the maps but never define
which pitches count as similar. Maps are t-SNE layouts (openTSNE). Triple-A
pitches are placed on the fixed MLB maps beside their 15 nearest MLB pitches.
MLB baselines are never refit on Triple-A data.

## Data

The pitch data belongs to MLB. It comes from Baseball Savant's public
`statcast_search` CSV export and the public statsapi schedule, neither of which
needs authentication. The raw cache and parquet store are not tracked, so a
clone re-fetches from MLB. The fetcher sends one request per game-date with 5
workers and backs off on 429 and 5xx responses. Please keep it that polite.
