> **Superseded 2026-11-04** by [umap_benchmark.md](umap_benchmark.md): the maps now leave usage out
> (27 measures) and the new benchmark adds t-SNE, PaCMAP, held-out placement, outcome signal,
> stability and a settings grid. Numbers below are from the 32-measure build.

# Why the Pitch Universe uses UMAP

**Short answer:** pitches differ on far more than movement. A ride-vs-break plot shows 2 of the 32
measures we use, and only **8%** of a pitch's truly closest matches end up next to it there. A 2D
UMAP keeps **24%**, three times as many; the **3D UMAP the atlas now opens with keeps 34%**, four
times as many, with the fewest false neighbors of anything we tested. It is the most faithful
picture of overall pitch similarity, so it is the map. Comps and rankings are never read off the
map; they use all 32 measures directly.

## The question

"Why not just plot horizontal break against induced vertical break, like Savant?"

A movement plot is the right picture of *what one pitch does*, and the atlas shows one on every
pitcher card. The map has a different job: put pitches that are similar *overall* next to each
other, so that browsing a neighborhood surfaces genuinely comparable pitchers. Overall similarity
here means 32 measures, not 2:

| Block | Measures |
|---|---|
| Shape | velocity, spin rate, spin axis, induced vertical break, horizontal break, accelerations |
| Derived | approach and release angles vs. expected, ride/run vs. arm-slot expectation (dead zone), seam-shifted wake |
| Delivery | arm angle, release height and side, extension, extension relative to height |
| Intent | location (inside/away, height), zone rate |
| Consistency | release-point, movement and velocity scatter |
| Usage | overall, when ahead, when behind, vs. LHB, vs. RHB |

Each block is scaled to equal total weight, so no single block dominates.

## The test

One dot is one pitcher-season's version of one pitch type, 2024 to 2026 (7,889 in all), and each
pitch type is tested on its own map.

1. For every pitch, find its 10 most similar pitches of the same type using all 32 measures. These
   are its "true neighbors", the same similarity the atlas uses for comps.
2. For each 2D picture, find the pitch's 10 nearest dots.
3. Score the picture by the share of true neighbors that are also nearest dots. Random placement
   would score about 1%.

We also report *trustworthiness* (0 to 1; 1 means a picture shows no false neighbors) and, for the
All-pitches map, how often a dot's nearest dots are the same pitch type.

## Results

True neighbors kept on the picture, by pitch type:

| Pitch type | Movement (ride × break) | Velocity × ride | PCA, top 2 directions | UMAP 2D | **UMAP 3D** |
|---|---|---|---|---|---|
| Four-seam | 5% | 4% | 8% | 18% | **29%** |
| Sinker | 7% | 6% | 9% | 23% | **33%** |
| Cutter | 8% | 6% | 12% | 26% | **34%** |
| Slider | 7% | 4% | 8% | 23% | **32%** |
| Sweeper | 9% | 7% | 11% | 28% | **38%** |
| Curveball | 10% | 8% | 10% | 31% | **39%** |
| Changeup | 8% | 7% | 9% | 25% | **35%** |
| Splitter | 17% | 13% | 22% | 42% | **48%** |
| **All (weighted)** | **8%** | **6%** | **10%** | **24%** | **34%** |

| | Movement | Velocity × ride | PCA | UMAP 2D | **UMAP 3D** |
|---|---|---|---|---|---|
| Trustworthiness | 0.72 | 0.66 | 0.77 | 0.85 | **0.91** |

On the All-pitches map, **86%** of a dot's 10 nearest dots are the same pitch type on 3D UMAP (85%
in 2D), vs. **71%** on a movement plot.

UMAP is best for every pitch type, by a wide margin over both the movement plot and the best
linear 2D projection (PCA), and the third dimension adds about ten points of neighbor preservation
on top of 2D. The atlas therefore opens in 3D, with a 2D toggle for the group ellipses.

## What this does and does not claim

- **No low-dimensional picture keeps most neighbors, UMAP included (34% in 3D).** Compressing 32 measures into 2 or 3 loses
  information. That is why the atlas computes comps, uniqueness and every ranking on all 32
  measures and never from map distances. The map is for exploring; the numbers come from the full data.
- **UMAP's axes have no units, and the gaps between islands mean nothing.** Only "nearby means
  similar" carries meaning. A dot's absolute position, and how far apart two islands sit, should
  not be read.
- **UMAP compresses outliers toward the middle.** Distance from the center of the map understates
  how unusual a pitch is. "Unique unicorns" are measured on all 32 measures for this reason.
- **The layout depends on settings and a random seed.** We fix the seed (0), use 15 neighbors and
  a minimum distance of 0.3, for both the 2D and 3D layouts. Neighborhoods are stable across runs;
  exact positions are not.
- **3D trades some readability for faithfulness.** Points can hide behind each other until you rotate;
  the 2D toggle is there when a flat, ellipse-annotated view reads better.
- **The movement plot is still the right tool for one pitch's shape.** The two pictures answer
  different questions, and the atlas shows both.

## Reproduce

```bash
python scripts/why_umap.py
```

It reads `data/meta/profiles_type_season_2024_2026.parquet` and
`data/meta/atlas_type_2024_2026.parquet` (both built by `scripts/atlas_build.py`) and prints the
tables above. Results are as of the 2026-10-31 build.
