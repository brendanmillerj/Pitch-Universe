# MLB pitch store

Every pitch thrown in the 2026 MLB regular season, in a verified local parquet
store. Built to feed a "single-cell RNA-seq for pitchers" dashboard: each pitch
is a cell, the biophysical columns (`arm_angle`, `spin_axis`, `release_speed`,
`pfx_x/z`, `release_extension`, `release_pos_*`) are the features, pitchers and
teams are the samples.

## Current state

| | |
|---|---|
| Season | 2026 (in progress) |
| Pitches | 458,522 |
| Game-dates | 120 (2026-03-25 → 2026-07-25) |
| Pitchers | 758 |
| Parquet | 52 MB across 5 month partitions |
| Raw CSV cache | ~150 MB gzipped |
| Full backfill | 2m13s |
| Incremental refresh | ~2s (no-change day) |
| Per-pitcher query | ~21 ms |
| League-wide aggregation | ~9 ms |

## Install

```bash
pip install -e .
```

By default the store lives at `<repo>/data`. Set `MLBPITCH_DATA` to point
somewhere else (scratch, a shared group copy, or any non-editable install).

## Usage

```bash
mlbpitch backfill                 # fetch every completed date not yet cached
mlbpitch refresh                  # daily: new dates + 10-day trailing window
mlbpitch verify                   # reconcile against the schedule oracle
mlbpitch build                    # rebuild parquet from raw cache, no network
mlbpitch pitcher 645261 --live    # inspect one pitcher, compare to live Savant
```

Equivalently `python -m mlbpitch.ingest <cmd>`. Nothing is scheduled — run
`refresh` yourself, or see `scripts/refresh.sbatch`.

<details>
<summary>USC CARC specifics</summary>

The interpreter is a venv at `/project2/hassy_472/brendajm/envs/mlb/bin/python`
(`--system-site-packages` over the module python 3.11.9). A conda/mamba solve
gets OOM-killed in a 4 GB OnDemand allocation, which is why it is a venv.
Set `MLBPITCH_PYTHON` to override the interpreter used by `refresh.sbatch`.

</details>

Querying:

```python
from mlbpitch import store

df   = store.pitcher(645261)                      # 27 feature columns, ~21 ms
full = store.pitcher(645261, columns="*")         # all 113 columns, ~51 ms
top  = store.query("SELECT pitch_name, count(*) n FROM pitches GROUP BY 1 ORDER BY n DESC")
con  = store.connect()                            # cached DuckDB conn, `pitches` view
```

`connect()` caches its connection (opening one and creating the view costs
~11 ms, which otherwise dominates a ~10 ms query). `build_month` invalidates it
automatically; call `store.reset_connection()` if you rewrite parquet by some
other route.

## Data provenance

The code here is original. The underlying pitch data is **MLB's** — retrieved
from Baseball Savant's public `statcast_search` CSV export and the public
statsapi schedule endpoint, neither of which requires authentication. No data is
redistributed in this repository: `data/raw` and `data/pitches` are gitignored,
and only fitted model coefficients and the ingest manifest are tracked. Anyone
cloning this re-fetches from MLB themselves.

The fetcher is deliberately polite to a free public service: one request per
game-date, 5 concurrent workers, exponential backoff on 429/5xx. Please keep it
that way.

## Design

**Source is Baseball Savant's `statcast_search/csv`, not statsapi.** The
statsapi `playByPlay`/`feed/live` endpoints lack `arm_angle`, `spin_axis`, and
`release_extension` — the exact features this project is built on — and cost
2,456 requests / ~10–15 GB per season versus Savant's ~184 requests / ~500 MB.
statsapi is used only as the schedule oracle.

**Fetched one game-date per request.** Savant silently truncates at 25,000 rows:
a three-week request returns exactly 25,000 rows covering only the first seven
dates, with no error. The busiest single date observed is 5,052 pitches, so a
one-day chunk has ~5x headroom and truncation is structurally impossible.
`savant.fetch_day` still raises if a response reaches the cap, as a tripwire for
the API contract changing.

**Raw gzipped CSV is kept permanently.** The parquet store is fully rebuildable
from it (`ingest build`), so dtype or schema changes never require re-scraping.

**Month-level parquet partitions** (5–7 files/season), rows sorted by
`pitcher, game_date, at_bat_number, pitch_number` so row-group statistics let
DuckDB skip most groups on the dominant per-pitcher filter. Per-pitcher files
would mean 1,000+ small files, which is what hurts on this filesystem.

**Refresh re-fetches a trailing window, it does not just append.** Statcast
revises recent games; a pure append silently preserves stale pitch
classifications. Months are rebuilt only when a date's content sha256 changes.

## Gotchas encoded here

These were all found the hard way against the live API:

- **25,000-row silent truncation** — the reason for day-level chunking.
- **`player_name` is `"Last, First"`** and `des` is free text, both containing
  commas. Any `str.split(",")` parse misaligns every column after the sixth.
  `schema.read_raw_frame` uses a real CSV reader.
- **Suspended games are double-listed.** statsapi reports a suspended game under
  both its original date and its resumption date, but Savant files all its
  pitches under the original. `schedule.expected_games_by_date` attributes each
  `game_pk` to its earliest completed date; without this, the resumption date
  looks permanently one game short. (Hit this on 2026-06-17 / game 824912.)
- **Savant lags statsapi.** A game is `Final` in statsapi before its pitches are
  published. `verify` treats missing games on dates within `--lag-days` (default
  2) as a warning, not a failure.
- **Six `*_deprecated` columns** are always null and are dropped at build time.

## Pitch angles (VRA/HRA at release, VAA/HAA at the plate)

Four stored parquet columns computed in `physics.py` from the Statcast
trajectory fit. The fit parameters are defined at the **y=50 ft** tracking
plane, **not** at `release_pos_y` (~54 ft) — using the latter as the reference
frame would be wrong. `approach_angles` propagates forward to the front of the
plate (y=17/12 ft); `release_angles` propagates *backward* to each pitch's own
`release_pos_y`.

```
# approach (yf = 17/12);  release (yf = release_pos_y, per pitch)
vy_f = -sqrt(vy0^2 - 2*ay*(y0 - yf));  t = (vy_f - vy0)/ay
vertical   = -degrees(atan((vz0 + az*t) / vy_f))
horizontal = -degrees(atan((vx0 + ax*t) / vy_f))
```

|  | at release | at the plate |
|---|---|---|
| vertical | `vra` — 4-seam −1.85°, curveball **+0.50°** (lofted) | `vaa` — 4-seam −4.65°, curveball −9.60° |
| horizontal | `hra` | `haa` |

Gravity only ever steepens a pitch in flight, so **VRA > VAA for every pitch
type** (mean gain 6.7°) — asserted in `verify`. Less negative VAA is "flatter".

**Horizontal sign convention.** Negative = aimed toward a right-handed batter.
Established empirically rather than assumed: hit-by-pitches to RHB average
`plate_x` = −1.95, to LHB +2.02, so RHB occupy negative `plate_x`. Note this
makes right-handed *pitchers* average **positive** HRA/HAA — they release from
the third-base side (`release_pos_x` ≈ −1.86) and must throw rightward to reach
the plate. Pitcher handedness and target side are separate things.

**All four raw angles are location-confounded and should not be modelled
directly.** Measured here:

| angle | confounder | raw \|r\| | after adjustment |
|---|---|---|---|
| `vaa` | `plate_z` | 0.89 | **0.000** |
| `vra` | `plate_z` | 0.79 | **0.000** |
| `haa` | `release_pos_x` | 0.95 | **0.000** |
| `hra` | `release_pos_x` | 0.73 | **0.000** |

`store.add_angles_adjusted(df)` returns `{vaa,haa,vra,hra}_above_avg` —
handedness-mirrored residuals against per-pitch-type baselines (vertical angles
on `plate_z`; horizontal on `plate_x` + `release_pos_x`). Coefficients live in
`data/meta/angle_baseline_{season}.parquet`, refit on every build.

Model choices were made by fit, not assumption: linear in `plate_z` suffices
(quadratic/cubic add ≤0.002 R²), and the horizontal fits *require*
`release_pos_x` — with `plate_x` alone R²≈0.45, adding release point reaches
0.88–0.93.

These are deliberately not stored as parquet columns: the fit is season-wide while
`build_month` only sees one month, and the baseline legitimately shifts as the
season accumulates — a stored column would freeze a mid-season snapshot.

What `vaa_above_avg` buys, concretely: ranked by it, the flattest four-seamers in
2026 are Kimbrel, Sewald, Joe Ryan and Misiorowski — the actual known flat-fastball
arms — and it correlates **−0.85 with release height**, i.e. it reads arm slot
rather than location. Raw VAA would have partly just re-encoded `plate_z`.

## Dynamic Dead Zone deltas

The "dead zone" is the region of the movement plot where a fastball gets poor
results. It isn't a fixed box — higher arm slots naturally produce more induced
vertical break, so the dead zone moves with the pitcher. The Dynamic Dead Zone
(Max Bay's model, framed in the FanGraphs piece on Andrés Muñoz) conditions on
release characteristics and asks how far actual movement sits from what they
predict.

`ddz.add_ddz_delta(df)` appends:

| column | meaning |
|---|---|
| `ddz_delta_az` | observed `az` − predicted from arm angle + height-scaled extension |
| `ddz_delta_ax` | same, horizontal |
| `ddz_delta_az_velo` | additionally conditions on `release_speed` |
| `ddz_delta_ax_velo` | same, horizontal |

Units are **ft/sec²** — these are accelerations, not inches of break. Positive
`az` delta = more ride than the pitcher's slot and extension predict.

The article specifies a 4-D multivariate normal over (arm_angle, scaled
extension, ax, az). The quantity needed from it is E[(ax,az) | arm_angle,
scaled_ext], and for a joint Gaussian that conditional mean *is* the
least-squares fit — so coefficients are stored in their equivalent linear form
(`data/meta/ddz_baseline_{season}.parquet`), with residual SDs alongside for
z-scoring.

**Pitcher height** is required (extension is scaled by it: a 6'8" pitcher
releasing 7 ft down the mound is mechanically ordinary; a 5'10" pitcher doing the
same is not). Statcast carries no height, so `people.py` pulls it from statsapi
into `data/meta/player_bio_{season}.parquet`. The fetch is incremental, so
`ingest build` remains network-free after the first run.

### Reproducing the Muñoz result

The article's point is that Muñoz's four-seamer sends contradictory signals: a
flat approach angle (suggesting success) but dead-zone movement (suggesting the
opposite), while he actually generates grounders. Velocity-adjusting the delta
resolves it — hitters expect more ride than 99 mph delivers.

| | article (2025) | this store (2026) |
|---|---|---|
| VAA above average | +0.59 | **+0.48** |
| DDZ delta | +0.2 | **+1.15** |
| velocity-adjusted | −0.3 | **−0.82** |

The qualitative finding reproduces exactly — the velocity adjustment flips the
sign from positive to negative. Magnitudes differ, which is expected: different
season, and Max Bay's exact specification (training population, filters, full
MVN vs. per-response fits) is not public.

## For the UMAP layer (not built yet)

Two transforms that must happen when building feature matrices. Both are
invisible failures — the UMAP will simply be subtly meaningless:

- **`spin_axis` is circular** (0–360°). Encode as `sin`/`cos`, never raw
  degrees, or 359° and 1° become maximally distant. Alcantara's 2026 arsenal
  shows why: his arm-side pitches sit at ~231–241° and his breaking balls at
  ~61–75°, straddling the wrap point.
- **Mirror left-handers.** Negate `release_pos_x` and `pfx_x` for LHP, or the
  dominant axis of any UMAP is just handedness rather than arsenal similarity.

Store raw values (as now); apply these in the feature layer.

## Layout

```
src/mlbpitch/
  paths.py      canonical on-disk locations
  schedule.py   statsapi -> {date: set(game_pk)} completeness oracle
  savant.py     day fetch, retry/backoff, truncation guard, raw cache
  schema.py     column selection + dtype policy
  store.py      parquet build, DuckDB query helpers, pitcher_dim
  ingest.py     CLI
data/
  raw/season=2026/date=YYYY-MM-DD.csv.gz
  pitches/season=2026/month=MM/part-0.parquet
  meta/{schedule,manifest,pitcher_dim}_2026.parquet
scripts/refresh.sbatch
```

Adding another season is `--season 2025 backfill`; the layout is already
partitioned for it.
