# mlbpitch — the pitch store

Statcast pitches in a verified local parquet store, queried with DuckDB.

## Modules

| | |
|---|---|
| `paths.py` | On-disk locations; data root is `<repo>/data` or `$MLBPITCH_DATA` |
| `schedule.py` | statsapi completeness oracle: `{date: set(game_pk)}` |
| `savant.py` | One-date fetch, retry/backoff, truncation guard, raw cache |
| `schema.py` | Column selection and dtype policy |
| `store.py` | Month-partition build, DuckDB query helpers, `pitcher_dim` |
| `physics.py` | Release and approach angles from the trajectory fit |
| `ddz.py` | Dynamic Dead Zone deltas |
| `people.py` | Pitcher height from statsapi (needed by `ddz`) |
| `profiles.py` | Pseudobulk pitcher × pitch-type profiles |
| `timeseries.py` | Continuous-time profiles and as-of snapshots |
| `ingest.py` | The `mlbpitch` CLI |

## CLI

```
mlbpitch [--season YYYY] [--level mlb|aaa] {backfill|refresh|verify|build|pitcher <id> [--live]}
```

`backfill` fetches every completed date not yet cached. `refresh` re-fetches
new dates plus a 10-day trailing window. `build` rebuilds parquet from the raw
cache with no network. `verify` reconciles against the schedule; run it after
every ingest. Nothing is scheduled.

## Layout

```
data/raw/season=YYYY/date=YYYY-MM-DD.csv.gz     raw cache, the rebuild source
data/pitches/season=YYYY/month=MM/part-0.parquet
data/meta/                                      schedule, manifest, pitcher_dim, bios, fitted baselines
data/aaa/{raw,pitches,meta}                     Triple-A, same layout
```

`store.pitcher(id)` returns the 34 `FEATURE_COLUMNS` by default; pass
`columns="*"` for all. DuckDB adds `season` and `month` as hive partition
columns, so the view has two more columns than the files on disk.
`store.connect()` caches its connection. `build_month` invalidates it; call
`store.reset_connection()` after writing parquet any other way.

## Design

**Savant, not statsapi, is the source.** The statsapi play-by-play feeds lack
`arm_angle`, `spin_axis` and `release_extension`, and cost ~2,456 requests
(10–15 GB) per season against Savant's ~184 (~500 MB). statsapi is used only as
the schedule oracle.

**One game-date per request.** Savant silently truncates at 25,000 rows. The
busiest date seen is ~5,000 pitches, so a one-day request cannot hit the cap.
`savant.fetch_day` still raises if a response reaches it.

**The raw gzipped CSV is kept.** Parquet is rebuilt from it (`build`), so a
schema change never needs a re-scrape.

**Month partitions**, rows sorted by `pitcher, game_date, at_bat_number,
pitch_number`, so row-group statistics skip most groups on a per-pitcher
filter. Per-pitcher files would mean 1,000+ small files on a shared filesystem.

**Refresh re-fetches a trailing window.** Statcast revises recent games, so an
append-only refresh would keep stale classifications. A month is rebuilt only
when a date's content hash changes.

## Gotchas

- `player_name` is `"Last, First"` and `des` is free text. Both contain commas,
  so never `str.split(",")` the CSV. `schema.read_raw_frame` uses a real reader.
- statsapi lists a suspended game under both its original and resumption dates;
  Savant files every pitch under the original. `schedule` attributes each
  `game_pk` to its earliest completed date.
- Savant lags statsapi. `verify` treats missing games within `--lag-days`
  (default 2) as a warning.
- The six `*_deprecated` columns are always null and are dropped at build.
- The Savant minors feed mixes levels; `--level aaa` keeps only game_pks on the
  Triple-A schedule.

## Pitch angles (`physics.py`)

Four stored columns from the Statcast trajectory fit, which is defined at
**y = 50 ft**, not at `release_pos_y` (~54 ft). `approach_angles` propagates
forward to the plate (y = 17/12 ft); `release_angles` propagates backward to each
pitch's own `release_pos_y`.

```
vy_f = -sqrt(vy0^2 - 2*ay*(y0 - yf));  t = (vy_f - vy0)/ay
vertical   = -degrees(atan((vz0 + az*t) / vy_f))
horizontal = -degrees(atan((vx0 + ax*t) / vy_f))
```

| | at release | at the plate |
|---|---|---|
| vertical | `vra` (4-seam −1.85°) | `vaa` (4-seam −4.65°, curveball −9.60°) |
| horizontal | `hra` | `haa` |

Gravity only steepens a pitch, so VRA > VAA always; `verify` asserts it.
Horizontal angles are negative when aimed toward a right-handed batter (RHB
stand at negative `plate_x`, confirmed from hit-by-pitch locations).
Right-handed pitchers therefore average positive HRA, since they release from
the third-base side and throw rightward.

**All four raw angles are location-confounded** (VAA vs `plate_z` r = 0.89, HAA
vs `release_pos_x` r = −0.95). `store.add_angles_adjusted(df)` adds
`{vaa,haa,vra,hra}_above_avg`: handedness-mirrored residuals against
per-pitch-type baselines (`data/meta/angle_baseline_{season}.parquet`, refit on
every build), which take each confounding correlation to 0.000. They are not
stored columns because the fit is season-wide while `build_month` sees one
month.

## Dynamic Dead Zone (`ddz.py`)

`ddz.add_ddz_delta(df)` adds `ddz_delta_{ax,az}`: observed acceleration minus
what the pitcher's arm angle and height-scaled extension predict, in ft/s².
Positive `az` is more ride than expected. The `_velo` variants also condition
on `release_speed`. The model is Max Bay's, from the FanGraphs piece on Andrés
Muñoz. Its 4-D normal conditional mean is the least-squares fit, so
coefficients are stored in linear form in `data/meta/ddz_baseline_{season}.parquet`.
Height comes from statsapi via `people.py`, fetched incrementally so `build`
stays offline after the first run.

The article's Muñoz result reproduces in sign. His four-seamer reads flat by
VAA (+0.48 above average) and in the dead zone by movement (+1.15), and the
velocity adjustment flips it (−0.82).

## Feature-layer transforms

Raw values are stored; apply these when building a feature matrix. Both fail
silently.

- `spin_axis` is circular. Encode it as sin/cos.
- Mirror left-handers: negate `release_pos_x` and `pfx_x`.
