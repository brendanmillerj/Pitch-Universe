"""Parquet store: build month partitions from the raw cache, then query them.

Layout is month-partitioned (7 files per season) rather than per-day or
per-pitcher. The filesystem here is VAST, where metadata load from many small
files is the thing that hurts; per-pitcher files would mean 1,000+ of them for
no query benefit that row-group statistics don't already provide.

Within each month file rows are sorted by pitcher, so the parquet row-group
min/max statistics let DuckDB skip most groups when filtering to one pitcher --
the dominant access pattern for the dashboard.
"""
from __future__ import annotations

import io
import os

from . import paths, physics, savant, schedule
from .schema import apply_dtypes, read_raw_frame

ROW_GROUP_SIZE = 32_768
COMPRESSION = "zstd"

SORT_KEY = ["pitcher", "game_date", "at_bat_number", "pitch_number"]


def month_of(date: str) -> str:
    return date[5:7]


def build_month(
    month: str, dates: list[str], season: int = paths.SEASON, level: str = paths.LEVEL
) -> dict:
    """Rebuild one month partition from the raw cache.

    Always a full rebuild of the month rather than an append: Statcast revises
    recent games (pitch types get reclassified, measurements corrected), so a
    re-fetched date must *replace* its old rows, not sit alongside them.

    For a minor-league level the raw feed carries every tracked level at once,
    so rows are kept only when their game_pk is on that level's schedule.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    frames = []
    for date in sorted(dates):
        raw = paths.raw_path(date, season, level)
        if not raw.exists():
            continue
        frames.append(read_raw_frame(raw))

    if not frames:
        return {"month": month, "rows": 0, "written": False}

    df = pd.concat(frames, ignore_index=True)
    if level != "mlb":
        keep = schedule.expected_game_pks(season, level)
        df = df[df["game_pk"].isin(keep)].reset_index(drop=True)
        if df.empty:
            return {"month": month, "rows": 0, "written": False}
    # Derived geometry before typing, so vaa/haa land in FLOAT32_COLUMNS with
    # every other measurement rather than staying float64. Concatenated in one
    # go -- assigning columns individually fragments a 113-column frame.
    vaa, haa = physics.approach_angles(df)
    vra, hra = physics.release_angles(df)
    df = pd.concat(
        [df, pd.DataFrame(
            {"vaa": vaa, "haa": haa, "vra": vra, "hra": hra}, index=df.index)],
        axis=1,
    )
    df = apply_dtypes(df)
    df = df.sort_values(SORT_KEY, kind="mergesort").reset_index(drop=True)

    dest = paths.month_path(month, season, level)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f".tmp{os.getpid()}.parquet")

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        tmp,
        compression=COMPRESSION,
        row_group_size=ROW_GROUP_SIZE,
        use_dictionary=True,
    )
    os.replace(tmp, dest)
    reset_connection(season, level)  # the parquet just changed under any open view

    return {
        "month": month,
        "rows": len(df),
        "dates": len(frames),
        "bytes": dest.stat().st_size,
        "written": True,
    }


def build_months(
    dates: list[str], season: int = paths.SEASON, level: str = paths.LEVEL
) -> list[dict]:
    """Rebuild every month touched by `dates`, using all cached dates in it."""
    from collections import defaultdict

    cached = cached_dates(season, level)
    touched = {month_of(d) for d in dates}

    by_month = defaultdict(list)
    for d in cached:
        by_month[month_of(d)].append(d)

    return [build_month(m, by_month[m], season, level) for m in sorted(touched)]


def cached_dates(season: int = paths.SEASON, level: str = paths.LEVEL) -> list[str]:
    """Dates present in the raw cache."""
    root = paths.raw_dir(level) / f"season={season}"
    if not root.exists():
        return []
    return sorted(p.name[len("date=") : -len(".csv.gz")] for p in root.glob("date=*.csv.gz"))


# --------------------------------------------------------------------------
# Query helpers
# --------------------------------------------------------------------------

def pitches_glob(season: int = paths.SEASON, level: str = paths.LEVEL) -> str:
    return str(paths.pitches_dir(level) / f"season={season}" / "month=*" / "*.parquet")


# The columns a pitch-level view actually needs: identity, context, and the
# biophysical features. Selecting these instead of all 113 columns is the
# single biggest query win -- parquet is columnar, so the unread columns cost
# nothing to skip (measured 51ms -> 13ms on a per-pitcher lookup).
FEATURE_COLUMNS = [
    "game_date", "game_pk", "pitcher", "player_name", "p_throws",
    "batter", "stand", "pitch_type", "pitch_name",
    "release_speed", "release_pos_x", "release_pos_z", "release_extension",
    "pfx_x", "pfx_z", "spin_axis", "release_spin_rate", "arm_angle",
    "vaa", "haa", "vra", "hra", "release_pos_y",
    "ax", "az",  # raw acceleration terms; DDZ deltas are computed from these
    "plate_x", "plate_z", "zone", "description", "events",
    "balls", "strikes", "at_bat_number", "pitch_number",
]

_CONNECTIONS: dict[tuple[int, str], "duckdb.DuckDBPyConnection"] = {}


def connect(season: int = paths.SEASON, fresh: bool = False, level: str = paths.LEVEL):
    """Cached DuckDB connection with a `pitches` view over one season+level.

    Reused across calls -- opening a connection and creating the view costs
    ~11 ms, which dominates an otherwise ~13 ms query. Call `reset_connection`
    (or pass fresh=True) after rewriting parquet files, since a live view can
    otherwise hold a stale file listing.
    """
    import duckdb

    key = (season, level)
    if fresh or key not in _CONNECTIONS:
        con = duckdb.connect()
        con.execute(
            "CREATE VIEW pitches AS SELECT * FROM "
            f"read_parquet('{pitches_glob(season, level)}')"
        )
        _CONNECTIONS[key] = con
    return _CONNECTIONS[key]


def reset_connection(season: int | None = None, level: str | None = None) -> None:
    """Drop cached connections so the next query re-reads the parquet layout.

    With no arguments every connection is dropped; with a season (and
    optionally a level) only the matching ones.
    """
    keys = [
        k for k in _CONNECTIONS
        if (season is None or k[0] == season) and (level is None or k[1] == level)
    ]
    for k in keys:
        con = _CONNECTIONS.pop(k, None)
        if con is not None:
            con.close()


def _select(columns) -> str:
    if columns is None or columns == "*":
        return "*"
    return ", ".join(columns)


def query(sql: str, season: int = paths.SEASON, level: str = paths.LEVEL):
    return connect(season, level=level).execute(sql).df()


def pitcher(
    pitcher_id: int, season: int = paths.SEASON, columns=FEATURE_COLUMNS,
    level: str = paths.LEVEL,
):
    """Every pitch thrown by one pitcher -- the core dashboard lookup.

    Defaults to FEATURE_COLUMNS; pass columns="*" for all 113.
    """
    return connect(season, level=level).execute(
        f"SELECT {_select(columns)} FROM pitches WHERE pitcher = ? "
        "ORDER BY game_date, at_bat_number, pitch_number",
        [pitcher_id],
    ).df()


# --------------------------------------------------------------------------
# Angle baselines (approach + release)
# --------------------------------------------------------------------------
#
# All four angles are strongly confounded by location -- measured on this
# store: VAA vs plate_z +0.89, HAA vs release_pos_x -0.95 (sweepers), VRA vs
# plate_z +0.79, HRA vs release_pos_x -0.73. Used raw as model features they
# would largely encode *where the pitch was thrown* rather than what it does.
# The fix (per the FanGraphs VAA/HAA primers) is the residual against a
# per-pitch-type location baseline.
#
# These are deliberately NOT parquet columns: the fit is season-wide, while
# build_month only ever sees one month, and the baseline legitimately shifts as
# the season accumulates. Storing coefficients and applying them at read time
# keeps "above average" honest instead of freezing a mid-season snapshot.

# Vertical angles regress on pitch height, horizontal angles on horizontal
# location plus release point. Linear in plate_z is sufficient -- quadratic and
# cubic terms added <=0.002 R^2 across every pitch type. The horizontal fits
# need release_pos_x as well as plate_x: plate_x alone gives R^2 ~0.45, adding
# release point reaches ~0.90.
ANGLE_PREDICTORS = {
    "vaa": ["plate_z"],
    "vra": ["plate_z"],
    "haa": ["plate_x", "release_pos_x"],
    "hra": ["plate_x", "release_pos_x"],
}

# Handedness is mirrored rather than modelled separately: LHP and RHP produce
# near mirror-image horizontal angles (HAA 4-seam: -1.38 vs +1.19 deg), so
# negating these for lefties lets one fit serve both hands at double the sample.
MIRROR_COLUMNS = ["haa", "hra", "plate_x", "release_pos_x"]

MIN_PITCHES_FOR_BASELINE = 200


def _mirror_lhp(df):
    """Negate x-axis quantities for left-handers -> handedness-neutral frame."""
    out = df.copy()
    left = (out["p_throws"] == "L").to_numpy()
    for c in MIRROR_COLUMNS:
        if c in out.columns:
            out.loc[left, c] = -out.loc[left, c]
    return out


def _ols(X, y):
    """Least-squares fit with an intercept prepended. Returns coef vector."""
    import numpy as np

    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _baseline_inputs(angle: str) -> list[str]:
    return [angle] + ANGLE_PREDICTORS[angle] + ["plate_x", "plate_z", "release_pos_x"]


def fit_angle_baseline(season: int = paths.SEASON):
    """Fit per-pitch-type location baselines for all four angles; persist coefs.

    MLB only, by design: the baseline is the big-league norm that every level
    is measured against, so it is never refitted on minor-league pitches.
    """
    import pandas as pd

    angles = list(ANGLE_PREDICTORS)
    needed = sorted({c for a in angles for c in _baseline_inputs(a)} | {"p_throws"})
    df = query(
        f"SELECT pitch_type, {', '.join(needed)} FROM pitches "
        "WHERE vaa IS NOT NULL AND pitch_type IS NOT NULL",
        season,
    )
    df = _mirror_lhp(df)

    rows = []
    for pitch_type, g in df.groupby("pitch_type", observed=True):
        if len(g) < MIN_PITCHES_FOR_BASELINE:
            continue
        row = {"pitch_type": pitch_type, "n": len(g)}
        for angle in angles:
            preds = ANGLE_PREDICTORS[angle]
            s = g.dropna(subset=[angle] + preds)
            if len(s) < MIN_PITCHES_FOR_BASELINE:
                continue
            X, y = s[preds].to_numpy(), s[angle].to_numpy()
            coef = _ols(X, y)
            resid = y - (coef[0] + X @ coef[1:])
            row[f"{angle}_intercept"] = coef[0]
            for name, c in zip(preds, coef[1:]):
                row[f"{angle}_{name}"] = c
            row[f"{angle}_resid_sd"] = float(resid.std())
            row[f"{angle}_n"] = len(s)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    paths.ensure_dirs()
    out.to_parquet(paths.angle_baseline_path(season), index=False)
    return out


def load_angle_baseline(season: int = paths.SEASON):
    import pandas as pd

    path = paths.angle_baseline_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"no angle baseline for {season}; run `ingest build` or "
            "`store.fit_angle_baseline()` first"
        )
    return pd.read_parquet(path)


def add_angles_adjusted(df, season: int = paths.SEASON, angles=None):
    """Append handedness-neutral, location-adjusted `<angle>_above_avg` columns.

    Positive `vaa_above_avg` = flatter than a league-average pitch of that type
    thrown to the same height. Positive `haa_above_avg` = more horizontal
    approach than expected given location and release point. Same reading for
    the release-side `vra_above_avg` / `hra_above_avg`.

    Requires pitch_type, p_throws, the angle itself, and its predictors. The
    baseline is always the MLB one, whichever level `df` came from.
    """
    base = load_angle_baseline(season).set_index("pitch_type")
    m = _mirror_lhp(df)
    coef = m[["pitch_type"]].join(base, on="pitch_type")
    out = df.copy()
    for angle in (angles or list(ANGLE_PREDICTORS)):
        preds = ANGLE_PREDICTORS[angle]
        if angle not in m.columns or f"{angle}_intercept" not in coef.columns:
            continue
        expected = coef[f"{angle}_intercept"].copy()
        for name in preds:
            expected = expected + coef[f"{angle}_{name}"] * m[name]
        out[f"{angle}_above_avg"] = m[angle] - expected
    return out


def build_pitcher_dim(season: int = paths.SEASON, level: str = paths.LEVEL):
    """Small per-pitcher summary so dashboard dropdowns never scan the big table."""
    df = connect(season, level=level).execute(
        """
        SELECT pitcher            AS pitcher_id,
               any_value(player_name) AS player_name,
               any_value(p_throws)    AS p_throws,
               count(*)               AS pitches,
               count(DISTINCT game_pk) AS games,
               count(DISTINCT pitch_type) AS n_pitch_types,
               min(game_date)         AS first_game,
               max(game_date)         AS last_game,
               avg(release_speed)     AS avg_velo,
               avg(arm_angle)         AS avg_arm_angle
        FROM pitches
        GROUP BY pitcher
        ORDER BY pitches DESC
        """
    ).df()

    paths.ensure_dirs(level)
    df.to_parquet(paths.pitcher_dim_path(season, level), index=False)
    return df
