"""Canonical on-disk locations for the pitch store.

The data root resolves in this order:

1. ``$MLBPITCH_DATA`` if set -- the escape hatch for pointing at a store that
   does not live beside the source (a scratch filesystem, a shared group copy,
   or any non-editable install).
2. ``<repo>/data`` inferred from this file's location. Correct for a source
   checkout or ``pip install -e .``; for a *non*-editable install this file
   lives in site-packages, where that inference is meaningless -- hence the
   environment variable above.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ["MLBPITCH_DATA"]) if os.environ.get("MLBPITCH_DATA") else ROOT / "data"

RAW = DATA / "raw"            # immutable gzipped Savant CSV, one per game-date
PITCHES = DATA / "pitches"    # parquet, partitioned season=/month=
META = DATA / "meta"          # schedule oracle, ingest manifest, pitcher dim

SEASON = 2026


def raw_path(date: str, season: int = SEASON) -> Path:
    return RAW / f"season={season}" / f"date={date}.csv.gz"


def month_path(month: str, season: int = SEASON) -> Path:
    return PITCHES / f"season={season}" / f"month={month}" / "part-0.parquet"


def schedule_path(season: int = SEASON) -> Path:
    return META / f"schedule_{season}.parquet"


def manifest_path(season: int = SEASON) -> Path:
    return META / f"manifest_{season}.parquet"


def pitcher_dim_path(season: int = SEASON) -> Path:
    return META / f"pitcher_dim_{season}.parquet"


def angle_baseline_path(season: int = SEASON) -> Path:
    return META / f"angle_baseline_{season}.parquet"


def player_bio_path(season: int = SEASON) -> Path:
    return META / f"player_bio_{season}.parquet"


def ddz_baseline_path(season: int = SEASON) -> Path:
    return META / f"ddz_baseline_{season}.parquet"


def ensure_dirs() -> None:
    for d in (RAW, PITCHES, META):
        d.mkdir(parents=True, exist_ok=True)
