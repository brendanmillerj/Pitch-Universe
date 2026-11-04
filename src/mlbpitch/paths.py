"""Canonical on-disk locations for the pitch store.

The data root resolves in this order:

1. ``$MLBPITCH_DATA`` if set -- the escape hatch for pointing at a store that
   does not live beside the source (a scratch filesystem, a shared group copy,
   or any non-editable install).
2. ``<repo>/data`` inferred from this file's location. Correct for a source
   checkout or ``pip install -e .``; for a *non*-editable install this file
   lives in site-packages, where that inference is meaningless -- hence the
   environment variable above.

Levels. MLB is the primary store and lives directly under the data root
(``data/raw``, ``data/pitches``, ``data/meta``), exactly as before levels
existed. Every other level gets its own sibling root (``data/aaa/raw`` ...)
with the same internal layout, so the MLB files never move and a level is one
extra directory rather than a new partition depth inside the parquet glob.

The angle and DDZ baselines are deliberately *not* per level: they are fitted
on MLB only and applied to every level, so an "above average" number for a
Triple-A pitcher reads as "against big-league norms". Hence those paths take
no level argument.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ["MLBPITCH_DATA"]) if os.environ.get("MLBPITCH_DATA") else ROOT / "data"

RAW = DATA / "raw"            # immutable gzipped Savant CSV, one per game-date
PITCHES = DATA / "pitches"    # parquet, partitioned season=/month=
META = DATA / "meta"          # schedule oracle, ingest manifest, pitcher dim

SEASON = 2026

# statsapi sportId per level. Savant's minors export (`minors=true`) carries
# every Hawk-Eye-equipped minor-league park in one undifferentiated feed, so
# the level of a pitch is established by joining game_pk to this schedule.
LEVEL = "mlb"
SPORT_ID = {"mlb": 1, "aaa": 11}
LEVELS = tuple(SPORT_ID)


def _check(level: str) -> str:
    if level not in SPORT_ID:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    return level


def level_root(level: str = LEVEL) -> Path:
    return DATA if _check(level) == "mlb" else DATA / level


def raw_dir(level: str = LEVEL) -> Path:
    return level_root(level) / "raw"


def pitches_dir(level: str = LEVEL) -> Path:
    return level_root(level) / "pitches"


def meta_dir(level: str = LEVEL) -> Path:
    return level_root(level) / "meta"


def raw_path(date: str, season: int = SEASON, level: str = LEVEL) -> Path:
    return raw_dir(level) / f"season={season}" / f"date={date}.csv.gz"


def month_path(month: str, season: int = SEASON, level: str = LEVEL) -> Path:
    return pitches_dir(level) / f"season={season}" / f"month={month}" / "part-0.parquet"


def schedule_path(season: int = SEASON, level: str = LEVEL) -> Path:
    return meta_dir(level) / f"schedule_{season}.parquet"


def manifest_path(season: int = SEASON, level: str = LEVEL) -> Path:
    return meta_dir(level) / f"manifest_{season}.parquet"


def pitcher_dim_path(season: int = SEASON, level: str = LEVEL) -> Path:
    return meta_dir(level) / f"pitcher_dim_{season}.parquet"


def player_bio_path(season: int = SEASON, level: str = LEVEL) -> Path:
    return meta_dir(level) / f"player_bio_{season}.parquet"


# MLB-only by design (see module docstring).
def angle_baseline_path(season: int = SEASON) -> Path:
    return META / f"angle_baseline_{season}.parquet"


def ddz_baseline_path(season: int = SEASON) -> Path:
    return META / f"ddz_baseline_{season}.parquet"


def ensure_dirs(level: str = LEVEL) -> None:
    for d in (raw_dir(level), pitches_dir(level), meta_dir(level)):
        d.mkdir(parents=True, exist_ok=True)
