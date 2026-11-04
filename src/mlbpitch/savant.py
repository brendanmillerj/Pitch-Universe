"""Baseball Savant pitch-level fetching, one game-date per request.

Why per-date and not per-week: the CSV endpoint silently truncates at 25,000
rows. A three-week request returns exactly 25,000 rows covering only the first
seven dates, with no error and no indication anything is missing. The busiest
observed single date (a 15-game slate) is ~4,400 pitches, so a one-day chunk
has ~5.7x headroom and truncation becomes structurally impossible rather than
merely unlikely. The guard below is therefore a tripwire for the API contract
changing, not a routine condition.
"""
from __future__ import annotations

import gzip
import io
import os
import time

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import paths
from .schema import MAX_ROWS_PER_DAY

CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# Savant serves the browser search page; a default python UA gets less reliable
# treatment. Identify as a normal client and accept gzip on the wire.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept-Encoding": "gzip, deflate",
}


class TruncatedResponse(RuntimeError):
    """Savant hit its row cap -- the response is silently incomplete."""


class TransientHTTPError(RuntimeError):
    """Retryable server-side condition (429/5xx)."""


def day_params(
    date: str, season: int = paths.SEASON, game_type: str = "R", level: str = paths.LEVEL
) -> dict:
    """The exact parameter set verified against the live endpoint.

    `minors=true` switches the same endpoint to the minor-league feed, which
    returns the identical column set (verified 2026-07-22: 119 columns, every
    tracking field populated at the MLB rate). The feed is not split by level
    -- Triple-A plus whichever lower-level parks have Hawk-Eye -- so the level
    is applied afterwards by game_pk (see store.build_month).
    """
    minors = {"minors": "true"} if level != "mlb" else {}
    return {
        **minors,
        "all": "true",
        "hfGT": f"{game_type}|",
        "hfSea": f"{season}|",
        "player_type": "pitcher",
        "min_pitches": 0,
        "min_results": 0,
        "min_pas": 0,
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc",
        "type": "details",
        "game_date_gt": date,
        "game_date_lt": date,
    }


@retry(
    retry=retry_if_exception_type(
        (TransientHTTPError, requests.exceptions.RequestException)
    ),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _get(session: requests.Session, params: dict) -> bytes:
    resp = session.get(CSV_URL, params=params, headers=HEADERS, timeout=300)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientHTTPError(f"HTTP {resp.status_code}")
    resp.raise_for_status()
    return resp.content


def fetch_day(
    date: str,
    session: requests.Session | None = None,
    season: int = paths.SEASON,
    game_type: str = "R",
    level: str = paths.LEVEL,
) -> bytes:
    """Raw CSV bytes for one game-date. Raises if Savant truncated the result."""
    own = session is None
    session = session or requests.Session()
    try:
        content = _get(session, day_params(date, season, game_type, level))
    finally:
        if own:
            session.close()

    n_rows = content.count(b"\n") - 1  # minus header
    if n_rows >= MAX_ROWS_PER_DAY:
        raise TruncatedResponse(
            f"{date}: {n_rows} rows >= Savant's {MAX_ROWS_PER_DAY} cap. "
            "A single date should never approach this; the export contract "
            "has changed and this date must be split into sub-ranges."
        )
    return content


def write_raw(
    date: str, content: bytes, season: int = paths.SEASON, level: str = paths.LEVEL
) -> "Path":
    """Persist the raw CSV gzipped, atomically (temp file + rename).

    The raw cache is what makes the parquet store rebuildable -- dtype or
    schema changes never require re-scraping Savant. For a minor-league level
    the file is the *whole* minors response for that date (every level Savant
    tracks), so a lower level can later be built from the same cache.
    """
    dest = paths.raw_path(date, season, level)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(content)
    os.replace(tmp, dest)
    return dest


def read_raw(date: str, season: int = paths.SEASON, level: str = paths.LEVEL) -> bytes:
    with gzip.open(paths.raw_path(date, season, level), "rb") as fh:
        return fh.read()


def fetch_pitcher_season(
    pitcher_id: int, season: int = paths.SEASON, game_type: str = "R",
    level: str = paths.LEVEL,
) -> "pd.DataFrame":
    """Ad-hoc escape hatch: one pitcher's full season in a single request.

    Not used by the ingest path -- this exists for spot-checking the local
    store against live Savant, and for one-off lookups of a pitcher whose
    date range is not yet backfilled.
    """
    from .schema import read_savant_csv

    params = day_params("", season, game_type, level)
    params.pop("game_date_gt")
    params.pop("game_date_lt")
    params["pitchers_lookup[]"] = pitcher_id

    with requests.Session() as session:
        content = _get(session, params)
    return read_savant_csv(io.BytesIO(content))
