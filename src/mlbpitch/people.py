"""Pitcher biographical data from statsapi -- currently just listed height.

Height is needed by the Dynamic Dead Zone model, which conditions on extension
*scaled by pitcher height*: a 6'8" pitcher releasing 7 ft down the mound is doing
something mechanically ordinary, while a 5'10" pitcher doing the same is not.
Raw extension conflates the two.

Statcast carries no height, so it comes from statsapi's people endpoint, which
accepts up to a few hundred ids per request (758 pitchers = 4 requests).
"""
from __future__ import annotations

import re

import requests

from . import paths

PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
BATCH = 200

_HEIGHT_RE = re.compile(r"(\d+)'\s*(\d+)")


def parse_height_ft(text: str | None) -> float | None:
    """statsapi reports height as e.g. `6' 5\"` -> 6.417 ft."""
    if not text:
        return None
    m = _HEIGHT_RE.match(text)
    return int(m.group(1)) + int(m.group(2)) / 12 if m else None


def fetch_bio(pitcher_ids):
    """One row per pitcher: id, name, height (ft), throwing hand."""
    import pandas as pd

    ids = [int(i) for i in pitcher_ids]
    people = []
    with requests.Session() as s:
        for i in range(0, len(ids), BATCH):
            chunk = ",".join(str(x) for x in ids[i : i + BATCH])
            r = s.get(PEOPLE_URL, params={"personIds": chunk}, timeout=90)
            r.raise_for_status()
            people += r.json().get("people", [])

    rows = [
        {
            "pitcher_id": p["id"],
            "full_name": p.get("fullName"),
            "height_ft": parse_height_ft(p.get("height")),
            "throws": p.get("pitchHand", {}).get("code"),
        }
        for p in people
    ]
    return pd.DataFrame(rows).drop_duplicates("pitcher_id").reset_index(drop=True)


def build_bio(season: int = paths.SEASON, level: str = paths.LEVEL):
    """Persist bio for every pitcher in pitcher_dim, fetching only what's new.

    Incremental so that `ingest build` stays a network-free operation after the
    first run -- only genuinely new pitchers trigger a request. Height does not
    change, so cached rows never need refreshing. A minor-league level first
    reuses rows already held in another level's bio for the same season
    (call-ups appear in both) before asking statsapi for the rest.
    """
    import pandas as pd

    dim = pd.read_parquet(paths.pitcher_dim_path(season, level))
    want = {int(i) for i in dim["pitcher_id"]}

    path = paths.player_bio_path(season, level)
    have = pd.read_parquet(path) if path.exists() else None
    known = set(have["pitcher_id"].astype(int)) if have is not None else set()

    missing = sorted(want - known)
    if missing:
        borrowed = []
        for other in paths.LEVELS:
            other_path = paths.player_bio_path(season, other)
            if other == level or not other_path.exists():
                continue
            ob = pd.read_parquet(other_path)
            borrowed.append(ob[ob["pitcher_id"].astype(int).isin(missing)])
        if borrowed:
            borrowed = pd.concat(borrowed, ignore_index=True).drop_duplicates("pitcher_id")
            have = borrowed if have is None else pd.concat([have, borrowed], ignore_index=True)
            missing = sorted(set(missing) - set(borrowed["pitcher_id"].astype(int)))
        if missing:
            fresh = fetch_bio(missing)
            have = fresh if have is None else pd.concat([have, fresh], ignore_index=True)
        have = have.drop_duplicates("pitcher_id").reset_index(drop=True)
        paths.ensure_dirs(level)
        have.to_parquet(path, index=False)

    return have


def load_bio(season: int = paths.SEASON, level: str = paths.LEVEL):
    import pandas as pd

    path = paths.player_bio_path(season, level)
    if not path.exists():
        raise FileNotFoundError(
            f"no player bio for {season}; run `ingest build` or "
            "`people.build_bio()` first"
        )
    return pd.read_parquet(path)
