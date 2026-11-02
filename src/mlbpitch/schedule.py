"""The completeness oracle: which games MLB says were played on each date.

Everything downstream is checked against this. A Savant day-fetch that returns
plausible-looking rows is only trustworthy once its game_pk set matches what
the schedule says was actually played, so this module is what turns "the file
is non-empty" into "the file is complete".

One request covers a whole season (~3 MB).
"""
from __future__ import annotations

import requests

from . import paths

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# Games that produced a full pitch record. Anything not in this set (postponed,
# suspended, cancelled) should not be expected to appear in the Savant export.
COMPLETE_STATES = {"Final", "Completed Early"}


def fetch_schedule(season: int = paths.SEASON, game_type: str = "R") -> "pd.DataFrame":
    """One row per scheduled game, with the state MLB reports for it."""
    import pandas as pd

    resp = requests.get(
        SCHEDULE_URL,
        params={
            "sportId": 1,
            "startDate": f"{season}-01-01",
            "endDate": f"{season}-12-31",
            "gameType": game_type,
        },
        timeout=90,
    )
    resp.raise_for_status()

    rows = []
    for date_entry in resp.json().get("dates", []):
        for game in date_entry.get("games", []):
            rows.append(
                {
                    "game_date": date_entry["date"],
                    "game_pk": int(game["gamePk"]),
                    "state": game["status"]["detailedState"],
                    "game_type": game.get("gameType", game_type),
                }
            )

    df = pd.DataFrame(rows)
    df["is_complete"] = df["state"].isin(COMPLETE_STATES)
    return df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def save_schedule(season: int = paths.SEASON, game_type: str = "R") -> "pd.DataFrame":
    paths.ensure_dirs()
    df = fetch_schedule(season, game_type)
    df.to_parquet(paths.schedule_path(season), index=False)
    return df


def load_schedule(season: int = paths.SEASON, refresh: bool = False) -> "pd.DataFrame":
    import pandas as pd

    path = paths.schedule_path(season)
    if refresh or not path.exists():
        return save_schedule(season)
    return pd.read_parquet(path)


def expected_games_by_date(season: int = paths.SEASON, refresh: bool = False) -> dict:
    """{game_date -> set(game_pk)} for games that actually finished.

    A suspended game is listed by statsapi under *both* the date it started
    and the date it resumed (carrying `resumedFrom`), but Savant files all of
    its pitches under the original date. Attributing each game_pk to its
    earliest completed date keeps the oracle consistent with how the pitch
    data is actually published -- otherwise the resumption date looks
    permanently short by one game.
    """
    df = load_schedule(season, refresh=refresh)
    done = df[df["is_complete"]].sort_values(["game_pk", "game_date"])
    done = done.drop_duplicates(subset="game_pk", keep="first")
    return {d: set(g["game_pk"]) for d, g in done.groupby("game_date")}


def playable_dates(season: int = paths.SEASON, refresh: bool = False) -> list[str]:
    """Dates with at least one completed game, ascending."""
    return sorted(expected_games_by_date(season, refresh=refresh))
