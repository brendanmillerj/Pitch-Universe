"""Column selection and dtypes for the Savant statcast_search/csv export.

Parsed with a real CSV reader only -- `player_name` is "Last, First" and `des`
contains free-text commas, so naive str.split(",") misaligns every column
after the sixth. pandas.read_csv (C parser) handles the quoting correctly.

The six `*_deprecated` columns are dropped: Savant has kept them in the export
for backward compatibility but they have been all-null since the Statcast
pitch-tracking system replaced TrackMan-era fields years ago. Verified null at
build time in the ingest CLI (`ingest.py verify`) rather than assumed here.
"""
from __future__ import annotations

DEPRECATED_COLUMNS = [
    "spin_dir",
    "spin_rate_deprecated",
    "break_angle_deprecated",
    "break_length_deprecated",
    "tfs_deprecated",
    "tfs_zulu_deprecated",
]

# dtype policy: float32 for continuous measurements (single-precision is far
# more than Statcast's actual measurement resolution), int32 for ids/counts,
# category for anything reused across millions of rows (dictionary-encoded in
# parquet automatically once dtype is "category").
FLOAT32_COLUMNS = [
    "release_speed", "release_pos_x", "release_pos_y", "release_pos_z",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot",
    "hit_distance_sc", "launch_speed", "launch_angle", "effective_speed",
    "release_spin_rate", "release_extension", "vx0", "vy0", "vz0",
    "ax", "ay", "az", "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "babip_value", "iso_value", "delta_home_win_exp", "delta_run_exp",
    "bat_speed", "swing_length", "estimated_slg_using_speedangle",
    "delta_pitcher_run_exp", "hyper_speed", "home_win_exp", "bat_win_exp",
    "api_break_z_with_gravity", "api_break_x_arm", "api_break_x_batter_in",
    "arm_angle", "attack_angle", "attack_direction", "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches", "spin_axis",
    # derived in physics (approach + release angles), attached during build_month
    "vaa", "haa", "vra", "hra",
]

INT32_COLUMNS = [
    "batter", "pitcher", "outs_when_up", "inning", "balls", "strikes",
    "game_year", "hit_location", "at_bat_number", "pitch_number",
    "home_score", "away_score", "bat_score", "fld_score",
    "post_away_score", "post_home_score", "post_bat_score", "post_fld_score",
    "game_pk", "n_thruorder_pitcher", "n_priorpa_thisgame_player_at_bat",
]

CATEGORY_COLUMNS = [
    "pitch_type", "pitch_name", "player_name", "events", "description",
    "zone", "game_type", "stand", "p_throws", "home_team", "away_team",
    "type", "bb_type", "inning_topbot", "if_fielding_alignment",
    "of_fielding_alignment",
]

# game_date stays as a date, sv_id/des kept as-is (free text / low value to
# type further). Fielder_2..9 and on_1b/2b/3b are nullable ids -> Int32.
NULLABLE_INT_COLUMNS = [
    "on_1b", "on_2b", "on_3b",
    "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    "launch_speed_angle",
]

DATE_COLUMNS = ["game_date"]

MAX_ROWS_PER_DAY = 25_000  # Savant's hard cap; see savant.py truncation guard


def read_raw_frame(path_or_buf):
    """Parse a Savant CSV export with no dtype coercion applied yet.

    Kept separate from `apply_dtypes` so that a month build can concatenate
    many days *before* typing: categorical columns built per-day carry
    per-day category sets, and concatenating those silently degrades them
    back to object. Type once, after the concat.
    """
    import pandas as pd

    return pd.read_csv(path_or_buf, low_memory=False, parse_dates=DATE_COLUMNS)


def apply_dtypes(df):
    """Drop deprecated columns and coerce to the storage dtype policy."""
    df = df.drop(columns=[c for c in DEPRECATED_COLUMNS if c in df.columns])

    for c in FLOAT32_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    for c in INT32_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("Int32")
    for c in NULLABLE_INT_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("Int32")
    for c in CATEGORY_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df


def read_savant_csv(path_or_buf):
    """Read a single Savant CSV export, fully typed."""
    return apply_dtypes(read_raw_frame(path_or_buf))
