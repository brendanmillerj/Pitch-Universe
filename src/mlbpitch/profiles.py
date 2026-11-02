"""Pseudobulk pitch profiles -- the "sample x cell type" level of the analogy.

Two tables, both keyed by half-season so a pitcher can be followed across six
timepoints (1st/2nd half of each season):

  * type profiles     -- one row per pitcher x half x pitch_type
  * pitcher profiles  -- one row per pitcher x half, the type rows pivoted wide
                         plus arsenal-level deception features

Every lateral quantity is in a handedness-neutral frame: lefties are mirrored
in pfx_x, ax, release_pos_x AND spin_axis (360 - theta). Mirroring only some of
them is the bug that tore LHP into their own islands in the first pitch atlas.

Outcomes (run value, whiff, CSW, xwOBA) are carried in OUTCOME_COLUMNS so they
can label an embedding -- they must never be fed into one, or "similar to
successful pitchers" becomes circular.

Derived features use each season's own baselines (`add_angles_adjusted`,
`add_ddz_delta`), so they are measured against that season's league -- which
also absorbs league-wide tracking recalibration between seasons.
"""
from __future__ import annotations

from . import ddz, paths, store

SEASONS = (2024, 2025, 2026)
MIN_PITCHES = 40  # per pitcher x half x type; a half of relief work is small

# Grip variants folded into their family; anything else rare is dropped.
TYPE_MAP = {"KC": "CU", "FO": "FS"}
PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]
FASTBALLS = ["FF", "SI", "FC"]  # primary-fastball candidates, in priority order

COLUMNS = [
    "game_date", "pitcher", "player_name", "p_throws", "stand", "pitch_type",
    "release_speed", "release_spin_rate", "spin_axis", "pfx_x", "pfx_z", "ax", "az",
    "arm_angle", "release_pos_x", "release_pos_z", "release_extension",
    "vaa", "haa", "vra", "hra", "plate_x", "plate_z", "zone",
    "balls", "strikes", "description", "delta_pitcher_run_exp",
    "estimated_woba_using_speedangle",
]

WHIFFS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
SWINGS = WHIFFS | {"foul", "hit_into_play"}

# Per-type means, shrunk toward the league type mean (see _shrink).
MEAN_FEATURES = {
    "shape": ["release_speed", "release_spin_rate", "spin_sin", "spin_cos",
              "pfx_x", "pfx_z", "ax", "az"],
    "derived": ["vaa_above_avg", "haa_above_avg", "vra_above_avg", "hra_above_avg",
                "ddz_delta_ax", "ddz_delta_az", "ddz_delta_az_velo", "ssw_dev"],
    "delivery": ["arm_angle", "release_pos_x", "release_pos_z", "release_extension",
                 "scaled_extension"],
    "intent": ["plate_x_in", "plate_z", "in_zone"],
}
SD_FEATURES = ["release_sd", "movement_sd", "velo_sd"]          # consistency block
USAGE_FEATURES = ["usage", "usage_ahead", "usage_behind", "usage_vs_lhb", "usage_vs_rhb"]
OUTCOME_COLUMNS = ["rv100", "whiff_pct", "csw_pct", "xwoba_contact"]
KEYS = ["season", "half", "pitcher"]


def half_split_date(season: int) -> str:
    """First game-date of the second half: the date ending the longest summer gap.

    The schedule oracle holds regular-season games only, so the All-Star break
    shows up as the longest June-August stretch with no games (3-4 days).
    """
    import pandas as pd

    dates = pd.to_datetime(
        pd.read_parquet(paths.schedule_path(season)).game_date.drop_duplicates()
    ).sort_values()
    summer = dates[dates.dt.month.isin([6, 7, 8])].reset_index(drop=True)
    return summer[summer.diff().idxmax()].strftime("%Y-%m-%d")


def load_season(season: int, by_half: bool = True):
    """Pitch-level frame for one season, mirrored, with derived columns attached.

    by_half=False labels every pitch half=0, so the same KEYS give pitcher-season rows.
    """
    import numpy as np

    df = store.query(
        f"SELECT {', '.join(COLUMNS)} FROM pitches WHERE pitch_type IS NOT NULL",
        season,
    )
    df["pitch_type"] = df.pitch_type.astype(str)
    df = df[df.pitch_type.isin(PITCH_TYPES + list(TYPE_MAP))].copy()

    # Baselines are per raw pitch_type (KC has its own), so apply them before
    # folding grip variants. Both functions mirror internally -- raw frame in.
    df = store.add_angles_adjusted(df, season)
    df = ddz.add_ddz_delta(df, season)
    df["pitch_type"] = df.pitch_type.replace(TYPE_MAP)
    df = ddz._with_scaled_extension(df, season)
    # Height is missing for a handful of pitchers; 6.2 ft is the league median.
    df["scaled_extension"] = df.scaled_extension.fillna(df.release_extension / 6.2)

    left = (df.p_throws == "L").to_numpy()
    for c in ["pfx_x", "ax", "release_pos_x"]:
        df.loc[left, c] = -df.loc[left, c]
    df.loc[left, "spin_axis"] = (360 - df.loc[left, "spin_axis"]) % 360

    th = np.deg2rad(df.spin_axis.astype("float64"))
    df["spin_sin"], df["spin_cos"] = np.sin(th), np.cos(th)

    # Seam-shifted wake proxy: measured spin axis minus the axis the movement
    # direction implies. Convention checked on 2026: four-seam ~+8 deg, curve ~0,
    # RHP sinker ~-20 and changeup ~-14 -- the pitches known for SSW.
    implied = (180 + np.degrees(np.arctan2(-df.pfx_x, df.pfx_z))) % 360
    df["ssw_dev"] = (df.spin_axis - implied + 180) % 360 - 180

    # Horizontal location relative to the batter: positive = inside.
    df["plate_x_in"] = np.where(df.stand == "R", -df.plate_x, df.plate_x)
    df["in_zone"] = df.zone.between(1, 9).astype(float)

    desc = df.description.astype(str)
    df["whiff"] = desc.isin(WHIFFS).astype(float)
    df["swing"] = desc.isin(SWINGS).astype(float)
    df["csw"] = (desc.isin(WHIFFS) | (desc == "called_strike")).astype(float)

    df["season"] = season
    if by_half:
        split = half_split_date(season)
        df["half"] = np.where(df.game_date.astype(str) < split, 1, 2)
    else:
        df["half"] = 0
    return df


def _shrink(prof, means, raw_var, cols):
    """Empirical-Bayes shrinkage of per-row means toward the league type mean.

    shrunk = (n * xbar + k * mu) / (n + k),  k = sigma2_within / tau2_between,
    estimated per pitch type and feature. A 45-pitch relief half moves a lot; a
    400-pitch starter half barely moves.
    """
    import numpy as np

    out = means.astype("float64")  # store columns are float32
    n = prof["n"]
    for pt, idx in prof.groupby("pitch_type").groups.items():
        for c in cols:
            x, v, m = means.loc[idx, c], raw_var.loc[idx, c], n.loc[idx]
            ok = x.notna()
            if ok.sum() < 5:
                continue
            mu = np.average(x[ok], weights=m[ok])
            s2w = np.average(v[ok].fillna(0), weights=m[ok])
            tau2 = max(x[ok].var() - (s2w / m[ok]).mean(), 1e-9 * max(s2w, 1e-12))
            k = s2w / tau2
            out.loc[idx, c] = (m * x + k * mu) / (m + k)
    return out


def build_type_profiles(seasons=SEASONS, min_pitches=MIN_PITCHES, by_half=True):
    """One row per pitcher x half x pitch_type (rows below min_pitches dropped).

    Also returns untruncated usage (pitcher x half, wide, every type) -- the
    min_pitches cut makes a reliever's third pitch flicker in and out of the
    type table, which would read as adding/dropping it.
    """
    import numpy as np
    import pandas as pd

    frames, usage_all = [], []
    for season in seasons:
        df = load_season(season, by_half)
        g = df.groupby(KEYS + ["pitch_type"], observed=True)
        mean_cols = [c for block in MEAN_FEATURES.values() for c in block]

        u = g.size().unstack("pitch_type", fill_value=0).reindex(columns=PITCH_TYPES, fill_value=0)
        tot = u.sum(axis=1)
        u = u.div(tot, axis=0).add_suffix("_usage_all")
        u["n_all"] = tot
        # Whole-arsenal outcomes over every pitch (the type table drops <min_pitches rows).
        gk = df.groupby(KEYS, observed=True)
        u["whiff_pct_all"] = gk.whiff.sum() / gk.swing.sum().replace(0, np.nan)
        u["csw_pct_all"] = gk.csw.mean()
        usage_all.append(u.reset_index())

        prof = g.size().rename("n").to_frame()
        means, var = g[mean_cols].mean(), g[mean_cols].var()

        prof["release_sd"] = np.sqrt(g.release_pos_x.var() + g.release_pos_z.var())
        prof["movement_sd"] = np.sqrt(g.pfx_x.var() + g.pfx_z.var())
        prof["velo_sd"] = g.release_speed.std()

        # Usage: share of the pitcher-half's pitches, overall and by situation.
        def share(mask, name):
            sub = df[mask]
            c = sub.groupby(KEYS + ["pitch_type"], observed=True).size()
            tot = sub.groupby(KEYS, observed=True).size()
            prof[name] = (c / tot.reindex(c.index.droplevel("pitch_type")).to_numpy()
                          ).reindex(prof.index).fillna(0)

        share(np.ones(len(df), bool), "usage")
        share(df.strikes > df.balls, "usage_ahead")
        share(df.balls > df.strikes, "usage_behind")
        share(df.stand == "L", "usage_vs_lhb")
        share(df.stand == "R", "usage_vs_rhb")

        prof["rv100"] = 100 * g.delta_pitcher_run_exp.mean()  # + = good for pitcher
        rv_var = (100 ** 2) * g.delta_pitcher_run_exp.var()
        prof["whiff_pct"] = g.whiff.sum() / g.swing.sum().replace(0, np.nan)
        prof["csw_pct"] = g.csw.mean()
        prof["xwoba_contact"] = g.estimated_woba_using_speedangle.mean()

        prof = prof.join(means)
        prof["player_name"] = g.player_name.first()
        prof["p_throws"] = g.p_throws.first()
        prof = prof[prof.n >= min_pitches].reset_index()

        idx = pd.MultiIndex.from_frame(prof[KEYS + ["pitch_type"]])
        shrunk = _shrink(
            prof,
            prof[mean_cols + ["rv100"]],
            pd.concat([var.reindex(idx).reset_index(drop=True)[mean_cols],
                       rv_var.reindex(idx).reset_index(drop=True).rename("rv100")],
                      axis=1),
            mean_cols + ["rv100"],
        )
        prof["rv100_raw"] = prof.rv100
        prof[mean_cols + ["rv100"]] = shrunk
        frames.append(prof)
        del df

    return pd.concat(frames, ignore_index=True), pd.concat(usage_all, ignore_index=True)


def build_pitcher_profiles(types, usage_all=None):
    """One row per pitcher x half: type rows pivoted wide + deception features.

    Absent pitch types are left NaN with `throws_<T>` = 0; the embedding step
    decides how to fill (z-score then zero-fill = "league average for that type").
    `usage_all` (from build_type_profiles) adds untruncated `<T>_usage_all`.
    """
    import numpy as np
    import pandas as pd

    per_type = (["usage", "usage_ahead", "usage_behind", "usage_vs_lhb", "usage_vs_rhb"]
                + MEAN_FEATURES["shape"] + MEAN_FEATURES["derived"]
                + ["plate_x_in", "plate_z", "in_zone"] + SD_FEATURES)
    wide = types.pivot_table(index=KEYS, columns="pitch_type", values=per_type,
                             aggfunc="first", observed=True)
    wide.columns = [f"{t}_{f}" for f, t in wide.columns]
    for t in PITCH_TYPES:
        wide[f"throws_{t}"] = wide.get(f"{t}_usage", pd.Series(0, index=wide.index)).notna().astype(float)

    g = types.groupby(KEYS, observed=True)
    w = types.n
    out = pd.DataFrame(index=wide.index)
    out["n"] = g.n.sum()
    out["player_name"] = g.player_name.first()
    out["p_throws"] = g.p_throws.first()
    out["n_types"] = g.size()

    # Delivery is a pitcher trait: pitch-weighted across the arsenal.
    for c in MEAN_FEATURES["delivery"]:
        out[c] = (types[c] * w).groupby([types[k] for k in KEYS]).sum() / out["n"]

    # Tunneling proxy: how far apart the per-type release points sit.
    def spread(df):
        wt = df.n / df.n.sum()
        cx = (df.release_pos_x * wt).sum()
        cz = (df.release_pos_z * wt).sum()
        return np.sqrt((wt * ((df.release_pos_x - cx) ** 2 + (df.release_pos_z - cz) ** 2)).sum())
    out["release_spread"] = g[["n", "release_pos_x", "release_pos_z"]].apply(spread)

    # Primary fastball = most-thrown of FF/SI/FC; secondaries described relative to it.
    fb = types[types.pitch_type.isin(FASTBALLS)].sort_values("n")
    fb = fb.drop_duplicates(KEYS, keep="last").set_index(KEYS)
    out["primary_fb"] = fb.pitch_type
    rel = types.join(fb[["pitch_type", "release_speed", "pfx_x", "pfx_z", "vaa_above_avg"]],
                     on=KEYS, rsuffix="_fb")
    rel = rel[rel.pitch_type != rel.pitch_type_fb]
    for c, name in [("release_speed", "d_velo"), ("pfx_x", "d_pfx_x"),
                    ("pfx_z", "d_pfx_z"), ("vaa_above_avg", "d_vaa")]:
        rel[name] = rel[c] - rel[f"{c}_fb"]
    d = rel.pivot_table(index=KEYS, columns="pitch_type",
                        values=["d_velo", "d_pfx_x", "d_pfx_z", "d_vaa"], aggfunc="first")
    d.columns = [f"{t}_{f}" for f, t in d.columns]

    # Pitcher-level outcome: pitch-weighted run value across the arsenal.
    out["rv100"] = (types.rv100 * w).groupby([types[k] for k in KEYS]).sum() / out["n"]
    out["rv100_raw"] = (types.rv100_raw * w).groupby([types[k] for k in KEYS]).sum() / out["n"]

    out = out.join(wide).join(d)
    if usage_all is not None:
        out = out.join(usage_all.set_index(KEYS))
    return out.reset_index()


def profiles_path(kind: str, seasons=SEASONS, by_half=True):
    tag = "" if by_half else "season_"
    return paths.META / f"profiles_{kind}_{tag}{min(seasons)}_{max(seasons)}.parquet"


def main(seasons=SEASONS):
    types, usage_all = build_type_profiles(seasons)
    types.to_parquet(profiles_path("type", seasons), index=False)
    pitchers = build_pitcher_profiles(types, usage_all)
    pitchers.to_parquet(profiles_path("pitcher", seasons), index=False)
    for kind, df in [("type", types), ("pitcher", pitchers)]:
        print(f"{kind}: {len(df):,} rows x {df.shape[1]} cols -> {profiles_path(kind, seasons)}")
    print(types.groupby(["season", "half"]).pitcher.nunique().unstack())
    return types, pitchers


if __name__ == "__main__":
    import sys

    main(tuple(int(s) for s in sys.argv[1:]) or SEASONS)
