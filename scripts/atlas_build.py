"""Interactive pitch atlas: pitcher-season profiles -> embeddings -> comps -> one HTML page.

Exploratory only -- nothing here predicts. One dot = one pitcher-season's version
of one pitch (2024-2026). No whole-arsenal pitcher map, by design.

  all pitches        pitcher x season x pitch type, global scaling (types separate)
  <type> universe    one UMAP per pitch type, fit on that type's rows alone
  comps              10 nearest DISTINCT other pitchers in the PCA space each map
                     is drawn from (not the 2D map), any season

Levels. The maps, comps, percentiles and baselines are all MLB. Triple-A
pitchers (level "aaa", from the Triple-A store) are *placed* on the fixed MLB
maps -- each pitch lands among its nearest big-league pitches of the same type
in the PCA space the map was drawn from -- and their comps are MLB pitchers, so
every number on a Triple-A card reads as "against big-league norms". The MLB
layout is never refitted because of them.

Outcomes shown beside comps come from Savant's custom leaderboard (official
xERA, ERA, IP, K%, BB%, xwOBA, whiff%), cached in data/meta/savant_leaders_<season>.parquet.

Free agents: MLB Stats API /people/freeAgents for the offseason after the latest
season (official; former team, declared date, and once signed: new team, date,
contract length -- no dollar values). Refetched on every run so signings show up;
falls back to data/meta/free_agents_<season>.parquet if the request fails.

Outputs:
  data/meta/profiles_{type,pitcher}_season_2024_2026.parquet       (data/aaa/meta/... for Triple-A)
  data/meta/atlas_type_2024_2026.parquet             coords + comps, both levels, for Python use
  data/meta/trajectory_2024_2026.parquet             per-pitch trajectory means, for the flight animation
  analysis/pitch_atlas.html                          self-contained page (data inlined)

Usage:
  python scripts/atlas_build.py                       # reuse cached profiles/leaders
  python scripts/atlas_build.py --rebuild-profiles    # after an ingest
  python scripts/atlas_build.py --refetch-leaders     # refresh Savant outcomes
  python scripts/atlas_build.py --html-only           # re-inject data into the template
"""
import argparse
import base64
import io
import json
import urllib.request

import numpy as np
import pandas as pd

from comps import other_pitcher_neighbours
from mlbpitch import paths, profiles, store
from profile_embed import TYPE_BLOCKS, block_scale, embed, type_space, zscore

SEASONS = profiles.SEASONS
RANGE = f"{min(SEASONS)}_{max(SEASONS)}"
LEVELS = ("mlb", "aaa")   # MLB first: it is the reference every other level is measured against
K_COMPS = 10
NBHD = 30          # neighbourhood stored per pitch: first K_COMPS are the displayed comps, the rest feed
                   # the "what goes with lower xERA among his lookalikes" analysis on the page
NBHD_LEVEL = {"mlb": NBHD, "aaa": K_COMPS}   # that analysis needs his own xERA, which Triple-A has none of
K_PLACE = 15       # MLB neighbours a Triple-A pitch is placed among (the UMAP's own n_neighbors)
TEMPLATE = paths.ROOT / "scripts" / "atlas_template.html"
PITCHER_TYPES = paths.ROOT / "scripts" / "pitcher_types.json"   # "Find a type of pitcher" rules (editable)
OUT_HTML = paths.ROOT / "analysis" / "pitch_atlas.html"
ATLAS = paths.META / f"atlas_type_{RANGE}.parquet"
MOVEMENT = paths.META / f"movement_sample_{RANGE}.parquet"
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId={sport}&season={season}"
MOVE_PER_ROW = 50          # pitches sampled per pitcher x season x pitch type for the movement chart
TRAJ = paths.META / f"trajectory_{RANGE}.parquet"
# Statcast's 9-parameter trajectory fit (quoted at y = 50 ft), raw Statcast frame: +x catcher's right,
# +y toward the pitcher, +z up. Drives the card's flight animation.
TRAJ_COLS = ["release_pos_x", "release_pos_y", "release_pos_z", "vx0", "vy0", "vz0", "ax", "ay", "az"]
TRAJ_SCALE = 100           # int16 hundredths: 0.01 ft / 0.01 ft/s / 0.01 ft/s^2, well under the fit's own noise
TRAJ_NA = -32768
LEADERS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/custom?year={season}&type=pitcher"
    "&filter=&min=1&selections=p_formatted_ip,pa,k_percent,bb_percent,p_era,xera,"
    "xwoba,whiff_percent&csv=true"
)
ACTIVE_SPIN_URL = ("https://baseballsavant.mlb.com/leaderboard/active-spin?year={season}_spin-based"
                   "&min=1&hand=&csv=true")
# Savant column -> our pitch type (slurve has no type of its own here and is dropped)
ACTIVE_SPIN_TYPES = {"fourseam": "FF", "sinker": "SI", "cutter": "FC", "changeup": "CH", "splitter": "FS",
                     "curve": "CU", "slider": "SL", "sweeper": "ST"}
FA_URL = "https://statsapi.mlb.com/api/v1/people/freeAgents?season={season}"
TYPE_FEATURES = [c for cs in TYPE_BLOCKS.values() for c in cs]


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------- inputs
def level_seasons(level):
    """Seasons with a parquet store at this level (a level can trail the MLB backfill)."""
    import glob
    return tuple(s for s in SEASONS if glob.glob(store.pitches_glob(s, level)))


def season_profiles(rebuild, level="mlb"):
    p_type = profiles.profiles_path("type", SEASONS, by_half=False, level=level)
    p_pitch = profiles.profiles_path("pitcher", SEASONS, by_half=False, level=level)
    if rebuild or not (p_type.exists() and p_pitch.exists()):
        types, usage_all = profiles.build_type_profiles(level_seasons(level), by_half=False, level=level)
        pitchers = profiles.build_pitcher_profiles(types, usage_all)
        types.to_parquet(p_type, index=False)
        pitchers.to_parquet(p_pitch, index=False)
        log(f"profiles {level}: {len(types):,} type rows, {len(pitchers):,} pitcher rows")
    return pd.read_parquet(p_type), pd.read_parquet(p_pitch)


def leaders(season, refetch):
    path = paths.META / f"savant_leaders_{season}.parquet"
    if path.exists() and not refetch:
        return pd.read_parquet(path)
    req = urllib.request.Request(LEADERS_URL.format(season=season),
                                 headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=120).read().decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(raw), dtype={"p_formatted_ip": str})
    parts = df.p_formatted_ip.fillna("0").str.partition(".")   # "184.1" = 184 1/3 IP
    df["ip"] = parts[0].astype(float) + pd.to_numeric(parts[2], errors="coerce").fillna(0) / 3
    df = df.rename(columns={"player_id": "pitcher", "year": "season", "k_percent": "k_pct",
                            "bb_percent": "bb_pct", "p_era": "era", "whiff_percent": "whiff_sv"})
    out = df[["pitcher", "season", "ip", "pa", "k_pct", "bb_pct", "era", "xera", "xwoba", "whiff_sv"]]
    out = out.apply(pd.to_numeric, errors="coerce")
    out.to_parquet(path, index=False)
    log(f"leaders {season}: {len(out)} pitchers")
    return out


def active_spin(season, refetch):
    """Savant spin efficiency (active spin %, Hawk-Eye spin-based) per pitcher x season x pitch type."""
    path = paths.META / f"active_spin_{season}.parquet"
    if path.exists() and not refetch:
        return pd.read_parquet(path)
    req = urllib.request.Request(ACTIVE_SPIN_URL.format(season=season), headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=120).read().decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(raw))
    long = df.melt(id_vars=["entity_id"], value_vars=[f"active_spin_{k}" for k in ACTIVE_SPIN_TYPES],
                   var_name="col", value_name="spin_eff")
    long["pitch_type"] = long.col.str.removeprefix("active_spin_").map(ACTIVE_SPIN_TYPES)
    long["spin_eff"] = pd.to_numeric(long.spin_eff, errors="coerce")
    out = (long.dropna(subset=["spin_eff"]).rename(columns={"entity_id": "pitcher"})
           .assign(season=season)[["pitcher", "season", "pitch_type", "spin_eff"]])
    out.to_parquet(path, index=False)
    log(f"active spin {season}: {out.pitcher.nunique()} pitchers, {len(out)} pitcher-pitch rows")
    return out


def free_agents(season=max(SEASONS)):
    path = paths.META / f"free_agents_{season}.parquet"
    try:
        req = urllib.request.Request(FA_URL.format(season=season), headers={"User-Agent": "Mozilla/5.0"})
        rows = json.load(urllib.request.urlopen(req, timeout=60))["freeAgents"]
        fa = pd.DataFrame([{
            "pitcher": r["player"]["id"],
            "fa_from": r.get("originalTeam", {}).get("name"),
            "fa_to": (r.get("newTeam") or {}).get("name"),
            "fa_signed": r.get("dateSigned"),
            "fa_declared": r.get("dateDeclared"),
            "fa_notes": r.get("notes"),
            "fa_pos": r.get("position", {}).get("abbreviation"),
        } for r in rows])
        fa.to_parquet(path, index=False)
        log(f"free agents {season}-{season + 1 - 2000}: {len(fa)} players, "
            f"{fa.fa_to.notna().sum()} signed")
    except Exception as e:  # offline or API change: keep the last good copy
        if not path.exists():
            raise
        fa = pd.read_parquet(path)
        log(f"free agents: fetch failed ({e}); using cached {path.name}")
    return fa


def movement_sample(rebuild, level="mlb"):
    """Up to MOVE_PER_ROW pitches per pitcher x season x pitch type: raw (catcher's view, not mirrored)
    horizontal and induced vertical break in inches, for the Savant-style movement chart."""
    path = paths.meta_dir(level) / MOVEMENT.name
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    frames = []
    for season in level_seasons(level):
        df = store.query("SELECT pitcher, pitch_type, pfx_x, pfx_z FROM pitches "
                         "WHERE pitch_type IS NOT NULL AND pfx_x IS NOT NULL AND pfx_z IS NOT NULL", season, level)
        df["pitch_type"] = df.pitch_type.astype(str).replace(profiles.TYPE_MAP)
        df = df[df.pitch_type.isin(profiles.PITCH_TYPES)]
        df = df.sample(frac=1, random_state=0).groupby(["pitcher", "pitch_type"]).head(MOVE_PER_ROW)
        df["season"] = season
        df["mx"] = (24 * df.pfx_x).round().clip(-127, 127).astype("int8")   # half-inch units
        df["mz"] = (24 * df.pfx_z).round().clip(-127, 127).astype("int8")
        frames.append(df[["pitcher", "season", "pitch_type", "mx", "mz"]])
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)
    log(f"movement sample {level}: {len(out):,} pitches")
    return out


def movement_blob(types, mv):
    """Per type row (in export order): count, then all x then all z as int8, base64."""
    g = {k: v for k, v in mv.groupby(["level", "pitcher", "season", "pitch_type"])}
    counts, xs, zs = [], [], []
    for r in types[["level", "pitcher", "season", "pitch_type"]].itertuples(index=False):
        part = g.get((r.level, r.pitcher, r.season, r.pitch_type))
        n = 0 if part is None else len(part)
        counts.append(n)
        if n:
            xs.append(part.mx.to_numpy()); zs.append(part.mz.to_numpy())
    x = np.concatenate(xs).astype("int8"); z = np.concatenate(zs).astype("int8")
    return {"counts": counts, "b64": base64.b64encode(np.concatenate([x, z]).tobytes()).decode()}


def trajectory_means(rebuild, level="mlb"):
    """Raw (unshrunk) per pitcher x season x pitch type means of the trajectory parameters, plus league means
    per season x hand x type (pitcher = -1) and each season's mean strike zone (sz_top/sz_bot, every row).

    The trajectory is linear in its parameters, so the path built from the mean parameters is the mean path.
    """
    path = paths.meta_dir(level) / TRAJ.name
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    frames = []
    for season in level_seasons(level):
        cols = ", ".join(TRAJ_COLS)
        df = store.query(f"SELECT pitcher, p_throws, pitch_type, sz_top, sz_bot, {cols} FROM pitches "
                         "WHERE pitch_type IS NOT NULL AND vy0 IS NOT NULL AND release_pos_y IS NOT NULL", season, level)
        df["pitch_type"] = df.pitch_type.astype(str).replace(profiles.TYPE_MAP)
        df = df[df.pitch_type.isin(profiles.PITCH_TYPES)]
        per = df.groupby(["pitcher", "p_throws", "pitch_type"], observed=True)[TRAJ_COLS].mean()
        per["n"] = df.groupby(["pitcher", "p_throws", "pitch_type"], observed=True).size()
        lg = df.groupby(["p_throws", "pitch_type"], observed=True)[TRAJ_COLS].mean()
        lg["n"] = df.groupby(["p_throws", "pitch_type"], observed=True).size()
        lg["pitcher"] = -1
        out = pd.concat([per.reset_index(), lg.reset_index()], ignore_index=True)
        out["season"] = season
        out["sz_top"], out["sz_bot"] = df.sz_top.mean(), df.sz_bot.mean()
        frames.append(out)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)
    log(f"trajectories {level}: {len(out):,} pitcher/league x season x type rows")
    return out


def trajectory_blob(types, tj):
    """Per type row (in export order): the 9 trajectory parameters as int16 hundredths, base64 (TRAJ_NA where
    missing); league rows keyed "season|hand|type"; strike zone per season."""
    # keyed by level as well: a call-up's Triple-A row must not borrow his MLB flight. League rows are MLB's,
    # the reference every level's flight is compared with.
    # a switch-pitcher has a row per hand: keep the hand he threw that pitch with more
    own = (tj[tj.pitcher >= 0].sort_values("n")
           .drop_duplicates(["level", "pitcher", "season", "pitch_type"], keep="last")
           .set_index(["level", "pitcher", "season", "pitch_type"])[TRAJ_COLS])
    idx = pd.MultiIndex.from_frame(types[["level", "pitcher", "season", "pitch_type"]])
    m = own.reindex(idx).to_numpy(dtype="float64")
    q = np.where(np.isfinite(m), np.round(m * TRAJ_SCALE), TRAJ_NA)
    assert (np.abs(q[np.isfinite(m)]) < 32768).all(), "trajectory value out of int16 range"
    lg = tj[(tj.pitcher < 0) & (tj.level == "mlb")]
    league = {f"{r.season}|{r.p_throws}|{r.pitch_type}": [round(float(getattr(r, c)), 3) for c in TRAJ_COLS] + [int(r.n)]
              for r in lg.itertuples()}
    zone = {str(s): [round(float(g.sz_top.iloc[0]), 2), round(float(g.sz_bot.iloc[0]), 2)]
            for s, g in tj[tj.level == "mlb"].groupby("season")}   # the zone drawn is MLB's
    return {"cols": TRAJ_COLS, "scale": TRAJ_SCALE, "na": TRAJ_NA, "league": league, "zone": zone,
            "b64": base64.b64encode(q.astype("<i2").tobytes()).decode()}


def pitcher_teams(level="mlb"):
    """Team(s) per pitcher-season, in order of first appearance ("NYY,CHC" if traded),
    plus team_main = the team he threw the most pitches for (used for colour).

    The pitching team is the home team in the top of an inning. OAK (2024) is
    folded into ATH so the franchise filters as one team.
    """
    frames = []
    for season in level_seasons(level):
        df = store.query(
            "SELECT pitcher, CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS team, "
            "min(game_date) AS first, count(*) AS n FROM pitches GROUP BY 1, 2", season, level)
        df["season"] = season
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["team"] = df.team.replace({"OAK": "ATH"})
    df = df.groupby(["pitcher", "season", "team"], as_index=False).agg(first=("first", "min"), n=("n", "sum"))
    g = df.sort_values("first").groupby(["pitcher", "season"])
    out = g.team.agg(",".join).rename("teams").to_frame()
    out["team_main"] = df.sort_values("n").groupby(["pitcher", "season"]).team.last()
    return out.reset_index().assign(level=level)


def minor_teams(level="aaa"):
    """Triple-A team directory: Savant code -> [club name, parent MLB code], from the MLB Stats API.

    Affiliations move now and then, so every season in the atlas is fetched and the latest wins.
    Cached per season under the level's meta dir; the page colours a Triple-A pitcher by the parent club.
    Savant's codes match statsapi abbreviations at both levels (checked 2026-11-03).
    """
    frames = []
    for season in level_seasons(level):
        path = paths.meta_dir(level) / f"teams_{season}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path)); continue
        req = urllib.request.Request(TEAMS_URL.format(sport=1, season=season), headers={"User-Agent": "Mozilla/5.0"})
        parent = {t["id"]: t["abbreviation"] for t in json.load(urllib.request.urlopen(req, timeout=60))["teams"]}
        req = urllib.request.Request(TEAMS_URL.format(sport=paths.SPORT_ID[level], season=season),
                                     headers={"User-Agent": "Mozilla/5.0"})
        rows = [{"code": t["abbreviation"], "name": t["name"], "parent": parent.get(t.get("parentOrgId")), "season": season}
                for t in json.load(urllib.request.urlopen(req, timeout=60))["teams"]]
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        log(f"{level} teams {season}: {len(df)} clubs")
        frames.append(df)
    if not frames:
        return {}
    df = pd.concat(frames).sort_values("season").drop_duplicates("code", keep="last")
    return {r.code: [r.name, r.parent] for r in df.itertuples()}


# ----------------------------------------------------------------- embedding
def _fill(F, med, by_type):
    if by_type is None:
        return F.fillna(med)
    fill = med.reindex(np.asarray(by_type))
    fill.index = F.index
    return F.fillna(fill)


def scaler_fit(F, by_type=None):
    """The feature scaling of one map, fitted on MLB rows so another level can be put in the same space.

    Reproduces block_scale(zscore(F.fillna(median))) exactly for the rows it was fitted on. `by_type`
    (those rows' pitch types) makes the fill median per type, as the all-pitches map does.
    """
    F = F.astype("float64")
    med = F.median() if by_type is None else F.groupby(np.asarray(by_type)).median()
    Ff = _fill(F, med, by_type)
    mu, sd = Ff.mean(), Ff.std(ddof=0).replace(0, 1)
    Z = (Ff - mu) / sd
    fac = {}
    for name, cols in TYPE_BLOCKS.items():
        tot = np.nansum(np.nanvar(Z[cols].to_numpy(dtype="float64"), axis=0))
        fac[name] = np.sqrt(tot if tot > 0 else 1)
    return {"med": med, "mu": mu, "sd": sd, "fac": fac}


def scaler_apply(sc, F, by_type=None):
    Z = (_fill(F.astype("float64"), sc["med"], by_type) - sc["mu"]) / sc["sd"]
    return np.nan_to_num(np.hstack([Z[cols].to_numpy(dtype="float64") / sc["fac"][n] for n, cols in TYPE_BLOCKS.items()]))


def place_level(types_mlb, space_mlb, types_new):
    """Put another level's pitches on the fixed MLB maps.

    Each pitch lands at the inverse-distance-weighted mean position of its K_PLACE nearest MLB pitches
    of the same type in the PCA space that map was drawn from (global scaling for the all-pitches map,
    within-type scaling for each type's map), and takes their weighted-majority cluster. Nothing MLB moves.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    types_mlb, space_mlb = types_mlb.reset_index(drop=True), space_mlb.reset_index(drop=True)
    types_new = types_new.reset_index(drop=True)
    assert (types_mlb.pitcher.to_numpy() == space_mlb.pitcher.to_numpy()).all()
    G_COLS, T_COLS = ["gx", "gy", "gx3", "gy3", "gz3"], ["tx", "ty", "tx3", "ty3", "tz3"]
    out = types_new[profiles.KEYS + ["pitch_type"]].copy()
    for c in G_COLS + T_COLS:
        out[c] = np.nan
    out["tcluster"] = -1

    def land(P_fit, P_new, rows_fit, rows_new, cols, cluster):
        d, i = NearestNeighbors(n_neighbors=min(K_PLACE, len(rows_fit))).fit(P_fit).kneighbors(P_new)
        w = 1 / (d + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        for c in cols:
            if c in space_mlb:
                out.loc[rows_new, c] = (w * space_mlb[c].to_numpy(dtype="float64")[rows_fit][i]).sum(axis=1)
        if cluster:
            cl = space_mlb.tcluster.to_numpy()[rows_fit][i]
            out.loc[rows_new, "tcluster"] = [int(np.bincount(r[r >= 0], weights=ww[r >= 0]).argmax()) if (r >= 0).any() else -1
                                             for r, ww in zip(cl, w)]

    Fm, Fn = types_mlb[TYPE_FEATURES], types_new[TYPE_FEATURES]
    gm, gn = types_mlb.groupby("pitch_type").indices, types_new.groupby("pitch_type").indices
    # all-pitches map: one global scaling, same-type neighbours
    sc = scaler_fit(Fm, types_mlb.pitch_type)
    Xm = scaler_apply(sc, Fm, types_mlb.pitch_type)
    pca = PCA(n_components=min(30, Xm.shape[1]), random_state=0).fit(Xm)
    Pm, Pn = pca.transform(Xm), pca.transform(scaler_apply(sc, Fn, types_new.pitch_type))
    for pt, rn in gn.items():
        if pt in gm:
            land(Pm[gm[pt]], Pn[rn], gm[pt], rn, G_COLS, False)
    # each type's map: that type's own scaling and PCA
    for pt, rn in gn.items():
        if pt not in gm:
            continue
        rm = gm[pt]
        sc = scaler_fit(Fm.iloc[rm])
        Xm = scaler_apply(sc, Fm.iloc[rm])
        pca = PCA(n_components=min(30, Xm.shape[1]), random_state=0).fit(Xm)
        land(pca.transform(Xm), pca.transform(scaler_apply(sc, Fn.iloc[rn])), rm, rn, T_COLS, True)
    log(f"placed {len(out):,} pitches on the MLB maps among their {K_PLACE} nearest big-league pitches")
    return out


def comps_for(P, pid, rows=None):
    """k nearest distinct other pitchers; indices returned in the caller's frame."""
    rows = np.arange(len(P)) if rows is None else rows
    i, d = other_pitcher_neighbours(P[rows], pid[rows], K_COMPS)
    gi = np.where(i >= 0, rows[np.clip(i, 0, None)], -1)
    first = np.sort(d[:, 0][~np.isnan(d[:, 0])])
    pct = np.searchsorted(first, np.nan_to_num(d, nan=np.inf)) / len(first)
    return gi, np.where(np.isnan(d), np.nan, pct)


ARM_TOLS = (7.5, 10, 12.5, 15, 20, 30, np.inf)   # arm-angle window, widened only when too few comps


def type_comps(types, query_level=None, cand_level="mlb"):
    """Comps per pitch: same pitch type, SAME HAND, arm angle within ARM_TOLS[0] degrees (the window
    widens step by step only when fewer than K_COMPS distinct other pitchers qualify), ranked by
    distance in that type's PCA space (the same space the per-type maps are drawn from).
    Returns comp rows, fit percentile, and the arm-angle window actually used per row.

    `types` may hold several levels (column `level`). The space is always fitted on MLB rows.
    `cand_level` picks the candidates and `query_level` the rows that get comps (None = every row):
      type_comps(t)                              every pitch's comps among MLB pitchers (the default)
      type_comps(t, None, "aaa")                 every pitch's lookalikes among Triple-A pitchers
    A pitcher never comps to himself, at either level. Rows that are not queries get no comps."""
    from scipy.spatial.distance import cdist
    from sklearn.decomposition import PCA

    types = types.reset_index(drop=True)
    n = len(types)
    rows_out = np.full((n, NBHD), -1)
    dist_out = np.full((n, NBHD), np.nan)
    tol_out = np.full(n, np.nan)
    pid = types.pitcher.to_numpy()
    arm = types.arm_angle.to_numpy(dtype="float64")
    hand = types.p_throws.to_numpy()
    lvl = types.level.to_numpy() if "level" in types else np.full(n, "mlb")
    is_query = np.ones(n, bool) if query_level is None else lvl == query_level
    nbhd_of = (lambda l: NBHD_LEVEL.get(l, NBHD)) if cand_level == "mlb" else (lambda l: K_COMPS)
    for pt, rows in types.groupby("pitch_type").indices.items():
        is_m = lvl[rows] == "mlb"
        mrows = rows[is_m]
        if not len(mrows):
            continue
        sc = scaler_fit(types.loc[mrows, TYPE_FEATURES])
        pca = PCA(n_components=min(30, len(TYPE_FEATURES)), random_state=0)
        Pc = np.zeros((len(rows), pca.n_components))
        Pc[is_m] = pca.fit_transform(scaler_apply(sc, types.loc[mrows, TYPE_FEATURES]))
        if (~is_m).any():
            Pc[~is_m] = pca.transform(scaler_apply(sc, types.loc[rows[~is_m], TYPE_FEATURES]))
        is_c = lvl[rows] == cand_level
        for h in np.unique(hand[rows]):
            local = np.flatnonzero((hand[rows] == h) & is_query[rows])   # queries
            cand = np.flatnonzero((hand[rows] == h) & is_c)              # candidates
            g, gc = rows[local], rows[cand]
            if len(gc) < 2 or not len(g):
                continue
            D = cdist(Pc[local], Pc[cand])
            for a in range(len(g)):
                nbhd = nbhd_of(lvl[g[a]])
                order = np.argsort(D[a])
                darm = np.abs(arm[gc] - arm[g[a]])
                not_self = gc[order] != g[a]
                for tol in ARM_TOLS:
                    ok = order[not_self & (np.isinf(tol) | (darm[order] <= tol))]
                    seen, picks = {pid[g[a]]}, []
                    for b in ok:
                        if pid[gc[b]] in seen:
                            continue
                        seen.add(pid[gc[b]]); picks.append(b)
                        if len(picks) == K_COMPS:
                            break
                    if len(picks) == K_COMPS or np.isinf(tol):
                        break
                # extend to the NBHD neighbourhood without touching the first K_COMPS: same window first,
                # widening further only if needed; appended in distance order within each window
                for tol2 in [t2 for t2 in ARM_TOLS if t2 >= tol]:
                    if len(picks) >= nbhd:
                        break
                    ok = order[not_self & (np.isinf(tol2) | (darm[order] <= tol2))]
                    for b in ok:
                        if pid[gc[b]] in seen:
                            continue
                        seen.add(pid[gc[b]]); picks.append(b)
                        if len(picks) == nbhd:
                            break
                rows_out[g[a], :len(picks)] = gc[picks]
                dist_out[g[a], :len(picks)] = D[a, picks]
                tol_out[g[a]] = tol
    # fit percentile against the reference distribution of nearest-comp distances: MLB pitches among
    # MLB candidates, or the queried level among its own
    ref = lvl == cand_level   # MLB pitches among MLB candidates, or Triple-A pitches among Triple-A
    first = np.sort(dist_out[ref, 0][~np.isnan(dist_out[ref, 0])])
    pct = np.searchsorted(first, np.nan_to_num(dist_out, nan=np.inf)) / len(first)
    widened = np.nanmean(tol_out[is_query] > ARM_TOLS[0])
    log(f"comps among {cand_level} for {'every' if query_level is None else query_level} pitch: same hand, "
        f"arm angle within {ARM_TOLS[0]} deg ({100 * widened:.1f}% of pitches needed a wider window)")
    return rows_out, np.where(np.isnan(dist_out), np.nan, pct), tol_out


def umap_3d(types):
    """3D UMAP coordinates on the same PCA spaces as the 2D maps (2D layouts untouched).

    All pitches: global z-scoring (pitch types separate), like profile_embed.type_space("global").
    Each pitch type: within-type z-scoring, like type_comps. Same neighbours/min_dist/seed as 2D.
    """
    import umap as umap_mod
    from sklearn.decomposition import PCA

    types = types.reset_index(drop=True)
    out = pd.DataFrame(index=types.index, columns=["gx3", "gy3", "gz3", "tx3", "ty3", "tz3"], dtype="float64")
    F = types[TYPE_FEATURES].astype("float64")
    F = F.fillna(F.groupby(types.pitch_type).transform("median"))
    Pg = PCA(n_components=30, random_state=0).fit_transform(block_scale(zscore(F), TYPE_BLOCKS))
    out[["gx3", "gy3", "gz3"]] = umap_mod.UMAP(n_components=3, n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(Pg)
    log("umap 3d: all pitches done")
    for pt, rows in types.groupby("pitch_type").indices.items():
        Ft = types.loc[rows, TYPE_FEATURES].astype("float64")
        Pt = PCA(n_components=30, random_state=0).fit_transform(block_scale(zscore(Ft.fillna(Ft.median())), TYPE_BLOCKS))
        out.loc[rows, ["tx3", "ty3", "tz3"]] = umap_mod.UMAP(n_components=3, n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(Pt)
        log(f"umap 3d: {pt} done")
    return out


def build_type_space(types):
    types = types.reset_index(drop=True)
    glob = type_space(types)                     # global scaling: the "all pitches" map
    out = types[profiles.KEYS + ["pitch_type"]].copy()
    out["gx"], out["gy"] = glob.global_umap1.to_numpy(), glob.global_umap2.to_numpy()
    out["tx"] = out["ty"] = np.nan
    out["tcluster"] = -1
    comp_rows = np.full((len(types), K_COMPS), -1)
    comp_pct = np.full((len(types), K_COMPS), np.nan)
    pid = types.pitcher.to_numpy()
    for pt, rows in types.groupby("pitch_type").indices.items():
        F = types.loc[rows, TYPE_FEATURES].astype("float64")
        X = block_scale(zscore(F.fillna(F.median())), TYPE_BLOCKS)
        P_sub, _, leiden, U = embed(X, f"type/{pt}")
        out.loc[rows, ["tx", "ty"]] = U
        out.loc[rows, "tcluster"] = leiden
        P = np.zeros((len(types), P_sub.shape[1]))
        P[rows] = P_sub
        ci, cp = comps_for(P, pid, rows)
        comp_rows[rows], comp_pct[rows] = ci, cp
    out["comp_rows"], out["comp_pct"] = list(comp_rows), list(comp_pct)
    return out


# -------------------------------------------------------------------- export
def first_last(name):
    last, _, first = str(name).partition(", ")
    return f"{first} {last}".strip()


def spin_axis_deg(s, c):
    return np.degrees(np.arctan2(s, c)) % 360


def type_display(t):
    """Profile columns -> display units. Movement/location signs: arm-side +, inside +."""
    d = pd.DataFrame(index=t.index)
    d["velo"] = t.release_speed
    d["spin"] = t.release_spin_rate
    d["spin_eff"] = t.spin_eff        # Savant active spin %, NaN where Savant has no value
    d["spin_axis"] = spin_axis_deg(t.spin_sin, t.spin_cos)
    d["ivb"] = 12 * t.pfx_z
    d["hb"] = -12 * t.pfx_x
    d["ax"] = -t.ax
    d["az"] = t.az
    for c in ["vaa", "haa", "vra", "hra"]:
        d[c] = t[f"{c}_above_avg"]
    d["ddz_ax"] = -t.ddz_delta_ax
    d["ddz_az"] = t.ddz_delta_az
    d["ddz_az_velo"] = t.ddz_delta_az_velo
    d["ssw"] = t.ssw_dev
    d["arm_angle"] = t.arm_angle
    d["rel_height"] = t.release_pos_z
    d["rel_side"] = -t.release_pos_x
    d["extension"] = t.release_extension
    d["scaled_ext"] = t.scaled_extension
    d["loc_inside"] = 12 * t.plate_x_in
    d["loc_height"] = t.plate_z
    d["zone_pct"] = 100 * t.in_zone
    d["release_sd"] = 12 * t.release_sd
    d["movement_sd"] = 12 * t.movement_sd
    d["velo_sd"] = t.velo_sd
    for c in profiles.USAGE_FEATURES:
        d[c] = 100 * t[c]
    d["whiff"] = 100 * t.whiff_pct
    d["csw"] = 100 * t.csw_pct
    d["xwoba_con"] = t.xwoba_contact
    d["rv100"] = t.rv100
    d["uniq_z"] = uniqueness(t)      # SDs from the typical pitch of its type (all measures)
    return d


def uniqueness(types):
    """Per pitch: how far it sits from its pitch type's typical pitch, in SDs.

    Same feature space the comps use (block-scaled, z within type); distance to the type's
    median vector, then z-scored within type. "Unique unicorns" on the page are > 2.
    The typical pitch, the scaling and the SD are MLB's; other levels are measured against them.
    """
    out = pd.Series(np.nan, index=types.index)
    lvl = types.level if "level" in types else pd.Series("mlb", index=types.index)
    for pt, rows in types.groupby("pitch_type").groups.items():
        mrows = rows[lvl.loc[rows].to_numpy() == "mlb"]
        if not len(mrows):
            continue
        sc = scaler_fit(types.loc[mrows, TYPE_FEATURES])
        Xm = scaler_apply(sc, types.loc[mrows, TYPE_FEATURES])
        centre = np.median(Xm, axis=0)
        dm = np.linalg.norm(Xm - centre, axis=1)
        d = np.linalg.norm(scaler_apply(sc, types.loc[rows, TYPE_FEATURES]) - centre, axis=1)
        out.loc[rows] = (d - dm.mean()) / dm.std()
    return out


def pitcher_display(p):
    d = pd.DataFrame(index=p.index)
    d["arm_angle"] = p.arm_angle
    d["rel_height"] = p.release_pos_z
    d["rel_side"] = -p.release_pos_x
    d["extension"] = p.release_extension
    d["scaled_ext"] = p.scaled_extension
    d["release_spread"] = 12 * p.release_spread
    d["n_types"] = p.n_types
    for t in profiles.PITCH_TYPES:
        d[f"use_{t}"] = 100 * p[f"{t}_usage_all"].fillna(0)
    d["whiff"] = 100 * p.whiff_pct_all
    d["csw"] = 100 * p.csw_pct_all
    d["rv100"] = p.rv100
    for c in ["xera", "era", "ip", "k_pct", "bb_pct", "xwoba", "whiff_sv"]:
        d[c] = p[c]
    d["luck"] = p.era - p.xera   # + = unlucky: more runs than K/BB/contact quality predict
    # per nine innings, from Savant's K% / BB% of batters faced
    d["k9"] = 9 * (p.k_pct / 100 * p.pa) / p.ip.where(p.ip > 0)
    d["bb9"] = 9 * (p.bb_pct / 100 * p.pa) / p.ip.where(p.ip > 0)
    return d


def columnar(df, decimals=3):
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            vals = [None if not np.isfinite(v) else float(f"{v:.{decimals}g}")
                    for v in s.astype("float64")]
            out[c] = vals
        elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
            out[c] = [int(v) for v in s]
        else:
            out[c] = [None if pd.isna(v) else str(v) for v in s]
    return out


def comps_columnar(rows, pct):
    return [[[int(r), round(float(q), 3)] for r, q in zip(rr, qq) if r >= 0]
            for rr, qq in zip(rows, pct)]


def export(types, pitchers, t_space, lead, fa, spin, mv, tj, teams, levels, teams_aaa):
    """`types`, `pitchers`, `t_space`, `mv`, `tj` and `teams` carry a `level` column; Savant outcomes and
    spin are MLB-only and join on level too, so a call-up's Triple-A row never borrows his big-league xERA."""
    key = profiles.KEYS + ["level"]
    lead, spin = lead.assign(level="mlb"), spin.assign(level="mlb")
    pitchers = pitchers.merge(lead, on=["pitcher", "season", "level"], how="left").merge(
        teams, on=["pitcher", "season", "level"], how="left")
    types = types.merge(t_space, on=key + ["pitch_type"], how="inner").merge(
        spin, on=["pitcher", "season", "pitch_type", "level"], how="left")
    # comp_rows index t_space rows; the merge kept its order (inner on unique keys)
    assert len(types) == len(t_space)
    assert (types.pitcher.to_numpy() == t_space.pitcher.to_numpy()).all()
    assert (types.level.to_numpy() == t_space.level.to_numpy()).all()

    t_out = types[["pitcher", "season", "level", "pitch_type", "p_throws", "n", "gx", "gy", "tx", "ty",
                   "tcluster", "comp_tol", "gx3", "gy3", "gz3", "tx3", "ty3", "tz3"]
                  + (["comp_tol_aaa"] if "comp_tol_aaa" in types else [])].copy()
    t_out = pd.concat([t_out, type_display(types)], axis=1)
    p_out = pitchers[["pitcher", "season", "level", "p_throws", "n", "primary_fb", "teams", "team_main"]].copy()
    p_out = pd.concat([p_out, pitcher_display(pitchers)], axis=1)
    p_out["name"] = pitchers.player_name.map(first_last)
    t_out["name"] = types.player_name.map(first_last)
    mlb_p, mlb_t = pitchers.level == "mlb", types.level == "mlb"
    log(f"Savant outcomes matched for {pitchers[mlb_p].xera.notna().mean():.1%} of MLB pitcher-seasons; "
        f"team for {pitchers.teams.notna().mean():.1%}; "
        f"{fa.pitcher.isin(pitchers.pitcher).sum()} free agents appear in the atlas; "
        f"spin efficiency for {types[mlb_t].spin_eff.notna().mean():.1%} of MLB pitch rows; "
        + "; ".join(f"{l}: {int((pitchers.level == l).sum()):,} pitcher-seasons, {int((types.level == l).sum()):,} pitches"
                    for l in levels))

    data = {
        "seasons": list(SEASONS),
        "levels": list(levels),
        "teams_aaa": {c: v for c, v in teams_aaa.items() if c in set(",".join(teams.teams).split(","))},
        "pitchers": columnar(p_out),
        "types": columnar(t_out),
        "tcomps": comps_columnar(types.comp_rows, types.comp_pct),
        # every pitch's lookalikes among Triple-A pitchers
        **({"tcomps_aaa": comps_columnar(types.comp_rows_aaa, types.comp_pct_aaa)} if "comp_rows_aaa" in types else {}),
        "movement": movement_blob(types, mv),
        "flight": trajectory_blob(types, tj),
        "archetypes": json.loads(PITCHER_TYPES.read_text()) if PITCHER_TYPES.exists() else [],
        "fa_season": max(SEASONS),
        "fa": {str(r.pitcher): [None if pd.isna(v) else v for v in
                                (r.fa_from, r.fa_to, r.fa_signed, r.fa_declared, r.fa_notes)]
               for r in fa[fa.pitcher.isin(pitchers.pitcher)].itertuples()},
    }
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False)
    html = TEMPLATE.read_text().replace("/*__ATLAS_DATA__*/null", blob)
    OUT_HTML.write_text(html)
    log(f"wrote {OUT_HTML}  ({len(html) / 1e6:.1f} MB; {len(p_out):,} pitcher-seasons, "
        f"{len(t_out):,} pitch rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-profiles", action="store_true")
    ap.add_argument("--refetch-leaders", action="store_true")
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()

    types, pitchers = season_profiles(a.rebuild_profiles, "mlb")
    lead = pd.concat([leaders(s, a.refetch_leaders) for s in SEASONS], ignore_index=True)
    if a.html_only and ATLAS.exists():
        t_space = pd.read_parquet(ATLAS)
        if "level" in t_space:   # other levels are re-placed below; only the MLB layout is cached
            t_space = t_space[t_space.level == "mlb"].drop(columns="level").reset_index(drop=True)
    else:
        t_space = build_type_space(types)
    # 3D layouts: computed once and cached in the atlas table
    if not {"gx3", "tx3"} <= set(t_space.columns):
        u3 = umap_3d(types)
        for c in u3.columns:
            t_space[c] = u3[c].to_numpy()

    # other levels ride on the MLB maps: placed among their nearest big-league pitches, never refitted
    levels = ["mlb"] + [l for l in LEVELS if l != "mlb" and level_seasons(l)]
    types, pitchers, t_space = [df.reset_index(drop=True).assign(level="mlb") for df in (types, pitchers, t_space)]
    parts = {"types": [types], "pitchers": [pitchers], "space": [t_space], "mv": [movement_sample(a.rebuild_profiles)],
             "tj": [trajectory_means(a.rebuild_profiles)], "teams": [pitcher_teams()]}
    teams_aaa = {}
    for level in levels[1:]:
        t_l, p_l = season_profiles(a.rebuild_profiles, level)
        parts["types"].append(t_l.assign(level=level)); parts["pitchers"].append(p_l.assign(level=level))
        parts["space"].append(place_level(types, t_space, t_l).assign(level=level))
        parts["mv"].append(movement_sample(a.rebuild_profiles, level))
        parts["tj"].append(trajectory_means(a.rebuild_profiles, level))
        parts["teams"].append(pitcher_teams(level))
        if level == "aaa":
            teams_aaa = minor_teams(level)
    types, pitchers, t_space, teams = [pd.concat(parts[k], ignore_index=True) for k in ("types", "pitchers", "space", "teams")]
    mv = pd.concat([m.assign(level=l) for m, l in zip(parts["mv"], levels)], ignore_index=True)
    tj = pd.concat([t.assign(level=l) for t, l in zip(parts["tj"], levels)], ignore_index=True)

    # comps are cheap and rule-driven (same hand, similar arm angle): always recompute
    cr, cp, ctol = type_comps(types)
    t_space["comp_rows"], t_space["comp_pct"], t_space["comp_tol"] = list(cr), list(cp), ctol
    # every pitch, MLB or Triple-A, also gets its lookalikes among Triple-A pitchers (the page offers both:
    # "MLB similar | AAA similar"), so an MLB arm's Triple-A doppelgangers can be saved and followed to that map
    if "aaa" in levels:
        cr, cp, ctol = type_comps(types, None, "aaa")
        t_space["comp_rows_aaa"], t_space["comp_pct_aaa"], t_space["comp_tol_aaa"] = list(cr), list(cp), ctol
    t_space.to_parquet(ATLAS, index=False)
    spin = pd.concat([active_spin(s, a.refetch_leaders) for s in SEASONS], ignore_index=True)
    export(types, pitchers, t_space, lead, free_agents(), spin, mv, tj, teams, levels, teams_aaa)


if __name__ == "__main__":
    main()
