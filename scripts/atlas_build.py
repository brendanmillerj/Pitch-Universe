"""Interactive pitch atlas: pitcher-season profiles -> embeddings -> comps -> one HTML page.

Exploratory only -- nothing here predicts. One dot = one pitcher-season's version
of one pitch (2024-2026). No whole-arsenal pitcher map, by design.

  all pitches        pitcher x season x pitch type, global scaling (types separate)
  <type> universe    one t-SNE map per pitch type, fit on that type's rows alone
  (maps, comps and uniqueness all use the 15 pitch-identity measures, profile_embed.IDENTITY_BLOCKS:
   shape, derived and delivery without near-duplicates; location, scatter and usage are shown and
   filtered on but do not define "similar" -- user decision 2026-11-04, analysis/umap_benchmark.md)
  comps              10 nearest DISTINCT other pitchers in the PCA space each map
                     is drawn from (not the 2D map), any season

Levels. The maps, comps, percentiles and baselines are all MLB. Triple-A
pitchers (level "aaa", from the Triple-A store) are *placed* on the fixed MLB
maps -- a call-up who also has an MLB row of that pitch type is placed by the map's own t-SNE
transform (beside his big-league self); anyone else lands at the inverse-distance mean of his 15
nearest big-league pitches of the same type in the PCA space -- and their comps are MLB pitchers, so
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
  python scripts/atlas_build.py --cold-layout         # refit the maps from scratch (default: restart from the
                                                      # previous layout when it used the same measures)
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
from profile_embed import (IDENTITY_BLOCKS, TSNE_PERPLEXITY, TSNE_RESTART_ITERS, TYPE_BLOCKS, block_scale, embed,
                           tsne_layout, zscore)

SEASONS = profiles.SEASONS
RANGE = f"{min(SEASONS)}_{max(SEASONS)}"
LEVELS = ("mlb", "aaa")   # MLB first: it is the reference every other level is measured against
K_COMPS = 10
NBHD = 30          # neighbourhood stored per pitch: first K_COMPS are the displayed comps, the rest feed
                   # the "what goes with lower xERA among his lookalikes" analysis on the page
NBHD_LEVEL = {"mlb": NBHD, "aaa": K_COMPS}   # that analysis needs his own xERA, which Triple-A has none of
K_PLACE = 15       # MLB neighbours a Triple-A stranger is placed among (inverse-distance mean)
TEMPLATE = paths.ROOT / "scripts" / "atlas_template.html"
PITCHER_TYPES = paths.ROOT / "scripts" / "pitcher_types.json"   # "Find a type of pitcher" rules (editable)
OUT_HTML = paths.ROOT / "analysis" / "pitch_atlas.html"
ATLAS = paths.META / f"atlas_type_{RANGE}.parquet"
MOVEMENT = paths.META / f"movement_sample_{RANGE}.parquet"
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId={sport}&season={season}"
MOVE_PER_ROW = 50          # pitches sampled per pitcher x season x pitch type for the movement chart
LOCBB = paths.META / f"location_bb_{RANGE}.parquet"
# "Find a type of pitch": consensus clusters + subclusters within each pitch type (scripts/pitch_subclusters.py, MLB only);
# Triple-A pitches take the subcluster most of their MLB lookalikes are in. Of the hand-written pitcher types
# (PITCHER_TYPES) only these stay, as card tags and filters (user decision 2026-11-05).
PCL_PARQ = paths.META / f"pitch_subclusters_{RANGE}.parquet"
PCL_JSON = paths.META / f"pitch_subclusters_{RANGE}.json"
TAG_RULES = ("pronator", "supinator")
# Card heat map of where a pitch crosses the plate: a grid over HEAT_X x HEAT_Z (feet, catcher's view), HEAT_N cells a
# side, built from HEAT_FINE x HEAT_FINE finer cells smoothed with a Gaussian of HEAT_SIGMA ft. Batted-ball shares need
# BBE_MIN balls in play.
HEAT_X, HEAT_Z, HEAT_N, HEAT_FINE, HEAT_SIGMA = (-2.0, 2.0), (0.4, 4.4), 10, 4, 0.3
BBE_MIN = 10
HEAT_MIN = 15   # pitches against a batter hand before its map is drawn
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
# Maps (2D, 3D, clusters, Triple-A placement), comps and uniqueness are all built on what a pitch IS: shape,
# derived and delivery, near-duplicates dropped (15 measures). Usage left the maps on 2026-11-03; location and
# scatter left maps and comps, and usage left the comps, on 2026-11-04 (user decisions; the half-to-half
# reliability of those blocks is 0.6 to 0.86 against 0.92 to 0.97, and they drove 79% of a pitcher's season-to-
# season movement in the old comps space). Cards, filters, colours and the heat table still show all 32 measures.
MAP_BLOCKS = IDENTITY_BLOCKS
MAP_FEATURES = [c for cs in MAP_BLOCKS.values() for c in cs]
LAYOUT_META = paths.META / f"atlas_layout_{RANGE}.json"   # how the cached layout was made; a restart needs a match
EMBEDS = {}        # (map, dims) -> (openTSNE embedding fitted this run, its coordinates' mean as stored)


def tsne_place(key, Q):
    """Place rows Q (PCA space of that map) with the map's t-SNE transform, in the stored map's frame. openTSNE
    re-centres the fitted embedding in place on the first transform() (a pure translation; checked 2026-11-04),
    so the result is shifted back by however far the reference moved."""
    E, centre = EMBEDS[key]
    T = np.asarray(E.transform(Q), dtype="float64")
    return T + (centre - np.asarray(E, dtype="float64").mean(axis=0))


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


def free_agents_asof(season=max(SEASONS)):
    """The day the free-agent list was last fetched (the cached file's date), shown as "as of" on the page."""
    import datetime
    path = paths.META / f"free_agents_{season}.parquet"
    if not path.exists():
        return None
    return datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()


def data_through(level):
    """Last game date with pitches in the store at this level: the page's "data through" line."""
    last = None
    for season in level_seasons(level):
        m = pd.read_parquet(paths.meta_dir(level) / f"manifest_{season}.parquet")
        m = m[m.n_pitches > 0]
        if len(m):
            last = max(last or "", str(m.game_date.max()))
    return last


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


def _heat_grids(df, keys):
    """Smoothed HEAT_N x HEAT_N location grid per group of `keys`: catcher's view, scaled to the group's own peak,
    HEAT_N * HEAT_N digits 0-9, top row (high) first, left to right; pitches outside the window are left out."""
    from scipy.ndimage import gaussian_filter
    nf = HEAT_N * HEAT_FINE
    g = df.groupby(keys, observed=True, sort=True)
    out = g.size().rename("n").reset_index()
    code = g.ngroup().to_numpy()
    x, z = df.plate_x.to_numpy(dtype="float64"), df.plate_z.to_numpy(dtype="float64")
    ix = np.floor((x - HEAT_X[0]) / (HEAT_X[1] - HEAT_X[0]) * nf)
    iz = np.floor((z - HEAT_Z[0]) / (HEAT_Z[1] - HEAT_Z[0]) * nf)
    ok = np.isfinite(ix) & np.isfinite(iz) & (ix >= 0) & (ix < nf) & (iz >= 0) & (iz < nf)
    H = np.zeros((len(out), nf, nf), dtype="float32")
    np.add.at(H, (code[ok], iz[ok].astype(int), ix[ok].astype(int)), 1)
    out["n"] = np.bincount(code[ok], minlength=len(out))          # pitches inside the window
    H = gaussian_filter(H, sigma=(0, HEAT_SIGMA * nf / (HEAT_Z[1] - HEAT_Z[0]), HEAT_SIGMA * nf / (HEAT_X[1] - HEAT_X[0])),
                        mode="constant")
    B = H.reshape(len(out), HEAT_N, HEAT_FINE, HEAT_N, HEAT_FINE).mean(axis=(2, 4))[:, ::-1, :]   # top row = high
    peak = B.max(axis=(1, 2), keepdims=True)
    Q = (np.rint(9 * B / np.where(peak > 0, peak, 1)) + 48).astype("uint8")
    out["heat"] = [bytes(q.ravel()).decode() if pk > 0 else None for q, pk in zip(Q, peak.ravel())]
    return out


def location_bb(rebuild, level="mlb"):
    """Per pitcher x season x pitch type: where the pitch crosses the plate against left- and right-handed batters
    (two smoothed HEAT_N x HEAT_N grids, `_heat_grids`; None under HEAT_MIN pitches against that hand) and its
    batted-ball mix (ground-ball and fly-ball shares of balls in play, Statcast bb_type; NaN under BBE_MIN balls in
    play). Mean sz_top / sz_bot ride along so the page can draw the strike zone."""
    path = paths.meta_dir(level) / LOCBB.name
    if path.exists() and not rebuild:
        cached = pd.read_parquet(path)
        if {"heat_l", "heat_r"} <= set(cached.columns):   # older caches had one map for both hands
            return cached
    frames = []
    for season in level_seasons(level):
        df = store.query("SELECT pitcher, pitch_type, stand, plate_x, plate_z, sz_top, sz_bot, bb_type FROM pitches "
                         "WHERE pitch_type IS NOT NULL", season, level)
        df["pitch_type"] = df.pitch_type.astype(str).replace(profiles.TYPE_MAP)
        df = df[df.pitch_type.isin(profiles.PITCH_TYPES)].reset_index(drop=True)
        df["stand"] = df.stand.astype(str)
        bb = df.bb_type.astype(object)
        df["bbe"], df["gb"], df["fb"] = bb.notna(), bb == "ground_ball", bb == "fly_ball"
        k = ["pitcher", "pitch_type"]
        out = df.groupby(k, observed=True, sort=True).agg(
            n_loc=("plate_x", "count"), bbe=("bbe", "sum"), gb=("gb", "sum"), fb=("fb", "sum"),
            sz_top=("sz_top", "mean"), sz_bot=("sz_bot", "mean")).reset_index()
        hs = _heat_grids(df[df.stand.isin(["L", "R"])].reset_index(drop=True), k + ["stand"])
        for side in ("L", "R"):
            h = hs[hs.stand == side].drop(columns="stand")
            h.loc[h.n < HEAT_MIN, "heat"] = None
            out = out.merge(h.rename(columns={"n": f"heat_n{side.lower()}", "heat": f"heat_{side.lower()}"}), on=k, how="left")
        bbe = out.bbe.to_numpy(dtype="float64")
        out["gb_pct"] = np.where(bbe >= BBE_MIN, 100 * out.gb / np.maximum(bbe, 1), np.nan)
        out["fb_pct"] = np.where(bbe >= BBE_MIN, 100 * out.fb / np.maximum(bbe, 1), np.nan)
        out["season"] = season
        frames.append(out[["pitcher", "season", "pitch_type", "n_loc", "bbe", "gb_pct", "fb_pct", "sz_top", "sz_bot",
                           "heat_l", "heat_nl", "heat_r", "heat_nr"]])
        del df
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)
    log(f"location + batted balls {level}: {len(out):,} pitcher-season-types, {out.gb_pct.notna().mean():.0%} with >= {BBE_MIN} "
        f"balls in play; maps vs LHB {out.heat_l.notna().mean():.0%}, vs RHB {out.heat_r.notna().mean():.0%}")
    return out


def pitch_clusters(types, t_out):
    """Adds cl1 / cl2 (cluster and subcluster ids such as "FF-A", "FF-A1") and conf1 / conf2 (consensus confidence) to the
    exported type columns, and returns the definitions for the page. MLB rows come from pitch_subclusters.py; Triple-A rows
    (and MLB rows it did not see) take the subcluster most of their MLB lookalikes are in, with that share as confidence."""
    if not (PCL_PARQ.exists() and PCL_JSON.exists()):
        return None
    from collections import Counter
    raw = json.loads(PCL_JSON.read_text())
    defs, core = raw.get("types", raw), raw.get("core", 0.6)
    A = pd.read_parquet(PCL_PARQ).astype({"pitcher": "int64", "season": "int64"})
    k = ["pitcher", "season", "pitch_type"]
    sub = A.set_index(k).reindex(pd.MultiIndex.from_frame(types[k].astype({"pitcher": "int64", "season": "int64"})))
    mlb = (types.level == "mlb").to_numpy()
    cl1 = np.where(mlb, sub.cl1.to_numpy(dtype=object), None)
    cl2 = np.where(mlb, sub.cl2.to_numpy(dtype=object), None)
    conf1 = np.where(mlb, sub.conf1.to_numpy(dtype="float64"), np.nan)
    conf2 = np.where(mlb, sub.conf2.to_numpy(dtype="float64"), np.nan)
    cr = np.stack(types.comp_rows.to_numpy())
    placed = 0
    for i in np.flatnonzero(pd.isna(cl2)):
        labs = [(cl1[j], cl2[j]) for j in cr[i] if j >= 0 and mlb[j] and cl2[j] is not None]
        if not labs:
            continue
        (c1, c2), n = Counter(labs).most_common(1)[0]
        cl1[i], cl2[i], conf2[i] = c1, c2, n / len(labs)
        conf1[i] = sum(a == c1 for a, _ in labs) / len(labs)
        placed += 1
    t_out["cl1"], t_out["cl2"], t_out["conf1"], t_out["conf2"] = cl1, cl2, conf1, conf2
    log(f"pitch clusters: {pd.notna(cl1[mlb]).mean():.1%} of MLB rows from pitch_subclusters, {placed:,} rows placed by their "
        f"MLB lookalikes; {sum(len(d['clusters']) for d in defs.values())} clusters, "
        f"{sum(len(c['subs']) for d in defs.values() for c in d['clusters'])} subclusters")
    keep = ("id", "letter", "name", "n", "n_core", "numbers")
    return {"core": core, "types": {pt: {"name": d["name"], "n": d["n"], "k": d["k"], "repro": round(d["repro"], 2),
                                         "clusters": [{**{x: c[x] for x in keep}, "subs": [{x: s_[x] for x in keep if x in s_} for s_ in c["subs"]]}
                                                      for c in d["clusters"]]}
                                    for pt, d in defs.items()}}


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


def scaler_fit(F, by_type=None, blocks=TYPE_BLOCKS):
    """The feature scaling of one map, fitted on MLB rows so another level can be put in the same space.

    Reproduces block_scale(zscore(F.fillna(median)), blocks) exactly for the rows it was fitted on. `by_type`
    (those rows' pitch types) makes the fill median per type, as the all-pitches map does.
    """
    F = F.astype("float64")
    med = F.median() if by_type is None else F.groupby(np.asarray(by_type)).median()
    Ff = _fill(F, med, by_type)
    mu, sd = Ff.mean(), Ff.std(ddof=0).replace(0, 1)
    Z = (Ff - mu) / sd
    fac = {}
    for name, cols in blocks.items():
        tot = np.nansum(np.nanvar(Z[cols].to_numpy(dtype="float64"), axis=0))
        fac[name] = np.sqrt(tot if tot > 0 else 1)
    return {"med": med, "mu": mu, "sd": sd, "fac": fac, "blocks": blocks}


def scaler_apply(sc, F, by_type=None):
    Z = (_fill(F.astype("float64"), sc["med"], by_type) - sc["mu"]) / sc["sd"]
    return np.nan_to_num(np.hstack([Z[cols].to_numpy(dtype="float64") / sc["fac"][n] for n, cols in sc["blocks"].items()]))


def place_level(types_mlb, space_mlb, types_new, cached=None):
    """Put another level's pitches on the fixed MLB maps. Nothing MLB moves.

    A call-up who also has an MLB row of the same pitch type (any season) is placed by the map's own t-SNE
    transform, fitted in this run (EMBEDS): on the 2026-11-04 benchmark that puts a pitcher's new data beside his
    old row far more often than any average does (71% vs 29-36% in 3D). Everyone else (a stranger) lands at the
    inverse-distance-weighted mean position of his K_PLACE nearest MLB pitches of the same type in the PCA space
    the map was drawn from (42% in 3D, level with every transform). Clusters always come from the weighted
    majority of those K_PLACE neighbours. With --html-only there are no embeddings: rows in `cached` (this level's
    previous placements) keep them, new rows are placed as strangers.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    types_mlb, space_mlb = types_mlb.reset_index(drop=True), space_mlb.reset_index(drop=True)
    types_new = types_new.reset_index(drop=True)
    assert (types_mlb.pitcher.to_numpy() == space_mlb.pitcher.to_numpy()).all()
    G2, G3, T2, T3 = ["gx", "gy"], ["gx3", "gy3", "gz3"], ["tx", "ty"], ["tx3", "ty3", "tz3"]
    out = types_new[profiles.KEYS + ["pitch_type"]].copy()
    for c in G2 + G3 + T2 + T3:
        out[c] = np.nan
    out["tcluster"] = -1
    known = set(zip(types_mlb.pitcher, types_mlb.pitch_type))
    callup = np.array([(p, t) in known for p, t in zip(types_new.pitcher, types_new.pitch_type)])
    counts = {"transform": 0, "cached": 0, "IDW": 0}

    def land(P_fit, P_new, rows_fit, rows_new, groups, cluster):
        """groups: [(cols, embedding key)]; IDW everywhere first, then transform for call-ups when fitted."""
        d, i = NearestNeighbors(n_neighbors=min(K_PLACE, len(rows_fit))).fit(P_fit).kneighbors(P_new)
        w = 1 / (d + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        for cols, key in groups:
            for c in cols:
                out.loc[rows_new, c] = (w * space_mlb[c].to_numpy(dtype="float64")[rows_fit][i]).sum(axis=1)
            cu = callup[rows_new]
            if key in EMBEDS and cu.any():
                out.loc[rows_new[cu], cols] = tsne_place(key, P_new[cu])
        if cluster:
            cl = space_mlb.tcluster.to_numpy()[rows_fit][i]
            out.loc[rows_new, "tcluster"] = [int(np.bincount(r[r >= 0], weights=ww[r >= 0]).argmax()) if (r >= 0).any() else -1
                                             for r, ww in zip(cl, w)]

    Fm, Fn = types_mlb[MAP_FEATURES], types_new[MAP_FEATURES]
    gm, gn = types_mlb.groupby("pitch_type").indices, types_new.groupby("pitch_type").indices
    # all-pitches map: one global scaling (as fitted), same-type neighbours for the average; the transform uses
    # the global embedding, which was fitted on every MLB row in this same PCA space
    sc = scaler_fit(Fm, types_mlb.pitch_type, MAP_BLOCKS)
    Xm = scaler_apply(sc, Fm, types_mlb.pitch_type)
    pca = PCA(n_components=min(30, Xm.shape[1]), random_state=0).fit(Xm)
    Pm, Pn = pca.transform(Xm), pca.transform(scaler_apply(sc, Fn, types_new.pitch_type))
    for pt, rn in gn.items():
        if pt not in gm:
            continue
        land(Pm[gm[pt]], Pn[rn], gm[pt], rn, [(G2, None), (G3, None)], False)
    if ("global", 2) in EMBEDS and callup.any():   # transform against the whole global map in one call
        for cols, key in [(G2, ("global", 2)), (G3, ("global", 3))]:
            out.loc[callup, cols] = tsne_place(key, Pn[callup])
    # each type's map: that type's own scaling and PCA
    for pt, rn in gn.items():
        if pt not in gm:
            continue
        rm = gm[pt]
        sc = scaler_fit(Fm.iloc[rm], blocks=MAP_BLOCKS)
        Xm = scaler_apply(sc, Fm.iloc[rm])
        pca = PCA(n_components=min(30, Xm.shape[1]), random_state=0).fit(Xm)
        land(pca.transform(Xm), pca.transform(scaler_apply(sc, Fn.iloc[rn])), rm, rn, [(T2, (pt, 2)), (T3, (pt, 3))], True)
    if EMBEDS:
        counts["transform"], counts["IDW"] = int(callup.sum()), int((~callup).sum())
    elif cached is not None and len(cached):
        c = cached.set_index(["pitcher", "season", "pitch_type"])
        idx = pd.MultiIndex.from_frame(out[["pitcher", "season", "pitch_type"]])
        hit = idx.isin(c.index)
        cols = [x for x in G2 + G3 + T2 + T3 + ["tcluster"] if x in c.columns]
        out.loc[hit, cols] = c.reindex(idx[hit])[cols].to_numpy()
        counts["cached"], counts["IDW"] = int(hit.sum()), int((~hit).sum())
    else:
        counts["IDW"] = len(out)
    log(f"placed {len(out):,} pitches on the MLB maps: {counts['transform']:,} call-ups by t-SNE transform, "
        f"{counts['cached']:,} kept from the cached build, {counts['IDW']:,} among their {K_PLACE} nearest big-league pitches")
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
        sc = scaler_fit(types.loc[mrows, MAP_FEATURES], blocks=MAP_BLOCKS)
        pca = PCA(n_components=min(30, len(MAP_FEATURES)), random_state=0)
        Pc = np.zeros((len(rows), pca.n_components))
        Pc[is_m] = pca.fit_transform(scaler_apply(sc, types.loc[mrows, MAP_FEATURES]))
        if (~is_m).any():
            Pc[~is_m] = pca.transform(scaler_apply(sc, types.loc[rows[~is_m], MAP_FEATURES]))
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


def global_space(types):
    """The all-pitches map's input: per-type median fill, z over all rows (pitch types separate), block-scaled.
    place_level's scaler_fit(F, by_type, MAP_BLOCKS) reproduces it exactly."""
    F = types[MAP_FEATURES].astype("float64")
    F = F.fillna(F.groupby(types.pitch_type).transform("median"))
    return block_scale(zscore(F), MAP_BLOCKS)


def type_input(types, rows):
    """One pitch type's map input: within-type median fill, z within type, block-scaled."""
    F = types.loc[rows, MAP_FEATURES].astype("float64")
    return block_scale(zscore(F.fillna(F.median())), MAP_BLOCKS)


def pcs(X):
    from sklearn.decomposition import PCA
    return PCA(n_components=min(30, X.shape[1]), random_state=0).fit_transform(X)


def layout_signature():
    return {"method": "tsne", "perplexity": TSNE_PERPLEXITY, "features": MAP_FEATURES}


def previous_layout(cold):
    """The cached MLB coordinates, keyed by pitcher/season/pitch type, if they were laid out the same way."""
    if cold or not ATLAS.exists() or not LAYOUT_META.exists():
        return None
    if json.loads(LAYOUT_META.read_text()) != layout_signature():
        log("cached layout used other measures or method: fitting the maps from scratch")
        return None
    a = pd.read_parquet(ATLAS)
    if "level" in a:
        a = a[a.level == "mlb"]
    return a.drop_duplicates(["pitcher", "season", "pitch_type"]).set_index(["pitcher", "season", "pitch_type"])


def restart_from(prev, keys, cols, P):
    """Start coordinates for a restart: each row's previous position; a row new since then starts at the
    inverse-distance mean of its K_PLACE nearest previous rows in P. None (a cold fit) without a usable cache."""
    from sklearn.neighbors import NearestNeighbors
    if prev is None or not set(cols) <= set(prev.columns):
        return None
    old = prev.reindex(pd.MultiIndex.from_frame(keys[["pitcher", "season", "pitch_type"]]))[cols].to_numpy(dtype="float64")
    have = ~np.isnan(old).any(axis=1)
    if have.sum() < max(K_PLACE, 0.5 * len(old)):
        return None
    init = old.copy()
    if (~have).any():
        d, i = NearestNeighbors(n_neighbors=K_PLACE).fit(P[have]).kneighbors(P[~have])
        w = 1 / (d + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        init[~have] = (w[:, :, None] * old[have][i]).sum(axis=1)
    return init


def layout_3d(types, prev=None):
    """3D t-SNE coordinates on the same PCA spaces as the 2D maps. Restarts from `prev` when given."""
    types = types.reset_index(drop=True)
    out = pd.DataFrame(index=types.index, columns=["gx3", "gy3", "gz3", "tx3", "ty3", "tz3"], dtype="float64")
    P = pcs(global_space(types))
    init = restart_from(prev, types, ["gx3", "gy3", "gz3"], P)
    U, E = tsne_layout(P, 3, init)
    out[["gx3", "gy3", "gz3"]], EMBEDS[("global", 3)] = U, (E, U.mean(axis=0))
    log(f"t-SNE 3d: all pitches done ({'restart' if init is not None else 'cold'})")
    for pt, rows in types.groupby("pitch_type").indices.items():
        Pt = pcs(type_input(types, rows))
        init = restart_from(prev, types.loc[rows], ["tx3", "ty3", "tz3"], Pt)
        U, E = tsne_layout(Pt, 3, init)
        EMBEDS[(pt, 3)] = (E, U.mean(axis=0))
        out.loc[rows, ["tx3", "ty3", "tz3"]] = U
        log(f"t-SNE 3d: {pt} done ({'restart' if init is not None else 'cold'})")
    return out


def build_type_space(types, prev=None):
    """2D t-SNE maps (all pitches + one per type), Leiden islands per type. Restarts from `prev` when given."""
    types = types.reset_index(drop=True)
    X = global_space(types)
    _, _, _, U = embed(X, "type/global", "tsne", restart_from(prev, types, ["gx", "gy"], pcs(X)))
    EMBEDS[("global", 2)] = (embed.last, U.mean(axis=0))
    out = types[profiles.KEYS + ["pitch_type"]].copy()
    out["gx"], out["gy"] = U[:, 0], U[:, 1]
    out["tx"] = out["ty"] = np.nan
    out["tcluster"] = -1
    comp_rows = np.full((len(types), K_COMPS), -1)
    comp_pct = np.full((len(types), K_COMPS), np.nan)
    pid = types.pitcher.to_numpy()
    for pt, rows in types.groupby("pitch_type").indices.items():
        X = type_input(types, rows)
        P_sub, _, leiden, U = embed(X, f"type/{pt}", "tsne", restart_from(prev, types.loc[rows], ["tx", "ty"], pcs(X)))
        EMBEDS[(pt, 2)] = (embed.last, U.mean(axis=0))
        out.loc[rows, ["tx", "ty"]] = U
        out.loc[rows, "tcluster"] = leiden
        P = np.zeros((len(types), P_sub.shape[1]))
        P[rows] = P_sub
        ci, cp = comps_for(P, pid, rows)
        comp_rows[rows], comp_pct[rows] = ci, cp
    out["comp_rows"], out["comp_pct"] = list(comp_rows), list(comp_pct)
    return out



# -------------------------------------------------------------------- export
LQ_K = 10   # neighbourhood size for "true neighbours kept", as in analysis/umap_benchmark.md (Table 3)


def layout_quality(types, space):
    """True neighbours kept (the benchmark's self-free recall@10), for the page's "Information projected" line: of each
    pitch's ten nearest pitches in the space a map is built from (global_space for all pitches, type_input per type),
    leaving out the same pitcher's other seasons, the share that are also among its ten nearest dots on the map. Scored
    for the t-SNE layouts (2D and 3D) and for the movement chart as the page draws it (arm-side HB x IVB, inches).
    `types` and `space` are the MLB rows, in the same order."""
    from sklearn.neighbors import NearestNeighbors

    pid = types.pitcher.to_numpy()

    def near_free(X, ids, k=LQ_K):
        kq = min(len(X) - 1, k + 12)
        nb = NearestNeighbors(n_neighbors=kq + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
        out = np.full((len(X), k), -1)
        for r in range(len(X)):
            o = nb[r][ids[nb[r]] != ids[r]][:k]
            out[r, :len(o)] = o
        return out

    def kept(a, b):
        return round(float(np.mean([len(set(x[x >= 0]) & set(y[y >= 0])) / max(1, (x >= 0).sum()) for x, y in zip(a, b)])), 3)

    mv = pd.DataFrame({"hb": -12 * types.pfx_x, "ivb": 12 * types.pfx_z})
    mv = mv.fillna(mv.groupby(types.pitch_type.to_numpy()).transform("median")).to_numpy()

    def score(rows, X, c2, c3):
        ids = pid[rows]
        truth = near_free(np.asarray(X), ids)
        out = {"tsne": kept(truth, near_free(space.loc[rows, c2].to_numpy(), ids)), "move": kept(truth, near_free(mv[rows], ids))}
        if set(c3) <= set(space.columns):
            out["tsne3"] = kept(truth, near_free(space.loc[rows, c3].to_numpy(), ids))
        return out

    allrows = np.arange(len(types))
    maps = {"all": score(allrows, global_space(types), ["gx", "gy"], ["gx3", "gy3", "gz3"])}
    for pt, rows in types.groupby("pitch_type").indices.items():
        if len(rows) > LQ_K + 13:
            maps[pt] = score(rows, type_input(types, rows), ["tx", "ty"], ["tx3", "ty3", "tz3"])
    a = maps["all"]
    log(f"true neighbours kept (k={LQ_K}, self-free), all pitches: t-SNE 2D {a['tsne']:.0%}, 3D {a.get('tsne3', float('nan')):.0%}, "
        f"movement chart {a['move']:.0%}")
    return {"k": LQ_K, "maps": maps}


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
    d["uniq_z"] = uniqueness(t)      # SDs from the typical pitch of its type (the 15 identity measures)
    return d


def look_deltas(types):
    """Per pitch: how he differs from his lookalikes (same type, same hand, similar arm angle, nearest on the
    identity measures; up to 30 for MLB pitches, 10 MLB lookalikes for Triple-A): whiff rate (points), location
    height (inches) and zone rate (points), his value minus their mean. Same pitch, different location: on the
    2026-11-04 benchmark the location gap explains the whiff gap twice as well among identity lookalikes as it
    did when location was part of "similar". NaN with fewer than 5 lookalikes."""
    cr = np.stack(types.comp_rows.to_numpy())
    valid = cr >= 0
    out = pd.DataFrame(index=types.index)
    for name, v in [("whiff_vs_look", 100 * types.whiff_pct), ("loc_height_vs_look", 12 * types.plate_z),
                    ("zone_vs_look", 100 * types.in_zone)]:
        v = v.to_numpy(dtype="float64")
        M = np.where(valid, v[np.clip(cr, 0, None)], np.nan)
        ok = (~np.isnan(M)).sum(axis=1) >= 5
        out[name] = np.where(ok, v - np.nanmean(np.where(ok[:, None], M, 0.0), axis=1), np.nan)
    return out


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
        sc = scaler_fit(types.loc[mrows, MAP_FEATURES], blocks=MAP_BLOCKS)
        Xm = scaler_apply(sc, types.loc[mrows, MAP_FEATURES])
        centre = np.median(Xm, axis=0)
        dm = np.linalg.norm(Xm - centre, axis=1)
        d = np.linalg.norm(scaler_apply(sc, types.loc[rows, MAP_FEATURES]) - centre, axis=1)
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


def export(types, pitchers, t_space, lead, fa, spin, mv, tj, teams, levels, teams_aaa, quality=None, lb=None):
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
    t_out = pd.concat([t_out, type_display(types), look_deltas(types)], axis=1)
    pcl = pitch_clusters(types, t_out)
    heat_grid = None
    if lb is not None:   # location heat map + batted-ball mix (location_bb), by key, in the types' order
        k = ["level", "pitcher", "season", "pitch_type"]
        lb = lb.astype({"pitcher": "int64", "season": "int64"})
        sub = lb.set_index(k).reindex(pd.MultiIndex.from_frame(types[k].astype({"pitcher": "int64", "season": "int64"})))
        for c in ("gb_pct", "fb_pct", "bbe", "heat_nl", "heat_nr"):
            t_out[c] = sub[c].to_numpy(dtype="float64")
        for c in ("heat_l", "heat_r"):
            t_out[c] = sub[c].to_numpy(dtype=object)
        m = lb[lb.level == "mlb"].dropna(subset=["sz_top", "sz_bot"])
        w = m.n_loc.to_numpy(dtype="float64")
        heat_grid = {"x": list(HEAT_X), "z": list(HEAT_Z), "n": HEAT_N,
                     "zone": [round(float(np.average(m.sz_bot, weights=w)), 2), round(float(np.average(m.sz_top, weights=w)), 2)]}
        log(f"heat maps vs LHB for {pd.notna(t_out.heat_l).mean():.1%} and vs RHB for {pd.notna(t_out.heat_r).mean():.1%} of pitch rows; "
            f"GB/FB for {t_out.gb_pct.notna().mean():.1%}; "
            f"strike zone {heat_grid['zone']} ft")
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
        "archetypes": [r for r in json.loads(PITCHER_TYPES.read_text()) if r.get("id") in TAG_RULES] if PITCHER_TYPES.exists() else [],
        **({"pitch_clusters": pcl} if pcl else {}),
        **({"layout_quality": quality} if quality else {}),
        **({"heat_grid": heat_grid} if heat_grid else {}),
        "fa_season": max(SEASONS),
        "fa_asof": free_agents_asof(),
        "data_through": {l: data_through(l) for l in levels},
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
    ap.add_argument("--cold-layout", action="store_true", help="fit the maps from scratch instead of restarting")
    a = ap.parse_args()

    types, pitchers = season_profiles(a.rebuild_profiles, "mlb")
    lead = pd.concat([leaders(s, a.refetch_leaders) for s in SEASONS], ignore_index=True)
    cached_levels = {}
    if a.html_only and ATLAS.exists():
        t_space = pd.read_parquet(ATLAS)
        if "level" in t_space:   # other levels: their cached placements are kept for rows that still exist
            cached_levels = {l: g.drop(columns="level") for l, g in t_space[t_space.level != "mlb"].groupby("level")}
            t_space = t_space[t_space.level == "mlb"].drop(columns="level").reset_index(drop=True)
        prev = None
    else:
        prev = previous_layout(a.cold_layout)
        t_space = build_type_space(types, prev)
    # 3D layouts: computed with the 2D ones and cached in the atlas table
    if not {"gx3", "tx3"} <= set(t_space.columns):
        u3 = layout_3d(types, prev)
        for c in u3.columns:
            t_space[c] = u3[c].to_numpy()

    # how much each map keeps of the true neighbourhoods (the page's "Information projected" line)
    quality = layout_quality(types.reset_index(drop=True), t_space.reset_index(drop=True))

    # other levels ride on the MLB maps: placed among their nearest big-league pitches, never refitted
    levels = ["mlb"] + [l for l in LEVELS if l != "mlb" and level_seasons(l)]
    types, pitchers, t_space = [df.reset_index(drop=True).assign(level="mlb") for df in (types, pitchers, t_space)]
    parts = {"types": [types], "pitchers": [pitchers], "space": [t_space], "mv": [movement_sample(a.rebuild_profiles)],
             "tj": [trajectory_means(a.rebuild_profiles)], "teams": [pitcher_teams()], "lb": [location_bb(a.rebuild_profiles)]}
    teams_aaa = {}
    for level in levels[1:]:
        t_l, p_l = season_profiles(a.rebuild_profiles, level)
        parts["types"].append(t_l.assign(level=level)); parts["pitchers"].append(p_l.assign(level=level))
        parts["space"].append(place_level(types, t_space, t_l, cached_levels.get(level)).assign(level=level))
        parts["mv"].append(movement_sample(a.rebuild_profiles, level))
        parts["tj"].append(trajectory_means(a.rebuild_profiles, level))
        parts["lb"].append(location_bb(a.rebuild_profiles, level))
        parts["teams"].append(pitcher_teams(level))
        if level == "aaa":
            teams_aaa = minor_teams(level)
    types, pitchers, t_space, teams = [pd.concat(parts[k], ignore_index=True) for k in ("types", "pitchers", "space", "teams")]
    mv = pd.concat([m.assign(level=l) for m, l in zip(parts["mv"], levels)], ignore_index=True)
    tj = pd.concat([t.assign(level=l) for t, l in zip(parts["tj"], levels)], ignore_index=True)
    lb = pd.concat([x.assign(level=l) for x, l in zip(parts["lb"], levels)], ignore_index=True)

    # comps are cheap and rule-driven (same hand, similar arm angle): always recompute
    cr, cp, ctol = type_comps(types)
    t_space["comp_rows"], t_space["comp_pct"], t_space["comp_tol"] = list(cr), list(cp), ctol
    # every pitch, MLB or Triple-A, also gets its lookalikes among Triple-A pitchers (the page offers both:
    # "MLB similar | AAA similar"), so an MLB arm's Triple-A doppelgangers can be saved and followed to that map
    if "aaa" in levels:
        cr, cp, ctol = type_comps(types, None, "aaa")
        t_space["comp_rows_aaa"], t_space["comp_pct_aaa"], t_space["comp_tol_aaa"] = list(cr), list(cp), ctol
    t_space.to_parquet(ATLAS, index=False)
    if not a.html_only:
        LAYOUT_META.write_text(json.dumps(layout_signature(), indent=1))
    spin = pd.concat([active_spin(s, a.refetch_leaders) for s in SEASONS], ignore_index=True)
    export(types, pitchers, t_space, lead, free_agents(), spin, mv, tj, teams, levels, teams_aaa, quality, lb)


if __name__ == "__main__":
    main()
