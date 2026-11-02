"""Interactive pitch atlas: pitcher-season profiles -> embeddings -> comps -> one HTML page.

Exploratory only -- nothing here predicts. One dot = one pitcher-season's version
of one pitch (2024-2026). No whole-arsenal pitcher map, by design.

  all pitches        pitcher x season x pitch type, global scaling (types separate)
  <type> universe    one UMAP per pitch type, fit on that type's rows alone
  comps              10 nearest DISTINCT other pitchers in the PCA space each map
                     is drawn from (not the 2D map), any season

Outcomes shown beside comps come from Savant's custom leaderboard (official
xERA, ERA, IP, K%, BB%, xwOBA, whiff%), cached in data/meta/savant_leaders_<season>.parquet.

Free agents: MLB Stats API /people/freeAgents for the offseason after the latest
season (official; former team, declared date, and once signed: new team, date,
contract length -- no dollar values). Refetched on every run so signings show up;
falls back to data/meta/free_agents_<season>.parquet if the request fails.

Outputs:
  data/meta/profiles_{type,pitcher}_season_2024_2026.parquet
  data/meta/atlas_type_2024_2026.parquet             coords + comps, for Python use
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
K_COMPS = 10
NBHD = 30          # neighbourhood stored per pitch: first K_COMPS are the displayed comps, the rest feed
                   # the "what goes with lower xERA among his lookalikes" analysis on the page
TEMPLATE = paths.ROOT / "scripts" / "atlas_template.html"
PITCHER_TYPES = paths.ROOT / "scripts" / "pitcher_types.json"   # "Find a type of pitcher" rules (editable)
OUT_HTML = paths.ROOT / "analysis" / "pitch_atlas.html"
ATLAS = paths.META / f"atlas_type_{RANGE}.parquet"
MOVEMENT = paths.META / f"movement_sample_{RANGE}.parquet"
MOVE_PER_ROW = 50          # pitches sampled per pitcher x season x pitch type for the movement chart
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
def season_profiles(rebuild):
    p_type = profiles.profiles_path("type", SEASONS, by_half=False)
    p_pitch = profiles.profiles_path("pitcher", SEASONS, by_half=False)
    if rebuild or not (p_type.exists() and p_pitch.exists()):
        types, usage_all = profiles.build_type_profiles(SEASONS, by_half=False)
        pitchers = profiles.build_pitcher_profiles(types, usage_all)
        types.to_parquet(p_type, index=False)
        pitchers.to_parquet(p_pitch, index=False)
        log(f"profiles: {len(types):,} type rows, {len(pitchers):,} pitcher rows")
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


def movement_sample(rebuild):
    """Up to MOVE_PER_ROW pitches per pitcher x season x pitch type: raw (catcher's view, not mirrored)
    horizontal and induced vertical break in inches, for the Savant-style movement chart."""
    if MOVEMENT.exists() and not rebuild:
        return pd.read_parquet(MOVEMENT)
    frames = []
    for season in SEASONS:
        df = store.query("SELECT pitcher, pitch_type, pfx_x, pfx_z FROM pitches "
                         "WHERE pitch_type IS NOT NULL AND pfx_x IS NOT NULL AND pfx_z IS NOT NULL", season)
        df["pitch_type"] = df.pitch_type.astype(str).replace(profiles.TYPE_MAP)
        df = df[df.pitch_type.isin(profiles.PITCH_TYPES)]
        df = df.sample(frac=1, random_state=0).groupby(["pitcher", "pitch_type"]).head(MOVE_PER_ROW)
        df["season"] = season
        df["mx"] = (24 * df.pfx_x).round().clip(-127, 127).astype("int8")   # half-inch units
        df["mz"] = (24 * df.pfx_z).round().clip(-127, 127).astype("int8")
        frames.append(df[["pitcher", "season", "pitch_type", "mx", "mz"]])
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(MOVEMENT, index=False)
    log(f"movement sample: {len(out):,} pitches")
    return out


def movement_blob(types, mv):
    """Per type row (in export order): count, then all x then all z as int8, base64."""
    g = {k: v for k, v in mv.groupby(["pitcher", "season", "pitch_type"])}
    counts, xs, zs = [], [], []
    for r in types[["pitcher", "season", "pitch_type"]].itertuples(index=False):
        part = g.get((r.pitcher, r.season, r.pitch_type))
        n = 0 if part is None else len(part)
        counts.append(n)
        if n:
            xs.append(part.mx.to_numpy()); zs.append(part.mz.to_numpy())
    x = np.concatenate(xs).astype("int8"); z = np.concatenate(zs).astype("int8")
    return {"counts": counts, "b64": base64.b64encode(np.concatenate([x, z]).tobytes()).decode()}


def pitcher_teams():
    """Team(s) per pitcher-season, in order of first appearance ("NYY,CHC" if traded),
    plus team_main = the team he threw the most pitches for (used for colour).

    The pitching team is the home team in the top of an inning. OAK (2024) is
    folded into ATH so the franchise filters as one team.
    """
    frames = []
    for season in SEASONS:
        df = store.query(
            "SELECT pitcher, CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS team, "
            "min(game_date) AS first, count(*) AS n FROM pitches GROUP BY 1, 2", season)
        df["season"] = season
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["team"] = df.team.replace({"OAK": "ATH"})
    df = df.groupby(["pitcher", "season", "team"], as_index=False).agg(first=("first", "min"), n=("n", "sum"))
    g = df.sort_values("first").groupby(["pitcher", "season"])
    out = g.team.agg(",".join).rename("teams").to_frame()
    out["team_main"] = df.sort_values("n").groupby(["pitcher", "season"]).team.last()
    return out.reset_index()


# ----------------------------------------------------------------- embedding
def comps_for(P, pid, rows=None):
    """k nearest distinct other pitchers; indices returned in the caller's frame."""
    rows = np.arange(len(P)) if rows is None else rows
    i, d = other_pitcher_neighbours(P[rows], pid[rows], K_COMPS)
    gi = np.where(i >= 0, rows[np.clip(i, 0, None)], -1)
    first = np.sort(d[:, 0][~np.isnan(d[:, 0])])
    pct = np.searchsorted(first, np.nan_to_num(d, nan=np.inf)) / len(first)
    return gi, np.where(np.isnan(d), np.nan, pct)


ARM_TOLS = (7.5, 10, 12.5, 15, 20, 30, np.inf)   # arm-angle window, widened only when too few comps


def type_comps(types):
    """Comps per pitch: same pitch type, SAME HAND, arm angle within ARM_TOLS[0] degrees (the window
    widens step by step only when fewer than K_COMPS distinct other pitchers qualify), ranked by
    distance in that type's PCA space (the same space the per-type maps are drawn from).
    Returns comp rows, fit percentile, and the arm-angle window actually used per row."""
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
    for pt, rows in types.groupby("pitch_type").indices.items():
        F = types.loc[rows, TYPE_FEATURES].astype("float64")
        X = block_scale(zscore(F.fillna(F.median())), TYPE_BLOCKS)
        Pc = PCA(n_components=min(30, X.shape[1]), random_state=0).fit_transform(X)
        for h in np.unique(hand[rows]):
            local = np.flatnonzero(hand[rows] == h)
            g = rows[local]
            if len(g) < 2:
                continue
            D = cdist(Pc[local], Pc[local])
            for a in range(len(g)):
                order = np.argsort(D[a])
                darm = np.abs(arm[g] - arm[g[a]])
                for tol in ARM_TOLS:
                    ok = order[(order != a) & (np.isinf(tol) | (darm[order] <= tol))]
                    seen, picks = {pid[g[a]]}, []
                    for b in ok:
                        if pid[g[b]] in seen:
                            continue
                        seen.add(pid[g[b]]); picks.append(b)
                        if len(picks) == K_COMPS:
                            break
                    if len(picks) == K_COMPS or np.isinf(tol):
                        break
                # extend to the NBHD neighbourhood without touching the first K_COMPS: same window first,
                # widening further only if needed; appended in distance order within each window
                for tol2 in [t2 for t2 in ARM_TOLS if t2 >= tol]:
                    if len(picks) >= NBHD:
                        break
                    ok = order[(order != a) & (np.isinf(tol2) | (darm[order] <= tol2))]
                    for b in ok:
                        if pid[g[b]] in seen:
                            continue
                        seen.add(pid[g[b]]); picks.append(b)
                        if len(picks) == NBHD:
                            break
                rows_out[g[a], :len(picks)] = g[picks]
                dist_out[g[a], :len(picks)] = D[a, picks]
                tol_out[g[a]] = tol
    first = np.sort(dist_out[:, 0][~np.isnan(dist_out[:, 0])])
    pct = np.searchsorted(first, np.nan_to_num(dist_out, nan=np.inf)) / len(first)
    widened = np.mean(tol_out > ARM_TOLS[0])
    log(f"comps: same hand, arm angle within {ARM_TOLS[0]} deg ({100 * widened:.1f}% of pitches needed a wider window)")
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
    """
    out = pd.Series(np.nan, index=types.index)
    for pt, rows in types.groupby("pitch_type").groups.items():
        F = types.loc[rows, TYPE_FEATURES].astype("float64")
        X = block_scale(zscore(F.fillna(F.median())), TYPE_BLOCKS)
        d = np.linalg.norm(X - np.median(X, axis=0), axis=1)
        out.loc[rows] = (d - d.mean()) / d.std()
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


def export(types, pitchers, t_space, lead, fa, spin, mv):
    key = profiles.KEYS
    pitchers = pitchers.merge(lead, on=["pitcher", "season"], how="left").merge(
        pitcher_teams(), on=["pitcher", "season"], how="left")
    types = types.merge(t_space, on=key + ["pitch_type"], how="inner").merge(
        spin, on=["pitcher", "season", "pitch_type"], how="left")
    # comp_rows index t_space rows; the merge kept its order (inner on unique keys)
    assert len(types) == len(t_space)
    assert (types.pitcher.to_numpy() == t_space.pitcher.to_numpy()).all()

    t_out = types[["pitcher", "season", "pitch_type", "p_throws", "n", "gx", "gy", "tx", "ty",
                   "tcluster", "comp_tol", "gx3", "gy3", "gz3", "tx3", "ty3", "tz3"]].copy()
    t_out = pd.concat([t_out, type_display(types)], axis=1)
    p_out = pitchers[["pitcher", "season", "p_throws", "n", "primary_fb", "teams", "team_main"]].copy()
    p_out = pd.concat([p_out, pitcher_display(pitchers)], axis=1)
    p_out["name"] = pitchers.player_name.map(first_last)
    t_out["name"] = types.player_name.map(first_last)
    log(f"Savant outcomes matched for {pitchers.xera.notna().mean():.1%} of pitcher-seasons; "
        f"team for {pitchers.teams.notna().mean():.1%}; "
        f"{fa.pitcher.isin(pitchers.pitcher).sum()} free agents appear in the atlas; "
        f"spin efficiency for {types.spin_eff.notna().mean():.1%} of pitch rows")

    data = {
        "seasons": list(SEASONS),
        "pitchers": columnar(p_out),
        "types": columnar(t_out),
        "tcomps": comps_columnar(types.comp_rows, types.comp_pct),
        "movement": movement_blob(types, mv),
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

    types, pitchers = season_profiles(a.rebuild_profiles)
    lead = pd.concat([leaders(s, a.refetch_leaders) for s in SEASONS], ignore_index=True)
    if a.html_only and ATLAS.exists():
        t_space = pd.read_parquet(ATLAS)
    else:
        t_space = build_type_space(types)
    # 3D layouts: computed once and cached in the atlas table
    if not {"gx3", "tx3"} <= set(t_space.columns):
        u3 = umap_3d(types)
        for c in u3.columns:
            t_space[c] = u3[c].to_numpy()
    # comps are cheap and rule-driven (same hand, similar arm angle): always recompute
    cr, cp, ctol = type_comps(types.reset_index(drop=True))
    t_space["comp_rows"], t_space["comp_pct"], t_space["comp_tol"] = list(cr), list(cp), ctol
    t_space.to_parquet(ATLAS, index=False)
    spin = pd.concat([active_spin(s, a.refetch_leaders) for s in SEASONS], ignore_index=True)
    mv = movement_sample(a.rebuild_profiles)
    export(types.reset_index(drop=True), pitchers.reset_index(drop=True), t_space, lead, free_agents(), spin, mv)


if __name__ == "__main__":
    main()
