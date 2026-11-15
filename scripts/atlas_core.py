"""Pitch atlas analysis: pitcher-season profiles -> t-SNE maps -> comps, uniqueness and changes.

One dot = one pitcher-season's version of one pitch (2024-2026).

  all pitches        pitcher x season x pitch type, global scaling (types separate)
  <type> universe    one t-SNE map per pitch type, fit on that type's rows alone
  (maps, comps and uniqueness all use the 15 pitch-identity measures, profile_embed.IDENTITY_BLOCKS:
   shape, derived and delivery without near-duplicates; location, scatter and usage are not part of
   "similar" -- analysis/umap_benchmark.md)
  comps              10 nearest DISTINCT other pitchers in the PCA space each map
                     is drawn from (not the 2D map), any season

Levels. The maps, comps, percentiles and baselines are all MLB. Triple-A pitchers are *placed* on the
fixed MLB maps: a call-up who also has an MLB row of that pitch type by the map's own t-SNE transform
(tsne_place), anyone else at the inverse-distance mean of his 15 nearest big-league pitches of the same
type in the PCA space (place_level). The MLB layout is never refitted because of them.

Writes data/meta/profiles_{type,pitcher}_season_2024_2026.parquet (season_profiles) and
analysis/biggest_changes.csv (biggest_changes); build_atlas writes the maps, placements and comps to
data/meta/atlas_type_2024_2026.parquet.

Usage:
  python scripts/atlas_core.py                    # restart from the cached layout when it used the same measures
  python scripts/atlas_core.py --rebuild-profiles # after an ingest
  python scripts/atlas_core.py --cold-layout      # refit the maps from scratch
"""
import argparse
import json

import numpy as np
import pandas as pd

from comps import other_pitcher_neighbours
from mlbpitch import paths, profiles, store
from profile_embed import (IDENTITY_BLOCKS, TSNE_PERPLEXITY, TYPE_BLOCKS, block_scale, embed,
                           tsne_layout, zscore)


SEASONS = profiles.SEASONS
RANGE = f"{min(SEASONS)}_{max(SEASONS)}"
LEVELS = ("mlb", "aaa")   # MLB first: it is the reference every other level is measured against
K_COMPS = 10
NBHD = 30          # neighbourhood stored per pitch: first K_COMPS are the displayed comps, the rest feed
                   # the "what goes with lower xERA among his lookalikes" analysis on the page
NBHD_LEVEL = {"mlb": NBHD, "aaa": K_COMPS}   # that analysis needs his own xERA, which Triple-A has none of
K_PLACE = 15       # MLB neighbours a Triple-A stranger is placed among (inverse-distance mean)
ATLAS = paths.META / f"atlas_type_{RANGE}.parquet"
TYPE_FEATURES = [c for cs in TYPE_BLOCKS.values() for c in cs]
# Maps (2D, 3D, clusters, Triple-A placement), comps and uniqueness are all built on what a pitch IS: shape,
# derived and delivery, near-duplicates dropped (15 measures). Usage left the maps on 2026-11-03; location and
# scatter left maps and comps, and usage left the comps, on 2026-11-04 (user decisions; the half-to-half
# reliability of those blocks is 0.6 to 0.86 against 0.92 to 0.97, and they drove 79% of a pitcher's season-to-
# season movement in the old comps space). Cards, filters, colours and the heat table still show all 32 measures.
MAP_BLOCKS = IDENTITY_BLOCKS
MAP_FEATURES = [c for cs in MAP_BLOCKS.values() for c in cs]
# Evolution (how far a pitch moved since last season; Big Evolvers; home's Biggest changes) measures the move on the map's
# identity measures WITHOUT spin direction (user, 2026-11-15): Savant infers it from the movement, and it swings on pitches
# that barely move. The maps keep it.
EVO_BLOCKS = {b: [c for c in cs if c not in ("spin_sin", "spin_cos")] for b, cs in MAP_BLOCKS.items()}
EVO_FEATURES = [c for cs in EVO_BLOCKS.values() for c in cs]
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
    # fit percentile, fair at every rank (user, 2026-11-14): the k-th comp's distance against every reference pitch's
    # k-th comp distance, within the pitch type (each type has its own PCA space). Reference: MLB pitches among MLB
    # candidates, or the queried level among its own. Before, every rank was measured against the nearest comp
    # pooled over types, so comps 2-10 read "Loose" almost always.
    ref = lvl == cand_level
    ptype = types.pitch_type.to_numpy()
    pct = np.full(dist_out.shape, np.nan)
    for t_ in np.unique(ptype):
        m = ptype == t_
        for k in range(dist_out.shape[1]):
            refk = dist_out[ref & m, k]
            refk = np.sort(refk[~np.isnan(refk)])
            if len(refk) < 20:   # too few at this rank in this type: fall back to every type's k-th
                refk = dist_out[ref, k]
                refk = np.sort(refk[~np.isnan(refk)])
            if not len(refk):
                continue
            col = dist_out[m, k]
            pct[m, k] = np.where(np.isnan(col), np.nan, np.searchsorted(refk, np.nan_to_num(col, nan=np.inf)) / len(refk))
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
    d["evo"] = evolution(t)          # how far the pitch moved since his previous season (as the home page's list)
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


def evolution(types):
    """Per pitch: how far it moved since the pitcher's previous season at the same level, on the all-pitches map's
    13 identity measures without spin direction (EVO_FEATURES), scaled as that map scales them (fitted on MLB rows, as biggest_changes does): the distance
    to the nearest pitch he threw >= CHANGE_THEN of the time the season before. A relabel (the same slider now called
    a sweeper) scores ~0 and a new pitch scores high. NaN in a pitcher's first season at that level. The home page's
    per-pitcher score is the usage-weighted mean of these over his pitches thrown >= CHANGE_NOW of the time."""
    out = np.full(len(types), np.nan)
    lvl = (types.level if "level" in types else pd.Series("mlb", index=types.index)).to_numpy()
    mlb = lvl == "mlb"
    pt = types.pitch_type.to_numpy()
    sc = scaler_fit(types.loc[mlb, EVO_FEATURES], pt[mlb], EVO_BLOCKS)
    X = scaler_apply(sc, types[EVO_FEATURES], pt)
    key = pd.DataFrame({"lvl": lvl, "pid": types.pitcher.to_numpy(), "s": types.season.to_numpy(),
                        "u": types.usage.to_numpy(dtype="float64")})   # positional index 0..n-1, like X
    for _, g in key.groupby(["lvl", "pid"]):
        for s in np.unique(g.s):
            then = g.index[(g.s == s - 1) & (g.u >= CHANGE_THEN)].to_numpy()
            if not len(then):
                continue
            now = g.index[g.s == s].to_numpy()
            out[now] = np.linalg.norm(X[now][:, None, :] - X[then][None, :, :], axis=2).min(axis=1)
    return pd.Series(out, index=types.index)


def evo_too_few(types, pitchers):
    """Rows whose Evolution is left blank (user, 2026-11-15), all with no row for this pitch last season at this level:
    he threw it (>= CHANGE_NEW of his pitches) but too few for a row (under profiles.MIN_PITCHES), so there is nothing of
    the same pitch to compare with -- the distance to his nearest other pitch would rank an injury-shortened season's
    curveball as a big mover; or it is under CHANGE_ADD of his pitches now, too rare to call a new pitch (2% -> 3%).
    A pitch under CHANGE_NEW last season and at least CHANGE_ADD now is new, and keeps its Evolution (how unlike
    anything he threw)."""
    lvl = types.level if "level" in types else pd.Series("mlb", index=types.index)
    have = set(zip(types.pitcher, types.season, types.pitch_type, lvl))
    now_use = types.usage.to_numpy(dtype="float64")
    pk = pitchers.set_index(["pitcher", "season", "level"])
    out = np.zeros(len(types), dtype=bool)
    for k, (pid, s, pt, lv) in enumerate(zip(types.pitcher, types.season, types.pitch_type, lvl)):
        if (pid, s - 1, pt, lv) in have or (pid, s - 1, lv) not in pk.index:
            continue
        before = pk.at[(pid, s - 1, lv), f"{pt}_usage_all"] if f"{pt}_usage_all" in pk.columns else np.nan
        out[k] = before >= CHANGE_NEW or now_use[k] < CHANGE_ADD
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


# ---- the home page's "Biggest changes since <last season>" (user, 2026-11-14)
# First-draft thresholds, to tune after reading analysis/biggest_changes.csv.
CHANGE_MIN_N = {"mlb": 500, "aaa": 300}   # pitches thrown in each of the two seasons, at that level
CHANGE_NOW, CHANGE_THEN, CHANGE_NEW = 0.10, 0.05, 0.02   # usage that counts this season / last season; under 2% last season = new
CHANGE_ADD = 0.05   # ...and a new pitch needs at least 5% now (user, 2026-11-15: 2% -> 3% is no new pitch); the page's NEW_MIN_NOW
CHANGE_TOP, CHANGE_REVIEW = 5, 25
CHANGE_CSV = paths.ROOT / "analysis" / "biggest_changes.csv"
PTN_NAME = {"FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper", "CU": "Curveball",
            "CH": "Changeup", "FS": "Splitter"}   # the page's PTN
# the measures a reason can name, in the page's display units (type_display): key, value, unit, decimals, words (up, down)
CHANGE_WORDS = [
    ("velo", lambda t: t.release_speed, "mph", 1, ("harder", "softer")),
    ("ivb", lambda t: 12 * t.pfx_z, "in", 1, None),
    ("hb", lambda t: -12 * t.pfx_x, "in", 1, None),
    ("spin", lambda t: t.release_spin_rate, "rpm", 0, ("more spin", "less spin")),
    # no spin direction (user, 2026-11-15): Savant infers it from the movement, and it swings on pitches that barely move
    ("arm_angle", lambda t: t.arm_angle, "°", 1, ("higher arm angle", "lower arm angle")),
    ("rel_height", lambda t: t.release_pos_z, "ft", 2, ("higher release", "lower release")),
    ("rel_side", lambda t: -t.release_pos_x, "ft", 2, ("wider release", "narrower release")),   # arm-side +, as the page's rel_side
    ("extension", lambda t: t.release_extension, "ft", 2, ("more extension", "less extension")),
    ("vaa", lambda t: t.vaa_above_avg, "°", 2, ("flatter approach", "steeper approach")),
]


def _delta(key, x0, x1):
    """x1 - x0; for the spin axis the short way round the circle."""
    return (x1 - x0 + 180) % 360 - 180 if key == "spin_axis" else x1 - x0


def _y2y_scale(t, W):
    """How much each reason measure usually moves for the same MLB pitch from one season to the next, per pitch type
    (a robust SD, 1.4826 x MAD, so the big changes being looked for don't set their own yardstick). An 11-degree drop
    in arm angle is the headline though arm angles vary a lot between pitchers; a 1 mph change is not."""
    keys = list(W)
    V = pd.DataFrame(W).assign(pitcher=t.pitcher.to_numpy(), season=t.season.to_numpy(), pitch_type=t.pitch_type.to_numpy())
    V = V[(t.level == "mlb").to_numpy() & (t.n >= 100).to_numpy()]
    parts = []
    for s in SEASONS[1:]:
        a = V[V.season == s - 1].set_index(["pitcher", "pitch_type"])
        b = V[V.season == s].set_index(["pitcher", "pitch_type"])
        j = a.index.intersection(b.index)
        parts.append(pd.DataFrame({k: _delta(k, a.loc[j, k].to_numpy(), b.loc[j, k].to_numpy()) for k in keys})
                     .assign(pitch_type=j.get_level_values("pitch_type")))
    dd = pd.concat(parts, ignore_index=True)
    mad = lambda x: 1.4826 * float((x - x.median()).abs().median())
    return {k: dd.groupby("pitch_type")[k].agg(mad) for k in keys}


def _change_word(key, x0, x1, words, tip=False):
    """A change in plain words, worded as the page's SHAPE_DIFFS: a pitch's "ride" is drop when it sinks. tip: the
    form that follows an amount ("8.8 in more ride", not "8.8 in rides more")."""
    up = x1 > x0
    if words:
        return words[0] if up else words[1]
    if key == "ivb":
        if x0 >= 0 and x1 >= 0:
            return "more ride" if up else "less ride"
        if x0 <= 0 and x1 <= 0:
            return "less drop" if up else "more drop"
        return ("more ride" if up else "more drop") if tip else ("rides more" if up else "drops more")
    if x0 >= 0 and x1 >= 0:   # hb, arm-side +
        return "more arm-side run" if up else "less arm-side run"
    if x0 <= 0 and x1 <= 0:
        return "less glove-side break" if up else "more glove-side break"
    return ("more arm-side break" if up else "more glove-side break") if tip else ("breaks more to the arm side" if up else "breaks more to the glove side")


def biggest_changes(types, pitchers, write_csv=True):
    """The pitchers whose pitches changed most since last season, five per level, for the home page.

    In the all-pitches map's space (the 13 identity measures without spin direction, scaled as that map scales them, fitted on MLB rows so
    Triple-A is measured the same way), each pitch he throws >= CHANGE_NOW of the time this season is matched to the
    nearest pitch he threw >= CHANGE_THEN of the time last season, and his score is those distances averaged by
    usage: how far his arsenal moved on the map. Matching to the nearest pitch means a relabel (the same slider now
    called a sweeper) scores ~0, and a new arm slot moves every pitch. The reason names the pitch that moved most:
    "New sweeper" when he threw that type under CHANGE_NEW of the time last season (trajectories.py's add threshold),
    else its one or two measures that moved most for how much they usually move in a year (_y2y_scale), in words; numbers go in `tip`.
    Writes the top CHANGE_REVIEW per level to CHANGE_CSV for review.
    """
    S = max(SEASONS)
    S0 = S - 1
    t = types.reset_index(drop=True)
    mlb = (t.level == "mlb").to_numpy()
    sc = scaler_fit(t.loc[mlb, EVO_FEATURES], t.loc[mlb, "pitch_type"].to_numpy(), EVO_BLOCKS)
    X = scaler_apply(sc, t[EVO_FEATURES], t.pitch_type.to_numpy())
    W = {k: f(t).astype("float64") for k, f, *_ in CHANGE_WORDS}
    sd = _y2y_scale(t, W)
    pk = pitchers.set_index(["pitcher", "season", "level"])
    rows = []
    for (level, pid), g in t[t.season.isin([S0, S])].groupby(["level", "pitcher"]):
        a, b = g[g.season == S0], g[g.season == S]
        if a.empty or b.empty or (pid, S0, level) not in pk.index or (pid, S, level) not in pk.index:
            continue
        p0, p1 = pk.loc[(pid, S0, level)], pk.loc[(pid, S, level)]
        if not (p0.n >= CHANGE_MIN_N.get(level, 500) and p1.n >= CHANGE_MIN_N.get(level, 500)):
            continue
        now, then = b[b.usage >= CHANGE_NOW], a[a.usage >= CHANGE_THEN]
        # a pitch he threw last season but too rarely for a row has nothing to compare with (as evo_too_few): left out
        few = [pt not in set(a.pitch_type) and p0.get(f"{pt}_usage_all", np.nan) >= CHANGE_NEW for pt in now.pitch_type]
        now = now[~np.array(few, dtype=bool)]
        if now.empty or then.empty:
            continue
        D = np.linalg.norm(X[now.index.to_numpy()][:, None, :] - X[then.index.to_numpy()][None, :, :], axis=2)
        d, w = D.min(axis=1), now.usage.to_numpy(dtype="float64")
        k = int(np.argmax(w * d))
        top, near = now.iloc[k], then.iloc[int(np.argmin(D[k]))]
        pt, name = top.pitch_type, PTN_NAME.get(top.pitch_type, top.pitch_type)
        before = p0.get(f"{pt}_usage_all", np.nan)
        moved = []
        if not (before >= CHANGE_NEW):   # NaN (never thrown) or under 2%: a new pitch
            reason = f"New {name.lower()}"
            tip = f"New {name.lower()} in {S}: {100 * top.usage:.0f}% of his pitches ({100 * (before if before == before else 0):.0f}% in {S0})"
        else:
            same = a[a.pitch_type == pt]
            ref = same.iloc[0] if len(same) else near   # his same pitch last season (it has a row with 40+ pitches)
            for key, _, unit, dec, words in CHANGE_WORDS:
                x0, x1, s = W[key][ref.name], W[key][top.name], sd[key].get(pt, np.nan)
                if np.isfinite(x0) and np.isfinite(x1) and s > 0:
                    moved.append((abs(_delta(key, x0, x1)) / s, key, x0, x1, unit, dec, words))
            moved.sort(key=lambda m: -m[0])
            pick = [m for m in moved[:2] if m[0] >= 1.5] or moved[:1]   # at least 1.5 of its usual season-to-season moves
            reason = f"{name}: " + ", ".join(_change_word(m[1], m[2], m[3], m[6]) for m in pick)
            amt = lambda m: f"{abs(_delta(m[1], m[2], m[3])):.{m[5]}f}" + (m[4] if m[4] == "°" else " " + m[4])
            said = lambda m: f"{amt(m)} change in spin direction" if m[1] == "spin_axis" else f"{amt(m)} {_change_word(m[1], m[2], m[3], m[6], tip=True)}"
            tip = f"{name}, {S0} to {S}: " + ", ".join(said(m) for m in pick)
        rows.append({"level": level, "pitcher": int(pid), "name": first_last(top.player_name), "season": S, "pt": pt,
                     "score": float((w * d).sum() / w.sum()), "reason": reason, "tip": tip,
                     "moved_most": float(d[k]), "usage": float(top.usage), "n_before": int(p0.n), "n_now": int(p1.n),
                     **{f"d_{m[1]}": round(_delta(m[1], m[2], m[3]), 3) for m in moved},
                     **{f"z_{m[1]}": round(m[0], 2) for m in moved}})
    if not rows:
        return None
    res = pd.DataFrame(rows).sort_values(["level", "score"], ascending=[True, False])
    res["rank"] = res.groupby("level").cumcount() + 1
    if write_csv:
        res[res["rank"] <= CHANGE_REVIEW].to_csv(CHANGE_CSV, index=False, float_format="%.3f")
    top5 = res[res["rank"] <= CHANGE_TOP]
    log("biggest changes since " + str(S0) + ": " + "; ".join(f"{l}: " + ", ".join(f"{r.name} ({r.reason})" for r in g.itertuples())
                                                              for l, g in top5.groupby("level")))
    return {"from": S0, "to": S, "changes": {l: [{k: r[k] for k in ("pitcher", "season", "pt", "name", "reason", "tip")}
                                                 for r in g.to_dict("records")] for l, g in top5.groupby("level")}}


def build_atlas(rebuild_profiles=False, cold_layout=False, reuse=False):
    """Profiles -> MLB t-SNE maps -> Triple-A placement -> comps, written to ATLAS.

    reuse: keep the MLB layout and the placements cached in ATLAS (rows that still exist) instead of refitting.
    Returns types, pitchers, t_space (every level, with a `level` column), layout quality and the levels built.
    """
    types, pitchers = season_profiles(rebuild_profiles, "mlb")
    cached_levels = {}
    if reuse and ATLAS.exists():
        t_space = pd.read_parquet(ATLAS)
        if "level" in t_space:   # other levels: their cached placements are kept for rows that still exist
            cached_levels = {l: g.drop(columns="level") for l, g in t_space[t_space.level != "mlb"].groupby("level")}
            t_space = t_space[t_space.level == "mlb"].drop(columns="level").reset_index(drop=True)
        prev = None
    else:
        prev = previous_layout(cold_layout)
        t_space = build_type_space(types, prev)
    # 3D layouts: computed with the 2D ones and cached in the atlas table
    if not {"gx3", "tx3"} <= set(t_space.columns):
        u3 = layout_3d(types, prev)
        for c in u3.columns:
            t_space[c] = u3[c].to_numpy()

    # how much each map keeps of the true neighbourhoods
    quality = layout_quality(types.reset_index(drop=True), t_space.reset_index(drop=True))

    # other levels ride on the MLB maps: placed among their nearest big-league pitches, never refitted
    levels = ["mlb"] + [l for l in LEVELS if l != "mlb" and level_seasons(l)]
    types, pitchers, t_space = [df.reset_index(drop=True).assign(level="mlb") for df in (types, pitchers, t_space)]
    parts = {"types": [types], "pitchers": [pitchers], "space": [t_space]}
    for level in levels[1:]:
        t_l, p_l = season_profiles(rebuild_profiles, level)
        parts["types"].append(t_l.assign(level=level)); parts["pitchers"].append(p_l.assign(level=level))
        parts["space"].append(place_level(types, t_space, t_l, cached_levels.get(level)).assign(level=level))
    types, pitchers, t_space = [pd.concat(parts[k], ignore_index=True) for k in ("types", "pitchers", "space")]

    # comps are cheap and rule-driven (same hand, similar arm angle): always recompute
    cr, cp, ctol = type_comps(types)
    t_space["comp_rows"], t_space["comp_pct"], t_space["comp_tol"] = list(cr), list(cp), ctol
    # every pitch, MLB or Triple-A, also gets its lookalikes among Triple-A pitchers
    if "aaa" in levels:
        cr, cp, ctol = type_comps(types, None, "aaa")
        t_space["comp_rows_aaa"], t_space["comp_pct_aaa"], t_space["comp_tol_aaa"] = list(cr), list(cp), ctol
    t_space.to_parquet(ATLAS, index=False)
    if not reuse:
        LAYOUT_META.write_text(json.dumps(layout_signature(), indent=1))
    return types, pitchers, t_space, quality, levels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild-profiles", action="store_true", help="after an ingest")
    ap.add_argument("--cold-layout", action="store_true", help="fit the maps from scratch instead of restarting")
    ap.add_argument("--reuse-layout", action="store_true", help="keep the cached maps; recompute comps only")
    a = ap.parse_args()
    build_atlas(a.rebuild_profiles, a.cold_layout, reuse=a.reuse_layout)


if __name__ == "__main__":
    main()
