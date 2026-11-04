"""Embed the pseudobulk profiles: block-scaled PCA -> kNN -> Leiden -> UMAP.

Two tables from `mlbpitch.profiles`, one shared space across all six halves so
a pitcher's timepoints are directly comparable:

  type     pitcher x half x pitch_type. Two scalings:
             global  -- z over all rows; pitch types separate (the atlas picture)
             within  -- z inside each pitch type; what comps and trajectories use,
                        since under global scaling velo barely varies *within* a type
  pitcher  pitcher x half. Per-type columns z-scored, weighted by sqrt(usage) so a
           5% pitch doesn't count like the primary fastball, absent types = 0.

Block scaling: every feature block is divided by sqrt(its total variance), so a
16-column block carries the same weight as a 3-column one.

Outcomes (profiles.OUTCOME_COLUMNS) are asserted absent from every feature matrix.

Checks printed at the end:
  * self-continuity -- a pitcher's consecutive halves should sit far closer than
    random pairs; and how often a row's nearest neighbour is the same pitcher
  * season batch -- among kNN neighbours from DIFFERENT pitchers, the share from
    the same season should be ~1/3 (chance); well above means a season effect

Usage:  python scripts/profile_embed.py
"""
import numpy as np
import pandas as pd

from mlbpitch import paths, profiles

SEED = 0
K = 15
N_PCS = 30
RESOLUTION = 1.0

TYPE_BLOCKS = {
    **profiles.MEAN_FEATURES,
    "consistency": profiles.SD_FEATURES,
    "usage": profiles.USAGE_FEATURES,
}
PER_TYPE = profiles.MEAN_FEATURES["shape"] + profiles.MEAN_FEATURES["derived"]


def log(msg):
    print(msg, flush=True)


def zscore(X):
    return (X - X.mean()) / X.std(ddof=0).replace(0, 1)


def block_scale(Z, blocks):
    """Concatenate blocks, each scaled to unit total variance."""
    parts = []
    for name, cols in blocks.items():
        B = Z[cols].to_numpy(dtype="float64")
        tot = np.nansum(np.nanvar(B, axis=0))
        parts.append(B / np.sqrt(tot if tot > 0 else 1))
    return np.nan_to_num(np.hstack(parts))


def assert_no_outcomes(blocks):
    cols = {c for cs in blocks.values() for c in cs}
    leaked = cols & set(profiles.OUTCOME_COLUMNS + ["rv100_raw"])
    assert not leaked, f"outcome columns in features: {leaked}"


def embed(X, label):
    """PCA -> kNN -> Leiden -> UMAP. Returns (PCs, knn_idx, leiden, umap)."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    import igraph as ig
    import leidenalg
    import umap as umap_mod

    n_pcs = min(N_PCS, X.shape[1])
    pca = PCA(n_components=n_pcs, random_state=SEED).fit(X)
    P = pca.transform(X)
    log(f"[{label}] {X.shape[0]:,} x {X.shape[1]} -> {n_pcs} PCs "
        f"({100 * pca.explained_variance_ratio_.sum():.1f}% var)")

    idx = NearestNeighbors(n_neighbors=K + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
    edges = {(min(i, j), max(i, j)) for i, row in enumerate(idx) for j in row}
    g = ig.Graph(n=len(P), edges=sorted(edges))
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=RESOLUTION, seed=SEED)
    leiden = np.array(part.membership)
    log(f"[{label}] Leiden: {leiden.max() + 1} clusters at resolution {RESOLUTION}")

    U = umap_mod.UMAP(n_neighbors=K, min_dist=0.3, random_state=SEED).fit_transform(P)
    return P, idx, leiden, U


def continuity_checks(meta, P, idx, label):
    """Self-continuity across halves and season mixing among kNN neighbours."""
    rng = np.random.default_rng(SEED)
    m = meta.reset_index(drop=True)
    per = 2 if m.half.max() > 0 else 1                            # half=0: season rows
    m["t"] = (m.season - m.season.min()) * per + m.half.clip(lower=1) - 1
    group = ["pitcher"] + (["pitch_type"] if "pitch_type" in m else [])
    m["row"] = np.arange(len(m))
    nxt = m.merge(m.assign(t=m.t - 1), on=group + ["t"], suffixes=("", "_next"))
    d_self = np.linalg.norm(P[nxt.row] - P[nxt.row_next], axis=1)

    # Random pairs matched on pitch type (type table) so the baseline isn't
    # inflated by sinker-vs-curveball distances.
    a = rng.integers(0, len(m), 20 * len(nxt))
    if "pitch_type" in m:
        by_type = m.groupby("pitch_type").row.apply(np.array)
        b = np.array([rng.choice(by_type[pt]) for pt in m.pitch_type.to_numpy()[a]])
    else:
        b = rng.integers(0, len(m), len(a))
    keep = m.pitcher.to_numpy()[a] != m.pitcher.to_numpy()[b]
    d_rand = np.linalg.norm(P[a[keep]] - P[b[keep]], axis=1)
    log(f"[{label}] consecutive-half distance median {np.median(d_self):.2f} vs "
        f"random different-pitcher {np.median(d_rand):.2f} "
        f"(ratio {np.median(d_self) / np.median(d_rand):.2f}, n={len(d_self):,})")

    same_pitcher = m.pitcher.to_numpy()[idx] == m.pitcher.to_numpy()[:, None]
    has_other = m.groupby(group).row.transform("size").to_numpy() > 1
    log(f"[{label}] nearest neighbour is the same pitcher: "
        f"{100 * same_pitcher[has_other, 0].mean():.1f}% of rows with >1 half")

    season = m.season.to_numpy()
    other = ~same_pitcher
    same_season = (season[idx] == season[:, None])[other].mean()
    chance = (m.season.value_counts(normalize=True) ** 2).sum()
    log(f"[{label}] different-pitcher neighbours from the same season: "
        f"{100 * same_season:.1f}% (chance {100 * chance:.1f}%)")
    if "pitch_type" in m:
        pt = m.pitch_type.to_numpy()
        log(f"[{label}] neighbours of the same pitch type: "
            f"{100 * (pt[idx] == pt[:, None]).mean():.1f}%")


def type_space(types, blocks=TYPE_BLOCKS):
    """`blocks` picks the feature blocks that build the space (the atlas maps leave out usage)."""
    keys = profiles.KEYS + ["pitch_type", "player_name", "p_throws", "n"]
    assert_no_outcomes(blocks)
    cols = [c for cs in blocks.values() for c in cs]
    out = types[keys + profiles.OUTCOME_COLUMNS + ["rv100_raw"]].copy()

    for scaling in ("global", "within"):
        F = types[cols].astype("float64")
        F = F.fillna(F.groupby(types.pitch_type).transform("median"))
        Z = zscore(F) if scaling == "global" else pd.concat(
            zscore(g) for _, g in F.groupby(types.pitch_type)).loc[F.index]
        X = block_scale(Z, blocks)
        P, idx, leiden, U = embed(X, f"type/{scaling}")
        continuity_checks(types[keys], P, idx, f"type/{scaling}")
        out[[f"{scaling}_pc{i + 1}" for i in range(P.shape[1])]] = P
        out[f"{scaling}_leiden"] = leiden
        out[[f"{scaling}_umap1", f"{scaling}_umap2"]] = U
    return out


def pitcher_space(pitchers):
    keys = profiles.KEYS + ["player_name", "p_throws", "n", "primary_fb"]
    T = profiles.PITCH_TYPES
    per_type = {t: [f"{t}_{f}" for f in PER_TYPE + profiles.SD_FEATURES] for t in T}
    blocks = {
        "usage": [f"{t}_usage_all" for t in T],   # untruncated: no 40-pitch flicker
        "delivery": profiles.MEAN_FEATURES["delivery"],
        "deception": ["release_spread"] + [f"{t}_{d}" for t in T
                                           for d in ("d_velo", "d_pfx_x", "d_pfx_z", "d_vaa")],
        "arsenal": [c for cs in per_type.values() for c in cs],
        "situational": [f"{t}_{u}" for t in T
                        for u in ("usage_ahead", "usage_behind", "usage_vs_lhb", "usage_vs_rhb")],
    }
    blocks = {k: [c for c in v if c in pitchers] for k, v in blocks.items()}
    assert_no_outcomes(blocks)

    F = pitchers[[c for cs in blocks.values() for c in cs]].astype("float64")
    Z = zscore(F).fillna(0)       # absent type / no primary FB -> league average
    for t in T:
        w = np.sqrt(pitchers[f"{t}_usage_all"].to_numpy())[:, None]
        Z[per_type[t]] = Z[per_type[t]].to_numpy() * w
    X = block_scale(Z, blocks)

    P, idx, leiden, U = embed(X, "pitcher")
    continuity_checks(pitchers[keys], P, idx, "pitcher")
    out = pitchers[keys + ["rv100", "rv100_raw"]].copy()
    out[[f"pc{i + 1}" for i in range(P.shape[1])]] = P
    out["leiden"] = leiden
    out[["umap1", "umap2"]] = U
    return out


def embed_path(kind):
    return paths.META / f"embed_{kind}_{min(profiles.SEASONS)}_{max(profiles.SEASONS)}.parquet"


if __name__ == "__main__":
    types = pd.read_parquet(profiles.profiles_path("type"))
    pitchers = pd.read_parquet(profiles.profiles_path("pitcher"))
    for kind, df in [("type", type_space(types)), ("pitcher", pitcher_space(pitchers))]:
        df.to_parquet(embed_path(kind), index=False)
        log(f"wrote {embed_path(kind)}  ({len(df):,} rows)")
