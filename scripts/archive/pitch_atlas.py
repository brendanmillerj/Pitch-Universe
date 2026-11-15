"""Pitch atlas -- an snRNAseq-style pipeline with **pitch = cell**.

Ports the standard single-nucleus workflow onto Statcast pitches. The mapping is
deliberate, and where a step has no honest analogue it is dropped rather than
faked (see "What is NOT ported" below).

    snRNAseq                          this pipeline
    ---------------------------------------------------------------------
    QC: min genes / min UMIs          complete feature vector + physical
                                      plausibility bounds
    QC: % mitochondrial               kNN isolation score (see note)
    normalize + log1p                 StandardScaler only (no counts to log)
    HVG selection                     fixed biophysical feature block
    batch correction (scVI)           Harmony on home_team -- OFF, no real
                                      batch effect once the pitcher confound
                                      is controlled (see embed_and_cluster)
    Leiden overcluster (res 2.0)      Leiden overcluster (res 2.0)
    CellTypist reference annotation   Statcast's own pitch_type classifier
    majority-vote over partitions     majority-vote over partitions
    Scrublet doublet removal          falls out of majority-vote disagreement
    microglia subcluster (res 0.4)    fastball subcluster (res 0.4)
    DAM signature score               ddz_delta_az (Dynamic Dead Zone)
    UMAP for visualization            UMAP for visualization

Two framing decisions drive the rest:

1. **Pitch = cell, pitcher ignored.** So there is no library size and no
   aggregation: the gene-count/UMI filters have no analogue at all and are gone.

2. **`pitch_type` is a model's prediction, not truth.** Statcast's classifier is
   this project's CellTypist -- trained elsewhere, applied here, and wrong in the
   same places any reference model is wrong. Leiden runs *unsupervised*, and
   majority-voting Statcast's calls over geometrically pure partitions denoises
   them. A pitch that disagrees with its partition's majority is the flagged
   ambiguous/misclassified one, which is why no separate Scrublet port is
   needed -- the same signal falls out of the annotation step.

**Stratified by handedness** -- two independent runs. pfx_x, ax and spin_axis are
all lab-frame, so a lefty's break and spin axis are the reflection of an
identical righty's. Pooling requires mirroring *every* such column into a common
frame and missing even one is silent and severe: an earlier combined fit mirrored
pfx_x but not spin_axis, which left the features in opposite frames and tore
every LHP fastball/sinker/changeup into its own detached island (separation
scaled with each type's L-vs-R spin gap: sweeper 124deg -> 2.25 sd, cutter 19deg
-> 0.28 sd, the control). Stratifying removes the failure mode outright.
Handedness is a biological covariate like sex -- stratify it; it is NOT batch.

**What is NOT ported.** log-normalization and HVG selection exist to fix count
sparsity and variable sequencing depth; these features are dense continuous
physical measurements with neither, so z-scoring is the whole normalization step.
scVI is deliberately not used: it models counts under a negative-binomial
likelihood, and would also be *expanding* 7 dims to 30 rather than compressing
20k. Harmony is the right continuous-data equivalent. And % mitochondrial has no
analogue -- it is inherently compositional (a fraction of reads *within* one
cell) and a pitch has no reads. The kNN isolation score replaces its function (a
continuous per-cell quality scalar you threshold), not its definition.

Env vars:
    ATLAS_SAMPLE=N   fit on a random N-row sample (smoke testing)
    ATLAS_OUTDIR     override output directory (default <repo>/analysis)
    ATLAS_HARMONY=1  enable park batch correction (off by default -- measured
                     as unnecessary; see embed_and_cluster)
"""
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mlbpitch import ddz, paths, store

t0 = time.time()
SAMPLE = int(os.environ.get("ATLAS_SAMPLE", "0"))
OUTDIR = Path(os.environ.get("ATLAS_OUTDIR", paths.ROOT / "analysis"))
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42
K_NEIGHBORS = 15
RES_OVERCLUSTER = 2.0    # matches the snRNAseq recipe's overclustering resolution
RES_SUBCLUSTER = 0.4     # matches its microglia/DAM subclustering resolution
ISOLATION_PCT = 99.5     # drop the most isolated 0.5% (tracking artifacts)
SUBCLUSTER_FAMILY = "Fastball"

FEATURES = ["release_speed", "pfx_x", "pfx_z", "release_extension", "arm_angle",
            "spin_sin", "spin_cos"]

# Expected-vs-actual movement. Measured placement, not assumed:
#
#   eta^2 vs pitch_type      raw pfx_x 0.860   raw pfx_z 0.782
#   delta, baseline fit PER pitch type   0.000            0.000
#   delta, baseline fit POOLED           0.853            0.791
#
# A per-type residual is orthogonal to type by construction -- exactly 0.000, so
# it cannot help the global atlas. The pooled variant keeps the signal but is
# r=0.993/0.986 with raw movement (delivery explains only 1.4%/2.7% of movement
# variance ACROSS types), so it is a near-duplicate column, not a new axis;
# including both merely double-weights movement in the Euclidean metric.
#
# The information is real but it is a WITHIN-type quantity -- which is why DDZ is
# fit per type in the first place. So the deltas drive SUBCLUSTERING (where every
# pitch shares a family and "unusual for its kind" is the whole question) and are
# carried as stored columns + overlays, never as global atlas features. Same
# discipline as a DAM score: overlaid on the embedding, not fed into the HVGs.
MOVEMENT = ["ax", "az"]                          # DDZ's own responses, ft/s^2
DELIVERY = ["arm_angle", "scaled_extension"]     # DDZ's own conditioners
SUBCLUSTER_FEATURES = FEATURES + [f"ddz_delta_{c}" for c in MOVEMENT]

# Optionally put a POOLED ddz delta into the global atlas (ATLAS_DDZ_FEATURES=1),
# replacing pfx_x/pfx_z. Measured to be inert rather than harmful:
#
#   delivery R^2, pooled over all types      ax 0.024   az 0.044
#   delivery R^2, within Sinker              ax 0.004   az 0.587
#   delivery R^2, within 4-Seam              ax 0.329   az 0.342
#
# Pooling collapses exactly the effect DDZ exists to measure (13x on az), so the
# pooled delta ends up r=0.988/0.977 with raw ax/az -- which is in turn r=0.990/
# 0.985 with the pfx_x/pfx_z already in FEATURES. Swapping them changes the
# pitch_type silhouette from 0.0504 to 0.0499, i.e. not at all. Left available
# because it is a defensible framing ("movement not attributable to delivery"),
# not because it improves anything.
USE_DDZ_FEATURES = bool(os.environ.get("ATLAS_DDZ_FEATURES"))
if USE_DDZ_FEATURES:
    FEATURES = [f for f in FEATURES if f not in ("pfx_x", "pfx_z")] + \
               [f"ddz_pooled_{c}" for c in MOVEMENT]
    SUBCLUSTER_FEATURES = FEATURES + [f"ddz_delta_{c}" for c in MOVEMENT]

# Physical plausibility bounds -- the "dead cell" filter. Values outside these
# are tracking failures or non-competitive lobs, not pitch archetypes.
BOUNDS = {
    "release_speed": (40.0, 108.0),
    "release_extension": (3.5, 8.5),
    "arm_angle": (-30.0, 95.0),
}


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


# ==========================================================================
# Stage 0 -- load
# ==========================================================================
sample_clause = f"USING SAMPLE {SAMPLE} ROWS" if SAMPLE else ""
raw = store.query(f"""
    SELECT pitcher, player_name, p_throws, pitch_type, pitch_name, home_team,
           game_date, inning, pitch_number,
           release_speed, pfx_x, pfx_z, release_extension, arm_angle, spin_axis,
           ax, az
    FROM (SELECT * FROM pitches {sample_clause})
    WHERE pitch_type IS NOT NULL
      AND release_speed IS NOT NULL AND pfx_x IS NOT NULL AND pfx_z IS NOT NULL
      AND release_extension IS NOT NULL AND arm_angle IS NOT NULL
      AND spin_axis IS NOT NULL
""")
log(f"loaded {len(raw):,} pitches | {raw.pitch_type.nunique()} types | "
    f"{raw.home_team.nunique()} parks")

# spin_axis is circular (0-360 deg) -- carry as sin/cos, never as a raw degree.
# No handedness mirror: each run below sees one hand, so the frame is common.
th = np.deg2rad(raw.spin_axis.astype("float64").to_numpy())
raw["spin_sin"], raw["spin_cos"] = np.sin(th), np.cos(th)

# DAM-equivalent signature score. ddz_delta_az = observed az minus what the
# pitcher's arm angle and height-scaled extension predict; positive = more ride
# than expected. Structurally a module score: a residual against a baseline,
# which is exactly how a DAM score is built and thresholded.
try:
    raw = ddz.add_ddz_delta(raw)
    HAVE_DDZ = "ddz_delta_az" in raw.columns
except (FileNotFoundError, KeyError) as e:
    log(f"WARNING: DDZ baseline unavailable ({e}); skipping signature scoring")
    HAVE_DDZ = False
log(f"DDZ signature available: {HAVE_DDZ}")


from mlbpitch import people

raw["scaled_extension"] = raw.release_extension / raw.pitcher.map(
    people.load_bio(paths.SEASON).set_index("pitcher_id")["height_ft"]
)
# Height is missing for a handful of pitchers; unscaled extension is a fine
# stand-in for the delivery term rather than dropping those pitches entirely.
raw["scaled_extension"] = raw.scaled_extension.fillna(raw.release_extension / 6.2)


def add_expected_movement(df):
    """Attach expected_ax/az to pair with ddz.py's ddz_delta_ax/az.

    ddz.add_ddz_delta already gives the residual (actual minus what the
    pitcher's arm angle and height-scaled extension predict) using the stored
    per-pitch-type baseline, which is the right conditioning: delivery explains
    up to 59% of sinker az variance within a type but only 4% pooled across
    types. This just recovers the other half of the pair -- the expectation
    itself -- so "expected vs actual" is directly readable per pitch.

    Also fits a POOLED (type-agnostic) variant for the optional atlas features;
    see USE_DDZ_FEATURES for why it is off by default.
    """
    out = df.copy()
    for c in MOVEMENT:
        if f"ddz_delta_{c}" in out.columns:
            out[f"expected_{c}"] = out[c] - out[f"ddz_delta_{c}"]

    # Pooled baseline, fit per hand, ALL pitch types together.
    for hand, g in out.groupby("p_throws"):
        s = g.dropna(subset=DELIVERY + MOVEMENT)
        if len(s) < 200:
            continue
        A = np.column_stack([np.ones(len(s))] + [s[c].to_numpy(float) for c in DELIVERY])
        for c in MOVEMENT:
            coef = np.linalg.lstsq(A, s[c].to_numpy(float), rcond=None)[0]
            out.loc[s.index, f"ddz_pooled_{c}"] = s[c].to_numpy(float) - A @ coef
    return out


raw = add_expected_movement(raw)
if HAVE_DDZ:
    log(f"expected/actual attached | ddz_delta_az sd = {raw.ddz_delta_az.std():.3f} "
        f"vs raw az sd = {raw.az.std():.3f}")
if USE_DDZ_FEATURES:
    log(f"ATLAS_DDZ_FEATURES set -- atlas features: {FEATURES}")

# A NaN in any feature silently drops that pitch from every kNN graph
# downstream, so fail loudly here instead.
_feat_na = raw[FEATURES + [f"ddz_delta_{c}" for c in MOVEMENT if f"ddz_delta_{c}" in raw]].isna().sum()
if _feat_na.any():
    log(f"WARNING: NaNs in features, dropping those rows:\n{_feat_na[_feat_na > 0]}")
    raw = raw.dropna(subset=[c for c in _feat_na[_feat_na > 0].index])
    log(f"  -> {len(raw):,} pitches remain")


# ==========================================================================
# Stage 1 -- QC
# ==========================================================================
def qc_filter(df):
    """Physical plausibility + kNN isolation. Returns (kept, report)."""
    from sklearn.preprocessing import StandardScaler
    from pynndescent import NNDescent

    n0 = len(df)
    rep = {"input": n0}

    ok = np.ones(n0, dtype=bool)
    for col, (lo, hi) in BOUNDS.items():
        bad = ~df[col].between(lo, hi).to_numpy()
        rep[f"drop_{col}"] = int(bad.sum())
        ok &= ~bad
    df = df.loc[ok].copy()

    # Isolation score: mean distance to the K nearest neighbours in scaled space.
    # Functionally what %mito does -- a continuous per-cell quality scalar with a
    # threshold -- without pretending a pitch has a mitochondrial compartment.
    Z = StandardScaler().fit_transform(df[FEATURES].to_numpy(dtype="float32"))
    idx = NNDescent(Z, n_neighbors=K_NEIGHBORS, metric="euclidean",
                    random_state=SEED)
    _, dists = idx.neighbor_graph
    iso = dists[:, 1:].mean(axis=1)
    df["isolation_score"] = iso
    cut = np.percentile(iso, ISOLATION_PCT)
    keep = iso <= cut
    rep["drop_isolated"] = int((~keep).sum())
    rep["isolation_cut"] = float(cut)
    df = df.loc[keep].copy()

    rep["kept"] = len(df)
    rep["pct_kept"] = round(100 * len(df) / n0, 2)
    return df, rep


# ==========================================================================
# Stage 2-4 -- normalize, batch-correct, graph, cluster
# ==========================================================================
def embed_and_cluster(df, hand):
    """Scale -> Harmony(park) -> kNN graph -> Leiden overcluster -> UMAP."""
    from sklearn.preprocessing import StandardScaler
    from pynndescent import NNDescent
    import igraph as ig
    import leidenalg
    import umap as umap_mod
    from umap.umap_ import fuzzy_simplicial_set

    # --- normalize (the whole of it; no log, no HVG -- see module docstring)
    Z = StandardScaler().fit_transform(df[FEATURES].to_numpy(dtype="float32"))
    Z = np.ascontiguousarray(Z, dtype="float64")

    # --- batch correction: park. Every stadium has its own Hawk-Eye install
    # with its own calibration, and altitude/air density genuinely shift break.
    # That is the technical covariate here -- the analogue of the 10x run.
    # OFF BY DEFAULT -- measured, not assumed. Park *looks* like a batch effect
    # (same-park neighbours run 2.7x chance) but that is almost entirely the
    # pitcher-composition confound: a pitcher throws ~60% of his pitches at one
    # park, so "same park" is mostly "same pitcher". Restricting to neighbour
    # pairs from DIFFERENT pitchers collapses the enrichment to 1.12x chance
    # (0.0376 vs 0.0337), while same-pitcher pairs run 53x. Correcting on
    # home_team therefore removes pitcher signal -- the thing that makes an
    # archetype an archetype -- to fix a batch effect that isn't there.
    # Set ATLAS_HARMONY=1 to enable and re-measure if that ever changes.
    shift = float("nan")
    if not os.environ.get("ATLAS_HARMONY"):
        Z = Z.astype("float32")
    else:
        import harmonypy
        meta = df[["home_team"]].reset_index(drop=True)
        ho = harmonypy.run_harmony(Z, meta, ["home_team"], max_iter_harmony=20,
                                   random_state=SEED, verbose=False)
        # harmonypy >=2.0 returns Z_corr as (cells, dims); 0.0.x returned it
        # transposed, so orient defensively rather than assuming either.
        Zc = np.asarray(ho.Z_corr)
        if Zc.shape != Z.shape:
            Zc = Zc.T
        if Zc.shape != Z.shape:
            raise RuntimeError(f"Harmony returned {Zc.shape}, expected {Z.shape}")
        Zc = np.ascontiguousarray(Zc, dtype="float32")
        shift = float(np.linalg.norm(Zc - Z, axis=1).mean())
        log(f"  {hand}HP Harmony done | mean per-pitch shift {shift:.3f} sd "
            f"(features are z-scored, so this is in sd units)")
        Z = Zc

    df = df.copy()
    df["harmony_shift"] = shift

    # --- kNN graph, then Leiden. Edge weights are UMAP's fuzzy simplicial set,
    # the same connectivity scanpy hands to Leiden.
    log(f"  {hand}HP building kNN graph (k={K_NEIGHBORS}) ...")
    index = NNDescent(Z, n_neighbors=K_NEIGHBORS, metric="euclidean",
                      random_state=SEED)
    knn_idx, knn_dist = index.neighbor_graph
    conn = fuzzy_simplicial_set(
        Z, n_neighbors=K_NEIGHBORS, random_state=np.random.RandomState(SEED),
        metric="euclidean", knn_indices=knn_idx, knn_dists=knn_dist,
    )
    if isinstance(conn, tuple):
        conn = conn[0]
    conn = conn.tocoo()

    upper = conn.row < conn.col          # de-duplicate the symmetric graph
    edges = list(zip(conn.row[upper].tolist(), conn.col[upper].tolist()))
    weights = conn.data[upper].tolist()
    g = ig.Graph(n=Z.shape[0], edges=edges)
    g.es["weight"] = weights

    log(f"  {hand}HP Leiden overcluster (res={RES_OVERCLUSTER}) on "
        f"{g.vcount():,} nodes / {g.ecount():,} edges ...")
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=RES_OVERCLUSTER, seed=SEED,
    )
    df["leiden"] = np.asarray(part.membership)
    log(f"  {hand}HP -> {df.leiden.nunique()} partitions")

    # --- UMAP for visualization only (never the space clustering happens in,
    # same discipline as clustering the scVI latent rather than the UMAP)
    log(f"  {hand}HP UMAP ...")
    emb = umap_mod.UMAP(n_neighbors=K_NEIGHBORS, min_dist=0.1,
                        metric="euclidean", random_state=SEED,
                        verbose=False).fit_transform(Z)
    df["x"], df["y"] = emb[:, 0], emb[:, 1]
    return df, g


# ==========================================================================
# Stage 5 -- annotation by majority vote over overclustered partitions
# ==========================================================================
def majority_vote(df):
    """Denoise Statcast's per-pitch calls using partition majorities.

    This is CellTypist's majority-vote mode: overcluster, then let each
    partition's dominant reference label overwrite its members'. Disagreement
    between a pitch's own call and its partition's majority is the flagged
    ambiguous/misclassified pitch -- the Scrublet step, for free.
    """
    maj = (df.groupby("leiden").pitch_type
             .agg(lambda s: s.value_counts().idxmax()))
    purity = (df.groupby("leiden").pitch_type
                .agg(lambda s: s.value_counts(normalize=True).iloc[0]))
    df = df.copy()
    df["label_denoised"] = df.leiden.map(maj)
    df["partition_purity"] = df.leiden.map(purity)
    df["label_disagree"] = df.label_denoised != df.pitch_type
    return df


# ==========================================================================
# Stage 6-7 -- subcluster one family, score its substates
# ==========================================================================
GENERAL_OF = {
    "FF": "Fastball", "SI": "Fastball", "FC": "Fastball",
    "CH": "Offspeed", "FS": "Offspeed", "FO": "Offspeed",
    "CU": "Breaking", "KC": "Breaking", "CS": "Breaking",
    "SL": "Breaking", "ST": "Breaking", "SV": "Breaking",
    "KN": "Knuckleball",
    "EP": "Other Pitches", "FA": "Other Pitches", "PO": "Other Pitches",
    "UN": "Other Pitches",
}


def subcluster(df, hand, family=SUBCLUSTER_FAMILY):
    """Re-embed one pitch family and Leiden it at low resolution.

    The microglia->DAM move: isolate a parent population, subcluster it, then
    score the substates with a signature. Here the signature is ddz_delta_az.
    """
    from sklearn.preprocessing import StandardScaler
    from pynndescent import NNDescent
    import igraph as ig
    import leidenalg
    from umap.umap_ import fuzzy_simplicial_set

    sub = df[df.general == family].copy()
    if len(sub) < 1000:
        log(f"  {hand}HP subcluster skipped ({family}: {len(sub)} pitches)")
        return None

    # SUBCLUSTER_FEATURES, not FEATURES: inside one family every pitch shares a
    # kind, so "moves more than its delivery predicts" is exactly the substate
    # axis -- the same reason DAM is called on a within-microglia signature.
    Z = StandardScaler().fit_transform(sub[SUBCLUSTER_FEATURES].to_numpy(dtype="float32"))
    Z = np.ascontiguousarray(Z, dtype="float64")
    index = NNDescent(Z, n_neighbors=K_NEIGHBORS, metric="euclidean",
                      random_state=SEED)
    knn_idx, knn_dist = index.neighbor_graph
    conn = fuzzy_simplicial_set(
        Z, n_neighbors=K_NEIGHBORS, random_state=np.random.RandomState(SEED),
        metric="euclidean", knn_indices=knn_idx, knn_dists=knn_dist,
    )
    if isinstance(conn, tuple):
        conn = conn[0]
    conn = conn.tocoo()
    upper = conn.row < conn.col
    g = ig.Graph(n=Z.shape[0],
                 edges=list(zip(conn.row[upper].tolist(), conn.col[upper].tolist())))
    g.es["weight"] = conn.data[upper].tolist()
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=RES_SUBCLUSTER, seed=SEED,
    )
    sub["subcluster"] = np.asarray(part.membership)
    log(f"  {hand}HP {family} subcluster (res={RES_SUBCLUSTER}) -> "
        f"{sub.subcluster.nunique()} substates")
    return sub


def marker_features(df, group_col):
    """Wilcoxon rank-sum per group vs. rest, per feature.

    scanpy's rank_genes_groups with method='wilcoxon', same test -- this is how
    a cluster gets named ("high velo, low ride, long extension") rather than
    left as an integer.
    """
    from scipy.stats import mannwhitneyu

    rows = []
    for grp, g in df.groupby(group_col):
        rest = df[df[group_col] != grp]
        if len(g) < 20 or len(rest) < 20:
            continue
        for f in SUBCLUSTER_FEATURES:
            a, b = g[f].to_numpy(), rest[f].to_numpy()
            try:
                u, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                continue
            # rank-biserial correlation: bounded, sign = direction
            eff = 2 * u / (len(a) * len(b)) - 1
            rows.append({group_col: grp, "feature": f, "n": len(g),
                         "effect": eff, "p": p,
                         "mean": float(a.mean()),
                         "mean_rest": float(b.mean())})
    return pd.DataFrame(rows)


# ==========================================================================
# Run both hands
# ==========================================================================
HANDS = [("R", "Right-handed pitchers"), ("L", "Left-handed pitchers")]

panels, subpanels, qc_reports, marker_tables = {}, {}, {}, {}
for hand, title in HANDS:
    log(f"=== {hand}HP ===")
    d = raw[raw.p_throws == hand].copy()
    d, rep = qc_filter(d)
    qc_reports[hand] = rep
    log(f"  {hand}HP QC: {rep['input']:,} -> {rep['kept']:,} ({rep['pct_kept']}% kept) "
        f"| bounds {sum(v for k, v in rep.items() if k.startswith('drop_') and k != 'drop_isolated'):,}"
        f" | isolated {rep['drop_isolated']:,}")

    d, _ = embed_and_cluster(d, hand)
    d = majority_vote(d)
    d["general"] = d.pitch_type.map(GENERAL_OF)
    d["general_denoised"] = d.label_denoised.map(GENERAL_OF)
    dis = d.label_disagree.mean()
    log(f"  {hand}HP majority-vote: {100*dis:.2f}% of pitches disagree with "
        f"their partition's majority label")
    panels[hand] = d

    s = subcluster(d, hand)
    if s is not None:
        subpanels[hand] = s
        marker_tables[hand] = marker_features(s, "subcluster")

# A sampled run is a smoke test, not the atlas -- keep its parquet out of
# data/meta so it can never be mistaken for the real thing downstream.
PARQUET_DIR = OUTDIR if SAMPLE else paths.META

out = pd.concat(panels.values(), ignore_index=True)
out_path = PARQUET_DIR / "pitch_atlas.parquet"
out.to_parquet(out_path)
log(f"wrote {out_path} ({len(out):,} rows, {out.shape[1]} cols)")

if subpanels:
    sub_all = pd.concat(subpanels.values(), ignore_index=True)
    sub_path = PARQUET_DIR / "pitch_atlas_subclusters.parquet"
    sub_all.to_parquet(sub_path)
    log(f"wrote {sub_path} ({len(sub_all):,} rows)")

# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
report_lines = ["# Pitch atlas -- QC and annotation report", ""]
report_lines.append("## QC")
report_lines.append(pd.DataFrame(qc_reports).to_string())
report_lines.append("")
report_lines.append("## Statcast label vs. partition majority (disagreement rate)")
report_lines.append(
    "absorbed=True means the type never wins a majority in ANY partition, so its\n"
    "100% disagreement is an artifact of majority-voting, not evidence that\n"
    "Statcast is wrong. This is the known failure mode of CellTypist's\n"
    "majority-vote mode erasing rare populations: read the disagreement rate\n"
    "only for absorbed=False types."
)
for hand in panels:
    d = panels[hand]
    winners = set(d.label_denoised.unique())
    tab = (d.groupby("pitch_name")
             .agg(n=("label_disagree", "size"),
                  disagree_pct=("label_disagree", lambda s: round(100 * s.mean(), 1)),
                  mean_purity=("partition_purity", lambda s: round(s.mean(), 3)),
                  pitch_type=("pitch_type", "first"))
             .sort_values("n", ascending=False))
    tab["absorbed"] = ~tab.pitch_type.isin(winners)
    real = d[d.pitch_type.isin(winners)]
    report_lines += ["", f"### {hand}HP  (overall {100*d.label_disagree.mean():.2f}%; "
                     f"excluding absorbed types "
                     f"{100*real.label_disagree.mean():.2f}%)",
                     tab.drop(columns="pitch_type").to_string()]
for hand, mk in marker_tables.items():
    if mk.empty:
        continue
    top = (mk.reindex(mk.effect.abs().sort_values(ascending=False).index)
             .groupby("subcluster").head(3)
             .sort_values(["subcluster", "effect"]))
    report_lines += ["", f"## {hand}HP {SUBCLUSTER_FAMILY} substate markers "
                     f"(top 3 by |rank-biserial|)", top.to_string(index=False)]

report_path = OUTDIR / "pitch_atlas_report.txt"
report_path.write_text("\n".join(report_lines) + "\n")
log(f"wrote {report_path}")
print("\n".join(report_lines))

# `raw` is the full pre-QC frame and is no longer needed; the panels carry
# everything the figures use. Drop it before the rasterized point clouds.
del raw
gc.collect()

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from adjustText import adjust_text

plt.rcParams.update({
    "font.family": "DejaVu Sans",  # Helvetica/Arial not installed on this host
    "axes.linewidth": 1, "axes.edgecolor": "black", "axes.grid": False,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.labelsize": 14, "axes.titlesize": 15,
    "legend.fontsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})

GENERAL_PALETTE = {
    "Fastball": "#C0392B", "Offspeed": "#4CA64C", "Breaking": "#D1A32E",
    "Knuckleball": "#4B3FC4", "Other Pitches": "#8C8C8C",
}
GENERAL_ORDER = ["Fastball", "Offspeed", "Breaking", "Knuckleball", "Other Pitches"]
MIN_LABEL_N = 100


# Rasterized pixels allowed per figure. The scatter layers are rasterized (the
# axes/text stay vector), and a rasterized artist allocates a full-figure buffer
# at save dpi -- measured at ~16 bytes/px, so an 18x10.5in figure at dpi=600
# costs 1.1GB for one collection and 2.5GB across six panels. That is what was
# OOM-killing figure 5. dpi only sets the resolution of the RASTERIZED layer in
# a PDF (text and axes are vector at any dpi), and ~400dpi is already past
# useful for a dense alpha-blended point cloud, so capping costs no real quality.
PIXEL_BUDGET = 30e6


def save(fig, name):
    w, h = fig.get_size_inches()
    cap = int((PIXEL_BUDGET / (w * h)) ** 0.5)
    for ext, dpi in (("pdf", 600), ("png", 300)):
        eff = min(dpi, cap)
        fig.savefig(OUTDIR / f"{name}.{ext}", dpi=eff, bbox_inches="tight")
        gc.collect()
    log(f"wrote {OUTDIR / name}.{{pdf,png}}  (raster dpi cap {cap})")


def label_clusters(ax, panel, label_col, coord=("x", "y")):
    cen = (panel.groupby(label_col)
                .agg(x=(coord[0], "median"), y=(coord[1], "median"), n=(coord[0], "size"))
                .query("n >= @MIN_LABEL_N"))
    if cen.empty:
        return
    texts = [ax.text(r.x, r.y, str(r.Index), fontsize=8, ha="center", va="center",
                     path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
             for r in cen.itertuples()]
    adjust_text(texts, ax=ax, expand=(1.3, 1.6),
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")


# --- Figure 1: the atlas, Savant classes + subcategory labels --------------
fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8))
for ax, (hand, title) in zip(axes, HANDS):
    p = panels[hand]
    for g in reversed([g for g in GENERAL_ORDER if g in set(p.general)]):
        m = p.general == g
        ax.scatter(p.loc[m, "x"], p.loc[m, "y"], s=2.5, alpha=0.3,
                   color=GENERAL_PALETTE[g], edgecolors="none", rasterized=True)
    label_clusters(ax, p, "pitch_name")
    ax.set_title(f"{title}  (n = {len(p):,})", fontsize=13)
    bare(ax)
handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=GENERAL_PALETTE[g],
                      markersize=8, label=g)
           for g in GENERAL_ORDER if g in set(out.general)]
axes[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                frameon=False, title="Pitch class", handletextpad=0.6, labelspacing=0.7)
fig.suptitle(f"Pitch-level UMAP — every pitch, 2026 season (n = {len(out):,})",
             fontsize=17, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.97))
save(fig, "pitch_umap")
plt.close(fig)

# --- Figure 2: annotation QC ----------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14.5, 12))
for row, (hand, title) in enumerate(HANDS):
    p = panels[hand]
    ax = axes[row, 0]
    rng = np.random.RandomState(SEED)
    shuffled = rng.permutation(p.leiden.nunique())
    cmap = plt.get_cmap("tab20")
    colors = [cmap(shuffled[c] % 20) for c in p.leiden]
    ax.scatter(p.x, p.y, s=2, alpha=0.35, c=colors, edgecolors="none",
               rasterized=True)
    label_clusters(ax, p, "leiden")
    ax.set_title(f"{title} — Leiden res {RES_OVERCLUSTER} "
                 f"({p.leiden.nunique()} partitions)", fontsize=12)
    bare(ax)

    ax = axes[row, 1]
    ax.scatter(p.loc[~p.label_disagree, "x"], p.loc[~p.label_disagree, "y"],
               s=2, alpha=0.15, color="#CCCCCC", edgecolors="none", rasterized=True)
    m = p.label_disagree
    ax.scatter(p.loc[m, "x"], p.loc[m, "y"], s=3, alpha=0.5, color="#C0392B",
               edgecolors="none", rasterized=True)
    ax.set_title(f"{title} — Statcast call ≠ partition majority "
                 f"({100*m.mean():.1f}%)", fontsize=12)
    bare(ax)
fig.suptitle("De novo partitions vs. Statcast's classifier", fontsize=17, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.975))
save(fig, "pitch_annotation_qc")
plt.close(fig)

# --- Figure 3: confusion, denoised label x Statcast label ------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
for ax, (hand, title) in zip(axes, HANDS):
    p = panels[hand]
    ct = pd.crosstab(p.pitch_name, p.label_denoised.map(
        lambda t: p.pitch_name[p.pitch_type == t].iloc[0] if (p.pitch_type == t).any() else t),
        normalize="index")
    im = ax.imshow(ct.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index, fontsize=8)
    ax.set_xlabel("partition-majority label"); ax.set_ylabel("Statcast label")
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row fraction")
fig.suptitle("Statcast label vs. majority-vote denoised label", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pitch_label_confusion")
plt.close(fig)

# --- Figure 4: subclustered family + DDZ signature ------------------------
if subpanels:
    ncol = 2 if HAVE_DDZ else 1
    fig, axes = plt.subplots(len(subpanels), ncol,
                             figsize=(7.2 * ncol, 6.0 * len(subpanels)),
                             squeeze=False)
    for row, (hand, s) in enumerate(subpanels.items()):
        ax = axes[row][0]
        cmap = plt.get_cmap("tab10")
        for c in sorted(s.subcluster.unique()):
            m = s.subcluster == c
            ax.scatter(s.loc[m, "x"], s.loc[m, "y"], s=3, alpha=0.4,
                       color=cmap(c % 10), edgecolors="none", rasterized=True,
                       label=f"{c} (n={int(m.sum()):,})")
        label_clusters(ax, s, "subcluster")
        ax.set_title(f"{hand}HP {SUBCLUSTER_FAMILY} — Leiden res {RES_SUBCLUSTER}",
                     fontsize=13)
        bare(ax)
        ax.legend(frameon=False, fontsize=8, markerscale=3, loc="best")

        if HAVE_DDZ:
            ax = axes[row][1]
            v = s.ddz_delta_az
            lim = np.nanpercentile(np.abs(v), 98)
            sc = ax.scatter(s.x, s.y, s=3, alpha=0.5, c=v, cmap="RdBu_r",
                            vmin=-lim, vmax=lim, edgecolors="none", rasterized=True)
            ax.set_title(f"{hand}HP — DDZ signature (ddz_delta_az)", fontsize=13)
            bare(ax)
            fig.colorbar(sc, ax=ax, fraction=0.046,
                         label="ft/s²  (+ = more ride than expected)")
    fig.suptitle(f"{SUBCLUSTER_FAMILY} substates and their signature score",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "pitch_substates")
    plt.close(fig)

# --- Figure 5: expected vs actual movement, per pitch ---------------------
# The quantity itself, before any embedding: what each pitch did against what
# its kind-and-delivery predicted.
fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
for row, (hand, title) in enumerate(HANDS):
    # Draw commonest first so the rare classes still land on top, which the
    # per-category loop used to get from plot order.
    p = panels[hand].iloc[
        panels[hand].general.map(panels[hand].general.value_counts())
                    .to_numpy().argsort()[::-1]
    ]
    for col, c in enumerate(MOVEMENT):
        ax = axes[row][col]
        lim = np.nanpercentile(np.abs(np.r_[p[c], p[f"expected_{c}"]]), 99.5)
        ax.axline((0, 0), slope=1, color="#444444", lw=1, ls="--", zorder=3)
        # ONE rasterized collection, not one per category: each rasterized
        # artist allocates a full-figure buffer at save dpi, and this figure has
        # the most panels -- looping per category here OOMs even on 8k rows.
        ax.scatter(p[f"expected_{c}"], p[c], s=2, alpha=0.25,
                   c=p.general.map(GENERAL_PALETTE).to_numpy(),
                   edgecolors="none", rasterized=True)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"expected {c}  (ft/s², from pitch type + delivery)")
        ax.set_ylabel(f"actual {c}  (ft/s²)")
        ax.set_title(f"{hand}HP — {c}: actual vs expected", fontsize=12)

    ax = axes[row][2]
    lim = np.nanpercentile(np.abs(p.ddz_delta_az), 98)
    sc = ax.scatter(p.x, p.y, s=2.5, alpha=0.45, c=p.ddz_delta_az, cmap="RdBu_r",
                    vmin=-lim, vmax=lim, edgecolors="none", rasterized=True)
    label_clusters(ax, p, "pitch_name")
    ax.set_title(f"{hand}HP — ddz_delta_az on the atlas", fontsize=12)
    bare(ax)
    fig.colorbar(sc, ax=ax, fraction=0.046,
                 label="ddz_delta_az, ft/s²  (+ = more ride than predicted)")
fig.suptitle("Expected vs. actual movement, per pitch", fontsize=17, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "pitch_expected_vs_actual")
plt.close(fig)

log("done")
