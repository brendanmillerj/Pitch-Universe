"""Benchmark of the Pitch Universe maps: UMAP against t-SNE, PaCMAP, PHATE, PCA and the movement plot.

Version 2 (2026-11-04), rebuilt after three independent reviews of the first draft. One row = one
pitcher-season's version of one pitch type (MLB 2024-2026, 7,889 rows). Maps are drawn from the 27
MAP_FEATURES, z-scored within pitch type, block-scaled, PCA (27 components, so the "full space" is the
scaled feature space itself). Every number is reported with a pitcher-clustered bootstrap interval;
"self-free" neighbour scores exclude a pitcher's own other seasons from both truth and view.

Tests (each written to analysis/umap_benchmark/ as CSV / parquet; figures come from umap_benchmark_figures.py):

  A  views      neighbour preservation of 2D/3D pictures on equal footing: recall@k (self-free and all),
                recall-over-k curves, trustworthiness / continuity, co-ranking Q_local / Q_global (Lee &
                Verleysen), distance rank-correlation on a fixed pair set, agreement with full-space
                Leiden islands (kNN accuracy, ARI), same-pitcher-other-season recovery by pitch count,
                handedness / season mixing, sample-size drift; truth sensitivity (no block weights, cosine).
  B  held-out   refresh-style: fit on every first-half profile, place each second half (library
                transform; inverse-distance mean of 15 nearest first halves excluding / including the
                pitcher's own rows), rank it against the embedded first halves; is his own first half
                nearest? Stranger-style (pitcher folds, both halves placed) kept as a secondary test.
  C  outcomes   whiff / CSW / xwOBA-on-contact / run value (shrunk and raw) predicted from the k nearest
                dots of other pitchers (k = 10, 30, 100); sensitivity without the intent block.
  D  stability  five seeds (neighbourhood overlap, island share, Procrustes), and a real refresh:
                layout fit on first-half profiles, refit on full-season profiles, cold start vs warm
                start (init from the previous layout) vs AlignedUMAP; 2D and 3D.
  E  settings   UMAP grids (2D and 3D), PaCMAP and t-SNE grids, output dimensions.
  G  global     the All-pitches map: same metrics plus pitch-type purity / kNN accuracy; refresh and
                held-out tests for the main methods.

Per-type work runs in parallel (one process per pitch type).

Version 3 (2026-11-04, evening): the truth is the atlas's new identity space (15 measures: shape, derived, delivery,
near-duplicates dropped; atlas_build.MAP_FEATURES); the old 27-measure space is a comparison; section L measures
"same pitch, different location".

Usage:  python scripts/umap_benchmark.py                        (full; ~30-40 min on 8 CPUs)
        UMAP_BENCH_FAST=1 python scripts/umap_benchmark.py     (two pitch types, small grids)
"""
import os

for _v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import procrustes  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import Isomap, trustworthiness  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlbpitch import paths, profiles  # noqa: E402
from profile_embed import block_scale, zscore  # noqa: E402
from atlas_build import MAP_BLOCKS, MAP_FEATURES, scaler_fit, scaler_apply  # noqa: E402

warnings.filterwarnings("ignore")
FAST = bool(os.environ.get("UMAP_BENCH_FAST"))
K = 10
SEED = 0
N_PCS = 30
RANGE = f"{min(profiles.SEASONS)}_{max(profiles.SEASONS)}"
OUT = paths.ROOT / "analysis" / "umap_benchmark"
OUTCOMES = ["whiff_pct", "csw_pct", "xwoba_contact", "rv100", "rv100_raw"]
SAVANT5 = ["release_speed", "pfx_z", "pfx_x", "release_pos_z", "release_extension"]
# the atlas's space before 2026-11-04 (27 measures: + location, scatter and the duplicate columns), and the
# identity space with location added back: the comparisons for the location story
OLD_BLOCKS = {**profiles.MEAN_FEATURES, "consistency": profiles.SD_FEATURES}
OLD_FEATURES = [c for cs in OLD_BLOCKS.values() for c in cs]
WITH_LOC = {**MAP_BLOCKS, "intent": profiles.MEAN_FEATURES["intent"]}
WITH_LOC_FEATURES = [c for cs in WITH_LOC.values() for c in cs]
K_CURVE = [1, 2, 5, 10, 20, 30, 50, 100, 200]
SEEDS = [0, 1] if FAST else [0, 1, 2, 3, 4]
N_BOOT = 200 if FAST else 1000
WORKERS = int(os.environ.get("UMAP_BENCH_WORKERS", "2" if FAST else "8"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ fitters (one registry for every test)
def fit_umap(P, d, seed, nn=15, md=0.3, init="spectral", densmap=False):
    import umap
    red = umap.UMAP(n_components=d, n_neighbors=nn, min_dist=md, init=init, random_state=seed, densmap=densmap).fit(P)
    return red.embedding_, red


def fit_tsne(P, d, seed, perp=30, exag=None, init=None, uniform_k=None):
    import openTSNE
    from openTSNE import affinity
    kw = dict(n_components=d, perplexity=perp, random_state=seed, n_jobs=1, negative_gradient_method="bh")
    if exag:
        kw["exaggeration"] = exag
    if init is not None:                                 # warm start from a previous layout: no early exaggeration
        kw["early_exaggeration_iter"] = 0
    aff = affinity.Uniform(P, k_neighbors=uniform_k, random_state=seed, n_jobs=1) if uniform_k else None
    E = openTSNE.TSNE(**kw).fit(P, affinities=aff, initialization=init)
    return np.asarray(E, dtype="float64"), E


def fit_pacmap(P, d, seed, nn=15, mn=0.5, local=False, init="pca"):
    import pacmap
    cls = pacmap.LocalMAP if local else pacmap.PaCMAP
    r = cls(n_components=d, n_neighbors=nn, MN_ratio=mn, random_state=seed, save_tree=True)
    return r.fit_transform(P, init=init), r


def fit_phate(P, d, seed):
    import phate
    op = phate.PHATE(n_components=d, knn=15, n_jobs=1, random_state=seed, verbose=0)
    return op.fit_transform(P), op


def fit_trimap(P, d, seed):
    import trimap
    np.random.seed(seed)
    return trimap.TRIMAP(n_dims=d).fit_transform(P), None


def fit_isomap(P, d, seed):
    return Isomap(n_neighbors=15, n_components=d).fit_transform(P), None


def place(obj, Pnew, Ptr):
    """The library's own out-of-sample placement."""
    import pacmap
    if hasattr(obj, "embedding_"):                       # umap
        return obj.transform(Pnew)
    if isinstance(obj, (pacmap.PaCMAP, pacmap.LocalMAP)):
        return obj.transform(Pnew, basis=Ptr)
    return np.asarray(obj.transform(Pnew), dtype="float64")   # openTSNE, PHATE


TIMES = {}


def timed(name, fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    TIMES.setdefault(name, []).append(time.time() - t0)
    return out


# the methods under comparison: name -> (dims, fitter, kwargs)
def METHODS():
    m = {
        "t-SNE 2D": (2, fit_tsne, {}),
        "t-SNE 2D, exag 4": (2, fit_tsne, {"exag": 4}),
        "t-SNE 2D, uniform k=15": (2, fit_tsne, {"uniform_k": 15}),
        "t-SNE 2D, perp 10": (2, fit_tsne, {"perp": 10}),
        "t-SNE 2D, perp 50": (2, fit_tsne, {"perp": 50}),
        "t-SNE 3D": (3, fit_tsne, {}),
        "UMAP 2D (15, 0.3)": (2, fit_umap, {}),
        "UMAP 2D (15, 0.3), PCA init": (2, fit_umap, {"init": "pca"}),
        "UMAP 2D (10, 0.1)": (2, fit_umap, {"nn": 10, "md": 0.1}),
        "UMAP 2D (5, 0.0)": (2, fit_umap, {"nn": 5, "md": 0.0}),
        "UMAP 3D (15, 0.3)": (3, fit_umap, {}),
        "UMAP 3D (10, 0.1)": (3, fit_umap, {"nn": 10, "md": 0.1}),
        "densMAP 3D (15, 0.3)": (3, fit_umap, {"densmap": True}),
        "PaCMAP 2D (10)": (2, fit_pacmap, {"nn": 10}),
        "PaCMAP 2D (15)": (2, fit_pacmap, {}),
        "PaCMAP 3D (15)": (3, fit_pacmap, {}),
        "LocalMAP 2D (15)": (2, fit_pacmap, {"local": True}),
        "PHATE 2D": (2, fit_phate, {}),
        "PHATE 3D": (3, fit_phate, {}),
        "TriMap 2D": (2, fit_trimap, {}),
        "Isomap 2D": (2, fit_isomap, {}),
    }
    if FAST:
        for k in ["t-SNE 2D, perp 10", "t-SNE 2D, perp 50", "LocalMAP 2D (15)", "TriMap 2D", "PHATE 2D"]:
            m.pop(k)
    return m


# the subset carried through the held-out, outcome and stability tests
CORE = ["UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "UMAP 3D (15, 0.3)", "UMAP 3D (10, 0.1)",
        "t-SNE 2D", "t-SNE 2D, exag 4", "t-SNE 3D", "PaCMAP 2D (15)", "PaCMAP 3D (15)"]
STAB = {"UMAP (15, 0.3)": (fit_umap, {}), "UMAP (10, 0.1)": (fit_umap, {"nn": 10, "md": 0.1}),
        "t-SNE (perp 30)": (fit_tsne, {}), "t-SNE (exag 4)": (fit_tsne, {"exag": 4}), "PaCMAP (15)": (fit_pacmap, {})}


# ------------------------------------------------------------------ metric helpers
def knn_idx(X, k):
    k = min(k, len(X) - 1)
    return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]


def knn_idx_free(X, k, pid):
    """k nearest excluding rows of the same pitcher (self-free)."""
    kq = min(len(X) - 1, k + 12)
    nb = knn_idx(X, kq)
    out = np.full((len(X), k), -1)
    for r in range(len(X)):
        o = nb[r][pid[nb[r]] != pid[r]][:k]
        out[r, :len(o)] = o
    return out


def row_recall(truth, view):
    """Per-row share of true neighbours also among the view's neighbours (padding -1 ignored)."""
    return np.array([len(set(a[a >= 0]) & set(b[b >= 0])) / max(1, (a >= 0).sum()) for a, b in zip(truth, view)])


def within_space(F, blocks=MAP_BLOCKS):
    X = block_scale(zscore(F.fillna(F.median())), blocks)
    return X, PCA(n_components=min(N_PCS, X.shape[1]), random_state=SEED).fit_transform(X)


def pair_set(n, rng, n_pairs=50_000):
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    m = i != j
    return i[m], j[m]


def dist_spearman(P, V, pairs):
    i, j = pairs
    return float(spearmanr(np.linalg.norm(P[i] - P[j], axis=1), np.linalg.norm(V[i] - V[j], axis=1)).statistic)


def rank_matrix(X):
    """rank[i, j] = rank of j among i's neighbours by distance (1 = nearest), self excluded (0)."""
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)
    R = np.empty_like(order)
    rows = np.arange(len(X))[:, None]
    R[rows, order] = np.arange(1, len(X) + 1)[None, :]
    R[np.arange(len(X)), np.arange(len(X))] = 0
    return R


def coranking(P, V):
    """Q_NX(K) for every K, LCMC, K_max, Q_local, Q_global (Lee & Verleysen 2009; Lee et al. 2013)."""
    n = len(P)
    Rh, Rl = rank_matrix(P), rank_matrix(V)
    mx = np.maximum(Rh, Rl)[~np.eye(n, dtype=bool)]
    counts = np.bincount(mx, minlength=n + 1)[1:n]          # pairs whose both ranks <= K, by max rank
    Ks = np.arange(1, n)
    qnx = np.cumsum(counts) / (Ks * n)
    lcmc = qnx - Ks / (n - 1)
    kmax = int(np.argmax(lcmc)) + 1
    return qnx, kmax, float(qnx[:kmax].mean()), float(qnx[kmax:].mean())


def leiden_labels(P, k=15):
    import igraph as ig
    import leidenalg
    idx = knn_idx(P, k)
    edges = {(min(a, b), max(a, b)) for a, row in enumerate(idx) for b in row}
    part = leidenalg.find_partition(ig.Graph(n=len(P), edges=sorted(edges)),
                                    leidenalg.RBConfigurationVertexPartition, resolution_parameter=1.0, seed=SEED)
    return np.array(part.membership)


def knn_label_accuracy(V, lab, k=10):
    nb = knn_idx(V, k)
    pred = np.array([np.bincount(lab[r]).argmax() for r in nb])
    return float(np.mean(pred == lab))


def idw_place(P_fit, P_new, U_fit, k=15, exclude=None):
    """Inverse-distance mean of the k nearest fitted rows' map positions (the atlas's Triple-A placement).
    `exclude`: boolean (n_new, n_fit) mask of fitted rows a new row may not use (e.g. the pitcher's own rows;
    on the All-pitches map a pitcher can have 24 of them, so the query reaches further)."""
    if exclude is None:
        d, i = NearestNeighbors(n_neighbors=min(len(P_fit), k)).fit(P_fit).kneighbors(P_new)
        w = 1 / (d + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        return (w[:, :, None] * U_fit[i]).sum(axis=1), i
    d, i = NearestNeighbors(n_neighbors=min(len(P_fit), k + 40)).fit(P_fit).kneighbors(P_new)
    out, idx = np.empty((len(P_new), U_fit.shape[1])), np.full((len(P_new), k), -1)
    for r in range(len(P_new)):
        keep = ~exclude[r, i[r]]
        ii, dd = i[r][keep][:k], d[r][keep][:k]
        w = 1 / (dd + 1e-6)
        w /= w.sum()
        out[r] = (w[:, None] * U_fit[ii]).sum(axis=0)
        idx[r, :len(ii)] = ii
    return out, idx


def ranks_of_target(Vq, Vref, target):
    """Rank of each query's target row among reference rows (1 = nearest); ties count half."""
    D = cdist(Vq, Vref)
    own = D[np.arange(len(Vq)), target]
    return (D < own[:, None]).sum(axis=1) + 0.5 * ((D == own[:, None]).sum(axis=1) - 1) + 1


# ------------------------------------------------------------------ per-type work
def run_type(pt, t_all, h_all):
    global TIMES
    TIMES = {}
    rng = np.random.default_rng(SEED)
    g = t_all[t_all.pitch_type == pt].reset_index(drop=True)
    pid = g.pitcher.to_numpy()
    X, P = within_space(g[MAP_FEATURES].astype("float64"))
    res = {}

    # ---------- A: views
    truth_all, truth_free = knn_idx(P, K), knn_idx_free(P, K, pid)
    truth30 = knn_idx(P, 30)
    truth_curve = {k: knn_idx(P, k) for k in K_CURVE}
    lab = leiden_labels(P)
    pairs = pair_set(len(g), rng)
    sav = zscore(g[SAVANT5].astype("float64").fillna(g[SAVANT5].median())).to_numpy()
    views = {
        "Movement (ride x HB)": zscore(pd.DataFrame({"ivb": g.pfx_z, "hb": g.pfx_x})).to_numpy(),
        "Savant 5 (PCA 2)": PCA(n_components=2, random_state=SEED).fit_transform(sav),
        "Savant 5 space (5-D)": sav,
        "PCA 2": P[:, :2],
        "PCA 3": P[:, :3],
    }
    objs = {}
    for name, (d, fn, kw) in METHODS().items():
        views[name], objs[name] = timed(name, fn, P, d, SEED, **kw)
    views["Random 2D"] = rng.normal(size=(len(g), 2))
    views["Full space"] = P

    # other-season recovery: rows whose pitcher has another season of this type
    other = np.array([np.flatnonzero((pid == p) & (np.arange(len(g)) != r)) for r, p in enumerate(pid)], dtype=object)
    has_other = np.array([len(o) > 0 for o in other])
    nq = pd.qcut(g.n, 4, labels=False, duplicates="drop").to_numpy()
    hand, season = g.p_throws.to_numpy(), g.season.to_numpy()
    chance_hand = float((pd.Series(hand).value_counts(normalize=True) ** 2).sum())
    chance_season = float((pd.Series(season).value_counts(normalize=True) ** 2).sum())
    self_rate = float(np.mean(pid[truth_all[:, 0]] == pid))
    self_in10 = float(np.mean(pid[truth_all] == pid[:, None]))

    A_rows, A_long, A_curve, A_cor = [], [], [], []
    for name, V in views.items():
        nb_all, nb_free = knn_idx(V, K), knn_idx_free(V, K, pid)
        r_all, r_free = row_recall(truth_all, nb_all), row_recall(truth_free, nb_free)
        hit = np.where(has_other, [np.isin(o, nb).any() if len(o) else False for o, nb in zip(other, knn_idx(V, K))], np.nan)
        nb10 = knn_idx(V, K)
        same_hand, same_season = float(np.mean(hand[nb10] == hand[:, None])), float(np.mean(season[nb10] == season[:, None]))
        centroid_d = np.linalg.norm(V - V.mean(axis=0), axis=1)
        row = {"type": pt, "n": len(g), "view": name, "dims": V.shape[1],
               "recall10_free": float(r_free.mean()), "recall10_all": float(r_all.mean()),
               "recall30_all": float(row_recall(truth30, knn_idx(V, 30)).mean()),
               "trust10": float(trustworthiness(P, V, n_neighbors=K)) if name != "Full space" else 1.0,
               "continuity10": float(trustworthiness(V, P, n_neighbors=K)) if name != "Full space" else 1.0,
               "dist_spearman": dist_spearman(P, V, pairs),
               "island_knn_acc": knn_label_accuracy(V, lab),
               "island_ari": float(adjusted_rand_score(lab, leiden_labels(V))),
               "other_season_top10": float(np.nanmean(hit)),
               "same_hand": same_hand, "chance_hand": chance_hand, "same_season": same_season, "chance_season": chance_season,
               "n_drift_spearman": float(spearmanr(g.n, centroid_d).statistic),
               "fit_seconds": float(np.median(TIMES[name])) if name in TIMES else 0.0}
        if V.shape[1] <= 5 and name != "Full space":
            qnx, kmax, ql, qg = coranking(P, V)
            row.update({"q_local": ql, "q_global": qg, "k_max": kmax})
            A_cor.append(pd.DataFrame({"type": pt, "view": name, "K": np.arange(1, len(g)), "qnx": qnx}))
            for k in K_CURVE:
                if k < len(g):
                    A_curve.append({"type": pt, "n": len(g), "view": name, "k": k,
                                    "recall": float(row_recall(truth_curve[k], knn_idx(V, k)).mean())})
        A_rows.append(row)
        A_long.append(pd.DataFrame({"type": pt, "row": np.arange(len(g)), "pitcher": pid, "n": g.n.to_numpy(), "nq": nq,
                                    "view": name, "recall10_free": r_free, "recall10_all": r_all, "other_season_hit": hit}))
    res["A_views"] = pd.DataFrame(A_rows).assign(self_rate_nn1=self_rate, self_share_in10=self_in10)
    # coordinates of the main 2D maps + the full space, for the figures (Shepard diagrams, example maps)
    coords = g[["pitcher", "season", "pitch_type", "player_name", "p_throws", "n", "release_speed", "pfx_x", "pfx_z"]].copy()
    coords["island"] = lab
    for j in range(P.shape[1]):
        coords[f"p{j + 1}"] = P[:, j]
    for name in ["PCA 2", "t-SNE 2D", "t-SNE 2D, exag 4", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "PaCMAP 2D (15)", "PHATE 2D", "Movement (ride x HB)"]:
        if name in views:
            coords[f"{name}|x"], coords[f"{name}|y"] = views[name][:, 0], views[name][:, 1]
    res["A_coords"] = coords
    res["A_rows"] = pd.concat(A_long, ignore_index=True)
    res["A_curve"] = pd.DataFrame(A_curve)
    res["A_coranking"] = pd.concat(A_cor, ignore_index=True)
    # truth sensitivity: other definitions of "truly nearest"
    _, P_old = within_space(g[OLD_FEATURES].astype("float64"), OLD_BLOCKS)
    alt = {"block-weighted (default)": P,
           "old atlas space (27: + location, scatter, duplicates)": P_old,
           "plain z-score (no block weights)": zscore(g[MAP_FEATURES].astype("float64").fillna(g[MAP_FEATURES].median())).to_numpy(),
           "cosine on block-weighted": X / np.linalg.norm(X, axis=1, keepdims=True)}
    sens = []
    for tname, T in alt.items():
        tr = knn_idx_free(T, K, pid)
        for name in ["Movement (ride x HB)", "PCA 2", "t-SNE 2D", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "UMAP 3D (15, 0.3)", "PaCMAP 2D (15)"]:
            sens.append({"type": pt, "n": len(g), "truth": tname, "view": name,
                         "recall10_free": float(row_recall(tr, knn_idx_free(views[name], K, pid)).mean())})
    res["A_truth_sensitivity"] = pd.DataFrame(sens)
    # PCA dimensions needed
    res["A_pca_dims"] = pd.DataFrame([{"type": pt, "n": len(g), "dims": d,
                                       "recall10_free": float(row_recall(truth_free, knn_idx_free(P[:, :d], K, pid)).mean())}
                                      for d in [2, 3, 4, 5, 7, 10, 15, 20, 27] if d <= P.shape[1]])
    log(f"A {pt}: " + " ".join(f"{r['view']} {100 * r['recall10_free']:.0f}" for r in A_rows
                               if r["view"] in ("PCA 2", "t-SNE 2D", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "UMAP 3D (15, 0.3)", "t-SNE 3D", "PaCMAP 3D (15)", "PHATE 3D")))

    # ---------- C: outcomes from neighbours (views from A)
    Y = g[OUTCOMES].to_numpy(dtype="float64")
    C_rows = []
    c_views = ["Full space", "Savant 5 space (5-D)", "Movement (ride x HB)", "PCA 2", "t-SNE 2D", "t-SNE 3D", "UMAP 2D (15, 0.3)",
               "UMAP 2D (10, 0.1)", "UMAP 3D (15, 0.3)", "UMAP 3D (10, 0.1)", "PaCMAP 2D (15)", "PaCMAP 3D (15)", "PHATE 3D", "Random 2D"]
    for name in c_views:
        V = views[name]
        for k in [10, 30, 100]:
            nb = knn_idx_free(V, k, pid)
            pred = np.array([Y[r[r >= 0]].mean(axis=0) for r in nb])
            for c, oc in enumerate(OUTCOMES):
                C_rows.append(pd.DataFrame({"type": pt, "row": np.arange(len(g)), "pitcher": pid, "view": name, "k": k,
                                            "outcome": oc, "actual": Y[:, c], "pred": pred[:, c]}))
    # sensitivity: location added back (identity + intent), and the old 27-measure space
    _, Pl = within_space(g[WITH_LOC_FEATURES].astype("float64"), WITH_LOC)
    Ul, _ = timed("t-SNE 3D + location", fit_tsne, Pl, 3, SEED)
    for name, V in {"Full space, + location": Pl, "t-SNE 3D, + location": Ul, "Full space, old 27 measures": P_old}.items():
        nb = knn_idx_free(V, 30, pid)
        pred = np.array([Y[r[r >= 0]].mean(axis=0) for r in nb])
        for c, oc in enumerate(OUTCOMES):
            C_rows.append(pd.DataFrame({"type": pt, "row": np.arange(len(g)), "pitcher": pid, "view": name, "k": 30,
                                        "outcome": oc, "actual": Y[:, c], "pred": pred[:, c]}))
    res["C_rows"] = pd.concat(C_rows, ignore_index=True)
    log(f"C {pt}: done")

    # ---------- L: same pitch, different location. For every pitch, his value minus the mean of his 30 nearest
    # other-pitcher neighbours, in the identity space, the old 27-measure space and on the built map (t-SNE 3D)
    L_rows = []
    gap_cols = {"d_height_in": 12 * g.plate_z.to_numpy(dtype="float64"), "d_zone_pts": 100 * g.in_zone.to_numpy(dtype="float64"),
                "d_inside_in": 12 * g.plate_x_in.to_numpy(dtype="float64"), "d_whiff_pts": 100 * g.whiff_pct.to_numpy(dtype="float64"),
                "d_csw_pts": 100 * g.csw_pct.to_numpy(dtype="float64"), "d_xwcon": g.xwoba_contact.to_numpy(dtype="float64"),
                "d_release_sd_in": 12 * g.release_sd.to_numpy(dtype="float64"), "d_velo": g.release_speed.to_numpy(dtype="float64")}
    for sname, V in {"Identity space (15)": P, "Old atlas space (27)": P_old, "t-SNE 3D map": views["t-SNE 3D"],
                     "Movement plot": views["Movement (ride x HB)"]}.items():
        nb = knn_idx_free(V, 30, pid)
        row = {"type": pt, "row": np.arange(len(g)), "pitcher": pid, "n": g.n.to_numpy(), "space": sname,
               "height_ft": g.plate_z.to_numpy(dtype="float64")}
        for c, v in gap_cols.items():
            row[c] = v - np.array([np.nanmean(v[r[r >= 0]]) for r in nb])
        L_rows.append(pd.DataFrame(row))
    res["L_rows"] = pd.concat(L_rows, ignore_index=True)

    # ---------- B: held-out second halves
    gh = h_all[h_all.pitch_type == pt]
    h1, h2 = gh[gh.half == 1].reset_index(drop=True), gh[gh.half == 2].reset_index(drop=True)
    F1, F2 = h1[MAP_FEATURES].astype("float64"), h2[MAP_FEATURES].astype("float64")
    key1 = {(p, s): k for k, (p, s) in enumerate(zip(h1.pitcher, h1.season))}
    own = np.array([key1.get((p, s), -1) for p, s in zip(h2.pitcher, h2.season)])
    q = np.flatnonzero(own >= 0)                       # second halves with a first half
    same_pitcher = h1.pitcher.to_numpy()[None, :] == h2.pitcher.to_numpy()[:, None]
    # refresh-style: fit on every first half, place the second halves
    sc = scaler_fit(F1, blocks=MAP_BLOCKS)
    pca = PCA(n_components=min(N_PCS, len(MAP_FEATURES)), random_state=SEED).fit(scaler_apply(sc, F1))
    P1, P2 = pca.transform(scaler_apply(sc, F1)), pca.transform(scaler_apply(sc, F2))
    mv = pd.DataFrame({"ivb": gh.pfx_z, "hb": gh.pfx_x})
    mu, sd = mv[gh.half == 1].mean(), mv[gh.half == 1].std(ddof=0)
    M1, M2 = ((mv[gh.half == 1] - mu) / sd).to_numpy(), ((mv[gh.half == 2] - mu) / sd).to_numpy()
    B_rows = []

    def add_B(scenario, view, placement, Vref, Vq, qrows, targets, fold=-1):
        rk = ranks_of_target(Vq, Vref, targets)
        B_rows.append(pd.DataFrame({"type": pt, "scenario": scenario, "view": view, "placement": placement, "fold": fold,
                                    "pitcher": h2.pitcher.to_numpy()[qrows], "n": h2.n.to_numpy()[qrows],
                                    "n_ref": len(Vref), "rank": rk}))

    add_B("refresh", "Full space", "as is", P1, P2[q], q, own[q])
    add_B("refresh", "Movement (ride x HB)", "as is", M1, M2[q], q, own[q])
    add_B("refresh", "PCA 2", "as is", P1[:, :2], P2[q, :2], q, own[q])
    for name in CORE:
        d, fn, kw = METHODS()[name]
        U1, obj = timed(name, fn, P1, d, SEED, **kw)
        add_B("refresh", name, "transform", U1, place(obj, P2[q], P1), q, own[q])
        add_B("refresh", name, "IDW 15, other pitchers", U1, idw_place(P1, P2[q], U1, exclude=same_pitcher[q])[0], q, own[q])
        add_B("refresh", name, "IDW 15, all rows", U1, idw_place(P1, P2[q], U1)[0], q, own[q])
    # stranger-style: pitcher folds, both halves placed on a map fit without that pitcher
    folds = 5
    pit_all = np.unique(np.concatenate([h1.pitcher, h2.pitcher]))
    fold_of = {p: k % folds for k, p in enumerate(np.random.default_rng(SEED).permutation(pit_all))}
    f1, f2 = np.array([fold_of[p] for p in h1.pitcher]), np.array([fold_of[p] for p in h2.pitcher])
    for f in range(folds):
        tr = f1 != f
        scf = scaler_fit(F1[tr], blocks=MAP_BLOCKS)
        pcf = PCA(n_components=min(N_PCS, len(MAP_FEATURES)), random_state=SEED).fit(scaler_apply(scf, F1[tr]))
        Ptr, Pte1, Pte2 = pcf.transform(scaler_apply(scf, F1[tr])), pcf.transform(scaler_apply(scf, F1[~tr])), pcf.transform(scaler_apply(scf, F2))
        te1 = np.flatnonzero(~tr)
        pos1 = {r: k for k, r in enumerate(te1)}
        qf = np.array([r for r in q if f2[r] == f and own[r] in pos1])
        if len(qf) == 0:
            continue
        tgt = np.array([pos1[own[r]] for r in qf])        # target index within the placed first halves
        muf, sdf = mv[gh.half == 1][tr].mean(), mv[gh.half == 1][tr].std(ddof=0)
        Mtr, Mte1, Mte2 = ((mv[gh.half == 1][tr] - muf) / sdf).to_numpy(), ((mv[gh.half == 1][~tr] - muf) / sdf).to_numpy(), ((mv[gh.half == 2] - muf) / sdf).to_numpy()

        def add_S(view, placement, Utr, Ute1, Ute2):
            # reference = fitted first halves + the held-out pitchers' placed first halves
            ref = np.vstack([Utr, Ute1])
            add_B("stranger", view, placement, ref, Ute2[qf], qf, len(Utr) + tgt, fold=f)

        add_S("Full space", "as is", Ptr, Pte1, Pte2)
        add_S("Movement (ride x HB)", "as is", Mtr, Mte1, Mte2)
        add_S("PCA 2", "as is", Ptr[:, :2], Pte1[:, :2], Pte2[:, :2])
        for name in ["UMAP 2D (15, 0.3)", "UMAP 3D (15, 0.3)", "t-SNE 2D", "t-SNE 3D", "PaCMAP 3D (15)"]:
            d, fn, kw = METHODS()[name]
            Utr, obj = timed(name, fn, Ptr, d, SEED, **kw)
            add_S(name, "transform", Utr, place(obj, Pte1, Ptr), place(obj, Pte2, Ptr))
            add_S(name, "IDW 15", Utr, idw_place(Ptr, Pte1, Utr)[0], idw_place(Ptr, Pte2, Utr)[0])
    res["B_rows"] = pd.concat(B_rows, ignore_index=True)
    log(f"B {pt}: {len(q)} refresh pairs")

    # ---------- D: stability (seeds; refresh with cold / warm / aligned starts), 2D and 3D
    D_rows = []
    # refresh data: first-half profiles vs full-season profiles of the same type
    th = h1                                           # first-half rows (fit 1)
    Fh = th[MAP_FEATURES].astype("float64")
    Xh, Ph = within_space(Fh)
    keyh = {(p, s): k for k, (p, s) in enumerate(zip(th.pitcher, th.season))}
    common_s = np.array([keyh.get((p, s), -1) for p, s in zip(g.pitcher, g.season)])
    cs = np.flatnonzero(common_s >= 0)                # season rows that also have a first-half row
    ch = common_s[cs]                                 # their first-half rows
    lab_s = lab
    for mname, (fn, kw) in STAB.items():
        for d in (2, 3):
            # seeds
            embs = [timed(f"{mname} {d}D", fn, P, d, s, **kw)[0] for s in SEEDS]
            nb10 = [knn_idx(E, 10) for E in embs]
            nb50 = [knn_idx(E, 50) for E in embs]
            pr = [(a, b) for ia, a in enumerate(range(len(SEEDS))) for b in range(ia + 1, len(SEEDS))]
            D_rows.append({"type": pt, "n": len(g), "method": mname, "dims": d, "test": "seeds", "runs": len(SEEDS),
                           "overlap10": float(np.mean([row_recall(nb10[a], nb10[b]).mean() for a, b in pr])),
                           "overlap50": float(np.mean([row_recall(nb50[a], nb50[b]).mean() for a, b in pr])),
                           "procrustes": float(np.mean([procrustes(embs[a], embs[b])[2] for a, b in pr])),
                           "island_share_mean": float(np.mean([np.mean(lab_s[nb] == lab_s[:, None]) for nb in nb10])),
                           "recall10_free_min": float(min(row_recall(truth_free, knn_idx_free(E, K, pid)).mean() for E in embs)),
                           "recall10_free_max": float(max(row_recall(truth_free, knn_idx_free(E, K, pid)).mean() for E in embs))})
            # refresh: first halves -> full season
            E1, _ = timed(f"{mname} {d}D", fn, Ph, d, SEED, **kw)
            init = np.zeros((len(g), d))
            init[cs] = E1[ch]
            new = np.setdiff1d(np.arange(len(g)), cs)
            if len(new):
                init[new] = idw_place(P[cs], P[new], E1[ch])[0]
            starts = {"cold": None, "warm": init}
            for sname, st in starts.items():
                if st is None:
                    E2, _ = timed(f"{mname} {d}D", fn, P, d, SEED, **kw)
                elif fn is fit_umap:
                    E2, _ = timed(f"{mname} {d}D", fn, P, d, SEED, init=st, **kw)
                elif fn is fit_tsne:
                    E2, _ = timed(f"{mname} {d}D", fn, P, d, SEED, init=st, **kw)
                else:
                    E2, _ = timed(f"{mname} {d}D", fn, P, d, SEED, init=st, **kw)
                a, b = E1[ch], E2[cs]
                D_rows.append({"type": pt, "n": len(cs), "method": mname, "dims": d, "test": f"refresh {sname}", "runs": 1,
                               "overlap10": float(row_recall(knn_idx(a, 10), knn_idx(b, 10)).mean()),
                               "overlap50": float(row_recall(knn_idx(a, 50), knn_idx(b, 50)).mean()),
                               "procrustes": float(procrustes(a, b)[2]),
                               "island_share_mean": float(np.mean(lab_s[cs][knn_idx(b, 10)] == lab_s[cs][:, None])),
                               "recall10_free_min": float(row_recall(truth_free, knn_idx_free(E2, K, pid)).mean()),
                               "recall10_free_max": np.nan})
            if fn is fit_umap:
                import umap
                t0 = time.time()
                al = umap.AlignedUMAP(n_neighbors=kw.get("nn", 15), min_dist=kw.get("md", 0.3), n_components=d,
                                      random_state=SEED).fit([Ph, P], relations=[{int(hh): int(ss) for hh, ss in zip(ch, cs)}])
                TIMES.setdefault(f"AlignedUMAP {d}D", []).append(time.time() - t0)
                a, b = al.embeddings_[0][ch], al.embeddings_[1][cs]
                D_rows.append({"type": pt, "n": len(cs), "method": mname, "dims": d, "test": "refresh aligned", "runs": 1,
                               "overlap10": float(row_recall(knn_idx(a, 10), knn_idx(b, 10)).mean()),
                               "overlap50": float(row_recall(knn_idx(a, 50), knn_idx(b, 50)).mean()),
                               "procrustes": float(procrustes(a, b)[2]),
                               "island_share_mean": float(np.mean(lab_s[cs][knn_idx(b, 10)] == lab_s[cs][:, None])),
                               "recall10_free_min": float(row_recall(truth_free, knn_idx_free(al.embeddings_[1], K, pid)).mean()),
                               "recall10_free_max": np.nan})
    res["D_stability"] = pd.DataFrame(D_rows)
    log(f"D {pt}: done")

    # ---------- E: settings
    E_rows = []

    def score(E, extra):
        E_rows.append({"type": pt, "n": len(g), **extra,
                       "recall10_free": float(row_recall(truth_free, knn_idx_free(E, K, pid)).mean()),
                       "trust10": float(trustworthiness(P, E, n_neighbors=K)),
                       "continuity10": float(trustworthiness(E, P, n_neighbors=K)),
                       "dist_spearman": dist_spearman(P, E, pairs),
                       "island_share": float(np.mean(lab[knn_idx(E, 10)] == lab[:, None]))})

    nn_grid = [5, 15, 50] if FAST else [5, 10, 15, 30, 50]
    md_grid = [0.1, 0.3] if FAST else [0.0, 0.1, 0.3, 0.5]
    for nn in nn_grid:
        for md in md_grid:
            score(timed("grid", fit_umap, P, 2, SEED, nn=nn, md=md)[0], {"method": "UMAP", "dims": 2, "a": nn, "b": md})
    for nn in ([5, 15] if FAST else [5, 10, 15, 30]):
        for md in ([0.1, 0.3] if FAST else [0.0, 0.1, 0.3]):
            score(timed("grid", fit_umap, P, 3, SEED, nn=nn, md=md)[0], {"method": "UMAP", "dims": 3, "a": nn, "b": md})
    for nn in ([10, 30] if FAST else [10, 15, 30]):
        for mn in ([0.5] if FAST else [0.5, 1.0]):
            score(timed("grid", fit_pacmap, P, 2, SEED, nn=nn, mn=mn)[0], {"method": "PaCMAP", "dims": 2, "a": nn, "b": mn})
    for perp in ([10, 30] if FAST else [10, 30, 50]):
        for ex in ([1, 4] if FAST else [1, 2, 4]):
            score(timed("grid", fit_tsne, P, 2, SEED, perp=perp, exag=(None if ex == 1 else ex))[0],
                  {"method": "t-SNE", "dims": 2, "a": perp, "b": ex})
    for d in ([2, 3] if FAST else [2, 3, 4, 5]):
        for nn, md in [(15, 0.3), (10, 0.1)]:
            score(timed("grid", fit_umap, P, d, SEED, nn=nn, md=md)[0], {"method": f"UMAP ({nn}, {md}) dims", "dims": d, "a": nn, "b": md})
    res["E_settings"] = pd.DataFrame(E_rows)
    log(f"E {pt}: done")

    # ---------- feature redundancy (within-type correlations)
    Z = zscore(g[MAP_FEATURES].astype("float64").fillna(g[MAP_FEATURES].median()))
    C = Z.corr().abs()
    red = [{"type": pt, "a": a, "b": b, "abs_r": float(C.loc[a, b])}
           for i, a in enumerate(MAP_FEATURES) for b in MAP_FEATURES[i + 1:] if C.loc[a, b] >= 0.8]
    res["F_redundancy"] = pd.DataFrame(red)
    res["times"] = pd.DataFrame([{"type": pt, "fit": k, "n": len(v), "median_s": float(np.median(v)), "max_s": float(np.max(v))}
                                 for k, v in TIMES.items()])
    return pt, res


# ------------------------------------------------------------------ addenda (second review round)
def run_addenda(pt, t_all, h_all):
    """(i) Refresh with symmetric warm starts: UMAP init=prev with n_epochs 100 (and lr 0.5), AlignedUMAP with
    regularisation 0.01 and 0.1, t-SNE init=prev with 250 iterations; recall after refit for all.
    (ii) Stranger test for UMAP (10, 0.1) in 2D and 3D and PaCMAP 2D. (iii) 3D grids for PaCMAP and t-SNE."""
    import umap
    global TIMES
    TIMES = {}
    g = t_all[t_all.pitch_type == pt].reset_index(drop=True)
    pid = g.pitcher.to_numpy()
    X, P = within_space(g[MAP_FEATURES].astype("float64"))
    truth_free = knn_idx_free(P, K, pid)
    lab = leiden_labels(P)
    rng = np.random.default_rng(SEED)
    pairs = pair_set(len(g), rng)
    res = {}
    # ---- (i) refresh
    gh = h_all[h_all.pitch_type == pt]
    h1 = gh[gh.half == 1].reset_index(drop=True)
    Xh, Ph = within_space(h1[MAP_FEATURES].astype("float64"))
    keyh = {(p, s): k for k, (p, s) in enumerate(zip(h1.pitcher, h1.season))}
    common_s = np.array([keyh.get((p, s), -1) for p, s in zip(g.pitcher, g.season)])
    cs = np.flatnonzero(common_s >= 0); ch = common_s[cs]
    new = np.setdiff1d(np.arange(len(g)), cs)
    rows = []

    def score_refresh(method, d, start, E1, E2):
        a, b = E1[ch], E2[cs]
        rows.append({"type": pt, "n": len(cs), "method": method, "dims": d, "test": f"refresh {start}",
                     "overlap10": float(row_recall(knn_idx(a, 10), knn_idx(b, 10)).mean()),
                     "overlap50": float(row_recall(knn_idx(a, 50), knn_idx(b, 50)).mean()),
                     "procrustes": float(procrustes(a, b)[2]),
                     "island_share_mean": float(np.mean(lab[cs][knn_idx(b, 10)] == lab[cs][:, None])),
                     "recall10_free": float(row_recall(truth_free, knn_idx_free(E2, K, pid)).mean())})

    for mname, nn, md in [("UMAP (15, 0.3)", 15, 0.3), ("UMAP (10, 0.1)", 10, 0.1)]:
        for d in (2, 3):
            E1 = umap.UMAP(n_components=d, n_neighbors=nn, min_dist=md, random_state=SEED).fit(Ph).embedding_
            init = np.zeros((len(g), d)); init[cs] = E1[ch]
            if len(new):
                init[new] = idw_place(P[cs], P[new], E1[ch])[0]
            E2 = umap.UMAP(n_components=d, n_neighbors=nn, min_dist=md, random_state=SEED).fit(P).embedding_
            score_refresh(mname, d, "cold", E1, E2)
            for label, kw in [("warm 500 epochs", {}), ("warm 100 epochs", {"n_epochs": 100}), ("warm 100 epochs, lr 0.5", {"n_epochs": 100, "learning_rate": 0.5}),
                              ("warm 30 epochs", {"n_epochs": 30})]:
                E2 = umap.UMAP(n_components=d, n_neighbors=nn, min_dist=md, init=init, random_state=SEED, **kw).fit(P).embedding_
                score_refresh(mname, d, label, E1, E2)
            for reg in (0.01, 0.1, 1.0):
                al = umap.AlignedUMAP(n_neighbors=nn, min_dist=md, n_components=d, alignment_regularisation=reg,
                                      random_state=SEED).fit([Ph, P], relations=[{int(a): int(b) for a, b in zip(ch, cs)}])
                score_refresh(mname, d, f"aligned reg {reg}", al.embeddings_[0], al.embeddings_[1])
    for d in (2, 3):
        E1 = fit_tsne(Ph, d, SEED)[0]
        init = np.zeros((len(g), d)); init[cs] = E1[ch]
        if len(new):
            init[new] = idw_place(P[cs], P[new], E1[ch])[0]
        score_refresh("t-SNE (perp 30)", d, "cold", E1, fit_tsne(P, d, SEED)[0])
        import openTSNE
        for label, n_iter in [("warm 500 iters", 500), ("warm 250 iters", 250), ("warm 100 iters", 100)]:
            E2 = np.asarray(openTSNE.TSNE(n_components=d, perplexity=30, random_state=SEED, n_jobs=1, negative_gradient_method="bh",
                                          early_exaggeration_iter=0, n_iter=n_iter).fit(P, initialization=init), dtype="float64")
            score_refresh("t-SNE (perp 30)", d, label, E1, E2)
    res["D_refresh2"] = pd.DataFrame(rows)
    log(f"addenda D {pt}: done")
    # ---- (ii) stranger test for the built setting and PaCMAP 2D
    h2 = gh[gh.half == 2].reset_index(drop=True)
    F1, F2 = h1[MAP_FEATURES].astype("float64"), h2[MAP_FEATURES].astype("float64")
    key1 = {(p, s): k for k, (p, s) in enumerate(zip(h1.pitcher, h1.season))}
    own = np.array([key1.get((p, s), -1) for p, s in zip(h2.pitcher, h2.season)])
    q = np.flatnonzero(own >= 0)
    folds = 5
    pit_all = np.unique(np.concatenate([h1.pitcher, h2.pitcher]))
    fold_of = {p: k % folds for k, p in enumerate(np.random.default_rng(SEED).permutation(pit_all))}
    f1, f2 = np.array([fold_of[p] for p in h1.pitcher]), np.array([fold_of[p] for p in h2.pitcher])
    B_rows = []
    for f in range(folds):
        tr = f1 != f
        scf = scaler_fit(F1[tr], blocks=MAP_BLOCKS)
        pcf = PCA(n_components=min(N_PCS, len(MAP_FEATURES)), random_state=SEED).fit(scaler_apply(scf, F1[tr]))
        Ptr, Pte1, Pte2 = pcf.transform(scaler_apply(scf, F1[tr])), pcf.transform(scaler_apply(scf, F1[~tr])), pcf.transform(scaler_apply(scf, F2))
        te1 = np.flatnonzero(~tr); pos1 = {r: k for k, r in enumerate(te1)}
        qf = np.array([r for r in q if f2[r] == f and own[r] in pos1])
        if len(qf) == 0:
            continue
        tgt = np.array([pos1[own[r]] for r in qf])
        for name in ["UMAP 2D (10, 0.1)", "UMAP 3D (10, 0.1)", "PaCMAP 2D (15)"]:
            d, fn, kw = METHODS()[name]
            Utr, obj = fn(Ptr, d, SEED, **kw)
            for placement, (Ute1, Ute2) in {"transform": (place(obj, Pte1, Ptr), place(obj, Pte2, Ptr)),
                                            "IDW 15": (idw_place(Ptr, Pte1, Utr)[0], idw_place(Ptr, Pte2, Utr)[0])}.items():
                ref = np.vstack([Utr, Ute1])
                rk = ranks_of_target(Ute2[qf], ref, len(Utr) + tgt)
                B_rows.append(pd.DataFrame({"type": pt, "scenario": "stranger", "view": name, "placement": placement, "fold": f,
                                            "pitcher": h2.pitcher.to_numpy()[qf], "n": h2.n.to_numpy()[qf], "n_ref": len(ref), "rank": rk}))
    res["B_rows2"] = pd.concat(B_rows, ignore_index=True)
    log(f"addenda B {pt}: done")
    # ---- (iii) 3D grids
    E_rows = []

    def score(E, extra):
        E_rows.append({"type": pt, "n": len(g), **extra, "recall10_free": float(row_recall(truth_free, knn_idx_free(E, K, pid)).mean()),
                       "trust10": float(trustworthiness(P, E, n_neighbors=K)), "continuity10": float(trustworthiness(E, P, n_neighbors=K)),
                       "dist_spearman": dist_spearman(P, E, pairs), "island_share": float(np.mean(lab[knn_idx(E, 10)] == lab[:, None]))})
    for nn in [10, 15, 30]:
        score(fit_pacmap(P, 3, SEED, nn=nn)[0], {"method": "PaCMAP", "dims": 3, "a": nn, "b": 0.5})
    for perp in [10, 30, 50]:
        for ex in [1, 2]:
            score(fit_tsne(P, 3, SEED, perp=perp, exag=(None if ex == 1 else ex))[0], {"method": "t-SNE", "dims": 3, "a": perp, "b": ex})
    res["E_settings3d"] = pd.DataFrame(E_rows)
    log(f"addenda E {pt}: done")
    return pt, res


# ------------------------------------------------------------------ G: the All-pitches map
def run_global(t, h):
    global TIMES
    TIMES = {}
    rng = np.random.default_rng(SEED)
    pid = t.pitcher.to_numpy()
    F = t[MAP_FEATURES].astype("float64")
    F = F.fillna(F.groupby(t.pitch_type).transform("median"))
    X = block_scale(zscore(F), MAP_BLOCKS)
    P = PCA(n_components=min(N_PCS, X.shape[1]), random_state=SEED).fit_transform(X)
    lab = t.pitch_type.to_numpy()
    # a pitcher's pitch types share a pitcher id: self-free here means other rows of the same pitcher AND type
    key = (t.pitcher.astype(str) + "|" + t.pitch_type).to_numpy()
    truth_free = knn_idx_free(P, K, key)
    pairs = pair_set(len(t), rng)
    methods = {k: v for k, v in METHODS().items() if k in ("t-SNE 2D", "t-SNE 2D, exag 4", "t-SNE 3D", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)",
                                                             "UMAP 3D (15, 0.3)", "UMAP 3D (10, 0.1)", "PaCMAP 2D (15)", "PaCMAP 3D (15)")}
    views = {"Movement (ride x HB)": zscore(pd.DataFrame({"ivb": t.pfx_z, "hb": t.pfx_x})).to_numpy(), "PCA 2": P[:, :2]}
    for name, (d, fn, kw) in methods.items():
        views[name] = timed(name, fn, P, d, SEED, **kw)[0]
        log(f"G fit {name}")
    views["Random 2D"] = rng.normal(size=(len(t), 2))
    views["Full space"] = P
    rows, long = [], []
    for name, V in views.items():
        nb = knn_idx(V, K)
        rf = row_recall(truth_free, knn_idx_free(V, K, key))
        rows.append({"view": name, "n": len(t), "dims": V.shape[1], "recall10_free": float(rf.mean()),
                     "type_purity": float(np.mean(lab[nb] == lab[:, None])), "type_knn_acc": knn_label_accuracy(V, pd.factorize(lab)[0]),
                     "trust10": float(trustworthiness(P, V, n_neighbors=K)) if name != "Full space" else 1.0,
                     "continuity10": float(trustworthiness(V, P, n_neighbors=K)) if name != "Full space" else 1.0,
                     "dist_spearman": dist_spearman(P, V, pairs),
                     "fit_seconds": float(np.median(TIMES[name])) if name in TIMES else 0.0})
        long.append(pd.DataFrame({"row": np.arange(len(t)), "pitcher": pid, "view": name, "recall10_free": rf}))
    out = {"G_views": pd.DataFrame(rows), "G_rows": pd.concat(long, ignore_index=True)}
    # coordinates of the main maps, for the figures
    coords = t[["pitcher", "season", "pitch_type", "player_name", "p_throws", "n", "release_speed"]].copy()
    for name in ["PCA 2", "t-SNE 2D", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "PaCMAP 2D (15)"]:
        coords[f"{name}|x"], coords[f"{name}|y"] = views[name][:, 0], views[name][:, 1]
    out["G_coords"] = coords
    # refresh (cold / warm) and held-out placement for the main methods on the all-pitches map
    h1, h2 = h[h.half == 1].reset_index(drop=True), h[h.half == 2].reset_index(drop=True)
    F1 = h1[MAP_FEATURES].astype("float64")
    sc = scaler_fit(F1, h1.pitch_type, MAP_BLOCKS)
    pca = PCA(n_components=min(N_PCS, len(MAP_FEATURES)), random_state=SEED).fit(scaler_apply(sc, F1, h1.pitch_type))
    P1 = pca.transform(scaler_apply(sc, F1, h1.pitch_type))
    P2 = pca.transform(scaler_apply(sc, h2[MAP_FEATURES].astype("float64"), h2.pitch_type))
    key1 = {(p, s, y): k for k, (p, s, y) in enumerate(zip(h1.pitcher, h1.season, h1.pitch_type))}
    own = np.array([key1.get((p, s, y), -1) for p, s, y in zip(h2.pitcher, h2.season, h2.pitch_type)])
    q = np.flatnonzero(own >= 0)
    same = h1.pitcher.to_numpy()[None, :] == h2.pitcher.to_numpy()[q][:, None]
    B = []
    mv1 = pd.DataFrame({"ivb": h1.pfx_z, "hb": h1.pfx_x}); mu, sd = mv1.mean(), mv1.std(ddof=0)
    M1, M2 = ((mv1 - mu) / sd).to_numpy(), ((pd.DataFrame({"ivb": h2.pfx_z, "hb": h2.pfx_x}) - mu) / sd).to_numpy()
    for name, (Vref, Vq) in {"Full space": (P1, P2[q]), "Movement (ride x HB)": (M1, M2[q]), "PCA 2": (P1[:, :2], P2[q, :2])}.items():
        B.append(pd.DataFrame({"scenario": "refresh", "view": name, "placement": "as is", "pitcher": h2.pitcher.to_numpy()[q],
                               "n_ref": len(Vref), "rank": ranks_of_target(Vq, Vref, own[q])}))
    keyh = {(p, s, y): k for k, (p, s, y) in enumerate(zip(h1.pitcher, h1.season, h1.pitch_type))}
    common_s = np.array([keyh.get((p, s, y), -1) for p, s, y in zip(t.pitcher, t.season, t.pitch_type)])
    cs = np.flatnonzero(common_s >= 0); ch = common_s[cs]
    D = []
    for name in ["UMAP 2D (15, 0.3)", "UMAP 3D (15, 0.3)", "UMAP 2D (10, 0.1)", "t-SNE 2D", "t-SNE 3D"]:
        d, fn, kw = METHODS()[name]
        U1, obj = timed(name, fn, P1, d, SEED, **kw)
        B.append(pd.DataFrame({"scenario": "refresh", "view": name, "placement": "transform", "pitcher": h2.pitcher.to_numpy()[q],
                               "n_ref": len(U1), "rank": ranks_of_target(place(obj, P2[q], P1), U1, own[q])}))
        B.append(pd.DataFrame({"scenario": "refresh", "view": name, "placement": "IDW 15, other pitchers", "pitcher": h2.pitcher.to_numpy()[q],
                               "n_ref": len(U1), "rank": ranks_of_target(idw_place(P1, P2[q], U1, exclude=same)[0], U1, own[q])}))
        init = np.zeros((len(t), d)); init[cs] = U1[ch]
        new = np.setdiff1d(np.arange(len(t)), cs)
        if len(new):
            init[new] = idw_place(P[cs], P[new], U1[ch])[0]
        for sname, st in {"cold": None, "warm": init}.items():
            E2 = timed(name, fn, P, d, SEED, **({} if st is None else {"init": st}), **kw)[0]
            a, b = U1[ch], E2[cs]
            D.append({"method": name, "dims": d, "test": f"refresh {sname}", "n": len(cs),
                      "overlap10": float(row_recall(knn_idx(a, 10), knn_idx(b, 10)).mean()),
                      "overlap50": float(row_recall(knn_idx(a, 50), knn_idx(b, 50)).mean()),
                      "procrustes": float(procrustes(a, b)[2]),
                      "type_purity": float(np.mean(lab[cs][knn_idx(b, 10)] == lab[cs][:, None]))})
        log(f"G {name}: held-out + refresh done")
    out["G_B_rows"] = pd.concat(B, ignore_index=True)
    out["G_D"] = pd.DataFrame(D)
    out["G_times"] = pd.DataFrame([{"fit": k, "n": len(v), "median_s": float(np.median(v))} for k, v in TIMES.items()])
    return out


# ------------------------------------------------------------------ main
def load():
    t = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False)).reset_index(drop=True)
    h = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=True)).reset_index(drop=True)
    types = ["FF", "FS"] if FAST else sorted(t.pitch_type.unique())
    return t, h, types


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    t, h, types = load()
    log(f"{len(t):,} rows, {len(h):,} half rows, types {types}, FAST={FAST}, workers {WORKERS}")
    if "--addenda" in sys.argv:                     # second-review additions, per type in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=min(WORKERS, len(types))) as ex:
            for pt, res in ex.map(run_addenda, types, [t] * len(types), [h] * len(types)):
                results[pt] = res
                log(f"== {pt} addenda finished")
        for key in results[types[0]]:
            df = pd.concat([results[pt][key] for pt in types], ignore_index=True)
            (df.to_parquet if key.endswith("rows2") else df.to_csv)(OUT / f"{key}.{'parquet' if key.endswith('rows2') else 'csv'}", index=False)
        log(f"addenda done in {(time.time() - t0) / 60:.1f} min")
        return
    if "--global-only" in sys.argv:                 # rerun just the All-pitches section
        G = run_global(t, h)
        for key, df in G.items():
            (df.to_parquet if key in ("G_rows", "G_B_rows", "G_coords") else df.to_csv)(OUT / f"{key}.{'parquet' if key in ('G_rows', 'G_B_rows', 'G_coords') else 'csv'}", index=False)
        write_meta(t, h, types, t0)
        return
    results = {}
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(types))) as ex:
        for pt, res in ex.map(run_type, types, [t] * len(types), [h] * len(types)):
            results[pt] = res
            log(f"== {pt} finished")
    for key in results[types[0]]:
        df = pd.concat([results[pt][key] for pt in types], ignore_index=True)
        if key in ("A_rows", "B_rows", "C_rows", "A_coranking", "A_coords", "L_rows"):
            df.to_parquet(OUT / f"{key}.parquet", index=False)
        else:
            df.to_csv(OUT / f"{key}.csv", index=False)
    log("per-type results written; starting the All-pitches map")
    G = run_global(t, h)
    for key, df in G.items():
        if key in ("G_rows", "G_B_rows", "G_coords"):
            df.to_parquet(OUT / f"{key}.parquet", index=False)
        else:
            df.to_csv(OUT / f"{key}.csv", index=False)
    write_meta(t, h, types, t0)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


def write_meta(t, h, types, t0):
    import importlib.metadata as im
    meta = {"rows": int(len(t)), "half_rows": int(len(h)), "types": types, "fast": FAST, "k": K, "seeds": SEEDS, "n_boot": N_BOOT,
            "total_minutes": (time.time() - t0) / 60, "features": MAP_FEATURES,
            "versions": {p: im.version(p) for p in ["umap-learn", "openTSNE", "pacmap", "phate", "trimap", "scikit-learn", "numpy"]}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
