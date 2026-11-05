"""Tables (with pitcher-clustered bootstrap intervals) and figures for the UMAP benchmark.

Reads analysis/umap_benchmark/*.csv|parquet written by umap_benchmark.py; writes tables.md and
fig1..fig6 + figS1..S2 (PDF + PNG) into the same folder. Every interval is a 95% percentile
bootstrap over PITCHERS (all of a pitcher's rows move together), 1,000 draws, the same draws for every
method so paired differences are honest. "Weighted" pools rows across pitch types (so each type counts
by its row count); "unweighted" averages the per-type means.

Usage:  python scripts/umap_benchmark_figures.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlbpitch import paths  # noqa: E402

OUT = paths.ROOT / "analysis" / "umap_benchmark"
SEED = 0
META = json.loads((OUT / "meta.json").read_text())
N_BOOT = int(META.get("n_boot", 1000))
TYPE_NAMES = {"FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper", "CU": "Curveball", "CH": "Changeup", "FS": "Splitter"}
MAIN = ["Movement (ride x HB)", "Savant 5 (PCA 2)", "PCA 2", "PCA 3", "Isomap 2D", "TriMap 2D", "PHATE 2D", "PaCMAP 2D (10)", "PaCMAP 2D (15)", "LocalMAP 2D (15)",
        "UMAP 2D (15, 0.3)", "UMAP 2D (15, 0.3), PCA init", "UMAP 2D (10, 0.1)", "UMAP 2D (5, 0.0)", "t-SNE 2D", "t-SNE 2D, exag 4", "t-SNE 2D, uniform k=15",
        "PHATE 3D", "PaCMAP 3D (15)", "densMAP 3D (15, 0.3)", "UMAP 3D (15, 0.3)", "UMAP 3D (10, 0.1)", "t-SNE 3D"]
SHORT = {"Movement (ride x HB)": "Movement", "Savant 5 (PCA 2)": "Savant-5 PCA", "Savant 5 space (5-D)": "Savant-5 (5-D)", "Full space": "Full space",
         "UMAP 2D (15, 0.3)": "UMAP 2D", "UMAP 2D (10, 0.1)": "UMAP 2D 10/0.1", "UMAP 2D (5, 0.0)": "UMAP 2D 5/0", "UMAP 3D (15, 0.3)": "UMAP 3D",
         "UMAP 3D (10, 0.1)": "UMAP 3D 10/0.1", "PaCMAP 2D (15)": "PaCMAP 2D", "PaCMAP 3D (15)": "PaCMAP 3D", "t-SNE 2D, exag 4": "t-SNE 2D ex4",
         "t-SNE 2D, uniform k=15": "t-SNE 2D unif", "UMAP 2D (15, 0.3), PCA init": "UMAP 2D PCA-init", "PaCMAP 2D (10)": "PaCMAP 2D (10)", "densMAP 3D (15, 0.3)": "densMAP 3D", "LocalMAP 2D (15)": "LocalMAP 2D", "Random 2D": "Random"}


def short(v):
    return SHORT.get(v, v)


# ------------------------------------------------------------------ markdown
def md(df, index=True, digits=2):
    df = df.reset_index() if index else df
    cols = [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else (f"{v:.{digits}f}" if not float(v).is_integer() or digits == 0 else f"{v:.0f}")
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def ci_str(m, lo, hi, pct=True, d=0):
    if pct:
        return f"{100 * m:.{d}f} [{100 * lo:.{d}f}, {100 * hi:.{d}f}]"
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


# ------------------------------------------------------------------ bootstrap over pitchers
class Boot:
    """Shared pitcher resampling. mult: (reps, n_pitchers) multiplicities."""

    def __init__(self, pitchers, reps=N_BOOT, seed=SEED):
        self.p = np.unique(pitchers)
        self.idx = {p: i for i, p in enumerate(self.p)}
        rng = np.random.default_rng(seed)
        self.mult = rng.multinomial(len(self.p), np.full(len(self.p), 1 / len(self.p)), size=reps).astype("float64")

    def mean(self, pitchers, values):
        """Bootstrap distribution of the mean of `values` (rows clustered by pitcher)."""
        i = np.array([self.idx[p] for p in pitchers])
        v = np.asarray(values, dtype="float64")
        ok = ~np.isnan(v)
        s = np.bincount(i[ok], weights=v[ok], minlength=len(self.p))
        c = np.bincount(i[ok], minlength=len(self.p)).astype("float64")
        return (self.mult @ s) / (self.mult @ c)

    def corr(self, pitchers, a, b):
        """Bootstrap distribution of the Pearson correlation of a and b (use ranks for Spearman)."""
        i = np.array([self.idx[p] for p in pitchers])
        a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
        ok = ~(np.isnan(a) | np.isnan(b))
        i, a, b = i[ok], a[ok], b[ok]
        W = self.mult[:, i]                                   # (reps, n_rows) multiplicities per row
        n = W.sum(axis=1)
        ma, mb = (W @ a) / n, (W @ b) / n
        cov = (W @ (a * b)) / n - ma * mb
        va, vb = (W @ (a * a)) / n - ma ** 2, (W @ (b * b)) / n - mb ** 2
        return cov / np.sqrt(np.clip(va * vb, 1e-12, None))


def summarize(dist):
    return float(np.mean(dist)), float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


# ------------------------------------------------------------------ load
def load():
    d = {}
    for f in OUT.glob("*.csv"):
        d[f.stem] = pd.read_csv(f)
    for f in OUT.glob("*.parquet"):
        d[f.stem] = pd.read_parquet(f)
    return d


# ------------------------------------------------------------------ tables

def coranking_summary(D, view):
    """From the stored Q_NX(K) curves: K_max (type median), Q_NX at K = 10 / 50 / 200 and the log-K-weighted
    R_NX area under the curve (Lee et al. 2013), weighted over types."""
    cor = D["A_coranking"]
    c = cor[cor.view == view]
    if c.empty:
        return {"K_max": np.nan, "Q_NX(10)": np.nan, "Q_NX(50)": np.nan, "Q_NX(200)": np.nan, "R_NX AUC": np.nan}
    n_by = D["A_views"].groupby("type").n.first()
    out, kmax, wsum = {"Q_NX(10)": 0.0, "Q_NX(50)": 0.0, "Q_NX(200)": 0.0, "R_NX AUC": 0.0}, [], 0.0
    for pt, g in c.groupby("type"):
        n = n_by[pt]
        K, q = g.K.to_numpy(), g.qnx.to_numpy()
        lcmc = q - K / (n - 1)
        kmax.append(int(K[np.argmax(lcmc)]))
        rnx = ((n - 1) * q - K) / (n - 1 - K)
        rnx = rnx[:-1]; Kr = K[:-1]
        auc = float(np.sum(rnx / Kr) / np.sum(1 / Kr))
        for kk in (10, 50, 200):
            out[f"Q_NX({kk})"] += n * float(q[K == kk][0]) if (K == kk).any() else 0.0
        out["R_NX AUC"] += n * auc
        wsum += n
    return {"K_max": float(np.median(kmax)), **{k: v / wsum for k, v in out.items()}}


def reliability():
    """Spearman between a pitcher's first-half and second-half outcome (same season, type), weighted by type: the
    ceiling any predictor of a season outcome can reach at these sample sizes."""
    from scipy.stats import spearmanr
    from mlbpitch import profiles
    h = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=True))
    mm = h[h.half == 1].merge(h[h.half == 2], on=["pitcher", "season", "pitch_type"], suffixes=("_1", "_2"))
    n = mm.groupby("pitch_type").size()
    out = {}
    for oc in ["whiff_pct", "csw_pct", "xwoba_contact", "rv100_raw", "rv100"]:
        per = mm.groupby("pitch_type").apply(lambda d: spearmanr(d[oc + "_1"], d[oc + "_2"]).statistic, include_groups=False)
        out[oc] = float((per * n).sum() / n.sum())
    return out


def pruned_truth(D):
    """recall@10 self-free of the stored 2D maps against a truth with the duplicate features removed."""
    from mlbpitch import profiles
    from umap_benchmark import MAP_BLOCKS, knn_idx_free, row_recall, within_space
    drop = {"haa_above_avg", "hra_above_avg", "ax", "az", "ddz_delta_az_velo", "scaled_extension"}
    blocks = {k: [c for c in v if c not in drop] for k, v in MAP_BLOCKS.items()}
    feats = [c for cs in blocks.values() for c in cs]
    full = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False)).reset_index(drop=True)
    coords = D["A_coords"]
    views = [v for v in ["Movement (ride x HB)", "PCA 2", "PaCMAP 2D (15)", "UMAP 2D (10, 0.1)", "UMAP 2D (15, 0.3)", "t-SNE 2D"] if f"{v}|x" in coords]
    tot, cnt = {v: 0.0 for v in views}, 0
    for pt, c in coords.groupby("pitch_type"):
        g = full[full.pitch_type == pt].reset_index(drop=True)
        assert (g.pitcher.to_numpy() == c.pitcher.to_numpy()).all()
        _, P = within_space(g[feats].astype("float64"), blocks)
        pid = g.pitcher.to_numpy()
        truth = knn_idx_free(P, 10, pid)
        for v in views:
            V = c[[f"{v}|x", f"{v}|y"]].to_numpy()
            tot[v] += row_recall(truth, knn_idx_free(V, 10, pid)).sum()
        cnt += len(g)
    return {v: 100 * tot[v] / cnt for v in views}

def tables(D, B):
    L = ["# Benchmark tables\n", f"Pitcher-clustered bootstrap, {N_BOOT} draws, 95% percentile intervals in brackets. "
         "Weighted = rows pooled across pitch types. Self-free = a pitcher's other seasons are excluded from both the true and the "
         "on-map neighbour sets.\n"]
    A, Ar = D["A_views"], D["A_rows"]
    views = [v for v in MAIN if v in A.view.unique()] + ["Random 2D"]
    # A1: headline per view
    rows = []
    ref2 = Ar[Ar.view == "t-SNE 2D"].set_index(["type", "row"]).recall10_free
    ref3 = Ar[Ar.view == "t-SNE 3D"].set_index(["type", "row"]).recall10_free
    boot_rows = {}
    for v in views + ["Full space", "Savant 5 space (5-D)"]:
        a = Ar[Ar.view == v]
        if a.empty:
            continue
        dist = B.mean(a.pitcher, a.recall10_free)
        boot_rows[v] = dist
        s = A[A.view == v]
        w = s.n / s.n.sum()
        d2 = B.mean(a.pitcher, a.recall10_free.to_numpy() - ref2.reindex(pd.MultiIndex.from_frame(a[["type", "row"]])).to_numpy())
        d3 = B.mean(a.pitcher, a.recall10_free.to_numpy() - ref3.reindex(pd.MultiIndex.from_frame(a[["type", "row"]])).to_numpy())
        rows.append({"view": v, "dims": int(s.dims.iloc[0]), "recall@10 self-free % [CI]": ci_str(*summarize(dist)),
                     "unweighted %": f"{100 * s.recall10_free.mean():.0f}",
                     "vs t-SNE 2D (pts)": ci_str(*summarize(d2)), "vs t-SNE 3D (pts)": ci_str(*summarize(d3)),
                     "all-rows %": f"{100 * (s.recall10_all * w).sum():.0f}", "recall@30 %": f"{100 * (s.recall30_all * w).sum():.0f}",
                     "trust": (s.trust10 * w).sum(), "continuity": (s.continuity10 * w).sum(), "dist corr": (s.dist_spearman * w).sum(),
                     **coranking_summary(D, v),
                     "island kNN acc": (s.island_knn_acc * w).sum(), "island ARI": (s.island_ari * w).sum(),
                     "fit s": s.fit_seconds.median()})
    A1 = pd.DataFrame(rows)
    L.append("## A1. Neighbour preservation by picture (all types, weighted)\n")
    L.append(md(A1, index=False) + "\n")
    chance = (10 / (A.groupby("type").n.first() - 1)).mean()
    L.append(f"Chance for recall@10: {100 * chance:.1f}%. Nearest true neighbour is the same pitcher's other season for "
             f"{100 * A.groupby('type').self_rate_nn1.first().mean():.0f}% of rows (type mean); self rows are "
             f"{100 * A.groupby('type').self_share_in10.first().mean():.0f}% of all-rows 10-NN sets, which is why the self-free score is the headline.\n")
    # A2 per type
    sub = ["Movement (ride x HB)", "PCA 2", "PaCMAP 2D (15)", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "t-SNE 2D", "PaCMAP 3D (15)", "UMAP 3D (15, 0.3)", "UMAP 3D (10, 0.1)", "t-SNE 3D"]
    sub = [v for v in sub if v in A.view.unique()]
    T = (A[A.view.isin(sub)].pivot(index="type", columns="view", values="recall10_free")[sub] * 100).round(0)
    T.columns = [short(c) for c in T.columns]
    T.insert(0, "n", A.groupby("type").n.first())
    L.append("## A2. recall@10 self-free (%) by pitch type\n")
    L.append(md(T, digits=0) + "\n")
    # A3 PCA dims
    pdims = D["A_pca_dims"]
    w = pdims.groupby("dims").apply(lambda d: (d.recall10_free * d.n).sum() / d.n.sum(), include_groups=False) * 100
    L.append("## A3. A linear picture: recall@10 self-free (%) by number of principal components\n")
    L.append(md(w.round(0).to_frame("recall@10 %").T, digits=0) + "\n")
    # A4 truth sensitivity (+ a pruned-feature truth computed here from the stored 2D coordinates)
    ts = D["A_truth_sensitivity"]
    T = ts.groupby(["truth", "view"]).apply(lambda d: (d.recall10_free * d.n).sum() / d.n.sum(), include_groups=False).unstack() * 100
    T.columns = [short(c) for c in T.columns]
    L.append("## A4. Truth sensitivity: recall@10 self-free (%) under other definitions of 'truly nearest'\n")
    L.append("Default: the 15 identity measures, block-weighted. The old atlas space is the 27 measures used before 2026-11-04 "
             "(adds location, scatter and the six near-duplicate columns); the maps are still fit on the identity space.\n")
    L.append(md(T.round(0), digits=0) + "\n")
    # A5 mixing and drift
    rows = []
    for v in ["Full space"] + sub:
        s = A[A.view == v]
        w = s.n / s.n.sum()
        rows.append({"view": short(v), "same hand %": 100 * (s.same_hand * w).sum(), "chance %": 100 * (s.chance_hand * w).sum(),
                     "same season %": 100 * (s.same_season * w).sum(), "chance season %": 100 * (s.chance_season * w).sum(),
                     "other-season top-10 %": 100 * (s.other_season_top10 * w).sum(), "n-drift Spearman": (s.n_drift_spearman * w).sum()})
    L.append("## A5. Neighbourhood mixing (10-NN), same-pitcher-other-season recovery, sample-size drift\n")
    L.append(md(pd.DataFrame(rows), index=False) + "\n")
    # A6 other-season by n quartile
    o = Ar[Ar.view.isin(["Full space"] + sub) & Ar.other_season_hit.notna()]
    T = o.groupby(["view", "nq"]).other_season_hit.mean().unstack() * 100
    T.index = [short(v) for v in T.index]
    T.columns = [f"n quartile {int(c) + 1}" for c in T.columns]
    L.append("## A6. Same pitcher, other season, within 10 nearest (%) by pitch-count quartile of the query row\n")
    L.append(md(T.round(0), digits=0) + "\n")

    # B
    Br = D["B_rows"]
    if "B_rows2" in D:
        Br = pd.concat([Br, D["B_rows2"]], ignore_index=True)
    L.append("## B1. Refresh-style held-out test: second half placed on a map of first halves\n")
    L.append("Own first half among the 10 nearest embedded first halves (%, CI); top-1; median rank. "
             f"Chance for top-10 is {100 * (10 / Br[Br.scenario == 'refresh'].groupby('type').n_ref.first()).mean():.1f}%.\n")
    rows = []
    for (v, pl), s in Br[Br.scenario == "refresh"].groupby(["view", "placement"], sort=False):
        dist = B.mean(s.pitcher, (s["rank"] <= 10).astype(float))
        rows.append({"view": short(v), "placement": pl, "top-10 % [CI]": ci_str(*summarize(dist)), "top-1 %": 100 * (s["rank"] <= 1).mean(),
                     "median rank": s["rank"].median(), "pairs": len(s)})
    L.append(md(pd.DataFrame(rows), index=False) + "\n")
    L.append("## B2. Stranger-style held-out test (pitcher folds, both halves placed on a map fit without that pitcher)\n")
    rows = []
    for (v, pl), s in Br[Br.scenario == "stranger"].groupby(["view", "placement"], sort=False):
        dist = B.mean(s.pitcher, (s["rank"] <= 10).astype(float))
        fold = s.groupby("fold").apply(lambda d: (d["rank"] <= 10).mean(), include_groups=False)
        rows.append({"view": short(v), "placement": pl, "top-10 % [CI]": ci_str(*summarize(dist)),
                     "fold range %": f"{100 * fold.min():.0f}-{100 * fold.max():.0f}", "median rank": s["rank"].median()})
    L.append(md(pd.DataFrame(rows), index=False) + "\n")
    L.append("## B3. Refresh-style top-10 (%) by pitch-count quartile (second-half pitch count), transform placement\n")
    r = Br[(Br.scenario == "refresh") & (Br.placement.isin(["as is", "transform"]))].copy()
    r["nq"] = pd.qcut(r.n, 4, labels=False, duplicates="drop")
    T = r.groupby(["view", "nq"]).apply(lambda d: (d["rank"] <= 10).mean(), include_groups=False).unstack() * 100
    T.index = [short(v) for v in T.index]
    T.columns = [f"n quartile {int(c) + 1}" for c in T.columns]
    L.append(md(T.round(0), digits=0) + "\n")

    # C
    Cr = D["C_rows"]
    types = sorted(Cr.type.unique())
    nt = Cr[(Cr.view == "Full space") & (Cr.k == 30) & (Cr.outcome == "whiff_pct")].groupby("type").size()

    def spear_boot(s):
        """Weighted mean over types of within-type Spearman, bootstrapped over pitchers."""
        tot = np.zeros(B.mult.shape[0])
        est = 0.0
        for pt in types:
            st = s[s.type == pt]
            if st.empty:
                continue
            ra, rp = st.actual.rank().to_numpy(), st.pred.rank().to_numpy()
            dist = B.corr(st.pitcher, ra, rp)
            tot += nt[pt] * dist
            est += nt[pt] * np.corrcoef(ra, rp)[0, 1]
        return est / nt.sum(), tot / nt.sum()

    L.append("## C1. Outcome signal in neighbourhoods: Spearman between a pitch's outcome and the mean of its 30 nearest other-pitcher dots\n")
    rel = reliability()
    rows = [{"view": "Half-to-half reliability r (pitchers with 40+ pitches in both halves)", **{k: f"{v:.2f}" for k, v in rel.items()}},
            {"view": "Ceiling for a season outcome, sqrt(2r / (1 + r)) (Spearman-Brown)", **{k: f"{np.sqrt(2 * v / (1 + v)):.2f}" for k, v in rel.items()}}]
    for v in ["Full space", "Savant 5 space (5-D)", "Movement (ride x HB)", "PCA 2", "t-SNE 2D", "t-SNE 3D", "PaCMAP 2D (15)", "PaCMAP 3D (15)", "PHATE 3D",
              "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "UMAP 3D (15, 0.3)", "UMAP 3D (10, 0.1)", "Random 2D"]:
        row = {"view": short(v)}
        for oc in ["whiff_pct", "csw_pct", "xwoba_contact", "rv100_raw", "rv100"]:
            s = Cr[(Cr.view == v) & (Cr.k == 30) & (Cr.outcome == oc)]
            if s.empty:
                continue
            est, dist = spear_boot(s)
            row[oc] = ci_str(est, *np.percentile(dist, [2.5, 97.5]), pct=False)
        rows.append(row)
    L.append(md(pd.DataFrame(rows), index=False) + "\n")
    L.append("## C2. Whiff-rate Spearman by neighbourhood size k\n")
    rows = []
    for v in ["Full space", "Movement (ride x HB)", "PCA 2", "t-SNE 2D", "UMAP 2D (15, 0.3)", "UMAP 3D (15, 0.3)", "PaCMAP 3D (15)"]:
        row = {"view": short(v)}
        for k in [10, 30, 100]:
            s = Cr[(Cr.view == v) & (Cr.k == k) & (Cr.outcome == "whiff_pct")]
            row[f"k={k}"] = f"{spear_boot(s)[0]:.2f}"
        rows.append(row)
    L.append(md(pd.DataFrame(rows), index=False) + "\n")
    L.append("## C3. Location added back (identity + location, zone rate) and the old 27-measure space: k = 30\n")
    rows = []
    for v in ["Full space", "Full space, + location", "Full space, old 27 measures", "t-SNE 3D", "t-SNE 3D, + location"]:
        row = {"view": v}
        for oc in ["whiff_pct", "csw_pct", "xwoba_contact", "rv100_raw"]:
            s = Cr[(Cr.view == v) & (Cr.k == 30) & (Cr.outcome == oc)]
            row[oc] = f"{spear_boot(s)[0]:.2f}"
        rows.append(row)
    L.append(md(pd.DataFrame(rows), index=False) + "\n")
    L.append("## C4. R² of the 30-neighbour mean as a predictor (weighted)\n")
    rows = []
    for v in ["Full space", "Movement (ride x HB)", "PCA 2", "t-SNE 2D", "UMAP 2D (15, 0.3)", "UMAP 3D (15, 0.3)", "PaCMAP 3D (15)", "Random 2D"]:
        row = {"view": short(v)}
        for oc in ["whiff_pct", "csw_pct", "xwoba_contact", "rv100_raw", "rv100"]:
            s = Cr[(Cr.view == v) & (Cr.k == 30) & (Cr.outcome == oc)]
            r2 = s.groupby("type").apply(lambda d: 1 - ((d.actual - d.pred) ** 2).sum() / ((d.actual - d.actual.mean()) ** 2).sum(), include_groups=False)
            row[oc] = f"{(r2 * nt).sum() / nt.sum():.2f}"
        rows.append(row)
    L.append(md(pd.DataFrame(rows), index=False) + "\n")

    # D
    Ds = D["D_stability"]

    def wm(d, c):
        return (d[c] * d.n).sum() / d.n.sum()
    L.append("## D1. Seed stability (5 seeds): neighbourhood overlap between runs, Procrustes disparity, island share, recall range\n")
    rows = []
    for (m, dd), s in Ds[Ds.test == "seeds"].groupby(["method", "dims"]):
        rows.append({"method": m, "dims": dd, "overlap@10": wm(s, "overlap10"), "overlap@50": wm(s, "overlap50"),
                     "Procrustes": wm(s, "procrustes"), "island share": wm(s, "island_share_mean"),
                     "recall@10 min-max %": f"{100 * wm(s, 'recall10_free_min'):.0f}-{100 * wm(s, 'recall10_free_max'):.0f}"})
    L.append(md(pd.DataFrame(rows), index=False) + "\n")
    L.append("## D2. Refresh: layout fit on first-half profiles, refit on full-season profiles (common rows compared)\n")
    L.append(f"Reference (not a bound): the full space's own neighbourhoods are {100 * REFRESH_REF[10]:.0f}% (k = 10) and "
             f"{100 * REFRESH_REF[50]:.0f}% (k = 50) the same between the two profile tables (71% at k = 50 in the old 27-measure space); "
             "a map above it is suppressing real change. "
             "Single seeded fit per type; weighted over types.\n")
    rows = []
    done = set(D["D_refresh2"].method.unique()) if "D_refresh2" in D else set()
    for (m, dd, tst), s in Ds[Ds.test.str.startswith("refresh") & ~Ds.method.isin(done)].groupby(["method", "dims", "test"]):
        rows.append({"method": m, "dims": dd, "start": tst.replace("refresh ", "") + (" (500 epochs)" if tst == "refresh warm" else " (reg 0.01)" if tst == "refresh aligned" else ""),
                     "overlap@10": wm(s, "overlap10"), "overlap@50": wm(s, "overlap50"),
                     "Procrustes": wm(s, "procrustes"), "island share": wm(s, "island_share_mean"), "recall@10 % after refit": 100 * wm(s, "recall10_free_min")})
    if "D_refresh2" in D:
        D2 = D["D_refresh2"]
        for (m, dd, tst), s in D2.groupby(["method", "dims", "test"]):
            rows.append({"method": m, "dims": dd, "start": tst.replace("refresh ", ""), "overlap@10": wm(s, "overlap10"), "overlap@50": wm(s, "overlap50"),
                         "Procrustes": wm(s, "procrustes"), "island share": wm(s, "island_share_mean"), "recall@10 % after refit": 100 * wm(s, "recall10_free")})
    L.append(md(pd.DataFrame(rows).sort_values(["method", "dims", "start"]), index=False) + "\n")

    # E
    E = D["E_settings"]
    for meth, a, b, title in [("UMAP", "n_neighbors", "min_dist", "E1. UMAP 2D grid"), ("PaCMAP", "n_neighbors", "MN_ratio", "E3. PaCMAP 2D grid"),
                              ("t-SNE", "perplexity", "exaggeration", "E4. t-SNE 2D grid")]:
        for dd in sorted(E[E.method == meth].dims.unique()):
            s = E[(E.method == meth) & (E.dims == dd)]
            L.append(f"## {title.replace('2D', f'{dd}D') if meth == 'UMAP' else title}: recall@10 self-free % / distance corr / continuity / island share "
                     f"(rows {a}, columns {b})\n")
            for col, nm, dg in [("recall10_free", "recall@10 %", 0), ("dist_spearman", "distance corr", 2), ("continuity10", "continuity", 2), ("island_share", "island share", 2)]:
                T = s.groupby(["a", "b"]).apply(lambda d: wm(d, col), include_groups=False).unstack()
                if dg == 0:
                    T = T * 100
                T.index.name, T.columns.name = a, None
                L.append(f"{nm}:\n\n" + md(T.round(dg), digits=dg) + "\n")
    if "E_settings3d" in D:
        E3 = D["E_settings3d"]
        for meth, a, title in [("PaCMAP", "n_neighbors", "E3b. PaCMAP 3D grid"), ("t-SNE", "perplexity", "E4b. t-SNE 3D grid (columns: exaggeration)")]:
            s3 = E3[E3.method == meth]
            T = s3.groupby(["a", "b"]).apply(lambda d: pd.Series({"recall@10 %": 100 * wm(d, "recall10_free"), "dist corr": wm(d, "dist_spearman"),
                                                                 "continuity": wm(d, "continuity10"), "island share": wm(d, "island_share")}), include_groups=False)
            T.index.names = [a, "b"]
            L.append(f"## {title}\n")
            L.append(md(T.round(2)) + "\n")
    s = E[E.method.str.contains("dims")]
    T = s.groupby(["method", "dims"]).apply(lambda d: pd.Series({"recall@10 %": 100 * wm(d, "recall10_free"), "trust": wm(d, "trust10"),
                                                                 "continuity": wm(d, "continuity10"), "dist corr": wm(d, "dist_spearman")}), include_groups=False)
    L.append("## E5. Output dimensions\n")
    L.append(md(T.round(2)) + "\n")

    # F
    F = D["F_redundancy"]
    T = F.groupby(["a", "b"]).agg(types=("type", "size"), mean_abs_r=("abs_r", "mean")).reset_index().sort_values("mean_abs_r", ascending=False)
    L.append("## F. Feature pairs with |r| >= 0.8 within a pitch type (count of types, mean |r|)\n")
    L.append(md(T, index=False) + "\n")

    # G
    if "G_views" not in D:
        (OUT / "tables.md").write_text("\n".join(L))
        return A1
    G = D["G_views"]
    L.append("## G1. The All-pitches map (global scaling, 7,889 dots)\n")
    Gr = D["G_rows"]
    rows = []
    for v in G.view:
        s = Gr[Gr.view == v]
        dist = B.mean(s.pitcher, s.recall10_free)
        g = G[G.view == v].iloc[0]
        rows.append({"view": short(v), "dims": int(g.dims), "recall@10 self-free % [CI]": ci_str(*summarize(dist)), "type purity %": 100 * g.type_purity,
                     "type kNN acc %": 100 * g.type_knn_acc, "trust": g.trust10, "continuity": g.continuity10, "dist corr": g.dist_spearman, "fit s": g.fit_seconds})
    L.append(md(pd.DataFrame(rows), index=False) + "\n")
    GB = D["G_B_rows"]
    L.append("## G2. All-pitches map, refresh-style held-out placement (own first half among 10 nearest, %)\n")
    rows = []
    for (v, pl), s in GB.groupby(["view", "placement"], sort=False):
        dist = B.mean(s.pitcher, (s["rank"] <= 10).astype(float))
        rows.append({"view": short(v), "placement": pl, "top-10 % [CI]": ci_str(*summarize(dist)), "median rank": s["rank"].median()})
    L.append(md(pd.DataFrame(rows), index=False) + "\n")
    L.append("## G3. All-pitches map, refresh stability (cold vs warm start)\n")
    L.append(md(D["G_D"].round(3), index=False) + "\n")
    L.append("## Software\n")
    L.append(md(pd.DataFrame([META["versions"]]).T.rename(columns={0: "version"})) + "\n")
    (OUT / "tables.md").write_text("\n".join(L))
    return A1


# ------------------------------------------------------------------ figures
def figures(D, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from figure_style import PITCH_COLORS, PITCH_NAMES, apply_style, cell_tag, save, style_axes

    apply_style()
    A, Ar, Br, Cr, Ds, E = D["A_views"], D["A_rows"], D["B_rows"], D["C_rows"], D["D_stability"], D["E_settings"]
    if "B_rows2" in D:
        Br = pd.concat([Br, D["B_rows2"]], ignore_index=True)
    coords = D["A_coords"]
    types = sorted(A.type.unique())
    W = 174 / 25.4
    METHOD_COLOR = {"UMAP": "#4575b4", "t-SNE": "#d73027", "PaCMAP": "#e0a800", "PHATE": "#7f7f7f", "PCA": "#999999", "Movement": "#999999",
                    "densMAP": "#74add1", "LocalMAP": "#f0c040", "Isomap": "#bbbbbb", "TriMap": "#bbbbbb", "Savant": "#999999", "Random": "#cccccc"}

    def mcolor(v):
        if "(10, 0.1)" in v or "10/0.1" in v:
            return "#1a3f8f"            # the built UMAP setting: darker blue
        if v in ("Full space", "Full space, no intent"):
            return "#222222"
        if v.startswith("Movement"):
            return "#8c8c8c"
        if v.startswith("PCA"):
            return "#b5b5b5"
        for k, c in METHOD_COLOR.items():
            if v.startswith(k):
                return c
        return "#999999"

    def lstyle(v):
        if "3D" in v:
            return "--"
        if v.startswith("Movement"):
            return "-."
        if v.startswith("PCA"):
            return ":"
        return "-"

    def ci_bar(ax, names, longdf, col, transform=lambda x: x, horizontal=True, label_fn=short):
        ys, ms, los, his = [], [], [], []
        for v in names:
            s = longdf[longdf.view == v]
            m, lo, hi = summarize(B.mean(s.pitcher, transform(s[col].to_numpy())))
            ms.append(m); los.append(lo); his.append(hi)
        y = np.arange(len(names))
        if horizontal:
            ax.barh(y, ms, color=[mcolor(v) for v in names], height=0.7)
            ax.errorbar(ms, y, xerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
            ax.set_yticks(y); ax.set_yticklabels([label_fn(v) for v in names]); ax.invert_yaxis()
        style_axes(ax)
        return ms

    # ---- Fig 1: example maps (largest type), islands and velocity
    pt = "FF" if "FF" in types else types[0]
    c = coords[coords.pitch_type == pt]
    maps = [m for m in ["PCA 2", "PaCMAP 2D (15)", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "t-SNE 2D"] if f"{m}|x" in c]
    fig = plt.figure(figsize=(W, W * 0.5))
    gs = gridspec.GridSpec(2, len(maps) + 2, figure=fig, width_ratios=[1] * len(maps) + [0.28, 1.15], wspace=0.08, hspace=0.12, left=0.06, right=0.98)
    isl = c.island.to_numpy()
    cmap = plt.get_cmap("tab10")
    axes_for_cb = []
    for j, m in enumerate(maps):
        for row, (col, title) in enumerate([("island", "full-space Leiden island"), ("release_speed", "velocity (mph)")]):
            ax = fig.add_subplot(gs[row, j])
            x, y = c[f"{m}|x"], c[f"{m}|y"]
            if col == "island":
                ax.scatter(x, y, c=[cmap(i % 10) for i in isl], s=2, lw=0, alpha=0.8)
            else:
                sc = ax.scatter(x, y, c=c.release_speed, cmap="viridis", s=2, lw=0, alpha=0.8)
                axes_for_cb.append(ax)
            ax.set_xticks([]); ax.set_yticks([])
            style_axes(ax)
            if row == 0:
                ax.set_title(short(m).replace(" 2D", ""), fontsize=6.5)
            if j == 0:
                ax.set_ylabel(title, fontsize=6)
            if row == 0 and j == 0:
                cell_tag(ax, "A", x=-0.05)
    # annotated: same pitcher across seasons on the production map; cases chosen in the FEATURE space
    # (largest and smallest season-to-season displacement among pitchers with 3 seasons of >= 150 pitches), not on the map
    ax = fig.add_subplot(gs[:, len(maps) + 1])
    m = "t-SNE 2D"
    ax.scatter(c[f"{m}|x"], c[f"{m}|y"], c="#d0d0d0", s=2, lw=0)
    # displacement on SHAPE + DELIVERY measures only (z-scored within the type): the consistency block's scatter
    # statistics spike in single seasons and would otherwise pick the "movers"
    from mlbpitch import profiles as _prof
    _full = pd.read_parquet(_prof.profiles_path("type", _prof.SEASONS, by_half=False))
    _g = _full[_full.pitch_type == pt].reset_index(drop=True)
    assert (_g.pitcher.to_numpy() == c.pitcher.to_numpy()).all()
    _cols = _prof.MEAN_FEATURES["shape"] + _prof.MEAN_FEATURES["delivery"]
    _Z = _g[_cols].astype("float64")
    _Z = ((_Z - _Z.mean()) / _Z.std(ddof=0)).fillna(0).to_numpy()
    cc = c.assign(_i=np.arange(len(c)))
    multi = cc.groupby("pitcher").filter(lambda d: d.season.nunique() == 3 and d.n.min() >= 150)
    disp = multi.groupby("pitcher").apply(lambda d: np.linalg.norm(np.diff(_Z[d.sort_values("season")._i.to_numpy()], axis=0), axis=1).sum(), include_groups=False)
    pick = list(disp.nsmallest(3).index) + list(disp.nlargest(3).index)
    xmid = float(np.median(c[f"{m}|x"]))
    for i, p_ in enumerate(pick):
        d = multi[multi.pitcher == p_].sort_values("season")
        col = cmap(i % 10)
        ax.plot(d[f"{m}|x"], d[f"{m}|y"], "-o", color=col, ms=2.5, lw=0.8, mec="none")
        x0, y0 = d[f"{m}|x"].iloc[-1], d[f"{m}|y"].iloc[-1]
        right = x0 > xmid
        ax.annotate(d.player_name.iloc[0].split(",")[0], (x0, y0), fontsize=5, color=col, ha="right" if right else "left",
                    xytext=(-3 if right else 3, 3), textcoords="offset points")
    span = np.ptp(c[[f"{m}|x", f"{m}|y"]].to_numpy(), axis=0).max()
    with open(OUT / "fig1b_cases.csv", "w") as fh:
        fh.write("pitcher,name,group,shape_delivery_displacement,map_path_share\n")
        for i, p_ in enumerate(pick):
            d = multi[multi.pitcher == p_].sort_values("season")
            path = np.linalg.norm(np.diff(d[[f"{m}|x", f"{m}|y"]].to_numpy(), axis=0), axis=1).sum() / span
            fh.write(f"{p_},{d.player_name.iloc[0].replace(',', '')},{'stable' if i < 3 else 'mover'},{disp[p_]:.2f},{path:.3f}\n")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("same pitcher, 2024 to 2026", fontsize=7)
    style_axes(ax)
    cell_tag(ax, "B", x=-0.02)
    cax = axes_for_cb[-1].inset_axes([1.05, 0.1, 0.06, 0.8])
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.set_title("mph", fontsize=5); cb.ax.tick_params(labelsize=5)
    save(fig, str(OUT / "fig1_maps"))

    # ---- Fig 2: neighbour preservation
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.78))
    ax = axes[0, 0]
    names = [v for v in MAIN if v in A.view.unique()] + ["Random 2D"]
    names = sorted(names, key=lambda v: -A[A.view == v].recall10_free.mean())
    ci_bar(ax, names, Ar, "recall10_free", transform=lambda x: 100 * x)
    ax.axvline(100 * (10 / (A.groupby("type").n.first() - 1)).mean(), color="black", lw=0.5, ls=":")
    ax.set_xlabel("True neighbours kept, self-free (%)")
    cell_tag(ax, "A", x=-0.45)
    ax = axes[0, 1]
    cur = D["A_curve"]
    for v in ["Movement (ride x HB)", "PCA 2", "PaCMAP 2D (15)", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "t-SNE 2D", "UMAP 3D (15, 0.3)", "UMAP 3D (10, 0.1)", "t-SNE 3D"]:
        s = cur[cur.view == v]
        if s.empty:
            continue
        w = s.groupby("k").apply(lambda d: (d.recall * d.n).sum() / d.n.sum(), include_groups=False)
        ax.plot(w.index, 100 * w.values, "o", ms=2, lw=0.9, color=mcolor(v), ls=lstyle(v), label=short(v))
    ax.set_xscale("log"); ax.set_xlabel("neighbourhood size k"); ax.set_ylabel("True neighbours kept (%)")
    ax.legend(frameon=False, fontsize=5, ncol=2)
    style_axes(ax, legend=True)
    cell_tag(ax, "B")
    ax = axes[1, 0]
    labelled = {"Movement (ride x HB)", "PCA 2", "PCA 3", "PaCMAP 2D (15)", "PaCMAP 3D (15)", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)",
                "UMAP 3D (15, 0.3)", "UMAP 3D (10, 0.1)", "t-SNE 2D", "t-SNE 3D", "densMAP 3D (15, 0.3)", "PHATE 3D", "TriMap 2D", "t-SNE 2D, exag 4", "Random 2D"}
    for v in names:
        s = A[A.view == v]
        w = s.n / s.n.sum()
        x, y = (s.dist_spearman * w).sum(), 100 * (s.recall10_free * w).sum()
        ax.plot(x, y, "o" if "3D" not in v else "^", color=mcolor(v), ms=4, mec="none")
        if v in labelled:
            off = {"t-SNE 2D": (-30, 6), "UMAP 3D (10, 0.1)": (6, 3), "UMAP 3D (15, 0.3)": (6, 0), "UMAP 2D (10, 0.1)": (-48, 6),
                   "UMAP 2D (15, 0.3)": (-40, 4), "TriMap 2D": (6, 2), "PaCMAP 2D (15)": (14, -16), "PaCMAP 3D (15)": (6, -6),
                   "PHATE 3D": (8, -2), "t-SNE 2D, exag 4": (6, -3), "PCA 2": (6, 2), "Movement (ride x HB)": (6, -9),
                   "PCA 3": (6, 2), "densMAP 3D (15, 0.3)": (-20, 9), "Random 2D": (4, 2), "t-SNE 3D": (6, 1)}.get(v, (3, 2))
            ax.annotate(short(v), (x, y), fontsize=4.5, xytext=off, textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.3, color="#666666", shrinkA=0, shrinkB=2))
    for meth, mk in [("UMAP", "."), ("PaCMAP", "x"), ("t-SNE", "+")]:
        s = E[(E.method == meth) & (E.dims == 2)]
        g = s.groupby(["a", "b"]).apply(lambda d: pd.Series({"x": (d.dist_spearman * d.n).sum() / d.n.sum(), "y": 100 * (d.recall10_free * d.n).sum() / d.n.sum()}), include_groups=False)
        ax.plot(g.x, g.y, mk, color=mcolor(meth), ms=3, alpha=0.5)
    ax.set_xlabel("Distance rank-correlation (global structure)"); ax.set_ylabel("True neighbours kept, self-free (%)")
    style_axes(ax)
    cell_tag(ax, "C")
    ax = axes[1, 1]
    cor = D["A_coranking"]
    cc = cor[cor.type == pt]
    for v in ["PCA 2", "PaCMAP 2D (15)", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "t-SNE 2D", "UMAP 3D (15, 0.3)", "t-SNE 3D"]:
        s = cc[cc.view == v]
        if s.empty:
            continue
        ax.plot(s.K, s.qnx, lw=0.9, color=mcolor(v), ls=lstyle(v), label=short(v))
    nK = int(cc.K.max()) + 1
    ax.plot(np.arange(1, nK), np.arange(1, nK) / (nK - 1), color="black", lw=0.5, ls=":", label="chance")
    ax.set_xscale("log"); ax.set_xlabel("K"); ax.set_ylabel(f"Q_NX(K), {PITCH_NAMES.get(pt, pt)}")
    ax.legend(frameon=False, fontsize=5)
    style_axes(ax, legend=True)
    cell_tag(ax, "D")
    fig.tight_layout()
    save(fig, str(OUT / "fig2_preservation"))

    # ---- Fig 3: held-out
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.78))
    ax = axes[0, 0]
    r = Br[Br.scenario == "refresh"]
    combos = [("Full space", "as is"), ("Movement (ride x HB)", "as is"), ("PCA 2", "as is"), ("t-SNE 2D", "transform"), ("t-SNE 2D", "IDW 15, all rows"),
              ("PaCMAP 2D (15)", "transform"), ("UMAP 2D (15, 0.3)", "transform"), ("UMAP 2D (10, 0.1)", "transform"), ("UMAP 2D (10, 0.1)", "IDW 15, all rows"),
              ("t-SNE 3D", "transform"), ("t-SNE 3D", "IDW 15, all rows"), ("PaCMAP 3D (15)", "transform"), ("PaCMAP 3D (15)", "IDW 15, all rows"),
              ("UMAP 3D (15, 0.3)", "transform"), ("UMAP 3D (15, 0.3)", "IDW 15, all rows"), ("UMAP 3D (10, 0.1)", "transform"), ("UMAP 3D (10, 0.1)", "IDW 15, all rows")]
    labels, ms, los, his = [], [], [], []
    for v, pl in combos:
        s = r[(r.view == v) & (r.placement == pl)]
        if s.empty:
            continue
        m, lo, hi = summarize(B.mean(s.pitcher, 100 * (s["rank"] <= 10).astype(float)))
        labels.append(short(v) + ("" if pl in ("as is", "transform") else " (IDW)")); ms.append(m); los.append(lo); his.append(hi)
    y = np.arange(len(labels))
    ax.barh(y, ms, color=[mcolor(l) for l in labels], height=0.7)
    ax.errorbar(ms, y, xerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("Own first half in 10 nearest (%)")
    style_axes(ax)
    cell_tag(ax, "A", x=-0.5)
    ax = axes[0, 1]
    for v, pl in [("Full space", "as is"), ("Movement (ride x HB)", "as is"), ("PCA 2", "as is"), ("t-SNE 2D", "transform"), ("UMAP 2D (15, 0.3)", "transform"),
                  ("UMAP 3D (15, 0.3)", "transform"), ("t-SNE 3D", "transform")]:
        s = r[(r.view == v) & (r.placement == pl)]
        if s.empty:
            continue
        rk = np.sort(s["rank"].to_numpy())
        ax.plot(rk, np.arange(1, len(rk) + 1) / len(rk), lw=0.9, color=mcolor(v), ls=lstyle(v), label=short(v))
    ax.set_xscale("log"); ax.set_xlabel("rank of own first half"); ax.set_ylabel("share of second halves")
    ax.legend(frameon=False, fontsize=5)
    style_axes(ax, legend=True)
    cell_tag(ax, "B")
    ax = axes[1, 0]
    o = Ar[Ar.other_season_hit.notna()]
    for v in ["Full space", "Movement (ride x HB)", "PCA 2", "t-SNE 2D", "UMAP 2D (15, 0.3)", "UMAP 3D (15, 0.3)", "t-SNE 3D"]:
        s = o[o.view == v]
        if s.empty:
            continue
        g = s.groupby("nq").other_season_hit.mean() * 100
        ax.plot(g.index + 1, g.values, "o", ms=2.5, lw=0.9, color=mcolor(v), ls=lstyle(v), label=short(v))
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("pitch-count quartile of the row"); ax.set_ylabel("Other season in 10 nearest (%)")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, fontsize=4.5, loc="upper left", ncol=4, columnspacing=0.8, handlelength=1.6)
    style_axes(ax, legend=True)
    cell_tag(ax, "C")
    ax = axes[1, 1]
    st = Br[Br.scenario == "stranger"]
    labels, ms, los, his = [], [], [], []
    for v, pl in [("Full space", "as is"), ("Movement (ride x HB)", "as is"), ("PCA 2", "as is"), ("t-SNE 2D", "transform"), ("t-SNE 2D", "IDW 15"),
                  ("PaCMAP 2D (15)", "transform"), ("UMAP 2D (15, 0.3)", "transform"), ("UMAP 2D (15, 0.3)", "IDW 15"), ("UMAP 2D (10, 0.1)", "transform"),
                  ("UMAP 2D (10, 0.1)", "IDW 15"), ("t-SNE 3D", "transform"), ("t-SNE 3D", "IDW 15"), ("PaCMAP 3D (15)", "transform"), ("PaCMAP 3D (15)", "IDW 15"),
                  ("UMAP 3D (15, 0.3)", "transform"), ("UMAP 3D (15, 0.3)", "IDW 15"), ("UMAP 3D (10, 0.1)", "transform"), ("UMAP 3D (10, 0.1)", "IDW 15")]:
        s = st[(st.view == v) & (st.placement == pl)]
        if s.empty:
            continue
        m, lo, hi = summarize(B.mean(s.pitcher, 100 * (s["rank"] <= 10).astype(float)))
        labels.append(short(v) + ("" if pl in ("as is", "transform") else " (IDW)")); ms.append(m); los.append(lo); his.append(hi)
    y = np.arange(len(labels))
    ax.barh(y, ms, color=[mcolor(l) for l in labels], height=0.7)
    ax.errorbar(ms, y, xerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("Stranger: own first half in 10 nearest (%)")
    style_axes(ax)
    cell_tag(ax, "D", x=-0.5)
    fig.tight_layout()
    save(fig, str(OUT / "fig3_heldout"))

    # ---- Fig 4: outcomes
    nt = Cr[(Cr.view == "Full space") & (Cr.k == 30) & (Cr.outcome == "whiff_pct")].groupby("type").size()

    def spear(s):
        tot, est = np.zeros(B.mult.shape[0]), 0.0
        for t_ in types:
            st_ = s[s.type == t_]
            if st_.empty:
                continue
            ra, rp = st_.actual.rank().to_numpy(), st_.pred.rank().to_numpy()
            tot += nt[t_] * B.corr(st_.pitcher, ra, rp); est += nt[t_] * np.corrcoef(ra, rp)[0, 1]
        return est / nt.sum(), np.percentile(tot / nt.sum(), [2.5, 97.5])
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.72))
    ax = axes[0, 0]
    cv = ["Full space", "Savant 5 space (5-D)", "Movement (ride x HB)", "PCA 2", "PaCMAP 2D (15)", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "t-SNE 2D",
          "PHATE 3D", "PaCMAP 3D (15)", "UMAP 3D (15, 0.3)", "UMAP 3D (10, 0.1)", "t-SNE 3D", "Random 2D"]
    cv = [v for v in cv if v in Cr.view.unique()]
    ms, los, his = [], [], []
    for v in cv:
        est, (lo, hi) = spear(Cr[(Cr.view == v) & (Cr.k == 30) & (Cr.outcome == "whiff_pct")])
        ms.append(est); los.append(lo); his.append(hi)
    y = np.arange(len(cv))
    bars = ax.barh(y, ms, color=[mcolor(v) for v in cv], height=0.7)
    for b, v in zip(bars, cv):
        if v in ("Full space", "Savant 5 space (5-D)"):
            b.set_hatch("////"); b.set_edgecolor("white")
    ax.errorbar(ms, y, xerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
    ax.set_yticks(y); ax.set_yticklabels([short(v) + (" (space)" if v in ("Full space", "Savant 5 space (5-D)") else "") for v in cv]); ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Whiff rate: Spearman with 30-neighbour mean")
    style_axes(ax)
    cell_tag(ax, "A", x=-0.45)
    ax = axes[0, 1]
    for v in ["Full space", "Movement (ride x HB)", "PCA 2", "t-SNE 2D", "UMAP 2D (15, 0.3)", "UMAP 3D (15, 0.3)", "PaCMAP 3D (15)"]:
        ys = [spear(Cr[(Cr.view == v) & (Cr.k == k) & (Cr.outcome == "whiff_pct")])[0] for k in [10, 30, 100]]
        ax.plot([10, 30, 100], ys, "o", ms=2.5, lw=0.9, color=mcolor(v), ls=lstyle(v), label=short(v))
    ax.set_xscale("log"); ax.set_xticks([10, 30, 100]); ax.set_xticklabels(["10", "30", "100"])
    ax.set_xlabel("neighbourhood size k"); ax.set_ylabel("Whiff rate Spearman")
    ax.legend(frameon=False, fontsize=5)
    style_axes(ax, legend=True)
    cell_tag(ax, "B")
    ax = axes[1, 0]
    ocs = ["whiff_pct", "csw_pct", "xwoba_contact", "rv100_raw", "rv100"]
    oc_names = ["whiff", "CSW", "xwOBA con.", "RV/100 raw", "RV/100 shrunk"]
    vv = ["Full space", "Movement (ride x HB)", "UMAP 2D (15, 0.3)", "UMAP 3D (15, 0.3)", "t-SNE 2D"]
    wdt = 0.8 / len(vv)
    for i, v in enumerate(vv):
        est = [spear(Cr[(Cr.view == v) & (Cr.k == 30) & (Cr.outcome == oc)]) for oc in ocs]
        ys = [e[0] for e in est]
        ax.bar(np.arange(len(ocs)) + i * wdt - 0.4 + wdt / 2, ys, width=wdt, color=mcolor(v), alpha=0.9 if "3D" not in v else 0.6, label=short(v),
               yerr=[[y - e[1][0] for y, e in zip(ys, est)], [e[1][1] - y for y, e in zip(ys, est)]], error_kw={"elinewidth": 0.5, "capsize": 1})
    ax.set_xticks(range(len(ocs))); ax.set_xticklabels(oc_names, rotation=20, ha="right"); ax.set_ylabel("Spearman, k = 30"); ax.axhline(0, color="black", lw=0.5)
    ax.legend(frameon=False, fontsize=5)
    style_axes(ax, legend=True)
    cell_tag(ax, "C")
    ax = axes[1, 1]
    vv = ["Full space", "Full space, + location", "t-SNE 3D", "t-SNE 3D, + location"]
    ocs2 = ["whiff_pct", "csw_pct", "xwoba_contact"]
    wdt = 0.8 / len(vv)
    for i, v in enumerate(vv):
        est = [spear(Cr[(Cr.view == v) & (Cr.k == 30) & (Cr.outcome == oc)]) for oc in ocs2]
        ys = [e[0] for e in est]
        ax.bar(np.arange(len(ocs2)) + i * wdt - 0.4 + wdt / 2, ys, width=wdt, color=mcolor(v), alpha=0.5 if "location" in v else 0.9, label=v,
               yerr=[[y - e[1][0] for y, e in zip(ys, est)], [e[1][1] - y for y, e in zip(ys, est)]], error_kw={"elinewidth": 0.5, "capsize": 1})
    ax.set_xticks(range(len(ocs2))); ax.set_xticklabels(["whiff", "CSW", "xwOBA con."]); ax.set_ylabel("Spearman, k = 30"); ax.axhline(0, color="black", lw=0.5)
    ax.legend(frameon=False, fontsize=5)
    style_axes(ax, legend=True)
    cell_tag(ax, "D")
    fig.tight_layout()
    save(fig, str(OUT / "fig4_outcomes"))

    # ---- Fig 5: stability (per-type dots on every bar; refresh ceiling line)
    def wm(d, c):
        return (d[c] * d.n).sum() / d.n.sum()
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.72))
    methods = list(Ds.method.unique())
    for ax, col, title, tag in [(axes[0, 0], "overlap50", "Seeds: 50-neighbourhood overlap", "A"), (axes[0, 1], "procrustes", "Seeds: Procrustes disparity", "B")]:
        for j, dd in enumerate([2, 3]):
            sub = [Ds[(Ds.test == "seeds") & (Ds.method == m) & (Ds.dims == dd)] for m in methods]
            xs = np.arange(len(methods)) + (j - 0.5) * 0.38
            ax.bar(xs, [wm(d, col) for d in sub], width=0.36, color=[mcolor(m) for m in methods], alpha=0.9 if dd == 2 else 0.5, label=f"{dd}D")
            for x, d in zip(xs, sub):
                ax.plot(np.full(len(d), x), d[col], ".", color="black", ms=2, alpha=0.6)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=25, ha="right"); ax.set_ylabel(title)
        ax.legend(frameon=False, fontsize=5)
        style_axes(ax, legend=True)
        cell_tag(ax, tag)
    D2 = D.get("D_refresh2")
    series = [("t-SNE (perp 30)", ["refresh cold", "refresh warm 500 iters", "refresh warm 250 iters", "refresh warm 100 iters"], ["cold", "500", "250", "100"]),
              ("UMAP (10, 0.1)", ["refresh cold", "refresh aligned reg 0.1", "refresh aligned reg 1.0"], ["cold", "A 0.1", "A 1.0"]),
              ("UMAP (15, 0.3)", ["refresh cold", "refresh aligned reg 0.1", "refresh aligned reg 1.0"], ["cold", "A 0.1", "A 1.0"])]
    for ax, dd, tag in [(axes[1, 0], 2, "C"), (axes[1, 1], 3, "D")]:
        for m, tests, labs in series:
            pts = []
            for t_ in tests:
                d = D2[(D2.method == m) & (D2.dims == dd) & (D2.test == t_)]
                pts.append((wm(d, "overlap50"), 100 * wm(d, "recall10_free")))
            pts = np.array(pts)
            col = mcolor("UMAP 2D (10, 0.1)" if m == "UMAP (10, 0.1)" else m)
            ax.plot(pts[:, 0], pts[:, 1], "-o", color=col, ms=3, lw=0.9, label=m)
            for (x, y), lab in zip(pts, labs):
                ax.annotate(lab, (x, y), fontsize=4.5, xytext=(3, 2), textcoords="offset points", color=col)
            # UMAP warm restart (100 epochs): a single grey-edged marker, to show it does not move
            if m.startswith("UMAP"):
                d = D2[(D2.method == m) & (D2.dims == dd) & (D2.test == "refresh warm 100 epochs")]
                ax.plot(wm(d, "overlap50"), 100 * wm(d, "recall10_free"), "s", mfc="white", mec=col, ms=3.5)
        for m in ["PaCMAP (15)", "t-SNE (exag 4)"]:
            d = Ds[(Ds.method == m) & (Ds.dims == dd) & (Ds.test == "refresh cold")]
            ax.plot(wm(d, "overlap50"), 100 * wm(d, "recall10_free_min"), "D", color=mcolor(m), ms=3.5, label=f"{m}, cold")
        ax.axvline(REFRESH_REF[50], color="black", lw=0.5, ls=":")
        ax.text(REFRESH_REF[50] + 0.003, ax.get_ylim()[0] + 0.5, "full-space\nreference", fontsize=4.5, va="bottom")
        ax.set_xlabel(f"Refresh {dd}D: 50-neighbourhood overlap"); ax.set_ylabel("True neighbours kept after refit (%)")
        ax.legend(frameon=False, fontsize=4.5, loc="lower left")
        style_axes(ax, legend=True)
        cell_tag(ax, tag)
    fig.tight_layout()
    save(fig, str(OUT / "fig5_stability"))

    # ---- Fig 6: settings
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.72))

    def heat(ax, s, col, title, scale=1, fmt="{:.0f}", vmin=None, vmax=None):
        g = s.groupby(["a", "b"]).apply(lambda d: wm(d, col), include_groups=False).unstack() * scale
        im = ax.imshow(g.to_numpy(), cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(g.shape[1])); ax.set_xticklabels(g.columns); ax.set_yticks(range(g.shape[0])); ax.set_yticklabels(g.index)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, fmt.format(g.iloc[i, j]), ha="center", va="center", fontsize=5.5, color="white" if g.iloc[i, j] > g.to_numpy().mean() else "black")
        ax.set_title(title, fontsize=7)
        style_axes(ax)
    s = E[(E.method == "UMAP") & (E.dims == 2)]
    heat(axes[0, 0], s, "recall10_free", "UMAP 2D: true neighbours kept (%)", 100, vmin=10, vmax=40); axes[0, 0].set_xlabel("min_dist"); axes[0, 0].set_ylabel("n_neighbors"); cell_tag(axes[0, 0], "A")
    heat(axes[0, 1], s, "dist_spearman", "UMAP 2D: distance rank-correlation", 1, "{:.2f}"); axes[0, 1].set_xlabel("min_dist"); cell_tag(axes[0, 1], "B")
    s3 = E[(E.method == "UMAP") & (E.dims == 3)]
    heat(axes[1, 0], s3, "recall10_free", "UMAP 3D: true neighbours kept (%)", 100, vmin=10, vmax=40); axes[1, 0].set_xlabel("min_dist"); axes[1, 0].set_ylabel("n_neighbors"); cell_tag(axes[1, 0], "C")
    ax = axes[1, 1]
    for m in sorted(E[E.method.str.contains("dims")].method.unique()):
        s = E[E.method == m]
        g = s.groupby("dims").apply(lambda d: 100 * wm(d, "recall10_free"), include_groups=False)
        col = mcolor(m)
        ax.plot(g.index, g.values, "-o", ms=3, lw=0.9, color=col, label=m.replace(" dims", ""))
        for pt_, d in s.groupby("type"):
            ax.plot(d.dims, 100 * d.recall10_free, ".", color=col, ms=2, alpha=0.4)
    ax.set_xlabel("output dimensions"); ax.set_ylabel("True neighbours kept (%)"); ax.set_xticks([2, 3, 4, 5])
    ax.legend(frameon=False, fontsize=5)
    style_axes(ax, legend=True)
    cell_tag(ax, "D")
    fig.tight_layout()
    save(fig, str(OUT / "fig6_settings"))

    # ---- Fig S1: Shepard diagrams (largest type)
    pc = [c_ for c_ in coords.columns if c_.startswith("p") and c_[1:].isdigit()]
    Pf = c[pc].to_numpy()
    rng = np.random.default_rng(SEED)
    i, j = rng.integers(0, len(c), 20000), rng.integers(0, len(c), 20000)
    m_ = i != j
    dh = np.linalg.norm(Pf[i[m_]] - Pf[j[m_]], axis=1)
    sh = [v for v in ["PCA 2", "PaCMAP 2D (15)", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "t-SNE 2D"] if f"{v}|x" in c]
    fig, axes = plt.subplots(1, len(sh), figsize=(W, W / len(sh) * 1.05))
    for ax, v in zip(np.atleast_1d(axes), sh):
        V = c[[f"{v}|x", f"{v}|y"]].to_numpy()
        dl = np.linalg.norm(V[i[m_]] - V[j[m_]], axis=1)
        ax.scatter(dh, dl, s=0.5, alpha=0.15, color=mcolor(v), lw=0)
        from scipy.stats import spearmanr
        ax.set_title(f"{short(v)}  rho = {spearmanr(dh, dl).statistic:.2f}", fontsize=6)
        ax.set_xlabel("full-space distance"); ax.set_yticks([])
        style_axes(ax)
    axes[0].set_ylabel("map distance")
    fig.tight_layout()
    save(fig, str(OUT / "figS1_shepard"))

    # ---- Fig S2: feature correlation (largest type)
    from mlbpitch import profiles
    feats = [f for fs in [profiles.MEAN_FEATURES[b] for b in ("shape", "derived", "delivery", "intent")] + [profiles.SD_FEATURES] for f in fs]
    full = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False))
    Cm = full[full.pitch_type == pt][feats].astype("float64").corr()
    fig, ax = plt.subplots(figsize=(W * 0.7, W * 0.7))
    im = ax.imshow(Cm.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=4.5); ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=4.5)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(f"{PITCH_NAMES.get(pt, pt)}: feature correlations", fontsize=7)
    style_axes(ax)
    fig.tight_layout()
    save(fig, str(OUT / "figS2_correlation"))


def figure_global(D):
    """Fig. 7: the All-pitches map on five 2D layouts, coloured by pitch type."""
    if "G_coords" not in D:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure_style import PITCH_COLORS, PITCH_NAMES, apply_style, cell_tag, save, style_axes
    apply_style()
    c = D["G_coords"]
    maps = [m for m in ["PCA 2", "PaCMAP 2D (15)", "UMAP 2D (15, 0.3)", "UMAP 2D (10, 0.1)", "t-SNE 2D"] if f"{m}|x" in c]
    W = 174 / 25.4
    fig, axes = plt.subplots(1, len(maps), figsize=(W, W / len(maps) * 1.15))
    for j, (ax, m) in enumerate(zip(np.atleast_1d(axes), maps)):
        for pt, g in c.groupby("pitch_type"):
            ax.scatter(g[f"{m}|x"], g[f"{m}|y"], s=1.2, lw=0, alpha=0.7, color=PITCH_COLORS.get(pt, "k"), label=PITCH_NAMES.get(pt, pt))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(short(m), fontsize=7)
        style_axes(ax)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=5, markerscale=4, handletextpad=0.2)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save(fig, str(OUT / "fig7_allpitches"))


# first-half vs full-season overlap of the identity space's own neighbourhoods (computed 2026-11-04 on the 15 measures)
REFRESH_REF = {10: 0.77, 50: 0.84}
LOC_GROUPS = {"Fastballs (FF, SI)": ["FF", "SI"], "Breaking (SL, ST, CU, FC)": ["SL", "ST", "CU", "FC"], "Offspeed (CH, FS)": ["CH", "FS"]}


def location_tables(D, B):
    """L1: per pitch type, how his location gap to his lookalikes goes with his whiff / CSW gap, in each space."""
    from scipy.stats import spearmanr
    Lr = D.get("L_rows")
    if Lr is None:
        return ""
    L = ["## L1. Same pitch, different location: Spearman between his gap to his 30 nearest lookalikes in location and in results\n",
         "Signed, within pitch type. Height: + higher. Whiff and CSW in points. Identity space = the atlas's new definition of similar; "
         "old atlas space = with location and scatter in it. 95% pitcher-clustered bootstrap intervals for the identity space.\n"]
    rows = []
    for pt, g in Lr.groupby("type"):
        row = {"type": pt, "n": int((g.space == "Identity space (15)").sum())}
        for sname, sk in [("Identity space (15)", "identity"), ("Old atlas space (27)", "old"), ("t-SNE 3D map", "t-SNE 3D"), ("Movement plot", "movement")]:
            d = g[g.space == sname]
            row[f"height~whiff, {sk}"] = float(spearmanr(d.d_height_in, d.d_whiff_pts, nan_policy="omit").statistic)
        d = g[g.space == "Identity space (15)"]
        ra, rb = d.d_height_in.rank().to_numpy(), d.d_whiff_pts.rank().to_numpy()
        lo, hi = np.nanpercentile(B.corr(d.pitcher, ra, rb), [2.5, 97.5])
        row["identity CI"] = f"[{lo:.2f}, {hi:.2f}]"
        row["zone~CSW, identity"] = float(spearmanr(d.d_zone_pts, d.d_csw_pts, nan_policy="omit").statistic)
        row["inside~whiff, identity"] = float(spearmanr(d.d_inside_in, d.d_whiff_pts, nan_policy="omit").statistic)
        row["release scatter~whiff, identity"] = float(spearmanr(d.d_release_sd_in, d.d_whiff_pts, nan_policy="omit").statistic)
        ok = d[["d_height_in", "d_whiff_pts"]].dropna()
        row["whiff pts per inch higher"] = float(np.polyfit(ok.d_height_in, ok.d_whiff_pts, 1)[0])
        row["SD of height gap, in"] = float(d.d_height_in.std())
        rows.append(row)
    T = pd.DataFrame(rows).set_index("type")
    L.append(md(T.round(2)) + "\n")
    return "\n".join(L)


def figure_location(D, B):
    """Fig. 8: same pitch, different location."""
    Lr = D.get("L_rows")
    if Lr is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from figure_style import PITCH_COLORS, PITCH_NAMES, apply_style, cell_tag, save, style_axes
    apply_style()
    W = 174 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.78))
    idn = Lr[Lr.space == "Identity space (15)"]
    gcol = {"Fastballs (FF, SI)": "#d73027", "Breaking (SL, ST, CU, FC)": "#4575b4", "Offspeed (CH, FS)": "#7f7f7f"}
    # A: binned height gap vs whiff gap, by family
    ax = axes[0, 0]
    for gname, ts in LOC_GROUPS.items():
        d = idn[idn.type.isin(ts)].dropna(subset=["d_height_in", "d_whiff_pts"])
        q = pd.qcut(d.d_height_in, 10, labels=False, duplicates="drop")
        m = d.groupby(q).agg(x=("d_height_in", "mean"), y=("d_whiff_pts", "mean"), s=("d_whiff_pts", "std"), n=("d_whiff_pts", "size"))
        ax.errorbar(m.x, m.y, yerr=1.96 * m.s / np.sqrt(m.n), fmt="-o", ms=2.5, lw=0.9, elinewidth=0.5, capsize=1, color=gcol[gname], label=gname)
    ax.axhline(0, color="black", lw=0.4); ax.axvline(0, color="black", lw=0.4)
    ax.set_xlabel("Location height vs. his lookalikes (in, + higher)"); ax.set_ylabel("Whiff rate vs. his lookalikes (pts)")
    ax.legend(frameon=False, fontsize=5)
    style_axes(ax, legend=True)
    cell_tag(ax, "A")
    # B: signed Spearman per pitch type, identity vs old space vs movement plot
    ax = axes[0, 1]
    types = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]
    spaces = [("Identity space (15)", "Lookalikes on what the pitch is", "#1a3f8f"), ("Old atlas space (27)", "Old: location in the definition", "#9ecae1"),
              ("Movement plot", "Movement-plot neighbours", "#bbbbbb")]
    wdt = 0.26
    for i, (sname, lab, col) in enumerate(spaces):
        ys, lo, hi = [], [], []
        for t in types:
            d = Lr[(Lr.type == t) & (Lr.space == sname)].dropna(subset=["d_height_in", "d_whiff_pts"])
            ys.append(spearmanr(d.d_height_in, d.d_whiff_pts).statistic)
            dist = B.corr(d.pitcher, d.d_height_in.rank().to_numpy(), d.d_whiff_pts.rank().to_numpy())
            lo.append(np.percentile(dist, 2.5)); hi.append(np.percentile(dist, 97.5))
        xs = np.arange(len(types)) + (i - 1) * wdt
        ax.bar(xs, ys, width=wdt, color=col, label=lab)
        ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], fmt="none", ecolor="black", elinewidth=0.5, capsize=1)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xticks(range(len(types))); ax.set_xticklabels([PITCH_NAMES[t] for t in types], rotation=35, ha="right")
    ax.set_ylabel("Spearman: height gap vs. whiff gap")
    ax.legend(frameon=False, fontsize=4.5, loc="lower left")
    style_axes(ax, legend=True)
    cell_tag(ax, "B")
    # C: whiff signal of neighbourhoods: identity, + location, old 27
    ax = axes[1, 0]
    Cr = D["C_rows"]
    nt = Cr[(Cr.view == "Full space") & (Cr.k == 30) & (Cr.outcome == "whiff_pct")].groupby("type").size()
    vv = [("Full space", "Identity (15)"), ("Full space, + location", "Identity + location"), ("Full space, old 27 measures", "Old atlas (27)"),
          ("t-SNE 3D", "t-SNE 3D map"), ("t-SNE 3D, + location", "t-SNE 3D, + location")]
    for i, oc in enumerate(["whiff_pct", "csw_pct"]):
        ys = []
        for v, _ in vv:
            sub = Cr[(Cr.view == v) & (Cr.k == 30) & (Cr.outcome == oc)]
            est = sum(nt[t] * np.corrcoef(g.actual.rank(), g.pred.rank())[0, 1] for t, g in sub.groupby("type")) / nt.sum()
            ys.append(est)
        ax.bar(np.arange(len(vv)) + (i - 0.5) * 0.38, ys, width=0.36, color=["#1a3f8f", "#6baed6", "#bbbbbb", "#d73027", "#f4a582"],
               alpha=1.0 if i == 0 else 0.55, label=["whiff", "CSW"][i])
    ax.set_xticks(range(len(vv))); ax.set_xticklabels([l for _, l in vv], rotation=25, ha="right")
    ax.set_ylabel("Spearman with 30-neighbour mean")
    ax.legend(frameon=False, fontsize=5)
    style_axes(ax, legend=True)
    cell_tag(ax, "C")
    # D: four-seamers, every pitch
    ax = axes[1, 1]
    d = idn[idn.type == "FF"].dropna(subset=["d_height_in", "d_whiff_pts"])
    ax.scatter(d.d_height_in, d.d_whiff_pts, s=1.5, lw=0, alpha=0.4, color=PITCH_COLORS["FF"])
    b = np.polyfit(d.d_height_in, d.d_whiff_pts, 1)
    xs = np.linspace(d.d_height_in.quantile(0.01), d.d_height_in.quantile(0.99), 20)
    ax.plot(xs, np.polyval(b, xs), color="black", lw=0.8)
    ax.text(0.03, 0.95, f"{b[0]:+.2f} whiff pts per inch higher\nrho = {spearmanr(d.d_height_in, d.d_whiff_pts).statistic:.2f}, n = {len(d):,}",
            transform=ax.transAxes, fontsize=5, va="top")
    ax.axhline(0, color="black", lw=0.4); ax.axvline(0, color="black", lw=0.4)
    ax.set_xlabel("Four-seamer height vs. lookalikes (in)"); ax.set_ylabel("Whiff vs. lookalikes (pts)")
    style_axes(ax)
    cell_tag(ax, "D")
    fig.tight_layout()
    save(fig, str(OUT / "fig8_location"))


def main():
    D = load()
    pitchers = pd.concat([D[k].pitcher for k in ("A_rows", "B_rows", "C_rows", "G_rows", "L_rows") if k in D]).unique()
    B = Boot(pitchers)
    tables(D, B)
    lt = location_tables(D, B)
    if lt:
        with open(OUT / "tables.md", "a") as fh:
            fh.write("\n" + lt)
    print("tables.md written")
    figures(D, B)
    figure_global(D)
    figure_location(D, B)
    print("figures written")


if __name__ == "__main__":
    main()
