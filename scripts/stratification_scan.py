"""Pitch "ancestry": a GWAS-style control for population structure in pitch-outcome associations.

In a GWAS, a SNP can associate with a phenotype only because both vary with ancestry; the fix is to put the top
principal components of the genotype matrix in the model (Price et al. 2006), or to match on ancestry, or to compare
siblings. The pitch analogue:

  SNP          -> one measure of a pitch (location height, release scatter, usage, velocity, ...)
  phenotype    -> whiff rate (also CSW, xwOBA on contact, raw run value), rank-inverse-normal within pitch type
  ancestry     -> what the pitch IS: principal components of the atlas's 15 identity measures, within pitch type
  continents   -> pitch types: every analysis runs separately within one type; types are never pooled
  relatedness  -> rows of the same pitcher (standard errors clustered by pitcher throughout)
  LOCO         -> a measure that is itself one of the identity measures is tested against PCs built WITHOUT its
                  own block (shape / derived / delivery), as GWAS LMMs leave the tested chromosome out of the GRM

Control ladder (each a model of phenotype on the standardised measure):
  type       no adjustment (every test, here and below, is within one pitch type: types are never pooled)
  pc3        within type + top 3 PCs                                  (the GWAS standard)
  pcall      within type + every PC of the identity space             (linear adjustment for the whole pitch)
  lookalike  within type + the mean phenotype and mean measure of the 30 nearest other-pitcher lookalikes
             (ancestry matching, in the spirit of the atlas's Fig. 4 gaps; the lookalike mean phenotype plays the role of
             REGENIE's LOCO polygenic prediction. A plain gap-on-gap regression gives conservative SEs because
             neighbouring rows share lookalikes: lambda 0.4 on permuted phenotypes in the smoke test)
  within     within pitcher x pitch type, across seasons, + all PCs   (sibling / within-family design; unlike a
             genotype, a pitch's "ancestry" can change between seasons, so the PCs are demeaned and kept in)

Calibration: synthetic null measures that track pitch ancestry but have no other link to the phenotype ("broad":
random combinations of PCs 1-3; "fine": a kNN-smoothed random field over the identity space), and permuted
phenotypes. QQ plots and lambda_GC = median(chi2) / 0.4549.

Outputs in analysis/stratification/: assoc.parquet (every test), nulls.parquet, lambda.csv, ladder.csv,
location_slopes.csv, pcs.csv (variance + loadings), Figure5.{pdf,png}, report.md.

    python scripts/stratification_scan.py            # full run (minutes)
    python scripts/stratification_scan.py --quick    # FF + SL only, few nulls (smoke test)
"""
import argparse
import glob
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from mlbpitch import paths, profiles
from profile_embed import IDENTITY_BLOCKS, block_scale, zscore

OUT = paths.ROOT / "analysis" / "stratification"
TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]
OUTCOMES = {"whiff_pct": "Whiff rate", "csw_pct": "CSW", "xwoba_contact": "xwOBA on contact", "rv100_raw": "Run value per 100"}
K_LOOK = 30
SEED = 0

# what is tested ("SNPs"), by block. Identity blocks include the near-duplicates dropped from the atlas's space.
TESTED = {
    "shape": profiles.MEAN_FEATURES["shape"],
    "derived": profiles.MEAN_FEATURES["derived"],
    "delivery": profiles.MEAN_FEATURES["delivery"],
    "location": profiles.MEAN_FEATURES["intent"],
    "scatter": list(profiles.SD_FEATURES),
    "usage": list(profiles.USAGE_FEATURES) + ["n"],
    "pitcher": ["height_in"],          # negative control: a trait of the pitcher, not the pitch
}
LOCO = {"shape", "derived", "delivery"}
MODELS = ["type", "pc3", "pcall", "lookalike", "within"]
MODEL_NAMES = {"type": "No adjustment", "pc3": "+ 3 PCs", "pcall": "+ all PCs",
               "lookalike": "Lookalike-matched", "within": "Within pitcher + PCs"}
# pre-specified case examples for the ladder (measure, type)
CASES = [("release_speed", "FF"), ("release_spin_rate", "FF"), ("plate_z", "FF"), ("plate_z", "SL"), ("plate_z", "CU"),
         ("release_sd", "FF"), ("usage", "SL"), ("height_in", "FF")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def int_transform(y):
    """Rank-based inverse normal transform (Blom), NaN kept."""
    y = np.asarray(y, dtype="float64")
    out = np.full_like(y, np.nan)
    ok = np.isfinite(y)
    r = pd.Series(y[ok]).rank(method="average").to_numpy()
    out[ok] = norm.ppf((r - 0.375) / (ok.sum() + 0.25))
    return out


def standardise(x):
    x = np.asarray(x, dtype="float64")
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s if s > 0 else 1.0)


def ols_cluster(y, x, C, groups, intercept=True):
    """Coefficient of x in y ~ x (+ 1) (+ C), pitcher-clustered sandwich SE (CR1). Rows with NaN dropped."""
    cols = [x[:, None]]
    if intercept:
        cols.append(np.ones((len(x), 1)))
    if C is not None and C.shape[1]:
        cols.append(C)
    X = np.hstack(cols)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X, g = y[ok], X[ok], np.asarray(groups)[ok]
    n, k = X.shape
    if n <= k + 2 or np.nanstd(X[:, 0]) < 1e-8:   # no variation (e.g. height within a pitcher): untestable
        return np.nan, np.nan, n, 0
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    codes, uniq = pd.factorize(g)
    G = len(uniq)
    S = np.zeros((G, k))
    np.add.at(S, codes, X * e[:, None])
    V = XtX_inv @ (S.T @ S) @ XtX_inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    return float(b[0]), float(np.sqrt(max(V[0, 0], 0))), int(n), int(G)


def identity_space(g, drop_block=None):
    blocks = {b: cols for b, cols in IDENTITY_BLOCKS.items() if b != drop_block}
    cols = [c for cs in blocks.values() for c in cs]
    F = g[cols].astype("float64")
    F = F.fillna(F.median())
    X = block_scale(zscore(F), blocks)
    pca = PCA(n_components=X.shape[1], random_state=SEED).fit(X)
    return pca.transform(X), pca, cols


def lookalikes(P, pid, k=K_LOOK):
    """Indices of the k nearest rows thrown by other pitchers."""
    nn = NearestNeighbors(n_neighbors=min(len(P), k + 12)).fit(P)
    idx = nn.kneighbors(P, return_distance=False)
    out = np.empty((len(P), k), dtype=int)
    for i, r in enumerate(idx):
        r = r[pid[r] != pid[i]][:k]
        out[i] = r
    return out


def gap(v, nb):
    return v - np.nanmean(v[nb], axis=1)


def within_demean(v, grp):
    s = pd.Series(v)
    return (s - s.groupby(grp).transform("mean")).to_numpy()


class TypeContext:
    """Everything the models need for one pitch type: PCs and lookalikes in the full and leave-one-block-out spaces."""

    def __init__(self, g):
        self.g = g
        self.pid = g.pitcher.to_numpy()
        self.space = {}
        for drop in (None, "shape", "derived", "delivery"):
            P, pca, cols = identity_space(g, drop)
            self.space[drop] = {"P": P, "pca": pca, "cols": cols, "nb": lookalikes(P, self.pid)}
        counts = pd.Series(self.pid).map(pd.Series(self.pid).value_counts()).to_numpy()
        self.multi = counts >= 2          # pitcher has 2+ seasons of this type

    def fit(self, model, y, x, block):
        sp = self.space[block if block in LOCO else None]
        if model == "type":
            return ols_cluster(y, x, None, self.pid)
        if model == "pc3":
            return ols_cluster(y, x, sp["P"][:, :3], self.pid)
        if model == "pcall":
            return ols_cluster(y, x, sp["P"], self.pid)
        if model == "lookalike":
            nb = sp["nb"]
            return ols_cluster(y, x, np.column_stack([np.nanmean(y[nb], axis=1), np.nanmean(x[nb], axis=1)]), self.pid)
        if model == "within":
            m = self.multi
            C = np.column_stack([within_demean(sp["P"][m, j], self.pid[m]) for j in range(sp["P"].shape[1])])
            return ols_cluster(within_demean(y[m], self.pid[m]), within_demean(x[m], self.pid[m]), C, self.pid[m], intercept=False)
        raise ValueError(model)


def load():
    t = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False)).reset_index(drop=True)
    bio = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(paths.DATA / "meta" / "player_bio_*.parquet")))])
    h = bio.dropna(subset=["height_ft"]).drop_duplicates("pitcher_id", keep="last").set_index("pitcher_id").height_ft * 12
    t["height_in"] = t.pitcher.map(h)
    return t


def row(outcome, model, pt, var, block, res):
    b, se, n, G = res
    z = b / se if se and np.isfinite(se) and se > 0 else np.nan
    return {"outcome": outcome, "model": model, "type": pt, "variable": var, "block": block, "beta": b, "se": se,
            "z": z, "p": 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan, "n": n, "n_pitchers": G}


def lam(z):
    z = np.asarray(z, dtype="float64")
    z = z[np.isfinite(z)]
    return float(np.median(z ** 2) / 0.454936) if len(z) else np.nan


def run(types, n_null, n_perm):
    t = load()
    rng = np.random.default_rng(SEED)
    tests, nulls, slopes, pcs, perms = [], [], [], [], []
    var_block = {v: b for b, vs in TESTED.items() for v in vs}

    for pt in types:
        g = t[t.pitch_type == pt].reset_index(drop=True)
        ctx = TypeContext(g)
        P = ctx.space[None]["P"]
        pca = ctx.space[None]["pca"]
        for j in range(P.shape[1]):
            pcs.append({"type": pt, "pc": j + 1, "var_explained": float(pca.explained_variance_ratio_[j]),
                        **{f"load_{c}": float(pca.components_[j, i]) for i, c in enumerate(ctx.space[None]["cols"])}})
        Ys = {oc: int_transform(g[oc]) for oc in OUTCOMES}
        for oc, y in Ys.items():
            for v, b in var_block.items():
                x = standardise(g[v])
                for m in MODELS:
                    tests.append(row(oc, m, pt, v, b, ctx.fit(m, y, x, b)))
        # natural units: whiff points per inch of location height
        yw, xh = 100 * g.whiff_pct.to_numpy(dtype="float64"), 12 * g.plate_z.to_numpy(dtype="float64")
        for m in MODELS:
            b_, se, n, G = ctx.fit(m, yw, xh, "location")
            slopes.append({"type": pt, "model": m, "slope": b_, "lo": b_ - 1.96 * se, "hi": b_ + 1.96 * se, "n": n})
        # synthetic nulls tracking pitch ancestry
        y = Ys["whiff_pct"]
        Pz = (P - P.mean(0)) / P.std(0)
        nb50 = lookalikes(P, ctx.pid, 50)          # fine structure from other pitchers only
        for fam in ("broad", "fine"):
            for _ in range(n_null):
                if fam == "broad":
                    s = Pz[:, :3] @ rng.normal(size=3)
                else:
                    u = rng.normal(size=len(P))
                    s = u[nb50].mean(1)
                s = standardise(s)
                r2 = rng.uniform(0.05, 0.5)
                x = np.sqrt(r2) * s + np.sqrt(1 - r2) * rng.normal(size=len(P))
                for m in MODELS:
                    nulls.append(row("whiff_pct", m, pt, fam, "null", ctx.fit(m, y, x, "null")) | {"family": fam})
        # permuted phenotype, real measures: checks the clustered SEs
        for k in range(n_perm):
            yp = rng.permutation(y)
            for v, b in var_block.items():
                x = standardise(g[v])
                for m in MODELS:
                    perms.append(row("whiff_pct", m, pt, v, b, ctx.fit(m, yp, x, b)) | {"perm": k})
        log(f"{pt}: {len(g)} rows done")

    A, N, S = pd.DataFrame(tests), pd.DataFrame(nulls), pd.DataFrame(slopes)
    Pm = pd.DataFrame(perms)
    lam_rows = []
    for m in MODELS:
        r = {"model": m}
        for oc in OUTCOMES:
            r[f"real_{oc}"] = lam(A[(A.model == m) & (A.outcome == oc)].z)
        for fam in ("broad", "fine"):
            r[f"null_{fam}"] = lam(N[(N.model == m) & (N.family == fam)].z)
        r["permuted"] = lam(Pm[Pm.model == m].z) if len(Pm) else np.nan
        lam_rows.append(r)
    L = pd.DataFrame(lam_rows)
    lad = []
    for v, pt in CASES:
        if pt not in types:
            continue
        for m in MODELS:
            s = A[(A.outcome == "whiff_pct") & (A.model == m) & (A.variable == v) & (A.type == pt)]
            if len(s):
                q = s.iloc[0]
                lad.append({"variable": v, "type": pt, "model": m, "beta": q.beta, "lo": q.beta - 1.96 * q.se,
                            "hi": q.beta + 1.96 * q.se, "p": q.p})
    return t, A, N, S, L, pd.DataFrame(lad), pd.DataFrame(pcs), Pm


# ------------------------------------------------------------------ figure
def figure(t, A, N, S, L, lad, types):
    sys.path.insert(0, str(paths.ROOT / "scripts"))
    import matplotlib.pyplot as plt
    import umap_benchmark_plots as ubp
    from figure_style import PITCH_COLORS, PITCH_NAMES, PRINT_W, save
    SZ = ubp.setup()
    lw, finish = ubp.lw, ubp.finish
    MM = 1 / 25.4
    fig = plt.figure(figsize=(PRINT_W, 190 * MM), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 0.95, 1.15])
    r1 = gs[0].subgridspec(1, 3)
    r2 = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 2.1])
    r3 = gs[2].subgridspec(1, 2, width_ratios=[1.15, 1])

    # A: ancestry PCs, fitted within each pitch type: cumulative variance explained
    axA = fig.add_subplot(r1[0, 0])
    for pt in types:
        _, pca_t, _ = identity_space(t[t.pitch_type == pt].reset_index(drop=True))
        cv = 100 * np.cumsum(pca_t.explained_variance_ratio_)
        axA.plot(np.arange(1, len(cv) + 1), cv, "-o", ms=1.6, lw=lw(0.75), color=PITCH_COLORS[pt], label=PITCH_NAMES[pt])
    axA.axvline(3, color="black", lw=lw(0.5), ls=":")
    axA.set_xlabel("Principal components"); axA.set_ylabel("Variance explained, cumulative (%)")
    axA.set_ylim(0, 102); axA.set_xticks([1, 3, 5, 10, 15])
    axA.set_title("Ancestry PCs, within each type", fontsize=SZ.axis_ttl, pad=2)
    axA.legend(loc="lower right", ncol=2, handlelength=1.2, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.2, fontsize=SZ.legend)
    finish(axA, legend=True)
    # B, C: four-seamers, coloured by whiff rate and by location height
    g = t[t.pitch_type == "FF"].reset_index(drop=True)
    P, _, _ = identity_space(g)
    for k, (col, lab, scale) in enumerate([("whiff_pct", "Whiff rate (%)", 100), ("plate_z", "Location height (in)", 12)]):
        ax = fig.add_subplot(r1[0, k + 1])
        v = scale * g[col].to_numpy(dtype="float64")
        lo, hi = np.nanpercentile(v, [5, 95])
        sc = ax.scatter(P[:, 0], P[:, 1], c=v, cmap="cividis", vmin=lo, vmax=hi, s=1.0, lw=0, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(lab, fontsize=SZ.axis_text); cb.outline.set_linewidth(lw(0.5)); cb.ax.tick_params(length=0, labelsize=SZ.axis_text)
        Xc = np.column_stack([np.ones(len(P)), P[:, :3]])
        ok = np.isfinite(v)
        res = v[ok] - Xc[ok] @ np.linalg.lstsq(Xc[ok], v[ok], rcond=None)[0]
        rsq = 1 - res.var() / v[ok].var()
        ax.text(0.03, 0.97, f"Four-seamers\nPCs 1-3 explain {100 * rsq:.0f}%", transform=ax.transAxes, va="top", fontsize=SZ.annot)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_xticks([]); ax.set_yticks([])
        finish(ax)
        if k == 0:
            axB = ax
        else:
            axC = ax

    # D, E: QQ plots of the synthetic nulls
    shades = {"type": "#b3b3b3", "pc3": "#92c5de", "pcall": "#2166ac", "lookalike": "#b2182b", "within": "#000000"}
    qq_axes = []
    for k, fam in enumerate(["broad", "fine"]):
        ax = fig.add_subplot(r2[0, k])
        mx = 0
        for m in MODELS:
            p = np.sort(N[(N.model == m) & (N.family == fam)].p.dropna().to_numpy())
            if not len(p):
                continue
            e = -np.log10((np.arange(1, len(p) + 1) - 0.5) / len(p))
            o = -np.log10(np.clip(p, 1e-300, 1))
            o = np.minimum(o, 30)
            ax.plot(e, o, ".", ms=1.4, color=shades[m], label=f"{MODEL_NAMES[m]} ({lam(N[(N.model == m) & (N.family == fam)].z):.2f})")
            mx = max(mx, e.max())
        ax.plot([0, mx], [0, mx], color="black", lw=lw(0.5), ls=":")
        ax.set_xlabel("Expected -log10 p"); ax.set_ylabel("Observed -log10 p")
        ax.set_title("Null measures: broad structure" if fam == "broad" else "Null measures: fine structure", fontsize=SZ.axis_ttl, pad=2)
        ax.legend(loc="upper left", markerscale=3, handletextpad=0.1, borderaxespad=0.2, fontsize=SZ.legend, title="Control (lambda)",
                  title_fontsize=SZ.legend)
        finish(ax, legend=True)
        qq_axes.append(ax)

    # F: Miami plot, real measures, whiff: no adjustment (up) vs + 3 PCs (down)
    axF = fig.add_subplot(r2[0, 2])
    bcol = {"shape": "#2166ac", "derived": "#92c5de", "delivery": "#542788", "location": "#b2182b", "scatter": "#e08214",
            "usage": "#8c510a", "pitcher": "#000000"}
    order = [v for b in TESTED for v in TESTED[b]]
    xpos = {v: i for i, v in enumerate(order)}
    W = A[A.outcome == "whiff_pct"]
    ntest = len(W[W.model == "type"])
    for m, sign in [("type", 1), ("pc3", -1)]:
        s = W[W.model == m]
        jit = s.type.map({pt: (i - 3.5) * 0.08 for i, pt in enumerate(TYPES)}).fillna(0)
        axF.scatter(s.variable.map(xpos) + jit, sign * -np.log10(s.p.clip(lower=1e-30)), s=2.5, lw=0,
                    c=s.block.map(bcol), rasterized=True)
    thr = -np.log10(0.05 / max(ntest, 1))
    for sgn in (1, -1):
        axF.axhline(sgn * thr, color="black", lw=lw(0.5), ls=":")
    axF.axhline(0, color="black", lw=lw(0.5))
    axF.set_xticks(range(len(order))); axF.set_xticklabels(order, rotation=90, fontsize=SZ.axis_text * 0.92)
    yt = axF.get_yticks()
    axF.set_yticks(yt); axF.set_yticklabels([f"{abs(v):g}" for v in yt])
    axF.set_ylim(-31, 31)
    axF.set_ylabel("-log10 p, whiff rate\n+ 3 PCs      no adjustment")
    axF.set_xlim(-0.7, len(order) - 0.3)
    finish(axF)

    # G: ladder for the case examples
    axG = fig.add_subplot(r3[0, 0])
    cases = [c for c in CASES if c[1] in types]
    ys = np.arange(len(cases))[::-1]
    mcol = shades
    for y0, (v, pt) in zip(ys, cases):
        s = lad[(lad.variable == v) & (lad.type == pt)].set_index("model")
        for j, m in enumerate(MODELS):
            if m not in s.index:
                continue
            yy = y0 + (2 - j) * 0.15
            q = s.loc[m]
            axG.plot([q.lo, q.hi], [yy, yy], color="black", lw=lw(0.5))
            axG.plot(q.beta, yy, "o", ms=2.6, mfc=mcol[m], mec="black", mew=lw(0.5))
    axG.axvline(0, color="black", lw=lw(0.5))
    axG.set_yticks(ys); axG.set_yticklabels([f"{v}, {PITCH_NAMES[pt]}" for v, pt in cases])
    axG.set_xlabel("Whiff rate per SD of the measure (SD units)")
    h = [plt.Line2D([], [], ls="", marker="o", ms=2.6, mfc=mcol[m], mec="black", mew=lw(0.5), label=MODEL_NAMES[m]) for m in MODELS]
    axG.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
    finish(axG, legend=True)

    # H: location in natural units across the ladder
    axH = fig.add_subplot(r3[0, 1])
    ys = np.arange(len(types))[::-1]
    for y0, pt in zip(ys, types):
        s = S[S.type == pt].set_index("model")
        for j, m in enumerate(MODELS):
            q = s.loc[m]
            yy = y0 + (2 - j) * 0.15
            axH.plot([q.lo, q.hi], [yy, yy], color="black", lw=lw(0.5))
            axH.plot(q.slope, yy, "o", ms=2.6, mfc=mcol[m], mec="black", mew=lw(0.5))
    axH.axvline(0, color="black", lw=lw(0.5))
    axH.set_yticks(ys); axH.set_yticklabels([PITCH_NAMES[pt] for pt in types])
    axH.set_xlabel("Whiff points per inch higher")
    finish(axH)
    ubp.place_tags(fig, [(axA, "A"), (axB, "B"), (axC, "C"), (qq_axes[0], "D"), (qq_axes[1], "E"), (axF, "F"), (axG, "G"), (axH, "H")])
    save(fig, str(OUT / "Figure5"))


# ------------------------------------------------------------------ report
def md_table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def report(A, N, S, L, lad, pcs, Pm, types, secs):
    lines = ["# Pitch ancestry: GWAS-style control for population structure", "",
             f"*Generated by `scripts/stratification_scan.py` on {time.strftime('%Y-%m-%d %H:%M')} ({secs:.0f} s). "
             "Draft numbers for review; the narrative is to be written after reading them.*", "",
             "Analogy: measure = SNP, whiff rate (rank-inverse-normal within type) = phenotype, PCs of the 15 identity "
             "measures within pitch type = ancestry PCs, pitch type = continental group (analysed separately, never pooled), rows of one pitcher = relatives "
             "(SEs clustered by pitcher), leave-one-block-out PCs for identity measures = LOCO. See the module docstring.", "",
             "![Figure 5](Figure5.png)", "",
             "## 1. Inflation (lambda_GC) by control", "",
             "Real measures include true effects, so their lambda need not reach 1; the synthetic nulls and the "
             "permuted phenotype should.", "", md_table(L), "",
             "## 2. Case examples, whiff rate, SD units (beta [95% CI])", ""]
    if len(lad):
        lad = lad.assign(cell=lambda d: d.apply(lambda r: f"{r.beta:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]" if np.isfinite(r.beta) else "not testable", axis=1))
        piv = lad.pivot_table(index=["variable", "type"], columns="model", values="cell", aggfunc="first").reindex(columns=MODELS).rename(columns=MODEL_NAMES).reset_index()
        lines += [md_table(piv), ""]
    lines += ["## 3. Location height in natural units: whiff points per inch higher", ""]
    S2 = S.assign(cell=lambda d: d.apply(lambda r: f"{r.slope:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]", axis=1))
    lines += [md_table(S2.pivot_table(index="type", columns="model", values="cell", aggfunc="first").reindex(columns=MODELS).reindex(types).rename(columns=MODEL_NAMES).reset_index()), ""]
    W = A[A.outcome == "whiff_pct"]
    a, b = W[W.model == "type"].set_index(["type", "variable"]), W[W.model == "pc3"].set_index(["type", "variable"])
    j = a[["beta", "p"]].join(b[["beta", "p"]], lsuffix="_type", rsuffix="_pc3").dropna()
    j["shrink"] = 1 - (j.beta_pc3.abs() / j.beta_type.abs())
    bonf = 0.05 / max(len(a), 1)
    lines += ["## 4. Whiff: associations that collapse once pitch ancestry is controlled", "",
              f"Significant with no adjustment (p < {bonf:.1e}, Bonferroni over {len(a)} tests) and shrinking most with 3 PCs.", "",
              md_table(j[j.p_type < bonf].sort_values("shrink", ascending=False).head(15).reset_index(), "{:.3g}"), "",
              "## 5. Whiff: associations that survive every control", ""]
    keep = None
    for m in ("type", "pc3", "pcall", "lookalike", "within"):
        s = W[(W.model == m) & (W.p < bonf)].set_index(["type", "variable"]).beta
        keep = s.rename(m).to_frame() if keep is None else keep.join(s.rename(m), how="inner")
    lines += [md_table(keep.rename(columns=MODEL_NAMES).reset_index(), "{:+.2f}") if keep is not None and len(keep) else "None.", "",
              "## 6. Count of Bonferroni-significant tests by outcome and control", ""]
    cnt = A.assign(sig=lambda d: d.p < bonf).groupby(["outcome", "model"]).sig.sum().unstack().reindex(columns=MODELS)
    lines += [md_table(cnt.rename(columns=MODEL_NAMES).reset_index(), "{:.0f}"), "",
              "## 7. What the ancestry PCs are (variance explained and top loadings, PCs 1-3)", ""]
    pr = []
    for (pt, pc), d in pcs[pcs.pc <= 3].groupby(["type", "pc"]):
        ld = d.filter(like="load_").iloc[0]
        top = ld.abs().sort_values(ascending=False).index[:3]
        pr.append({"type": pt, "pc": pc, "var_explained": d.var_explained.iloc[0],
                   "top loadings": ", ".join(f"{c[5:]} {ld[c]:+.2f}" for c in top)})
    lines += [md_table(pd.DataFrame(pr)), ""]
    (OUT / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="FF and SL only, 20 nulls, 1 permutation")
    ap.add_argument("--nulls", type=int, default=200)
    ap.add_argument("--perms", type=int, default=20)
    args = ap.parse_args()
    types = ["FF", "SL"] if args.quick else TYPES
    n_null, n_perm = (20, args.perms) if args.quick else (args.nulls, args.perms)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    t, A, N, S, L, lad, pcs, Pm = run(types, n_null, n_perm)
    A.to_parquet(OUT / "assoc.parquet", index=False)
    N.to_parquet(OUT / "nulls.parquet", index=False)
    Pm.to_parquet(OUT / "perms.parquet", index=False)
    S.to_csv(OUT / "location_slopes.csv", index=False)
    L.to_csv(OUT / "lambda.csv", index=False)
    lad.to_csv(OUT / "ladder.csv", index=False)
    pcs.to_csv(OUT / "pcs.csv", index=False)
    log("tables written")
    print(L.round(2).to_string(index=False), flush=True)
    figure(t, A, N, S, L, lad, types)
    log("figure written")
    report(A, N, S, L, lad, pcs, Pm, types, time.time() - t0)
    log(f"done in {time.time() - t0:.0f} s; outputs in {OUT}")


if __name__ == "__main__":
    main()
