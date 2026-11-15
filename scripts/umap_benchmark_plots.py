"""Figures for analysis/umap_benchmark.md, in the house style (figure_aesthetics: Cell print rules via figure_style).

Main figures tell one story, in order:
  Figure 1  What "similar" should mean: which measures repeat, what drove season-to-season movement, the candidate
            definitions, and the payoff (a pitcher's other season found again).
  Figure 2  Which picture keeps that similarity: the same pitches on four pictures, an example of the atlas's map,
            then how many true neighbours each keeps, at every scale, and whether it finds a pitcher's other season.
  Figure 3  Placing pitchers the map has not seen (returning and stranger), in the built atlas, and refreshing it.
  Figure 4  Same pitch, different location.
Supplementary S1 to S7: every method, Shepard diagrams, settings, seeds, outcomes, the all-pitches map, correlations.

Rules followed (figure_aesthetics/cell_figure_guidelines.md): Arial (Arial Unicode MS where Arial is absent), text
6-8 pt at print, strokes 0.5-1.5 pt, 174 mm wide and at most 200 mm tall, capital panel letters, no red with green,
grey fills 10-80% and at least 20% apart, individual points beside means, figure titles and legends in the text not
the image, PDF (fonts embedded) plus a 300-dpi PNG for review.

Called from umap_benchmark_figures.py (which builds the bootstrap); data prep for Figures 1 and 3 is cached as CSV.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from figure_style import PITCH_COLORS, PITCH_NAMES, PRINT_W, apply_style, save, style_axes
from cell_typography import cell_lw

SZ = None
MM = 1 / 25.4
TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]

# Method colours: hues per family, a light tint for 2D and the full tone for 3D. No green anywhere (red is used).
COL = {"t-SNE 3D": "#b2182b", "t-SNE 2D": "#f4a582", "UMAP 3D": "#2166ac", "UMAP 2D": "#92c5de",
       "PaCMAP 3D": "#8c6d1f", "PaCMAP 2D": "#dfc27d", "Full space": "#333333", "Movement": "#737373",
       "PCA": "#b3b3b3", "Old": "#b3b3b3", "Identity": "#2166ac"}
# Leiden islands: up to 12 categories without green
ISLAND = ["#2166ac", "#92c5de", "#542788", "#b2abd2", "#e08214", "#fdb863", "#8c510a", "#dfc27d",
          "#4d4d4d", "#c51b7d", "#000000", "#bababa"]
MAIN = [("Movement (ride x HB)", "Movement plot", "Movement"), ("PCA 2", "PCA", "PCA"),
        ("PaCMAP 2D (15)", "PaCMAP", "PaCMAP 2D"), ("UMAP 2D (10, 0.1)", "UMAP", "UMAP 2D"), ("t-SNE 2D", "t-SNE", "t-SNE 2D"),
        ("PaCMAP 3D (15)", "PaCMAP", "PaCMAP 3D"), ("UMAP 3D (10, 0.1)", "UMAP", "UMAP 3D"), ("t-SNE 3D", "t-SNE", "t-SNE 3D")]


# ------------------------------------------------------------------ style helpers
def setup():
    global SZ
    SZ = apply_style()
    plt.rcParams["hatch.linewidth"] = lw(0.5)
    return SZ


def lw(pt=0.75):
    return cell_lw(pt)


def finish(ax, legend=False):
    style_axes(ax, legend=legend)
    ax.tick_params(length=0, pad=2)


def place_tags(fig, pairs):
    """Capital panel letters at the top-left corner of each panel's full extent (tick labels, title and legend
    included), on one line across a row: panels in the same row of the figure's top-level grid share the highest
    letter. Drawn as axes text so the constrained layout makes room for them; positions refined."""
    texts = [ax.text(0, 1, letter, transform=ax.transAxes, fontsize=SZ.tag, fontweight="bold", ha="left", va="bottom")
             for ax, letter in pairs]
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        boxes = []
        for (ax, _l), t in zip(pairs, texts):
            t.set_visible(False)
            boxes.append((ax.get_tightbbox(r), ax.get_window_extent(r)))
            t.set_visible(True)
        rows = [ax.get_subplotspec().get_topmost_subplotspec().rowspan.start for ax, _l in pairs]
        for (bb, pos), t, row in zip(boxes, texts, rows):
            top = max(b.y1 for (b, _p), r in zip(boxes, rows) if r == row)
            t.set_position(((bb.x0 - pos.x0) / pos.width, (top - pos.y0) / pos.height))


def wmean(d, col, w="n"):
    return float((d[col] * d[w]).sum() / d[w].sum())


def ci_bars(ax, ys, means, los, his, colors, per_type=None, height=0.7, orient="h"):
    """Bars (weighted mean), 95% CI whiskers and one dot per pitch type (individual points beside the mean)."""
    ys = np.asarray(ys, dtype=float)
    if orient == "h":
        ax.barh(ys, means, height=height, color=colors, edgecolor="none", zorder=1)
        ax.errorbar(means, ys, xerr=[np.subtract(means, los), np.subtract(his, means)], fmt="none", ecolor="black",
                    elinewidth=lw(0.5), capsize=0, zorder=3)
        if per_type is not None:
            for y, pts in zip(ys, per_type):
                ax.plot(pts, np.full(len(pts), y), "o", ms=1.6, mfc="white", mec="black", mew=lw(0.5), zorder=4)
    else:
        ax.bar(ys, means, width=height, color=colors, edgecolor="none", zorder=1)
        ax.errorbar(ys, means, yerr=[np.subtract(means, los), np.subtract(his, means)], fmt="none", ecolor="black",
                    elinewidth=lw(0.5), capsize=0, zorder=3)
        if per_type is not None:
            for x, pts in zip(ys, per_type):
                ax.plot(np.full(len(pts), x), pts, "o", ms=1.6, mfc="white", mec="black", mew=lw(0.5), zorder=4)


# ------------------------------------------------------------------ data prep (cached)
def figure1_data(OUT):
    """Reliability per measure, season-to-season movement share per block (old space), candidate definitions."""
    p_rel, p_mov, p_def = OUT / "F1_reliability.csv", OUT / "F1_movement.csv", OUT / "F1_definitions.csv"
    if p_rel.exists() and p_mov.exists() and p_def.exists():
        return pd.read_csv(p_rel), pd.read_csv(p_mov), pd.read_csv(p_def)
    from mlbpitch import profiles
    from profile_embed import DUPLICATE_FEATURES, IDENTITY_BLOCKS, zscore
    t = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False)).reset_index(drop=True)
    h = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=True)).reset_index(drop=True)
    blocks = {"Shape": profiles.MEAN_FEATURES["shape"], "Derived": profiles.MEAN_FEATURES["derived"],
              "Delivery": profiles.MEAN_FEATURES["delivery"], "Location": profiles.MEAN_FEATURES["intent"],
              "Scatter": profiles.SD_FEATURES, "Usage": profiles.USAGE_FEATURES}
    ident = {c for cs in IDENTITY_BLOCKS.values() for c in cs}
    mm = h[h.half == 1].merge(h[h.half == 2], on=["pitcher", "season", "pitch_type"], suffixes=("_1", "_2"))
    n = mm.groupby("pitch_type").size()
    rel = []
    for b, cols in blocks.items():
        for c in cols:
            per = mm.groupby("pitch_type").apply(lambda d: spearmanr(d[c + "_1"], d[c + "_2"], nan_policy="omit").statistic,
                                                 include_groups=False)
            rel.append({"block": b, "measure": c, "reliability": float((per * n).sum() / n.sum()),
                        "role": "identity" if c in ident else ("duplicate" if c in DUPLICATE_FEATURES else "excluded")})
    rel = pd.DataFrame(rel)

    def space(g, bl, weights=None):
        cols = [c for cs in bl.values() for c in cs]
        Z = zscore(g[cols].astype("float64").fillna(g[cols].median()))
        parts = {k: (weights or {}).get(k, 1.0) * Z[cs].to_numpy() / np.sqrt(np.nansum(np.nanvar(Z[cs].to_numpy(), axis=0)))
                 for k, cs in bl.items()}
        return np.nan_to_num(np.hstack(list(parts.values()))), parts

    old = {k: v for k, v in blocks.items() if k != "Usage"}
    tot = {k: 0.0 for k in old}
    for pt, g in t.groupby("pitch_type"):
        g = g.reset_index(drop=True)
        _, parts = space(g, old)
        for p, d in g.sort_values(["pitcher", "season"]).groupby("pitcher"):
            if len(d) < 2:
                continue
            for k in old:
                tot[k] += np.nansum(np.diff(np.nan_to_num(parts[k][d.index.to_numpy()]), axis=0) ** 2)
    s = sum(tot.values())
    mov = pd.DataFrame([{"block": k, "share": v / s} for k, v in tot.items()])

    dedup = {k: [c for c in v if c not in DUPLICATE_FEATURES] for k, v in old.items()}
    designs = {"Old: all five blocks": (old, None), "Pitch only": ({k: old[k] for k in ("Shape", "Derived", "Delivery")}, None),
               "Pitch only, duplicates dropped (adopted)": ({k: dedup[k] for k in ("Shape", "Derived", "Delivery")}, None),
               "Drop scatter": ({k: old[k] for k in ("Shape", "Derived", "Delivery", "Location")}, None),
               "Location and scatter at half weight": (old, {"Location": 0.5, "Scatter": 0.5})}
    m_rel = rel.groupby("block").reliability.mean()
    designs["Weighted by reliability"] = (old, {k: m_rel[k] / m_rel.max() for k in old})

    def knn(X, k):
        return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    rows = []
    for name, (bl, w) in designs.items():
        for pt, g in t.groupby("pitch_type"):
            g = g.reset_index(drop=True)
            pid = g.pitcher.to_numpy()
            X, _ = space(g, bl, w)
            nb = knn(X, 10)
            other = [np.flatnonzero((pid == p) & (np.arange(len(g)) != r)) for r, p in enumerate(pid)]
            has = np.array([len(o) > 0 for o in other])
            hit = np.mean([np.isin(o, b).any() for o, b, hh in zip(other, nb, has) if hh])
            nb45 = knn(X, 45)
            nf = [r[pid[r] != pid[i]][:30] for i, r in enumerate(nb45)]
            wh = g.whiff_pct.to_numpy()
            pred = np.array([wh[r].mean() for r in nf])
            rows.append({"design": name, "type": pt, "n": len(g), "n_other": int(has.sum()), "same_pitcher": hit,
                         "whiff_rho": spearmanr(wh, pred).statistic})
    dfn = pd.DataFrame(rows)
    dfn.to_csv(p_def, index=False)
    rel.to_csv(p_rel, index=False)
    mov.to_csv(p_mov, index=False)
    return rel, mov, dfn


def cases_data(D, OUT):
    """Six four-seam pitchers for the example panel, chosen on SHAPE + DELIVERY change (z within four-seamers), never on
    the map: the three that changed least and the three that changed most, among pitchers with three seasons of 150+."""
    path = OUT / "F2_cases.csv"
    if path.exists():
        return pd.read_csv(path)
    from mlbpitch import profiles
    c = D["A_coords"]; c = c[c.pitch_type == "FF"].reset_index(drop=True)
    full = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False))
    g = full[full.pitch_type == "FF"].reset_index(drop=True)
    assert (g.pitcher.to_numpy() == c.pitcher.to_numpy()).all()
    cols = profiles.MEAN_FEATURES["shape"] + profiles.MEAN_FEATURES["delivery"]
    Z = g[cols].astype("float64")
    Z = ((Z - Z.mean()) / Z.std(ddof=0)).fillna(0).to_numpy()
    cc = c.assign(_i=np.arange(len(c)))
    multi = cc.groupby("pitcher").filter(lambda d: d.season.nunique() == 3 and d.n.min() >= 150)
    disp = multi.groupby("pitcher").apply(lambda d: np.linalg.norm(np.diff(Z[d.sort_values("season")._i.to_numpy()], axis=0), axis=1).sum(),
                                          include_groups=False)
    rows = [{"pitcher": p, "name": multi[multi.pitcher == p].player_name.iloc[0].replace(",", ""), "group": grp, "displacement": disp[p]}
            for grp, ps in (("stable", disp.nsmallest(3).index), ("mover", disp.nlargest(3).index)) for p in ps]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def callup_data(OUT):
    """Built atlas: does a Triple-A call-up land among the 10 MLB dots nearest his own MLB row? Stored placement
    (t-SNE transform for call-ups) against the 15-neighbour average, per pitch type and map."""
    path = OUT / "F3_callups.csv"
    if path.exists():
        return pd.read_csv(path)
    from sklearn.decomposition import PCA
    import atlas_core as ab
    from mlbpitch import profiles
    a = pd.read_parquet(ab.ATLAS)
    m, x = a[a.level == "mlb"].reset_index(drop=True), a[a.level == "aaa"].reset_index(drop=True)
    tm = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False)).reset_index(drop=True)
    ta = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False, level="aaa")).reset_index(drop=True)
    assert (tm.pitcher.to_numpy() == m.pitcher.to_numpy()).all() and (ta.pitcher.to_numpy() == x.pitcher.to_numpy()).all()
    rows = []
    for pt in TYPES:
        rm, rn = np.flatnonzero(m.pitch_type == pt), np.flatnonzero(x.pitch_type == pt)
        sc = ab.scaler_fit(tm.loc[rm, ab.MAP_FEATURES], blocks=ab.MAP_BLOCKS)
        Xm = ab.scaler_apply(sc, tm.loc[rm, ab.MAP_FEATURES])
        pca = PCA(n_components=min(30, Xm.shape[1]), random_state=0).fit(Xm)
        Pm, Pn = pca.transform(Xm), pca.transform(ab.scaler_apply(sc, ta.loc[rn, ab.MAP_FEATURES]))
        d, i = NearestNeighbors(n_neighbors=15).fit(Pm).kneighbors(Pn)
        w = 1 / (d + 1e-6)
        w /= w.sum(1, keepdims=True)
        pidm = m.pitcher.to_numpy()[rm]
        for cols in (["tx", "ty"], ["tx3", "ty3", "tz3"]):
            M = m.loc[rm, cols].to_numpy(float)
            idw = (w[:, :, None] * M[i]).sum(1)
            tr = x.loc[rn, cols].to_numpy(float)
            for k, r in enumerate(rn):
                own = np.flatnonzero(pidm == x.pitcher[r])
                if not len(own):
                    continue
                for name, pos in (("t-SNE transform (built)", tr[k]), ("Average of 15 neighbours", idw[k])):
                    dd = np.linalg.norm(M - pos, axis=1)
                    rows.append({"type": pt, "dims": len(cols), "placement": name, "pitcher": int(x.pitcher[r]),
                                 "hit": float((dd < dd[own].min()).sum() + 1 <= 10)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------------ Figure 1
def fig1(D, B, OUT):
    rel, mov, dfn = figure1_data(OUT)
    fig = plt.figure(figsize=(PRINT_W, 112 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])
    order = ["Shape", "Derived", "Delivery", "Location", "Scatter", "Usage"]
    xs = np.arange(len(order))
    # A: reliability per measure
    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(0)
    for j, b in enumerate(order):
        r = rel[rel.block == b]
        ax.plot([j - 0.32, j + 0.32], [r.reliability.mean()] * 2, color="black", lw=lw(1.0), zorder=2)
        for _, q in r.iterrows():
            c = COL["Identity"] if q.role == "identity" else "white"
            ec = COL["Identity"] if q.role in ("identity", "duplicate") else "#737373"
            ax.plot(j + rng.uniform(-0.22, 0.22), q.reliability, "o", ms=3, mfc=c if q.role != "excluded" else "#b3b3b3",
                    mec=ec, mew=lw(0.6), zorder=3)
    ax.axvspan(2.5, 5.5, color="#f2f2f2", zorder=0, lw=0)
    ax.set_xticks(xs); ax.set_xticklabels(order)
    ax.set_xlim(-0.6, 5.6); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Half-to-half reliability (Spearman)")
    ax.text(1, 0.43, "defines “similar”", ha="center", va="bottom", fontsize=SZ.annot)
    ax.text(4, 0.43, "shown, not used for similarity", ha="center", va="bottom", fontsize=SZ.annot)
    h = [plt.Line2D([], [], ls="", marker="o", ms=3, mfc=COL["Identity"], mec=COL["Identity"], label="Used"),
         plt.Line2D([], [], ls="", marker="o", ms=3, mfc="white", mec=COL["Identity"], label="Near-duplicate, dropped"),
         plt.Line2D([], [], ls="", marker="o", ms=3, mfc="#b3b3b3", mec="#737373", label="Not used")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.69, 1.0), handletextpad=0.2, borderaxespad=0.3)
    finish(ax, legend=True)
    # B: what moved a pitcher between seasons, old space
    ax2 = fig.add_subplot(gs[0, 1])
    mv = mov.set_index("block").share.reindex(order[:5]) * 100
    cols = [COL["Identity"]] * 3 + ["#b3b3b3"] * 2
    ax2.bar(np.arange(5), mv.values, color=cols, width=0.7)
    for j, v in enumerate(mv.values):
        ax2.text(j, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=SZ.annot)
    ax2.set_xticks(np.arange(5)); ax2.set_xticklabels(order[:5])
    ax2.set_ylabel("Share of season-to-season movement (%)")
    ax2.set_ylim(0, 48)
    ax2.text(0.02, 0.97, "Old 27-measure space, every pitcher\nwith two or more seasons", transform=ax2.transAxes,
             ha="left", va="top", fontsize=SZ.annot)
    finish(ax2)
    # C: the candidate definitions
    ax3 = fig.add_subplot(gs[1, 0])
    agg = dfn.groupby("design").apply(lambda d: pd.Series({"same": 100 * (d.same_pitcher * d.n_other).sum() / d.n_other.sum(),
                                                             "whiff": (d.whiff_rho * d.n).sum() / d.n.sum()}), include_groups=False)
    lab = {"Old: all five blocks": ("Old: all five blocks", (4, -6), "left"),
           "Pitch only": ("Pitch only", (-5, -2), "right"),
           "Pitch only, duplicates dropped (adopted)": ("Pitch only, duplicates\ndropped (adopted)", (-4, 5), "right"),
           "Drop scatter": ("Drop scatter", (4, 3), "left"),
           "Location and scatter at half weight": ("Location and scatter\nat half weight", (-4, -5), "right"),
           "Weighted by reliability": ("Weighted by reliability", (4, 3), "left")}
    for name, r in agg.iterrows():
        adopted = "adopted" in name
        ax3.plot(r.same, r.whiff, "o", ms=4.5 if adopted else 3.5, mfc=COL["Identity"] if adopted else "white",
                 mec=COL["Identity"] if adopted else "black", mew=lw(0.75), zorder=3)
        t_, off, ha = lab[name]
        ax3.annotate(t_, (r.same, r.whiff), xytext=off, textcoords="offset points", ha=ha, va="bottom" if off[1] > 0 else "top",
                     fontsize=SZ.annot, fontweight="bold" if adopted else "normal")
    ax3.set_xlabel("Other season of the same pitcher\namong his 10 nearest (%)")
    ax3.set_ylabel("Whiff rate of the 30 nearest lookalikes\nvs. his own (Spearman)")
    ax3.set_xlim(72, 92); ax3.set_ylim(0.18, 0.43)
    finish(ax3)
    # D: the payoff, per pitch type
    ax4 = fig.add_subplot(gs[1, 1])
    o = dfn[dfn.design == "Old: all five blocks"].set_index("type").same_pitcher * 100
    a = dfn[dfn.design == "Pitch only, duplicates dropped (adopted)"].set_index("type").same_pitcher * 100
    ys = np.arange(len(TYPES))[::-1]
    for y, t_ in zip(ys, TYPES):
        ax4.plot([o[t_], a[t_]], [y, y], color="#b3b3b3", lw=lw(1.0), zorder=1)
        ax4.plot(o[t_], y, "o", ms=3.5, mfc="white", mec="black", mew=lw(0.75), zorder=2)
        ax4.plot(a[t_], y, "o", ms=3.5, mfc=COL["Identity"], mec=COL["Identity"], zorder=3)
    ax4.set_yticks(ys); ax4.set_yticklabels([PITCH_NAMES[t_] for t_ in TYPES])
    ax4.set_xlabel("Other season of the same pitcher among his 10 nearest (%)")
    ax4.set_xlim(55, 100)
    h = [plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", mec="black", label="Old definition"),
         plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc=COL["Identity"], mec=COL["Identity"], label="What the pitch is")]
    ax4.legend(handles=h, loc="lower left", handletextpad=0.2)
    finish(ax4, legend=True)
    place_tags(fig, [(ax, "A"), (ax2, "B"), (ax3, "C"), (ax4, "D")])
    save(fig, str(OUT / "Figure1"))


# ------------------------------------------------------------------ Figure 2
def fig2(D, B, OUT):
    A, Ar, coords, cur = D["A_views"], D["A_rows"], D["A_coords"], D["A_curve"]
    c = coords[coords.pitch_type == "FF"].reset_index(drop=True)
    fig = plt.figure(figsize=(PRINT_W, 132 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 5, height_ratios=[0.62, 1], width_ratios=[1, 1, 1, 1, 1.25])
    # A: one pitch type on four pictures, coloured by full-space island
    maps = [("Movement (ride x HB)", "Movement plot"), ("PCA 2", "PCA"), ("UMAP 2D (10, 0.1)", "UMAP"), ("t-SNE 2D", "t-SNE (the atlas)")]
    isl = c.island.to_numpy()
    cols = [ISLAND[k % len(ISLAND)] for k in isl]
    axA = []
    for j, (m, title) in enumerate(maps):
        ax = fig.add_subplot(gs[0, j])
        ax.scatter(c[f"{m}|x"], c[f"{m}|y"], c=cols, s=1.2, lw=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=SZ.axis_ttl, pad=2)
        ax.set_box_aspect(1)
        finish(ax)
        axA.append(ax)
    axA[0].set_ylabel("Four-seam fastballs,\ncoloured by island", fontsize=SZ.axis_text)
    # B: the atlas map, six pitchers over three seasons
    axB = fig.add_subplot(gs[0, 4])
    m = "t-SNE 2D"
    axB.scatter(c[f"{m}|x"], c[f"{m}|y"], c="#d9d9d9", s=1.2, lw=0, rasterized=True)
    cases = cases_data(D, OUT)
    if len(cases):
        xmid = float(np.median(c[f"{m}|x"]))
        dy = {0: 3, 1: -13, 2: -9, 3: 3, 4: 3, 5: 3}   # stable three cluster together: alternate above and below
        for k, (_, r) in enumerate(cases.iterrows()):
            d = c[c.pitcher == r.pitcher].sort_values("season")
            col = COL["t-SNE 3D"] if r.group == "mover" else COL["UMAP 3D"]
            axB.plot(d[f"{m}|x"], d[f"{m}|y"], "-o", color=col, ms=2.2, lw=lw(0.75), mec="none")
            x0, y0 = d[f"{m}|x"].iloc[-1], d[f"{m}|y"].iloc[-1]
            right = x0 > xmid
            axB.annotate(str(r["name"]).split(" ")[0], (x0, y0), xytext=(-3 if right else 3, dy.get(k, 3)),
                         textcoords="offset points", ha="right" if right else "left", fontsize=SZ.annot, color=col)
    axB.set_xticks([]); axB.set_yticks([])
    axB.set_title("Same pitcher, 2024 to 2026", fontsize=SZ.axis_ttl, pad=2)
    axB.set_box_aspect(1)
    h = [plt.Line2D([], [], color=COL["UMAP 3D"], marker="o", ms=2.2, lw=lw(0.75), label="Pitch unchanged"),
         plt.Line2D([], [], color=COL["t-SNE 3D"], marker="o", ms=2.2, lw=lw(0.75), label="Pitch changed")]
    axB.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, handlelength=1.2, columnspacing=0.6, handletextpad=0.3)
    finish(axB, legend=True)
    # C: true neighbours kept
    gs2 = gs[1, :].subgridspec(1, 3, width_ratios=[1.15, 1, 1])
    axC = fig.add_subplot(gs2[0, 0])
    labels, means, los, his, colors, pts = [], [], [], [], [], []
    for v, name, ck in MAIN:
        s = Ar[Ar.view == v]
        mu, lo, hi = _summ(B.mean(s.pitcher, 100 * s.recall10_free.to_numpy()))
        labels.append(name + (" 3D" if "3D" in v else " 2D" if v not in ("Movement (ride x HB)", "PCA 2") else ""))
        means.append(mu); los.append(lo); his.append(hi); colors.append(COL[ck])
        pts.append(100 * A[A.view == v].recall10_free.to_numpy())
    ys = np.arange(len(labels))[::-1]
    ci_bars(axC, ys, means, los, his, colors, pts)
    axC.set_yticks(ys); axC.set_yticklabels(labels)
    axC.axvline(100 * (10 / (A.groupby("type").n.first() - 1)).mean(), color="black", lw=lw(0.5), ls=":")
    axC.set_xlabel("True neighbours kept (%)")
    axC.set_xlim(0, 60)
    finish(axC)
    # D: at every neighbourhood size
    axD = fig.add_subplot(gs2[0, 1])
    for v, name, ck in [MAIN[0], MAIN[1], MAIN[3], MAIN[4], MAIN[6], MAIN[7]]:
        s = cur[cur.view == v]
        w = s.groupby("k").apply(lambda d: (d.recall * d.n).sum() / d.n.sum(), include_groups=False)
        axD.plot(w.index, 100 * w.values, "-o", ms=1.8, lw=lw(0.75), color=COL[ck], ls="--" if "3D" in v else "-",
                 label=name + (" 3D" if "3D" in v else " 2D" if ck not in ("Movement", "PCA") else ""))
    axD.set_xscale("log")
    axD.set_xlabel("Neighbourhood size (k)")
    axD.set_ylabel("Nearest k kept (%)")
    axD.set_ylim(0, 85)
    axD.legend(loc="upper right", ncol=2, handlelength=1.6, columnspacing=0.6, handletextpad=0.3)
    finish(axD, legend=True)
    # E: a pitcher's other season found again
    axE = fig.add_subplot(gs2[0, 2])
    o = Ar[Ar.other_season_hit.notna()]
    rowsE = [("Full space", "Full space", "Full space")] + [MAIN[0], MAIN[1], MAIN[3], MAIN[4], MAIN[6], MAIN[7]]
    labels, means, los, his, colors, pts = [], [], [], [], [], []
    for v, name, ck in rowsE:
        s = o[o.view == v]
        mu, lo, hi = _summ(B.mean(s.pitcher, 100 * s.other_season_hit.to_numpy()))
        labels.append(name + (" 3D" if "3D" in v else " 2D" if ck not in ("Movement", "PCA", "Full space") else ""))
        means.append(mu); los.append(lo); his.append(hi); colors.append(COL[ck])
        pts.append(100 * A[A.view == v].other_season_top10.to_numpy())
    ys = np.arange(len(labels))[::-1]
    ci_bars(axE, ys, means, los, his, colors, pts)
    axE.set_yticks(ys); axE.set_yticklabels(labels)
    axE.set_xlabel("Other season of the same pitcher\namong his 10 nearest (%)")
    axE.set_xlim(0, 100)
    finish(axE)
    place_tags(fig, [(axA[0], "A"), (axB, "B"), (axC, "C"), (axD, "D"), (axE, "E")])
    save(fig, str(OUT / "Figure2"))


def _summ(dist):
    return float(np.mean(dist)), float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


# ------------------------------------------------------------------ Figure 3
def fig3(D, B, OUT):
    Br = pd.concat([D["B_rows"], D.get("B_rows2", pd.DataFrame())], ignore_index=True)
    cu = callup_data(OUT)
    fig = plt.figure(figsize=(PRINT_W, 118 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    def grouped(ax, scen, rows):
        r = Br[Br.scenario == scen]
        ys, k = [], 0
        ticks, ticklabels = [], []
        for name, ck, pl in rows:
            items = pl if isinstance(pl, list) else [pl]
            for j, (v, place, alpha, hatch) in enumerate(items):
                s = r[(r.view == v) & (r.placement == place)]
                mu, lo, hi = _summ(B.mean(s.pitcher, 100 * (s["rank"] <= 10).astype(float).to_numpy()))
                per = s.groupby("type").apply(lambda d: 100 * (d["rank"] <= 10).mean(), include_groups=False).to_numpy()
                y = -(k + 0.42 * j)
                ax.barh(y, mu, height=0.38 if len(items) > 1 else 0.6, color=COL[ck] if alpha == 1 else "white",
                        edgecolor=COL[ck], hatch=hatch, lw=lw(0.75))
                ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], fmt="none", ecolor="black", elinewidth=lw(0.5))
                ax.plot(per, np.full(len(per), y), "o", ms=1.6, mfc="white", mec="black", mew=lw(0.5), zorder=4)
            ticks.append(-(k + 0.21 * (len(items) - 1))); ticklabels.append(name)
            k += 1
        ax.set_yticks(ticks); ax.set_yticklabels(ticklabels)
        ax.set_xlim(0, 100)

    tsne, umap3, pac3 = "t-SNE 3D", "UMAP 3D (10, 0.1)", "PaCMAP 3D (15)"
    # A: a returning pitcher (refresh / call-up with an MLB row)
    axA = fig.add_subplot(gs[0, 0])
    grouped(axA, "refresh", [("Full space", "Full space", [("Full space", "as is", 1, None)]),
                             ("Movement plot", "Movement", [("Movement (ride x HB)", "as is", 1, None)]),
                             ("t-SNE 3D", "t-SNE 3D", [(tsne, "transform", 1, None), (tsne, "IDW 15, all rows", 0, None)]),
                             ("UMAP 3D", "UMAP 3D", [(umap3, "transform", 1, None), (umap3, "IDW 15, all rows", 0, None)]),
                             ("PaCMAP 3D", "PaCMAP 3D", [(pac3, "transform", 1, None), (pac3, "IDW 15, all rows", 0, None)])])
    axA.set_xlabel("Second half placed among the\n10 dots nearest his first half (%)")
    h = [plt.Rectangle((0, 0), 1, 1, fc="#737373", ec="#737373", lw=lw(0.75), label="Map's placement"),
         plt.Rectangle((0, 0), 1, 1, fc="white", ec="#737373", lw=lw(0.75), label="Neighbour average")]
    axA.legend(handles=h, loc="center right", bbox_to_anchor=(1.0, 0.73), handlelength=1.2, handletextpad=0.3)
    axA.set_title("A pitcher already on the map", fontsize=SZ.axis_ttl, pad=2)
    finish(axA, legend=True)
    # B: a stranger
    axB = fig.add_subplot(gs[0, 1])
    grouped(axB, "stranger", [("Full space", "Full space", [("Full space", "as is", 1, None)]),
                              ("Movement plot", "Movement", [("Movement (ride x HB)", "as is", 1, None)]),
                              ("t-SNE 3D", "t-SNE 3D", [(tsne, "transform", 1, None), (tsne, "IDW 15", 0, None)]),
                              ("UMAP 3D", "UMAP 3D", [(umap3, "transform", 1, None), (umap3, "IDW 15", 0, None)]),
                              ("PaCMAP 3D", "PaCMAP 3D", [(pac3, "transform", 1, None), (pac3, "IDW 15", 0, None)])])
    axB.set_xlabel("Second half placed among the\n10 dots nearest his first half (%)")
    axB.set_title("A pitcher the map has never seen", fontsize=SZ.axis_ttl, pad=2)
    finish(axB)
    # C: the built atlas, Triple-A call-ups, per pitch type
    axC = fig.add_subplot(gs[1, 0])
    c3 = cu[cu.dims == 3]
    ys = np.arange(len(TYPES))[::-1]
    for y, t_ in zip(ys, TYPES):
        a_ = 100 * c3[(c3.type == t_) & (c3.placement.str.startswith("Average"))].hit.mean()
        b_ = 100 * c3[(c3.type == t_) & (c3.placement.str.startswith("t-SNE"))].hit.mean()
        axC.plot([a_, b_], [y, y], color="#d9d9d9", lw=lw(1.0), zorder=1)
        axC.plot(a_, y, "o", ms=3.5, mfc="white", mec=COL["t-SNE 3D"], mew=lw(0.75), zorder=2)
        axC.plot(b_, y, "o", ms=3.5, mfc=COL["t-SNE 3D"], mec=COL["t-SNE 3D"], zorder=3)
    axC.set_yticks(ys); axC.set_yticklabels([PITCH_NAMES[t_] for t_ in TYPES])
    axC.set_xlim(0, 100)
    axC.set_xlabel("Triple-A call-ups among the 10 MLB dots\nnearest their own MLB row (%)")
    h = [plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc=COL["t-SNE 3D"], mec=COL["t-SNE 3D"], label="t-SNE placement"),
         plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", mec=COL["t-SNE 3D"], label="Neighbour average")]
    axC.legend(handles=h, loc="lower left", handletextpad=0.2)
    axC.set_title("The atlas: 3,352 call-ups, per-type 3D maps", fontsize=SZ.axis_ttl, pad=2)
    finish(axC, legend=True)
    # D: refreshing the map
    axD = fig.add_subplot(gs[1, 1])
    D2, Ds = D["D_refresh2"], D["D_stability"]
    series = [("t-SNE (perp 30)", "t-SNE 3D", ["refresh cold", "refresh warm 500 iters", "refresh warm 250 iters", "refresh warm 100 iters"],
               ["fresh", "500", "250 (built)", "100"], [(3, 3), (-2, 5), (-3, -9), (4, 0)]),
              ("UMAP (10, 0.1)", "UMAP 3D", ["refresh cold", "refresh aligned reg 0.1", "refresh aligned reg 1.0"], ["fresh", "aligned 0.1", "aligned 1.0"],
               [(3, 3), (3, 3), (0, -9)])]
    for m, ck, tests, labs, offs in series:
        pts = np.array([[wmean(D2[(D2.method == m) & (D2.dims == 3) & (D2.test == t_)], "overlap50"),
                         100 * wmean(D2[(D2.method == m) & (D2.dims == 3) & (D2.test == t_)], "recall10_free")] for t_ in tests])
        axD.plot(pts[:, 0], pts[:, 1], "-o", color=COL[ck], ms=2.5, lw=lw(0.75), label=ck.split(" ")[0])
        for (x, y), l, o in zip(pts, labs, offs):
            axD.annotate(l, (x, y), xytext=o, textcoords="offset points", fontsize=SZ.annot, color=COL[ck], ha="left" if o[0] > 0 else "right" if o[0] < 0 else "center")
    for m, ck, l in [("PaCMAP (15)", "PaCMAP 3D", "PaCMAP")]:
        d = Ds[(Ds.method == m) & (Ds.dims == 3) & (Ds.test == "refresh cold")]
        axD.plot(wmean(d, "overlap50"), 100 * wmean(d, "recall10_free_min"), "D", ms=3, color=COL[ck], label=l)
    axD.axvline(0.84, color="black", lw=lw(0.5), ls=":")
    axD.text(0.842, 36.2, "the measures'\nown change", fontsize=SZ.annot, va="bottom")
    axD.set_xlabel("Neighbourhoods (50) unchanged after a refresh")
    axD.set_ylabel("True neighbours kept after refit (%)")
    axD.set_xlim(0.5, 0.97); axD.set_ylim(35, 50)
    axD.legend(loc="lower left", handletextpad=0.3)
    axD.set_title("Refreshing the map (3D)", fontsize=SZ.axis_ttl, pad=2)
    finish(axD, legend=True)
    place_tags(fig, [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")])
    save(fig, str(OUT / "Figure3"))


# ------------------------------------------------------------------ Figure 4
FAMILIES = {"Fastballs (four-seam, sinker)": (["FF", "SI"], "#b2182b"), "Breaking (slider, sweeper, curve, cutter)": (["SL", "ST", "CU", "FC"], "#2166ac"),
            "Offspeed (changeup, splitter)": (["CH", "FS"], "#737373")}


def fig4(D, B, OUT):
    Lr, Cr = D["L_rows"], D["C_rows"]
    idn = Lr[Lr.space == "Identity space (15)"].dropna(subset=["d_height_in", "d_whiff_pts"])
    old = Lr[Lr.space == "Old atlas space (27)"].dropna(subset=["d_height_in", "d_whiff_pts"])
    fig = plt.figure(figsize=(PRINT_W, 128 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05])
    # A: binned, by family
    axA = fig.add_subplot(gs[0, 0:2])
    for name, (ts, col) in FAMILIES.items():
        d = idn[idn.type.isin(ts)]
        q = pd.qcut(d.d_height_in, 10, labels=False, duplicates="drop")
        m = d.groupby(q).agg(x=("d_height_in", "mean"), y=("d_whiff_pts", "mean"), s=("d_whiff_pts", "std"), n=("d_whiff_pts", "size"))
        axA.errorbar(m.x, m.y, yerr=1.96 * m.s / np.sqrt(m.n), fmt="-o", ms=2.5, lw=lw(0.75), elinewidth=lw(0.5), capsize=0, color=col, label=name)
    axA.axhline(0, color="black", lw=lw(0.5)); axA.axvline(0, color="black", lw=lw(0.5))
    axA.set_xlabel("Location height vs. his lookalikes (in; + higher)")
    axA.set_ylabel("Whiff rate vs. his lookalikes (points)")
    axA.legend(loc="upper right", handlelength=1.5, handletextpad=0.3)
    finish(axA, legend=True)
    # B: whiff points per inch, per pitch type: lookalikes on what the pitch is vs. the old space
    axB = fig.add_subplot(gs[0, 2])

    def slope(d):
        x, y = d.d_height_in.to_numpy(), d.d_whiff_pts.to_numpy()
        return np.polyfit(x, y, 1)[0]

    def slope_boot(d):
        i = np.array([B.idx[p] for p in d.pitcher])
        x, y = d.d_height_in.to_numpy(), d.d_whiff_pts.to_numpy()
        W = B.mult[:, i]
        n = W.sum(1)
        mx, my = (W @ x) / n, (W @ y) / n
        cov = (W @ (x * y)) / n - mx * my
        var = (W @ (x * x)) / n - mx ** 2
        return np.percentile(cov / var, [2.5, 97.5])
    ys = np.arange(len(TYPES))[::-1]
    for y, t_ in zip(ys, TYPES):
        for off, dd, filled in [(0.17, idn, True), (-0.17, old, False)]:
            d = dd[dd.type == t_]
            b, (lo, hi) = slope(d), slope_boot(d)
            col = PITCH_COLORS[t_]
            axB.plot([lo, hi], [y + off] * 2, color="black", lw=lw(0.5), zorder=2)
            axB.plot(b, y + off, "o", ms=3.2, mfc=col if filled else "white", mec=col, mew=lw(0.75), zorder=3)
    axB.axvline(0, color="black", lw=lw(0.5))
    axB.set_yticks(ys); axB.set_yticklabels([PITCH_NAMES[t_] for t_ in TYPES])
    axB.set_xlabel("Whiff points per inch higher")
    h = [plt.Line2D([], [], ls="", marker="o", ms=3.2, mfc="#4d4d4d", mec="#4d4d4d", label="Lookalikes on what the pitch is"),
         plt.Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec="#4d4d4d", label="Old: location in “similar”")]
    axB.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.2, borderaxespad=0.1)
    finish(axB, legend=True)
    # C, D: every four-seamer and every slider
    for k, (t_, letter, nm) in enumerate([("FF", "C", "Four-seam fastballs"), ("SL", "D", "Sliders")]):
        ax = fig.add_subplot(gs[1, k])
        d = idn[idn.type == t_]
        ax.scatter(d.d_height_in, d.d_whiff_pts, s=1.5, lw=0, alpha=0.5, color=PITCH_COLORS[t_], rasterized=True)
        b = np.polyfit(d.d_height_in, d.d_whiff_pts, 1)
        xs = np.linspace(d.d_height_in.quantile(0.01), d.d_height_in.quantile(0.99), 20)
        ax.plot(xs, np.polyval(b, xs), color="black", lw=lw(1.0))
        rho = spearmanr(d.d_height_in, d.d_whiff_pts).statistic
        ax.text(0.03, 0.97, f"{nm}, n = {len(d):,}\n{b[0]:+.2f} whiff points per inch\nSpearman {rho:+.2f}",
                transform=ax.transAxes, va="top", fontsize=SZ.annot, bbox=dict(fc="white", ec="none", pad=1.0))
        ax.axhline(0, color="black", lw=lw(0.5)); ax.axvline(0, color="black", lw=lw(0.5))
        ax.set_xlabel("Height vs. lookalikes (in)")
        ax.set_ylabel("Whiff vs. lookalikes (points)")
        ax.set_xlim(-8, 8); ax.set_ylim(-30, 40)
        finish(ax)
        if k == 0:
            axC = ax
        else:
            axD = ax
    # E: location carries outcome signal
    axE = fig.add_subplot(gs[1, 2])
    nt = Cr[(Cr.view == "Full space") & (Cr.k == 30) & (Cr.outcome == "whiff_pct")].groupby("type").size()
    vv = [("Full space", "What the pitch is", COL["Identity"], 1), ("Full space, + location", "+ location", COL["Identity"], 0),
          ("Full space, old 27 measures", "Old 27 measures", "#737373", 0)]
    xs = np.arange(len(vv))
    for i, oc in enumerate(["whiff_pct", "csw_pct"]):
        for j, (v, lab, col, filled) in enumerate(vv):
            sub = Cr[(Cr.view == v) & (Cr.k == 30) & (Cr.outcome == oc)]
            est, dist = 0.0, np.zeros(B.mult.shape[0])
            per = []
            for t_, g in sub.groupby("type"):
                ra, rp = g.actual.rank().to_numpy(), g.pred.rank().to_numpy()
                r_ = np.corrcoef(ra, rp)[0, 1]; per.append(r_)
                est += nt[t_] * r_; dist += nt[t_] * B.corr(g.pitcher, ra, rp)
            est, dist = est / nt.sum(), dist / nt.sum()
            lo, hi = np.percentile(dist, [2.5, 97.5])
            x = j + (i - 0.5) * 0.36
            axE.bar(x, est, width=0.34, color=col if (filled and i == 0) else ("white" if i == 1 else col),
                    edgecolor=col, hatch=None if i == 0 else "////", lw=lw(0.75), alpha=1.0 if filled or i == 1 else 0.55)
            axE.errorbar(x, est, yerr=[[est - lo], [hi - est]], fmt="none", ecolor="black", elinewidth=lw(0.5))
            axE.plot(np.full(len(per), x), per, "o", ms=1.6, mfc="white", mec="black", mew=lw(0.5), zorder=4)
    axE.set_xticks(xs); axE.set_xticklabels([l for _, l, _, _ in vv], rotation=20, ha="right")
    axE.set_ylabel("Outcome of the 30 nearest\nlookalikes vs. his own (Spearman)")
    axE.set_ylim(0, 0.55)
    h = [plt.Rectangle((0, 0), 1, 1, fc="#737373", ec="#737373", lw=lw(0.75), label="Whiff rate"),
         plt.Rectangle((0, 0), 1, 1, fc="white", ec="#737373", hatch="////", lw=lw(0.75), label="Called + swinging strikes")]
    axE.legend(handles=h, loc="upper right", handlelength=1.2, handletextpad=0.3)
    finish(axE, legend=True)
    place_tags(fig, [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"), (axE, "E")])
    save(fig, str(OUT / "Figure4"))


# ------------------------------------------------------------------ Supplementary
def figS1(D, B, OUT):
    """Every method: neighbours kept, local vs. global, co-ranking."""
    from umap_benchmark_figures import MAIN as ALLV, short
    A, Ar, E = D["A_views"], D["A_rows"], D["E_settings"]
    names = [v for v in ALLV if v in A.view.unique()] + ["Random 2D"]
    names = sorted(names, key=lambda v: -A[A.view == v].recall10_free.mean())
    fig = plt.figure(figsize=(PRINT_W, 120 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])

    def col(v):
        for k in ("t-SNE", "UMAP", "PaCMAP", "LocalMAP", "densMAP"):
            if v.startswith(k):
                base = "PaCMAP" if k in ("PaCMAP", "LocalMAP") else "UMAP" if k == "densMAP" else k
                return COL[base + (" 3D" if "3D" in v else " 2D")]
        return COL["Movement"] if v.startswith(("Movement", "Savant")) else "#e6e6e6" if v.startswith("Random") else COL["PCA"]
    ax = fig.add_subplot(gs[0, 0])
    means, los, his, pts = [], [], [], []
    for v in names:
        s = Ar[Ar.view == v]
        mu, lo, hi = _summ(B.mean(s.pitcher, 100 * s.recall10_free.to_numpy()))
        means.append(mu); los.append(lo); his.append(hi); pts.append(100 * A[A.view == v].recall10_free.to_numpy())
    ys = np.arange(len(names))[::-1]
    ci_bars(ax, ys, means, los, his, [col(v) for v in names], pts)
    ax.set_yticks(ys); ax.set_yticklabels([short(v) for v in names])
    ax.set_xlabel("True neighbours kept (%)")
    finish(ax)
    ax2 = fig.add_subplot(gs[0, 1])
    for v in names:
        s = A[A.view == v]
        ax2.plot(wmean(s, "dist_spearman"), 100 * wmean(s, "recall10_free"), "^" if "3D" in v else "o", ms=3, color=col(v), mec="black", mew=lw(0.5))
    for meth, mk in [("UMAP", "UMAP 2D"), ("PaCMAP", "PaCMAP 2D"), ("t-SNE", "t-SNE 2D")]:
        s = E[(E.method == meth) & (E.dims == 2)]
        g = s.groupby(["a", "b"]).apply(lambda d: pd.Series({"x": wmean(d, "dist_spearman"), "y": 100 * wmean(d, "recall10_free")}), include_groups=False)
        ax2.plot(g.x, g.y, "o", ms=1.6, color=COL[mk], zorder=0)
    for v, off in [("t-SNE 3D", (3, 0)), ("t-SNE 2D", (3, -5)), ("UMAP 3D (10, 0.1)", (0, 5)), ("UMAP 2D (10, 0.1)", (-3, -6)), ("densMAP 3D (15, 0.3)", (-3, 3)),
                   ("PCA 3", (-3, 3)), ("PCA 2", (-3, 3)), ("Movement (ride x HB)", (3, -3))]:
        s = A[A.view == v]
        ax2.annotate(short(v), (wmean(s, "dist_spearman"), 100 * wmean(s, "recall10_free")), xytext=off, textcoords="offset points",
                     ha="left" if off[0] > 0 else "right" if off[0] < 0 else "center", fontsize=SZ.annot)
    ax2.set_xlabel("Distance rank-correlation (global)")
    ax2.set_ylabel("True neighbours kept (%; local)")
    finish(ax2)
    ax3 = fig.add_subplot(gs[0, 2])
    cor = D["A_coranking"]
    cc = cor[cor.type == "FF"]
    for v, ck in [("PCA 2", "PCA"), ("UMAP 2D (10, 0.1)", "UMAP 2D"), ("t-SNE 2D", "t-SNE 2D"), ("UMAP 3D (10, 0.1)", "UMAP 3D"), ("t-SNE 3D", "t-SNE 3D")]:
        s = cc[cc.view == v]
        ax3.plot(s.K, s.qnx, lw=lw(0.75), color=COL[ck], ls="--" if "3D" in v else "-", label=short(v))
    nK = int(cc.K.max()) + 1
    ax3.plot(np.arange(1, nK), np.arange(1, nK) / (nK - 1), color="black", lw=lw(0.5), ls=":", label="Random")
    ax3.set_xscale("log"); ax3.set_xlabel("Neighbourhood size K"); ax3.set_ylabel("K-neighbourhoods kept, four-seamers")
    ax3.legend(loc="upper left", handlelength=1.6)
    finish(ax3, legend=True)
    place_tags(fig, [(ax, "A"), (ax2, "B"), (ax3, "C")])
    save(fig, str(OUT / "FigureS1"))


def figS2(D, B, OUT):
    """Shepard diagrams, four-seamers."""
    c = D["A_coords"]; c = c[c.pitch_type == "FF"]
    pc = [x for x in c.columns if x[0] == "p" and x[1:].isdigit()]
    Pf = c[pc].to_numpy()
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(c), 20000), rng.integers(0, len(c), 20000)
    m_ = i != j
    dh = np.linalg.norm(Pf[i[m_]] - Pf[j[m_]], axis=1)
    sh = [("PCA 2", "PCA", "PCA"), ("PaCMAP 2D (15)", "PaCMAP", "PaCMAP 2D"), ("UMAP 2D (10, 0.1)", "UMAP", "UMAP 2D"), ("t-SNE 2D", "t-SNE", "t-SNE 2D")]
    fig, axes = plt.subplots(1, 4, figsize=(PRINT_W, 50 * MM), layout="constrained")
    for ax, (v, name, ck), L in zip(axes, sh, "ABCD"):
        V = c[[f"{v}|x", f"{v}|y"]].to_numpy()
        dl = np.linalg.norm(V[i[m_]] - V[j[m_]], axis=1)
        ax.scatter(dh, dl, s=0.4, alpha=0.2, color=COL[ck] if ck != "PCA" else "#737373", lw=0, rasterized=True)
        ax.text(0.03, 0.97, f"{name}\nSpearman {spearmanr(dh, dl).statistic:.2f}", transform=ax.transAxes, va="top", fontsize=SZ.annot)
        ax.set_xlabel("Distance, 15 measures"); ax.set_yticks([])
        finish(ax)
    axes[0].set_ylabel("Distance on the map")
    place_tags(fig, list(zip(axes, "ABCD")))
    save(fig, str(OUT / "FigureS2"))


def figS3(D, B, OUT):
    """Settings."""
    E = D["E_settings"]
    fig = plt.figure(figsize=(PRINT_W, 95 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 0.95, 1])

    def heat(ax, s, col, scale, fmt, title, vmin, vmax, xl, yl):
        g = s.groupby(["a", "b"]).apply(lambda d: wmean(d, col), include_groups=False).unstack() * scale
        ax.imshow(g.to_numpy(), cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(g.shape[1])); ax.set_xticklabels([f"{c:g}" for c in g.columns]); ax.set_yticks(range(g.shape[0])); ax.set_yticklabels(g.index)
        for a in range(g.shape[0]):
            for b_ in range(g.shape[1]):
                v = g.iloc[a, b_]
                ax.text(b_, a, fmt.format(v), ha="center", va="center", fontsize=SZ.annot, color="white" if (v - vmin) / (vmax - vmin) > 0.6 else "black")
        ax.set_title(title, fontsize=SZ.axis_ttl, pad=2); ax.set_xlabel(xl); ax.set_ylabel(yl)
        finish(ax)
    a1 = fig.add_subplot(gs[0, 0]); heat(a1, E[(E.method == "UMAP") & (E.dims == 2)], "recall10_free", 100, "{:.0f}", "UMAP 2D", 20, 50, "min_dist", "n_neighbors")
    a2 = fig.add_subplot(gs[0, 1]); heat(a2, E[(E.method == "UMAP") & (E.dims == 3)], "recall10_free", 100, "{:.0f}", "UMAP 3D", 20, 50, "min_dist", "n_neighbors")
    a3 = fig.add_subplot(gs[0, 2]); heat(a3, E[(E.method == "t-SNE") & (E.dims == 2)], "recall10_free", 100, "{:.0f}", "t-SNE 2D", 20, 50, "exaggeration", "perplexity")
    a4 = fig.add_subplot(gs[0, 3])
    for m, ck in [("UMAP (10, 0.1) dims", "UMAP 3D"), ("UMAP (15, 0.3) dims", "UMAP 2D")]:
        s = E[E.method == m]
        g = s.groupby("dims").apply(lambda d: 100 * wmean(d, "recall10_free"), include_groups=False)
        a4.plot(g.index, g.values, "-o", ms=2.5, lw=lw(0.75), color=COL[ck], label=m.replace(" dims", ""))
        for _, d in s.groupby("type"):
            a4.plot(d.dims, 100 * d.recall10_free, "o", ms=1.4, mfc="white", mec=COL[ck], mew=lw(0.5))
    a4.set_xticks([2, 3, 4, 5]); a4.set_xlabel("Output dimensions"); a4.set_ylabel("True neighbours kept (%)")
    a4.legend(loc="lower right")
    finish(a4, legend=True)
    place_tags(fig, [(a1, "A"), (a2, "B"), (a3, "C"), (a4, "D")])
    save(fig, str(OUT / "FigureS3"))


def figS4(D, B, OUT):
    """Seed stability and the 2D refresh."""
    Ds, D2 = D["D_stability"], D["D_refresh2"]
    fig = plt.figure(figsize=(PRINT_W, 70 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 3)
    meth = [("UMAP (10, 0.1)", "UMAP"), ("t-SNE (perp 30)", "t-SNE"), ("PaCMAP (15)", "PaCMAP")]
    axs = []
    for k, (col, ylab) in enumerate([("overlap50", "Neighbourhoods (50) shared\nbetween seeds"), ("procrustes", "Layout change between seeds\n(Procrustes disparity)")]):
        ax = fig.add_subplot(gs[0, k])
        for i, (m, base) in enumerate(meth):
            for j, d in enumerate([2, 3]):
                s = Ds[(Ds.test == "seeds") & (Ds.method == m) & (Ds.dims == d)]
                x = i + (j - 0.5) * 0.38
                ax.bar(x, wmean(s, col), width=0.36, color=COL[f"{base} {d}D"])
                ax.plot(np.full(len(s), x), s[col], "o", ms=1.6, mfc="white", mec="black", mew=lw(0.5))
        ax.set_xticks(range(3)); ax.set_xticklabels([b for _, b in meth]); ax.set_ylabel(ylab)
        if k == 1:  # one key for both bar panels (same colours)
            h = [plt.Rectangle((0, 0), 1, 1, color="#b3b3b3", label="2D"), plt.Rectangle((0, 0), 1, 1, color="#333333", label="3D")]
            ax.legend(handles=h, loc="upper right")
        finish(ax, legend=True)
        axs.append(ax)
    ax = fig.add_subplot(gs[0, 2])
    for m, ck, tests, labs in [("t-SNE (perp 30)", "t-SNE 2D", ["refresh cold", "refresh warm 500 iters", "refresh warm 250 iters", "refresh warm 100 iters"], ["fresh", "500", "250", "100"]),
                               ("UMAP (10, 0.1)", "UMAP 2D", ["refresh cold", "refresh aligned reg 0.1", "refresh aligned reg 1.0"], ["fresh", "aligned 0.1", "aligned 1.0"])]:
        pts = np.array([[wmean(D2[(D2.method == m) & (D2.dims == 2) & (D2.test == t_)], "overlap50"),
                         100 * wmean(D2[(D2.method == m) & (D2.dims == 2) & (D2.test == t_)], "recall10_free")] for t_ in tests])
        dark = "t-SNE 3D" if ck.startswith("t-SNE") else "UMAP 3D"
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=COL[dark], ms=2.5, lw=lw(0.75), label=ck.split(" ")[0])
        for (x, y), l in zip(pts, labs):
            left = l == "aligned 0.1"  # right of the point it would sit on the reference line
            ax.annotate(l, (x, y), xytext=(-4, -1) if left else (3, 3), textcoords="offset points", fontsize=SZ.annot,
                        color=COL[dark], ha="right" if left else "left", va="center" if left else "bottom")
    ax.set_xlim(0.38, 1.02); ax.margins(y=0.1)
    ax.axvline(0.84, color="black", lw=lw(0.5), ls=":")
    ax.set_xlabel("Neighbourhoods (50) unchanged after a refresh"); ax.set_ylabel("True neighbours kept after refit (%)")
    ax.set_title("Refreshing the map (2D)", fontsize=SZ.axis_ttl, pad=2)
    ax.legend(loc="lower left")
    finish(ax, legend=True)
    place_tags(fig, [(axs[0], "A"), (axs[1], "B"), (ax, "C")])
    save(fig, str(OUT / "FigureS4"))


def figS5(D, B, OUT):
    """Outcome signal in neighbourhoods, every picture."""
    Cr = D["C_rows"]
    nt = Cr[(Cr.view == "Full space") & (Cr.k == 30) & (Cr.outcome == "whiff_pct")].groupby("type").size()

    def spear(sub):
        est, dist = 0.0, np.zeros(B.mult.shape[0])
        for t_, g in sub.groupby("type"):
            ra, rp = g.actual.rank().to_numpy(), g.pred.rank().to_numpy()
            est += nt[t_] * np.corrcoef(ra, rp)[0, 1]; dist += nt[t_] * B.corr(g.pitcher, ra, rp)
        return est / nt.sum(), np.percentile(dist / nt.sum(), [2.5, 97.5])
    views = [("Full space", "Full space", "Full space"), ("Savant 5 space (5-D)", "Savant-5 space", "Movement"), MAIN[0], MAIN[1], MAIN[2], MAIN[3], MAIN[4], MAIN[5], MAIN[6], MAIN[7]]
    fig = plt.figure(figsize=(PRINT_W, 80 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0, 0])
    ocs = [("whiff_pct", "Whiff"), ("csw_pct", "CSW"), ("xwoba_contact", "xwOBA on contact"), ("rv100_raw", "Run value")]
    wdt = 0.8 / len(ocs)
    for i, (oc, lab) in enumerate(ocs):
        ests = [spear(Cr[(Cr.view == v) & (Cr.k == 30) & (Cr.outcome == oc)]) for v, _, _ in views]
        xs = np.arange(len(views)) + i * wdt - 0.4 + wdt / 2
        ax.bar(xs, [e[0] for e in ests], width=wdt, color=["#333333", "#737373", "#b3b3b3", "#e6e6e6"][i], label=lab)
        ax.errorbar(xs, [e[0] for e in ests], yerr=[[e[0] - e[1][0] for e in ests], [e[1][1] - e[0] for e in ests]], fmt="none", ecolor="black", elinewidth=lw(0.5))
    ax.axhline(0, color="black", lw=lw(0.5))
    ax.set_xticks(range(len(views))); ax.set_xticklabels([n + (" 3D" if "3D" in v else " 2D" if c not in ("Movement", "PCA", "Full space") else "") for v, n, c in views],
                                                       rotation=30, ha="right")
    ax.set_ylabel("Outcome of the 30 nearest vs. his own (Spearman)")
    ax.legend(loc="upper right", ncol=2)
    finish(ax, legend=True)
    ax2 = fig.add_subplot(gs[0, 1])
    for v, name, ck in [("Full space", "Full space", "Full space"), MAIN[0], MAIN[1], MAIN[4], MAIN[7], MAIN[6]]:
        ys = [spear(Cr[(Cr.view == v) & (Cr.k == k) & (Cr.outcome == "whiff_pct")])[0] for k in [10, 30, 100]]
        ax2.plot([10, 30, 100], ys, "-o", ms=2.5, lw=lw(0.75), color=COL[ck], ls="--" if "3D" in v else "-",
                 label=name + (" 3D" if "3D" in v else " 2D" if ck not in ("Movement", "PCA", "Full space") else ""))
    ax2.set_xscale("log"); ax2.set_xticks([10, 30, 100]); ax2.set_xticklabels(["10", "30", "100"])
    ax2.set_xlabel("Lookalikes averaged (k)"); ax2.set_ylabel("Whiff rate (Spearman)")
    ax2.legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)
    ax2.set_ylim(0.07, 0.30)
    finish(ax2, legend=True)
    place_tags(fig, [(ax, "A"), (ax2, "B")])
    save(fig, str(OUT / "FigureS5"))


def figS6(D, B, OUT):
    """The all-pitches map."""
    c, G, Gr = D["G_coords"], D["G_views"], D["G_rows"]
    maps = [("PCA 2", "PCA"), ("PaCMAP 2D (15)", "PaCMAP"), ("UMAP 2D (10, 0.1)", "UMAP"), ("t-SNE 2D", "t-SNE (the atlas)")]
    fig = plt.figure(figsize=(PRINT_W, 66 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.25])
    axs = []
    for j, (m, title) in enumerate(maps):
        ax = fig.add_subplot(gs[0, j])
        for pt, g in c.groupby("pitch_type"):
            ax.scatter(g[f"{m}|x"], g[f"{m}|y"], s=0.6, lw=0, alpha=0.7, color=PITCH_COLORS.get(pt, "k"), label=PITCH_NAMES.get(pt, pt), rasterized=True)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        ax.set_title(title, fontsize=SZ.axis_ttl, pad=2)
        finish(ax)
        axs.append(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=8, markerscale=5, handletextpad=0.1, columnspacing=0.8)
    ax = fig.add_subplot(gs[0, 4])
    rows = [("Movement (ride x HB)", "Movement plot", "Movement"), ("PCA 2", "PCA", "PCA"), ("PaCMAP 2D (15)", "PaCMAP 2D", "PaCMAP 2D"),
            ("UMAP 2D (10, 0.1)", "UMAP 2D", "UMAP 2D"), ("t-SNE 2D", "t-SNE 2D", "t-SNE 2D"), ("PaCMAP 3D (15)", "PaCMAP 3D", "PaCMAP 3D"),
            ("UMAP 3D (10, 0.1)", "UMAP 3D", "UMAP 3D"), ("t-SNE 3D", "t-SNE 3D", "t-SNE 3D")]
    means, los, his = [], [], []
    for v, _, _ in rows:
        s = Gr[Gr.view == v]
        mu, lo, hi = _summ(B.mean(s.pitcher, 100 * s.recall10_free.to_numpy()))
        means.append(mu); los.append(lo); his.append(hi)
    ys = np.arange(len(rows))[::-1]
    ci_bars(ax, ys, means, los, his, [COL[ck] for _, _, ck in rows])
    ax.set_yticks(ys); ax.set_yticklabels([n for _, n, _ in rows]); ax.set_xlabel("True neighbours kept (%)")
    finish(ax)
    place_tags(fig, [(axs[0], "A"), (ax, "B")])
    save(fig, str(OUT / "FigureS6"))


def figS7(D, B, OUT):
    """Correlations among the 27 measures of the old space, four-seamers."""
    from mlbpitch import profiles
    feats = [f for b in ("shape", "derived", "delivery", "intent") for f in profiles.MEAN_FEATURES[b]] + list(profiles.SD_FEATURES)
    full = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False))
    Cm = full[full.pitch_type == "FF"][feats].astype("float64").corr()
    fig, ax = plt.subplots(figsize=(120 * MM, 112 * MM), layout="constrained")
    im = ax.imshow(Cm.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90); ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Correlation, four-seam fastballs")
    cb.outline.set_linewidth(lw(0.5))
    finish(ax)
    save(fig, str(OUT / "FigureS7"))


def make_all(D, B, OUT):
    setup()
    for f in (fig1, fig2, fig3, fig4, figS1, figS2, figS3, figS4, figS5, figS6, figS7):
        f(D, B, OUT)
        plt.close("all")
