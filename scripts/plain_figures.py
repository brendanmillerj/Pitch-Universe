"""Plain-units versions of the trajectory and comps results.

The analysis figures (trajectories.pdf, comps.pdf) live in embedding space --
UMAP arrows, PCA loadings, partial residuals. These redraw the same findings in
baseball units: movement charts in inches, changes in mph / degrees, whiff %.

  analysis/fig_change.pdf   how pitchers change between halves
  analysis/fig_comps.pdf    who looks like whom, and whether that predicts

Movement is in each pitcher's handedness-neutral frame (lefties mirrored), shown
as arm-side break (+ = toward the pitcher's arm side) vs induced vertical break.

Usage:  python scripts/plain_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import (PITCH_COLORS, PITCH_NAMES, PRINT_W, apply_style, cell_lw,
                          cell_tag, save, style_axes)
from mlbpitch import paths, profiles

META = paths.META
FIG_DIR = paths.ROOT / "analysis"
HALF_LABEL = ["2024 H1", "2024 H2", "2025 H1", "2025 H2", "2026 H1", "2026 H2"]

# (pitcher, the change the panel is about, a pitch relabel to draw as one path).
# Picked from the data among pitchers present in all six halves: largest
# arm-angle drop, largest usage-based "addition", largest four-seam ride gain.
# Anderson's "added sweeper" is his slider reshaped AND relabeled by Statcast
# (SL 35-45% in 2024 -> ST 37-39% from 2025), so the panel says so.
EXAMPLES = [("Hancock, Emerson", "Arm angle 27° → 11°; fastballs lost ride", None),
            ("Anderson, Grant", "Slider reshaped into a sweeper", ("SL", "ST")),
            ("Fried, Max", "Four-seam gained 5 in of ride", None)]
COMP_EXAMPLE = ("Gausman, Kevin", "FF", 2026, 2)
LATEST = (2026, 2)


def first_last(name):
    """'Last, First' -> 'First Last'."""
    last, _, first = name.partition(", ")
    return f"{first} {last}".strip()


def movement_panel(ax, types, name, subtitle, lims, link=None):
    d = types[types.player_name == name].copy()
    d["t"] = (d.season - 2024) * 2 + d.half - 1
    d["hb"], d["ivb"] = -12 * d.pfx_x, 12 * d.pfx_z
    ends, labels, points = {}, [], []
    for pt, g in d.groupby("pitch_type"):
        g = g.sort_values("t")
        c = PITCH_COLORS[pt]
        ends[pt] = g
        ax.plot(g.hb, g.ivb, "-", color=c, lw=cell_lw(0.75), alpha=0.6)
        ax.scatter(g.hb.iloc[:-1], g.ivb.iloc[:-1], s=8, facecolor="white",
                   edgecolor=c, lw=cell_lw(0.5), zorder=3)
        ax.scatter(g.hb.iloc[-1], g.ivb.iloc[-1], s=22, color=c, zorder=4)
        u0, u1 = g.usage.iloc[0], g.usage.iloc[-1]
        use = f"{u1:.0%}" if len(g) == 1 else f"{u0:.0%}→{u1:.0%}"
        labels.append(ax.text(g.hb.iloc[-1] + 1, g.ivb.iloc[-1] + 0.8,
                              f"{PITCH_NAMES[pt]} {use}", fontsize=6, color=c))
        points.append((g.hb.iloc[-1], g.ivb.iloc[-1]))
    if link and all(p in ends for p in link):
        a, b = ends[link[0]].iloc[-1], ends[link[1]].iloc[0]
        ax.annotate("", (b.hb, b.ivb), (a.hb, a.ivb),
                    arrowprops=dict(arrowstyle="->", ls="--", color="0.4", lw=cell_lw(0.75)))
        ax.text((a.hb + b.hb) / 2, (a.ivb + b.ivb) / 2 - 2.2, "relabeled\n2024 → 2025",
                fontsize=6, color="0.4", ha="center", va="top")
    ax.axhline(0, color="0.8", lw=cell_lw(0.5), zorder=0)
    ax.axvline(0, color="0.8", lw=cell_lw(0.5), zorder=0)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    from adjustText import adjust_text
    px, py = zip(*points)
    adjust_text(labels, x=list(px), y=list(py), ax=ax, expand=(1.1, 1.3),
                arrowprops=dict(arrowstyle="-", color="0.6", lw=cell_lw(0.5)))
    ax.set_xlabel("Arm-side break (in)")
    ax.set_title(f"{first_last(name)}\n{subtitle}", loc="left")
    style_axes(ax)


def change_hist(ax, tr, col, scale, unit, label):
    off = tr.half == 2           # H2 -> next H1 crosses an offseason
    vals = {"Within a season": scale * tr.loc[~off, col].dropna(),
            "Across an offseason": scale * tr.loc[off, col].dropna()}
    lo, hi = np.percentile(np.r_[vals["Within a season"], vals["Across an offseason"]], [1, 99])
    bins = np.linspace(lo, hi, 36)
    # Shares, not counts: there are ~2x as many within-season transitions.
    rng = {k: np.percentile(v, [10, 90]) for k, v in vals.items()}
    lab = {k: f"{k}\n  80% between {a:+.1f} and {b:+.1f} {unit}" for k, (a, b) in rng.items()}
    w = lambda v: np.full(len(v), 100 / len(v))
    ax.hist(vals["Within a season"], bins=bins, weights=w(vals["Within a season"]),
            color="0.78", label=lab["Within a season"])
    ax.hist(vals["Across an offseason"], bins=bins, weights=w(vals["Across an offseason"]),
            histtype="step", color="#4575b4", lw=cell_lw(1), label=lab["Across an offseason"])
    ax.axvline(0, color="black", lw=cell_lw(0.5))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
    ax.legend(loc="upper left", fontsize=6, handlelength=1.2,
              frameon=True, framealpha=1, edgecolor="none")
    ax.set_xlabel(label)
    ax.set_ylabel("Share of pitchers (%)")
    style_axes(ax, legend=True)


def fig_change(types, tr):
    fig = plt.figure(figsize=(PRINT_W, 5.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ex = types[types.player_name.isin([e[0] for e in EXAMPLES])]
    hb, ivb = -12 * ex.pfx_x, 12 * ex.pfx_z
    lims = ((hb.min() - 3, hb.max() + 6), (ivb.min() - 2, ivb.max() + 3))
    for j, (name, sub, link) in enumerate(EXAMPLES):
        ax = fig.add_subplot(gs[0, j])
        movement_panel(ax, types, name, sub, lims, link)
        if j == 0:
            ax.set_ylabel("Induced vertical break (in)")
        cell_tag(ax, "ABC"[j])
    # legend for the open/filled markers
    ax0 = fig.axes[0]
    ax0.scatter([], [], s=8, facecolor="white", edgecolor="0.3", label="Earlier halves")
    ax0.scatter([], [], s=22, color="0.3", label="Latest half (2026 H2)")
    ax0.legend(loc="lower left", fontsize=6, handletextpad=0.2)

    for j, (col, scale, unit, label) in enumerate([
            ("d_fb_release_speed", 1, "mph", "Change in fastball velocity (mph)"),
            ("d_arm_angle", 1, "°", "Change in arm angle (°)"),
            ("d_fb_pfx_z", 12, "in", "Change in fastball ride (in)")]):
        ax = fig.add_subplot(gs[1, j])
        change_hist(ax, tr, col, scale, unit, label)
        cell_tag(ax, "DEF"[j])
    save(fig, FIG_DIR / "fig_change")


def fingerprint(ax, types, comps):
    name, pt, season, half = COMP_EXAMPLE
    pool = types[types.pitch_type == pt].copy()
    pool["hb"], pool["ivb"] = -12 * pool.pfx_x, 12 * pool.pfx_z
    pool["in_zone"] = 100 * pool.in_zone
    pool["usage"] = 100 * pool.usage
    # One or two rows from each block the comps are matched on.
    feats = [("release_speed", "Velocity", "mph"), ("ivb", "Ride (iVB)", "in"),
             ("hb", "Arm-side run", "in"), ("vaa_above_avg", "VAA vs. expected", "°"),
             ("arm_angle", "Arm angle", "°"), ("release_extension", "Extension", "ft"),
             ("in_zone", "In zone", "%"), ("usage", "Usage", "%")]
    for c, _, _ in feats:
        pool[f"{c}_pct"] = 100 * pool[c].rank(pct=True)
    key = profiles.KEYS + ["pitch_type"]
    me = pool[(pool.player_name == name) & (pool.season == season) & (pool.half == half)].iloc[0]
    sel = comps[(comps.player_name == name) & (comps.season == season)
                & (comps.half == half) & (comps.pitch_type == pt)]
    cm = pool.merge(sel[["comp_pitcher", "comp_season", "comp_half"]].rename(
        columns={"comp_pitcher": "pitcher", "comp_season": "season", "comp_half": "half"}),
        on=["pitcher", "season", "half"])
    y = np.arange(len(feats))[::-1]
    for yi, (c, lab, unit) in zip(y, feats):
        ax.plot([0, 100], [yi, yi], color="0.88", lw=cell_lw(1.5), zorder=0)
        ax.scatter(cm[f"{c}_pct"], np.full(len(cm), yi), s=10, color="#74add1",
                   edgecolor="none", zorder=2)
        ax.scatter(me[f"{c}_pct"], yi, s=30, color="#d73027", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab} ({round(me[c], 1) + 0.0:.1f} {u})" for c, lab, u in feats])
    ax.set_xlim(-2, 102)
    ax.set_xlabel(f"Percentile among all {PITCH_NAMES[pt].lower()}s")
    ax.scatter([], [], s=30, color="#d73027", label=f"{first_last(name)} ({season} H{half})")
    ax.scatter([], [], s=10, color="#74add1", label="His 10 closest comps")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=6)
    ax.text(0.5, -0.24, "Comps are matched on 32 measurements at once, so no single row lines up exactly",
            transform=ax.transAxes, ha="center", va="top", fontsize=6, color="0.35")
    ax.set_title(f"{first_last(name).split()[-1]}'s four-seam vs. its 10 closest comps\n"
                 f"Comps whiffed {100 * sel.comp_whiff_pct.mean():.0f}%; "
                 f"his whiffed {100 * me.whiff_pct:.0f}%", loc="left")
    style_axes(ax, legend=True)


def predict_panel(ax, uq):
    m = uq.assign(t=(uq.season - 2024) * 2 + uq.half - 1)
    key = ["pitcher", "pitch_type"]
    pr = m.merge(m[key + ["t", "whiff_pct"]].assign(t=lambda d: d.t - 1),
                 on=key + ["t"], suffixes=("", "_next")).dropna(subset=["whiff_pct", "whiff_pct_next",
                                                                     "exp_whiff_pct"])
    x, y, own = 100 * pr.exp_whiff_pct, 100 * pr.whiff_pct_next, 100 * pr.whiff_pct
    ax.scatter(x, y, s=1.5, color="0.7", edgecolor="none", rasterized=True)
    bins = pd.qcut(x, 10, labels=False)
    b = pd.DataFrame({"x": x, "y": y, "bin": bins}).groupby("bin").agg(
        x=("x", "mean"), y=("y", "mean"), se=("y", lambda s: s.std() / np.sqrt(len(s))))
    ax.errorbar(b.x, b.y, yerr=1.96 * b.se, fmt="o-", ms=3, color="#4575b4",
                lw=cell_lw(1), capsize=1.5, label="Mean of each tenth (95% CI)")
    lim = (0, np.percentile(np.r_[x, y], 99.5))
    ax.plot(lim, lim, "--", color="black", lw=cell_lw(0.5), label="Perfect prediction")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    r2 = lambda a: np.corrcoef(a, y)[0, 1] ** 2
    X = np.column_stack([np.ones(len(x)), x, own])
    both = 1 - ((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ax.text(0.03, 0.97,
            f"Share of next-half variation explained by\n"
            f"  its comps this half: {100 * r2(x):.0f}%\n"
            f"  its own whiff % this half: {100 * r2(own):.0f}%\n"
            f"  both together: {100 * both:.0f}%",
            transform=ax.transAxes, va="top", fontsize=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xlabel("What similar pitches whiffed this half (%)")
    ax.set_ylabel("What this pitch whiffed next half (%)")
    ax.set_title(f"Comps predict the next half\n({len(pr):,} pitch-type pairs, 2024–2026)", loc="left")
    ax.legend(loc="lower right", fontsize=6)
    style_axes(ax)
    return pr


def buylow_panel(ax, uq, pr):
    # Historical hit rate for the same rule, so the list comes with a track record.
    # The fair baseline is NOT all pitches: a pitch whiffing far below normal tends
    # to bounce back on its own. Compare against pitches of the same type with an
    # equally low whiff rate (same within-type decile) but ordinary comps.
    pr = pr.copy()
    gap = pr.exp_whiff_pct - pr.whiff_pct
    top = (gap >= gap.quantile(0.9)).to_numpy()
    up = (pr.whiff_pct_next > pr.whiff_pct).to_numpy()
    pr["dec"] = pr.groupby("pitch_type").whiff_pct.transform(
        lambda s: np.ceil(10 * s.rank(pct=True)))
    rate = pr[~top].assign(up=up[~top]).groupby(["pitch_type", "dec"]).up.mean()
    matched = rate.reindex(pd.MultiIndex.from_frame(pr.loc[top, ["pitch_type", "dec"]])).mean()
    up_top = up[top].mean()

    d = uq[(uq.season == LATEST[0]) & (uq.half == LATEST[1]) & (uq.n >= 150)].copy()
    d["gap"] = d.exp_whiff_pct - d.whiff_pct
    d = d.sort_values("gap", ascending=False).head(10)[::-1]
    y = np.arange(len(d))
    ax.hlines(y, 100 * d.whiff_pct, 100 * d.exp_whiff_pct, color="0.7", lw=cell_lw(1))
    ax.scatter(100 * d.whiff_pct, y, s=18, facecolor="white", edgecolor="#d73027",
               lw=cell_lw(0.75), zorder=3, label="Actual")
    ax.scatter(100 * d.exp_whiff_pct, y, s=18, color="#4575b4", zorder=3,
               label="What its comps got")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{first_last(n)} {PITCH_NAMES[p].lower()}"
                        for n, p in zip(d.player_name, d.pitch_type)])
    ax.set_xlabel("Whiff % (2026 second half)")
    ax.set_title("Buy-low: whiffing far below their comps\n"
                 f"Past cases: {100 * up_top:.0f}% improved next half\n"
                 f"(vs {100 * matched:.0f}% of equally low-whiff pitches)", loc="left")
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=6)
    style_axes(ax, legend=True)


def rarity_panel(ax, uq):
    d = uq.dropna(subset=["whiff_pct"]).copy()
    d["u_pct"] = d.groupby("pitch_type").uniqueness.rank(pct=True)
    d["decile"] = np.ceil(10 * d.u_pct).clip(1, 10)
    jit = np.random.default_rng(0).uniform(-0.3, 0.3, len(d))
    ax.scatter(d.decile + jit, 100 * d.whiff_pct, s=1, color="0.75", edgecolor="none",
               rasterized=True)
    b = d.groupby("decile").whiff_pct.agg(["mean", "std", "size"])
    ax.errorbar(b.index, 100 * b["mean"], yerr=196 * b["std"] / np.sqrt(b["size"]),
                fmt="o-", ms=3, color="#d73027", lw=cell_lw(1), capsize=1.5)
    ax.set_ylim(0, np.percentile(100 * d.whiff_pct, 99))
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Unusualness in its type (1 typical → 10 rare)")
    ax.set_ylabel("Whiff % (red: mean ± 95% CI)")
    ax.set_title(f"Unusual pitches whiff a little more\n"
                 f"{100 * b['mean'].iloc[0]:.0f}% (typical) → "
                 f"{100 * b['mean'].iloc[-1]:.0f}% (rare)", loc="left")
    style_axes(ax)


def fig_comps(types, comps, uq):
    fig = plt.figure(figsize=(PRINT_W, 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0]); fingerprint(ax, types, comps); cell_tag(ax, "A", x=-0.4)
    ax = fig.add_subplot(gs[0, 1]); pr = predict_panel(ax, uq); cell_tag(ax, "B")
    ax = fig.add_subplot(gs[1, 0]); buylow_panel(ax, uq, pr); cell_tag(ax, "C", x=-0.4)
    ax = fig.add_subplot(gs[1, 1]); rarity_panel(ax, uq); cell_tag(ax, "D")
    save(fig, FIG_DIR / "fig_comps")


if __name__ == "__main__":
    apply_style()
    types = pd.read_parquet(profiles.profiles_path("type"))
    tr = pd.read_parquet(META / "trajectories_2024_2026.parquet")
    comps = pd.read_parquet(META / "comps_type_2024_2026.parquet")
    uq = pd.read_parquet(META / "uniqueness_type_2024_2026.parquet")
    fig_change(types, tr)
    fig_comps(types, comps, uq)
