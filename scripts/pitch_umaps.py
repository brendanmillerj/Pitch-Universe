"""Per-pitch-type landscapes and per-pitcher comp cards, on as-of snapshots.

Rows are filtered state-space estimates (mlbpitch.timeseries): one pitcher's
version of one pitch type as of one date (15th of Apr-Sep + season end,
2024-2026), using only games up to that date. Comps come from
scripts/snapshot_comps.py, STRICT set: a comp is never dated after the card's
date, so a backdated card (--asof) shows only what was knowable then.

The UMAP is a LANDSCAPE only -- what is typical vs rare, where outcomes run
high. It is not used to judge who resembles whom: on these continuous data a 2D
map keeps only ~20-50% of each row's true nearest neighbours. Comps are shown
in real units instead.

  analysis/pitch_atlas_by_type.pdf     8 landscapes, coloured by outcome percentile
  analysis/cards/<Last_First>.pdf      one pitcher on one date, one row per pitch:
                                       landscape | fingerprint vs comps | comps' outcome

Usage:
  python scripts/pitch_umaps.py                         # atlas + example cards
  python scripts/pitch_umaps.py "Gausman, Kevin" 676106  # names or MLBAM ids
  python scripts/pitch_umaps.py "Fried, Max" --asof 2025-06-15
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figure_style import PITCH_NAMES, PRINT_W, apply_style, cell_lw, cell_tag, save, style_axes
from mlbpitch import paths, profiles

SEED = 0
RANGE = f"{min(profiles.SEASONS)}_{max(profiles.SEASONS)}"
FIG_DIR = paths.ROOT / "analysis"
UMAP_PATH = paths.META / f"snap_umap_{RANGE}.parquet"
EMBED_PATH = paths.META / f"snap_embed_{RANGE}.parquet"
EXAMPLES = ["Gausman, Kevin", "Hancock, Emerson"]
ME_COLOR, COMP_COLOR = "#d73027", "#4575b4"
CMAP = "cividis"            # blue -> yellow: no green under the red marker
MAX_ROWS = 4                # pitches per card; Cell caps a figure at 20 cm tall
MIN_USAGE = 0.03            # rarer pitches are listed, not given a row
MIN_PITCHES = 50            # season-to-date pitches of the type for a full row
GLOVE_SIDE = {"SL", "ST", "FC", "CU"}
DROP_TYPES = {"FS", "CH", "CU", "SI"}   # pitches valued for depth, not ride
# Headline outcome per pitch type: sinkers are contact pitches, so CSW not whiff.
OUTCOME = {pt: ("whiff_pct", "Whiff", "Whiff % per swing") for pt in profiles.PITCH_TYPES}
OUTCOME["SI"] = ("csw_pct", "CSW", "CSW % per pitch")
FIT_BINS = [(0.5, "close"), (0.9, "fair"), (1.01, "loose")]

# (column, label, unit, scale, decimals); the movement rows are resolved per type
FINGERPRINT = [("release_speed", "Velocity", " mph", 1, 1),
               ("ivb", None, " in", 1, 1),
               ("hb", None, " in", 1, 1),
               ("vaa_above_avg", "VAA vs. exp. (+ flatter)", "°", 1, 2),
               ("arm_angle", "Arm angle", "°", 1, 1),
               ("release_extension", "Extension", " ft", 1, 1),
               ("in_zone", "In zone", "%", 100, 0),
               ("usage", "Usage", "%", 100, 0)]


def first_last(name):
    last, _, first = name.partition(", ")
    return f"{first} {last}".strip()


def name_labels(snap):
    """pitcher id -> unambiguous short label. 'F. Last' unless two pitchers share
    it (27 do: Luis Castillo x2, Tyler/Trevor Rogers...), then the full first
    name, then the MLBAM id."""
    ids = snap.groupby("pitcher").player_name.last()
    short = ids.map(lambda n: (lambda l, f: f"{f[:1]}. {l}" if f else l)(*n.partition(", ")[::2]))
    full = ids.map(first_last)
    lab = short.copy()
    dup = short.duplicated(keep=False)
    lab[dup] = full[dup]
    dup2 = lab.duplicated(keep=False)
    lab[dup2] = lab[dup2] + " (" + lab[dup2].index.astype(str) + ")"
    assert lab.is_unique
    return lab.to_dict()


def date_label(ts, season_end):
    """Snapshot date as what it means: estimate through that date."""
    if ts == season_end.get(ts.year):
        return f"'{str(ts.year)[2:]} final"
    return f"thru {ts.month}/{ts.day}/{str(ts.year)[2:]}"


def pct_str(x):
    return "<1%" if x < 0.01 else f"{100 * x:.0f}%"


def load():
    snap = pd.read_parquet(paths.META / f"snapshots_{RANGE}.parquet")
    snap["ivb"], snap["hb"] = 12 * snap.pfx_z, -12 * snap.pfx_x     # mirrored frame, inches
    snap["gb"], snap["drop"] = -snap.hb, -snap.ivb
    comps = pd.read_parquet(paths.META / f"snap_comps_strict_{RANGE}.parquet")
    games = pd.read_parquet(paths.META / f"ts_games_{RANGE}.parquet",
                            columns=["pitcher", "pitch_type", "game_date", "n", "season"])
    return snap, comps, games


def umap_by_type():
    """One UMAP per pitch type on the snapshot PCs; cached, rebuilt when the
    embedding is newer than the cache."""
    if UMAP_PATH.exists() and UMAP_PATH.stat().st_mtime > EMBED_PATH.stat().st_mtime:
        return pd.read_parquet(UMAP_PATH)
    import umap as umap_mod

    emb = pd.read_parquet(EMBED_PATH)
    pcs = [c for c in emb if c.startswith("pc")]
    out = []
    for pt in profiles.PITCH_TYPES:
        d = emb[emb.pitch_type == pt].copy()
        U = umap_mod.UMAP(n_neighbors=15, min_dist=0.3, random_state=SEED).fit_transform(d[pcs].to_numpy())
        d["x"], d["y"] = U[:, 0], U[:, 1]
        out.append(d[["asof", "season", "pitcher", "pitch_type", "x", "y"]])
        print(f"  umap {pt}: {len(d):,} rows", flush=True)
    out = pd.concat(out, ignore_index=True)
    out.to_parquet(UMAP_PATH, index=False)
    return out


def with_outcome_pct(u, snap):
    """Landscape colour = within-type percentile of the headline outcome estimate,
    so every pitch type uses the full 0-100 scale (raw whiff saturates sinkers)."""
    key = ["asof", "season", "pitcher", "pitch_type"]
    d = u.merge(snap[key + ["whiff_pct", "csw_pct"]], on=key)
    d["out_pct"] = np.nan
    for pt, (col, *_) in OUTCOME.items():
        m = d.pitch_type == pt
        d.loc[m, "out_pct"] = 100 * d.loc[m, col].rank(pct=True)
    return d


def fig_atlas(u):
    fig, axes = plt.subplots(2, 4, figsize=(PRINT_W, 3.9), layout="constrained")
    for ax, pt, tag in zip(axes.flat, profiles.PITCH_TYPES, "ABCDEFGH"):
        d = u[u.pitch_type == pt].sample(frac=1, random_state=SEED)   # no plot-order bias
        sc = ax.scatter(d.x, d.y, c=d.out_pct, s=1.5, cmap=CMAP, vmin=0, vmax=100,
                        linewidths=0, rasterized=True)
        ax.set_title(f"{PITCH_NAMES[pt]} ({OUTCOME[pt][1]}, n = {len(d):,})", loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        style_axes(ax)
        cell_tag(ax, tag, x=-0.04)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("Whiff percentile within pitch type (sinker: CSW)")
    save(fig, FIG_DIR / "pitch_atlas_by_type")


def resolve(snap, who):
    """Name ('Last, First') or MLBAM id -> one pitcher id."""
    if str(who).isdigit():
        return int(who)
    ids = snap.loc[snap.player_name == who, "pitcher"].unique()
    if len(ids) != 1:
        raise SystemExit(f"{who!r} matches {len(ids)} pitchers ({list(ids)}); pass the MLBAM id")
    return int(ids[0])


def fingerprint_rows(pt):
    rows = []
    for c, lab, unit, scale, dec in FINGERPRINT:
        if c == "ivb":
            c, lab = ("drop", "Drop (−iVB)") if pt in DROP_TYPES else ("ivb", "Ride (iVB)")
        elif c == "hb":
            c, lab = ("gb", "Glove-side break") if pt in GLOVE_SIDE else ("hb", "Arm-side run")
        rows.append((c, lab, unit, scale, dec))
    return rows


def fingerprint_panel(ax, pool, me, cm, pt, show_x, n_ref):
    rows = fingerprint_rows(pt)
    y = np.arange(len(rows))[::-1]
    for yi, (c, *_r) in zip(y, rows):
        ax.plot([0, 100], [yi, yi], color="0.9", lw=cell_lw(1.5), zorder=0)
        ax.scatter(cm[f"{c}_pct"], np.full(len(cm), yi), s=6, color=COMP_COLOR, linewidths=0, zorder=2)
        med = cm[f"{c}_pct"].median()
        ax.plot([med, med], [yi - 0.32, yi + 0.32], color=COMP_COLOR, lw=cell_lw(1), zorder=2)
        ax.scatter(me[f"{c}_pct"], yi, s=20, color=ME_COLOR, zorder=3)
    labels = [f"{lab} {round(scale * me[c], dec) + 0.0:.{dec}f}{unit}" for c, lab, unit, scale, dec in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel(f"Percentile among active {PITCH_NAMES[pt].lower()}s that date" if show_x else "")
    ax.text(1.0, 1.02, f"n = {n_ref}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="0.4")
    style_axes(ax)


def comps_panel(ax, sel, me, pt, xmax, season_end, labels):
    col, _, axis_lab = OUTCOME[pt]
    s = sel.sort_values("rank", ascending=False)        # closest at the top
    y = np.arange(len(s))
    lo = 100 * max(me[col] - 1.96 * me[f"{col}_sd"], 0.0)
    hi = 100 * min(me[col] + 1.96 * me[f"{col}_sd"], 1.0)
    ax.axvspan(lo, hi, color=ME_COLOR, alpha=0.15, linewidth=0, zorder=0)
    ax.barh(y, 100 * s[f"comp_{col}"], height=0.65, color=COMP_COLOR, alpha=0.85, zorder=1)
    ax.axvline(100 * me[col], color=ME_COLOR, lw=cell_lw(1), zorder=2)
    ax.axvline(100 * s[f"comp_{col}"].mean(), color=COMP_COLOR, ls="--", lw=cell_lw(0.75), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{labels[p]} {date_label(a, season_end)}"
                        for p, a in zip(s.comp_pitcher, s.comp_asof)])
    ax.set_xlim(0, xmax)
    ax.set_xlabel(axis_lab)
    style_axes(ax)


def comp_fit(comps):
    """Each query row's median comp distance, as a percentile among all rows of
    its pitch type -- the same statistic on both sides, so 'close' (< 50th) is
    half the rows by construction."""
    med = comps.groupby(["asof", "pitcher", "pitch_type"]).distance.median().rename("med")
    med = med.reset_index()
    med["fit_pct"] = med.groupby("pitch_type").med.rank(pct=True)
    return med.set_index(["asof", "pitcher", "pitch_type"]).fit_pct


def card(snap, u, comps, games, labels, fit, who, asof=None):
    pid = resolve(snap, who)
    mine = snap[snap.pitcher == pid]
    name = mine.player_name.iloc[0]
    D = mine["asof"].max() if asof is None else pd.Timestamp(asof)
    now = mine[mine["asof"] == D].sort_values("usage", ascending=False).copy()
    if now.empty:
        raise SystemExit(f"{name} has no snapshot rows on {D.date()}")
    season_end = snap.groupby("season")["asof"].max().to_dict()

    # season-to-date pitches and last game per pitch type, as of D
    g = games[(games.pitcher == pid) & (games.season == D.year) & (games.game_date <= D)]
    std = g.groupby("pitch_type").agg(n_std=("n", "sum"), last=("game_date", "max"))
    now = now.join(std, on="pitch_type")
    now["n_std"] = now.n_std.fillna(0)
    rare = now[now.usage < MIN_USAGE]
    thin = now[(now.usage >= MIN_USAGE) & (now.n_std < MIN_PITCHES)]
    ok = now[(now.usage >= MIN_USAGE) & (now.n_std >= MIN_PITCHES)]
    over = ok.iloc[MAX_ROWS:]
    now = ok.head(MAX_ROWS)

    sels = {pt: comps[(comps.pitcher == pid) & (comps.pitch_type == pt) & (comps["asof"] == D)]
            for pt in now.pitch_type}
    xmax = 5 + max(100 * max(sels[r.pitch_type][f"comp_{OUTCOME[r.pitch_type][0]}"].max(),
                             getattr(r, OUTCOME[r.pitch_type][0]))
                   for r in now.itertuples())

    fig = plt.figure(figsize=(PRINT_W, 0.95 + 1.6 * len(now)), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02, hspace=0.02)
    gs = fig.add_gridspec(len(now), 3, width_ratios=[0.7, 1.25, 1.3])
    for i, r in enumerate(now.itertuples()):
        pt = r.pitch_type
        pool = snap[(snap.pitch_type == pt) & (snap["asof"] == D)].copy()   # same-date reference
        for c, *_ in fingerprint_rows(pt):
            pool[f"{c}_pct"] = 100 * pool[c].rank(pct=True)
        me = pool[pool.pitcher == pid].iloc[0]
        sel = sels[pt]
        # comps live on their own dates: rank each against ITS date's active pool
        cm = []
        for (a, p) in zip(sel.comp_asof, sel.comp_pitcher):
            ref = snap[(snap.pitch_type == pt) & (snap["asof"] == a)].copy()
            for c, *_ in fingerprint_rows(pt):
                ref[f"{c}_pct"] = 100 * ref[c].rank(pct=True)
            row = ref[ref.pitcher == p].iloc[0]
            cm.append(row)
        cm = pd.DataFrame(cm)
        bg = u[u.pitch_type == pt].sample(frac=1, random_state=SEED)
        me_xy = bg[(bg.pitcher == pid) & (bg["asof"] == D)].iloc[0]
        col, lab, _ = OUTCOME[pt]
        f_pct = fit.get((D, pid, pt), np.nan)
        f_lab = next(l for cut, l in FIT_BINS if f_pct < cut) if np.isfinite(f_pct) else "n/a"

        ax = fig.add_subplot(gs[i, 0])
        sc = ax.scatter(bg.x, bg.y, c=bg.out_pct, s=1.2, cmap=CMAP, vmin=0, vmax=100,
                        linewidths=0, rasterized=True)
        ax.scatter(me_xy.x, me_xy.y, s=36, facecolor=ME_COLOR, edgecolor="black",
                   lw=cell_lw(0.75), zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        style_axes(ax)
        ax.set_title(f"{PITCH_NAMES[pt]} ({pct_str(me.usage)})", loc="left")
        cell_tag(ax, "ABCDEFGH"[i], x=-0.1)

        ax = fig.add_subplot(gs[i, 1])
        fingerprint_panel(ax, pool, me, cm, pt, i == len(now) - 1, len(pool))

        ax = fig.add_subplot(gs[i, 2])
        comps_panel(ax, sel, me, pt, xmax, season_end, labels)
        last = pd.Timestamp(r.last)
        ax.set_title(f"{lab} {100 * me[col]:.0f}% ({100 * max(me[col] - 1.96 * me[f'{col}_sd'], 0):.0f}"
                     f"–{100 * min(me[col] + 1.96 * me[f'{col}_sd'], 1):.0f}), fit {f_lab}\n"
                     f"{int(r.n_std)} pitches '{str(D.year)[2:]}, last {last.month}/{last.day}",
                     loc="left")

    cb = fig.colorbar(sc, ax=fig.axes[0::3], location="bottom", shrink=0.9, aspect=12, pad=0.01)
    cb.set_label("Whiff pctile (sinker: CSW)")
    handles = [Line2D([], [], marker="o", ls="", color=ME_COLOR, label=first_last(name)),
               Line2D([], [], marker="o", ls="", color=COMP_COLOR, ms=4, label="10 closest comps"),
               Line2D([], [], marker="|", ls="", color=COMP_COLOR, ms=7, label="Comps median"),
               Line2D([], [], color=ME_COLOR, lw=cell_lw(1), label="His estimate, 95% shaded"),
               Line2D([], [], color=COMP_COLOR, ls="--", lw=cell_lw(0.75), label="Comps mean")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=6, frameon=False)

    notes = []
    if len(over):
        notes.append("Not shown (row limit): " + ", ".join(f"{PITCH_NAMES[p]} {pct_str(x)}"
                                                         for p, x in zip(over.pitch_type, over.usage)))
    if len(thin):
        notes.append(f"Under {MIN_PITCHES} pitches in {D.year}: " + ", ".join(
            f"{PITCH_NAMES[p]} ({int(n)})" for p, n in zip(thin.pitch_type, thin.n_std)))
    if len(rare):
        notes.append("Rarely thrown (<3%): " + ", ".join(f"{PITCH_NAMES[p]} {pct_str(x)}"
                                                        for p, x in zip(rare.pitch_type, rare.usage)))
    fig.suptitle(f"{first_last(name)}, as of {D.date()}: state-space estimates from games to that date. "
                 f"Comps: nearest on 6 equally weighted groups (shape, angle residuals, delivery, "
                 f"location, consistency, usage), none dated after {D.date()}; comps fit = their median "
                 f"distance vs. all rows of the type (close < 50th pct, fair < 90th, else loose). "
                 f"Percentiles rank raw values among pitchers active that date."
                 + ("  " + "; ".join(notes) + "." if notes else ""),
                 fontsize=6, x=0.01, ha="left", wrap=True)
    out = FIG_DIR / "cards"
    out.mkdir(exist_ok=True)
    stem = out / name.replace(", ", "_").replace(" ", "_")
    _check_clipping(fig)
    save(fig, stem)


def _check_clipping(fig):
    """Warn if any title runs past the figure edge (constrained layout ignores them)."""
    fig.canvas.draw()
    W = fig.bbox.width
    for ax in fig.axes:
        for t in (ax.title, ax._left_title):
            if t.get_text() and t.get_window_extent().x1 > W + 1:
                print(f"  WARNING clipped title: {t.get_text()!r}", flush=True)


if __name__ == "__main__":
    apply_style()
    args = sys.argv[1:]
    asof = None
    if "--asof" in args:
        i = args.index("--asof")
        asof = args[i + 1]
        args = args[:i] + args[i + 2:]
    snap, comps, games = load()
    u = with_outcome_pct(umap_by_type(), snap)
    labels, fit = name_labels(snap), comp_fit(comps)
    if not args:
        fig_atlas(u)
        args = EXAMPLES
    for who in args:
        card(snap, u, comps, games, labels, fit, who, asof)
