"""Pitch-design forecast: when a pitcher changes a pitch (it moves on the map), does the change in what its
lookalikes get predict the change in whiff rate he actually gets?

The decision this tests (player development / pitch lab): "moving your slider here is worth about +X whiff points".

Unit: MLB pitcher x pitch type, consecutive seasons (2024->25, 2025->26). Secondary: first half -> second half of a
season. Per pitch type, in the atlas's identity space (15 measures, block-scaled, `stratification_scan.identity_space`):

  V   = mean whiff (points) of the 100 nearest rows thrown by OTHER pitchers (any period): what pitches like this get
        (100, not the atlas's 30-lookalike list: with 30 the value is too noisy to difference and overstates changes
        about twofold, calibration slope 0.43; with 100 it is 0.83. The report has the k sensitivity.)
  dV  = V(later pitch) - V(earlier pitch)                       the map's forecast of the change
  dW  = whiff(later) - whiff(earlier), demeaned within type x period pair (league-wide drift removed)

Forecasts, scored out of sample over 5 pitcher folds (coefficients fit on 4 folds, scored on the 5th):
  rtm        regression to the mean: dW ~ type-specific slope on the earlier whiff (deviation from type mean)
  linear     rtm + dLin   (change in the 15 measures x cross-sectional linear coefficients, fit on training pitchers)
  movement   rtm + dVm    (lookalike value on the movement plot only: ride and run)
  lookalike  rtm + dV     (the map)
  + location rtm + dV + change in location height x type-specific slope (Fig. 4)
Skill = 1 - SSE / SSE(no change). Pitcher-clustered bootstrap intervals over the out-of-fold predictions.
Calibration: slope of dW on dV given the earlier whiff (1 = the forecast is on the right scale).

Outputs analysis/design_forecast/: pairs.parquet, skill.csv, calibration.csv, quartiles.csv, examples.csv,
report.md, Figure6.{pdf,png}.

    python scripts/design_forecast.py            # full run (about a minute)
    python scripts/design_forecast.py --quick    # four-seamers only
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from mlbpitch import paths, profiles
from profile_embed import IDENTITY_BLOCKS
from stratification_scan import identity_space, lookalikes, ols_cluster

OUT = paths.ROOT / "analysis" / "design_forecast"
TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]
ID_COLS = [c for cs in IDENTITY_BLOCKS.values() for c in cs]
K_VALUE = 100        # neighbours behind the value surface (see docstring); module-level so the sensitivity loop can set it
SEED = 0
N_BOOT = 1000
FORECASTS = ["rtm", "linear", "movement", "lookalike", "lookalike_loc"]
NAMES = {"rtm": "Regression to the mean", "linear": "+ linear model of the 15 measures", "movement": "+ movement-plot lookalikes",
         "lookalike": "+ lookalikes (the map)", "lookalike_loc": "+ lookalikes + location"}
CASES = {676974: "Meyer", 676106: "Hancock", 608331: "Fried", 673513: "Matsui", 592332: "Gausman", 605400: "Nola"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def knn_value(Q, y, pid, k=None):
    nb = lookalikes(Q, pid, k or K_VALUE)
    assert not (pid[nb] == pid[:, None]).any(), "focal pitcher inside his own lookalike set"
    return np.nanmean(y[nb], axis=1), nb


def type_rows(g):
    """Per-row values for one pitch type: identity PCs, lookalike value V, movement-only value Vm, z-scored measures."""
    g = g.reset_index(drop=True)
    pid = g.pitcher.to_numpy()
    P, _, _ = identity_space(g)
    w = 100 * g.whiff_pct.to_numpy(dtype="float64")
    V, _ = knn_value(P, w, pid)
    mv = g[["pfx_x", "pfx_z"]].astype("float64")
    M = ((mv - mv.mean()) / mv.std(ddof=0)).fillna(0).to_numpy()
    Vm, _ = knn_value(M, w, pid)
    F = g[ID_COLS].astype("float64")
    Z = ((F - F.mean()) / F.std(ddof=0)).fillna(0)
    out = g[["pitcher", "pitch_type", "period", "n", "player_name"]].copy()
    out["whiff"], out["V"], out["Vm"] = w, V, Vm
    out["height_in"] = 12 * g.plate_z.to_numpy(dtype="float64")
    for j in range(P.shape[1]):
        out[f"pc{j}"] = P[:, j]
    for c in ID_COLS:
        out[f"z_{c}"] = Z[c].to_numpy()
    return out


def make_pairs(rows, transitions, min_n):
    """Consecutive periods of the same pitcher and type."""
    a = rows.rename(columns=lambda c: c if c in ("pitcher", "pitch_type") else c + "_0")
    b = rows.rename(columns=lambda c: c if c in ("pitcher", "pitch_type") else c + "_1")
    out = []
    for p0, p1 in transitions:
        m = a[a.period_0 == p0].merge(b[b.period_1 == p1], on=["pitcher", "pitch_type"])
        out.append(m)
    d = pd.concat(out, ignore_index=True)
    d = d[(d.n_0 >= min_n) & (d.n_1 >= min_n)].copy()
    d["pair"] = d.period_0.astype(str) + "->" + d.period_1.astype(str)
    d["dW_raw"] = d.whiff_1 - d.whiff_0
    grp = d.groupby(["pitch_type", "pair"])
    d["dW"] = d.dW_raw - grp.dW_raw.transform("mean")
    d["w0"] = d.whiff_0 - grp.whiff_0.transform("mean")
    d["dV"] = d.V_1 - d.V_0
    d["dVm"] = d.Vm_1 - d.Vm_0
    d["dH"] = d.height_in_1 - d.height_in_0
    pcs = sorted({c[:-2] for c in d.columns if c.startswith("pc") and c[2:-2].isdigit()}, key=lambda s: int(s[2:]))
    d["move"] = np.sqrt(sum((d[f"{c}_1"] - d[f"{c}_0"]) ** 2 for c in pcs))
    d["move_q"] = d.groupby("pitch_type").move.transform(lambda s: pd.qcut(s, 4, labels=False, duplicates="drop"))
    return d.reset_index(drop=True)


def design(d, cols_extra, rows_train=None, lin_coef=None):
    """Design matrix: type intercepts, type-specific RTM slopes, then extra columns (some type-specific)."""
    X = []
    for pt in TYPES:
        m = (d.pitch_type == pt).to_numpy(float)
        X += [m, m * d.w0.to_numpy()]
    for c in cols_extra:
        if c == "dH":
            for pt in TYPES:
                X.append((d.pitch_type == pt).to_numpy(float) * d.dH.to_numpy())
        else:
            X.append(d[c].to_numpy())
    return np.column_stack(X)


def lin_feature(d, rows, train_pitchers):
    """dLin: change in the z-scored 15 measures x cross-sectional coefficients (whiff ~ measures within type),
    fit on training pitchers' rows only."""
    out = np.zeros(len(d))
    for pt in TYPES:
        r = rows[(rows.pitch_type == pt) & rows.pitcher.isin(train_pitchers)]
        if len(r) < 50:
            continue
        Z = r[[f"z_{c}" for c in ID_COLS]].to_numpy()
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(Z)), Z]), r.whiff.to_numpy(), rcond=None)[0][1:]
        m = (d.pitch_type == pt).to_numpy()
        dz = d.loc[m, [f"z_{c}_1" for c in ID_COLS]].to_numpy() - d.loc[m, [f"z_{c}_0" for c in ID_COLS]].to_numpy()
        out[m] = dz @ beta
    return out


EXTRA = {"rtm": [], "linear": ["dLin"], "movement": ["dVm"], "lookalike": ["dV"], "lookalike_loc": ["dV", "dH"]}


def out_of_fold(d, rows, n_folds=5, seed=SEED):
    rng = np.random.default_rng(seed)
    pit = d.pitcher.unique()
    fold_of = dict(zip(pit, rng.integers(0, n_folds, len(pit))))
    fold = d.pitcher.map(fold_of).to_numpy()
    pred = {f: np.zeros(len(d)) for f in FORECASTS}
    d = d.copy()
    d["dLin"] = 0.0
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        train_p = set(pit[[fold_of[p] != k for p in pit]])
        d["dLin"] = lin_feature(d, rows, train_p)
        y = d.dW.to_numpy()
        for f in FORECASTS:
            X = design(d, EXTRA[f])
            beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            pred[f][te] = X[te] @ beta
    d["dLin"] = lin_feature(d, rows, set(pit))          # full-data feature for calibration slopes
    return d, pred


def skill(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum(y ** 2)


def boot_skill(d, pred, n_boot=N_BOOT, seed=SEED):
    """Pitcher-clustered bootstrap of out-of-fold skill and of paired differences against rtm."""
    rng = np.random.default_rng(seed + 1)
    pit = d.pitcher.unique()
    idx = {p: np.flatnonzero(d.pitcher.to_numpy() == p) for p in pit}
    y = d.dW.to_numpy()
    draws = {f: [] for f in FORECASTS}
    for _ in range(n_boot):
        rows = np.concatenate([idx[p] for p in rng.choice(pit, len(pit))])
        for f in FORECASTS:
            draws[f].append(skill(y[rows], pred[f][rows]))
    out = []
    for f in FORECASTS:
        s, dd = skill(y, pred[f]), np.array(draws[f])
        diff = dd - np.array(draws["rtm"])
        out.append({"forecast": f, "skill": s, "lo": np.percentile(dd, 2.5), "hi": np.percentile(dd, 97.5),
                    "vs_rtm": s - skill(y, pred["rtm"]), "vs_rtm_lo": np.percentile(diff, 2.5), "vs_rtm_hi": np.percentile(diff, 97.5),
                    "spearman": spearmanr(y, pred[f]).statistic,
                    **{f"skill_{pt}": skill(y[(d.pitch_type == pt).to_numpy()], pred[f][(d.pitch_type == pt).to_numpy()])
                       for pt in TYPES if (d.pitch_type == pt).any()}})
    return pd.DataFrame(out)


def calibration(d):
    """Slope of dW on each forecast change, given type intercepts and type-specific RTM slopes; pitcher-clustered SE."""
    C = design(d, [])
    out = []
    for c, lab in [("dV", "lookalikes (the map)"), ("dLin", "linear model"), ("dVm", "movement-plot lookalikes")]:
        b, se, n, G = ols_cluster(d.dW.to_numpy(), d[c].to_numpy(), C, d.pitcher.to_numpy(), intercept=False)
        out.append({"forecast": c, "label": lab, "slope": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "n": n, "pitchers": G,
                    "sd_change": float(d[c].std())})
    return pd.DataFrame(out)


def quartiles(d, pred):
    y = d.dW.to_numpy()
    C = design(d, [])
    out = []
    for q in sorted(d.move_q.dropna().unique()):
        m = (d.move_q == q).to_numpy()
        b, se, n, G = ols_cluster(y[m], d.dV.to_numpy()[m], C[m], d.pitcher.to_numpy()[m], intercept=False)
        out.append({"move_quartile": int(q) + 1, "n": int(m.sum()), "median_move": float(d.move[m].median()),
                    "sd_dV": float(d.dV[m].std()),
                    **{f"gain_{f}": skill(y[m], pred[f][m]) - skill(y[m], pred["rtm"][m]) for f in FORECASTS[1:]},
                    "slope_dV": b, "slope_lo": b - 1.96 * se, "slope_hi": b + 1.96 * se})
    return pd.DataFrame(out)


def permuted_gain(d, rows, n_perm=20, seed=SEED):
    rng = np.random.default_rng(seed + 2)
    gains = []
    for _ in range(n_perm):
        dp = d.copy()
        dp["dV"] = dp.groupby("pitch_type").dV.transform(lambda s: rng.permutation(s.to_numpy()))
        dp2, pr = out_of_fold(dp, rows)
        gains.append(skill(dp2.dW.to_numpy(), pr["lookalike"]) - skill(dp2.dW.to_numpy(), pr["rtm"]))
    return np.array(gains)


def build(by_half, types):
    t = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=by_half)).reset_index(drop=True)
    t = t[t.pitch_type.isin(types)]
    t["period"] = (t.season * 10 + t.half) if by_half else t.season
    rows = pd.concat([type_rows(g) for _, g in t.groupby("pitch_type")], ignore_index=True)
    if by_half:
        trans = [(s * 10 + 1, s * 10 + 2) for s in profiles.SEASONS]
    else:
        ss = sorted(profiles.SEASONS)
        trans = list(zip(ss[:-1], ss[1:]))
    return t, rows, trans


# ------------------------------------------------------------------ figure
def figure(d, rows, pred, sk, cal, qt, types):
    sys.path.insert(0, str(paths.ROOT / "scripts"))
    import matplotlib.pyplot as plt
    import umap_benchmark_plots as ubp
    from figure_style import PITCH_NAMES, PRINT_W, save
    SZ = ubp.setup()
    lw, finish = ubp.lw, ubp.finish
    MM = 1 / 25.4
    fig = plt.figure(figsize=(PRINT_W, 150 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    # A: four-seam map coloured by the lookalike whiff landscape, with the movers' paths
    axA = fig.add_subplot(gs[0, 0])
    at = pd.read_parquet(paths.DATA / "meta" / "atlas_type_2024_2026.parquet")
    at = at[(at.level == "mlb") & (at.pitch_type == "FF")][["pitcher", "season", "tx", "ty"]]
    ff = rows[rows.pitch_type == "FF"].rename(columns={"period": "season"}).merge(at, on=["pitcher", "season"])
    lo, hi = np.nanpercentile(ff.V, [5, 95])
    sc = axA.scatter(ff.tx, ff.ty, c=ff.V, cmap="cividis", vmin=lo, vmax=hi, s=1.6, lw=0, rasterized=True)
    cb = fig.colorbar(sc, ax=axA, fraction=0.04, pad=0.01)
    cb.set_label("Whiff rate of 100 nearest lookalikes (%)", fontsize=SZ.axis_text); cb.outline.set_linewidth(lw(0.5))
    cb.ax.tick_params(length=0, labelsize=SZ.axis_text)
    for pid, name in CASES.items():
        s = ff[ff.pitcher == pid].sort_values("season")
        if len(s) < 2:
            continue
        if name not in ("Meyer", "Hancock", "Fried"):      # the stable three barely move; label the movers only
            continue
        col = "#b2182b"
        for (_, r0), (_, r1) in zip(s.iloc[:-1].iterrows(), s.iloc[1:].iterrows()):
            axA.annotate("", (r1.tx, r1.ty), (r0.tx, r0.ty), arrowprops=dict(arrowstyle="-|>", color=col, lw=lw(1.0),
                                                                              shrinkA=0, shrinkB=0, mutation_scale=6))
        e = d[(d.pitcher == pid) & (d.pitch_type == "FF")]
        txt = f"{name}: {e.dV.sum():+.1f} / {e.dW_raw.sum():+.1f}" if len(e) else name
        r = s.iloc[-1]
        axA.annotate(txt, (r.tx, r.ty), xytext=(3, -3), textcoords="offset points", fontsize=SZ.annot, color=col,
                     bbox=dict(fc="black", ec="none", alpha=0.45, pad=0.6))
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_title("Four-seamers, 2024 to 2026: predicted / actual change", fontsize=SZ.axis_ttl, pad=2)
    finish(axA)
    # B: binned forecast vs actual (after regression to the mean), by family
    axB = fig.add_subplot(gs[0, 1])
    fam = {"Fastballs": (["FF", "SI"], "#b2182b"), "Breaking": (["SL", "ST", "CU", "FC"], "#2166ac"), "Offspeed": (["CH", "FS"], "#737373")}
    resid = d.dW.to_numpy() - pred["rtm"]
    allx = []
    for name, (ts, col) in fam.items():
        m = d.pitch_type.isin(ts).to_numpy()
        if m.sum() < 40:
            continue
        x, y = d.dV.to_numpy()[m], resid[m]
        q = pd.qcut(x, 8, labels=False, duplicates="drop")
        g = pd.DataFrame({"x": x, "y": y, "q": q}).groupby("q").agg(x=("x", "mean"), y=("y", "mean"), s=("y", "std"), n=("y", "size"))
        axB.errorbar(g.x, g.y, yerr=1.96 * g.s / np.sqrt(g.n), fmt="-o", ms=2.5, lw=lw(0.75), elinewidth=lw(0.5), color=col,
                     label=f"{name} (n = {m.sum():,})")
        allx += list(g.x)
    lim = max(abs(min(allx)), abs(max(allx))) * 1.1 if allx else 5
    axB.plot([-lim, lim], [-lim, lim], color="black", lw=lw(0.5), ls=":")
    axB.axhline(0, color="black", lw=lw(0.5)); axB.axvline(0, color="black", lw=lw(0.5))
    c0 = cal.set_index("forecast").loc["dV"]
    axB.text(0.97, 0.03, f"slope {c0.slope:.2f} [{c0.lo:.2f}, {c0.hi:.2f}]", transform=axB.transAxes, ha="right", va="bottom", fontsize=SZ.annot)
    axB.set_xlabel("Predicted whiff change (points):\nlookalikes of the new pitch minus the old")
    axB.set_ylabel("Actual whiff change beyond\nregression to the mean (points)")
    axB.legend(loc="upper left", handlelength=1.5, handletextpad=0.3)
    finish(axB, legend=True)
    # C: out-of-fold skill
    axC = fig.add_subplot(gs[1, 0])
    sk_ = sk.set_index("forecast").reindex(FORECASTS)
    cols = ["#b3b3b3", "#737373", "#92c5de", "#2166ac", "#b2182b"]
    ys = np.arange(len(FORECASTS))[::-1]
    pts = [[100 * sk_.loc[f, f"skill_{pt}"] for pt in types if f"skill_{pt}" in sk_.columns] for f in FORECASTS]
    ubp.ci_bars(axC, ys, 100 * sk_.skill.to_numpy(), 100 * sk_.lo.to_numpy(), 100 * sk_.hi.to_numpy(), cols, pts)
    axC.set_yticks(ys); axC.set_yticklabels([NAMES[f] for f in FORECASTS])
    axC.axvline(0, color="black", lw=lw(0.5))
    axC.set_xlabel("Season-to-season whiff change explained, out of sample (%)")
    finish(axC)
    # D: added skill over regression to the mean, by how far the pitch moved
    axD = fig.add_subplot(gs[1, 1])
    for f, col, mk in [("linear", "#737373", "s"), ("movement", "#92c5de", "^"), ("lookalike", "#2166ac", "o"), ("lookalike_loc", "#b2182b", "D")]:
        lab = {"linear": "Linear model", "movement": "Movement-plot lookalikes", "lookalike": "Lookalikes (the map)",
               "lookalike_loc": "Lookalikes + location"}[f]
        axD.plot(qt.move_quartile, 100 * qt[f"gain_{f}"], "-" + mk, ms=2.8, lw=lw(0.75), color=col, label=lab)
    axD.axhline(0, color="black", lw=lw(0.5))
    axD.set_xticks(qt.move_quartile); axD.set_xticklabels(["Least", "2", "3", "Most"])
    axD.set_xlabel("How much the pitch changed\n(quartile of movement in the 15 measures)")
    axD.set_ylabel("Added explained change over\nregression to the mean (points of %)")
    axD.legend(loc="upper left", handlelength=1.5, handletextpad=0.3)
    finish(axD, legend=True)
    ubp.place_tags(fig, [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")])
    save(fig, str(OUT / "Figure6"))


# ------------------------------------------------------------------ report
def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--min-n", type=int, default=100)
    args = ap.parse_args()
    types = ["FF"] if args.quick else TYPES
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {}
    for label, by_half in [("season", False), ("half", True)]:
        t, rows, trans = build(by_half, types)
        for min_n in ([args.min_n, 40] if label == "season" else [args.min_n]):
            d = make_pairs(rows, trans, min_n)
            d, pred = out_of_fold(d, rows)
            sk = boot_skill(d, pred)
            cal = calibration(d)
            qt = quartiles(d, pred)
            res[(label, min_n)] = (d, rows, pred, sk, cal, qt)
            log(f"{label} min_n {min_n}: {len(d)} pairs, {d.pitcher.nunique()} pitchers; skill "
                + " ".join(f"{f} {100 * s:.1f}" for f, s in zip(sk.forecast, sk.skill)))
    d, rows, pred, sk, cal, qt = res[("season", args.min_n)]
    perm = permuted_gain(d, rows, n_perm=5 if args.quick else 20)
    global K_VALUE
    sens = []
    for k in (30, 60, 100, 200):
        K_VALUE = k
        _, rk, tk = build(False, types)
        dk, pk = out_of_fold(make_pairs(rk, tk, args.min_n), rk)
        yk, r0 = dk.dW.to_numpy(), skill(dk.dW.to_numpy(), pk["rtm"])
        ck = calibration(dk).set_index("forecast").loc["dV"]
        sens.append({"neighbours": k, "added_skill_lookalike": 100 * (skill(yk, pk["lookalike"]) - r0),
                     "added_skill_with_location": 100 * (skill(yk, pk["lookalike_loc"]) - r0),
                     "calibration_slope": ck.slope, "slope_lo": ck.lo, "slope_hi": ck.hi})
    K_VALUE = 100
    sens = pd.DataFrame(sens)
    log("k sensitivity done")
    log(f"permuted dV gain over rtm: mean {100 * perm.mean():+.2f} (max {100 * perm.max():+.2f})")
    ex = d[d.pitcher.isin(list(CASES))][["player_name_0", "pitch_type", "pair", "whiff_0", "whiff_1", "dW_raw", "V_0", "V_1", "dV", "dH", "move"]]
    d.drop(columns=[c for c in d.columns if c.startswith(("pc", "z_"))]).to_parquet(OUT / "pairs.parquet", index=False)
    sk.to_csv(OUT / "skill.csv", index=False); cal.to_csv(OUT / "calibration.csv", index=False)
    qt.to_csv(OUT / "quartiles.csv", index=False); ex.to_csv(OUT / "examples.csv", index=False)
    figure(d, rows, pred, sk, cal, qt, types)
    log("figure written")

    L = ["# Pitch-design forecast: does moving on the map predict the whiff change?", "",
         f"*Generated by `scripts/design_forecast.py` on {time.strftime('%Y-%m-%d %H:%M')} ({time.time() - t0:.0f} s). Draft numbers; "
         "see the module docstring for definitions.*", "", "![Figure 6](Figure6.png)", "",
         f"Sample: {len(d):,} season-to-season pairs (both seasons >= {args.min_n} pitches), {d.pitcher.nunique():,} pitchers. "
         f"Target dW = whiff change in points, demeaned within type x season pair (SD {d.dW.std():.2f}). "
         f"Predicted change dV has SD {d.dV.std():.2f} points.", "",
         "## 1. Out-of-sample skill (share of the whiff change explained; 95% pitcher-clustered bootstrap)", ""]
    for key, title in [(("season", args.min_n), f"Season to season, >= {args.min_n} pitches"), (("season", 40), "Season to season, >= 40 pitches"),
                       (("half", args.min_n), f"First half to second half, >= {args.min_n} pitches")]:
        s = res[key][3]
        tab = pd.DataFrame({"forecast": [NAMES[f] for f in s.forecast], "skill %": 100 * s.skill,
                            "95% CI": [f"[{100 * a:.1f}, {100 * b:.1f}]" for a, b in zip(s.lo, s.hi)],
                            "vs RTM": 100 * s.vs_rtm, "vs RTM 95% CI": [f"[{100 * a:+.1f}, {100 * b:+.1f}]" for a, b in zip(s.vs_rtm_lo, s.vs_rtm_hi)],
                            "Spearman": s.spearman})
        L += [f"**{title}** ({len(res[key][0]):,} pairs)", "", md(tab, "{:.2f}"), ""]
    L += [f"Permuted dV (shuffled within type, 20 draws): added skill over RTM mean {100 * perm.mean():+.2f}, max {100 * perm.max():+.2f} points of %.", "",
          "## 2. Calibration: whiff change per point of predicted change (given regression to the mean)", "", md(cal), "",
          "A slope of 1 means a predicted +1 whiff point arrives as +1 actual point; below 1 means the forecast overstates "
          "the change (some of the move is noise in the measures, or the lookalike value is noisy).", "",
          "## 2b. Neighbours behind the value surface (season to season; added skill in points of %)", "", md(sens, "{:.2f}"), "",
          "## 3. By how much the pitch changed (quartiles of movement in the 15-measure space)", "", md(qt), "",
          "## 4. Per pitch type, season to season (skill, % explained)", "",
          md(sk[["forecast"] + [c for c in sk.columns if c.startswith("skill_")]].assign(**{c: 100 * sk[c] for c in sk.columns if c.startswith("skill_")}), "{:.1f}"), "",
          "## 5. Worked examples (Fig. 2B pitchers; points)", "", md(ex, "{:.2f}"), ""]
    (OUT / "report.md").write_text("\n".join(L))
    log(f"done in {time.time() - t0:.0f} s; outputs in {OUT}")


if __name__ == "__main__":
    main()
