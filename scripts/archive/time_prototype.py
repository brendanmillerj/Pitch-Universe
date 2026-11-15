"""Prototype: model time continuously instead of half-season bins.

For one pitch type and a few features, each pitcher's game appearances form an
irregular time series of noisy readings of a latent "true" value:

  observation   y_g = x_g + v_g,      v_g ~ N(0, sigma2 / n_g + tau2)
                (n_g pitches in game g; sigma2 = pooled pitch-to-pitch variance
                 within a game, measured directly from the data, not fitted;
                 tau2 = game-to-game variation that does NOT shrink with pitch
                 count -- park, weather, tracking calibration, the day's feel.
                 Without it the fit read off-nights as true drift: 2.9 in of iVB
                 per 30 days, i.e. a smoother that chased noise.)
  state         x_g = x_{g-1} + w_g,  w_g ~ N(0, q_day * days)  within a season
                                      w_g ~ N(0, q_off)          across an offseason

q_day and q_off are fitted by maximum likelihood POOLED across every pitcher
with enough games of that pitch type, so the examples don't tune their own
smoothing. A Kalman filter + Rauch-Tung-Striebel smoother then gives the
latent value, with a 95% band, at every appearance.

Deliberate jumps (grip change, slot change, Statcast relabel) are what a
random walk blurs, so changepoints are found separately: optimal partitioning
(PELT) of the game means, weighted by each game's total variance
(sigma2/n + tau2 + one typical stretch of drift), with a conservative BIC-style penalty
and a minimum segment length.

Output: analysis/time_prototype.{pdf,png} -- smoothed path vs half-season bins.

Usage:  python scripts/time_prototype.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from figure_style import PRINT_W, apply_style, cell_lw, cell_tag, save, style_axes
from mlbpitch import paths, profiles, store

PITCH_TYPE = "FF"
SEASONS = (2024, 2025, 2026)
EXAMPLES = {"Hancock, Emerson": 676106, "Fried, Max": 608331, "Gausman, Kevin": 592332}
FEATURES = [("ivb", "Four-seam ride, iVB (in)"), ("arm_angle", "Arm angle (°)")]
MIN_GAMES_FIT = 30          # pitchers used to fit q_day / q_off
MIN_SEG = 5                 # changepoint: minimum games per segment
PEN_MULT = 4.0              # changepoint penalty = PEN_MULT * log(n_games)
DRIFT_DAYS = 30             # changepoint cost tolerates this much random-walk drift
SEASON_GAP = 25             # plotted days between seasons (axis compression)
FIG = paths.ROOT / "analysis" / "time_prototype"


def log(msg):
    print(msg, flush=True)


def load_games():
    """Per pitcher x game appearance: n, mean, within-game variance of each feature."""
    frames = []
    for s in SEASONS:
        d = store.query(
            f"SELECT pitcher, player_name, game_pk, game_date, pfx_z, arm_angle "
            f"FROM pitches WHERE pitch_type = '{PITCH_TYPE}'", s)
        d["ivb"] = 12 * d.pfx_z
        d["season"] = s
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    g = p.groupby(["pitcher", "season", "game_pk", "game_date"], observed=True)
    out = g.size().rename("n").to_frame()
    for f, _ in FEATURES:
        out[f] = g[f].mean()
        out[f"{f}_var"] = g[f].var()
    out["player_name"] = g.player_name.first()
    out = out.reset_index().sort_values(["pitcher", "game_date"])
    out["day"] = pd.to_datetime(out.game_date)
    return out


def pooled_sigma2(games, f):
    """Pitch-to-pitch variance within a game, pooled over all appearances."""
    ok = games.n >= 2
    return float(((games.n[ok] - 1) * games[f"{f}_var"][ok]).sum() / (games.n[ok] - 1).sum())


def series(games, pid, f):
    d = games[(games.pitcher == pid) & games[f].notna()]
    dt = d.day.diff().dt.days.to_numpy(dtype=float)
    new_season = (d.season.diff() != 0).to_numpy()
    return d, d[f].to_numpy(), d.n.to_numpy(dtype=float), dt, new_season


def kalman(y, n, dt, new_season, sigma2, q_day, q_off, tau2, smooth=True):
    """Local-level filter (+ RTS smoother) on an irregular series. Returns
    (loglik, smoothed mean, smoothed var). Diffuse-ish start at the first game."""
    T = len(y)
    r = sigma2 / n + tau2
    a_f, p_f, a_p, p_p = (np.empty(T) for _ in range(4))
    ll = 0.0
    a, p = y[0], 1e4
    for t in range(T):
        if t > 0:
            q = q_off if new_season[t] else q_day * max(dt[t], 1.0)
            a, p = a, p + q
        a_p[t], p_p[t] = a, p
        S = p + r[t]
        v = y[t] - a
        if t > 0:  # first observation only initialises the state
            ll += -0.5 * (np.log(2 * np.pi * S) + v * v / S)
        K = p / S
        a, p = a + K * v, (1 - K) * p
        a_f[t], p_f[t] = a, p
    if not smooth:
        return ll, None, None
    a_s, p_s = a_f.copy(), p_f.copy()
    for t in range(T - 2, -1, -1):
        J = p_f[t] / p_p[t + 1]
        a_s[t] = a_f[t] + J * (a_s[t + 1] - a_p[t + 1])
        p_s[t] = p_f[t] + J * J * (p_s[t + 1] - p_p[t + 1])
    return ll, a_s, p_s


def fit_hyper(games, f, sigma2):
    """Pooled ML for (q_day, q_off, tau2) across pitchers with >= MIN_GAMES_FIT games."""
    counts = games[games[f].notna()].groupby("pitcher").size()
    pids = counts[counts >= MIN_GAMES_FIT].index
    data = [series(games, pid, f)[1:] for pid in pids]

    def nll(theta):
        q_day, q_off, tau2 = np.exp(theta)
        return -sum(kalman(y, n, dt, ns, sigma2, q_day, q_off, tau2, smooth=False)[0]
                    for y, n, dt, ns in data)

    res = minimize(nll, x0=np.log([sigma2 / 500, sigma2 / 5, sigma2 / 20]), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 400})
    q_day, q_off, tau2 = np.exp(res.x)
    log(f"  {f}: pitch-to-pitch sd {np.sqrt(sigma2):.2f} | game-to-game sd {np.sqrt(tau2):.2f} "
        f"| true drift sd per 30 days {np.sqrt(30 * q_day):.2f} | offseason jump sd "
        f"{np.sqrt(q_off):.2f} | {len(pids)} pitchers, converged={res.success}")
    return q_day, q_off, tau2


def pelt(y, w, pen, min_seg):
    """Optimal partitioning of a weighted series into constant-mean segments.
    cost(segment) = sum w_i (y_i - weighted mean)^2. Returns segment start indices."""
    T = len(y)
    cw, cwy, cwy2 = (np.r_[0, np.cumsum(v)] for v in (w, w * y, w * y * y))

    def cost(s, e):  # segment y[s:e]
        W = cw[e] - cw[s]
        return (cwy2[e] - cwy2[s]) - (cwy[e] - cwy[s]) ** 2 / W

    F = np.full(T + 1, np.inf)
    F[0] = -pen
    last = np.zeros(T + 1, dtype=int)
    cands = [0]
    for e in range(min_seg, T + 1):
        vals = [(F[s] + cost(s, e) + pen, s) for s in cands if e - s >= min_seg]
        if vals:
            F[e], last[e] = min(vals)
        # PELT pruning
        cands = [s for s in cands if e - s < min_seg or F[s] + cost(s, e) <= F[e]] + \
                ([e - min_seg + 1] if e - min_seg + 1 > 0 else [])
        cands = sorted(set(cands))
    cps, e = [], T
    while e > 0:
        s = last[e]
        if s > 0:
            cps.append(s)
        e = s
    return sorted(cps)


def plot_x(d):
    """Compress offseasons: days since each season's first game + season offset."""
    starts = d.groupby("season").day.transform("min")
    lens = {s: 190 for s in SEASONS}
    off = {s: sum(lens[k] + SEASON_GAP for k in SEASONS if k < s) for s in SEASONS}
    return (d.day - starts).dt.days.to_numpy() + d.season.map(off).to_numpy(), off


def one_step(y, n, dt, new_season, q_day, q_off, tau2, sigma2):
    """Filtered prediction of each game from strictly earlier games (nan for game 0)."""
    pred = np.full(len(y), np.nan)
    a, p = y[0], sigma2 / n[0] + tau2
    for t in range(1, len(y)):
        p = p + (q_off if new_season[t] else q_day * max(dt[t], 1.0))
        pred[t] = a
        r = sigma2 / n[t] + tau2
        K = p / (p + r)
        a, p = a + K * (y[t] - a), (1 - K) * p
    return pred


def forecast_check(games, f, hyper, split):
    """Predict every game from earlier games only, three ways, across all pitchers
    used in the hyperparameter fit:
      state-space    one-step-ahead Kalman prediction
      previous half  n-weighted mean of the pitcher's most recent completed half
                     (what the half-season pipeline knows at that moment)
      season to date n-weighted mean of this season's earlier games
    Scored on games with >= 10 pitches, as RMSE; all three share the same
    irreducible game noise, so the differences are what matter.
    Note q_day/q_off/tau2 were fitted on these same pitchers' likelihoods (3
    pooled numbers over ~40k games), so the state-space column is near out-of-sample."""
    s2, q_day, q_off, tau2 = hyper
    counts = games[games[f].notna()].groupby("pitcher").size()
    errs = {"state-space": [], "previous half": [], "season to date": []}
    for pid in counts[counts >= MIN_GAMES_FIT].index:
        d, y, n, dt, ns = series(games, pid, f)
        ss = one_step(y, n, dt, ns, q_day, q_off, tau2, s2)
        half = np.where(d.day.to_numpy() < d.season.map(split).to_numpy(), 1, 2)
        key = d.season.to_numpy() * 10 + half
        for t in range(1, len(y)):
            if n[t] < 10:
                continue
            prev = key < key[t]
            if not prev.any():
                continue
            last = key[prev].max()
            m = key == last
            ph = np.average(y[m], weights=n[m])
            same = (d.season.to_numpy() == d.season.iloc[t]) & (np.arange(len(y)) < t)
            std = np.average(y[same], weights=n[same]) if same.any() else ph
            for k, v in (("state-space", ss[t]), ("previous half", ph), ("season to date", std)):
                errs[k].append(y[t] - v)
    out = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}
    log(f"  {f}: next-game RMSE over {len(errs['state-space']):,} games -- " +
        ", ".join(f"{k} {v:.2f}" for k, v in out.items()) +
        f"  (noise floor ~ {np.sqrt(s2 / 25 + tau2):.2f} for a 25-pitch game)")
    return out


def main():
    apply_style()
    games = load_games()
    log(f"{len(games):,} four-seam appearances, {games.pitcher.nunique():,} pitchers")
    types = pd.read_parquet(profiles.profiles_path("type"))
    split = {s: pd.Timestamp(profiles.half_split_date(s)) for s in SEASONS}

    hyper = {}
    for f, _ in FEATURES:
        s2 = pooled_sigma2(games, f)
        hyper[f] = (s2, *fit_hyper(games, f, s2))
    log("\nforecast check (predict each game from earlier games only):")
    for f, _ in FEATURES:
        forecast_check(games, f, hyper[f], split)

    fig, axes = plt.subplots(len(EXAMPLES), len(FEATURES), figsize=(PRINT_W, 6.3),
                             layout="constrained", sharex=True)
    summary = []
    for i, (name, pid) in enumerate(EXAMPLES.items()):
        for j, (f, flabel) in enumerate(FEATURES):
            ax = axes[i, j]
            s2, q_day, q_off, tau2 = hyper[f]
            d, y, n, dt, ns = series(games, pid, f)
            _, m, v = kalman(y, n, dt, ns, s2, q_day, q_off, tau2)
            x, off = plot_x(d)

            # raw game means, size = pitches thrown
            ax.scatter(x, y, s=np.clip(n, 3, 40) * 0.35, color="0.65", linewidths=0,
                       zorder=1, rasterized=True)
            # smoothed latent value + 95% band, broken at offseasons
            for s in SEASONS:
                k = (d.season == s).to_numpy()
                if k.any():
                    ax.fill_between(x[k], m[k] - 1.96 * np.sqrt(v[k]), m[k] + 1.96 * np.sqrt(v[k]),
                                    color="#4575b4", alpha=0.25, linewidth=0, zorder=2)
                    ax.plot(x[k], m[k], color="#4575b4", lw=cell_lw(1), zorder=3)
            # half-season bins (what the current pipeline uses), from the same
            # pitch type's row so both lines describe the four-seam only
            col, scale = ("pfx_z", 12) if f == "ivb" else ("arm_angle", 1)
            for s in SEASONS:
                for h in (1, 2):
                    k = ((d.season == s) & ((d.day < split[s]) if h == 1 else (d.day >= split[s]))).to_numpy()
                    row = types[(types.pitcher == pid) & (types.pitch_type == PITCH_TYPE)
                                & (types.season == s) & (types.half == h)]
                    if not k.any() or row.empty:
                        continue
                    ax.hlines(scale * row[col].iloc[0], x[k].min(), x[k].max(), color="#d73027",
                              lw=cell_lw(1.25), zorder=4)
            # changepoints
            w = 1.0 / (s2 / n + tau2 + q_day * DRIFT_DAYS)
            cps = pelt(y, w, PEN_MULT * np.log(len(y)), MIN_SEG)
            for c in cps:
                ax.axvline((x[c - 1] + x[c]) / 2, color="black", ls="--", lw=cell_lw(0.6), zorder=5)
                summary.append((name, f, d.game_date.iloc[c].date(), y[max(0, c - MIN_SEG):c].mean(),
                                y[c:c + MIN_SEG].mean()))
            summary.append((name, f, "start->end", m[0], m[-1]))
            ax.set_xlim(-5, max(off.values()) + 195)
            for s, o in off.items():
                ax.text(o + 95, 1.02, str(s), transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=6, color="0.35")
                if o > 0:
                    ax.axvspan(o - SEASON_GAP, o, color="0.95", zorder=0, linewidth=0)
            ax.set_xticks([])
            if i == 0:
                ax.set_title(flabel, loc="left", pad=10)
            if j == 0:
                ax.set_ylabel(name.split(", ")[1] + " " + name.split(", ")[0])
            style_axes(ax)
            cell_tag(ax, "ABCDEF"[i * len(FEATURES) + j], x=-0.1, y=1.02)

    # legend once
    ax = axes[0, 0]
    ax.scatter([], [], s=6, color="0.65", label="One game (size = pitches)")
    ax.plot([], [], color="#4575b4", lw=cell_lw(1), label="State-space estimate (95% band)")
    ax.plot([], [], color="#d73027", lw=cell_lw(1.25), label="Half-season bin (current)")
    ax.plot([], [], color="black", ls="--", lw=cell_lw(0.6), label="Detected changepoint")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6,
               bbox_to_anchor=(0.5, -0.03))
    for a in axes[-1]:
        a.set_xlabel("Game date (offseasons compressed, gray)")
    save(fig, FIG)

    log("\nchangepoints (mean of 5 games before -> after) and smoothed start->end:")
    for name, f, when, a, b in summary:
        log(f"  {name:<18} {f:<9} {str(when):<11} {a:7.2f} -> {b:7.2f}  ({b - a:+.2f})")


if __name__ == "__main__":
    main()
