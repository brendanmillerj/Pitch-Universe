"""Continuous-time pitch profiles: game-level state-space smoothing + as-of snapshots.

Replaces half-season binning (profiles.py) as the time layer. Every pitcher x
pitch type is a series of game appearances. Each feature is modelled as a latent
local level observed with noise:

  observation   y_g = x_g + v_g,   v_g ~ N(0, obs_var_g + tau2)
  state         x_g = x_{g-1} + w_g,   w_g ~ N(0, q_day * days)   within a season
                                       w_g ~ N(0, q_off)          across an offseason

obs_var_g is the KNOWN sampling variance of game g's value (sigma2 / n for a
mean, sigma2 measured from per-pitch sums of squares; p(1-p) / n for a share,
p = the filter's PREDICTED share; trigamma((n-1)/2) for a log within-game
variance), so a 6-pitch outing counts for little and a 30-pitch start for a lot.
Each series starts from a population prior (a0 = type mean, P0 = between-
pitcher variance), so a row built on one or two games is shrunk, not raw; a
pitcher who skips a season gets one q_off per offseason crossed. tau2 is
game-to-game variation that does not shrink with pitch count (park, weather,
tracking, the day's feel) -- without it the fit reads off-nights as drift.
(q_day, q_off, tau2) are fitted by maximum likelihood per (pitch type, feature),
POOLED over every pitcher with enough games, so no pitcher tunes his own
smoothing. scripts/time_prototype.py showed this cuts next-game forecast error
above the noise floor by about half versus half-season bins.

Feature families (names match profiles.py so downstream code is unchanged):
  means        24 per-pitch means (MEAN_FEATURES), series = games where the
               type was thrown
  usage        5 shares (overall, ahead, behind, vs LHB, vs RHB), series = every
               appearance of the pitcher (0 when the type wasn't thrown)
  consistency  3 within-game spreads (release, movement, velo), modelled as
               bias-corrected log variance, reported as SDs (their *_sd column
               is the posterior sd on the log-variance scale). Within-OUTING repeatability --
               not the half-season spread profiles.py used, which mixed in drift
  outcomes     whiff (per swing), CSW (per pitch), rv100, xwOBA on contact --
               labels only, never embedding features

Snapshots: on each as-of date D the FILTERED state (games on or before D only,
plus drift variance for the days since) -- what was knowable on D, so comps and
their prospective validation never see the future. A (pitcher, type) row exists
on D if the type was thrown within ACTIVE_DAYS before D in the same season.
Smoothed states (which do use later games) are kept per game for description.

Outputs (data/meta):
  ts_hyper_<range>.parquet      fitted (sigma2, q_day, q_off, tau2) per type x feature
  ts_games_<range>.parquet      per pitcher x type x game: n, smoothed mean + sd
  snapshots_<range>.parquet     per as-of date x pitcher x type: filtered estimate
                                + sd for every feature, outcomes, raw window counts

Usage:  python -m mlbpitch.timeseries
"""
from __future__ import annotations

from . import paths, profiles

SEASONS = profiles.SEASONS
PITCH_TYPES = profiles.PITCH_TYPES
MEAN_FEATURES = [c for b in profiles.MEAN_FEATURES.values() for c in b]
USAGE = {"usage": None, "usage_ahead": "ahead", "usage_behind": "behind",
         "usage_vs_lhb": "vs_lhb", "usage_vs_rhb": "vs_rhb"}
CONSISTENCY = {"release_sd": ["release_pos_x", "release_pos_z"],
               "movement_sd": ["pfx_x", "pfx_z"],
               "velo_sd": ["release_speed"]}
OUTCOMES = {"whiff_pct": ("whiff", "swing"), "csw_pct": ("csw", "n"),
            "rv100": ("rv", "n"), "xwoba_contact": ("xwoba", "bip")}
MIN_GAMES_FIT = 15          # series used for the pooled hyperparameter fit
MAX_FIT_SERIES = 600        # cap per (type, feature) fit, chosen deterministically
ACTIVE_DAYS = 45
SNAPSHOT_MONTHS = ["04-15", "05-15", "06-15", "07-15", "08-15", "09-15"]


def log(msg):
    print(msg, flush=True)


def _path(name):
    return paths.META / f"{name}_{min(SEASONS)}_{max(SEASONS)}.parquet"


# ---------------------------------------------------------------------------
# Game-level aggregation
# ---------------------------------------------------------------------------

def game_tables(seasons=SEASONS):
    """(type_games, pitcher_games). type_games: one row per pitcher x type x game
    with n, per-feature game means and within-game variances, outcome sums and
    situational counts. pitcher_games: totals per pitcher x game (usage denominators)."""
    import numpy as np
    import pandas as pd

    tg, pg = [], []
    for season in seasons:
        df = profiles.load_season(season)
        df["game_date"] = pd.to_datetime(df.game_date)
        df["ahead"] = (df.strikes > df.balls).astype(float)
        df["behind"] = (df.balls > df.strikes).astype(float)
        df["vs_lhb"] = (df.stand == "L").astype(float)
        df["vs_rhb"] = (df.stand == "R").astype(float)
        df["rv"] = df.delta_pitcher_run_exp.astype(float)
        df["xwoba"] = df.estimated_woba_using_speedangle.astype(float)
        df["bip"] = df.xwoba.notna().astype(float)
        df["rv2"], df["xwoba2"] = df.rv ** 2, df.xwoba ** 2
        key = ["pitcher", "pitch_type", "game_date"]
        g = df.groupby(key, observed=True)
        t = g.size().rename("n").to_frame()
        t = t.join(g[MEAN_FEATURES].mean())
        # within-game variance of every mean feature: pooled sigma2 for the mean
        # series, and the observations for the consistency series
        t = t.join(g[MEAN_FEATURES].var().add_suffix("_wvar"))
        t = t.join(g[["whiff", "swing", "csw", "rv", "bip", "ahead", "behind",
                      "vs_lhb", "vs_rhb"]].sum())
        t["xwoba"] = g.xwoba.sum(min_count=1)
        t["rv2"], t["xwoba2"] = g.rv2.sum(), g.xwoba2.sum(min_count=1)
        t["player_name"] = g.player_name.first()
        t["p_throws"] = g.p_throws.first()
        t["season"] = season
        tg.append(t.reset_index())

        h = df.groupby(["pitcher", "game_date"], observed=True)
        p = h.size().rename("N").to_frame()
        p = p.join(h[["ahead", "behind", "vs_lhb", "vs_rhb"]].sum())
        p["season"] = season
        pg.append(p.reset_index())
        log(f"  {season}: {len(t):,} pitcher-type-games, {len(p):,} appearances")
        del df
    return pd.concat(tg, ignore_index=True), pd.concat(pg, ignore_index=True)


# ---------------------------------------------------------------------------
# Vectorised local-level Kalman filter / smoother over many series at once
# ---------------------------------------------------------------------------

class Panel:
    """Left-aligned padded arrays for S series of up to T observations."""

    def __init__(self, frame, sid, y, r, share_n=None):
        import numpy as np

        # share_n: binomial denominators -- the observation variance is then
        # computed inside the filter from the PREDICTED share, not from r
        self.share = share_n is not None
        frame = frame.assign(_y=y, _r=r, _N=share_n if self.share else 1.0)
        frame = frame.dropna(subset=["_y", "_r"])
        frame = frame[np.isfinite(frame._r) & (frame._r > 0)]
        frame = frame.sort_values([*sid, "game_date"])
        self.keys = frame[sid].drop_duplicates().reset_index(drop=True)
        codes = frame.groupby(sid, sort=False, observed=True).ngroup().to_numpy()
        pos = frame.groupby(sid, sort=False, observed=True).cumcount().to_numpy()
        S, T = len(self.keys), pos.max() + 1 if len(pos) else 0
        self.S, self.T = S, T
        self.y = np.zeros((S, T)); self.r = np.ones((S, T)); self.m = np.zeros((S, T), bool)
        self.N = np.ones((S, T))
        self.day = np.zeros((S, T), "datetime64[D]")
        self.season = np.zeros((S, T), int)
        self.y[codes, pos] = frame._y.to_numpy()
        self.r[codes, pos] = frame._r.to_numpy()
        self.N[codes, pos] = frame._N.to_numpy()
        self.m[codes, pos] = True
        self.day[codes, pos] = frame.game_date.to_numpy().astype("datetime64[D]")
        self.season[codes, pos] = frame.season.to_numpy()
        dd = np.diff(self.day, axis=1).astype(float)
        self.dt = np.concatenate([np.zeros((S, 1)), np.maximum(dd, 1.0)], axis=1)
        # number of offseasons crossed (2 for a pitcher who skipped a season)
        self.new = np.concatenate([np.zeros((S, 1), int),
                                   np.clip(np.diff(self.season, axis=1), 0, None)], axis=1)
        self.nobs = self.m.sum(axis=1)
        self.code, self.pos = codes, pos
        self.set_prior()

    def set_prior(self):
        """Population prior for each series' first game: a0 = precision-weighted
        mean of every observation, P0 = between-series variance of series means
        net of their sampling variance. Without it a 1-game row was its raw value."""
        import numpy as np

        w = np.where(self.m, 1.0 / self.r, 0.0)
        tot = w.sum(axis=1)
        ok = (self.nobs >= 5) & (tot > 0)
        if not ok.any():
            ok = tot > 0
        mean_s = (w * self.y).sum(axis=1)[ok] / tot[ok]
        self.a0 = float(np.average(mean_s))
        self.P0 = float(max(np.var(mean_s) - np.mean(1.0 / tot[ok]), 0.05 * np.var(mean_s), 1e-12))

    def subset(self, rows):
        import copy
        sub = copy.copy(self)
        for a in ("y", "r", "m", "N", "day", "season", "dt", "new", "nobs"):
            setattr(sub, a, getattr(self, a)[rows])
        sub.S = len(rows)
        return sub


def kalman(P, q_day, q_off, tau2, smooth=False):
    """Filter (and optionally RTS-smooth) every series in Panel P at once.
    Returns loglik and, per (s, t): filtered mean/var (and smoothed mean/var)."""
    import numpy as np

    S, T = P.S, P.T
    af, pf, ap, pp = (np.zeros((S, T)) for _ in range(4))
    a = np.full(S, P.a0)
    p = np.full(S, P.P0)
    ll = 0.0
    for t in range(T):
        m = P.m[:, t]
        if t > 0:
            q = np.where(P.new[:, t] > 0, q_off * P.new[:, t], q_day * P.dt[:, t])
            p = np.where(m, p + q, p)
        ap[:, t], pp[:, t] = a, p
        if P.share:
            pr = np.clip(a, 0.02, 0.98)
            R = pr * (1 - pr) / P.N[:, t] + tau2
        else:
            R = P.r[:, t] + tau2
        Sv = p + R
        v = P.y[:, t] - a
        ll += float(np.sum(np.where(m, -0.5 * (np.log(2 * np.pi * Sv) + v * v / Sv), 0.0)))
        K = np.where(m, p / Sv, 0.0)
        a = a + K * v
        p = p * R / Sv * m + p * (1 - m)   # (1-K)p without the cancellation
        af[:, t], pf[:, t] = a, p
    if not smooth:
        return ll, af, pf, None, None
    as_, ps_ = af.copy(), pf.copy()
    for t in range(T - 2, -1, -1):
        nxt = P.m[:, t + 1]
        J = np.where(nxt, pf[:, t] / np.maximum(pp[:, t + 1], 1e-12), 0.0)
        as_[:, t] = np.where(nxt, af[:, t] + J * (as_[:, t + 1] - ap[:, t + 1]), af[:, t])
        ps_[:, t] = np.where(nxt, pf[:, t] + J * J * (ps_[:, t + 1] - pp[:, t + 1]), pf[:, t])
    return ll, af, pf, as_, ps_


def fit_hyper(P, scale):
    """Pooled ML for (q_day, q_off, tau2). `scale` = typical observation variance,
    used for starting values. Fit on up to MAX_FIT_SERIES of the longest series."""
    import numpy as np
    from scipy.optimize import minimize

    rows = np.where(P.nobs >= MIN_GAMES_FIT)[0]
    if len(rows) < 20:
        rows = np.argsort(-P.nobs)[:max(20, len(rows))]
    rows = rows[np.argsort(-P.nobs[rows], kind="stable")][:MAX_FIT_SERIES]
    sub = P.subset(rows)
    sub_T = int(sub.nobs.max())
    for a in ("y", "r", "m", "N", "dt", "new"):
        setattr(sub, a, getattr(sub, a)[:, :sub_T])
    sub.T = sub_T

    def nll(th):
        q_day, q_off, tau2 = np.exp(th)
        return -kalman(sub, q_day, q_off, tau2)[0]

    # Nelder-Mead stalls on this surface when one variance heads for zero: FF
    # velocity once stopped at an offseason jump of 0.02 mph (true 0.89, +900
    # log-lik) and FF release height at 0.001 ft even from two starts (true
    # 0.107 ft, +737). So: three starts (one with a large offseason jump), then
    # restart from the best point -- a fresh simplex -- until it stops improving.
    def run(x0):
        return minimize(nll, x0, method="Nelder-Mead",
                        options={"xatol": 0.01, "fatol": 0.1, "maxiter": 600})

    best = min((run(x0) for x0 in (np.log([scale / 300, scale / 3, scale / 10]),
                                   np.log([scale / 1000, scale, scale / 5]),
                                   np.log([scale / 100, 10 * scale, scale / 3]))),
               key=lambda r: r.fun)
    for _ in range(3):
        again = run(best.x)
        if again.fun > best.fun - 0.5:
            break
        best = again
    return (*np.exp(best.x), len(rows), bool(best.success))


# ---------------------------------------------------------------------------
# Series definitions
# ---------------------------------------------------------------------------

def feature_series(tg, pg):
    """Yield (feature, pitch_type, frame, y, obs_kind, n, family) per series family.
    frame carries pitcher, pitch_type, game_date, season."""
    import numpy as np

    base = ["pitcher", "pitch_type", "game_date", "season"]
    # 1. per-pitch means: sigma2 = pooled within-game variance per pitch type
    for f in MEAN_FEATURES:
        for pt, d in tg.groupby("pitch_type"):
            yield f, pt, d[base], d[f].to_numpy(float), None, d.n.to_numpy(float), "mean"
    # 2. usage shares over EVERY appearance (0 when the type wasn't thrown)
    for f, sit in USAGE.items():
        denom = "N" if sit is None else sit
        for pt in PITCH_TYPES:
            # numerator renamed: pg has the same situational column names
            num = tg[tg.pitch_type == pt][["pitcher", "game_date", "n" if sit is None else sit]]
            num = num.set_axis(["pitcher", "game_date", "_num"], axis=1)
            d = pg.merge(num, on=["pitcher", "game_date"], how="left").fillna({"_num": 0})
            d = d[d[denom] > 0].assign(pitch_type=pt)
            # only pitchers who ever threw the type (otherwise every series is all-zero)
            d = d[d.groupby("pitcher")._num.transform("sum") > 0]
            share = d._num.to_numpy(float) / d[denom].to_numpy(float)
            yield f, pt, d[base], share, "share", d[denom].to_numpy(float), "usage"
    # 3. within-game spread (n >= 3), on the LOG-variance scale: log s^2 is
    # bias-corrected to E = log sigma^2 and has variance trigamma((n-1)/2),
    # independent of the level and far less sensitive to heavy tails than s^2
    # itself (one 2024 position-player game had a within-game velo variance of 127).
    from scipy.special import digamma
    for f, cols in CONSISTENCY.items():
        for pt, d in tg[tg.n >= 3].groupby("pitch_type"):
            v = sum(d[f"{c}_wvar"].to_numpy(float) for c in cols)
            ok = v > 0
            dd, v = d[ok], v[ok]
            h = (dd.n.to_numpy(float) - 1) / 2
            y = np.log(v) - digamma(h) + np.log(h)
            yield f, pt, dd[base], y, "logvar", dd.n.to_numpy(float), "consistency"
    # 4. outcomes, as rates over their own denominators
    for f, (num, den) in OUTCOMES.items():
        for pt, d in tg.groupby("pitch_type"):
            dd = d[d[den] > 0]
            y = dd[num].to_numpy(float) / dd[den].to_numpy(float)
            if f == "rv100":
                y = 100 * y
            yield f, pt, dd[base], y, None, dd[den].to_numpy(float), "outcome"


def obs_variance(y, n, kind, frame, pooled=None):
    """Known sampling variance of each game's value."""
    import numpy as np

    if kind == "share":
        # placeholder only: Panel(share_n=...) recomputes p(1-p)/N inside the
        # filter from the predicted share (no whole-series, future-using mean)
        return 0.25 / n
    if kind == "logvar":
        from scipy.special import polygamma
        return polygamma(1, (n - 1) / 2)
    return pooled / n


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def snapshot_dates(seasons=SEASONS):
    import pandas as pd

    out = []
    for s in seasons:
        sched = pd.read_parquet(paths.schedule_path(s))
        end = pd.to_datetime(sched.game_date).max()
        for md in SNAPSHOT_MONTHS:
            out.append((s, pd.Timestamp(f"{s}-{md}")))
        out.append((s, end))
    return out


def build(seasons=SEASONS):
    import numpy as np
    import pandas as pd

    log("game tables")
    tg, pg = game_tables(seasons)
    # Known per-observation variances: pooled pitch-to-pitch variance within a
    # game for each (type, mean feature); Bernoulli / dispersion for outcomes.
    pv = {}
    for pt, d in tg.groupby("pitch_type"):
        w = (d.n - 1).clip(lower=0).to_numpy(float)
        for f in MEAN_FEATURES:
            v = d[f"{f}_wvar"].to_numpy(float)
            ok = np.isfinite(v) & (w > 0)
            pv[(pt, f)] = float((v[ok] * w[ok]).sum() / w[ok].sum())
        p_whiff = d.whiff.sum() / d.swing.sum()
        p_csw = d.csw.sum() / d.n.sum()
        pv[(pt, "whiff_pct")] = p_whiff * (1 - p_whiff)
        pv[(pt, "csw_pct")] = p_csw * (1 - p_csw)
        # run value per pitch: E[rv^2] - E[rv]^2 isn't stored; use game-level
        # dispersion: Var(game mean) ~ sigma2 / n  =>  sigma2 ~ median(n * dev^2)
        # per-pitch variances from sums of squares (the earlier median of
        # n * dev^2 understated these 2-4x, so tau2 absorbed the gap)
        N, B = d.n.sum(), d.bip.sum()
        pv[(pt, "rv100")] = 1e4 * float(d.rv2.sum() / N - (d.rv.sum() / N) ** 2)
        pv[(pt, "xwoba_contact")] = float(d.xwoba2.sum() / B - (d.xwoba.sum() / B) ** 2)

    hyper, game_out, snaps = [], [], []
    dates = snapshot_dates(seasons)
    for f, pt, frame, y, kind, n, family in feature_series(tg, pg):
        r = obs_variance(y, n, kind, frame, pooled=pv.get((pt, f)))
        P = Panel(frame.reset_index(drop=True), ["pitcher", "pitch_type"], y, r,
                  share_n=n if kind == "share" else None)
        if P.S == 0:
            continue
        scale = float(np.nanmedian(r[np.isfinite(r)]))
        q_day, q_off, tau2, n_fit, ok = fit_hyper(P, scale)
        hyper.append(dict(pitch_type=pt, feature=f, family=family, sigma2=pv.get((pt, f)),
                          q_day=q_day, q_off=q_off, tau2=tau2, n_series=n_fit, converged=ok))
        _, af, pf, as_, ps_ = kalman(P, q_day, q_off, tau2, smooth=True)

        # per-game smoothed values (description)
        s_idx, t_idx = np.nonzero(P.m)
        gk = P.keys.iloc[s_idx].reset_index(drop=True)
        gk["game_date"] = P.day[s_idx, t_idx].astype("datetime64[ns]")
        gk[f] = as_[s_idx, t_idx]
        gk[f"{f}_sd"] = np.sqrt(np.maximum(ps_[s_idx, t_idx], 0))
        if family == "consistency":   # log-variance -> SD in natural units
            gk[f] = np.sqrt(np.exp(gk[f]))
        game_out.append(gk.set_index(["pitcher", "pitch_type", "game_date"]))

        # as-of snapshots from FILTERED states
        rows = []
        for season, D in dates:
            Dd = np.datetime64(D.date())
            ok_t = P.m & (P.day <= Dd) & (P.season == season)
            if not ok_t.any():
                continue
            # index of last observation on/before D (within D's season)
            idx = np.where(ok_t, np.arange(P.T)[None, :], -1).max(axis=1)
            has = idx >= 0
            s = np.where(has)[0]
            li = idx[s]
            gap = (Dd - P.day[s, li]).astype(float)
            act = gap <= ACTIVE_DAYS
            s, li, gap = s[act], li[act], gap[act]
            mean = af[s, li]
            var = pf[s, li] + q_day * gap
            k = P.keys.iloc[s].reset_index(drop=True)
            k["asof"] = D
            k["season"] = season
            k[f] = np.sqrt(np.exp(mean)) if family == "consistency" else mean
            k[f"{f}_sd"] = np.sqrt(var)   # consistency: sd on the log-variance scale
            rows.append(k)
        if rows:
            snaps.append(pd.concat(rows).set_index(["asof", "season", "pitcher", "pitch_type"]))
        log(f"  {pt} {f:<20} {family:<11} series {P.S:>5} | fit on {n_fit:>3} | "
            f"tau {np.sqrt(tau2):.3g} drift/30d {np.sqrt(30 * q_day):.3g} off {np.sqrt(q_off):.3g}"
            f"{'' if ok else '  (not converged)'}")

    log("assembling")
    hyper = pd.DataFrame(hyper)
    hyper.to_parquet(_path("ts_hyper"), index=False)

    def collapse(parts):
        out = {}
        for p in parts:
            for c in p.columns:
                out.setdefault(c, []).append(p[c])
        return pd.concat({c: pd.concat(v) for c, v in out.items()}, axis=1)

    games = collapse(game_out).reset_index()
    meta = tg[["pitcher", "pitch_type", "game_date", "n", "player_name", "p_throws", "season"]]
    games = meta.merge(games, on=["pitcher", "pitch_type", "game_date"], how="left")
    games.astype({c: "float32" for c in games.columns if games[c].dtype == "float64"}) \
         .to_parquet(_path("ts_games"), index=False)

    snap = collapse(snaps).reset_index()
    # a snapshot row needs the pitch-type means (type thrown recently), not just usage
    snap = snap[snap.release_speed.notna()].copy()
    # identity + raw counts: pitches in the trailing 30 days, and the NEXT window
    # (used for prospective validation -- never as features)
    names = tg.groupby("pitcher").agg(player_name=("player_name", "last"), p_throws=("p_throws", "last"))
    snap = snap.join(names, on="pitcher")
    snap = add_windows(snap, tg, dates)
    snap = snap.sort_values(["asof", "pitch_type", "pitcher"]).reset_index(drop=True)
    snap.to_parquet(_path("snapshots"), index=False)
    log(f"snapshots: {len(snap):,} rows over {snap['asof'].nunique()} dates; "
        f"games: {len(games):,}; hyper: {len(hyper)} fits "
        f"({(~hyper.converged).sum()} not converged)")
    return hyper, games, snap


def add_windows(snap, tg, dates):
    """Raw counts in the 30 days up to each snapshot ('_win') and in the window up
    to the NEXT snapshot of the same season ('_next') -- for display and for
    leak-free prospective checks."""
    import numpy as np
    import pandas as pd

    by_season = {}
    for s, D in dates:
        by_season.setdefault(s, []).append(D)
    cols = ["n", "whiff", "swing", "csw", "rv", "bip", "xwoba"]
    out = []
    for s, Ds in by_season.items():
        Ds = sorted(Ds)
        for i, D in enumerate(Ds):
            lo = D - pd.Timedelta(days=30)
            w = tg[(tg.season == s) & (tg.game_date > lo) & (tg.game_date <= D)]
            a = w.groupby(["pitcher", "pitch_type"])[cols].sum(min_count=1).add_suffix("_win")
            if i + 1 < len(Ds):
                nx = tg[(tg.season == s) & (tg.game_date > D) & (tg.game_date <= Ds[i + 1])]
                b = nx.groupby(["pitcher", "pitch_type"])[cols].sum(min_count=1).add_suffix("_next")
                a = a.join(b, how="outer")
            a["asof"] = D
            out.append(a.reset_index())
    win = pd.concat(out, ignore_index=True)
    return snap.merge(win, on=["asof", "pitcher", "pitch_type"], how="left")


if __name__ == "__main__":
    build()
