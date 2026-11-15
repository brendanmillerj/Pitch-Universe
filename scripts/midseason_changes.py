"""Midseason changes: league-wide changepoint scan of every pitch, within each season.

The season-to-season Evolution view can't tell a June change from a winter one. This scan follows each
pitcher x season x pitch type game by game and finds the dates where a measure jumped and stayed jumped.

Per game appearance the measure is the mean of that pitch type's pitches. Each game is a noisy reading:

    var(game mean) = sigma2 / n + tau2

sigma2 is the pitch-to-pitch variance inside a game (measured directly, pooled per pitch type); tau2 is the
game-to-game wobble that doesn't shrink with pitch count (park, weather, tracking calibration, the day's feel).
Gradual drift q_day per day between games is allowed for too. All three are pooled per level x pitch type x
measure (robust median fits on consecutive-game differences), so no pitcher tunes his own smoothing; the noise
model is fitted per level because the Triple-A tracking is its own instrument, but the yardstick that ranks a
change (below) is MLB's.

Changepoints: PELT optimal partitioning of the game means into constant-mean segments, weighted by
1 / (sigma2/n + tau2 + q_day * DRIFT_DAYS), with a conservative BIC-style penalty and at least MIN_SEG games
per segment -- the same detector as scripts/archive/time_prototype.py, run on everybody.

Measures: the Evolution reasons (velo, iVB, HB, spin, arm angle, release height and side, extension,
location-adjusted VAA) plus usage (share of the pitcher's pitches each game, games without the pitch counted as 0,
so a pitch added or shelved midseason is a usage change). Spin direction is left out: Savant infers it from the
movement, so it adds nothing iVB and HB don't, and on a pitch that barely moves (gyro sliders, Germán Márquez's
knuckle curve at ~0.1 ft of break) it swings 50-225 degrees game to game and topped the first run's list.
A usage drop in one pitch and a rise in another within EVENT_DAYS is one swap ("Sweeper -> slider").

Size of a change = |after - before| / how much the same MLB pitch usually moves between seasons (robust SD, per
pitch type, as the Evolution reasons use). A size of 1 is a typical full year's change made inside a season.
Changes to one pitch within EVENT_DAYS of each other are one event (a slot drop moves arm angle and release
height together); events are ranked by their largest change.

Output:
  data/meta/midseason_changes_<seasons>.parquet        MLB events (one row per event)
  data/aaa/meta/midseason_changes_<seasons>.parquet    Triple-A events
  ..._games parquet beside each: the game means of each event's lead measure for that pitcher-season (charts)
  analysis/midseason_changes.csv                       both levels, for reading

Usage:  python scripts/midseason_changes.py [--level mlb|aaa|both]
"""
import argparse

import numpy as np
import pandas as pd

from mlbpitch import paths, profiles

SEASONS = profiles.SEASONS
MIN_SEG = 5            # games per segment
PEN_MULT = 4.0         # penalty per changepoint = PEN_MULT * log(n games)
DRIFT_DAYS = 30        # a segment tolerates this much random-walk drift
EVENT_DAYS = 7         # changes to one pitch this close together are one event
MIN_SIZE = 1.0         # the scan keeps a change of at least this many typical year-to-year moves (finish() re-sizes)
MIN_GAMES_TYPE = 12    # a pitcher-season-type needs this many games with the pitch (usage: games overall)
MIN_PITCHES_TYPE = 100
GAP_DAYS = 20          # a gap this long around the change date is flagged (injured list, demotion, ...)
LEVELS = ("mlb", "aaa")
CSV = paths.ROOT / "analysis" / "midseason_changes.csv"

# key, source column (mirrored profile frame), unit, decimals, words (up, down) -- as atlas_core.CHANGE_WORDS
MEASURES = [
    ("velo", "release_speed", 1, "mph", 1, ("harder", "softer")),
    ("ivb", "pfx_z", 12, "in", 1, ("more ride", "less ride")),
    ("hb", "pfx_x", -12, "in", 1, ("more arm-side run", "more glove-side")),
    ("spin", "release_spin_rate", 1, "rpm", 0, ("more spin", "less spin")),
    ("arm_angle", "arm_angle", 1, "°", 1, ("higher arm angle", "lower arm angle")),
    ("rel_height", "release_pos_z", 1, "ft", 2, ("higher release", "lower release")),
    ("rel_side", "release_pos_x", -1, "ft", 2, ("wider release", "narrower release")),
    ("extension", "release_extension", 1, "ft", 2, ("more extension", "less extension")),
    ("vaa", "vaa_above_avg", 1, "°", 2, ("flatter approach", "steeper approach")),
]
USAGE = ("usage", None, 100, "%", 0, ("used more", "used less"))
ALL_MEASURES = MEASURES + [USAGE]
NAME = {m[0]: m for m in ALL_MEASURES}
PTN_NAME = {"FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper", "CU": "Curveball",
            "CH": "Changeup", "FS": "Splitter"}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------- data

def game_table(season, level):
    """Per pitcher x pitch type x game date: n, mean and within-game variance of every measure, plus the
    pitcher's total pitches that day (usage)."""
    df = profiles.load_season(season, by_half=False, level=level)
    df = df[["game_date", "pitcher", "player_name", "p_throws", "pitch_type"] +
            [m[1] for m in MEASURES]].copy()
    df["game_date"] = pd.to_datetime(df.game_date)
    for key, col, mult, *_ in MEASURES:
        df[key] = df[col].astype("float64") * mult
    k = ["pitcher", "pitch_type"]

    tot = df.groupby(["pitcher", "game_date"]).size().rename("n_all")
    g = df.groupby(k + ["game_date"], observed=True)
    out = g.size().rename("n").to_frame()
    keys = [m[0] for m in MEASURES]
    out = out.join(g[keys].mean()).join(g[keys].var().add_suffix("_var"))
    out = out.reset_index().join(tot, on=["pitcher", "game_date"])
    names = df.groupby("pitcher").agg(player_name=("player_name", "first"), p_throws=("p_throws", "first"))
    out["season"] = season
    return out, tot.reset_index(), names


def usage_table(games, totals, season):
    """Per pitcher x type x every game the pitcher pitched: share of his pitches (0 when he didn't throw it)."""
    pt = games.groupby(["pitcher", "pitch_type"]).n.sum()
    pt = pt[pt >= MIN_PITCHES_TYPE].reset_index()[["pitcher", "pitch_type"]]
    u = totals.merge(pt, on="pitcher").merge(games[["pitcher", "pitch_type", "game_date", "n"]],
                                             on=["pitcher", "pitch_type", "game_date"], how="left")
    u["n"] = u.n.fillna(0)
    u["usage"] = 100 * u.n / u.n_all
    u["season"] = season
    return u


# ---------------------------------------------------------------------------- noise model

def _scaled_median_var(d2, base):
    """V such that median(d2 / (base + V)) = 0.4549 (median of chi-square 1): a robust variance."""
    lo, hi = 0.0, max(float(np.median(d2)) * 10, 1e-9)
    if np.median(d2 / base) <= 0.4549:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.median(d2 / (base + mid)) > 0.4549:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_noise(series_frames, value, nfun):
    """Pooled (sigma2-free) noise: from consecutive games of the same pitcher-season-type,
    var(y_t - y_{t-1}) = r_t + r_{t-1} + q_day * dt with r = sampling part + tau2. Robust variance per
    days-apart bin, then tau2 and q_day by weighted least squares through the bins."""
    d2, base, dt = [], [], []
    for d in series_frames:
        y = d[value].to_numpy()
        s = nfun(d)
        dd = d.game_date.diff().dt.days.to_numpy()[1:]
        d2.append(np.diff(y) ** 2)
        base.append(s[1:] + s[:-1])
        dt.append(dd)
    d2, base, dt = (np.concatenate(v) for v in (d2, base, dt))
    ok = np.isfinite(d2) & np.isfinite(base)
    d2, base, dt = d2[ok], base[ok], dt[ok]
    bins = [(1, 3), (4, 6), (7, 10), (11, 20)]
    xs, vs, ws = [], [], []
    for a, b in bins:
        m = (dt >= a) & (dt <= b)
        if m.sum() < 50:
            continue
        xs.append(dt[m].mean())
        vs.append(_scaled_median_var(d2[m], base[m]))
        ws.append(m.sum())
    if not xs:
        return 0.0, 0.0
    xs, vs, ws = map(np.asarray, (xs, vs, ws))
    if len(xs) == 1:
        return vs[0] / 2, 0.0
    A = np.column_stack([np.full(len(xs), 2.0), xs]) * np.sqrt(ws)[:, None]
    tau2, q = np.linalg.lstsq(A, vs * np.sqrt(ws), rcond=None)[0]
    if q < 0:
        tau2, q = float(np.average(vs, weights=ws)) / 2, 0.0
    if tau2 < 0:
        tau2 = 0.0
        q = max(0.0, float(np.sum(ws * xs * vs) / np.sum(ws * xs * xs)))
    return float(tau2), float(q)


# ---------------------------------------------------------------------------- changepoints

def pelt(y, w, pen, min_seg):
    """Optimal partitioning of a weighted series into constant-mean segments (time_prototype.pelt, vectorised).
    cost(segment) = sum w (y - weighted mean)^2. Returns segment start indices."""
    T = len(y)
    cw, cwy, cwy2 = (np.r_[0, np.cumsum(v)] for v in (w, w * y, w * y * y))
    F = np.full(T + 1, np.inf)
    F[0] = -pen
    last = np.zeros(T + 1, dtype=int)
    cands = np.array([0])
    for e in range(min_seg, T + 1):
        s = cands[e - cands >= min_seg]
        if len(s):
            W = cw[e] - cw[s]
            cost = (cwy2[e] - cwy2[s]) - (cwy[e] - cwy[s]) ** 2 / W
            v = F[s] + cost + pen
            i = int(np.argmin(v))
            F[e], last[e] = v[i], s[i]
            keep = (e - cands < min_seg)
            keep[e - cands >= min_seg] = (F[s] + cost) <= F[e]
            cands = cands[keep]
        if e - min_seg + 1 > 0:
            cands = np.union1d(cands, [e - min_seg + 1])
    cps, e = [], T
    while e > 0:
        s = last[e]
        if s > 0:
            cps.append(int(s))
        e = s
    return sorted(cps)


def scan_series(d, value, r, pen_mult=PEN_MULT):
    """Changepoints of one series. r = per-game variance (sampling + tau2 + drift allowance)."""
    y = d[value].to_numpy(dtype="float64")
    ok = np.isfinite(y) & np.isfinite(r)
    if ok.sum() < 2 * MIN_SEG:
        return []
    d, y, w = d[ok], y[ok], 1.0 / r[ok]
    cps = pelt(y, w, pen_mult * np.log(len(y)), MIN_SEG)
    bounds = [0] + cps + [len(y)]
    days = d.game_date.to_numpy()
    out = []
    for i, c in enumerate(cps):
        a, b = slice(bounds[i], c), slice(c, bounds[i + 2])
        m0, m1 = np.average(y[a], weights=w[a]), np.average(y[b], weights=w[b])
        se = np.sqrt(1 / w[a].sum() + 1 / w[b].sum())
        out.append(dict(date=pd.Timestamp(days[c]), last_before=pd.Timestamp(days[c - 1]),
                        before=m0, after=m1, delta=m1 - m0, t=(m1 - m0) / se,
                        g_before=c - bounds[i], g_after=bounds[i + 2] - c,
                        gap_days=int((days[c] - days[c - 1]) / np.timedelta64(1, "D"))))
    return out


# ---------------------------------------------------------------------------- yardstick

def y2y_scale(seasonal):
    """Robust SD (1.4826 x MAD) of the same MLB pitch's season-to-season change, per pitch type and measure."""
    keys = [m[0] for m in ALL_MEASURES]
    V = seasonal[(seasonal.level == "mlb") & (seasonal.n >= MIN_PITCHES_TYPE)]
    parts = []
    for s in SEASONS[1:]:
        a = V[V.season == s - 1].set_index(["pitcher", "pitch_type"])
        b = V[V.season == s].set_index(["pitcher", "pitch_type"])
        j = a.index.intersection(b.index)
        dd = {}
        for k in keys:
            x0, x1 = a.loc[j, k].to_numpy(), b.loc[j, k].to_numpy()
            dd[k] = x1 - x0
        parts.append(pd.DataFrame(dd).assign(pitch_type=j.get_level_values("pitch_type")))
    dd = pd.concat(parts, ignore_index=True)
    mad = lambda x: 1.4826 * float((x - x.median()).abs().median())
    return dd.groupby("pitch_type")[keys].agg(mad)


# ---------------------------------------------------------------------------- main

def load_level(level):
    G, U, N = [], [], []
    for s in SEASONS:
        log(f"  {level} {s}: loading")
        g, tot, names = game_table(s, level)
        G.append(g)
        U.append(usage_table(g, tot, s))
        N.append(names)
    names = pd.concat(N)
    names = names[~names.index.duplicated(keep="last")]
    return pd.concat(G, ignore_index=True), pd.concat(U, ignore_index=True), names


def seasonal_means(G, U, level):
    keys = [m[0] for m in MEASURES]
    w = G[keys].mul(G.n, axis=0)
    S = w.groupby([G.pitcher, G.season, G.pitch_type]).sum().div(
        G.groupby(["pitcher", "season", "pitch_type"]).n.sum(), axis=0)
    S["n"] = G.groupby(["pitcher", "season", "pitch_type"]).n.sum()
    ug = U.groupby(["pitcher", "season", "pitch_type"])
    S["usage"] = 100 * ug.n.sum() / ug.n_all.sum()
    return S.reset_index().assign(level=level)


def scan_level(level, G, U, scale):
    keys = [m[0] for m in MEASURES]
    G = G.sort_values(["pitcher", "season", "pitch_type", "game_date"])
    U = U.sort_values(["pitcher", "season", "pitch_type", "game_date"])
    gk = ["pitcher", "season", "pitch_type"]
    size = G.groupby(gk).agg(games=("n", "size"), pitches=("n", "sum"))
    good = size[(size.games >= MIN_GAMES_TYPE) & (size.pitches >= MIN_PITCHES_TYPE)].index
    series = {k: d for k, d in G.groupby(gk) if k in good}
    useries = {k: d for k, d in U.groupby(gk) if k in good or len(d) >= MIN_GAMES_TYPE}
    log(f"  {level}: {len(series):,} pitcher-season-types, {len(useries):,} usage series")

    rows = []
    for pt in profiles.PITCH_TYPES:
        sub = {k: d for k, d in series.items() if k[2] == pt}
        if not sub:
            continue
        for key in keys:
            n_all = np.concatenate([d.n.to_numpy() for d in sub.values()])
            v_all = np.concatenate([d[f"{key}_var"].to_numpy() for d in sub.values()])
            ok = (n_all >= 2) & np.isfinite(v_all)
            s2 = float(((n_all[ok] - 1) * v_all[ok]).sum() / (n_all[ok] - 1).sum())
            tau2, q = fit_noise(list(sub.values()), key, lambda d: s2 / d.n.to_numpy())
            for k, d in sub.items():
                r = s2 / d.n.to_numpy() + tau2 + q * DRIFT_DAYS
                for c in scan_series(d, key, r):
                    rows.append(dict(pitcher=k[0], season=k[1], pitch_type=pt, measure=key, **c))
            log(f"    {pt} {key:<10} sigma {np.sqrt(s2):8.3f}  tau {np.sqrt(tau2):7.3f}  "
                f"drift/30d {np.sqrt(q * 30):7.3f}")
        # usage: binomial sampling around the pitcher-season's usage, + game-plan wobble
        usub = {k: d for k, d in useries.items() if k[2] == pt}

        def ufun(d):
            p = np.clip(d.n.sum() / d.n_all.sum(), 0.02, 0.98)
            return 1e4 * p * (1 - p) / d.n_all.to_numpy()
        tau2, q = fit_noise(list(usub.values()), "usage", ufun)
        for k, d in usub.items():
            for c in scan_series(d, "usage", ufun(d) + tau2 + q * DRIFT_DAYS):
                rows.append(dict(pitcher=k[0], season=k[1], pitch_type=pt, measure="usage", **c))
        log(f"    {pt} usage      tau {np.sqrt(tau2):7.3f} pts  drift/30d {np.sqrt(q * 30):7.3f}")

    ch = pd.DataFrame(rows)
    ch["scale"] = [scale.loc[p, m] if p in scale.index else np.nan for p, m in zip(ch.pitch_type, ch.measure)]
    ch["size"] = (ch.delta.abs() / ch.scale).round(3)
    ch = ch[ch["size"] >= MIN_SIZE].copy()
    ch["level"] = level
    return ch


def words(r):
    key, _, _, unit, dec, (up, down) = NAME[r.measure]
    d = r.delta
    amt = f"{abs(d):.{dec}f}{'' if unit in ('°',) else ' '}{unit}"
    if key == "usage":
        return f"{'up' if d > 0 else 'down'} {abs(d):.0f} pts of usage"
    if key == "ivb":
        return f"{amt} {'more' if d > 0 else 'less'} ride"
    if key == "hb":
        return f"{amt} {'more' if d > 0 else 'less'} arm-side break"
    return f"{amt} {up if d > 0 else down}"


def events(ch):
    """Group a pitch's changes within EVENT_DAYS into one event; lead = the largest."""
    out = []
    for (lvl, pid, s, pt), g in ch.sort_values("date").groupby(["level", "pitcher", "season", "pitch_type"]):
        cur = []
        for _, r in g.iterrows():
            if cur and (r.date - cur[0].date).days > EVENT_DAYS:
                out.append(cur)
                cur = []
            cur.append(r)
        out.append(cur)
    rows = []
    for ev in out:
        ev = sorted(ev, key=lambda r: -r["size"])
        lead = ev[0]
        rows.append(dict(
            level=lead.level, pitcher=lead.pitcher, season=lead.season, pitch_type=lead.pitch_type,
            date=min(r.date for r in ev), lead=lead.measure, size=lead["size"],
            before=lead.before, after=lead.after, delta=lead.delta, t=lead.t,
            g_before=lead.g_before, g_after=lead.g_after, gap_days=max(r.gap_days for r in ev),
            measures=",".join(r.measure for r in ev),
            tip="; ".join(words(r) for r in ev),
            detail=[dict(m=r.measure, before=round(float(r.before), 3), after=round(float(r.after), 3),
                         size=float(r["size"])) for r in ev]))
    return pd.DataFrame(rows)


def pair_swaps(ev):
    """A usage-led event in one pitch and an opposite one in another pitch of the same pitcher-season within
    EVENT_DAYS: one swap, kept on the pitch that rose ("Sweeper -> slider"), the fall's numbers in its tip.
    members = every pitch the row covers, [pitch type, before, after] (the page draws each)."""
    ev = ev.copy()
    ev["swap_from"] = None
    mem = {i: [[r.pitch_type, float(r.before), float(r.after)]] for i, r in ev.iterrows()}
    drop = set()
    u = ev[ev.lead == "usage"]
    for (lv, pid, s), g in u.groupby(["level", "pitcher", "season"]):
        ups, downs = g[g.delta > 0], g[g.delta < 0]
        for i, r in ups.sort_values("size", ascending=False).iterrows():
            near = downs[(downs.pitch_type != r.pitch_type) & ((downs.date - r.date).abs().dt.days <= EVENT_DAYS)
                         & ~downs.index.isin(drop)]
            if near.empty:
                continue
            j = (near.delta.abs() - abs(r.delta)).abs().idxmin()
            ev.at[i, "swap_from"] = near.at[j, "pitch_type"]
            ev.at[i, "tip"] = f"{PTN_NAME.get(near.at[j, 'pitch_type'])} {near.at[j, 'tip']}; " \
                              f"{PTN_NAME.get(r.pitch_type)} {r.tip}"
            ev.at[i, "size"] = max(r["size"], near.at[j, "size"])
            mem[i] = [[r.pitch_type, float(r.before), float(r.after)],
                      [near.at[j, "pitch_type"], float(near.at[j, "before"]), float(near.at[j, "after"])]]
            drop.add(j)
    ev["members"] = pd.Series(mem, dtype=object)
    return ev.drop(index=list(drop))


DELIVERY = ("arm_angle", "rel_height", "rel_side", "extension")


def merge_delivery(ev):
    """A delivery change (arm angle, release height or side, extension) moves every pitch: when two or more pitches
    of a pitcher-season lead with the same delivery measure, same direction, within EVENT_DAYS, they are one row,
    kept on the largest ("Delivery: narrower release, 4 pitches"), the others' numbers in its tip and every pitch in
    members ([pitch type, before, after], largest first)."""
    ev = ev.copy()
    if "n_pitches" not in ev:
        ev["n_pitches"] = 1
    mem = dict(ev.members.items())
    drop = set()
    d = ev[ev.lead.isin(DELIVERY) & ~ev.swap_from.map(lambda x: isinstance(x, str))]
    for (lv, pid, s, m), g in d.groupby(["level", "pitcher", "season", "lead"]):
        g = g.sort_values("size", ascending=False)
        for i, r in g.iterrows():
            if i in drop:
                continue
            near = g[(g.index != i) & ~g.index.isin(drop) & (g.pitch_type != r.pitch_type)
                     & ((g.date - r.date).abs().dt.days <= EVENT_DAYS) & (np.sign(g.delta) == np.sign(r.delta))]
            if near.empty:
                continue
            ev.at[i, "n_pitches"] = int(r.n_pitches) + len(near)
            mem[i] = [[x.pitch_type, float(x.before), float(x.after)] for x in [r, *[near.loc[j] for j in near.index]]]
            ev.at[i, "tip"] = "; ".join(f"{PTN_NAME.get(x.pitch_type, x.pitch_type)} {x.tip}"
                                        for x in [r, *[near.loc[j] for j in near.index]])
            drop.update(near.index)
    ev["members"] = pd.Series(mem, dtype=object)
    return ev.drop(index=list(drop))


def lasted(ev, gm, prof):
    """Did it last, and did it settle? This season: the last (up to) 10 games, as a share of the jump kept (k), and
    whether the measure moved again after the change (another changepoint before season end):
      steady to season end    k >= 2/3, no later change: the new level is where he finished
      kept going              k >= 4/3: he moved further the same way
      held, with later shifts k >= 2/3, but it moved again in between
      settled partway back    1/3 <= k < 2/3
      reverted <date>         k < 1/3, dated by the first game of the next level Next season: his next-season mean of the pitch (season profiles) on the same scale -- kept,
    partly kept, back. kept_now / kept_next are the shares (1 = all of the jump, 0 = back where he started)."""
    series = {k: g.sort_values("game_date") for k, g in gm.groupby(["pitcher", "season", "pitch_type", "measure"])}
    col = {"velo": ("release_speed", 1), "ivb": ("pfx_z", 12), "hb": ("pfx_x", -12), "spin": ("release_spin_rate", 1),
           "arm_angle": ("arm_angle", 1), "rel_height": ("release_pos_z", 1), "rel_side": ("release_pos_x", -1),
           "extension": ("release_extension", 1), "vaa": ("vaa_above_avg", 1), "usage": ("usage", 100)}
    prof = prof.set_index(["pitcher", "season", "pitch_type"])
    now, now_w, nxt, nxt_w, rev = [], [], [], [], []
    for r in ev.itertuples():
        g = series[(r.pitcher, r.season, r.pitch_type, r.lead)]
        g = g[np.isfinite(g.value)]
        days, v = g.game_date.to_numpy(), g.value.to_numpy()
        c = int(np.searchsorted(days, np.datetime64(r.date)))
        tail = v[c:][-10:]
        k = (tail.mean() - r.before) / r.delta if len(tail) else np.nan
        e = c + int(r.g_after)
        back = pd.Timestamp(days[e]) if e < len(days) and k < 1 / 3 else pd.NaT
        now.append(round(float(k), 2))
        later = e < len(days)
        now_w.append("kept going" if k >= 4 / 3 else ("held, with later shifts" if later else "steady to season end") if k >= 2 / 3
                     else "settled partway back" if k >= 1 / 3
                     else f"reverted {back:%b} {back.day}" if pd.notna(back) else "reverted")
        rev.append(back)
        key = (r.pitcher, r.season + 1, r.pitch_type)
        cname, mult = col[r.lead]
        if key in prof.index and pd.notna(prof.at[key, cname]):
            k2 = (mult * prof.at[key, cname] - r.before) / r.delta
            nxt.append(round(float(k2), 2))
            nxt_w.append(f"kept in {r.season + 1}" if k2 >= 2 / 3 else f"partly kept in {r.season + 1}" if k2 >= 1 / 3
                         else f"back in {r.season + 1}")
        else:
            nxt.append(np.nan)
            nxt_w.append("" if r.season == max(SEASONS) else f"no {r.season + 1} {PTN_NAME.get(r.pitch_type, r.pitch_type).lower()}")
    return ev.assign(kept_now=now, lasted_now=now_w, reverted=rev, kept_next=nxt, lasted_next=nxt_w)


PLURAL = {"FF": "four-seams", "SI": "sinkers", "FC": "cutters", "SL": "sliders", "ST": "sweepers", "CU": "curveballs",
          "CH": "changeups", "FS": "splitters"}


def reason(ev):
    """Short headline, as the Evolution reasons read: 'Four-seam: lower arm angle, lower release'. Usage reads as counts
    with the shares (user, 2026-11-15: "Four-seam -> sinker" was unclear): 'More four-seams, fewer sinkers',
    'Fewer cutters (29% -> 1%)'."""
    many = lambda t: PLURAL.get(t, t)
    if isinstance(getattr(ev, "swap_from", None), str):
        return f"More {many(ev.pitch_type)}, fewer {many(ev.swap_from)}"
    if ev.lead == "usage" and getattr(ev, "n_pitches", 1) <= 1:
        rest = [x for x in ev.measures.split(",")[1:2] if x != "usage"]
        more = f"{'More' if ev.delta > 0 else 'Fewer'} {many(ev.pitch_type)} ({ev.before:.0f}% → {ev.after:.0f}%)"
        if rest:
            d = next(x for x in ev.detail if x["m"] == rest[0])
            more += f", {NAME[rest[0]][5][0] if d['after'] > d['before'] else NAME[rest[0]][5][1]}"
        return more
    ms = ev.measures.split(",")[:2]
    bits = []
    for m in ms:
        d = next(x for x in ev.detail if x["m"] == m)
        up, down = NAME[m][5]
        bits.append(up if d["after"] > d["before"] else down)
    if getattr(ev, "n_pitches", 1) > 1:
        up, down = NAME[ev.lead][5]
        return f"Delivery: {up if ev.delta > 0 else down}"   # the page shows the pitches (swatches, "4 pitches") beside it
    return f"{PTN_NAME.get(ev.pitch_type, ev.pitch_type)}: {', '.join(dict.fromkeys(bits))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["mlb", "aaa", "both"], default="both")
    ap.add_argument("--post-only", action="store_true",
                    help="skip the scan: redo the delivery merge, 'did it last' and the CSV from the saved events")
    a = ap.parse_args()
    levels = LEVELS if a.level == "both" else (a.level,)
    if not a.post_only:
        scan(levels)
    finish(levels)


def scan(levels):
    data = {lv: load_level(lv) for lv in sorted(set(levels) | {"mlb"})}   # MLB always: it sets the yardstick
    scale = y2y_scale(seasonal_means(*data["mlb"][:2], "mlb"))
    log("yardstick (typical MLB year-to-year move):\n" + scale.round(2).to_string())

    for lv in levels:
        G, U, names = data[lv]
        ch = scan_level(lv, G, U, scale)
        ev = events(ch).join(names, on="pitcher")   # raw: one row per pitch; finish() pairs swaps and merges deliveries
        ev = ev.sort_values("size", ascending=False).reset_index(drop=True)
        # game means of each event's lead measure, for the page's step chart
        need = ev[["pitcher", "season", "pitch_type", "lead"]].drop_duplicates()
        gm = []
        for r in need.itertuples():
            src = U if r.lead == "usage" else G
            d = src[(src.pitcher == r.pitcher) & (src.season == r.season) & (src.pitch_type == r.pitch_type)]
            d = d.sort_values("game_date")
            gm.append(pd.DataFrame({"pitcher": r.pitcher, "season": r.season, "pitch_type": r.pitch_type,
                                    "measure": r.lead, "game_date": d.game_date.to_numpy(),
                                    "value": d[r.lead].to_numpy(), "n": d.n.to_numpy()}))
        gm = pd.concat(gm, ignore_index=True) if gm else pd.DataFrame()
        tag = f"{SEASONS[0]}_{SEASONS[-1]}"
        out = paths.meta_dir(lv) / f"midseason_changes_raw_{tag}.parquet"
        ev.assign(detail=ev.detail.map(lambda x: __import__("json").dumps(x))).to_parquet(out, index=False)
        gm.to_parquet(paths.meta_dir(lv) / f"midseason_changes_games_{tag}.parquet", index=False)
        log(f"  {lv}: {len(ch):,} changes -> {len(ev):,} events -> {out}")


def half_scale(tag):
    """The yardstick a change's size is measured in (user, 2026-11-15): how much the same MLB pitch usually moves from the
    first half of a season to the second -- a robust SD (1.4826 x MAD) per pitch type and measure, from the half-season
    type profiles (100+ pitches in each half). About two-thirds of the year-to-year move (four-seam velo 0.53 vs 0.79
    mph), so every change at 2 of these is above the scan's 1-year-to-year-move floor (MIN_SIZE) and no rescan is
    needed to change it."""
    t = pd.read_parquet(paths.meta_dir("mlb") / f"profiles_type_{tag}.parquet")
    t = t[t.n >= MIN_PITCHES_TYPE]
    col = {"velo": ("release_speed", 1), "ivb": ("pfx_z", 12), "hb": ("pfx_x", -12), "spin": ("release_spin_rate", 1),
           "arm_angle": ("arm_angle", 1), "rel_height": ("release_pos_z", 1), "rel_side": ("release_pos_x", -1),
           "extension": ("release_extension", 1), "vaa": ("vaa_above_avg", 1), "usage": ("usage", 100)}
    V = pd.DataFrame({m: mult * t[c].to_numpy() for m, (c, mult) in col.items()})
    V[["pitcher", "season", "half", "pitch_type"]] = t[["pitcher", "season", "half", "pitch_type"]].to_numpy()
    k = ["pitcher", "season", "pitch_type"]
    a, b = V[V.half == 1].set_index(k)[list(col)], V[V.half == 2].set_index(k)[list(col)]
    j = a.index.intersection(b.index)
    d = (b.loc[j] - a.loc[j]).astype(float).reset_index()
    mad = lambda x: 1.4826 * float((x - x.median()).abs().median())
    return d.groupby("pitch_type")[list(col)].agg(mad)


def finish(levels):
    """Post-processing on the raw events the scan saved: swaps paired, one row per delivery change, did it last,
    reasons; writes the page's events (midseason_changes_<tag>.parquet) and the CSV. Rerunnable (--post-only)."""
    import json
    tag = f"{SEASONS[0]}_{SEASONS[-1]}"
    allev = []
    for lv in levels:
        out = paths.meta_dir(lv) / f"midseason_changes_{tag}.parquet"
        ev = pd.read_parquet(paths.meta_dir(lv) / f"midseason_changes_raw_{tag}.parquet")
        ev["detail"] = ev.detail.map(json.loads)
        ev["date"] = pd.to_datetime(ev.date)
        gm = pd.read_parquet(paths.meta_dir(lv) / f"midseason_changes_games_{tag}.parquet")
        prof = pd.read_parquet(paths.meta_dir(lv) / f"profiles_type_season_{tag}.parquet")
        n0 = len(ev)
        hs = half_scale(tag)
        # size = the jump in SDs of the usual first-half-to-second-half move (the raw events carry the year-to-year one)
        ev["size_y2y"] = ev["size"]
        ev["size"] = [round(abs(r.delta) / hs.at[r.pitch_type, r.lead], 3) for r in ev.itertuples()]
        ev = lasted(merge_delivery(pair_swaps(ev)), gm, prof)
        ev["reason"] = [reason(r) for r in ev.itertuples()]
        ev = ev.sort_values("size", ascending=False).reset_index(drop=True)
        ev.assign(detail=ev.detail.map(json.dumps), members=ev.members.map(json.dumps)).to_parquet(out, index=False)
        log(f"  {lv}: {n0:,} -> {len(ev):,} events after pairing swaps and one row per delivery change; "
            + ", ".join(f"{k} {v:.0%}" for k, v in ev[ev["size"] >= 2].lasted_now.str.replace(r" [A-Z][a-z]{2} \d+$", "", regex=True)
                        .value_counts(normalize=True).items()) + " (size 2+)")
        allev.append(ev)

    ev = pd.concat(allev, ignore_index=True)
    cols = ["level", "pitcher", "player_name", "season", "pitch_type", "date", "reason", "tip", "size", "lead",
            "before", "after", "delta", "t", "g_before", "g_after", "gap_days", "lasted_now", "kept_now",
            "lasted_next", "kept_next", "measures"]
    ev.sort_values(["level", "size"], ascending=[False, False])[cols].round(3, ).to_csv(CSV, index=False)
    log(f"wrote {CSV}")
    for lv in levels:
        e = ev[ev.level == lv]
        log(f"\n{lv}: top 25")
        log(e.head(25)[["player_name", "season", "date", "reason", "size", "lasted_now", "lasted_next"]]
            .to_string(index=False, max_colwidth=70))


if __name__ == "__main__":
    main()
