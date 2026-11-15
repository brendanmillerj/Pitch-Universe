"""Comps on the as-of snapshots (mlbpitch.timeseries) + leak-free validation.

Same feature space as the half-season pipeline (profile_embed.TYPE_BLOCKS:
32 features in 6 blocks, each block scaled to equal total variance, z-scored
WITHIN pitch type), but every row is a filtered as-of estimate, so values carry
only information available on that date.

Two comp sets:
  display  pool = every snapshot row of the type (any date). For the cards.
  strict   pool = rows dated ON OR BEFORE the query row. Used for validation and
           for backdated cards. Comp rows and their outcome estimates never
           postdate the query; the scaling/PCA and the state-space
           hyperparameters are fitted on the full sample (measured: refitting
           the embedding on earlier seasons keeps ~95% of comps).

Validation (strict): for each row on date D, do the comps' estimated whiff %
add to predicting the pitcher's RAW whiff % in the window to the next snapshot
-- data the model never saw -- beyond his own estimate AND his pitch type?
Swing-weighted OLS with type fixed effects and a pitcher-cluster bootstrap;
buy-low is scored against the own+type prediction.

Outputs (data/meta): snap_embed_*, snap_comps_* (display), snap_comps_strict_*,
snap_validation_*.

Usage:  python scripts/snapshot_comps.py
"""
import numpy as np
import pandas as pd

from mlbpitch import paths, profiles
from profile_embed import TYPE_BLOCKS, block_scale, zscore

SEED = 0
K = 10
N_CAND = 300
N_PCS = 30
N_BOOT = 300
RANGE = f"{min(profiles.SEASONS)}_{max(profiles.SEASONS)}"
FEATURES = [c for cs in TYPE_BLOCKS.values() for c in cs]
ID = ["asof", "season", "pitcher", "pitch_type"]


def log(msg):
    print(msg, flush=True)


def embed_type(d):
    """Within-type z-score -> block scale -> PCA. Returns PC matrix."""
    from sklearn.decomposition import PCA

    F = d[FEATURES].astype("float64")
    F = F.fillna(F.median())
    X = block_scale(zscore(F), TYPE_BLOCKS)
    n = min(N_PCS, X.shape[1])
    return PCA(n_components=n, random_state=SEED).fit_transform(X)


def distinct_neighbours(P, pid, pool, query, out_i, out_d):
    """K nearest DISTINCT other pitchers for each query row, searching `pool`.
    Each pitcher appears up to ~21 times (once per snapshot), so search
    N_CAND candidates and keep each pitcher's nearest row."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(len(pool), N_CAND)).fit(P[pool])
    dist, idx = nn.kneighbors(P[query])
    for r, q in enumerate(query):
        cand, dd = pool[idx[r]], dist[r]
        keep = pid[cand] != pid[q]
        cand, dd = cand[keep], dd[keep]
        _, first = np.unique(pid[cand], return_index=True)   # nearest row per pitcher
        first = np.sort(first)[:K]
        out_i[q, :len(first)], out_d[q, :len(first)] = cand[first], dd[first]


def comps_for(d, P, strict):
    """Comps for every row of one pitch type. Returns long frame."""
    pid = d.pitcher.to_numpy()
    asof = d["asof"].to_numpy()
    out_i = np.full((len(d), K), -1)
    out_d = np.full((len(d), K), np.nan)
    if not strict:
        everything = np.arange(len(d))
        distinct_neighbours(P, pid, everything, everything, out_i, out_d)
    else:
        for D in np.unique(asof):
            distinct_neighbours(P, pid, np.where(asof <= D)[0], np.where(asof == D)[0],
                                out_i, out_d)
    short = (out_i[:, -1] < 0).sum()
    if short:
        log(f"    {short} rows found fewer than {K} distinct comps")
    first = np.sort(out_d[:, 0][np.isfinite(out_d[:, 0])])
    rows = []
    for r in range(len(d)):
        for k in range(K):
            j = out_i[r, k]
            if j >= 0:
                rows.append((r, k + 1, j, out_d[r, k]))
    c = pd.DataFrame(rows, columns=["row", "rank", "comp_row", "distance"])
    c["dist_pctile"] = np.searchsorted(first, c.distance) / len(first)
    left = d[ID + ["player_name"]].iloc[c.row].reset_index(drop=True)
    right = d[ID + ["player_name", "whiff_pct", "csw_pct", "swing_win", "n_win"]] \
        .iloc[c.comp_row].reset_index(drop=True).add_prefix("comp_")
    return pd.concat([left, c[["rank", "distance", "dist_pctile"]], right], axis=1)


def validate(snap, strict_comps):
    """Prospective test on next-window raw whiff %, net of pitch type.

    Every model carries pitch-type fixed effects: without them most of the comps'
    apparent value was just "splitters whiff more than sinkers" (own + type mean
    explained as much as own + comps). The headline is the comps' INCREMENTAL
    value over own estimate + type. Rows need >= 1 swing in the next window
    (swing weights handle the noise); requiring 20 kept survivors only.
    """
    exp = strict_comps.groupby(ID).comp_whiff_pct.mean().rename("exp_whiff")
    d = snap.join(exp, on=ID)
    d["whiff_next"] = d.whiff_next / d.swing_next
    # the season-end snapshot has no next window in this data; drop it
    not_last = d["asof"] < d.groupby("season")["asof"].transform("max")
    eligible = d[d.exp_whiff.notna() & d.whiff_pct.notna() & not_last]
    d = eligible[eligible.swing_next >= 1].reset_index(drop=True)
    y, w = d.whiff_next.to_numpy(), d.swing_next.to_numpy(float)
    fe = pd.get_dummies(d.pitch_type, dtype=float).to_numpy()      # type FE, no intercept

    def design(cols):
        return np.column_stack([fe] + [d[c].to_numpy(float) for c in cols])

    def fit(X, rows):
        W = np.sqrt(w[rows])[:, None]
        return np.linalg.lstsq(X[rows] * W, y[rows] * W[:, 0], rcond=None)[0]

    def wr2(pred):
        return 1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)

    allr = np.arange(len(d))
    X_type, X_own, X_both = design([]), design(["whiff_pct"]), design(["whiff_pct", "exp_whiff"])
    b_own, b_both = fit(X_own, allr), fit(X_both, allr)
    r2 = {"type": wr2(X_type @ fit(X_type, allr)), "own+type": wr2(X_own @ b_own),
          "own+type+comps": wr2(X_both @ b_both)}

    groups = d.groupby("pitcher").indices
    pids = np.array(list(groups))
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        rows = np.concatenate([groups[p] for p in rng.choice(pids, len(pids))])
        bb, bo = fit(X_both, rows), fit(X_own, rows)
        Wr = w[rows]
        tot = np.sum(Wr * (y[rows] - np.average(y[rows], weights=Wr)) ** 2)
        gain = (np.sum(Wr * (y[rows] - X_own[rows] @ bo) ** 2)
                - np.sum(Wr * (y[rows] - X_both[rows] @ bb) ** 2)) / tot
        boot.append((bb[-1], bb[-2], gain))
    boot = np.array(boot)
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    gain = r2["own+type+comps"] - r2["own+type"]
    log(f"validation (strict comps; next-window raw whiff; type fixed effects; "
        f"n = {len(d):,} rows / {d.pitcher.nunique()} pitchers):")
    log(f"  swing-weighted R^2: type {r2['type']:.3f} | own+type {r2['own+type']:.3f} | "
        f"own+type+comps {r2['own+type+comps']:.3f}")
    log(f"  comps' increment: dR^2 {gain:+.4f} (95% CI {ci[0, 2]:+.4f} to {ci[1, 2]:+.4f}); "
        f"coefficient {b_both[-1]:+.3f} ({ci[0, 0]:+.3f} to {ci[1, 0]:+.3f}); "
        f"own {b_both[-2]:+.3f} ({ci[0, 1]:+.3f} to {ci[1, 1]:+.3f})")
    per_type = {}
    for pt in sorted(d.pitch_type.unique()):
        k = (d.pitch_type == pt).to_numpy()
        A1 = np.column_stack([np.ones(k.sum()), d.whiff_pct[k]])
        A2 = np.column_stack([A1, d.exp_whiff[k]])
        Wk = np.sqrt(w[k])[:, None]
        r1 = y[k] - A1 @ np.linalg.lstsq(A1 * Wk, y[k] * Wk[:, 0], rcond=None)[0]
        r2_ = y[k] - A2 @ np.linalg.lstsq(A2 * Wk, y[k] * Wk[:, 0], rcond=None)[0]
        tot = np.sum(w[k] * (y[k] - np.average(y[k], weights=w[k])) ** 2)
        per_type[pt] = float((np.sum(w[k] * r1 ** 2) - np.sum(w[k] * r2_ ** 2)) / tot)
    log("  within-type dR^2 from comps: " + ", ".join(f"{k} {v:+.3f}" for k, v in per_type.items()))

    # Buy-low, scored against the own+type prediction so that plain regression
    # toward the type mean is NOT counted as a hit.
    resid = y - X_own @ b_own
    gap = (d.exp_whiff - d.whiff_pct).to_numpy()
    top = gap >= np.quantile(gap, 0.9)
    hit = float(np.average(resid[top] > 0, weights=w[top]))
    lift = float(np.average(resid[top], weights=w[top]))
    # survival: did the top-gap rows keep pitching (>= 20 swings next window)?
    eg = (eligible.exp_whiff - eligible.whiff_pct).to_numpy()
    etop = eg >= np.quantile(eg, 0.9)
    surv_top = float((eligible.swing_next.fillna(0).to_numpy()[etop] >= 20).mean())
    surv_all = float((eligible.swing_next.fillna(0) >= 20).mean())
    log(f"  buy-low (top 10% comps-minus-own): beat the own+type prediction "
        f"{100 * hit:.0f}% of the time (50% = no edge), mean +{100 * lift:.1f} pts whiff; "
        f"kept a role (>= 20 swings next window) {100 * surv_top:.0f}% vs {100 * surv_all:.0f}% overall")
    return dict(r2_type=r2["type"], r2_own=r2["own+type"], r2_both=r2["own+type+comps"],
                gain=gain, gain_lo=ci[0, 2], gain_hi=ci[1, 2], coef=b_both[-1],
                coef_lo=ci[0, 0], coef_hi=ci[1, 0], n=len(d), buylow_hit=hit,
                buylow_lift=lift, survival_top=surv_top, survival_all=surv_all,
                **{f"gain_{k}": v for k, v in per_type.items()})


if __name__ == "__main__":
    snap = pd.read_parquet(paths.META / f"snapshots_{RANGE}.parquet")
    log(f"{len(snap):,} snapshot rows, {snap['asof'].nunique()} dates")
    embeds, display, strict = [], [], []
    for pt in profiles.PITCH_TYPES:
        d = snap[snap.pitch_type == pt].reset_index(drop=True)
        P = embed_type(d)
        e = d[ID].copy()
        e[[f"pc{i + 1}" for i in range(P.shape[1])]] = P
        embeds.append(e)
        display.append(comps_for(d, P, strict=False))
        strict.append(comps_for(d, P, strict=True))
        log(f"  {pt}: {len(d):,} rows")
    pd.concat(embeds).to_parquet(paths.META / f"snap_embed_{RANGE}.parquet", index=False)
    display = pd.concat(display, ignore_index=True)
    display.to_parquet(paths.META / f"snap_comps_{RANGE}.parquet", index=False)
    strict = pd.concat(strict, ignore_index=True)
    strict.to_parquet(paths.META / f"snap_comps_strict_{RANGE}.parquet", index=False)
    res = validate(snap, strict)
    pd.DataFrame([res]).to_parquet(paths.META / f"snap_validation_{RANGE}.parquet", index=False)
    log(f"wrote snap_embed / snap_comps ({len(display):,} rows) / snap_validation")
