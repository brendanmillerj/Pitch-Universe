"""Goal 2 -- who resembles whom, who is unusual, and where success concentrates.

Inputs: profiles_{type,pitcher}_* and embed_{type,pitcher}_* (scripts/profile_embed.py).
Type level uses the WITHIN-type PCs and only ever compares a pitch with pitches
of the same type ("whose sweeper looks like X's sweeper").

  comps         K_COMPS nearest rows from OTHER pitchers (one row per pitcher:
                their closest half), with distance and percentile
  uniqueness    distance to the K_UNIQ-th such neighbour. Shrinkage pulls
                small-sample rows toward the type mean (= less unique), so the
                rarity-vs-success test controls for log(n)
  success       top quartile of rv100 (within pitch type at type level).
                Milo-style: each row's neighbourhood = its K_NBHD nearest
                DISTINCT other pitchers, so a pitcher's halves can't fake an
                enrichment; binomial test vs the 25% base rate, BH-FDR 10%
  near-misses   2026 2nd-half rows that are not top-quartile themselves but
                whose neighbourhood is significantly enriched -- they already
                look like the successful guys
  block value   GroupKFold-by-pitcher R^2 of gradient boosting on each feature
                block alone, and all-blocks minus each block, for rv100 and
                whiff%. Answers whether VAA/DDZ add anything beyond shape.

Usage:  python scripts/comps.py
"""
import numpy as np
import pandas as pd

from mlbpitch import paths, profiles
from profile_embed import TYPE_BLOCKS, embed_path

SEED = 0
K_COMPS = 10
K_UNIQ = 10
K_NBHD = 30
FDR = 0.10
SUCCESS_Q = 0.75
N_BOOT = 500
LATEST = (2026, 2)
FIG_DIR = paths.ROOT / "analysis"
OUT = paths.META


def log(msg):
    print(msg, flush=True)


def other_pitcher_neighbours(P, pid, k, pool=None):
    """For each row: indices/distances of its k nearest DISTINCT other pitchers.

    `pool` restricts candidates (e.g. same pitch type). A pitcher is represented
    by whichever of their halves is nearest.
    """
    from sklearn.neighbors import NearestNeighbors

    rows = np.arange(len(P)) if pool is None else pool
    nn = NearestNeighbors(n_neighbors=min(len(rows), 8 * k)).fit(P[rows])
    dist, idx = nn.kneighbors(P[rows])
    out_i = np.full((len(rows), k), -1)
    out_d = np.full((len(rows), k), np.nan)
    for r, (di, ii) in enumerate(zip(dist, idx)):
        seen = {pid[rows[r]]}
        j = 0
        for d, i in zip(di, ii):
            p = pid[rows[i]]
            if p in seen:
                continue
            seen.add(p)
            out_i[r, j], out_d[r, j] = rows[i], d
            j += 1
            if j == k:
                break
    return out_i, out_d


def neighbour_tables(meta, P, by_type):
    """kNN of distinct other pitchers for every row (within pitch type if by_type)."""
    pid = meta.pitcher.to_numpy()
    idx = np.full((len(P), K_NBHD), -1)
    dist = np.full((len(P), K_NBHD), np.nan)
    groups = meta.groupby("pitch_type").indices if by_type else {"all": np.arange(len(P))}
    for _, rows in groups.items():
        i, d = other_pitcher_neighbours(P, pid, K_NBHD, pool=rows)
        idx[rows], dist[rows] = i, d
    return idx, dist


def comps_long(meta, idx, dist, label_cols):
    """Long table: one row per (row, comp rank)."""
    rows = []
    for r in range(len(meta)):
        for k in range(K_COMPS):
            j = idx[r, k]
            if j < 0:
                continue
            rows.append((r, k + 1, j, dist[r, k]))
    c = pd.DataFrame(rows, columns=["row", "rank", "comp_row", "distance"])
    left = meta[label_cols].iloc[c.row].reset_index(drop=True)
    right = meta[label_cols].add_prefix("comp_").iloc[c.comp_row].reset_index(drop=True)
    # Distance percentile among all first-comp distances: "how close is close".
    first = np.sort(dist[:, 0][~np.isnan(dist[:, 0])])
    c["pctile"] = np.searchsorted(first, c.distance) / len(first)
    return pd.concat([left, c[["rank", "distance", "pctile"]], right], axis=1)


def success_regions(meta, idx, outcome, by_type):
    from scipy.stats import binom
    from scipy.stats import false_discovery_control

    grp = meta.pitch_type if by_type else pd.Series("all", index=meta.index)
    thr = meta.groupby(grp)[outcome].transform(lambda s: s.quantile(SUCCESS_Q))
    success = (meta[outcome] >= thr).to_numpy()
    valid = (idx >= 0).all(axis=1)
    hits = np.where(valid, success[np.where(idx >= 0, idx, 0)].sum(axis=1), 0)
    p = np.where(valid, binom.sf(hits - 1, K_NBHD, 1 - SUCCESS_Q), 1.0)
    q = false_discovery_control(p, method="bh")
    return success, hits / K_NBHD, q


def uniqueness_test(df, outcome, by_type):
    """OLS outcome ~ z(uniqueness) + log n (+ type FE), pitcher-cluster bootstrap."""
    d = df.dropna(subset=[outcome, "uniqueness"]).reset_index(drop=True)
    u = d.uniqueness
    if by_type:
        u = u / d.groupby("pitch_type").uniqueness.transform("median")
    z = (np.log(u) - np.log(u).mean()) / np.log(u).std()
    cols = [np.ones(len(d)), z, np.log(d.n)]
    if by_type:
        cols += [(d.pitch_type == t).astype(float) for t in profiles.PITCH_TYPES[1:]]
    X = np.column_stack(cols)
    y = d[outcome].to_numpy()
    fit = lambda r: np.linalg.lstsq(X[r], y[r], rcond=None)[0][1]
    beta = fit(np.arange(len(d)))
    groups = d.groupby("pitcher").indices
    pids = np.array(list(groups))
    rng = np.random.default_rng(SEED)
    boots = [fit(np.concatenate([groups[p] for p in rng.choice(pids, len(pids))]))
             for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return beta, lo, hi


def prospective(meta, outcome, by_type):
    """Does the comps-expected outcome at t predict the pitcher's own outcome at
    t+1 beyond their own outcome at t?  OLS y(t+1) ~ y(t) + expected(t) + log n,
    pitcher-cluster bootstrap. Returns (beta, lo, hi, pairs frame)."""
    key = ["pitcher"] + (["pitch_type"] if by_type else [])
    m = meta.assign(t=(meta.season - meta.season.min()) * 2 + meta.half - 1)
    cols = key + ["t", outcome, f"exp_{outcome}", "n"]
    pr = m[cols].merge(m[key + ["t", outcome]].assign(t=lambda d: d.t - 1),
                       on=key + ["t"], suffixes=("", "_next")).dropna()
    X = np.column_stack([np.ones(len(pr)), pr[outcome], pr[f"exp_{outcome}"], np.log(pr.n)])
    y = pr[f"{outcome}_next"].to_numpy()
    fit = lambda r: np.linalg.lstsq(X[r], y[r], rcond=None)[0]
    beta = fit(np.arange(len(pr)))
    groups = pr.groupby("pitcher").indices
    pids = np.array(list(groups))
    rng = np.random.default_rng(SEED)
    boots = np.array([fit(np.concatenate([groups[p] for p in rng.choice(pids, len(pids))]))
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    # Partial residual for plotting: next-half outcome with own level removed.
    pr["resid_next"] = y - X[:, [0, 1, 3]] @ beta[[0, 1, 3]]
    return beta, lo, hi, pr


def block_value(types):
    """CV R^2 by feature block for rv100 and whiff%, grouped by pitcher."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict

    onehot = pd.get_dummies(types.pitch_type, prefix="pt", dtype=float)
    blocks = dict(TYPE_BLOCKS)
    results = []
    for target in ["rv100", "whiff_pct"]:
        d = types.dropna(subset=[target])
        y, groups = d[target].to_numpy(), d.pitcher.to_numpy()

        def r2(cols):
            X = pd.concat([onehot.loc[d.index], d[cols]], axis=1).to_numpy(dtype=float)
            model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                  max_leaf_nodes=15, random_state=SEED)
            pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(5))
            return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

        base = r2([])
        allc = [c for cs in blocks.values() for c in cs]
        full = r2(allc)
        log(f"[{target}] pitch type only R^2 {base:.3f} | all blocks {full:.3f}")
        results.append((target, "pitch type only", "alone", base))
        results.append((target, "all", "alone", full))
        for b, cols in blocks.items():
            alone = r2(cols)
            drop = full - r2([c for c in allc if c not in cols])
            results.append((target, b, "alone", alone))
            results.append((target, b, "drop", drop))
            log(f"  {b:<12} alone {alone:.3f} (+{alone - base:.3f} over type)  "
                f"unique contribution {drop:+.3f}")
    return pd.DataFrame(results, columns=["target", "block", "kind", "r2"])


def run_level(kind, meta, P, by_type, label_cols):
    idx, dist = neighbour_tables(meta, P, by_type)
    meta = meta.copy()
    meta["uniqueness"] = dist[:, K_UNIQ - 1]
    meta["success"], meta["nbhd_success"], meta["nbhd_q"] = success_regions(
        meta, idx, "rv100", by_type)
    n_sig = (meta.nbhd_q < FDR).sum()
    log(f"[{kind}] {n_sig:,} of {len(meta):,} neighbourhoods enriched for top-quartile rv100 "
        f"at FDR {FDR:.0%} (binary Milo-style test -- underpowered for an outcome this noisy)")

    outcomes = ["rv100", "whiff_pct"] if by_type else ["rv100"]
    safe = np.where(idx >= 0, idx, 0)
    pairs = {}
    for outcome in outcomes:
        b, lo, hi = uniqueness_test(meta, outcome, by_type)
        log(f"[{kind}] uniqueness -> {outcome}: {b:+.4f} per SD of log-uniqueness "
            f"(95% CI {lo:+.4f} to {hi:+.4f})")
        # Continuous replacement for the binary test: what the 30 nearest other
        # pitchers' versions of this earn. Never includes the row's own outcome.
        vals = meta[outcome].to_numpy()[safe]
        meta[f"exp_{outcome}"] = np.where(idx >= 0, vals, np.nan).mean(axis=1)
        r = meta[[outcome, f"exp_{outcome}"]].corr().iloc[0, 1]
        beta, lo, hi, pr = prospective(meta, outcome, by_type)
        pairs[outcome] = pr
        log(f"[{kind}] comps-expected {outcome}: r with own = {r:+.3f}; predicts NEXT half "
            f"beyond own level: {beta[2]:+.3f} (95% CI {lo[2]:+.3f} to {hi[2]:+.3f}), "
            f"own level {beta[1]:+.3f}, n = {len(pr):,} pairs")
    comps = comps_long(meta, idx, dist, label_cols)
    comps.to_parquet(OUT / f"comps_{kind}_2024_2026.parquet", index=False)
    meta.to_parquet(OUT / f"uniqueness_{kind}_2024_2026.parquet", index=False)
    log(f"[{kind}] wrote comps ({len(comps):,} rows) and uniqueness tables")

    # Near-misses: latest half, decent sample, design says they should be earning
    # more than they are. Ranked by expected minus actual on the stabler outcome.
    target = "whiff_pct" if by_type else "rv100"
    latest = meta[(meta.season == LATEST[0]) & (meta.half == LATEST[1]) & (meta.n >= 150)].copy()
    latest["gap"] = latest[f"exp_{target}"] - latest[target]
    cols = ["player_name"] + (["pitch_type"] if by_type else []) + ["n", target, f"exp_{target}", "gap"]
    log(f"[{kind}] near-misses {LATEST[0]} H{LATEST[1]} (n>=150, largest expected-minus-actual "
        f"{target}):\n" + latest.sort_values("gap", ascending=False)[cols].head(12)
        .round(3).to_string(index=False))
    return meta, pairs


def plot(type_meta, type_pairs, pitcher_pairs, blocks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figure_style import apply_style, save, CMAP_DIVERGING

    apply_style()
    fig = plt.figure(figsize=(7.09, 5.4), layout="constrained")
    gs = fig.add_gridspec(2, 3)

    # (a) uniqueness vs whiff%, binned within type
    ax = fig.add_subplot(gs[0, 0])
    d = type_meta.dropna(subset=["whiff_pct"]).copy()
    d["u_rel"] = np.log(d.uniqueness / d.groupby("pitch_type").uniqueness.transform("median"))
    d["bin"] = pd.qcut(d.u_rel, 10, labels=False)
    b = d.groupby("bin").agg(u=("u_rel", "mean"), w=("whiff_pct", "mean"),
                             se=("whiff_pct", lambda s: s.std() / np.sqrt(len(s))))
    ax.errorbar(b.u, 100 * b.w, yerr=196 * b.se, fmt="o-", ms=2.5, lw=0.8,
                color="#d73027", capsize=1.5)
    ax.set_xlabel("Uniqueness (log, rel. to type median)")
    ax.set_ylabel("Whiff % (decile mean, 95% CI)")
    ax.set_title("a  Rarer pitches vs whiffs", loc="left")

    # (b) prospective check: does what your comps earn predict your next half?
    for j, (pr, outcome, scale, name) in enumerate([
            (type_pairs["whiff_pct"], "whiff_pct", 100, "Whiff % (pitch type)"),
            (pitcher_pairs["rv100"], "rv100", 1, "RV/100 (pitcher)")]):
        ax = fig.add_subplot(gs[0, 1 + j])
        pr = pr.assign(bin=pd.qcut(pr[f"exp_{outcome}"], 10, labels=False))
        b = pr.groupby("bin").agg(x=(f"exp_{outcome}", "mean"), y=("resid_next", "mean"),
                                  se=("resid_next", lambda s: s.std() / np.sqrt(len(s))))
        ax.errorbar(scale * b.x, scale * b.y, yerr=1.96 * scale * b.se, fmt="o-", ms=2.5,
                    lw=0.8, color="#4575b4", capsize=1.5)
        ax.set_xlabel(f"Comps-expected {name.split(' (')[0]}, half t")
        ax.set_ylabel("Next half, own level removed")
        ax.set_title(f"{'b' if j == 0 else ''}  {name}", loc="left")

    # (c, d) block value
    for j, target in enumerate(["rv100", "whiff_pct"]):
        ax = fig.add_subplot(gs[1, 0] if j == 0 else gs[1, 1:])
        r = blocks[blocks.target == target]
        base = r[(r.block == "pitch type only")].r2.iloc[0]
        alone = r[r.kind == "alone"].set_index("block").r2.drop(["pitch type only", "all"]) - base
        drop = r[r.kind == "drop"].set_index("block").r2
        order = alone.sort_values().index
        y = np.arange(len(order))
        ax.barh(y + 0.2, alone[order], height=0.4, color="#4575b4", label="Block alone (gain over type)")
        ax.barh(y - 0.2, drop[order], height=0.4, color="#d73027", label="Unique (drop-one loss)")
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Cross-validated R²")
        name = "RV/100" if target == "rv100" else "Whiff %"
        ax.set_title(f"{'cd'[j]}  What predicts {name} (type-only R² {base:.2f})", loc="left")
        if j == 1:
            ax.legend(frameon=False, loc="lower right")

    save(fig, FIG_DIR / "comps")


if __name__ == "__main__":
    types = pd.read_parquet(profiles.profiles_path("type"))
    t_emb = pd.read_parquet(embed_path("type"))
    p_emb = pd.read_parquet(embed_path("pitcher"))

    label = profiles.KEYS + ["player_name", "pitch_type", "n", "rv100", "whiff_pct"]
    P_type = t_emb[[c for c in t_emb if c.startswith("within_pc")]].to_numpy()
    t_meta, t_pairs = run_level("type", t_emb[label].reset_index(drop=True), P_type, True, label)

    plabel = profiles.KEYS + ["player_name", "n", "rv100"]
    P_p = p_emb[[c for c in p_emb if c.startswith("pc") and c[2:].isdigit()]].to_numpy()
    _, p_pairs = run_level("pitcher", p_emb[plabel].reset_index(drop=True), P_p, False, plabel)

    blocks = block_value(types)
    blocks.to_parquet(OUT / "block_value_2024_2026.parquet", index=False)
    plot(t_meta, t_pairs, p_pairs, blocks)
