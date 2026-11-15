"""Goal 1 -- how pitchers change across the six half-season timepoints.

Inputs: profiles_pitcher_* (interpretable features) and embed_pitcher_* (PCs),
both from `mlbpitch.profiles` / `scripts/profile_embed.py`.

One row per transition = a pitcher's adjacent halves (t -> t+1; t runs 0..5 over
2024H1..2026H2, so 2024H2 -> 2025H1 counts as adjacent -- the offseason is where
most deliberate change happens).

  move           |dP| in the pitcher PC space -- how far the pitcher moved
  deltas         interpretable feature changes, standardized by their SD across
                 all transitions (DELTA_FEATURES); usage is untruncated
  events         added / dropped a pitch (<2% -> >=5% usage and back), primary-
                 fastball switch, and any continuous delta beyond 2 SD
  modes          PCA of the standardized deltas. Not clusters: KMeans on these
                 gave silhouette 0.05 at every k (4-10) -- change is continuous,
                 so the honest summary is its principal axes, not archetypes
  nbhd_rv        mean rv100 of the K nearest rows from OTHER pitchers -- what that
                 region of pitch-design space earns, independent of the pitcher

Two regression-to-the-mean traps, both handled:
  * Arrow field. Noise alone makes edge points drift inward from t to t+1 --
    but equally from t+1 back to t. Real drift is time-asymmetric, so the field
    drawn is (forward - time-reversed) / 2, which cancels the symmetric pull.
  * Payoff. OLS d_rv100 ~ d_nbhd_rv + rv100(t) + nbhd_rv(t); both levels at t
    regress, so both are covariates. CIs by pitcher-cluster bootstrap.

Usage:  python scripts/trajectories.py
"""
import numpy as np
import pandas as pd

from mlbpitch import paths, profiles

SEED = 0
K = 15
N_BOOT = 1000
N_MODES = 4
EVENT_SD = 2.0
ADD_FROM, ADD_TO = 0.02, 0.05   # untruncated usage thresholds for add / drop

FB_FEATURES = ["release_speed", "release_spin_rate", "pfx_x", "pfx_z",
               "vaa_above_avg", "ddz_delta_az", "ssw_dev"]
DELIVERY = ["arm_angle", "release_pos_x", "release_pos_z", "release_extension",
            "release_spread"]
FIG_DIR = paths.ROOT / "analysis"


def log(msg):
    print(msg, flush=True)


def load():
    prof = pd.read_parquet(profiles.profiles_path("pitcher"))
    emb = pd.read_parquet(paths.META / "embed_pitcher_2024_2026.parquet")
    keys = profiles.KEYS
    df = prof.merge(emb[keys + [c for c in emb if c.startswith(("pc", "umap"))]], on=keys)
    df["t"] = (df.season - df.season.min()) * 2 + df.half - 1
    # Primary-fastball features under one name, so a sinker-baller and a
    # four-seam pitcher are compared on "their fastball".
    for f in FB_FEATURES:
        df[f"fb_{f}"] = np.nan
        for t in profiles.FASTBALLS:
            sel = df.primary_fb == t
            df.loc[sel, f"fb_{f}"] = df.loc[sel, f"{t}_{f}"]
    return df


def neighbourhood_rv(df, pcs):
    """Mean rv100 over the K nearest rows belonging to other pitchers."""
    from sklearn.neighbors import NearestNeighbors

    P = df[pcs].to_numpy()
    _, idx = NearestNeighbors(n_neighbors=K + 30).fit(P).kneighbors(P)
    pid, rv = df.pitcher.to_numpy(), df.rv100.to_numpy()
    out = np.empty(len(df))
    for i, row in enumerate(idx):
        other = row[pid[row] != pid[i]][:K]
        out[i] = rv[other].mean()
    return out


def transitions(df, pcs):
    delta_cols = ([f"{t}_usage_all" for t in profiles.PITCH_TYPES] + DELIVERY
                  + [f"fb_{f}" for f in FB_FEATURES])
    tr = df.merge(df.assign(t=df.t - 1), on=["pitcher", "t"], suffixes=("", "_next"))
    tr = tr[tr.primary_fb.notna() & tr.primary_fb_next.notna()].copy()

    tr["move"] = np.linalg.norm(tr[pcs].to_numpy() - tr[[f"{c}_next" for c in pcs]].to_numpy(), axis=1)
    for c in delta_cols:
        tr[f"d_{c}"] = tr[f"{c}_next"] - tr[c]
    same_fb = tr.primary_fb == tr.primary_fb_next   # fastball deltas: like with like
    for f in FB_FEATURES:
        tr.loc[~same_fb, f"d_fb_{f}"] = np.nan

    D = tr[[f"d_{c}" for c in delta_cols]]
    sd = D.std()
    Z = D / sd
    for c in delta_cols:
        tr[f"z_{c}"] = Z[f"d_{c}"]

    ev = []
    for t in profiles.PITCH_TYPES:
        u0, u1 = tr[f"{t}_usage_all"], tr[f"{t}_usage_all_next"]
        tr[f"added_{t}"] = (u0 < ADD_FROM) & (u1 >= ADD_TO)
        tr[f"dropped_{t}"] = (u0 >= ADD_TO) & (u1 < ADD_FROM)
        ev += [f"added_{t}", f"dropped_{t}"]
    tr["fb_switch"] = ~same_fb
    ev.append("fb_switch")
    for c in DELIVERY + [f"fb_{f}" for f in FB_FEATURES]:
        tr[f"big_{c}"] = Z[f"d_{c}"].abs() > EVENT_SD
        ev.append(f"big_{c}")
    tr["n_events"] = tr[ev].sum(axis=1)

    log(f"{len(tr):,} transitions from {tr.pitcher.nunique():,} pitchers "
        f"({same_fb.mean():.1%} kept the same primary fastball)")
    log("1 SD of half-to-half change: " + ", ".join(
        f"{c} {sd[f'd_{c}']:.2f}" for c in ["arm_angle", "release_extension",
                                            "fb_release_speed", "fb_pfx_z", "fb_ddz_delta_az"]))
    rates = tr[ev].mean().sort_values(ascending=False)
    log("event rates per transition (%):\n" + (100 * rates[rates > 0]).round(1).to_string())
    return tr, delta_cols, ev


def modes_of_change(tr, delta_cols):
    """PCA of standardized deltas -> loadings (modes x features) and scores."""
    from sklearn.decomposition import PCA

    Z = tr[[f"z_{c}" for c in delta_cols]].fillna(0).clip(-4, 4).to_numpy()
    pca = PCA(n_components=N_MODES, random_state=SEED).fit(Z)
    load = pd.DataFrame(pca.components_, columns=delta_cols,
                        index=[f"mode {i + 1}" for i in range(N_MODES)])
    # Sign convention: largest-magnitude loading positive, so "+" reads as the
    # direction the mode is named for.
    for m in load.index:
        if load.loc[m].iloc[load.loc[m].abs().argmax()] < 0:
            load.loc[m] *= -1
    scores = (Z - pca.mean_) @ load.to_numpy().T
    for i in range(N_MODES):
        tr[f"mode{i + 1}"] = scores[:, i]
    for i, (m, row) in enumerate(load.iterrows()):
        top = row.abs().sort_values(ascending=False).head(4).index
        log(f"  {m} ({100 * pca.explained_variance_ratio_[i]:.1f}% var): "
            + ", ".join(f"{c} {row[c]:+.2f}" for c in top))
    return load, pca.explained_variance_ratio_


def payoff(tr):
    """OLS d_rv ~ d_nbhd_rv + rv(t) + nbhd_rv(t), pitcher-cluster bootstrap CIs."""
    y = (tr.rv100_next - tr.rv100).to_numpy()
    X = np.column_stack([np.ones(len(tr)), tr.nbhd_rv_next - tr.nbhd_rv,
                         tr.rv100, tr.nbhd_rv])

    def fit(rows):
        return np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]

    beta = fit(np.arange(len(tr)))
    rng = np.random.default_rng(SEED)
    groups = tr.groupby("pitcher").indices
    pids = np.array(list(groups))
    boots = np.array([fit(np.concatenate([groups[p] for p in rng.choice(pids, len(pids))]))
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    names = ["d_nbhd_rv", "rv100(t)", "nbhd_rv(t)"]
    log("payoff OLS on d_rv100 (n = %s):" % f"{len(tr):,}")
    for j, nm in enumerate(names, start=1):
        log(f"  {nm:<11} {beta[j]:+.3f}  (95% CI {lo[j]:+.3f} to {hi[j]:+.3f})")
    return beta, lo, hi


def drift_field(tr, nb=16, min_n=10):
    """Time-asymmetric mean displacement on a UMAP grid: (forward - reversed) / 2."""
    u0, v0 = tr.umap1.to_numpy(), tr.umap2.to_numpy()
    u1, v1 = tr.umap1_next.to_numpy(), tr.umap2_next.to_numpy()
    allu, allv = np.r_[u0, u1], np.r_[v0, v1]
    gx = np.linspace(allu.min(), allu.max(), nb + 1)
    gy = np.linspace(allv.min(), allv.max(), nb + 1)

    def cell_mean(x, y, du, dv):
        ix = np.clip(np.digitize(x, gx) - 1, 0, nb - 1)
        iy = np.clip(np.digitize(y, gy) - 1, 0, nb - 1)
        g = pd.DataFrame({"ix": ix, "iy": iy, "du": du, "dv": dv}).groupby(["ix", "iy"])
        return g.mean()[g.size() >= min_n]

    fwd = cell_mean(u0, v0, u1 - u0, v1 - v0)           # leaving each cell, forward
    rev = cell_mean(u1, v1, u0 - u1, v0 - v1)           # leaving each cell, reversed
    both = fwd.join(rev, lsuffix="_f", rsuffix="_r", how="inner").dropna()
    field = pd.DataFrame({"du": (both.du_f - both.du_r) / 2,
                          "dv": (both.dv_f - both.dv_r) / 2}, index=both.index)
    raw = fwd.loc[both.index]
    ix, iy = field.index.get_level_values(0), field.index.get_level_values(1)
    cx, cy = (gx[ix] + gx[ix + 1]) / 2, (gy[iy] + gy[iy + 1]) / 2
    shrink = np.hypot(field.du, field.dv).median() / np.hypot(raw.du, raw.dv).median()
    log(f"drift field: {len(field)} cells; time-asymmetric arrows are "
        f"{shrink:.0%} the length of the raw forward arrows")
    return cx, cy, field, raw


def drift_null(tr, n_perm=200):
    """Is the time-asymmetric field bigger than noise? Null: reverse time for a
    random half of pitchers (whole pitcher, so within-pitcher dependence stays)."""
    import contextlib
    import io

    def median_len(t):
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, f, _ = drift_field(t)
        return np.hypot(f.du, f.dv).median()

    obs = median_len(tr)
    rng = np.random.default_rng(SEED)
    pids = tr.pitcher.unique()
    null = []
    for _ in range(n_perm):
        flip = tr.pitcher.isin(rng.choice(pids, len(pids) // 2, replace=False))
        t = tr.copy()
        for a in ("umap1", "umap2"):
            t.loc[flip, [a, f"{a}_next"]] = tr.loc[flip, [f"{a}_next", a]].to_numpy()
        null.append(median_len(t))
    null = np.array(null)
    log(f"drift vs time-reversal null: median arrow {obs:.3f} vs {np.median(null):.3f} "
        f"(95th pct {np.percentile(null, 95):.3f}), p = {(null >= obs).mean():.3f}")


def plot(df, tr, load, var):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figure_style import apply_style, save, CMAP_DIVERGING

    apply_style()
    fig = plt.figure(figsize=(7.09, 5.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8])
    cx, cy, field, raw = drift_field(tr)
    med = df.nbhd_rv.median()
    lim = np.nanpercentile(np.abs(df.nbhd_rv - med), 98)
    scale = np.percentile(np.hypot(raw.du, raw.dv), 90) / 0.06  # shared arrow scale

    for j, (vec, title) in enumerate([(raw, "a  Raw forward movement"),
                                      (field, "b  Time-asymmetric drift")]):
        ax = fig.add_subplot(gs[0, j])
        sc = ax.scatter(df.umap1, df.umap2, c=df.nbhd_rv, s=1.5, cmap=CMAP_DIVERGING,
                        vmin=med - lim, vmax=med + lim, linewidths=0, rasterized=True)
        ax.quiver(cx, cy, vec.du, vec.dv, angles="xy", scale_units="width", scale=scale,
                  width=0.005, color="0.1")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title(title, loc="left")
    cb = fig.colorbar(sc, ax=fig.axes[:2], shrink=0.7, pad=0.01)
    cb.set_label("Neighbourhood RV/100 (other pitchers)")

    ax = fig.add_subplot(gs[1, :])
    show = load.loc[:, load.abs().max() >= 0.2]
    lim = np.abs(show.to_numpy()).max()
    im = ax.imshow(show.to_numpy(), cmap=CMAP_DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([f"{m} ({100 * v:.0f}%)" for m, v in zip(show.index, var)])
    ax.set_xticks(range(show.shape[1]))
    ax.set_xticklabels([c.replace("_usage_all", " usage").replace("fb_", "FB ")
                        .replace("release_", "rel. ").replace("_above_avg", "")
                        .replace("ddz_delta_az", "DDZ az").replace("_", " ")
                        for c in show.columns], rotation=45, ha="right")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("c  Modes of half-to-half change (PCA loadings on standardized deltas)", loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("Loading")

    save(fig, FIG_DIR / "trajectories")


if __name__ == "__main__":
    df = load()
    pcs = [c for c in df if c.startswith("pc") and c[2:].isdigit()]
    df["nbhd_rv"] = neighbourhood_rv(df, pcs)
    tr, delta_cols, ev = transitions(df, pcs)
    load, var = modes_of_change(tr, delta_cols)
    payoff(tr)
    drift_null(tr)
    out = paths.META / "trajectories_2024_2026.parquet"
    keep = (["pitcher", "player_name", "season", "half", "t", "primary_fb", "primary_fb_next",
             "move", "n_events", "rv100", "rv100_next", "nbhd_rv", "nbhd_rv_next"]
            + [f"mode{i + 1}" for i in range(N_MODES)]
            + [f"d_{c}" for c in delta_cols] + ev)
    tr[keep].to_parquet(out, index=False)
    log(f"wrote {out}")
    plot(df, tr, load, var)
