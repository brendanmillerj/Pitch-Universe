"""Clusters and subclusters within each pitch type, for "Find a type of pitch" (replaces the hand-written tags).

Same space as the atlas's maps and lookalikes: one pitch type's 15 identity measures (atlas_build.type_input: within-
type median fill, z within type, block-scaled), PCA, a 15-nearest-neighbour graph, Leiden (RBConfiguration), as for
the islands. Within a pitch type the pitches form a continuum (single Leiden runs agree with resampled runs at an
adjusted Rand index of only 0.4-0.6), so the clusters are CONSENSUS clusters:

  1. For a resolution, run Leiden on 60 resamples that each leave out 20% of pitchers (all of a pitcher's rows together).
  2. Consensus matrix C[i, j] = share of the resamples holding both pitches that put them in the same cluster.
  3. Cut C into k clusters by average linkage (k = the resamples' median cluster count).
  4. Choose the resolution: k inside the level's range, every cluster >= its minimum size, then the highest
     reproducibility (adjusted Rand index between the cuts of two independent consensus matrices of 30 resamples each),
     ties to the lowest PAC (share of pairs with 0.1 < C < 0.9: how much is ambiguous).
  5. Confidence of a pitch = its mean C with the rest of its cluster. CORE members have confidence >= CORE; the rest
     are EDGE members, between two clusters.

Two levels, like subclustering a cell type: level 1 splits the pitch type into 3-6 clusters; level 2 splits each cluster
of >= 60 rows (its rows, a new 15-NN graph on the same PCs) into 2-5 subclusters of >= 15, kept only if reproducibility
clears MIN_REPRO, otherwise the cluster is not split. Persistence (a pitcher's same pitch in the same cluster in
consecutive seasons, against chance = sum of squared cluster shares) is reported, never used to choose.

Names: each cluster's standardised mean differences from the rest of its parent (the pitch type, or the sibling
subclusters) on plain measures; at most one phrase per idea (ride, run, slot), the two largest |SMD| (>= 0.5 for clusters,
>= 0.4 for subclusters). Subclusters are named only by what separates them from their siblings, in comparatives.

Outputs (MLB rows only; Triple-A is placed later by atlas_build from its MLB lookalikes, never fitted):
  data/meta/pitch_subclusters_2024_2026.parquet   pitcher, season, pitch_type, cl1, cl2, conf1, conf2
  data/meta/pitch_subclusters_2024_2026.json      per type: clusters (id, name, size, core size, markers, subclusters)
  analysis/pitch_subclusters/report.md, sweep.csv, Figure7.{pdf,png}

    python scripts/pitch_subclusters.py            # full (tens of minutes)
    python scripts/pitch_subclusters.py --quick    # four-seamers only, 8 resamples per half
"""
import argparse
import json
import string
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from mlbpitch import paths, profiles

sys.path.insert(0, str(paths.ROOT / "scripts"))
import atlas_build as ab  # noqa: E402
from figure_style import PITCH_NAMES  # noqa: E402

OUT = paths.ROOT / "analysis" / "pitch_subclusters"
META_PARQ = paths.META / "pitch_subclusters_2024_2026.parquet"
META_JSON = paths.META / "pitch_subclusters_2024_2026.json"
TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]
K, SEED, KEEP, N_HALF = 15, 0, 0.8, 30
RES1 = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
RES2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
N1, N2 = (3, 6), (2, 5)
MIN_CLUSTER, MIN_SUB, MIN_CLUSTER_TO_SPLIT, MIN_REPRO = 40, 15, 60, 0.7
CORE = 0.6

# (display column, idea, level-1 word if higher / lower, level-2 comparative if higher / lower)
MARKERS = [
    ("velo", "velo", "harder", "softer", "harder", "softer"),
    ("ivb", "ride", "more ride", "less ride", "more ride", "less ride"),
    ("ddz_az", "ride", "ride beyond slot", "less ride than slot", "more ride for the slot", "less ride for the slot"),
    ("hb", "run", "more arm-side run", "more glove-side break", "more arm-side run", "more glove-side"),
    ("ddz_ax", "run", "run beyond slot", "less run than slot", "more run for the slot", "less run for the slot"),
    ("spin", "spin", "high spin", "low spin", "more spin", "less spin"),
    ("vaa", "vaa", "flat approach", "steep approach", "flatter", "steeper"),
    ("arm_angle", "slot", "high slot", "low slot", "higher slot", "lower slot"),
    ("rel_height", "slot", "high release", "low release", "higher release", "lower release"),
    ("extension", "ext", "long extension", "short extension", "longer extension", "shorter extension"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def leiden(P, res, seed=SEED, k=K):
    import igraph as ig
    import leidenalg
    k = min(k, len(P) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
    edges = {(min(i, j), max(i, j)) for i, row in enumerate(idx) for j in row}
    g = ig.Graph(n=len(P), edges=sorted(edges))
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=seed)
    return np.array(part.membership)


def relabel(lab):
    """Clusters numbered by size, largest first."""
    order = pd.Series(lab).value_counts().index
    m = {c: i for i, c in enumerate(order)}
    return np.array([m[c] for c in lab])


def consensus_matrix(P, pid, res, n, seed):
    m = len(P)
    S, N = np.zeros((m, m), np.float32), np.zeros((m, m), np.float32)
    rng, pits, counts = np.random.default_rng(seed), np.unique(pid), []
    for b in range(n):
        rows = np.flatnonzero(np.isin(pid, rng.choice(pits, max(2, int(KEEP * len(pits))), replace=False)))
        lab = leiden(P[rows], res, seed=seed + 1 + b)
        counts.append(lab.max() + 1)
        ix = np.ix_(rows, rows)
        S[ix] += (lab[:, None] == lab[None, :])
        N[ix] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(N > 0, S / N, np.nan)
    return C, int(np.median(counts))


def cut(C, k):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = 1 - np.nan_to_num(C, nan=0.0)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    return relabel(fcluster(Z, k, criterion="maxclust") - 1)


def pac(C, lo=0.1, hi=0.9):
    v = C[np.triu_indices_from(C, 1)]
    v = v[np.isfinite(v)]
    return float(((v > lo) & (v < hi)).mean())


def confidence(C, lab):
    conf = np.empty(len(lab))
    Cz = np.nan_to_num(C, nan=0.0)
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c)
        conf[idx] = (Cz[np.ix_(idx, idx)].sum(1) - 1) / max(len(idx) - 1, 1)
    return conf


def persistence(lab, pid, season):
    d = pd.DataFrame({"p": pid, "s": season, "c": lab}).sort_values(["p", "s"])
    nxt = d.groupby("p").shift(-1)
    ok = nxt.s.notna() & (nxt.s == d.s + 1)
    same = float((d.c[ok] == nxt.c[ok]).mean()) if ok.any() else np.nan
    share = pd.Series(lab).value_counts(normalize=True)
    return same, float((share ** 2).sum())


def level(P, pid, season, grid, n_range, min_size, n_half, seed, tag):
    rows, out = [], None
    for r in grid:
        n0 = leiden(P, r).max() + 1
        if n0 < n_range[0]:
            continue
        if n0 > n_range[1] + 2:
            break
        Ca, ka = consensus_matrix(P, pid, r, n_half, seed)
        Cb, kb = consensus_matrix(P, pid, r, n_half, seed + 10_000)
        k = int(round((ka + kb) / 2))
        if k < 2:
            continue
        C = np.where(np.isfinite(Ca) & np.isfinite(Cb), (Ca + Cb) / 2, np.where(np.isfinite(Ca), Ca, Cb))
        repro = adjusted_rand_score(cut(Ca, k), cut(Cb, k))
        lab = cut(C, k)
        sizes = np.bincount(lab)
        pr, ch = persistence(lab, pid, season)
        rec = {"tag": tag, "res": r, "k": k, "pac": pac(C), "repro": float(repro), "min_size": int(sizes.min()),
               "persist": pr, "chance": ch, "core": float((confidence(C, lab) >= CORE).mean())}
        rows.append(rec)
        ok = n_range[0] <= k <= n_range[1] and sizes.min() >= min_size
        if ok and (out is None or (round(rec["repro"], 2), -rec["pac"]) > (round(out[0]["repro"], 2), -out[0]["pac"])):
            out = (rec, lab, C)
    return out, rows


def markers(D, rows_in, rows_rest, sd, lvl):
    out = []
    for col, idea, h1, l1, h2, l2 in MARKERS:
        a, b = D.loc[rows_in, col].astype(float), D.loc[rows_rest, col].astype(float)
        if a.notna().sum() < 5 or b.notna().sum() < 5 or not sd.get(col):
            continue
        smd = (a.mean() - b.mean()) / sd[col]
        word = (h1 if smd > 0 else l1) if lvl == 1 else (h2 if smd > 0 else l2)
        out.append({"measure": col, "idea": idea, "smd": float(smd), "word": word, "mean": float(a.mean())})
    return sorted(out, key=lambda m: -abs(m["smd"]))


def name_from(mk, thresh, fallback):
    words, ideas = [], set()
    for m in mk:
        if abs(m["smd"]) < thresh or m["idea"] in ideas:
            continue
        words.append(m["word"]); ideas.add(m["idea"])
        if len(words) == 2:
            break
    s = ", ".join(words) if words else fallback
    return s[0].upper() + s[1:]


def numbers(D, rows):
    g = D.loc[rows]
    f = lambda c, d: f"{g[c].astype(float).mean():.{d}f}" if g[c].notna().any() else "—"
    return f"{f('velo', 1)} mph · ride {f('ivb', 1)} in · run {f('hb', 1)} in · {f('spin', 0)} rpm · arm angle {f('arm_angle', 0)}°"


def run(types_run, n_half):
    t = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False)).reset_index(drop=True)
    D = ab.type_display(t.assign(spin_eff=np.nan))
    defs, assign, sweeps = {}, [], []
    for pt in types_run:
        rows = np.flatnonzero(t.pitch_type.to_numpy() == pt)
        X = ab.type_input(t, rows)
        P = PCA(n_components=min(30, X.shape[1]), random_state=SEED).fit_transform(X)
        pid, season = t.pitcher.to_numpy()[rows], t.season.to_numpy()[rows]
        sd = {m[0]: float(D.loc[rows, m[0]].astype(float).std()) for m in MARKERS}
        best, sw = level(P, pid, season, RES1, N1, MIN_CLUSTER, n_half, SEED, pt)
        sweeps += [dict(r, level=1) for r in sw]
        if best is None:
            log(f"{pt}: no reproducible partition in range; skipped"); continue
        rec, lab1, C1 = best
        conf1 = confidence(C1, lab1)
        log(f"{pt}: {len(rows)} rows -> {rec['k']} clusters (resolution {rec['res']}, reproducibility {rec['repro']:.2f}, "
            f"PAC {rec['pac']:.2f}, core {rec['core']:.0%}, same next season {rec['persist']:.0%} vs {rec['chance']:.0%})")
        cl1 = np.array([f"{pt}-{string.ascii_uppercase[c]}" for c in lab1], dtype=object)
        cl2, conf2 = cl1.copy(), conf1.copy()
        clusters = []
        for c in range(lab1.max() + 1):
            m = lab1 == c
            cid = f"{pt}-{string.ascii_uppercase[c]}"
            mk = markers(D, rows[m], rows[~m], sd, 1)
            entry = {"id": cid, "letter": string.ascii_uppercase[c], "n": int(m.sum()), "n_core": int((conf1[m] >= CORE).sum()),
                     "name": name_from(mk, 0.5, "middle of the type"), "markers": mk[:5], "numbers": numbers(D, rows[m]),
                     "split": None, "subs": []}
            if m.sum() >= MIN_CLUSTER_TO_SPLIT:
                b2, sw2 = level(P[m], pid[m], season[m], RES2, N2, MIN_SUB, n_half, SEED + 100 + c, cid)
                sweeps += [dict(r, level=2) for r in sw2]
                if b2 is not None and b2[0]["repro"] >= MIN_REPRO:
                    rec2, lab2, C2 = b2
                    cf2 = confidence(C2, lab2)
                    entry["split"] = {k: rec2[k] for k in ("res", "k", "pac", "repro", "persist", "chance", "core")}
                    sub_rows = rows[m]
                    for s in range(lab2.max() + 1):
                        ms = lab2 == s
                        mk2 = markers(D, sub_rows[ms], sub_rows[~ms], sd, 2)
                        entry["subs"].append({"id": f"{cid}{s + 1}", "n": int(ms.sum()), "n_core": int((cf2[ms] >= CORE).sum()),
                                              "name": name_from(mk2, 0.4, "middle of the group"), "markers": mk2[:5],
                                              "numbers": numbers(D, sub_rows[ms])})
                    idx = np.flatnonzero(m)
                    cl2[idx] = [f"{cid}{s + 1}" for s in lab2]
                    conf2[idx] = cf2
            clusters.append(entry)
            log(f"   {cid} ({entry['n']}, core {entry['n_core']}): {entry['name']}"
                + (f" -> {len(entry['subs'])} subclusters (reproducibility {entry['split']['repro']:.2f}): "
                   + " | ".join(f"{s['id'][-2:]} {s['name']}" for s in entry["subs"]) if entry["split"] else " (not split)"))
        defs[pt] = {"name": PITCH_NAMES[pt], "n": int(len(rows)), **{k: rec[k] for k in ("res", "k", "pac", "repro", "persist", "chance", "core")},
                    "clusters": clusters}
        assign.append(pd.DataFrame({"pitcher": pid, "season": season, "pitch_type": pt, "cl1": cl1, "cl2": cl2,
                                    "conf1": conf1.round(3), "conf2": conf2.round(3)}))
    return t, D, defs, pd.concat(assign, ignore_index=True), pd.DataFrame(sweeps)


def old_tags_overlap(t, A):
    """How the hand-written pitch-level tags fall across the new clusters (are known ideas recovered?)."""
    rules = json.loads((paths.ROOT / "scripts" / "pitcher_types.json").read_text())
    D = ab.type_display(t.assign(spin_eff=np.nan)).assign(pitcher=t.pitcher, season=t.season, pitch_type=t.pitch_type)
    checks = {"gyro-slider": ("SL", lambda d: d.ivb.abs().lt(4) & d.hb.abs().lt(4)),
              "kick-change": ("CH", lambda d: d.ivb.lt(5) & d.velo.ge(88)),
              "splinker": ("FS", lambda d: d.velo.ge(89)),
              "turbo-sinker": ("SI", lambda d: d.velo.ge(96)),
              "ride-beyond-slot": ("FF", lambda d: d.ddz_az.ge(1.5)),
              "cut-ride": ("FF", lambda d: d.ivb.ge(17) & d.hb.le(6)),
              "flat-vaa": ("FF", lambda d: d.vaa.ge(0.6))}
    names = {r["id"]: r["name"] for r in rules}
    out = []
    for tag, (pt, f) in checks.items():
        d = D[D.pitch_type == pt]
        hit = d[f(d).fillna(False)].merge(A, on=["pitcher", "season", "pitch_type"])
        if not len(hit):
            continue
        top = hit.cl2.value_counts()
        out.append({"tag": names.get(tag, tag), "pitches": len(hit), "clusters": ", ".join(f"{c} {n / len(hit):.0%}" for c, n in top.head(3).items())})
    return pd.DataFrame(out)


def figure(defs, A, types_run):
    import matplotlib.pyplot as plt
    import umap_benchmark_plots as ubp
    from figure_style import PRINT_W, save
    SZ = ubp.setup()
    at = pd.read_parquet(ab.ATLAS)
    at = at[at.level == "mlb"][["pitcher", "season", "pitch_type", "tx", "ty"]]
    M = A.merge(at, on=["pitcher", "season", "pitch_type"])
    pal = ["#2166ac", "#b2182b", "#e08214", "#542788", "#8c510a", "#4d4d4d"]
    n = len(types_run)
    cols = 4 if n > 1 else 1
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(PRINT_W, (44 * rows_ + 6) / 25.4), layout="constrained", squeeze=False)
    for ax, pt in zip(axes.ravel(), types_run):
        if pt not in defs:
            ax.axis("off"); continue
        d = M[M.pitch_type == pt]
        for k, c in enumerate(sorted(d.cl1.unique())):
            e = d[d.cl1 == c]
            core = e.conf1 >= CORE
            ax.scatter(e.tx[~core], e.ty[~core], s=1.0, lw=0, color=pal[k % len(pal)], alpha=0.25, rasterized=True)
            ax.scatter(e.tx[core], e.ty[core], s=1.2, lw=0, color=pal[k % len(pal)], rasterized=True)
            ax.text(e.tx[core].median() if core.any() else e.tx.median(), e.ty[core].median() if core.any() else e.ty.median(),
                    c.split("-")[1], fontsize=SZ.annot, fontweight="bold", ha="center", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.4))
        ax.set_title(f"{PITCH_NAMES[pt]} ({defs[pt]['k']})", fontsize=SZ.axis_ttl, pad=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        ubp.finish(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    save(fig, str(OUT / "Figure7"))


def report(defs, ov, secs):
    L = ["# Clusters and subclusters within each pitch type (consensus)", "",
         f"*Generated by `scripts/pitch_subclusters.py` on {time.strftime('%Y-%m-%d %H:%M')} ({secs:.0f} s). MLB 2024-2026; "
         "identity measures only (what the pitch is); results never used. Names are generated from markers.*", "",
         "![Figure 7](Figure7.png)", "",
         f"Figure 7. Each pitch type's t-SNE map (the atlas's), coloured by consensus cluster; solid dots are core members "
         f"(confidence >= {CORE}), faint dots edge members. Letters sit at each cluster's core.", "",
         "Reproducibility: adjusted Rand index between the clusters cut from two independent consensus matrices (30 resamples "
         "each, 20% of pitchers left out each time). PAC: share of pitch pairs that are ambiguous (together in 10-90% of "
         "resamples). Core: share of pitches with confidence >= " + str(CORE) + ". Same next season: a pitcher's same pitch in "
         "the same cluster the following season, against chance.", "",
         "| Pitch type | Rows | Clusters | Resolution | Reproducibility | PAC | Core | Same next season | Chance |",
         "|---|---|---|---|---|---|---|---|---|"]
    for pt, d in defs.items():
        L.append(f"| {d['name']} | {d['n']:,} | {d['k']} | {d['res']} | {d['repro']:.2f} | {d['pac']:.2f} | {d['core']:.0%} | "
                 f"{d['persist']:.0%} | {d['chance']:.0%} |")
    for pt, d in defs.items():
        L += ["", f"## {d['name']}", ""]
        for c in d["clusters"]:
            sp = c["split"]
            L.append(f"- **{c['letter']}** · {c['name']} ({c['n']:,}; core {c['n_core']:,}) · {c['numbers']}"
                     + (f" · {len(c['subs'])} subclusters (reproducibility {sp['repro']:.2f})" if sp else " · not split"))
            for s in c["subs"]:
                L.append(f"  - **{s['id'].split('-')[1]}** · {s['name']} ({s['n']:,}; core {s['n_core']:,}) · {s['numbers']}")
    if ov is not None and len(ov):
        L += ["", "## Do the clusters recover the hand-written pitch tags?", "",
              "Pitches meeting each old rule (approximated on season means) and the subclusters they fall in.", "",
              "| Old tag | Pitches | Main subclusters |", "|---|---|---|"]
        L += [f"| {r.tag} | {r.pitches} | {r.clusters} |" for r in ov.itertuples()]
    L += ["", "sweep.csv: every resolution tried at both levels (clusters, reproducibility, PAC, core share, persistence)."]
    (OUT / "report.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    types_run = ["FF"] if a.quick else TYPES
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    t, D, defs, A, sw = run(types_run, 8 if a.quick else N_HALF)
    sw.to_csv(OUT / "sweep.csv", index=False)
    if not a.quick:
        A.to_parquet(META_PARQ, index=False)
        META_JSON.write_text(json.dumps({"core": CORE, "types": defs}, indent=1))
    else:
        A.to_parquet(OUT / "quick_assign.parquet", index=False)
        (OUT / "quick_defs.json").write_text(json.dumps(defs, indent=1))
    ov = old_tags_overlap(t, A)
    figure(defs, A, types_run)
    report(defs, ov, time.time() - t0)
    log(f"done in {time.time() - t0:.0f} s; outputs in {OUT}")


if __name__ == "__main__":
    main()
