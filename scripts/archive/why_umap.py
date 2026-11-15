"""Does the UMAP earn its place? Neighbour preservation of 2D views of the pitch data.

For each pitch (one pitcher-season's version of one pitch type, 2024-2026): its K nearest
neighbours in the full feature space -- the same block-scaled, within-type z-scored features
the comps use, reduced to 30 PCs. Each 2D view is scored by the share of those K that are also
among the pitch's K nearest in the view, and by sklearn's trustworthiness. A second check asks,
on the All-pitches map, how often a dot's nearest dots are the same pitch type.

Results are written up in analysis/umap_justification.md.

Usage:  python scripts/why_umap.py
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from mlbpitch import paths, profiles
from profile_embed import TYPE_BLOCKS, block_scale, zscore

K = 10
RANGE = f"{min(profiles.SEASONS)}_{max(profiles.SEASONS)}"
COLS = [c for cs in TYPE_BLOCKS.values() for c in cs]


def knn(X, k=K):
    return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]


def overlap(A, B):
    return np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(A, B)])


def main():
    t = pd.read_parquet(profiles.profiles_path("type", profiles.SEASONS, by_half=False))
    a = pd.read_parquet(paths.META / f"atlas_type_{RANGE}.parquet")
    t = t.merge(a[["pitcher", "season", "pitch_type", "tx", "ty", "gx", "gy", "tx3", "ty3", "tz3", "gx3", "gy3", "gz3"]],
                on=["pitcher", "season", "pitch_type"])

    names = ["movement (ride x HB)", "velo x ride", "PCA (top 2 directions)", "UMAP 2D", "UMAP 3D"]
    rows = []
    for pt, g in t.groupby("pitch_type"):
        F = g[COLS].astype(float)
        X = block_scale(zscore(F.fillna(F.median())), TYPE_BLOCKS)
        P = PCA(n_components=min(30, X.shape[1]), random_state=0).fit_transform(X)
        truth = knn(P)
        views = {
            names[0]: zscore(pd.DataFrame({"ivb": g.pfx_z, "hb": g.pfx_x})).to_numpy(),
            names[1]: zscore(pd.DataFrame({"v": g.release_speed, "ivb": g.pfx_z})).to_numpy(),
            names[2]: P[:, :2],
            names[3]: g[["tx", "ty"]].to_numpy(),
            names[4]: g[["tx3", "ty3", "tz3"]].to_numpy(),
        }
        r = {"type": pt, "n": len(g)}
        for name, V in views.items():
            r[name] = overlap(truth, knn(V))
            r[name + " trust"] = trustworthiness(P, V, n_neighbors=K)
        rows.append(r)

    df = pd.DataFrame(rows).set_index("type")
    w = df.n / df.n.sum()
    print(f"Share of each pitch's {K} truly most similar pitches that are also its {K} nearest on the picture")
    print((df[names] * 100).round(0).astype(int).to_string())
    print("weighted overall:", {k: f"{100 * (df[k] * w).sum():.0f}%" for k in names})
    print("chance level:", f"{100 * (K / (df.n - 1) * w).sum():.1f}%")
    print("trustworthiness:", {k: round(float((df[k + ' trust'] * w).sum()), 3) for k in names})

    lab = t.pitch_type.to_numpy()
    purity = lambda V: np.mean(lab[knn(V)] == lab[:, None])
    mov = zscore(pd.DataFrame({"ivb": t.pfx_z, "hb": t.pfx_x})).to_numpy()
    print(f"\nAll pitches: same-type share of 10 nearest dots -- movement {100 * purity(mov):.0f}%, "
          f"UMAP 2D {100 * purity(t[['gx', 'gy']].to_numpy()):.0f}%, UMAP 3D {100 * purity(t[['gx3', 'gy3', 'gz3']].to_numpy()):.0f}%")


if __name__ == "__main__":
    main()
