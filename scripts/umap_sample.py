"""Sample pitcher-level UMAP: league manifold with the Cubs highlighted.

Fitted on all qualifying league pitchers rather than the 34 Cubs alone -- UMAP
needs enough points to learn a manifold, and the interesting question is *where
the Cubs land relative to everyone else*, which a Cubs-only fit cannot answer.
"""
import numpy as np
import pandas as pd
from mlbpitch import ddz, paths, store

MIN_PITCHES = 200
TYPES = ["FF", "SI", "SL", "ST", "CH", "CU", "FC", "FS"]

raw = store.query("""
    SELECT pitcher, player_name, p_throws, pitch_type,
           CASE WHEN inning_topbot='Top' THEN home_team ELSE away_team END AS team,
           release_speed, release_pos_x, release_pos_z, release_extension,
           pfx_x, pfx_z, spin_axis, release_spin_rate, arm_angle,
           plate_x, plate_z, vaa, haa, ax, az
    FROM pitches
    WHERE pitch_type IS NOT NULL AND release_speed IS NOT NULL AND arm_angle IS NOT NULL
""")
raw = store.add_angles_adjusted(raw)
raw = ddz.add_ddz_delta(raw)

# Mirror lefties so arsenal shape, not handedness, drives the embedding.
left = (raw.p_throws == "L").to_numpy()
for c in ["release_pos_x", "pfx_x"]:
    raw.loc[left, c] = -raw.loc[left, c]

# spin_axis is circular: carry it as the mean resultant vector, never as degrees.
th = np.deg2rad(raw.spin_axis.astype("float64").to_numpy())
raw["spin_sin"], raw["spin_cos"] = np.sin(th), np.cos(th)

counts = raw.groupby("pitcher").size()
keep = counts[counts >= MIN_PITCHES].index
raw = raw[raw.pitcher.isin(keep)]

# team = modal team (handles mid-season trades)
team = raw.groupby(["pitcher", "team"]).size().reset_index(name="n")
team = team.sort_values("n").drop_duplicates("pitcher", keep="last").set_index("pitcher")["team"]

rows = {}
# 1. arsenal usage fractions
usage = (raw.groupby(["pitcher", "pitch_type"], observed=True).size()
           .unstack(fill_value=0).reindex(columns=TYPES, fill_value=0))
usage = usage.div(usage.sum(axis=1), axis=0)
for t in TYPES:
    rows[f"use_{t}"] = usage[t]

# 2. release characteristics (whole arsenal)
g = raw.groupby("pitcher")
for c in ["arm_angle", "release_pos_x", "release_pos_z", "release_extension"]:
    rows[c] = g[c].mean()
rows["velo_overall"] = g.release_speed.mean()

# spread of velocity across the arsenal -- separates one-speed relievers from
# starters who change speeds
pt_velo = raw.groupby(["pitcher", "pitch_type"], observed=True).release_speed.mean()
rows["velo_spread"] = pt_velo.groupby("pitcher").max() - pt_velo.groupby("pitcher").min()

# 3. primary fastball profile (FF, else SI) -- the pitch the arsenal is built on
fb = raw[raw.pitch_type.isin(["FF", "SI"])]
prim = (fb.groupby(["pitcher", "pitch_type"], observed=True).size()
          .reset_index(name="n").sort_values("n")
          .drop_duplicates("pitcher", keep="last").set_index("pitcher")["pitch_type"])
fb = fb[[p in prim.index and prim[p] == t
         for p, t in zip(fb.pitcher, fb.pitch_type)]]
fg = fb.groupby("pitcher")
for c, name in [("release_speed", "fb_velo"), ("pfx_x", "fb_pfx_x"), ("pfx_z", "fb_pfx_z"),
                ("release_spin_rate", "fb_spin"), ("spin_sin", "fb_spin_sin"),
                ("spin_cos", "fb_spin_cos"), ("vaa_above_avg", "fb_vaa_aa"),
                ("ddz_delta_az_velo", "fb_ddz_v")]:
    rows[name] = fg[c].mean()

X = pd.DataFrame(rows)
X = X.dropna()
meta = pd.DataFrame({
    "player_name": raw.groupby("pitcher").player_name.first(),
    "p_throws": raw.groupby("pitcher").p_throws.first(),
    "team": team,
    "pitches": counts,
}).loc[X.index]

print(f"feature matrix: {X.shape[0]} pitchers x {X.shape[1]} features")
print(f"Cubs in matrix: {(meta.team == 'CHC').sum()}")

from sklearn.preprocessing import StandardScaler
import umap

Z = StandardScaler().fit_transform(X.to_numpy())
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                random_state=42).fit_transform(Z)

out = meta.copy()
out["x"], out["y"] = emb[:, 0], emb[:, 1]
out = out.join(X)
out.to_parquet(paths.META / "umap_pitchers_sample.parquet")
print("\nCubs pitchers in the embedding:")
print(out[out.team == "CHC"][["player_name", "p_throws", "pitches", "x", "y",
                              "arm_angle", "fb_velo", "fb_vaa_aa"]]
      .sort_values("pitches", ascending=False).round(2).to_string(index=False))
