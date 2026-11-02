"""Pitch-level UMAP: every pitch thrown this season, colored by pitch classification.

Unlike scripts/umap_sample.py (pitcher-level, ~516 points, one row per pitcher's
whole arsenal), this fits the manifold on individual pitches -- the "pitch
archetype atlas" from CLAUDE.md's two-UMAP plan. Each point is one pitch;
clusters are pitch archetypes (a hard slider, a low-slot sinker, ...), not
pitchers.

STRATIFIED BY HANDEDNESS -- two independent fits, RHP and LHP, one panel each.
pfx_x and spin_axis are both lab-frame, so a lefty's arm-side break and spin axis
are the sign/reflection opposite of an identical righty's. Fitting the two hands
together therefore requires mirroring *every* such column into a common frame,
and missing even one is silent and severe: an earlier combined fit mirrored pfx_x
but not spin_axis, which left the features in opposite frames and tore every LHP
fastball/sinker/changeup off into its own detached island (separation scaled with
each pitch type's L-vs-R spin gap -- sweeper 124deg -> 2.25 sd, cutter 19deg ->
0.28 sd). Stratifying removes the whole failure mode: within one hand the raw
columns are already in a common frame, so nothing needs mirroring or adjusting.

Trade-off: the two panels are independent embeddings, so coordinates are NOT
comparable across them. Compare cluster structure, not position.

Colors and grouping follow Baseball Savant's own pitch-classification scheme
(Fastball / Offspeed / Breaking [Curveball group, Slider group] / Knuckleball /
Other Pitches), per a reference screenshot of Savant's pitch-type filter UI --
not a from-scratch palette. Only null pitch_type is dropped (no real
classification to assign); everything else, including Savant's own "Other
Pitches" bucket (Eephus, Pitch Out, Other, Unknown), keeps its slot so the figure
covers every pitch thrown.
"""
import time

import numpy as np
import pandas as pd
from mlbpitch import paths, store

t0 = time.time()

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
raw = store.query("""
    SELECT pitcher, player_name, p_throws, pitch_type, pitch_name,
           release_speed, pfx_x, pfx_z, release_extension, arm_angle, spin_axis
    FROM pitches
    WHERE pitch_type IS NOT NULL
      AND release_speed IS NOT NULL AND pfx_x IS NOT NULL AND pfx_z IS NOT NULL
      AND release_extension IS NOT NULL AND arm_angle IS NOT NULL
      AND spin_axis IS NOT NULL
""")
print(f"[{time.time()-t0:.0f}s] loaded {len(raw):,} pitches, "
      f"{raw.pitch_type.nunique()} pitch types, {raw.pitcher.nunique()} pitchers")
print(raw.p_throws.value_counts().to_string())

# spin_axis is circular (0-360 deg) -- carry as its sin/cos, never as a raw degree.
# No handedness mirror here: each fit below sees a single hand, so the lab frame
# is already common within a panel.
th = np.deg2rad(raw.spin_axis.astype("float64").to_numpy())
raw["spin_sin"], raw["spin_cos"] = np.sin(th), np.cos(th)

FEATURES = ["release_speed", "pfx_x", "pfx_z", "release_extension", "arm_angle",
            "spin_sin", "spin_cos"]

# --------------------------------------------------------------------------
# UMAP -- one independent fit per handedness
# --------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
import umap

HANDS = [("R", "Right-handed pitchers"), ("L", "Left-handed pitchers")]

frames = []
for hand, _ in HANDS:
    sub = raw[raw.p_throws == hand].copy()
    # Scaler is also per-hand: standardizing on pooled data would reintroduce a
    # cross-hand reference frame through the back door.
    Z = StandardScaler().fit_transform(sub[FEATURES].to_numpy(dtype="float32")).astype("float32")
    print(f"[{time.time()-t0:.0f}s] fitting UMAP ({hand}HP) on {Z.shape[0]:,} x {Z.shape[1]} ...")
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                    random_state=42, verbose=True).fit_transform(Z)
    sub["x"], sub["y"] = emb[:, 0], emb[:, 1]
    frames.append(sub)
    print(f"[{time.time()-t0:.0f}s] UMAP ({hand}HP) done")

out = pd.concat(frames, ignore_index=True)
out = out[["pitcher", "player_name", "p_throws", "pitch_type", "pitch_name"]
          + FEATURES + ["x", "y"]]
out_path = paths.META / "umap_pitches.parquet"
out.to_parquet(out_path)
print(f"[{time.time()-t0:.0f}s] wrote {out_path} ({len(out):,} rows)")

# --------------------------------------------------------------------------
# Color = general category (one hue each, per Baseball Savant's own top-level
# pitch-classification groups: Fastball / Offspeed / Breaking / Knuckleball /
# Other Pitches). Individual pitch types (the subcategories) are no longer
# color-coded -- they're direct-labeled at their cluster centroid instead,
# same convention as analysis/example_umap_for_snrnaseq.R's cell-type labels
# (e.g. "VIP interneurons"): color carries the coarse identity, text carries
# the fine one.
# --------------------------------------------------------------------------
GENERAL_OF = {
    "FF": "Fastball", "SI": "Fastball", "FC": "Fastball",
    "CH": "Offspeed", "FS": "Offspeed", "FO": "Offspeed",
    "CU": "Breaking", "KC": "Breaking", "CS": "Breaking",
    "SL": "Breaking", "ST": "Breaking", "SV": "Breaking",
    "KN": "Knuckleball",
    "EP": "Other Pitches", "FA": "Other Pitches", "PO": "Other Pitches", "UN": "Other Pitches",
}
GENERAL_PALETTE = {
    "Fastball": "#C0392B",       # brick red
    "Offspeed": "#4CA64C",       # green
    "Breaking": "#D1A32E",       # gold
    "Knuckleball": "#4B3FC4",    # indigo
    "Other Pitches": "#8C8C8C",  # gray
}
GENERAL_ORDER = ["Fastball", "Offspeed", "Breaking", "Knuckleball", "Other Pitches"]

present = set(raw.pitch_type.unique())
missing = present - GENERAL_OF.keys()
if missing:
    raise ValueError(f"pitch_type(s) with no general-category mapping: {missing}")

out["general"] = out.pitch_type.map(GENERAL_OF)
pitch_name = raw.groupby("pitch_type").pitch_name.first()

# Subcategory (pitch-type) label positions: median UMAP coord per pitch_type,
# computed WITHIN each panel (the two embeddings have independent coordinates).
# Below MIN_LABEL_N, a "centroid" is just a handful of points -- ggrepel-style
# collision avoidance can then drag that label onto an unrelated, actually-
# populated cluster, misleading even though technically correct (same reasoning
# as MIN_LABEL_N in example_umap_for_snrnaseq.R).
MIN_LABEL_N = 100


def centroids_for(panel):
    c = (panel.groupby("pitch_type")
              .agg(x=("x", "median"), y=("y", "median"), n=("x", "size"))
              .query("n >= @MIN_LABEL_N"))
    c["label"] = c.index.map(pitch_name)
    return c


# --------------------------------------------------------------------------
# Figure -- two panels, shared legend
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",  # Helvetica/Arial not installed on this host
    "axes.linewidth": 1,
    "axes.edgecolor": "black",
    "axes.grid": False,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import matplotlib.patheffects as pe
from adjustText import adjust_text

fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8))

for ax, (hand, title) in zip(axes, HANDS):
    panel = out[out.p_throws == hand]
    present_here = [g for g in GENERAL_ORDER if g in set(panel.general)]
    for g in reversed(present_here):  # rarest category plotted last, on top
        m = panel.general == g
        ax.scatter(panel.loc[m, "x"], panel.loc[m, "y"], s=2.5, alpha=0.3,
                   color=GENERAL_PALETTE[g], edgecolors="none", rasterized=True)

    cen = centroids_for(panel)
    print(f"[labels] {hand}HP: {len(cen)} pitch types labeled (n >= {MIN_LABEL_N}); "
          f"skipped: {sorted(set(panel.pitch_type.unique()) - set(cen.index))}")
    texts = [ax.text(r.x, r.y, r.label, fontsize=8.5, fontweight="normal",
                     color="black", ha="center", va="center",
                     path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
             for r in cen.itertuples()]
    adjust_text(texts, ax=ax, expand=(1.3, 1.6),
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))

    ax.set_title(f"{title}  (n = {len(panel):,})", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)

# Legend: general category only (5 entries) -- blown-up swatches since the
# plotted points are tiny (same big_legend() idea as the R reference script).
handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=GENERAL_PALETTE[g],
                      markersize=8, label=g)
           for g in GENERAL_ORDER if g in set(out.general)]
axes[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                frameon=False, title="Pitch class", handletextpad=0.6, labelspacing=0.7)

fig.suptitle(f"Pitch-level UMAP — every pitch, 2026 season (n = {len(out):,})",
             fontsize=18, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.97))

pdf_path = paths.ROOT / "analysis" / "pitch_umap.pdf"
png_path = paths.ROOT / "analysis" / "pitch_umap.png"
pdf_path.parent.mkdir(exist_ok=True)
fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"[{time.time()-t0:.0f}s] wrote {pdf_path} and {png_path}")
