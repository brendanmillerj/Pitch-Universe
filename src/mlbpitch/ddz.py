"""Dynamic Dead Zone (DDZ) deltas -- observed movement vs. what a pitcher's
release characteristics predict.

The "dead zone" is the region of the movement plot where a fastball gets poor
results. It is not a fixed box: higher arm slots naturally produce more induced
vertical break, so the dead zone moves with the pitcher. Max Bay's Dynamic Dead
Zone conditions on release characteristics -- arm angle and height-scaled
extension -- and asks how far a pitch's actual acceleration sits from what those
characteristics predict.

Concept and framing per the FanGraphs piece on Andres Munoz being an analytical
blind spot; the DDZ model itself is Max Bay's. The article specifies a
4-dimensional multivariate normal over (arm_angle, height-scaled extension, ax,
az). The quantity we need from it is the conditional expectation
E[(ax,az) | arm_angle, scaled_ext], and for a joint Gaussian that conditional
mean is exactly the least-squares fit of the responses on the conditioners:

    E[Y | X=x] = mu_Y + Sigma_YX Sigma_XX^-1 (x - mu_X)   ==   OLS(Y ~ X)

So the coefficients below are the MVN conditional mean, stored in their
equivalent linear form because that round-trips through parquet cleanly. The
conditional (residual) SD is kept alongside so a delta can be expressed in
standard deviations rather than raw units when that is more useful.

Delta units are ft/sec^2 -- these are accelerations, not inches of break.
Positive `ddz_delta_az` = more vertical movement (more ride) than the pitcher's
slot and extension predict; negative = less ride than predicted.

Two variants are fitted:
  * base  -- conditions on arm_angle + scaled_extension
  * velo  -- additionally conditions on release_speed

The velocity variant is the one that resolves the Munoz puzzle in the article:
his four-seamer looks dead-zone-ish on the base model, but once his 99 mph is
accounted for, hitters are expecting more ride than he delivers.
"""
from __future__ import annotations

from . import paths

CONDITIONERS = {
    "": ["arm_angle", "scaled_extension"],
    "_velo": ["arm_angle", "scaled_extension", "release_speed"],
}
RESPONSES = ["ax", "az"]

# ax is arm-side signed: right-handers run the ball toward negative x. Mirror
# lefties so one league-wide fit serves both hands.
MIRROR_COLUMNS = ["ax"]

MIN_PITCHES = 200


def _mirror_lhp(df):
    out = df.copy()
    left = (out["p_throws"] == "L").to_numpy()
    for c in MIRROR_COLUMNS:
        if c in out.columns:
            out.loc[left, c] = -out.loc[left, c]
    return out


def _with_scaled_extension(df, season: int = paths.SEASON):
    """Attach height-scaled extension, joining pitcher height from bio."""
    from . import people

    bio = people.load_bio(season).set_index("pitcher_id")["height_ft"]
    out = df.copy()
    out["height_ft"] = out["pitcher"].map(bio)
    out["scaled_extension"] = out["release_extension"] / out["height_ft"]
    return out


def _ols(X, y):
    import numpy as np

    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def fit_ddz_baseline(season: int = paths.SEASON):
    """Fit per-pitch-type DDZ models; persist coefficients and residual SDs."""
    import pandas as pd

    from . import store

    df = store.query(
        "SELECT pitcher, pitch_type, p_throws, arm_angle, release_extension, "
        "release_speed, ax, az FROM pitches "
        "WHERE ax IS NOT NULL AND arm_angle IS NOT NULL AND pitch_type IS NOT NULL",
        season,
    )
    df = _mirror_lhp(_with_scaled_extension(df, season))

    rows = []
    for pitch_type, g in df.groupby("pitch_type", observed=True):
        if len(g) < MIN_PITCHES:
            continue
        row = {"pitch_type": pitch_type, "n": len(g)}
        for suffix, conds in CONDITIONERS.items():
            s = g.dropna(subset=conds + RESPONSES)
            if len(s) < MIN_PITCHES:
                continue
            X = s[conds].to_numpy()
            for resp in RESPONSES:
                y = s[resp].to_numpy()
                coef = _ols(X, y)
                resid = y - (coef[0] + X @ coef[1:])
                key = f"{resp}{suffix}"
                row[f"{key}_intercept"] = coef[0]
                for name, c in zip(conds, coef[1:]):
                    row[f"{key}_{name}"] = c
                row[f"{key}_resid_sd"] = float(resid.std())
            row[f"n{suffix}"] = len(s)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    paths.ensure_dirs()
    out.to_parquet(paths.ddz_baseline_path(season), index=False)
    return out


def load_ddz_baseline(season: int = paths.SEASON):
    import pandas as pd

    path = paths.ddz_baseline_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"no DDZ baseline for {season}; run `ingest build` or "
            "`ddz.fit_ddz_baseline()` first"
        )
    return pd.read_parquet(path)


def add_ddz_delta(df, season: int = paths.SEASON):
    """Append `ddz_delta_{ax,az}` and `ddz_delta_{ax,az}_velo` (ft/sec^2).

    Positive az delta = more ride than the pitcher's arm angle and scaled
    extension predict. Requires pitcher, pitch_type, p_throws, arm_angle,
    release_extension, release_speed, ax, az.
    """
    base = load_ddz_baseline(season).set_index("pitch_type")
    m = _mirror_lhp(_with_scaled_extension(df, season))
    coef = m[["pitch_type"]].join(base, on="pitch_type")

    out = df.copy()
    for suffix, conds in CONDITIONERS.items():
        for resp in RESPONSES:
            key = f"{resp}{suffix}"
            if f"{key}_intercept" not in coef.columns:
                continue
            expected = coef[f"{key}_intercept"].copy()
            for name in conds:
                expected = expected + coef[f"{key}_{name}"] * m[name]
            out[f"ddz_delta_{key}"] = m[resp] - expected
    return out
