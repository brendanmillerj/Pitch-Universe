"""Derived pitch geometry from the Statcast trajectory parameters.

Statcast reports a nine-parameter constant-acceleration fit -- position
(x0,y0,z0), velocity (vx0,vy0,vz0) and acceleration (ax,ay,az) -- all defined
at the **y = 50 ft** tracking plane, not at the release point. `release_pos_y`
(~54 ft) is where the ball actually left the hand and is *not* the reference
frame for these terms; using it here would be wrong.

Approach angles are the pitch's direction of travel as it reaches the front of
home plate (y = 17/12 ft), obtained by propagating the fit forward under
constant acceleration.

Formulas per the FanGraphs primers on VAA and HAA:
    vy_f = -sqrt(vy0^2 - 2*ay*(y0 - yf))
    t    = (vy_f - vy0) / ay
    vz_f = vz0 + az*t        vx_f = vx0 + ax*t
    VAA  = -degrees(atan(vz_f / vy_f))
    HAA  = -degrees(atan(vx_f / vy_f))
"""
from __future__ import annotations

import numpy as np

Y0 = 50.0        # ft -- the plane the Statcast fit parameters are defined at
YF = 17.0 / 12.0  # ft -- front of home plate


def approach_angles(df):
    """Return (vaa, haa) in degrees, aligned to df.index.

    VAA is negative for a descending pitch: about -4.6 deg for a league-average
    four-seamer and about -9.6 deg for a curveball, so *less negative* is the
    "flatter" plane that plays well up in the zone.

    HAA is signed by handedness -- right-handers are positive, left-handers
    negative, near mirror images. Use `store.add_angles_adjusted` for a
    handedness-neutral, location-adjusted version before modelling.

    Rows with missing trajectory parameters yield NaN rather than raising.
    """
    vy0 = df["vy0"].to_numpy(dtype="float64")
    vz0 = df["vz0"].to_numpy(dtype="float64")
    vx0 = df["vx0"].to_numpy(dtype="float64")
    ax = df["ax"].to_numpy(dtype="float64")
    ay = df["ay"].to_numpy(dtype="float64")
    az = df["az"].to_numpy(dtype="float64")

    # Guard the radicand: it is positive for every row observed so far, but a
    # negative value would silently produce NaN and hide a units/frame change.
    disc = vy0**2 - 2.0 * ay * (Y0 - YF)
    with np.errstate(invalid="ignore", divide="ignore"):
        vy_f = -np.sqrt(np.where(disc >= 0, disc, np.nan))
        t = (vy_f - vy0) / ay
        vaa = -np.degrees(np.arctan((vz0 + az * t) / vy_f))
        haa = -np.degrees(np.arctan((vx0 + ax * t) / vy_f))

    return vaa, haa


def release_angles(df):
    """Return (vra, hra) in degrees -- the angles the ball leaves the hand at.

    Same trajectory fit as `approach_angles`, propagated *backwards* from the
    y=50 ft plane to each pitch's own `release_pos_y` (51-58 ft here) rather
    than forwards to the plate. The back-propagation time is negative and the
    ball is necessarily faster at release than at 50 ft, both of which the
    verification checks assert.

    VRA is the launch angle out of the hand: about -1.85 deg for a four-seamer
    (leaving slightly downward) and +0.50 deg for a curveball, which has to be
    lofted. Gravity then steepens every pitch in flight, which is why VAA is
    always well below VRA.

    HRA sign follows the same convention as HAA: negative is aimed toward a
    right-handed batter (who stands at negative plate_x), positive toward a
    lefty. Note this makes right-handed *pitchers* average positive HRA --
    they release from the third-base side (release_pos_x ~ -1.9) and must
    throw rightward to reach the plate. Handedness of the pitcher and the
    target are separate things here.

    Rows missing trajectory parameters or `release_pos_y` yield NaN.
    """
    vy0 = df["vy0"].to_numpy(dtype="float64")
    vz0 = df["vz0"].to_numpy(dtype="float64")
    vx0 = df["vx0"].to_numpy(dtype="float64")
    ax = df["ax"].to_numpy(dtype="float64")
    ay = df["ay"].to_numpy(dtype="float64")
    az = df["az"].to_numpy(dtype="float64")
    yr = df["release_pos_y"].to_numpy(dtype="float64")

    disc = vy0**2 - 2.0 * ay * (Y0 - yr)
    with np.errstate(invalid="ignore", divide="ignore"):
        vy_r = -np.sqrt(np.where(disc >= 0, disc, np.nan))
        t = (vy_r - vy0) / ay  # negative: stepping back toward the hand
        vra = -np.degrees(np.arctan((vz0 + az * t) / vy_r))
        hra = -np.degrees(np.arctan((vx0 + ax * t) / vy_r))

    return vra, hra


def n_invalid_discriminant(df) -> int:
    """Rows where the propagation is unsolvable -- expected to be 0."""
    vy0 = df["vy0"].to_numpy(dtype="float64")
    ay = df["ay"].to_numpy(dtype="float64")
    disc = vy0**2 - 2.0 * ay * (Y0 - YF)
    return int((disc < 0).sum())
