"""Ingest CLI: backfill | refresh | build | verify | pitcher.

    python -m mlbpitch.ingest backfill
    python -m mlbpitch.ingest refresh          # daily; re-fetches a trailing window
    python -m mlbpitch.ingest verify
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import threading
from datetime import datetime, timezone

from . import ddz, paths, people, savant, schedule, store

MAX_WORKERS = 5          # polite to a free public service; ~120 dates in 2-3 min
TRAILING_WINDOW = 10     # days always re-fetched by `refresh`

_local = threading.local()


def _session():
    """One requests.Session per worker thread."""
    import requests

    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

MANIFEST_COLUMNS = [
    "game_date", "n_games_expected", "n_games_seen", "n_pitches",
    "csv_sha256", "fetched_at", "status",
]


def load_manifest(season: int = paths.SEASON):
    import pandas as pd

    path = paths.manifest_path(season)
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_parquet(path)


def save_manifest(df, season: int = paths.SEASON):
    paths.ensure_dirs()
    df = df.sort_values("game_date").reset_index(drop=True)
    df.to_parquet(paths.manifest_path(season), index=False)
    return df


def _summarize(content: bytes, date: str, expected: set[int]) -> dict:
    """Cheap per-date stats without materializing all 113 columns."""
    import pandas as pd

    seen = pd.read_csv(io.BytesIO(content), usecols=["game_pk"])["game_pk"]
    seen_set = set(seen.astype(int).unique())
    missing = expected - seen_set
    return {
        "game_date": date,
        "n_games_expected": len(expected),
        "n_games_seen": len(seen_set),
        "n_pitches": int(len(seen)),
        "csv_sha256": hashlib.sha256(content).hexdigest(),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ok" if not missing else f"missing:{len(missing)}",
    }


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def fetch_dates(dates: list[str], season: int = paths.SEASON, verbose: bool = True) -> list[dict]:
    """Fetch, cache, and summarize a list of game-dates concurrently."""
    expected_map = schedule.expected_games_by_date(season)
    results: list[dict] = []

    def one(date: str) -> dict:
        content = savant.fetch_day(date, session=_session(), season=season)
        savant.write_raw(date, content, season)
        return _summarize(content, date, expected_map.get(date, set()))

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(one, d): d for d in dates}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            date = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001 - surfaced per date, not swallowed
                row = {
                    "game_date": date, "n_games_expected": 0, "n_games_seen": 0,
                    "n_pitches": 0, "csv_sha256": "",
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "status": f"error:{type(exc).__name__}: {exc}",
                }
            results.append(row)
            if verbose:
                print(f"  [{i:>3}/{len(dates)}] {row['game_date']} "
                      f"{row['n_pitches']:>5} pitches  "
                      f"{row['n_games_seen']}/{row['n_games_expected']} games  "
                      f"{row['status']}", flush=True)
    return results


def _merge_manifest(rows: list[dict], season: int):
    import pandas as pd

    old = load_manifest(season)
    new = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    keep = old[~old["game_date"].isin(new["game_date"])] if len(old) else old
    return save_manifest(pd.concat([keep, new], ignore_index=True), season)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_backfill(args):
    season = args.season
    paths.ensure_dirs()
    schedule.save_schedule(season)

    want = schedule.playable_dates(season)
    have = set(store.cached_dates(season))
    todo = [d for d in want if d not in have] if not args.force else want

    print(f"season {season}: {len(want)} completed game-dates, "
          f"{len(have)} already cached, {len(todo)} to fetch")
    if not todo:
        print("nothing to fetch")
    else:
        rows = fetch_dates(todo, season)
        _merge_manifest(rows, season)

    print("\nbuilding parquet partitions...")
    for stat in store.build_months(store.cached_dates(season), season):
        if stat["written"]:
            print(f"  month={stat['month']}  {stat['rows']:>7,} rows  "
                  f"{stat['dates']:>3} dates  {stat['bytes']/1e6:>6.1f} MB")

    dim = store.build_pitcher_dim(season)
    print(f"\npitcher_dim: {len(dim)} pitchers")
    base = store.fit_angle_baseline(season)
    print(f"angle baseline: {len(base)} pitch types fitted")
    bio = people.build_bio(season)
    dz = ddz.fit_ddz_baseline(season)
    print(f"player bio: {len(bio)} pitchers | DDZ baseline: {len(dz)} pitch types")
    return 0


def cmd_refresh(args):
    """New dates plus a trailing window, because Statcast revises recent games."""
    import pandas as pd

    season = args.season
    schedule.save_schedule(season)

    want = schedule.playable_dates(season)
    have = set(store.cached_dates(season))
    new = [d for d in want if d not in have]
    trailing = [d for d in want if d in have][-args.window:]
    todo = sorted(set(new) | set(trailing))

    print(f"refresh: {len(new)} new dates, {len(trailing)} trailing re-fetch "
          f"(window={args.window}) -> {len(todo)} requests")
    if not todo:
        print("up to date")
        return 0

    before = load_manifest(season).set_index("game_date")["csv_sha256"].to_dict()
    rows = fetch_dates(todo, season)
    _merge_manifest(rows, season)

    changed = [r["game_date"] for r in rows if before.get(r["game_date"]) != r["csv_sha256"]]
    print(f"\n{len(changed)} dates changed content: {changed or 'none'}")

    if changed:
        for stat in store.build_months(changed, season):
            if stat["written"]:
                print(f"  rebuilt month={stat['month']}  {stat['rows']:>7,} rows")
        store.build_pitcher_dim(season)
        store.fit_angle_baseline(season)  # baselines shift as the season accrues
        people.build_bio(season)          # incremental: only new pitchers
        ddz.fit_ddz_baseline(season)
    return 0


def cmd_build(args):
    """Rebuild parquet from the raw cache without touching the network."""
    season = args.season
    dates = store.cached_dates(season)
    print(f"rebuilding all months from {len(dates)} cached dates")
    for stat in store.build_months(dates, season):
        if stat["written"]:
            print(f"  month={stat['month']}  {stat['rows']:>7,} rows  "
                  f"{stat['bytes']/1e6:>6.1f} MB")
    store.build_pitcher_dim(season)
    base = store.fit_angle_baseline(season)
    print(f"angle baseline: {len(base)} pitch types fitted")
    bio = people.build_bio(season)
    dz = ddz.fit_ddz_baseline(season)
    print(f"player bio: {len(bio)} pitchers | DDZ baseline: {len(dz)} pitch types")
    return 0


def cmd_verify(args):
    """Reconcile the store against the schedule oracle and the dtype policy."""
    import pandas as pd
    from .schema import DEPRECATED_COLUMNS, MAX_ROWS_PER_DAY

    season = args.season
    ok = True

    man = load_manifest(season)
    expected = schedule.expected_games_by_date(season)

    print(f"=== verify season {season} ===")
    print(f"manifest dates: {len(man)}   schedule completed dates: {len(expected)}")

    # 1. every completed date present
    missing_dates = sorted(set(expected) - set(man["game_date"]))
    if missing_dates:
        ok = False
        print(f"FAIL  {len(missing_dates)} completed dates absent from store: "
              f"{missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}")
    else:
        print("PASS  every completed game-date is in the store")

    # 2. per-date game completeness, checked against what is actually stored
    #    rather than the status frozen into the manifest at fetch time
    stored = store.query(
        "SELECT strftime(game_date, '%Y-%m-%d') AS d, "
        "list(DISTINCT game_pk) AS pks FROM pitches GROUP BY d",
        season,
    )
    stored_map = {r.d: set(int(x) for x in r.pks) for r in stored.itertuples()}

    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    lag_cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.lag_days)).strftime("%Y-%m-%d")

    gaps, lagging = [], []
    for date, want_pks in sorted(expected.items()):
        missing = want_pks - stored_map.get(date, set())
        if not missing:
            continue
        (lagging if date >= lag_cutoff else gaps).append((date, sorted(missing)))

    if gaps:
        ok = False
        print(f"FAIL  {len(gaps)} dates missing games beyond the "
              f"{args.lag_days}-day publication window:")
        for date, pks in gaps[:10]:
            print(f"        {date}  missing game_pk {pks}")
    else:
        print("PASS  every date's game_pk set matches the schedule")

    if lagging:
        print(f"WARN  {len(lagging)} recent date(s) not yet published by Savant "
              f"(statsapi marks the game Final first; normal lag):")
        for date, pks in lagging:
            marker = "  <- today" if date == today else ""
            print(f"        {date}  awaiting game_pk {pks}{marker}")

    # 3. truncation canary
    at_cap = man[man["n_pitches"] >= MAX_ROWS_PER_DAY]
    if len(at_cap):
        ok = False
        print(f"FAIL  {len(at_cap)} dates at/over Savant's {MAX_ROWS_PER_DAY} row cap "
              "-- responses were silently truncated")
    else:
        peak = int(man["n_pitches"].max()) if len(man) else 0
        print(f"PASS  no date near the {MAX_ROWS_PER_DAY:,} cap (busiest: {peak:,})")

    # 4. null rates on the headline features
    feat = store.query(
        """
        SELECT count(*) AS n,
               avg(CASE WHEN arm_angle      IS NULL THEN 0 ELSE 1 END) AS arm_angle,
               avg(CASE WHEN spin_axis      IS NULL THEN 0 ELSE 1 END) AS spin_axis,
               avg(CASE WHEN release_speed  IS NULL THEN 0 ELSE 1 END) AS release_speed,
               avg(CASE WHEN release_extension IS NULL THEN 0 ELSE 1 END) AS release_extension,
               avg(CASE WHEN pfx_x          IS NULL THEN 0 ELSE 1 END) AS pfx_x
        FROM pitches
        """,
        season,
    ).iloc[0]
    cols = store.query("SELECT * FROM pitches LIMIT 0", season).columns

    print(f"\nstored pitches: {int(feat['n']):,}")
    for name in ["arm_angle", "spin_axis", "release_speed", "release_extension", "pfx_x"]:
        rate = float(feat[name]) * 100
        flag = "PASS" if rate >= 95 else "FAIL"
        if rate < 95:
            ok = False
        print(f"  {flag}  {name:<20} {rate:5.1f}% non-null")

    # 5. approach angles: physical sanity, handedness symmetry, de-confounding
    print("\napproach angles (vaa/haa):")
    ang = store.query(
        "SELECT pitch_name, p_throws, avg(vaa) AS vaa, avg(haa) AS haa, "
        "count(*) AS n, min(vaa) AS vmin, max(vaa) AS vmax "
        "FROM pitches WHERE vaa IS NOT NULL GROUP BY pitch_name, p_throws",
        season,
    )
    tot = ang.groupby("pitch_name").apply(
        lambda g: (g["vaa"] * g["n"]).sum() / g["n"].sum(), include_groups=False
    )

    ff = tot.get("4-Seam Fastball")
    cu = tot.get("Curveball")
    if ff is None or not (-5.5 <= ff <= -3.5):
        ok = False
        print(f"  FAIL  4-seam mean VAA {ff} outside expected [-5.5, -3.5]")
    else:
        print(f"  PASS  4-seam mean VAA {ff:+.2f} deg (league norm ~-4.6)")
    if cu is None or not (cu < ff - 3.0):
        ok = False
        print(f"  FAIL  curveball VAA {cu} not >3 deg steeper than 4-seam")
    else:
        print(f"  PASS  curveball VAA {cu:+.2f} deg, {ff - cu:.1f} deg steeper than 4-seam")

    # Range is checked on competitive pitches only. A 30 mph position-player
    # eephus legitimately descends at -33 deg, so a blanket bound would flag
    # correct physics; the wide bound below is the absolute sanity floor.
    rng = store.query(
        "SELECT min(vaa) AS lo, max(vaa) AS hi, "
        "count(*) FILTER (WHERE vaa < -20 OR vaa > 2) AS n_out, "
        "arg_max(plate_z, greatest(vaa - 2, -20 - vaa)) AS worst_z FROM pitches "
        "WHERE vaa IS NOT NULL AND release_speed >= 60",
        season,
    ).iloc[0]
    # "Unknown" (UN) is Statcast's tracking-failure bucket: 2025 has a 21.7 mph
    # UN at plate_z -57.6 ft (VAA -69). Classified pitches carry the check.
    known = ang[ang.pitch_name != "Unknown"]
    lo_all, hi_all = known["vmin"].min(), known["vmax"].max()
    # -20 rather than something tighter: a spiked curveball (72 mph, plate_z
    # -2.55 ft, i.e. bounced) reaches -16.2 legitimately. This is an outlier
    # guard; the mean-VAA assertions above are what actually validate the physics.
    # A handful of escapes is tolerated: 2024 has one 87.5 mph changeup tracked at
    # plate_z 12.4 ft (over the backstop) -- a real throw, not a frame error.
    if rng.n_out > 5:
        ok = False
        print(f"  FAIL  {int(rng.n_out)} pitches >=60mph escape vaa [-20, 2] "
              f"(range [{rng.lo:.1f}, {rng.hi:.1f}])")
    elif rng.n_out:
        print(f"  PASS  vaa range for >=60mph pitches: {int(rng.n_out)} outlier(s) "
              f"tolerated (range [{rng.lo:.1f}, {rng.hi:.1f}], worst at "
              f"plate_z {rng.worst_z:.1f} ft)")
    else:
        print(f"  PASS  vaa range for >=60mph pitches [{rng.lo:.1f}, {rng.hi:.1f}]")
    if lo_all < -45 or hi_all > 5:
        ok = False
        print(f"  FAIL  vaa range overall [{lo_all:.1f}, {hi_all:.1f}] is unphysical")
    else:
        print(f"  PASS  vaa overall [{lo_all:.1f}, {hi_all:.1f}] (slow lobs are steep, expected)")

    # HAA should be near mirror-image across handedness -- an inverted sign
    # convention is the likeliest silent failure and would roughly double the
    # residual. Restricted to pitch types with real samples in both hands:
    # Eephus has 9 LHP vs 664 RHP, so its "asymmetry" is pure noise. The
    # tolerance is relative -- |L+R| against |L|+|R| -- because a flipped sign
    # makes those two equal (ratio 1) at any magnitude, while genuine population
    # differences stay small (2024 Slurve: |L+R| 0.76 of 7.37, ratio 0.10, from
    # a handful of pitchers per hand; an absolute 0.75 cut flagged it).
    hh = ang.pivot_table(index="pitch_name", columns="p_throws", values="haa")
    nn = ang.pivot_table(index="pitch_name", columns="p_throws", values="n")
    if {"L", "R"} <= set(hh.columns):
        keep = (nn["L"] >= 500) & (nn["R"] >= 500)
        sym = ((hh["L"] + hh["R"]).abs() / (hh["L"].abs() + hh["R"].abs()))[keep].dropna()
        worst_type = sym.idxmax() if len(sym) else None
        worst = sym.max() if len(sym) else None
    else:
        worst = worst_type = None
    if worst is None or worst > 0.5:
        ok = False
        print(f"  FAIL  HAA not handedness-symmetric (worst |L+R|/(|L|+|R|) = {worst} "
              f"on {worst_type}) -- check the sign convention")
    else:
        print(f"  PASS  HAA mirror-symmetric across handedness over {len(sym)} "
              f"well-sampled types (worst |L+R|/(|L|+|R|) = {worst:.2f} on {worst_type})")

    # The adjustment must actually remove the location signal it exists to remove.
    raw = store.query(
        "SELECT pitch_type, p_throws, vaa, haa, vra, hra, plate_x, plate_z, release_pos_x "
        "FROM pitches WHERE vaa IS NOT NULL AND pitch_type IS NOT NULL",
        season,
    )
    adj = store.add_angles_adjusted(raw, season)
    mirrored = store._mirror_lhp(adj)
    # Each adjustment must kill the confounder it was built to remove.
    for angle, confounder, before in [
        ("vaa", "plate_z", 0.89), ("vra", "plate_z", 0.79),
        ("haa", "release_pos_x", 0.95), ("hra", "release_pos_x", 0.73),
    ]:
        col = f"{angle}_above_avg"
        after = abs(mirrored[col].corr(mirrored[confounder]))
        if not (after <= 0.05):
            ok = False
            print(f"  FAIL  {col} vs {confounder}: |r| = {after:.3f}, "
                  "adjustment did not de-confound")
        else:
            print(f"  PASS  {col} vs {confounder}: |r| {before:.2f} -> {after:.3f}")

    # Release angles must be flatter than approach angles: gravity only ever
    # steepens a pitch in flight, so VRA > VAA for every pitch type.
    va = store.query(
        "SELECT pitch_name, avg(vra) AS vra, avg(vaa) AS vaa, count(*) AS n "
        "FROM pitches WHERE vra IS NOT NULL GROUP BY pitch_name HAVING count(*) > 1000",
        season,
    )
    bad = va[va["vra"] <= va["vaa"]]
    if len(bad):
        ok = False
        print(f"  FAIL  VRA not above VAA for: {list(bad.pitch_name)} "
              "-- gravity can only steepen a pitch")
    else:
        print(f"  PASS  VRA > VAA for all {len(va)} pitch types "
              f"(mean gain {(va.vra - va.vaa).mean():.1f} deg from gravity in flight)")

    # 6. dynamic dead zone
    print("\ndynamic dead zone:")
    try:
        bio = people.load_bio(season)
        dz = ddz.load_ddz_baseline(season)
    except FileNotFoundError as exc:
        ok = False
        print(f"  FAIL  {exc}")
    else:
        n_dim = len(pd.read_parquet(paths.pitcher_dim_path(season)))
        n_ht = int(bio["height_ft"].notna().sum())
        if n_ht < n_dim:
            ok = False
            print(f"  FAIL  height missing for {n_dim - n_ht} of {n_dim} pitchers "
                  "-- scaled_extension would be null for them")
        else:
            print(f"  PASS  height present for all {n_dim} pitchers "
                  f"({bio.height_ft.min():.2f}-{bio.height_ft.max():.2f} ft)")

        # The premise of the whole model: a higher arm slot yields more ride.
        # If this coefficient is not positive for fastballs, the fit is wrong.
        ff = dz[dz["pitch_type"].isin(["FF", "SI"])]
        badslot = ff[ff["az_arm_angle"] <= 0]
        if len(badslot):
            ok = False
            print(f"  FAIL  az~arm_angle not positive for {list(badslot.pitch_type)} "
                  "-- higher slots must produce more ride")
        else:
            print("  PASS  higher arm slot -> more ride for FF/SI "
                  f"({', '.join(f'{r.pitch_type} {r.az_arm_angle:+.3f}' for r in ff.itertuples())})")

        # Deltas are residuals, so the league mean must sit at zero.
        sample = store.query(
            "SELECT pitcher, pitch_type, p_throws, arm_angle, release_extension, "
            "release_speed, ax, az FROM pitches "
            "WHERE ax IS NOT NULL AND arm_angle IS NOT NULL AND pitch_type='FF'",
            season,
        )
        dd = ddz.add_ddz_delta(sample, season)
        for col in ["ddz_delta_ax", "ddz_delta_az",
                    "ddz_delta_ax_velo", "ddz_delta_az_velo"]:
            mu = dd[col].mean()
            if abs(mu) > 0.05:
                ok = False
                print(f"  FAIL  {col} league mean {mu:+.3f}, expected ~0 for a residual")
            else:
                print(f"  PASS  {col} league mean {mu:+.3f} (sd {dd[col].std():.2f} ft/s^2)")

    # 7. deprecated columns really are gone
    leftover = [c for c in DEPRECATED_COLUMNS if c in cols]
    print(f"\n{'FAIL' if leftover else 'PASS'}  deprecated columns dropped"
          f"{': ' + str(leftover) if leftover else ''}")
    if leftover:
        ok = False

    print(f"\n{'ALL CHECKS PASSED' if ok else 'VERIFICATION FAILED'}")
    return 0 if ok else 1


def cmd_pitcher(args):
    """Round-trip check: local store vs a live single-pitcher Savant fetch."""
    season = args.season
    local = store.pitcher(args.pitcher_id, season)
    print(f"local store: {len(local):,} pitches")

    if args.live:
        remote = savant.fetch_pitcher_season(args.pitcher_id, season)
        print(f"live savant: {len(remote):,} pitches")
        match = len(local) == len(remote)
        print(f"{'PASS' if match else 'FAIL'}  row counts "
              f"{'agree' if match else 'DISAGREE'}")
        lv, rv = local["release_speed"].mean(), remote["release_speed"].mean()
        print(f"  mean velo local={lv:.3f} live={rv:.3f} delta={abs(lv-rv):.4f}")

    if len(local):
        print()
        adj = {}
        try:
            local = store.add_angles_adjusted(local, season)
            adj |= {f"{a}_adj": (f"{a}_above_avg", "mean") for a in ("vaa", "haa")}
        except FileNotFoundError:
            pass
        try:
            local = ddz.add_ddz_delta(local, season)
            adj |= {"ddz_h": ("ddz_delta_ax", "mean"),
                    "ddz_v": ("ddz_delta_az", "mean"),
                    "ddz_h_velo": ("ddz_delta_ax_velo", "mean"),
                    "ddz_v_velo": ("ddz_delta_az_velo", "mean")}
        except FileNotFoundError:
            pass
        summary = local.groupby("pitch_name", observed=True).agg(
            n=("release_speed", "size"),
            velo=("release_speed", "mean"),
            arm_angle=("arm_angle", "mean"),
            spin_rpm=("release_spin_rate", "mean"),
            vra=("vra", "mean"),
            vaa=("vaa", "mean"),
            hra=("hra", "mean"),
            haa=("haa", "mean"),
            **adj,
        ).sort_values("n", ascending=False)
        # cast off float32 before rounding so the display isn't 97.300003
        summary = summary.astype({c: "float64" for c in summary.columns if c != "n"})
        print(summary.round(1).to_string())
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlbpitch.ingest")
    p.add_argument("--season", type=int, default=paths.SEASON)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backfill", help="fetch all completed dates not yet cached")
    b.add_argument("--force", action="store_true", help="re-fetch even if cached")
    b.set_defaults(func=cmd_backfill)

    r = sub.add_parser("refresh", help="fetch new dates + re-fetch trailing window")
    r.add_argument("--window", type=int, default=TRAILING_WINDOW)
    r.set_defaults(func=cmd_refresh)

    bd = sub.add_parser("build", help="rebuild parquet from raw cache (no network)")
    bd.set_defaults(func=cmd_build)

    v = sub.add_parser("verify", help="reconcile store against the schedule oracle")
    v.add_argument("--lag-days", type=int, default=2,
                   help="dates this recent may lag Savant publication (warn, not fail)")
    v.set_defaults(func=cmd_verify)

    pi = sub.add_parser("pitcher", help="inspect one pitcher; --live to compare to Savant")
    pi.add_argument("pitcher_id", type=int)
    pi.add_argument("--live", action="store_true")
    pi.set_defaults(func=cmd_pitcher)

    args = p.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
