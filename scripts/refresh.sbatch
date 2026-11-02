#!/bin/bash
# Daily incremental refresh of the pitch store.
#
# Fetches any newly completed game-dates and unconditionally re-fetches a
# trailing 10-day window, because Statcast revises recent games (pitch types
# get reclassified, measurements corrected). Parquet months are rebuilt only
# for dates whose content sha256 actually changed, then the angle/DDZ baselines
# and pitcher_dim are refit.
#
# This is a ~10 second job -- it only needs an allocation if you want it on a
# schedule. Otherwise just run:  mlbpitch refresh && mlbpitch verify
#
# SITE-SPECIFIC: --account and PYTHON below are for USC CARC. Edit both for
# another cluster; everything else derives from the script's own location.

#SBATCH --account=hassy_472
#SBATCH --partition=debug
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=mlb-refresh
#SBATCH --output=refresh-%j.log

set -euo pipefail

module purge 2>/dev/null || true

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MLBPITCH_PYTHON:-/project2/hassy_472/brendajm/envs/mlb/bin/python}"

cd "$REPO"
"$PYTHON" -m mlbpitch.ingest refresh
"$PYTHON" -m mlbpitch.ingest verify
