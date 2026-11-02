#!/bin/bash
# Backfill 2024 + 2025 and refresh 2026 to the end of the regular season, for
# the six half-season timepoints in the trajectory work. backfill/refresh both
# rebuild parquet + angle/DDZ baselines + player bio for their season.
#
# Submit from the repo root:  sbatch scripts/backfill_seasons.sbatch

#SBATCH --account=hassy_472
#SBATCH --partition=debug
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=mlb-backfill
#SBATCH --output=backfill-%j.log

set -euo pipefail

module purge 2>/dev/null || true

# SLURM runs a spooled copy of this script, so BASH_SOURCE is not the repo.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${MLBPITCH_PYTHON:-/project2/hassy_472/brendajm/envs/mlb/bin/python}"

# A verify failure is recorded, not fatal: one season's quirk shouldn't stop
# the others. Read every verify block in the log; the job exits 1 if any failed.
failed=""
verify() {
    "$PYTHON" -m mlbpitch.ingest --season "$1" verify || failed="$failed $1"
}

cd "$REPO"
for season in 2024 2025; do
    echo "=== ${season} backfill $(date) ==="
    "$PYTHON" -m mlbpitch.ingest --season "$season" backfill
    verify "$season"
done

echo "=== 2026 refresh $(date) ==="
"$PYTHON" -m mlbpitch.ingest --season 2026 refresh
verify 2026
echo "=== done $(date) ==="

if [ -n "$failed" ]; then
    echo "VERIFY FAILED for:${failed}"
    exit 1
fi
