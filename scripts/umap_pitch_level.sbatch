#!/bin/bash
# Full pitch-level UMAP: every pitch thrown this season (~450k rows), fit and
# figure. Two independent fits, stratified by handedness (321k RHP + 129k LHP),
# so budget for both. Needs its own SLURM allocation -- the interactive Claude
# Code session runs inside a small memory cgroup (~4GB) and OOM-killed on the
# full dataset; the same script ran fine there on <15k-row smoke-test samples.
#
# random_state is set for reproducibility, which forces UMAP to single-thread
# the neighbor search + embedding (a known umap-learn behavior -- n_jobs is
# silently overridden to 1). More CPUs won't speed that part up; they're only
# headroom for the DuckDB/pandas prep step.
#
# SITE-SPECIFIC: --account and PYTHON below are for USC CARC.
#
# Submit from the repo root:
#   sbatch scripts/umap_pitch_level.sbatch

#SBATCH --account=hassy_472
#SBATCH --partition=main
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=mlb-umap-pitch
#SBATCH --output=umap_pitch_level-%j.log

set -euo pipefail

module purge 2>/dev/null || true

# SLURM copies the script into a spool dir before running it, so
# ${BASH_SOURCE[0]}'s directory is NOT the repo -- must use SLURM_SUBMIT_DIR
# (set to the cwd `sbatch` was invoked from; falls back to BASH_SOURCE for a
# direct, non-sbatch run of this script).
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${MLBPITCH_PYTHON:-/project2/hassy_472/brendajm/envs/mlb/bin/python}"
SCRIPT="${REPO}/scripts/umap_pitch_level.py"

echo "=============================================="
echo "Pitch-level UMAP -- mlb"
echo "Job: ${SLURM_JOB_ID:-<interactive>}   Node: $(hostname)"
echo "Started: $(date)"
echo "=============================================="

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: python not found or not executable: ${PYTHON}"
    exit 1
fi
if [ ! -s "${SCRIPT}" ]; then
    echo "ERROR: script not found: ${SCRIPT}"
    exit 1
fi

cd "$REPO"
"$PYTHON" "$SCRIPT"

echo ""
echo "=============================================="
echo "Finished: $(date)"
echo "Outputs: data/meta/umap_pitches.parquet, analysis/pitch_umap.{pdf,png}"
echo "=============================================="
