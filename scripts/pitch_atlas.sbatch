#!/bin/bash
# Full pitch atlas: snRNAseq-style pipeline over every pitch this season
# (~450k rows), run twice -- once per handedness. Stages: QC -> Leiden
# overcluster (res 2.0) -> majority-vote annotation -> fastball subcluster
# (res 0.4) -> DDZ signature -> UMAP + figures.
#
# Needs its own SLURM allocation: the interactive Claude Code session runs in a
# small memory cgroup (~4GB) and OOM-killed on the full dataset. Smoke tests run
# fine there via ATLAS_SAMPLE.
#
# random_state is set for reproducibility, which forces UMAP to single-thread
# the neighbour search + embedding (a known umap-learn behavior -- n_jobs is
# silently overridden to 1). Extra CPUs only help the DuckDB/pandas prep and the
# pynndescent/Leiden stages.
#
# SITE-SPECIFIC: --account and PYTHON below are for USC CARC.
#
# Submit from the repo root:
#   sbatch scripts/pitch_atlas.sbatch

#SBATCH --account=hassy_472
#SBATCH --partition=main
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=mlb-atlas
#SBATCH --output=pitch_atlas-%j.log

set -euo pipefail

module purge 2>/dev/null || true

# SLURM copies the script into a spool dir before running it, so
# ${BASH_SOURCE[0]}'s directory is NOT the repo -- must use SLURM_SUBMIT_DIR
# (the cwd `sbatch` was invoked from; falls back to BASH_SOURCE for a direct,
# non-sbatch run of this script).
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${MLBPITCH_PYTHON:-/project2/hassy_472/brendajm/envs/mlb/bin/python}"
SCRIPT="${REPO}/scripts/pitch_atlas.py"

echo "=============================================="
echo "Pitch atlas -- mlb"
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
echo "Outputs:"
echo "  data/meta/pitch_atlas.parquet"
echo "  data/meta/pitch_atlas_subclusters.parquet"
echo "  analysis/pitch_umap.{pdf,png}"
echo "  analysis/pitch_annotation_qc.{pdf,png}"
echo "  analysis/pitch_label_confusion.{pdf,png}"
echo "  analysis/pitch_substates.{pdf,png}"
echo "  analysis/pitch_atlas_report.txt"
echo "=============================================="
