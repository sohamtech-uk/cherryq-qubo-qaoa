#!/bin/bash -l
#SBATCH --job-name=cherryq-q20
#SBATCH --account=project_462001763
#SBATCH --reservation=quantumfinance
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=cherryq-q20-%j.out
#SBATCH --error=cherryq-q20-%j.err

set -euo pipefail

PROJECT=project_462001763
REPO_ROOT="${CHERRYQ_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
MODE="${1:-bell}"
SHOTS="${CHERRYQ_Q20_SHOTS:-1000}"
RESULT_ROOT="${CHERRYQ_RESULT_ROOT:-/scratch/${PROJECT}/cherryq/q20}"
OUTPUT="${RESULT_ROOT}/${MODE}-${SLURM_JOB_ID}.json"

if [[ "$MODE" != "bell" && "$MODE" != "cherryq" ]]; then
  echo "Usage: sbatch lumi/q20_hardware.sbatch [bell|cherryq]" >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT"

module use /appl/local/quantum/modulefiles
module --ignore_cache load fiqci-vtt-qiskit-QxF
export DEVICES=("radiance20")
source "$RUN_SETUP"

export PYTHONPATH="${REPO_ROOT}/src"

echo "CherryQ Q20 mode: $MODE"
echo "Repository: $REPO_ROOT"
echo "Output: $OUTPUT"

python -u -m cherryq.hardware.q20 \
  --mode "$MODE" \
  --shots "$SHOTS" \
  --optimization-level 1 \
  --maxiter 80 \
  --seed 42 \
  --penalty 500 \
  --relaxation-multistart 32 \
  --output "$OUTPUT"
