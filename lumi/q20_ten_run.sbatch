#!/bin/bash -l
#SBATCH --job-name=cherryq-q20-10
#SBATCH --account=project_462001763
#SBATCH --reservation=quantumfinance
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=cherryq-q20-10-%j.out
#SBATCH --error=cherryq-q20-10-%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch lumi/q20_ten_run.sbatch <preflight-manifest.json>" >&2
  exit 2
fi

PROJECT=project_462001763
REPO_ROOT="${CHERRYQ_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
MANIFEST="$1"
RESULT_ROOT="${CHERRYQ_RESULT_ROOT:-/scratch/${PROJECT}/cherryq/q20-ten}"
OUTPUT="${RESULT_ROOT}/cherryq-ten-${SLURM_JOB_ID}.json"
SHOTS="${CHERRYQ_Q20_SHOTS:-1000}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing preflight manifest: $MANIFEST" >&2
  exit 2
fi

module use /appl/local/quantum/modulefiles
module --ignore_cache load fiqci-vtt-qiskit/18.0
export DEVICES=("radiance20")
: "${RUN_SETUP:?RUN_SETUP was not set by fiqci-vtt-qiskit/18.0}"
source "$RUN_SETUP"
: "${Q20_CORTEX_URL:?Q20_CORTEX_URL was not configured for radiance20}"

export PYTHONPATH="${REPO_ROOT}/src"

python -u -m cherryq.hardware.q20_ten \
  --mode run \
  --manifest "$MANIFEST" \
  --shots "$SHOTS" \
  --output "$OUTPUT"
