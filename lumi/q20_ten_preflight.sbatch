#!/bin/bash -l
#SBATCH --job-name=cherryq-q20-10-preflight
#SBATCH --account=project_462001763
#SBATCH --reservation=quantumfinance
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=cherryq-q20-10-preflight-%j.out
#SBATCH --error=cherryq-q20-10-preflight-%j.err

set -euo pipefail

PROJECT=project_462001763
REPO_ROOT="${CHERRYQ_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
RESULT_ROOT="${CHERRYQ_RESULT_ROOT:-/scratch/${PROJECT}/cherryq/q20-ten}"
RUN_DIR="${RESULT_ROOT}/preflight-${SLURM_JOB_ID}"

mkdir -p "$RUN_DIR"

module use /appl/local/quantum/modulefiles
module --ignore_cache load fiqci-vtt-qiskit/18.0
export DEVICES=("radiance20")
: "${RUN_SETUP:?RUN_SETUP was not set by fiqci-vtt-qiskit/18.0}"
source "$RUN_SETUP"
: "${Q20_CORTEX_URL:?Q20_CORTEX_URL was not configured for radiance20}"

export PYTHONPATH="${REPO_ROOT}/src"

python -u -m cherryq.hardware.q20_ten \
  --mode preflight \
  --output-dir "$RUN_DIR" \
  --optimization-level 3 \
  --seed-transpiler 42 \
  --maxiter 80 \
  --seed 42 \
  --penalty 250 \
  --relaxation-multistart 32

echo "PREFLIGHT_DIR=$RUN_DIR"
echo "MANIFEST=$RUN_DIR/ten-invoice-q20-preflight.json"
