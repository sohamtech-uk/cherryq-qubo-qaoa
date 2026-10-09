#!/bin/bash -l
#SBATCH --job-name=cherryq-q20-cost-audit
#SBATCH --account=project_462001763
#SBATCH --reservation=quantumfinance
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=cherryq-q20-cost-audit-%j.out
#SBATCH --error=cherryq-q20-cost-audit-%j.err

set -euo pipefail
PROJECT=project_462001763
REPO_ROOT="${CHERRYQ_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
RESULT_ROOT="${CHERRYQ_RESULT_ROOT:-/scratch/${PROJECT}/cherryq/q20-ten}"
RUN_DIR="${RESULT_ROOT}/cost-order-audit-${SLURM_JOB_ID}"

module use /appl/local/quantum/modulefiles
module --ignore_cache load fiqci-vtt-qiskit/18.0
export DEVICES=("radiance20")
: "${RUN_SETUP:?RUN_SETUP was not set by fiqci-vtt-qiskit/18.0}"
source "$RUN_SETUP"
: "${Q20_CORTEX_URL:?Q20_CORTEX_URL was not configured for radiance20}"
export PYTHONPATH="${REPO_ROOT}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Compilation and ideal validation only: no backend.run call, no QPU submission.
python -u -m cherryq.hardware.q20_routing_audit \
  --output-dir "$RUN_DIR" --seeds 42 7 123

echo "ROUTING_AUDIT=$RUN_DIR/routing-audit.json"
