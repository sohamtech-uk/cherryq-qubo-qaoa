#!/bin/bash -l
#SBATCH --job-name=cherryq-qaoa
#SBATCH --account=project_462001763
#SBATCH --partition=small
#SBATCH --array=0-39%10
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cherryq-qaoa-%A_%a.out
#SBATCH --error=cherryq-qaoa-%A_%a.err

set -euo pipefail

PROJECT=project_462001763
REPO_ROOT="${CHERRYQ_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SIF="${CHERRYQ_LUMI_SIF:-/projappl/${PROJECT}/cherryq/cherryq-lumi-c.sif}"
RESULT_ROOT="${CHERRYQ_RESULT_ROOT:-/scratch/${PROJECT}/cherryq/results}"
RUN_DIR="${RESULT_ROOT}/${SLURM_ARRAY_JOB_ID}"

mkdir -p "$RUN_DIR"

if [[ ! -f "$SIF" ]]; then
  echo "Missing CherryQ LUMI container: $SIF" >&2
  exit 2
fi

if [[ ! -d "${REPO_ROOT}/src/cherryq" ]]; then
  echo "REPO_ROOT does not look like the CherryQ repository: $REPO_ROOT" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}/src"

echo "Node: $(hostname)"
echo "Job: ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "Container: $SIF"
echo "Results: $RUN_DIR"

srun singularity exec \
  --bind="/pfs,/users,/projappl,/scratch,/project,/flash" \
  "$SIF" \
  python -m cherryq.lumi_sweep \
    --output-dir "$RUN_DIR" \
    --seed-start 1 \
    --seed-count 100 \
    --penalty 500 \
    --methods standard,warm-start-x,warm-start \
    --p 1,2 \
    --maxiter 80 \
    --shots 2048 \
    --relaxation-multistart 32 \
    --warm-start-jitter 0.05 \
    --workers "${SLURM_CPUS_PER_TASK}"
