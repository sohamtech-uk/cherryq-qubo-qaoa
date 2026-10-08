#!/bin/bash -l
#SBATCH --job-name=cherryq-aggregate
#SBATCH --account=project_462001763
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=cherryq-aggregate-%j.out
#SBATCH --error=cherryq-aggregate-%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch aggregate_results.sbatch <array-job-id>" >&2
  exit 2
fi

PROJECT=project_462001763
ARRAY_JOB_ID="$1"
REPO_ROOT="${CHERRYQ_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SIF="${CHERRYQ_LUMI_SIF:-/projappl/${PROJECT}/cherryq/cherryq-lumi-c.sif}"
RESULT_ROOT="${CHERRYQ_RESULT_ROOT:-/scratch/${PROJECT}/cherryq/results}"
RUN_DIR="${RESULT_ROOT}/${ARRAY_JOB_ID}"
OUTPUT="${RUN_DIR}/aggregate.json"

export PYTHONPATH="${REPO_ROOT}/src"

srun singularity exec \
  --bind="/pfs,/users,/projappl,/scratch,/project,/flash" \
  "$SIF" \
  python -m cherryq.lumi_aggregate \
    --input-dir "$RUN_DIR" \
    --output "$OUTPUT"

echo "Aggregate report: $OUTPUT"
