#!/bin/bash -l
set -euo pipefail

PROJECT=project_462001763
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DIR="/projappl/${PROJECT}/cherryq"
INSTALL_DIR="${CHERRYQ_LUMI_INSTALL_DIR:-$DEFAULT_DIR}"
SIF="${CHERRYQ_LUMI_SIF:-${INSTALL_DIR}/cherryq-lumi-c.sif}"

mkdir -p "$INSTALL_DIR"

module purge
module load CrayEnv
module load cotainr

echo "Building $SIF"
cotainr build "$SIF" --system=lumi-c --conda-env="${SCRIPT_DIR}/environment.yml"

echo "Validating container"
singularity exec "$SIF" python - <<'PY'
import qiskit
import numpy
import scipy
import sklearn
print("qiskit", qiskit.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
PY

echo "Container ready: $SIF"
