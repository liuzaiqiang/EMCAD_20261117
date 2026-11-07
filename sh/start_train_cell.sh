#!/usr/bin/env bash
# DSB18 and EM share the binary EMCAD trainer and checkpoint layout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export DATASET=Cell
export DATASET_NAME="${DATASET_NAME:-DSB18}"
case "${DATASET_NAME}" in DSB18|EM) ;; *) echo "[ERROR] DATASET_NAME must be DSB18 or EM" >&2; exit 2 ;; esac
export DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../../data/cell/target}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/model_pth/Cell}"
export IMG_SIZE="${IMG_SIZE:-256}"
export N_GPU="${N_GPU:-1}"
export USE_MULTI_SCALE_TRAINING="${USE_MULTI_SCALE_TRAINING:-0}"
if [[ "${DATASET_NAME}" == DSB18 ]]; then
  export MERGE_INSTANCE_MASKS="${MERGE_INSTANCE_MASKS:-1}"
else
  export MERGE_INSTANCE_MASKS="${MERGE_INSTANCE_MASKS:-0}"
fi
source "${SCRIPT_DIR}/start_train_polyp.sh"
