#!/usr/bin/env bash
# DSB18 and EM use the checkpoint-aware binary evaluator.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export DATASET=Cell
export DATASET_NAME="${DATASET_NAME:-DSB18}"
case "${DATASET_NAME}" in DSB18|EM) ;; *) echo "[ERROR] DATASET_NAME must be DSB18 or EM" >&2; exit 2 ;; esac
export DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../../data/cell/target}"
source "${SCRIPT_DIR}/start_test_polyp.sh"
