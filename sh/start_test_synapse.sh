#!/usr/bin/env bash
# Evaluate the selected Synapse checkpoint with its adjacent training config.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

CONDA_BASE="${CONDA_BASE:-/base/mambaforge}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/root/shared-nvme/lzq_conda/envs/sld_emcad}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"

CKPT="${CKPT:-}"
[[ -n "${CKPT}" && -f "${CKPT}" ]] || {
  echo "[ERROR] Set CKPT to an existing Synapse best.pth" >&2
  exit 1
}
CKPT="$(realpath "${CKPT}")"
CKPT_DIR="$(dirname "${CKPT}")"
[[ -f "${CKPT_DIR}/config.json" ]] || {
  echo "[ERROR] Training config not found: ${CKPT_DIR}/config.json" >&2
  exit 1
}

VOLUME_PATH="${VOLUME_PATH:-${PROJECT_DIR}/../../data/Synapse/test_vol_h5}"
LIST_DIR="${LIST_DIR:-${PROJECT_DIR}/../../data/Synapse/lists/lists_Synapse}"
[[ -d "${VOLUME_PATH}" && -f "${LIST_DIR}/test_vol.txt" ]] || {
  echo "[ERROR] Synapse test volumes or test_vol.txt not found" >&2
  exit 1
}

TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
RUN_ID="test_Synapse_${TS}_gpu${CUDA_VISIBLE_DEVICES}_RAND${RAND}"
RUN_DIR="${CKPT_DIR}/test_results/${RUN_ID}"
mkdir -p "${CKPT_DIR}/test_results"
mkdir "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/test.log"
PID_FILE="${RUN_DIR}/${RUN_ID}.pid"

printf '[INFO] CKPT=%s\n[INFO] CONFIG=%s\n[INFO] RUN_ID=%s\n[INFO] LOG_FILE=%s\n' \
  "${CKPT}" "${CKPT_DIR}/config.json" "${RUN_ID}" "${LOG_FILE}" | tee -a "${LOG_FILE}"

nohup env RUN_ID="${RUN_ID}" python -u test_synapse.py \
  --dataset Synapse \
  --checkpoint "${CKPT}" \
  --volume_path "${VOLUME_PATH}" \
  --list_dir "${LIST_DIR}" \
  --test_save_dir "${RUN_DIR}/outputs" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" > "${PID_FILE}"
echo "[INFO] PID=${PID} PID_FILE=${PID_FILE}"
