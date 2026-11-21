#!/usr/bin/env bash
# Run sibling EMCAD projects in order. Each project uses its own seed queue,
# launcher, environment, data paths, logs, and checkpoints.
#
# Manifest (UTF-8, one project per line):
#   folder_name|synapse|2222 3407 5678
#   folder_name|isic|2222 3407|ISIC2017
# Empty lines and lines beginning with # are ignored. Folder names are
# relative to the parent of this script's project directory.
#
#   bash sh/run_project_queue.sh --check sh/project_queue.txt
#   nohup bash sh/run_project_queue.sh sh/project_queue.txt >project_queue_start.log 2>&1 < /dev/null &

set -euo pipefail

usage() {
  echo "Usage: bash sh/run_project_queue.sh [--check] <manifest.txt>" >&2
  exit 2
}

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
  shift
fi
[[ $# == 1 ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# Usually the parent containing this project's sibling experiment folders.
# PROJECT_PARENT_DIR can point at another parent when projects live elsewhere.
PROJECT_PARENT="$(cd "${PROJECT_PARENT_DIR:-${SCRIPT_DIR}/../..}" && pwd -P)"
MANIFEST_ARG="$1"
[[ -f "${MANIFEST_ARG}" && -r "${MANIFEST_ARG}" ]] || {
  echo "[ERROR] Cannot read manifest: ${MANIFEST_ARG}" >&2
  exit 2
}
MANIFEST="$(cd "$(dirname "${MANIFEST_ARG}")" && pwd -P)/$(basename "${MANIFEST_ARG}")"

trim_field() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

PROJECT_DIRS=()
DATASET_KEYS=()
SEED_LISTS=()
DATASET_NAMES=()
DISPLAY_NAMES=()
line_number=0
while IFS= read -r line || [[ -n "${line}" ]]; do
  line_number=$((line_number + 1))
  line="${line%$'\r'}"  # Accept files edited on Windows.
  [[ -z "${line//[[:space:]]/}" || "${line}" =~ ^[[:space:]]*# ]] && continue
  separators="${line//[^|]/}"
  if (( ${#separators} < 2 || ${#separators} > 3 )); then
    echo "[ERROR] Manifest line ${line_number}: expected 3 or 4 pipe-separated fields" >&2
    exit 2
  fi
  IFS='|' read -r folder key seed_text dataset_name extra <<< "${line}"
  folder="$(trim_field "${folder}")"
  key="$(trim_field "${key}")"
  seed_text="$(trim_field "${seed_text}")"
  dataset_name="$(trim_field "${dataset_name:-}")"
  if [[ -z "${folder}" || -z "${key}" || -z "${seed_text}" || -n "${extra:-}" ]]; then
    echo "[ERROR] Manifest line ${line_number}: expected folder|dataset|seeds[|DATASET_NAME]" >&2
    exit 2
  fi
  if [[ "${folder}" == */* || "${folder}" == '.' || "${folder}" == '..' ]]; then
    echo "[ERROR] Manifest line ${line_number}: folder must be a direct sibling directory name" >&2
    exit 2
  fi
  case "${key,,}" in
    synapse|acdc|polyp|busi|isic|cell) key="${key,,}" ;;
    *) echo "[ERROR] Manifest line ${line_number}: unsupported dataset: ${key}" >&2; exit 2 ;;
  esac
  if [[ ! -d "${PROJECT_PARENT}/${folder}" ]]; then
    echo "[ERROR] Manifest line ${line_number}: project directory missing: ${PROJECT_PARENT}/${folder}" >&2
    exit 2
  fi
  project_dir="$(cd "${PROJECT_PARENT}/${folder}" && pwd -P)"
  if [[ "$(dirname "${project_dir}")" != "${PROJECT_PARENT}" ]]; then
    echo "[ERROR] Manifest line ${line_number}: folder resolves outside the project parent" >&2
    exit 2
  fi
  if [[ ! -f "${project_dir}/sh/run_seed_queue.sh" || ! -f "${project_dir}/sh/start_train_${key}.sh" ]]; then
    echo "[ERROR] Manifest line ${line_number}: missing seed queue or ${key} launcher in ${project_dir}/sh" >&2
    exit 2
  fi
  # Validate the launcher and queue without starting training.
  if ! bash -n "${project_dir}/sh/run_seed_queue.sh" ||
     ! bash -n "${project_dir}/sh/start_train_${key}.sh"; then
    echo "[ERROR] Manifest line ${line_number}: Bash syntax error in ${project_dir}/sh" >&2
    exit 2
  fi
  if [[ "${key}" == "cell" ]]; then
    [[ -f "${project_dir}/sh/start_train_polyp.sh" ]] || {
      echo "[ERROR] Manifest line ${line_number}: Cell launcher requires start_train_polyp.sh" >&2
      exit 2
    }
    bash -n "${project_dir}/sh/start_train_polyp.sh" || {
      echo "[ERROR] Manifest line ${line_number}: Bash syntax error in Cell's shared Polyp launcher" >&2
      exit 2
    }
  fi
  read -r -a seeds <<< "${seed_text}"
  for seed in "${seeds[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || {
      echo "[ERROR] Manifest line ${line_number}: invalid seed: ${seed}" >&2
      exit 2
    }
  done
  if [[ -n "${dataset_name}" && ! "${dataset_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[ERROR] Manifest line ${line_number}: invalid DATASET_NAME: ${dataset_name}" >&2
    exit 2
  fi
  PROJECT_DIRS+=("${project_dir}")
  DATASET_KEYS+=("${key}")
  SEED_LISTS+=("${seed_text}")
  DATASET_NAMES+=("${dataset_name}")
  DISPLAY_NAMES+=("${folder}")
done < "${MANIFEST}"

(( ${#PROJECT_DIRS[@]} > 0 )) || { echo "[ERROR] Manifest has no projects" >&2; exit 2; }
echo "[PROJECT_QUEUE] preflight OK; projects=${#PROJECT_DIRS[@]}; parent=${PROJECT_PARENT}"
for i in "${!PROJECT_DIRS[@]}"; do
  printf '[PROJECT_QUEUE] %d/%d folder=%s dataset=%s DATASET_NAME=%s seeds=%s\n' \
    "$((i + 1))" "${#PROJECT_DIRS[@]}" "${DISPLAY_NAMES[i]}" "${DATASET_KEYS[i]}" \
    "${DATASET_NAMES[i]:--}" "${SEED_LISTS[i]}"
done
(( CHECK_ONLY == 0 )) || exit 0

PROJECT_CONTINUE_ON_ERROR="${PROJECT_CONTINUE_ON_ERROR:-0}"
[[ "${PROJECT_CONTINUE_ON_ERROR}" == "0" || "${PROJECT_CONTINUE_ON_ERROR}" == "1" ]] || {
  echo "[ERROR] PROJECT_CONTINUE_ON_ERROR must be 0 or 1" >&2
  exit 2
}
LOG_DIR="${PROJECT_QUEUE_LOG_DIR:-${SCRIPT_DIR}/../logs}"
mkdir -p "${LOG_DIR}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
QUEUE_LOG="${LOG_DIR}/project_queue_${RUN_STAMP}.log"
STATUS_FILE="${LOG_DIR}/project_queue_${RUN_STAMP}.tsv"
printf 'index\tfolder\tdataset\tdataset_name\tseeds\texit_code\tfinished_at\n' > "${STATUS_FILE}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1
echo "[PROJECT_QUEUE] manifest=${MANIFEST}"
echo "[PROJECT_QUEUE] log=${QUEUE_LOG}"
echo "[PROJECT_QUEUE] status=${STATUS_FILE}"

ACTIVE_QUEUE_PID=""
stop_queue() {
  echo "[PROJECT_QUEUE] interrupted; active seed queue PID=${ACTIVE_QUEUE_PID:-none}"
  if [[ -n "${ACTIVE_QUEUE_PID}" ]] && kill -0 "${ACTIVE_QUEUE_PID}" 2>/dev/null; then
    kill -TERM "${ACTIVE_QUEUE_PID}" 2>/dev/null || true
    wait "${ACTIVE_QUEUE_PID}" 2>/dev/null || true
  fi
  exit 130
}
trap stop_queue INT TERM

failed=0
for i in "${!PROJECT_DIRS[@]}"; do
  read -r -a seeds <<< "${SEED_LISTS[i]}"
  echo "[PROJECT_QUEUE] START $((i + 1))/${#PROJECT_DIRS[@]} ${DISPLAY_NAMES[i]} $(date '+%F %T')"
  (
    cd "${PROJECT_DIRS[i]}"
    if [[ -n "${DATASET_NAMES[i]}" ]]; then
      export DATASET_NAME="${DATASET_NAMES[i]}"
    else
      unset DATASET_NAME
    fi
    exec bash sh/run_seed_queue.sh "${DATASET_KEYS[i]}" "${seeds[@]}"
  ) &
  ACTIVE_QUEUE_PID=$!
  if wait "${ACTIVE_QUEUE_PID}"; then
    status=0
  else
    status=$?
    failed=$((failed + 1))
  fi
  ACTIVE_QUEUE_PID=""
  printf '%d\t%s\t%s\t%s\t%s\t%d\t%s\n' \
    "$((i + 1))" "${DISPLAY_NAMES[i]}" "${DATASET_KEYS[i]}" \
    "${DATASET_NAMES[i]:--}" "${SEED_LISTS[i]}" "${status}" "$(date '+%F %T')" >> "${STATUS_FILE}"
  echo "[PROJECT_QUEUE] END ${DISPLAY_NAMES[i]} exit_code=${status} $(date '+%F %T')"
  if (( status != 0 && PROJECT_CONTINUE_ON_ERROR == 0 )); then
    echo "[PROJECT_QUEUE] stopped after failure; remaining projects were not started"
    exit "${status}"
  fi
done
echo "[PROJECT_QUEUE] finished; failed_projects=${failed}; status=${STATUS_FILE}"
(( failed == 0 ))
