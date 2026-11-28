#!/usr/bin/env bash
# Bash 严格模式：命令失败、未定义变量或管道中间失败时立即退出，避免带错误路径继续启动任务。
set -euo pipefail

# 本次实验要验证的目的；也可在启动命令前用 RUN_PURPOSE=... 覆盖。
RUN_PURPOSE="${RUN_PURPOSE:-未填写（请设置 RUN_PURPOSE）}"
#205
#CONDA_BASE="/home/elham/anaconda3"
#CONDA_ENV_PREFIX="/home/mlf/anaconda3/envs/sld_emcad"

#209服务器
#CONDA_BASE="${CONDA_BASE:-/home/mlf/anaconda3}"
#CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/mlf/anaconda3/envs/sld_emcad}"


#210服务器
CONDA_BASE="/home/lzq/lzqEnvs/miniconda3"
CONDA_ENV_PREFIX="/home/lzq/lzqEnvs/miniconda3/envs/sld_emcad"


#autoDL
#CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
#CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/root/miniconda3/envs/sld_emcad}"




source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"
export PYTHONNOUSERSITE=1
PYTHON_BIN="${PYTHON_BIN:-python}"


# 解析脚本所在目录为项目绝对路径，并切换过去，使后续相对路径和PID文件位置稳定。
#PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_DIR}"


export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
N_GPU="${N_GPU:-1}"


export PYTHONUNBUFFERED=1

SEED="${SEED:-2222}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-12}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"

# ACDC 是4类心脏MRI分割；这些环境变量可在调用脚本前覆盖。
DATASET="ACDC"
BASE_LR="${BASE_LR:-1e-4}"
SUPERVISION="${SUPERVISION:-mutation}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ENCODER="pvt_v2_b2"
EXPANSION_FACTOR="2"
LGAG_KS="3"
ACTIVATION_MSCB="relu6"
KERNEL_SIZES=(1 3 5)
KERNEL_SIZES_DISPLAY="${KERNEL_SIZES[*]}"


DETERMINISTIC="${DETERMINISTIC:-1}"


# 数据根目录包含 train/valid/test；列表目录保存各划分病例名；模型输出固定写入 model_pth/ACDC。
ROOT_PATH="${ROOT_PATH:-${PROJECT_DIR}/../../../../data/ACDC}"
LIST_DIR="${LIST_DIR:-${ROOT_PATH}/lists_ACDC}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/model_pth/ACDC}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${PROJECT_DIR}/../../../../pretrained_pth/pvt}"

# 启动前逐项验证训练/验证目录、列表和PVTv2-B2预训练权重；失败统一返回退出码1。
test -d "${ROOT_PATH}/train" || { echo "[ERROR] ROOT_PATH/train not found: ${ROOT_PATH}/train"; exit 1; }
test -d "${ROOT_PATH}/valid" || { echo "[ERROR] ROOT_PATH/valid not found: ${ROOT_PATH}/valid"; exit 1; }
test -f "${LIST_DIR}/train.txt" || { echo "[ERROR] train list not found: ${LIST_DIR}/train.txt"; exit 1; }
test -f "${LIST_DIR}/valid.txt" || { echo "[ERROR] valid list not found: ${LIST_DIR}/valid.txt"; exit 1; }
test -f "${PRETRAINED_DIR}/pvt_v2_b2.pth" || {
  echo "[ERROR] pretrained model not found: ${PRETRAINED_DIR}/pvt_v2_b2.pth"
  echo "[ERROR] This Synapse-style launcher uses the pretrained EMCAD encoder."
  exit 1
}

# 秒级时间戳加随机十六进制后缀组成唯一运行标识，避免并发启动相互覆盖日志/PID文件。
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
RUN_ID="train_${DATASET}_${TS}_gpu_${CUDA_VISIBLE_DEVICES}_SEED_${SEED}_RAND_${RAND}"


# tee -a把配置写入日志；RUN_ID同时打印到终端，供stop_train_acdc.sh作为参数使用。
echo "[INFO] RUN_ID=${RUN_ID}"

RUN_DIR="$("${PYTHON_BIN}" -m lib.experiment_paths \
  "${OUTPUT_DIR}" "${DATASET}" "${DATASET}" "${ENCODER}" "${IMG_SIZE}" \
  "${BATCH_SIZE}" "${MAX_EPOCHS}" "${SUPERVISION}" "${BASE_LR}" "${EXPANSION_FACTOR}" \
  "${LGAG_KS}" "${ACTIVATION_MSCB}" "${SEED}" "${KERNEL_SIZES[@]}")"
LOG_FILE="${RUN_DIR}/launcher.log"
PID_FILE="${RUN_DIR}/${RUN_ID}.pid"

PARAM_NAMES=(
  CONDA_BASE CONDA_ENV_PREFIX PYTHON_BIN PYTHONNOUSERSITE
  PROJECT_DIR CUDA_VISIBLE_DEVICES
  DATASET TS RAND RUN_ID RUN_DIR
  LOG_FILE PID_FILE SEED IMG_SIZE BATCH_SIZE
  MAX_EPOCHS BASE_LR SUPERVISION NUM_WORKERS N_GPU
  DETERMINISTIC LIST_DIR ROOT_PATH OUTPUT_DIR PRETRAINED_DIR
  ENCODER EXPANSION_FACTOR LGAG_KS ACTIVATION_MSCB RUN_PURPOSE
  KERNEL_SIZES_DISPLAY
)
{
  echo "[INFO] parameters:"
  for name in "${PARAM_NAMES[@]}"; do
    printf '[INFO] %-24s=%s\n' "$name" "${!name}"
  done
  echo "---------------------------ready to train---------------------------------"
} | tee -a "${LOG_FILE}"

# 整个反斜杠块是一条后台命令：nohup抵抗终端断开，env注入RUN_ID用于停止时核验进程身份。
# -u和PYTHONUNBUFFERED共同减少日志缓冲；stdout追加到日志，stderr合并，stdin断开，&立即返回控制权。
nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u train_acdc.py \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_dir "${RUN_DIR}" \
  --run_name "${RUN_ID}" \
  --encoder "${ENCODER}" \
  --pretrained_dir "${PRETRAINED_DIR}" \
  --kernel_sizes "${KERNEL_SIZES[@]}" \
  --expansion_factor "${EXPANSION_FACTOR}" \
  --lgag_ks "${LGAG_KS}" \
  --activation_mscb "${ACTIVATION_MSCB}" \
  --supervision "${SUPERVISION}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --num_workers "${NUM_WORKERS}" \
  --n_gpu "${N_GPU}" \
  --deterministic "${DETERMINISTIC}" \
  --seed "${SEED}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# 保存刚启动后台任务的PID；脚本本身成功结束不代表训练完成，只表示启动和PID记录成功。
PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
# 输出PID文件和日志位置，方便监控及后续停止。
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] RUN_OUTPUT=${RUN_DIR}"
