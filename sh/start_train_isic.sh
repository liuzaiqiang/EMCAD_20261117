#!/usr/bin/env bash
# Bash严格模式让路径检查、环境激活或后台命令准备失败时立即退出。
set -euo pipefail

# 本次实验要验证的目的；也可在启动命令前用 RUN_PURPOSE=... 覆盖。
RUN_PURPOSE="${RUN_PURPOSE:-未填写（请设置 RUN_PURPOSE）}"

# 固定项目根和工作目录，避免调用者所在目录影响相对路径。
#PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


cd "${PROJECT_DIR}"




#205
#CONDA_BASE="/home/elham/anaconda3"
#CONDA_ENV_PREFIX="/home/mlf/anaconda3/envs/sld_emcad"

#209服务器
#CONDA_BASE="${CONDA_BASE:-/home/mlf/anaconda3}"
#CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/mlf/anaconda3/envs/sld_emcad}"


#210服务器
#CONDA_BASE="/home/lzq/lzqEnvs/miniconda3"
#CONDA_ENV_PREFIX="/home/lzq/lzqEnvs/miniconda3/envs/sld_emcad"


#autoDL
#CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
#CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/root/miniconda3/envs/sld_emcad}"





source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"
export PYTHONNOUSERSITE=1

PYTHON_BIN="${PYTHON_BIN:-python}"


# 限定GPU并关闭Python标准流缓冲，便于实时观察nohup日志。
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
N_GPU="${N_GPU:-1}"
export PYTHONUNBUFFERED=1

# DATASET_NAME只允许ISIC2017或ISIC2018，默认训练ISIC2018。
DATASET="ISIC"
DATASET_NAME="${DATASET_NAME:-ISIC2018}"

# 二分类训练、验证、优化、阈值与学习率调度参数；外部同名环境变量优先。
IMG_SIZE="${IMG_SIZE:-352}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BASE_LR="${BASE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLIP="${CLIP:-0.5}"
THRESHOLD="${THRESHOLD:-0.5}"
SCHEDULER="${SCHEDULER:-constant}"
MIN_LR="${MIN_LR:-1e-6}"

# 数据加载、可复现性、保存/验证频率和调试上限；0上限表示由Python处理为不限制。
NUM_WORKERS="${NUM_WORKERS:-8}"
SCALE_RATES=(0.75 1.0 1.25)
SCALE_RATES_DISPLAY="${SCALE_RATES[*]}"
KERNEL_SIZES=(1 3 5)
KERNEL_SIZES_DISPLAY="${KERNEL_SIZES[*]}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
# 中间 epoch_N.pth 保存已停用；保留原配置便于审核。
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VALID_CASES="${MAX_VALID_CASES:-0}"
DEVICE="${DEVICE:-cuda}"

# EMCAD编码器和解码器关键结构参数；paper监督用于二分类论文五项损失。
ENCODER="${ENCODER:-pvt_v2_b2}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-2}"
LGAG_KS="${LGAG_KS:-3}"
ACTIVATION_MSCB="${ACTIVATION_MSCB:-relu6}"
SUPERVISION="${SUPERVISION:-paper}"


# prepared数据、模型输出与PVT预训练权重目录。
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../../../../data/isic/target}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/model_pth/ISIC}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${PROJECT_DIR}/../../../../pretrained_pth/pvt}"

# case白名单验证数据集名称；不匹配时打印原因并以退出码1结束。
case "${DATASET_NAME}" in
  ISIC2017|ISIC2018)
    ;;
  *)
    echo "[ERROR] DATASET_NAME must be ISIC2017 or ISIC2018"
    exit 1
    ;;
esac

# 依次检查训练/验证图像与掩膜、固定划分清单和划分摘要，防止缺文件时才在Python深处失败。
test -d "${DATA_ROOT}/${DATASET_NAME}/train/images" || {
  echo "[ERROR] train images not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/train/images"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/train/masks" || {
  echo "[ERROR] train masks not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/train/masks"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/val/images" || {
  echo "[ERROR] val images not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/val/images"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/val/masks" || {
  echo "[ERROR] val masks not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/val/masks"
  exit 1
}

test -f "${DATA_ROOT}/${DATASET_NAME}/split_manifest.csv" || {
  echo "[ERROR] split_manifest.csv not found"
  exit 1
}

test -f "${DATA_ROOT}/${DATASET_NAME}/split_summary.json" || {
  echo "[ERROR] split_summary.json not found"
  exit 1
}

# 仅PVTv2编码器要求本地同名预训练权重文件。
if [[ "${ENCODER}" == pvt_v2_* ]]; then
  test -f "${PRETRAINED_DIR}/${ENCODER}.pth" || {
    echo "[ERROR] pretrained model not found:"
    echo "${PRETRAINED_DIR}/${ENCODER}.pth"
    exit 1
  }
fi

# 输出根目录由shell创建，Python再建立本次RUN_ID子目录。

# 秒级时间戳和随机后缀生成唯一RUN_ID。
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"

# 日志、PID 文件和训练产物位于同一个实验目录。
RUN_ID="train_${DATASET}_${DATASET_NAME}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"

# 把关键配置追加进日志，终端单独打印RUN_ID供停止脚本使用。
RUN_DIR="$("${PYTHON_BIN}" -m lib.experiment_paths \
  "${OUTPUT_DIR}" "${DATASET}" "${DATASET_NAME}" "${ENCODER}" "${IMG_SIZE}" \
  "${BATCH_SIZE}" "${MAX_EPOCHS}" "${SUPERVISION}" "${BASE_LR}" "${EXPANSION_FACTOR}" \
  "${LGAG_KS}" "${ACTIVATION_MSCB}" "${SEED}" "${KERNEL_SIZES[@]}")"
LOG_FILE="${RUN_DIR}/launcher.log"
PID_FILE="${RUN_DIR}/${RUN_ID}.pid"

PARAM_NAMES=(
  CONDA_BASE CONDA_ENV_PREFIX PYTHON_BIN PYTHONNOUSERSITE
  PROJECT_DIR CUDA_VISIBLE_DEVICES
  DATASET TS RAND RUN_ID RUN_DIR
  LOG_FILE PID_FILE DATASET_NAME DATA_ROOT OUTPUT_DIR
  IMG_SIZE BATCH_SIZE VAL_BATCH_SIZE MAX_EPOCHS BASE_LR
  WEIGHT_DECAY SUPERVISION NUM_WORKERS SEED DETERMINISTIC
  RUN_PURPOSE KERNEL_SIZES_DISPLAY SCALE_RATES_DISPLAY ENCODER EXPANSION_FACTOR
  LGAG_KS ACTIVATION_MSCB PRETRAINED_DIR CLIP SCHEDULER
  MIN_LR N_GPU VALIDATE_EVERY THRESHOLD MAX_TRAIN_BATCHES
  MAX_VALID_CASES DEVICE
)
{
  echo "[INFO] parameters:"
  for name in "${PARAM_NAMES[@]}"; do printf '[INFO] %-24s=%s\n' "$name" "${!name}"; done
  echo "---------------------------ready to train---------------------------------"
} | tee -a "${LOG_FILE}"

# 整个续行块是一条命令：nohup+后台运行，RUN_ID注入进程环境，stdout/stderr追加到同一日志，stdin接/dev/null。
# 固定传递[1,3,5]卷积核和0.75/1/1.25多尺度；调度器、阈值及其余变量显式传给train_isic.py。
nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u train_isic.py \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_dir "${RUN_DIR}" \
  --run_name "${RUN_ID}" \
  --encoder "${ENCODER}" \
  --kernel_sizes "${KERNEL_SIZES[@]}" \
  --expansion_factor "${EXPANSION_FACTOR}" \
  --lgag_ks "${LGAG_KS}" \
  --activation_mscb "${ACTIVATION_MSCB}" \
  --supervision "${SUPERVISION}" \
  --pretrained_dir "${PRETRAINED_DIR}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --clip "${CLIP}" \
  --scheduler "${SCHEDULER}" \
  --min_lr "${MIN_LR}" \
  --scale_rates "${SCALE_RATES[@]}" \
  --num_workers "${NUM_WORKERS}" \
  --n_gpu "${N_GPU}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --validate_every "${VALIDATE_EVERY}" \
  --threshold "${THRESHOLD}" \
  --max_train_batches "${MAX_TRAIN_BATCHES}" \
  --max_valid_cases "${MAX_VALID_CASES}" \
  --device "${DEVICE}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# 记录后台PID并报告日志、PID文件和输出目录；启动脚本退出码0仅表示任务已成功放入后台。
PID=$!

echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] RUN_OUTPUT=${RUN_DIR}"
