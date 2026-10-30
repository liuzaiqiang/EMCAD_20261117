#!/usr/bin/env bash
# 新 Linux x86_64 NVIDIA 服务器：bash sh/install_sld_emcad.sh
# 不修改已有环境，不改模型/增强/损失代码；全部安装到新 sld_emcad。
# 对齐参考环境：Python 3.10.20 / PyTorch 2.5.1+cu121 / CUDA 12.1 / cuDNN 9.1。
# NVIDIA Linux 驱动最低 525.60.13，建议 >=530.30.02；无需另装系统 CUDA Toolkit。
# RTX 50/Blackwell 等新架构不适用这套旧版 PyTorch。
set -Eeuo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
LOG_ROOT="${INSTALL_LOG_ROOT:-${SCRIPT_DIR}/logs}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_ROOT}/install_sld_emcad_${RUN_STAMP}}"
mkdir -p "${RUN_LOG_DIR}"

# 默认由 nohup 启动独立后台进程；只有内部子进程使用 --foreground。
if [[ "${1:-}" != "--foreground" ]]; then
  RUN_LOG_DIR="${RUN_LOG_DIR}" nohup bash "${SCRIPT_PATH}" --foreground "$@" \
    > /dev/null 2>&1 < /dev/null &
  INSTALL_PID=$!
  printf '%s\n' "${INSTALL_PID}" > "${RUN_LOG_DIR}/installer.pid"
  printf '[STARTED] EMCAD 环境安装已转入后台。\nPID: %s\n日志目录: %s\n总日志: %s\n' \
    "${INSTALL_PID}" "${RUN_LOG_DIR}" "${RUN_LOG_DIR}/install.log"
  printf '查看进度: tail -f "%s/install.log"\n' "${RUN_LOG_DIR}"
  exit 0
fi
shift

MAIN_LOG="${RUN_LOG_DIR}/install.log"
STEP_NAME=bootstrap
STEP_LOG="${RUN_LOG_DIR}/00_${STEP_NAME}.log"
exec 3>&1
exec > >(tee -a "${MAIN_LOG}" "${STEP_LOG}" >&3) 2>&1

# 阶段日志分别记录命令输出；总日志汇总所有阶段，失败时标出阶段和行号。
start_step() {
  local step_key="$1"
  STEP_NAME="${step_key#*_}"
  echo "[STEP COMPLETE] ${STEP_NAME} $(date '+%F %T %Z')"
  STEP_LOG="${RUN_LOG_DIR}/${step_key}.log"
  exec > >(tee -a "${MAIN_LOG}" "${STEP_LOG}" >&3) 2>&1
  echo "[STEP START] ${step_key} $(date '+%F %T %Z')"
}
trap 'status=$?; echo "[STEP FAILED] ${STEP_NAME} line=${LINENO} exit=${status} $(date "+%F %T %Z")" >&2; echo "[ERROR] 环境可能尚未装完；日志目录：${RUN_LOG_DIR}" >&2' ERR
trap 'echo "[INTERRUPTED] 安装被中断于 ${STEP_NAME}；日志目录：${RUN_LOG_DIR}" >&2; exit 130' INT TERM

echo "[INSTALL] pid=$$ started=$(date '+%F %T %Z')"
echo "[INSTALL] run_log_dir=${RUN_LOG_DIR}"
echo "[INSTALL] main_log=${MAIN_LOG}"

# 镜像只对本次脚本生效，不改 ~/.condarc 或全局 pip 配置。
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
CONDA_MIRROR="${CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
# CUDA wheel 使用独立国内镜像，不能用普通 PyPI 的默认 CUDA 构建替代。
TORCH_MIRROR="${TORCH_MIRROR:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu121}"
ENV_NAME=sld_emcad
start_step 01_preflight
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo '[ERROR] 本脚本仅用于 Linux x86_64 NVIDIA 服务器。' >&2; exit 1;
}
echo "[INFO] 环境安装脚本目录：${SCRIPT_DIR}"

# 使用已有的 base 找到 Conda；不会在 base 中安装 pip 包。
if command -v conda >/dev/null 2>&1; then
  EMCAD_CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  EMCAD_CONDA_BASE="$("${CONDA_EXE}" info --base)"
else
  echo '[ERROR] 找不到 conda，请先 source <base路径>/etc/profile.d/conda.sh。' >&2
  exit 1
fi
source "${EMCAD_CONDA_BASE}/etc/profile.d/conda.sh"

# 已有同名环境就停止，防止碰到正在训练的环境。
EMCAD_ENV_LIST="$(conda env list --json)"
if "${EMCAD_CONDA_BASE}/bin/python" -c \
  'import json,sys,pathlib; sys.exit(0 if any(pathlib.Path(p).name=="sld_emcad" for p in json.loads(sys.argv[1])["envs"]) else 1)' \
  "${EMCAD_ENV_LIST}"; then
  echo '[ERROR] 已有 sld_emcad 环境；未做任何安装。请在没有该环境的新服务器运行。' >&2
  exit 1
fi

# 提前检查驱动是否可见；完整 GPU 兼容性在安装后实际验证。
command -v nvidia-smi >/dev/null 2>&1 || {
  echo '[ERROR] 找不到 nvidia-smi，请先配置 NVIDIA 驱动。' >&2; exit 1;
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# 创建 Conda 环境；该阶段完整输出单独保存。
start_step 02_create_conda_env
echo "[2/8] 创建 ${ENV_NAME}"
conda create -y --name "${ENV_NAME}" --override-channels \
  -c "${CONDA_MIRROR}" python=3.10.20
conda activate "${ENV_NAME}"

# 排除用户目录包和外部 PYTHONPATH，防止加载别的环境中的 NumPy。
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
# 忽略已有 pip.conf/环境变量中的其他源，避免混用多个下载源。
export PIP_CONFIG_FILE=/dev/null
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX PIP_CONSTRAINT PIP_REQUIREMENT
export PIP_INDEX_URL="${PIP_MIRROR}"
export PIP_DEFAULT_TIMEOUT=120
export PIP_DISABLE_PIP_VERSION_CHECK=1

# pip 和编译工具走国内 PyPI，避免 Conda main 镜像没有 pip=23.3.2 等构建包。
# MedPy 等包可能需要编译 C++ 扩展；工具只安装到新环境，不需要 sudo。
start_step 03_bootstrap_python_tools
python -m ensurepip --upgrade
python -m pip install --retries 5 --upgrade pip==23.3.2 \
  setuptools==69.5.1 wheel==0.43.0 ninja==1.11.1.1 cmake==3.29.6

# 固定直接依赖，并把这些版本同时作为约束传给解析器。
# 只有 headless OpenCV：服务器无需桌面库，避免两个 cv2 发行包互相覆盖。
REQ_FILE="${CONDA_PREFIX}/emcad-requirements-pinned.txt"
cat > "${REQ_FILE}" <<'REQ'
numpy==1.26.4
scipy==1.11.4
scikit-image==0.21.0
scikit-learn==1.3.2
pandas==2.1.4
matplotlib==3.7.5
seaborn==0.12.2
pillow==9.5.0
opencv-python-headless==4.8.1.78
albumentations==1.1.0
qudida==0.0.4
torch==2.5.1+cu121
torchvision==0.20.1+cu121
nvidia-cudnn-cu12==9.1.0.70
timm==0.6.12
transformers==4.21.3
tokenizers==0.12.1
huggingface-hub==0.11.0
torchmetrics==0.11.4
einops==0.6.1
thop==0.1.1.post2209072238
ptflops==0.7.1.2
torchprofile==0.0.4
torchsummary==1.5.1
torchsummaryx==1.3.0
warmup-scheduler==0.3
segmentation-mask-overlay==0.3.4
SimpleITK==2.3.1
nibabel==5.2.1
h5py==3.10.0
MedPy==0.5.2
tifffile==2023.7.10
tensorboardX==2.6.2.2
protobuf==3.20.3
ml-collections==0.1.1
loguru==0.7.2
tqdm==4.66.4
PyYAML==6.0.1
tabulate==0.9.0
REQ

start_step 04_numpy_scipy
echo '[4/8] 安装 NumPy/SciPy 二进制包'
python -m pip install --retries 5 --only-binary=:all: \
  -c "${REQ_FILE}" numpy scipy scikit-image scikit-learn h5py

# 只从 CUDA 12.1 镜像下载两个匹配的 wheel；其余依赖仍从国内 PyPI 安装。
# 先下载再安装，防止普通 PyPI 把 torch 解析成 CUDA 12.4 或 CPU 版本。
start_step 05_pytorch_cuda
echo '[5/8] 安装 PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121'
TORCH_WHEEL_DIR="${CONDA_PREFIX}/emcad-torch-wheels"
mkdir -p "${TORCH_WHEEL_DIR}"
python -m pip download --retries 5 --only-binary=:all: --no-deps \
  --index-url "${TORCH_MIRROR}" --dest "${TORCH_WHEEL_DIR}" \
  "torch==2.5.1+cu121" "torchvision==0.20.1+cu121"
python -m pip install --retries 5 -c "${REQ_FILE}" "${TORCH_WHEEL_DIR}"/*.whl

start_step 06_project_dependencies
echo '[6/8] 安装其余项目依赖并检查声明的依赖关系'
python -m pip install --retries 5 -c "${REQ_FILE}" -r "${REQ_FILE}"
python -m pip check

start_step 07_runtime_validation
echo '[7/8] 检查固定版本、通用依赖、增强和 CUDA 前后向'
python - <<'PY'
from importlib import metadata, import_module
import sys
import platform
import socket
import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.stats
import skimage
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision

for name in ('numpy', 'scipy'):
    versions = [d.version for d in metadata.distributions()
                if (d.metadata.get('Name') or '').lower() == name]
    assert len(versions) == 1, (name, versions)

# 检查所有主要直接依赖均可导入，不导入或读取任何 EMCAD 项目文件。
for module in (
    'sklearn', 'pandas', 'matplotlib', 'seaborn', 'PIL', 'cv2',
    'SimpleITK', 'nibabel', 'h5py', 'medpy', 'tifffile',
    'timm', 'transformers', 'huggingface_hub', 'torchmetrics', 'einops',
    'thop', 'ptflops', 'torchprofile', 'torchsummary', 'torchsummaryX',
    'warmup_scheduler', 'segmentation_mask_overlay', 'tensorboardX',
    'ml_collections', 'yaml', 'tabulate', 'loguru', 'tqdm',
):
    import_module(module)
    print('依赖导入通过:', module, flush=True)

transform = A.Compose([
    A.Rotate(limit=90, p=1), A.VerticalFlip(p=1),
    A.HorizontalFlip(p=1), A.Resize(352, 352),
    A.Normalize(), ToTensorV2(),
])
out = transform(image=np.zeros((64, 64, 3), dtype=np.uint8),
                mask=np.zeros((64, 64), dtype=np.uint8))
assert out['image'].shape == (3, 352, 352)
torch.from_numpy(np.zeros((2, 2), dtype=np.float32)).numpy()

# 严格核对参考环境中可通过安装控制的版本，错误构建不能通过验收。
assert platform.python_version() == '3.10.20', sys.version
assert torch.__version__ == '2.5.1+cu121', torch.__version__
assert torchvision.__version__ == '0.20.1+cu121', torchvision.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
assert torch.backends.cudnn.version() == 90100, torch.backends.cudnn.version()
print('Python:', platform.python_version(), '系统:', platform.platform())
print('主机:', socket.gethostname(), 'cuDNN:', torch.backends.cudnn.version())
assert torch.cuda.is_available(), 'CUDA 不可用，请检查驱动和显卡可见性'
# 每张可见卡都实际执行卷积及反向传播，不能只检查 is_available。
for index in range(torch.cuda.device_count()):
    device = torch.device('cuda', index)
    layer = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
    x = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
    layer(x).square().mean().backward()
    torch.cuda.synchronize(device)
    print('GPU通过:', index, torch.cuda.get_device_name(index),
          '计算能力:', torch.cuda.get_device_capability(index), flush=True)

print('PyTorch:', torch.__version__, 'CUDA运行时:', torch.version.cuda)
PY

start_step 08_cli_and_manifests
echo '[8/8] 保存已验证环境的依赖清单'
python -m pip freeze > "${CONDA_PREFIX}/emcad-pip-freeze.txt"
conda list --explicit > "${CONDA_PREFIX}/emcad-conda-explicit.txt"
echo "[SUCCESS] 环境验证通过：${CONDA_PREFIX}"
echo '以后使用：conda activate sld_emcad'
echo "启动脚本中的 CONDA_BASE 应填写：${EMCAD_CONDA_BASE}"
echo "启动脚本中的 CONDA_ENV_PREFIX 应填写：${CONDA_PREFIX}"
echo '系统内核、glibc、主机名和显卡由服务器决定，Conda 不会把它们改成参考机器。'
echo '显卡可见性由训练脚本 CUDA_DEVICE 控制；切片数/批次数由数据划分及 batch size 决定。'
echo '环境依赖和 GPU 运行时已独立检查；本脚本不读取 EMCAD 项目代码或数据集。'
echo "[STEP COMPLETE] ${STEP_NAME} $(date '+%F %T %Z')"
echo "[INSTALL SUCCESS] $(date '+%F %T %Z')"
