#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# DSB18 / EM 细胞二分类测试启动器
#
# 本脚本设置 Cell 数据集默认值，然后复用 start_test_polyp.sh 与 test_polyp.py 中的
# 检查点加载、验证指标、逐图输出和 CSV 逻辑，避免维护两套相同推理代码。
# 检查点目录中的 config.json 会恢复 encoder、输入通道、CAA 与 DSB18 掩膜解释设置，
# 因此测试时应传入本次训练目录中的 best.pth。
# -----------------------------------------------------------------------------

# 严格模式让缺失检查点、错误数据路径或非法参数在进入后台前就明确失败。
set -euo pipefail

# 两个模块开关必须相邻；0=不构造/不执行，1=开启。对应模块见PAIR_README.md。
export PDS="${PDS:-1}"
export BFC="${BFC:-1}"
case "${PDS}${BFC}" in
  00|01|10|11) ;;
  *) echo "PDS/BFC must be 0 or 1" >&2; exit 1 ;;
esac
echo "[PAIR] PDS=${PDS} BFC=${BFC}"


# 从本文件位置推导项目根，使相对数据和脚本路径不受启动目录影响。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"


#212服务器
#CONDA_BASE="/home/zym/anaconda3"
#CONDA_ENV_PREFIX="/home/zym/anaconda3/envs/sld_emcad"


# Cell 数据根默认位于项目目录上四级的 data/cell/target；服务器数据位置不同可传 DATA_ROOT 覆盖。
export DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../../../../data/cell/target}"

# 固定日志标签和模型系列名为 Cell。
export DATASET="Cell"

# 支持 DSB18 与 EM 两套评估数据；默认选择 DSB18。
export DATASET_NAME="${DATASET_NAME:-DSB18}"

# 限制合法数据集名，防止静默读取错误的测试目录。
case "${DATASET_NAME}" in
  DSB18|EM) ;;
  *)
    echo "[ERROR] DATASET_NAME must be DSB18 or EM; got: ${DATASET_NAME}" >&2
    exit 2
    ;;
esac

# 切换到项目根并 source 通用测试启动器；CKPT、SPLIT、GPU 与结果目录仍使用既有参数。
cd "${PROJECT_DIR}"
source "${SCRIPT_DIR}/start_test_polyp.sh"
