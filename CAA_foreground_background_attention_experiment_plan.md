# EMCAD + CAA + 潜在前景/背景解耦注意力

## 实际代码与机制

当前运行模型是 `lib/networks.py::EMCADNet`。编码器特征按
`x4,[x3,x2,x1]` 进入 `lib/decoders.py::EMCAD`；三个 EUCB 上采样可选 CAA，
三个 LGAG 将解码特征和同尺度 skip 融合。新模块 FBDA 放在 d3/d2/d1
各自的 skip 加法之后、原 CAB/SAB/MSCB 之前。它只处理融合特征，
不改变数据、标签、损失、类别数、测试阈值和 best 选择规则。

每一级从融合特征 X 学习单通道软先验 P=sigmoid(conv1x1(X))。
用 P 和 1-P 分别加权平均 X，得到两个独立的通道上下文向量；
两个 depthwise 1×1 层分别生成通道门，再用 P 与 1-P 在空间上组合。
输出为 `X + tanh(alpha) * X * selection`。`alpha` 从 0 初始化，
因此第一次前向严格等于旧模型；关闭开关时完全不实例化新层。
新层在隔离的 RNG 上下文中初始化，使原模型和输出头在相同 seed 下
保留相同初始化。这里的“前景/背景”是**模型学习到的潜在软区域**，
没有使用真值掩膜，也不能未经验证当成真实器官区域。

PVTv2-B2 的三层通道为 320/128/64，总新增参数
`Σ(5C+2)=2,566`。实际 FLOPs、延迟、显存需服务器测量；
软区域加权池化会增加访存，参数少不意味着一定更快。

## 12 数据集接入

| 训练/测试家族 | 数据集 |
| --- | --- |
| Synapse | Synapse |
| ACDC | ACDC |
| 通用 Polyp | BKAI、ClinicDB、ColonDB、ETIS、Kvasir |
| ISIC 包装器 | ISIC2017、ISIC2018 |
| BUSI 包装器 | BUSI |
| Cell 包装器 | DSB18、EM |

训练 shell 的 `USE_FOREGROUND_BACKGROUND_ATTENTION` 默认 0，只接受 0/1，
与 `USE_CONTENT_AWARE_ANTIALIAS` 相邻；五个实际训练启动器的
`PARAM_NAMES` 都记录该值。Cell 通过 Polyp 通用启动器记录。
Python 参数写进训练 `config.json`，实验目录增加 `fbda_0/1`。
测试根据选中 `best.pth` 同目录的配置恢复开关；旧配置缺字段视为 0，
显式冲突会报错，state_dict 仍严格加载。测试结果继续按 checkpoint
分开保存，既有运行不会相互覆盖。

## 消融与科研边界

| CAA | FBDA | 组别 |
| --- | --- | --- |
| 0 | 0 | EMCAD 基础路径 |
| 1 | 0 | EMCAD + CAA |
| 0 | 1 | EMCAD + FBDA |
| 1 | 1 | EMCAD + CAA + FBDA |

示例（后台启动器返回不代表训练完成，单卡应逐组等待）：

```bash
USE_CONTENT_AWARE_ANTIALIAS=0 USE_FOREGROUND_BACKGROUND_ATTENTION=0 SEED=2222 bash sh/start_train_synapse.sh
USE_CONTENT_AWARE_ANTIALIAS=1 USE_FOREGROUND_BACKGROUND_ATTENTION=0 SEED=2222 bash sh/start_train_synapse.sh
USE_CONTENT_AWARE_ANTIALIAS=0 USE_FOREGROUND_BACKGROUND_ATTENTION=1 SEED=2222 bash sh/start_train_synapse.sh
USE_CONTENT_AWARE_ANTIALIAS=1 USE_FOREGROUND_BACKGROUND_ATTENTION=1 SEED=2222 bash sh/start_train_synapse.sh
```

其他数据集用对应脚本及 `DATASET_NAME`，每个预先选定的 seed 都跑四组。
固定数据划分、预训练权重文件哈希、输入尺寸、epoch、增强、优化器、
损失、best 选择及测试代码。报告逐病例/图像 Dice、HD95、类别级结果、
空掩膜数量、跨 seed 均值和标准差。由于新模块与 CAB/SAB 都有注意力作用，
应增加参数量相近的普通通道/空间注意力对照，证明收益确实来自区域解耦。
软先验图需可视化并与真值边界作分析，但不得用测试真值参与模型前向。
Synapse 现有训练每轮用名为 `test_vol` 的列表选 `best.pth`；若它是最终
测试集，就存在测试集参与选模，正式实验必须先核实并解决。

前景/背景分离及相关注意力已有研究，例如
[FBA-Net](https://arxiv.org/abs/2306.15189)、
[COMA-Net](https://www.sciencedirect.com/science/article/pii/S1746809423006316)。
本代码是针对 EMCAD+CAA 的一种待验证实现，不可仅凭结构或静态测试宣称
原创、Dice 涨点或 HD95 降低。发表前需核查更接近的工作并完成真实对照。

本地已通过 Python AST、12 个 Bash 入口 `bash -n`、`git diff --check`
和入口/旧配置静态测试。本机缺 PyTorch，真实模型测试尚未执行。服务器命令：

```bash
python -m unittest tests.test_foreground_background_entrypoints tests.test_foreground_background_attention -v
```
