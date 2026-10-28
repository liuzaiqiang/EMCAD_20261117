# EMCAD + CAA + 方向性小波细节补偿

本次实现位于当前 `contentAware_and_fxxxbxjbcmk` 分支。CAA 实现保留在 `lib/decoders.py`，新增补偿模块位于 `lib/wavelet_detail.py`。本模块是待实验验证的研究方案，不能因为使用了小波、通过了程序测试就认定能提高 Dice 或恢复真实器官边界。

## 1. 接入位置与边界

每级数据流为：

```text
上一层解码特征 → EUCB（原始上采样或已有 CAA）→ 通道对齐 → u
原始编码器 skip → Haar 分解 → 三方向软阈值抑制 → 语义门控
             → LL=0 的逆 Haar 变换 → 无偏置通道投影 → r
u' = u + tanh(alpha) * r
原有 LGAG(g=u', x=原始 skip) → 原有 skip 相加 → CAB → SAB → MSCB
```

补偿发生在完整 EUCB 输出之后、LGAG 之前，三个位置对应 d3、d2、d1。读取的是原始 encoder skip，不是 LGAG 筛选后的 skip。原始 skip 本身及其既有融合路径保留。补偿后的解码特征也会作为 LGAG 的 gating 输入，因此它同时影响后续 LGAG 的筛选。

PVTv2-B2、224×224 输入时：

| 层级 | EUCB 输出与原始 skip | 小波高频子带 |
| --- | --- | --- |
| d3 / x3 | B×320×14×14 | B×320×3×7×7 |
| d2 / x2 | B×128×28×28 | B×128×3×14×14 |
| d1 / x1 | B×64×56×56 | B×64×3×28×28 |

补偿仅在特征空间工作，没有额外分割头、边界标签或辅助损失。小波高频也包含背景纹理、噪声与伪边界；阈值和门控只是提供学习抑制的能力，不能保证识别所有真实边界。

## 2. 数学定义与初始化

对每个通道的非重叠 2×2 小块 `[[a,b],[c,d]]`，使用正交 Haar 约定：

```text
LL = (a+b+c+d)/2
H  = (a+b-c-d)/2   # 沿行/y方向高通，响应水平边缘
V  = (a-b+c-d)/2   # 沿列/x方向高通，响应垂直边缘
D  = (a-b-c+d)/2   # 对角高频
```

H/V 名称在不同小波库中可能交换，本实现以上述公式为准。三种高频均保留正负号。奇数高宽仅在右侧、底部复制填充，逆变换后裁回原大小；不直接对降采样后的高频系数做插值。

对每个样本、通道、方向，计算：

```text
m = mean(abs(detail)).detach()
tau = softplus(theta) * m
shrunk = sign(detail) * relu(abs(detail) - tau)
```

`softplus(theta)` 初始为 0.1，形状为 `1×C×3×1×1`；空间均值来自当前样本，不维护跨样本统计。`m` 是特征幅值参考，不能解释为高斯噪声标准差，也不是 Donoho 的通用阈值公式。

门控读取下采样后的解码特征与原始 skip 的低频上下文，经 `1×1 Conv → ReLU → 1×1 Conv → sigmoid` 生成三张方向门控图，形状 `B×3×h×w`，每方向空间门控在通道上共享。采用独立 sigmoid，使三个方向可以同时受到抑制。

把门控后的三方向系数与零 LL 一起逆变换，得到只含高频的同分辨率细节，再用不带 bias 的 1×1 卷积对齐通道语义。每一级只有一个可学习残差标量 `alpha`，初始化为 0，实际缩放为 `tanh(alpha)`，范围为 (-1,1)。初始输出严格等于补偿前的 EUCB 特征；第一步先更新 alpha，后续细节分支开始学习。负缩放允许模型学习削弱或纠正不合适的细节响应。

新分支在原有解码层构造之后创建，并保存、恢复 CPU 随机数状态。同一个 CAA 设置下，开关小波分支不会改变原有共享层和输出头的初始权重。本实现无新增第三方依赖。

## 3. 开关与启动

所有主训练启动器新增 `USE_WAVELET_DETAIL`，默认 0，只有 0/1 合法。Python 对应 `--use_wavelet_detail 0/1`，模型构造对应 `use_wavelet_detail=False/True`。Cell 通过 source 通用 Polyp 启动器继承控制。

| USE_CONTENT_AWARE_ANTIALIAS | USE_WAVELET_DETAIL | 模型 |
| --- | --- | --- |
| 0 | 0 | EMCAD 基础模型 |
| 1 | 0 | EMCAD + 原有 CAA |
| 0 | 1 | EMCAD + 小波补偿，用于独立消融 |
| 1 | 1 | EMCAD + 原有 CAA + 小波补偿 |

在原有 Linux 训练环境，从项目根目录执行组合实验：

```bash
USE_CONTENT_AWARE_ANTIALIAS=1 USE_WAVELET_DETAIL=1 bash sh/start_train_synapse.sh
```

严格基础结构实验：

```bash
USE_CONTENT_AWARE_ANTIALIAS=0 USE_WAVELET_DETAIL=0 bash sh/start_train_synapse.sh
```

也可以直接编辑对应 `.sh` 中两个开关的默认数字。显式环境变量优先于默认值。原有其他数据集启动器的 CAA 默认值保留，因此消融时务必显式设置两个开关；若额外指定了 `CAA_MODE=aa_only/content_only`，应作为单独实验记录。

小波关闭时不实例化三个补偿模块，没有它们的参数、buffer 或相关张量计算。CAA 关闭时沿用原始 EUCB 最近邻上采样路径。双关只保证模型结构/计算恢复基础模型；数据路径按本次要求改变，若更换了数据内容，实验结果当然不能据此视为历史复现。

当前分支 Python 没有可靠性融合模块。此前部分 shell 残留的融合变量并未传入 Python，本次清理这些无效变量与日志，避免把不存在的功能记入实验条件；损失与输出规则没有因此变化。

### 全部数据集覆盖

Synapse 之外的 11 个数据集全部包含在本次接入范围内，总计 12 个。公共启动器通过 `DATASET_NAME` 选择具体数据集，没有为同一训练循环复制多份代码。

| 数据集 | 训练 / 测试脚本（均位于 sh/） | 默认输入 | 原有训练设置 |
| --- | --- | --- | --- |
| Synapse | start_train_synapse.sh / start_test_synapse.sh | 224×224 | 原有多类 mutation 监督 |
| ACDC | start_train_acdc.sh / start_test_acdc.sh | 224×224 | 4 类、原有 mutation 监督 |
| BKAI | start_train_polyp.sh / start_test_polyp.sh | 352×352 | paper 监督、多尺度训练 |
| ClinicDB | start_train_polyp.sh / start_test_polyp.sh | 352×352 | paper 监督、多尺度训练 |
| ColonDB | start_train_polyp.sh / start_test_polyp.sh | 352×352 | paper 监督、多尺度训练 |
| ETIS | start_train_polyp.sh / start_test_polyp.sh | 352×352 | paper 监督、多尺度训练 |
| Kvasir | start_train_polyp.sh / start_test_polyp.sh | 352×352 | paper 监督、多尺度训练 |
| ISIC2017 | start_train_isic.sh / start_test_isic.sh | 352×352 | paper 监督、多尺度训练、固定版本划分 |
| ISIC2018 | start_train_isic.sh / start_test_isic.sh | 352×352 | paper 监督、多尺度训练、固定版本划分 |
| BUSI | start_train_busi.sh / start_test_busi.sh | 256×256 | 三通道、固定尺度、原有划分校验 |
| DSB18 | start_train_cell.sh / start_test_cell.sh | 256×256 | 固定尺度、实例掩膜合并为语义前景 |
| EM | start_train_cell.sh / start_test_cell.sh | 256×256 | 固定尺度、配对语义掩膜 |

入口检查还修复了 BUSI 训练/测试缺少 `PYTHON_BIN` 默认值、BUSI/ISIC 训练缺少 `N_GPU` 默认值的问题；BUSI 测试去掉重复且引用未定义环境名的 Conda 激活。默认 Python 使用已激活环境中的 `python`，GPU 数默认 1，均支持原有同名环境变量覆盖。

多尺度训练保留现有 0.75/1/1.25 设置。这里列出的是本分支启动默认值，不表示所有数据集采用相同实验协议。二分类任务复用通用模型工厂和训练循环，BUSI/ISIC 包装器继续执行各自数据协议检查，Cell 继续通过 source 共用启动器及队列 PID 行为。

各系列启动示例（执行所需的一条；不建议未经资源安排一次并发全部任务）：

```bash
USE_CONTENT_AWARE_ANTIALIAS=1 USE_WAVELET_DETAIL=1 bash sh/start_train_acdc.sh
DATASET_NAME=BKAI USE_CONTENT_AWARE_ANTIALIAS=1 USE_WAVELET_DETAIL=1 bash sh/start_train_polyp.sh
DATASET_NAME=ISIC2017 USE_CONTENT_AWARE_ANTIALIAS=1 USE_WAVELET_DETAIL=1 bash sh/start_train_isic.sh
USE_CONTENT_AWARE_ANTIALIAS=1 USE_WAVELET_DETAIL=1 bash sh/start_train_busi.sh
DATASET_NAME=DSB18 USE_CONTENT_AWARE_ANTIALIAS=1 USE_WAVELET_DETAIL=1 bash sh/start_train_cell.sh
```

Polyp 的 `DATASET_NAME` 可取 BKAI/ClinicDB/ColonDB/ETIS/Kvasir；ISIC 可取 ISIC2017/ISIC2018；Cell 可取 DSB18/EM。每个系列均可将两开关设成 `0/0`、`1/0`、`0/1`、`1/1`。例如 EM 基础模型：

```bash
DATASET_NAME=EM USE_CONTENT_AWARE_ANTIALIAS=0 USE_WAVELET_DETAIL=0 bash sh/start_train_cell.sh
```

配置写入各训练入口已有的 `config.json`；统一实验目录加入 `wavelet_0/1` 层级。各测试入口恢复该字段；旧配置缺字段时解释为 0。测试组合 checkpoint 应使用训练保存的配置，不能靠 `strict=False` 掩盖结构不匹配。

Synapse 示例：

```bash
CKPT=/absolute/path/to/best.pth bash sh/start_test_synapse.sh
```

## 4. 数据目录改动

训练、测试、检查工具、BUSI 划分准备工具及 Synapse 预处理脚本中的运行时 `../data` 默认路径统一改为 `../../data`。`.sh` 使用 `${PROJECT_DIR}/../../data`，Python 直接运行时应从项目根目录启动；已有的显式 CLI 路径和环境变量覆盖规则保留。数据集子目录名称、大小写、清单名称及划分协议不变。历史实验文档未改写。

当前 Windows 项目对应目标：

```text
D:\files\data\Synapse\...
D:\files\data\ACDC\...
D:\files\data\polyp\target\...
D:\files\data\isic\target\...
D:\files\data\busi\target\...
D:\files\data\cell\target\...
```

检查时 `D:\files\data` 不存在，原有 `D:\files\Medical_Image_Segmentation_Projects\data` 存在。本次仅修改默认路径，没有移动、复制或重新划分数据。启动训练前需将实际数据准备到目标目录，或者通过现有路径参数显式指定正确位置。

## 5. 已完成的验证与证据范围

环境：本机 `D:\install\miniconda3\envs\Sim-MPNet\python.exe`，PyTorch 2.6.0+cpu、timm 1.0.15；没有可用 CUDA。

```bash
python -m unittest discover -s tests -p test_wavelet_detail.py -v
```

数值/回归测试共 10 项，全部通过，覆盖：

1. Haar 分解/重建、能量守恒、H/V/D 的解析方向与像素位置、奇数和单像素尺寸。
2. 常量 skip 不产生补偿；零初始化恒等；阈值、门控、投影、skip 梯度有限且可学习。
3. 双关 PVT-B2 与历史基础代码 `f70e6b0c06b4f5b6a4408b43b0c88be05608ac5b` 比较全部参数/buffer、随机数状态、四路 logits、训练梯度及 BN 更新，结果一致，并能严格加载其 state_dict。
4. 小波关闭、CAA 开启时，与本次编辑前代码 `768974840d2c4fd705162def6e3ad7d98c6ed5a1` 的 CAA 模型权重和输出一致。
5. 四种结构可运行；新增分支不改变共享权重初始化；模块确实接收原始 encoder skip；组合模型 state_dict 保存、严格重载后输出一致。

此外，PVT-B2、224×224、单通道输入、9 类输出进行了四种组合的 CPU 前向/反向检查。三层 alpha 在初始零值下均能获得非零梯度；第一次 AdamW 更新后，三层内部所有可学习参数均能获得非零梯度。已检查修改后的 Python 语法和 `.sh` 的 `bash -n`，并检查 diff 空白错误。

全数据集入口回归命令：

```bash
python -m unittest discover -s tests -p test_wavelet_entrypoints.py -v
```

该测试实际执行临时副本中的 shell，通过拦截 Conda 激活与 nohup 捕获命令，覆盖 12 个数据集的 48 次训练启动（每个四组开关组合）及 12 次测试启动。检查开关、数据根路径、数据集名称、默认图像尺寸、监督方式、固定/多尺度和实例掩膜设置；对捕获命令执行原始 Python 参数解析源码。BUSI/ISIC 另执行包装器的数据元数据校验之前的真实配置逻辑。ACDC/通用二分类及 BUSI/ISIC 的 checkpoint 配置检查覆盖旧配置缺字段默认关闭、新配置恢复开启；ACDC/通用二分类还覆盖显式冲突拒绝。

上述 4 项入口/配置测试全部通过（共 48 次训练启动与 12 次测试启动）。另对 PVT-B2、3 通道、1 类输出的 256×256 和 352×352 输入分别执行四组消融的 CPU 前向/反向，共 8 组，四路输出尺寸正确、输出与梯度均有限。这些输入来自随机张量。

临时数据和 checkpoint 是空夹具，只用于 shell 路径检查及参数传递，未加载图像或权重。AST 提取用于避开原有数据加载器依赖，不能替代完整入口导入、真实数据加载或正式训练。另有 1 类二分类与 4 类 ACDC 输出头的历史基础模型权重/输出一致性检查。

Synapse、ACDC 的训练/测试入口通过 `--help` 参数检查。本地环境缺少 `albumentations`，Polyp/BUSI/ISIC 入口在导入原有数据加载器时失败，因此这些任务的完整入口及真实数据加载尚未完成运行验证。新模块本身不依赖该库。

该配置参数量仅用于结构记录，不代表速度或分割精度：

| CAA | 小波 | 全模型参数量 |
| --- | --- | --- |
| 0 | 0 | 26,772,635 |
| 1 | 0 | 26,876,993 |
| 0 | 1 | 26,928,039 |
| 1 | 1 | 27,032,397 |

小波新增 155,404 个参数。未测量真实训练吞吐、CUDA 显存或实际数据集 Dice/HD95，没有执行正式训练，也没有读取服务器数据。

正式评估应冻结数据划分、增强、预处理、损失、优化器、学习率、监督方式、epoch、batch、编码器、预训练及 checkpoint 选择规则，预先确定同一组随机种子，报告四组消融的均值/标准差与逐器官 Dice/HD95。现有 Synapse 训练器使用 `test_vol` 选取 best；若要报告独立测试性能，需要另行确定独立验证/测试划分，不能把参与选 checkpoint 的集合直接称为独立测试集。本次未改变该既有协议。

## 6. 论文依据

以下页面/元数据已在线核实。引用支持基础架构、小波分析/重建和软阈值思想，不构成当前组合的创新性或有效性证明。

- Rahman et al., **EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation**, CVPR 2024, pp. 11769–11779。基础模型及 EUCB/LGAG 接口依据。[CVF 官方页面](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html)
- Liu et al., **Multi-Level Wavelet-CNN for Image Restoration**, CVPR Workshops 2018, pp. 773–782。借鉴特征域小波分析与逆变换思想；本实现不是其多级网络复现，也没有照搬其任务结论。[CVF 官方页面](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Liu_Multi-Level_Wavelet-CNN_for_CVPR_2018_paper.html)
- D. L. Donoho, **De-noising by soft-thresholding**, IEEE Transactions on Information Theory, 1995。借鉴符号保持的软阈值形式；本实现阈值由特征幅值及学习参数决定。[DOI](https://doi.org/10.1109/18.382009)
