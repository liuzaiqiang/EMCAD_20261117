# EMCAD + CAA + skip 局部频带选择门：代码与实验说明

## 代码路径和模块定义

当前实际入口是 `lib/networks.py::EMCADNet`，四级编码特征按
`x4, [x3,x2,x1]` 进入 `lib/decoders.py::EMCAD`。CAA 仅在三个 EUCB 的
上采样中工作。新模块 SFG 在每一级 `LGAG(g=d_i, x=skip_i)` 之前处理
编码器 skip，使用同尺度解码特征作为引导；三个门彼此独立。它不改编码器、
原 LGAG、损失、标签、选模指标或测试阈值。

SFG 以固定 3×3 二项式核把 skip 分成局部平滑分量 `L` 和残差 `H=skip-L`。
两者只是局部低/高频代理，不能称为精确 FFT 频谱。对每个通道和像素，
深度分组 1×1 卷积从解码特征生成两个分数；再加入归一化的 `|L|/|H|`
证据，得到低/高分量各自的有界门。输出是
`skip + tanh(alpha) * [tanh(score_L)*L + tanh(score_H)*H]`。
各级 `alpha` 初始化为零，因此开启时初次前向严格等于原 skip；关闭时
不实例化门、不增加 state_dict 键，并完全走原 EMCAD+CAA/off 代码路径。
新门初始化置于独立 RNG 上下文，原有分割头的随机初始化序列保持一致。

PVTv2-B2 三个 skip 通道数为 320/128/64；总新增可学习参数
`Σ(6C+1)=3,075`，另有 3 个固定 3×3 buffer。224 输入时固定模糊与
分组 1×1 卷积合计约 4.00M MAC；逐元素运算、访存与插值不包含在此数。
实际 FLOPs、延迟、峰值显存和对 Dice/HD95 的影响需在训练服务器测量。

## 接入范围

| 训练/测试家族 | 数据集 | 新开关来源 |
| --- | --- | --- |
| Synapse | Synapse | 自有 Python 入口 |
| ACDC | ACDC | 自有 Python 入口，公共 `acdc_utils.build_model` |
| Polyp | BKAI、ClinicDB、ColonDB、ETIS、Kvasir | 公共 `train_polyp`、`test_polyp` |
| ISIC | ISIC2017、ISIC2018 | 包装器复用 Polyp 模型 |
| BUSI | BUSI | 包装器复用 Polyp 模型 |
| Cell | DSB18、EM | shell 包装器复用 Polyp 入口 |

所有训练 shell 的 `USE_SKIP_FREQUENCY_GATE` 默认 0，只接受 0/1，并与
`USE_CONTENT_AWARE_ANTIALIAS` 相邻；`PARAM_NAMES` 会记录它。训练
`config.json` 保存 `use_skip_frequency_gate`，实验路径增加 `sfg_0/1`。
测试通过所选 `best.pth` 相邻配置恢复新开关，旧配置缺字段时视为 0；
显式冲突会报错，state_dict 保持严格加载。测试输出仍放在 checkpoint 下
独立的 `test_results/<run_id>`，不覆盖其它训练运行。

## 文献与学术边界

低/高频分解和 skip 频带增强已有明确先例，例如
[Octave Convolution（ICCV 2019）](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Drop_an_Octave_Reducing_Spatial_Redundancy_in_Convolutional_Neural_Networks_ICCV_2019_paper.html)、
[FcaNet（ICCV 2021）](https://openaccess.thecvf.com/content/ICCV2021/html/Qin_FcaNet_Frequency_Channel_Attention_Networks_ICCV_2021_paper.html)、
[PFESA（MICCAI 2025）](https://papers.miccai.org/miccai-2025/0689-Paper3694.html)以及
[FSR-Net（2026）](https://www.sciencedirect.com/science/article/abs/pii/S0950705126012402)。
尤其 PFESA 已明确在医学分割 skip 上做高低频处理；本实现采用**局部滤波代理、
解码器条件化的逐通道双门、零初始化残差以及与 EMCAD/CAA 的具体组合**，
FSR-Net 摘要也涉及解码语义引导的 skip 频率选择。因此不能仅凭这些差异
宣称方法首创。发表前需更全面的同类方法对比和真实消融。

## 四组消融与指标

| CAA | SFG | 含义 |
| --- | --- | --- |
| 0 | 0 | 当前 EMCAD 基础路径 |
| 1 | 0 | EMCAD + CAA |
| 0 | 1 | EMCAD + SFG |
| 1 | 1 | EMCAD + CAA + SFG |

例如 Synapse 一组训练：

```bash
USE_CONTENT_AWARE_ANTIALIAS=0 USE_SKIP_FREQUENCY_GATE=0 SEED=2222 bash sh/start_train_synapse.sh
USE_CONTENT_AWARE_ANTIALIAS=1 USE_SKIP_FREQUENCY_GATE=0 SEED=2222 bash sh/start_train_synapse.sh
USE_CONTENT_AWARE_ANTIALIAS=0 USE_SKIP_FREQUENCY_GATE=1 SEED=2222 bash sh/start_train_synapse.sh
USE_CONTENT_AWARE_ANTIALIAS=1 USE_SKIP_FREQUENCY_GATE=1 SEED=2222 bash sh/start_train_synapse.sh
```

各启动器采用后台任务；上述四行需**每次等待训练完成并核查日志后**逐一执行。
其他数据集替换对应启动器，Polyp/ISIC/Cell 按已有 `DATASET_NAME` 选子集。
对预先指定的多个 seed 重复，固定划分、预训练文件哈希、轮数、损失、增强、
best 选择和测试代码。用 `CKPT=/absolute/path/to/best.pth bash sh/start_test_<family>.sh`
测试每组；只用独立验证集选模，最终测试集只评估一次。Synapse 当前训练
`trainer.py` 每轮使用名为 `test_vol` 的列表选 `best.pth`；若服务器上该列表
就是最终测试集，则存在测试集参与选模，正式论文实验必须先解决协议问题。

报告每个数据集每个 seed 的病例/图像级 Dice、HD95、空掩膜数量和配对差值，
再报告跨 seed 均值与标准差。多类体数据与二分类图像的统计口径不同，
不能直接合并成一个总分；当前空掩膜的 HD95 约定在不同入口也不同，
不得只对新模块修改指标。若要统一评估协议，四组都必须重新计算。

## 验证状态

本地已通过 Python AST 解析、12 个 Bash 入口 `bash -n`、`git diff --check`
及六类入口静态贯通测试。本机没有 PyTorch，真实前向/反向、AMP、严格加载
和四组输出等价测试会跳过。服务器上运行：

```bash
python -m unittest tests.test_skip_frequency_gate tests.test_skip_frequency_entrypoints -v
```

代码可运行不等于 Dice 或 HD95 已改善；当前没有任何真实训练指标。
