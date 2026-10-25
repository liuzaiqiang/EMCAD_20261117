# 第二模块：互补跳连融合

## 做什么

EMCAD 的每层 LGAG 先筛选编码器 skip，再把它与上采样解码特征直接相加。新模块在这一步把已门控 skip 分成与主路同向的“共享响应”和其余“互补响应”，分别学习抑制和增强幅度。它作用于 `d3/d2/d1` 的特征形成过程；已有类别条件尺度融合只在四个预测头形成后选择 `p3/p2/p1`，两者接入位置不同。

模块对每个空间位置计算主路的单位通道方向，将 skip 在该方向上的正投影作为共享响应，余量作为互补响应。轻量逐通道门控读取主路、skip 及两者绝对差，并调节两项残差。修正增益经 `tanh` 限幅且初始化为零，因此新模块初始前向与原始 `decoder + gated_skip` 完全一致。关闭开关时连模块参数都不创建，直接执行原始相加。

每层增加 `6C` 个可学习标量；PVTv2-B2 三层合计 3,072 个参数。门控在特征图上运行，实际推理耗时和显存须在目标服务器测量。

## 为什么值得验证

LGAG 决定 skip 的空间位置，却未单独处理与解码主路重复的通道响应。这可能影响小目标召回或纹理背景下的假阳性。模块预计从这些错误中获益，但同向响应也可能是正确证据，抑制过度会降低 Dice。现有文献已研究可学习 skip 与语义差异，本模块不能仅凭结构宣称论文新颖性；需用跨数据集消融、等参数对照与错误分析证明其增量价值。

## 12 个数据集与消融

`USE_COMPLEMENTARY_SKIP_FUSION=1` 启用模块，默认 `0`。Synapse、ACDC、BKAI、ClinicDB、ColonDB、ETIS、Kvasir、ISIC2017、ISIC2018、BUSI、DSB18、EM 共用 `EMCADNet`；各训练入口和对应测试入口均接收该开关。ISIC/BUSI 复用 Polyp 接口，Cell 复用 Polyp 启动器。测试从训练 `config.json` 恢复两个开关，旧配置缺失时默认为 `0`。

四组：两个开关均关、仅尺度融合、仅互补跳连、两者均开；同数据划分、seed、编码器、训练设置。先看逐类 Dice、召回、误检及 HD95，再看两模块联合是否超过尺度融合单独开启。应增加参数量匹配的普通逐通道 skip 门控对照，避免把收益误判为互补分解本身。合成测试只能验证结构和梯度，不能保证任何数据集涨点。

相关依据：[EMCAD 原论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.pdf)、[可学习跳连与语义差异](https://www.sciencedirect.com/science/article/pii/S0893608024004702)、[FusionU-Net](https://proceedings.mlr.press/v222/li24a.html)。
