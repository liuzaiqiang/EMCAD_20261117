# CCSF + LRCI 分支

- `LBTJCDRH`：**类别条件尺度融合**（CCSF）。按类别为不同解码尺度分配权重，融合多尺度预测。
- `DZTDXFCJH`：**低秩通道协方差交互**（LRCI）。在深层解码特征中建模压缩后的通道关系。

训练和测试脚本中的两个开关均支持 `0`（关闭）或 `1`（开启），脚本默认 `11`；直接调用 Python 的默认值为 `00`。训练与测试必须使用相同组合。

```bash
LBTJCDRH=1 DZTDXFCJH=1 bash sh/start_train_synapse.sh
```

代码入口覆盖 12 个数据集：Synapse、ACDC、BKAI、ClinicDB、ColonDB、ETIS、Kvasir、BUSI、ISIC2017、ISIC2018、DSB18、EM。Synapse 和 ACDC 各有独立入口；Polyp/BUSI/ISIC/Cell 通过二值任务入口或对应包装脚本运行。此处的覆盖指代码路径，尚未完成 12 个真实数据集的训练和测试验证。

## 复杂度记录

每次训练启动后、优化器训练前，按 EMCAD 原作者的 `thop.profile` 口径做一次 batch=1 的**全网络前向**统计。`model_complexity.txt` 保存直接注册参数数、THOP 参数数、THOP 原始返回值和原始返回值除以 `1e9` 的 GFLOPs；日志同时打印。**不乘 2**。二值任务固定输入 `1×3×256×256`，与论文 Table 1 的全模型输入一致。Synapse/ACDC 使用实际 `1×1×img_size×img_size`，因此其结果不得直接与论文二值任务 Table 1 或只统计解码器的 Table 6 混比。

THOP 的默认模块钩子可能漏计函数式运算：本分支 CCSF 的 `einsum`，以及 LRCI 的 `F.conv2d` 和 `bmm`。记录值可按原作者的工具口径比较，但不是完整算子 FLOPs 或硬件实测开销；写论文时应明确称为 **THOP 估计 GFLOPs** 并注明输入尺寸和统计范围。参数数是 `sum(p.numel() for p in model.parameters())`，包括冻结参数。

原作者实现：[EMCAD utils.py](https://github.com/SLDGroup/EMCAD/blob/main/utils/utils.py)；论文：[EMCAD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.pdf)。
