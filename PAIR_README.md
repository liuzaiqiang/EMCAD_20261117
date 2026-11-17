# LRCI + BFC 分支

- `DZTDXFCJH`：**低秩通道协方差交互**（LRCI）。在深层解码特征中建模压缩后的通道关系；它增加注册参数，并参与前向推理。
- `SXTZYZX`：**双向特征一致性**（BFC）。对解码主路与跳连特征施加双向一致性损失；它没有新增注册参数，只在训练时计算辅助损失，默认权重 `BFC_WEIGHT=0.05`。

训练和测试脚本中的两个开关均支持 `0`（关闭）或 `1`（开启），脚本默认 `11`；直接调用 Python 的默认值为 `00`。训练与测试应使用相同组合。

```bash
DZTDXFCJH=1 SXTZYZX=1 bash sh/start_train_synapse.sh
```

代码入口覆盖 12 个数据集：Synapse、ACDC、BKAI、ClinicDB、ColonDB、ETIS、Kvasir、BUSI、ISIC2017、ISIC2018、DSB18、EM。Synapse 和 ACDC 各有独立入口；Polyp/BUSI/ISIC/Cell 通过二值任务入口或包装脚本运行。此处的覆盖指代码路径，尚未完成 12 个真实数据集的训练和测试验证。

## 复杂度记录

每次训练启动后、优化器训练前，按 EMCAD 原作者的 `thop.profile` 口径做一次 batch=1 的**全网络前向**统计。`model_complexity.txt` 保存直接注册参数数、THOP 参数数、THOP 原始返回值及其除以 `1e9` 的 GFLOPs；日志同时打印。**不乘 2**。二值任务固定输入 `1×3×256×256`，与论文 Table 1 的全模型输入一致。Synapse/ACDC 使用实际 `1×1×img_size×img_size`，因此其结果不得与论文二值任务 Table 1 或只统计解码器的 Table 6 直接混比。

THOP 的默认模块钩子会漏计 LRCI 的函数式 `F.conv2d` 与 `bmm`。BFC 仅在训练时产生辅助损失，不应算入单图**推理前向** GFLOPs。因此开启 LRCI 后，原始 THOP 的 GFLOPs 可能不变；该数仍符合原作者工具口径，但不是完整算子 FLOPs。论文中应称为 **THOP 估计 GFLOPs**，注明输入尺寸、统计范围和这一限制。参数数按 `sum(p.numel() for p in model.parameters())` 统计，包括冻结参数。

原作者实现：[EMCAD utils.py](https://github.com/SLDGroup/EMCAD/blob/main/utils/utils.py)；论文：[EMCAD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.pdf)。
