# LRCI + MCC 分支

- `DZTDXFCJH`：**低秩通道协方差交互**（LRCI）。在 d4 和 d3 的解码特征中建模压缩后的通道相关性。
- `DCDLKYZX`：**多尺度轮廓一致性**（MCC）。在分割头之前比较 d3/d2/d1 的特征轮廓响应，并反馈调节特征。

两者都是有可学习参数、参与推理的 `nn.Module`。训练和测试脚本中的两个开关均支持 `0`（关闭）或 `1`（开启），脚本默认 `11`；直接调用 Python 的默认值为 `00`。训练与测试应使用相同组合。

```bash
DZTDXFCJH=1 DCDLKYZX=1 bash sh/start_train_synapse.sh
```

代码入口覆盖 12 个数据集：Synapse、ACDC、BKAI、ClinicDB、ColonDB、ETIS、Kvasir、BUSI、ISIC2017、ISIC2018、DSB18、EM。Synapse 和 ACDC 各有独立入口；Polyp/BUSI/ISIC/Cell 通过二值任务入口或包装脚本运行。此处的覆盖指代码路径，尚未完成 12 个真实数据集的训练和测试验证。

## 来源分支对比与实验可比性

模块来源的精确提交见 `PAIR_SOURCES.json`。当前 `lib/pair_modules/lrci.py` 与来源 `research_emcad/low_rank_covariance.py`、当前 `lib/pair_modules/mcc.py` 与来源 `lib/contour_consistency.py` 的算法代码逐行一致，仅文件末尾空行不同。与已有的 `261117_CCSF_lbtjcdrh_LRCI_dztdxfcjh`、`261117_LRCI_dztdxfcjh_BFC_sxtzyzx`、`261117_FRD_qfblxjzl_MCC_dcdlkyzx`、`261117_PRF_xsjkkxdtrh_MCC_dcdlkyzx` 分支相比，对应模块文件逐行相同，相关运行接入语句也相同。LRCI 同样作用在 d4/d3，MCC 同样在 d3/d2/d1 输出送入分割头前执行。当前组合把模块注册在 `model.pair_modules`，旧单模块来源分支注册在 `decoder` 内，故旧检查点的模块参数键不能直接严格加载到当前结构。

实验设置存在明确差异：MCC 来源提交的 Synapse 启动脚本为 **400 epoch**，当前组合分支为 **300 epoch**，而 LRCI 来源提交也是 **300 epoch**。旧来源分支按最高 Dice 保存 `best.pth`；当前 Synapse 脚本默认开启动态 top-N（10 个候选、Dice 阈值 83.90）。因此旧 MCC 的 Dice/HD95 不能直接归因比较。旧来源分支和当前分支都在 `test_vol` 上计算用于选模型的 Dice；当前 top-N 记录的 `selection_split` 也明确为 `test_vol`。如果该 split 是最终测试集，这会产生测试集选择偏差；正式论文需使用独立验证集选模型，再只在测试集评估一次。要判断组合能否提高 Dice、降低 HD95，应在相同数据划分、预训练权重、输入尺寸、epoch、种子及 checkpoint 选择规则下重新运行 `00/10/01/11`，并分别报告指标。两模块串联会相互影响，单独增益不保证组合增益。

## 复杂度记录

每次训练启动后、优化器训练前，按 EMCAD 原作者的 `thop.profile` 口径做一次 batch=1 的**全网络前向**统计。`model_complexity.txt` 保存直接注册参数数、THOP 参数数、THOP 原始返回值及其除以 `1e9` 的 GFLOPs；日志同时打印。**不乘 2**。二值任务固定输入 `1×3×256×256`，与论文 Table 1 的全模型输入一致。Synapse/ACDC 使用实际 `1×1×img_size×img_size`，其结果不得与论文二值任务 Table 1 或只统计解码器的 Table 6 直接混比。

THOP 默认钩子漏计 LRCI 的函数式 `F.conv2d`/`bmm` 和 MCC 的函数式 Sobel 卷积；MCC 的注册反馈卷积会计入。记录值符合原作者工具口径，但不是完整算子 FLOPs。论文中应称为 **THOP 估计 GFLOPs**，注明输入尺寸和统计范围。参数数按 `sum(p.numel() for p in model.parameters())` 统计，包括冻结参数。

原作者实现：[EMCAD utils.py](https://github.com/SLDGroup/EMCAD/blob/main/utils/utils.py)；论文：[EMCAD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.pdf)。
