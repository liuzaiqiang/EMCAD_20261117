# EMCAD + 小器官监督 + PointRend 实验

来源：[PointRend 论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf)和[作者的语义分割实现](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)。

`POINTREND_ENABLED=1` 时，模型在 EMCAD 最终 H/4 预测头之后加入点细化。训练按作者设置每幅图采 196 点，过采样比例 3，不确定点占 0.75；点头使用三层 1×1 Conv1d、每层拼入粗预测，点级损失与原分割损失相加。推理从 H/4 逐级上采样两次，每步重预测最多 784 个不确定点，得到输入分辨率 logits。二分类不确定性为负 logit 绝对值，多分类为前两类 logit 的负间隔。细粒度特征取最终解码特征和从原分辨率输入提取的浅层特征。这个接法是适配 EMCAD 的语义分割移植，不是作者 Detectron2 骨干的逐行复现。

训练入口为 Synapse、ACDC、Polyp、ISIC2017、ISIC2018、BUSI、DSB18、EM，以及项目 Polyp 的各数据集，均使用同一可选模型参数。测试入口从 checkpoint 同目录的 `config.json` 恢复 `pointrend_enabled`。`SMALL_TARGET_HARD_SUPERVISION=0 POINTREND_ENABLED=0` 时，不创建点头参数，也不计算点级损失，模型回到 EMCAD 原始结构与前向路径。

六个训练 shell 的 `PARAM_NAMES` 会输出开关。Synapse 原有 TopN 保留；ACDC 与共用二分类训练入口可设 `TOPN_ENABLED=1 TOPN_SIZE=10`，记录每次验证的 Dice/HD95 并按 Dice 保留前 N 个 checkpoint。二分类 HD95 为像素单位，ACDC 在缺少体素间距时为体素单位。模型统计直接调用项目作者的 `CalParams(model, input_tensor)`；THOP 对采样、排序和散射等动态操作可能低估 FLOPs。

PointRend 针对插值模糊边界，因此 HD95 有下降的机制依据，但它也可能放大远处假阳性或误细化小目标，不能保证 Dice 和 HD95 同时改善。建议固定数据划分、种子、阈值、训练轮次和模型选择规则，比较 EMCAD、小器官监督、PointRend、两者结合，并报告逐类指标与失败病例。
