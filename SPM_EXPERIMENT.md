# EMCAD 的显式形状先验实验

SPM 参考 You 等人的 [论文](https://pubmed.ncbi.nlm.nih.gov/39331543/) 和 [作者代码](https://github.com/AlexYouXin/Explicit-Shape-Priors)。原实现处理三维体数据；本仓库将 SUB（类别形状 token 的全局自注意力）和 CUB（形状 token 与特征交互、局部卷积更新）转换为二维，接在 EMCAD 的三级跳连进入 LGAG 前。这是结构移植，不是论文三维实验的完全复现。

无需事先画模板或配置器官形状规则。`SPM_ENABLED=1` 时，类别形状图随机初始化，随模型从训练集图像与标注中学习；推理时由当前图像特征逐级更新。Synapse 使用 9 类，ACDC 使用 4 类，二分类任务内部使用背景和前景两个形状 token。SPM 不额外添加手工形状约束损失。测试入口从 checkpoint 同目录的 `config.json` 恢复结构开关。

训练脚本只提供小器官感知监督 `SMALL_TARGET_HARD_SUPERVISION` 与形状先验 `SPM_ENABLED` 两个功能开关，数值均为 0 或 1。两者全为 0 时，不注册 SPM 参数，解码器走 EMCAD 原始路径，小器官附加损失不计算。Synapse 原有 TopN 保留；ACDC、Polyp、ISIC、BUSI、Cell 可用 `TOPN_ENABLED=1 TOPN_SIZE=10`，记录验证 Dice/HD95 并按 Dice 保存前 N 个权重。参数量和 FLOPs 调用原项目的 `CalParams(model, input_tensor)`；THOP 可能不统计自定义注意力的全部算术操作。

建议固定数据划分、种子、训练轮次、阈值及模型选择规则，比较 EMCAD、小器官监督、SPM 和两者结合。SPM 可能减少孤立假阳性与不合理轮廓，但强形状偏置也可能损伤异形病灶或稀少类别；需逐数据集报告 Dice、HD95、失败病例和多种子结果。
