# EMCAD + 小器官感知监督 + CaraNet CFP 实验

本实现参考 [CaraNet 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9938296/)及[作者的 CFP 代码](https://github.com/AngeLouCN/CaraNet/blob/main/lib/context_module.py)。CaraNet 的 CFP 使用四路膨胀卷积、每路逐级堆叠、跨路逐级累加、通道融合及残差。作者代码在 `d=8` 时实际使用膨胀率 1、3、5、9。本项目保持这套 CFP 内部结构，将 EMCAD 的 x2、x3、x4 经 3×3 卷积投影到 32 通道，经过 CFP 后以 1×1 卷积投影回原通道并残差注入，再进入原 EMCAD 解码器。投影层是骨干通道不同所必需的适配，不是 CaraNet 原有的 Res2Net、部分解码器或轴向反向注意力。

六个训练脚本提供 `CFP_ENABLED=0/1`，与 `SMALL_TARGET_HARD_SUPERVISION=0/1` 独立；两者均为 0 时，模型没有 CFP 参数，也不计算小器官附加损失，执行 EMCAD 原始前向路径。现有十二个数据集是 Synapse、ACDC、BKAI、ClinicDB、ColonDB、ETIS、Kvasir、ISIC2017、ISIC2018、BUSI、DSB18、EM；五个息肉集与 ISIC、BUSI、Cell 通过共用二分类训练器接入。测试端从 checkpoint 同目录 `config.json` 恢复 CFP 架构开关。启动参数与开关会记入日志及配置快照。

Synapse 原有 TopN 保留。ACDC 及通用二分类训练入口可用 `TOPN_ENABLED=1 TOPN_SIZE=10`，保存每次验证的 Dice、HD95 轨迹，并按 Dice 保留前 N 个权重。二分类 HD95 是像素单位；ACDC 未传体素间距时是体素单位。模型统计调用项目原有 `CalParams(model, input_tensor)`，维持其 THOP 口径。

CFP 增大多尺度感受野，可能使小目标区域更完整，从而提升 Dice；较少的孤立预测或更完整的轮廓也可能降低 HD95。但 CaraNet 整网结果不能直接归因于 CFP，更不能保证本分支 12 个数据集全部涨点。应固定数据划分、种子、训练轮次、阈值与模型选择规则，比较 EMCAD、小器官监督、CFP、两者结合，并报告逐类 Dice、HD95 与失败病例。本分支未实现预测图辅助模块。
