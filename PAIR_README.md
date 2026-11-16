# PRF＋AAT：像素可靠性与小前景监督

研究假设：PRF改善多头读出，面积损失保留对p1小前景的监督；检验融合是否缓解小器官预测的不稳定。

基线：`261117_baseModel`，提交`f0f3c5ba7d1dda1c071e01727feda8235104d62e`。来源及精确提交见PAIR_SOURCES.json。原模块算法保留，组合调度见lib/pair_runtime.py。CAA在基础模型初始化完成后替换上采样，初始化随机序列与旧组合分支可能不同，不能把旧实验数值直接当成本分支结果。

## 开关与运行

所有sh/start_train_*.sh和start_test_*.sh中两个开关相邻。MODULE_A=PRF，MODULE_B=AAT。脚本默认11，直接运行Python默认00。

```bash
MODULE_A=1 MODULE_B=1 bash sh/start_train_synapse.sh
MODULE_A=1 MODULE_B=0 bash sh/start_train_synapse.sh
MODULE_A=0 MODULE_B=1 bash sh/start_train_synapse.sh
MODULE_A=0 MODULE_B=0 bash sh/start_train_synapse.sh
```

测试必须使用与训练相同的开关。先沿用原脚本设置本机conda、数据和预训练路径。12数据集入口：Synapse、ACDC各独立入口；5个Polyp数据集共用polyp入口；BUSI使用busi；ISIC2017/2018使用isic；DSB18/EM使用cell入口。训练损失在三类公共训练器接入，推理统一使用最终输出。00不创建任何附加模块参数、不计算辅助损失。

AAT_WEIGHT默认1，BFC_WEIGHT默认0.05；SDG_WEIGHT默认0.001为用户确认的新实验起始值，源分支未记录涨点权重。距离损失保持原像素距离、原求和归约，二分类仅用[0,z]转换为等价双类softmax。BFC使用源实现关闭空间对齐的模式，不额外加入第三模块。

## 验证边界与科研记录

CPU合成输入检查覆盖1/4/9输出类别、四种开关、前向反向、有限梯度、严格检查点重载。未执行真实数据集训练、CUDA/AMP或完整12数据集端到端实验；服务器数据与预训练文件需沿用既有配置。不能保证涨点或无环境差异。运行`python tests/test_pair_integration.py`可复查当前分支。

00已与精确基线同种子参数及输出逐项比较。原数据划分、指标实现和训练超参数保留；现有空掩膜规则、体素间距和按测试集选择checkpoint的风险应在正式论文实验前独立审查。不得把测试集用于组合筛选。报告00/10/01/11、多个随机种子、逐类Dice与HD95、参数量与运行开销，组合改善仍是待验证假设。

## 相关文献

- EMCAD：https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html
- 抗混叠：https://proceedings.mlr.press/v97/zhang19a.html （讨论降采样，不能直接证明本CAA上采样有效）
- 区域不均衡：https://arxiv.org/abs/1706.05721
- 距离监督：https://proceedings.mlr.press/v102/kervadec19a.html
- 形状与语义流：https://openaccess.thecvf.com/content_ICCV_2019/html/Takikawa_Gated-SCNN_Gated_Shape_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.html
- 二阶统计：https://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.pdf
- 梯度冲突：https://arxiv.org/abs/2001.06782 （需要测量，不能仅凭模块名断言冲突）

文献支持研究方向，不证明本组合的原创性或实验收益；不能穷尽全部互联网文献。本次优先20组可明确描述机制的实验候选，其余35组暂缓，未断言必然冲突。
