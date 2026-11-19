# PDS＋BFC：逐级语义传递与跨路径一致性

研究假设：PDS调整门控后的跳连，BFC约束门控前的双向表征；检验分布一致性与语义选择是否互补。

基线：`261117_baseModel`，提交`f0f3c5ba7d1dda1c071e01727feda8235104d62e`。来源及精确提交见PAIR_SOURCES.json。原模块算法保留，组合调度见lib/pair_runtime.py。CAA在基础模型初始化完成后替换上采样，初始化随机序列与旧组合分支可能不同，不能把旧实验数值直接当成本分支结果。

## 开关与运行

所有 sh/start_train_*.sh 和 start_test_*.sh 中两个开关相邻：`PDS` 控制逐级深层语义模块，`BFC` 控制双向特征一致性损失。脚本默认 11，直接运行 Python 默认 00。

```bash
PDS=1 BFC=1 bash sh/start_train_synapse.sh
PDS=1 BFC=0 bash sh/start_train_synapse.sh
PDS=0 BFC=1 bash sh/start_train_synapse.sh
PDS=0 BFC=0 bash sh/start_train_synapse.sh
```

测试必须使用与训练相同的开关。先沿用原脚本设置本机conda、数据和预训练路径。12数据集入口：Synapse、ACDC各独立入口；5个Polyp数据集共用polyp入口；BUSI使用busi；ISIC2017/2018使用isic；DSB18/EM使用cell入口。训练损失在三类公共训练器接入，推理统一使用最终输出。00不创建任何附加模块参数、不计算辅助损失。

AAT_WEIGHT默认1，BFC_WEIGHT默认0.05；SDG_WEIGHT默认0.001为用户确认的新实验起始值，源分支未记录涨点权重。距离损失保持原像素距离、原求和归约，二分类仅用[0,z]转换为等价双类softmax。BFC使用源实现关闭空间对齐的模式，不额外加入第三模块。

## 验证边界与科研记录

CPU合成输入检查覆盖1/4/9输出类别、四种开关、前向反向、有限梯度、严格检查点重载。未执行真实数据集训练、CUDA/AMP或完整12数据集端到端实验；服务器数据与预训练文件需沿用既有配置。不能保证涨点或无环境差异。运行`python tests/test_pair_integration.py`可复查当前分支。

00已与精确基线同种子参数及输出逐项比较。原数据划分、指标实现和训练超参数保留；现有空掩膜规则、体素间距和按测试集选择checkpoint的风险应在正式论文实验前独立审查。不得把测试集用于组合筛选。报告00/10/01/11、多个随机种子、逐类Dice与HD95、参数量与运行开销，组合改善仍是待验证假设。

## 复杂度统计口径

训练入口直接调用 `utils/utils.py::cal_params_flops`：单张图像调用 `thop.profile(model, inputs=(input,))`，按原函数打印 `flops / 1e9`、`params / 1e6` 和直接求和参数量，写入训练日志；`model_complexity.txt` 另存结构化数值。THOP 原始第一返回值**不乘 2**。整数 `size` 保持作者的三通道输入，NCHW 形状允许 Synapse/ACDC 按真实灰度输入统计。原作者将 THOP 第一返回值标作 FLOPs，THOP 也常称其为 MACs；记录中保留同一原始值，避免偷换计数口径。

复杂度文件只记录模型参数量与计算量，不存储开关组合。PDS 的参数与推理计算进入整网及解码器统计；BFC 是训练辅助损失，不增加推理参数或计算量。

二分类的 Polyp、BUSI、ISIC、Cell 统一按原论文表 1 的 3×256×256 整网输入统计，即使训练输入设为 352×352 也不改变论文对比尺寸。Synapse 与 ACDC 当前另记录其 1×224×224 整网值。所有入口还记录 `decoder_224`：用 7/14/28/56 四级编码器特征作为输入，对应原论文表 6 的 **224×224 解码器单独统计**。PDS 位于当前模型的解码路径但注册在 `pair_modules` 下；`decoder_224` 显式包含它，避免漏算。BFC 仅在训练辅助损失中执行，不计入推理 FLOPs。THOP 对函数式算子有覆盖边界，论文中应说明使用原作者的 THOP 口径、版本、模型配置和输入尺寸；这表示与原作者方法一致，不代表真实硬件指令数。

## 相关文献

- EMCAD：https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html
- 抗混叠：https://proceedings.mlr.press/v97/zhang19a.html （讨论降采样，不能直接证明本CAA上采样有效）
- 区域不均衡：https://arxiv.org/abs/1706.05721
- 距离监督：https://proceedings.mlr.press/v102/kervadec19a.html
- 形状与语义流：https://openaccess.thecvf.com/content_ICCV_2019/html/Takikawa_Gated-SCNN_Gated_Shape_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.html
- 二阶统计：https://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.pdf
- 梯度冲突：https://arxiv.org/abs/2001.06782 （需要测量，不能仅凭模块名断言冲突）

文献支持研究方向，不证明本组合的原创性或实验收益；不能穷尽全部互联网文献。本次优先20组可明确描述机制的实验候选，其余35组暂缓，未断言必然冲突。
