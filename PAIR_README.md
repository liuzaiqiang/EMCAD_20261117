# FRD＋MCC：跨尺度轮廓与原图细节

研究假设：MCC协调解码特征轮廓，FRD补充输入分辨率的细节；检查边界一致性和细节恢复是否相互支持。

基线：`261117_baseModel`，提交`f0f3c5ba7d1dda1c071e01727feda8235104d62e`。来源及精确提交见PAIR_SOURCES.json。原模块算法保留，组合调度见lib/pair_runtime.py。CAA在基础模型初始化完成后替换上采样，初始化随机序列与旧组合分支可能不同，不能把旧实验数值直接当成本分支结果。

## 开关与运行

所有 sh/start_train_*.sh 和 start_test_*.sh 中两个开关相邻。`QFBLXJZL` 是“全分辨率细节支路”的拼音首字母，对应 FRD；`DCDLKYZX` 是“多尺度轮廓一致性”的拼音首字母，对应 MCC。脚本默认 11，直接运行 Python 默认 00。有 `PARAM_NAMES` 的启动脚本会打印两个开关。

```bash
QFBLXJZL=1 DCDLKYZX=1 bash sh/start_train_synapse.sh
QFBLXJZL=1 DCDLKYZX=0 bash sh/start_train_synapse.sh
QFBLXJZL=0 DCDLKYZX=1 bash sh/start_train_synapse.sh
QFBLXJZL=0 DCDLKYZX=0 bash sh/start_train_synapse.sh
```

测试必须使用与训练相同的开关。先沿用原脚本设置本机conda、数据和预训练路径。12数据集入口：Synapse、ACDC各独立入口；5个Polyp数据集共用polyp入口；BUSI使用busi；ISIC2017/2018使用isic；DSB18/EM使用cell入口。训练损失在三类公共训练器接入，推理统一使用最终输出。00不创建任何附加模块参数、不计算辅助损失。

FRD 与 MCC 都是参与推理前向的有参数模块；MCC 的轮廓是特征梯度，不是额外的标注边界损失。

## 复杂度统计口径

训练日志和每次实验目录的 `model_complexity.txt` 按 EMCAD 原作者 `utils/utils.py::cal_params_flops` 调用 `thop.profile(model, inputs=(input,))`。THOP 第一返回值直接除以 1e9，沿用原作者称作 FLOPs 的标签，**不乘 2**；同时记录 THOP 参数数及 `sum(p.numel() for p in model.parameters())` 的直接参数数。二分类整网按原论文表 1 使用单张 3×256×256 输入，即使训练尺寸为 352 也保持论文对比尺寸；Synapse/ACDC 整网按各自输入尺寸统计。另记录 224×224 的 EMCAD 解码器核心加已开启 FRD/MCC 扩展的统计，明确不含编码器和原预测头；这个扩展范围由本项目定义，不能直接冒充原论文表 6 的纯 EMCAD 解码器。FRD 与 MCC 都纳入整网统计；MCC 内部函数式 Sobel 卷积不被原作者使用的 THOP 默认规则计入，故 THOP 数值可比的是**相同计数方法**，不代表完整硬件运算量。复杂度文件不存储开关组合，组合由实验配置和用户记录对应。当前本地 THOP 0.1.1 测得纯 B2 解码器 224×224 为 0.385190G，而原论文表 6 为 0.381G；论文使用时须如实注明此差异。

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
