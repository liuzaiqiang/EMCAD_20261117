# FRD＋MCC复杂度审计

本分支训练入口仍自动调用log_model_complexity。现在不依赖THOP模块钩子，改用PyTorch算子级拦截；同时覆盖nn.Conv2d、函数式Sobel卷积、Linear和注意力矩阵乘法。无需额外安装统计包，建议沿用服务器PyTorch2.5.1；旧版PyTorch缺少TorchDispatchMode时需先升级环境。

## 论文计数口径

- Parameters：模型已分配参数元素总数；trainable_parameters单列。包含已分配但本输入未使用的灰度适配器等参数；buffer不计入。
- MACs：仅卷积和矩阵乘加，1MAC=2FLOPs。卷积为输出元素数×每输出卷积核元素数，矩阵乘为输出元素数×归约维度。
- FLOPs字段为2×MACs，必须在论文明确此约定，不能称为所有标量运算的完整总数。
- 激活、归一化、池化、插值（含抗混叠）、逐元素操作及归约单独列入excluded_non_mac_calls；数据搬运不计算术量。它们没有被伪装成MACs。该口径用于四种消融的一致比较。
- scope为完整四预测头eval前向，包括FRD和MCC；不计损失、反向、优化器、数据处理。batch总量与每图量都记录。
- 未分类算子使统计失败并阻止训练启动，避免产生看似有效但漏项的数据。

## 已测数据

CPU、PyTorch2.5.1、PVTv2-B2、输入1×1×224×224、9类、完整四头前向：

| A(FRD) | B(MCC) | Parameters | MACs | FLOPs(2MACs) |
|---|---|---:|---:|---:|
|0|0|26,772,635|4,395,439,082|8,790,878,164|
|1|0|26,772,892|4,406,277,098|8,812,554,196|
|0|1|26,776,830|4,405,706,346|8,811,412,692|
|1|1|26,777,087|4,416,544,362|8,833,088,724|

FRD新增257参数，MCC新增4195参数，合计4452。以上不得套用到不同骨干、类别数、输入通道或分辨率。

统计在隔离的CPU模型副本执行，保留实时训练模型的参数、buffer、training状态和随机数序列；不注册THOP的total_ops/total_params。原日志参数量/MACs/FLOPs字段保留，并额外输出<原统计文件>.json，含算子清单、排除项、输入形状、开关、版本和计数口径。

测试：python tests/test_complexity_audit.py。包括函数式分组卷积和矩阵乘法手算核对、未知算子拒绝、实时模型不被修改。完整运行数据见COMPLEXITY_RESULTS.json。参考：https://docs.pytorch.org/docs/main/generated/torch.utils.flop_counter.FlopCounterMode.html 。普通torch.profiler的with_flops也只覆盖特定运算，不能直接视作所有运算总量：https://docs.pytorch.org/docs/main/profiler 。

独立交叉核对：PVTv2-B2、1×3×352×352、1类、11状态：参数26,768,791；MACs11,407,458,618；FLOPs22,814,917,236，与PyTorch FlopCounterMode一致。
