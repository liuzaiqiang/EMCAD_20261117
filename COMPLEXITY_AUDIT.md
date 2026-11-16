# FRD＋MCC复杂度审计

本分支训练入口仍自动调用log_model_complexity。现在不依赖THOP模块钩子，改用PyTorch算子级拦截；同时覆盖nn.Conv2d、函数式Sobel卷积、Linear和注意力矩阵乘法。无需额外安装统计包，建议沿用服务器PyTorch2.5.1；旧版PyTorch缺少TorchDispatchMode时需先升级环境。

## 论文计数口径

- Parameters：模型已分配参数元素总数；trainable_parameters单列。包含已分配但本输入未使用的灰度适配器等参数；buffer不计入。
- MACs：仅卷积和矩阵乘加，1MAC=2FLOPs。卷积为输出元素数×每输出卷积核元素数，矩阵乘为输出元素数×归约维度。
- FLOPs字段为2×MACs，必须在论文明确此约定，不能称为所有标量运算的完整总数。
- 激活、归一化、池化、插值（含抗混叠）、逐元素操作及归约单独列入excluded_non_mac_calls；数据搬运不计算术量。它们没有被伪装成MACs。该口径用于四种消融的一致比较。
- scope为完整四预测头eval前向，包括FRD和MCC；不计损失、反向、优化器、数据处理。batch总量与每图量都记录。
- 未分类算子使统计结果标为failed，MACs/FLOPs为空；默认记录错误并继续训练。COMPLEXITY_STRICT=1时中止，适合论文统计专项检查。

## 已测数据

CPU、PyTorch2.5.1、PVTv2-B2、输入1×1×224×224、9类、完整四头前向：

| A(FRD) | B(MCC) | Parameters | MACs | FLOPs(2MACs) |
|---|---|---:|---:|---:|
|0|0|26,772,635|4,395,439,082|8,790,878,164|
|1|0|26,772,892|4,406,277,098|8,812,554,196|
|0|1|26,776,830|4,405,706,346|8,811,412,692|
|1|1|26,777,087|4,416,544,362|8,833,088,724|

FRD新增257参数，MCC新增4195参数，合计4452。以上不得套用到不同骨干、类别数、输入通道或分辨率。

统计在隔离的CPU模型副本执行（先逐个复制参数和buffer到CPU，再复制模型结构，避免先在GPU复制整网），保留实时训练模型的参数、buffer、training状态和随机数序列；不注册THOP的total_ops/total_params。原日志参数量/MACs/FLOPs字段保留，并额外输出<原统计文件>.json，含算子清单、排除项、输入形状、开关、版本和计数口径。

测试：python tests/test_complexity_audit.py。包括函数式分组卷积和矩阵乘法手算核对、未知算子拒绝、实时模型不被修改。完整运行数据见COMPLEXITY_RESULTS.json。参考：https://docs.pytorch.org/docs/main/generated/torch.utils.flop_counter.FlopCounterMode.html 。普通torch.profiler的with_flops也只覆盖特定运算，不能直接视作所有运算总量：https://docs.pytorch.org/docs/main/profiler 。

独立交叉核对：PVTv2-B2、1×3×352×352、1类、11状态：参数26,768,791；MACs11,407,458,618；FLOPs22,814,917,236，与PyTorch FlopCounterMode一致。

## 第二次修复：扩展计数与运行安全

`macs`与`flops=2*macs`继续表示卷积/矩阵口径，供与采用相同口径的论文比较。另输出`supplemental_scalar_ops_estimate`、`supplemental_scalar_ops_by_operator`、`expanded_ops_estimate`。后者是明确公式下的算法估算，不能称为硬件指令数或与THOP/fvcore结果直接比较。

扩展口径：逐元素加减乘除、平方、开方、绝对值、tanh和比较各按每元素1次；sigmoid按4次、SiLU按5次、GELU按8次；sum按输入元素减输出元素，mean再加每输出1次除法；softmax按5N；LayerNorm按8N、推理BatchNorm按4N的声明估算；普通双线性插值按每输出4乘3加。抗混叠双线性按先水平后垂直的可分离三角滤波、边界裁剪的有效tap数计算，每点k乘+(k-1)加；尺寸不变轴跳过。池化按窗口覆盖量。卷积和矩阵偏置加法另计。

这些是统一的理论比较口径：不包含插值权重生成、坐标计算、索引、内存搬运；超越函数按约定成本，不表示实际硬件成本。尤其融合归一化/激活为估算。未知算子不得得到成功状态。论文可报告Params和GMACs，并在方法中明确口径；如果报告扩展量，必须使用“estimated expanded operations”并附规则。

新增6项专项测试覆盖：手算分组函数卷积与矩阵乘、MCC归约/普通与抗混叠插值公式、未知算子失败、失败不阻断训练、模型状态和共享参数及CPU随机数保持。CUDA显存行为依赖代码路径审查，尚未在本环境实测GPU。
