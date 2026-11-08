# CAA + 上下文细节细化：实施与实验方案

## 当前分支审计

实际模型入口是 `lib/networks.py` 的 `EMCADNet`。编码器输出四级特征；
`EMCAD` 依次计算 d4→d3→d2→d1，每次以 EUCB 上采样，再以 LGAG 筛选同尺度 skip，
相加后经 CAB/SAB/MSCB。四个 1×1 分割头输出 p4→p1，评估主要使用 p1。
CAA 只改变三个 EUCB 的上采样路径。当前分支没有历史文档提及的轮廓反馈、
全分辨率模块或有界形变模块的运行时代码；不能把历史说明当作当前实现。
本次只增加独立的上下文细节细化（CDR），不改变损失、数据、评估公式和模型选择。

12 个数据集由六类入口覆盖：Synapse、ACDC、五个 Polyp（BKAI、ClinicDB、
ColonDB、ETIS、Kvasir）、ISIC2017/18、BUSI、Cell（DSB18、EM）。
ISIC、BUSI 训练与测试包装器复用通用 Polyp 参数和模型构造；Cell shell 包装器
复用 Polyp shell。Synapse 与 ACDC 使用各自的多类体数据评估。
`config.json` 保存开关；测试从 checkpoint 相邻配置恢复，旧配置缺字段按 0。
严格 state_dict 加载防止新旧结构混用。实验路径增加 `cdr_0/1`。

## 方案比较与证据边界

| 候选 | 张量路径与动机 | 与 CAA 的关系 | 代价和风险 | 最小实验 |
| --- | --- | --- | --- | --- |
| CDR（本次实现） | d2 投影至 d1 通道并对齐空间；d1 局部高通提供细节；语义相关性门控有界残差，仅修正 p1 前的 d1 | CAA 管上采样，CDR 管解码完成后的细节筛选 | B2/224 增约 12,868 参数、约 21.08M 卷积 MAC；可能放大噪声或改善有限 | 四组合、逐病例 Dice/HD95 和边界可视化 |
| 全尺度轻量聚合 | 将 d4/d3/d2/d1 投影并对齐，融合后修正 d1；参考 SegFormer 的跨层聚合 | 作用节点不同，但容易与已有多尺度 MSCB 重叠 | 多次投影/插值，显存与延迟较高；低分辨率结构可能主导小目标 | 仅在代表性数据集比较单尺度、两尺度和全尺度 |
| 显式形状支路 | 从浅层特征提取边界支路，与语义支路门控融合；参考 Gated-SCNN | 理论上补足 CAA 的边界语义，但可能重复 LGAG 的门控 | 需要边界监督或可靠的伪边界定义，跨模态噪声风险较高 | 固定损失比较无监督支路与明确边界监督支路 |
| 密集 skip 变换 | 在编码器 skip 与解码器间增加逐级语义桥；参考 UNet++ | 与 CAA 正交，但对现有 LGAG/MSCAM 有明显重叠 | 参数与改动面大，难以把收益归因于一个轻量模块 | 一层桥与多层桥分开消融 |

文献只支持这些机制值得测试，不能证明本实现的 Dice/HD95 会改善：
[EMCAD（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html)、
[Gated-SCNN（ICCV 2019）](https://openaccess.thecvf.com/content_ICCV_2019/html/Takikawa_Gated-SCNN_Gated_Shape_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.html)、
[SegFormer（NeurIPS 2021）](https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)、
[UNet++（2018）](https://arxiv.org/abs/1807.10165)。
CDR 是基于这些一般思路在当前 EMCAD 路径上的待验证设计，尚无原创性或性能结论。

## CDR 定义

令 `F=d1`、`C=resize(conv1x1(d2), size(F))`，
`D=F-avgpool3x3(F)`（复制边界填充）。由 `|tanh(D)|` 的通道均值和
`tanh(F)×tanh(C)` 的通道均值生成单通道 sigmoid 门 `G`。
`R=tanh(pointwise(depthwise3x3(D)))`，最终 `F'=F+tanh(alpha)×G×R`。
`alpha` 初始化为 0，初始前向严格等于原 d1。仅启用时实例化该模块，且在旧输出头之后
实例化，因此关闭时旧 state_dict 和初始化顺序保持不变。启用时非常规 H/W 在入网前
复制填充到 32 的倍数，四个 logits 裁回原尺寸；标准 224/256/352 输入不变。

参数量（仅 CDR）：`C2×C1 + 9×C1 + C1×C1 + 4`；B2 的 C2=128、C1=64，
得到 12,868。224 输入下卷积 MAC 约 21.08M；若按乘加=2 FLOPs 计约 42.16M，
不含插值、池化、逐元素操作。实际延迟、峰值显存与整网 FLOPs 必须在服务器测量。

## 实验执行

先在 Synapse、ACDC、ClinicDB、BUSI 四种模态/任务做筛选；但 Synapse 训练代码
使用 `test_vol.txt` 每轮选 `best.pth`。若服务器该列表是最终测试集，现有协议存在
测试泄漏，必须先按预先确定的独立验证/测试清单重新设计正式实验；不能把现有
`test_vol` 指标作为最终无偏测试结果。不要在看过测试结果后挑模块。

每个数据集、每个预先选定 seed（建议至少 3 个）按如下四组合运行；保持当前脚本的
epoch、学习率、监督、预训练文件哈希、划分、增强和评估规则不变。
这些启动器使用 `nohup ... &`，命令返回只代表任务已启动；单卡上应等上一次训练
结束并检查日志后，再启动下一组，不能把下列四行直接当作串行训练队列。

```bash
USE_CONTENT_AWARE_ANTIALIAS=0 USE_CONTEXT_DETAIL_REFINEMENT=0 SEED=2222 bash sh/start_train_synapse.sh
USE_CONTENT_AWARE_ANTIALIAS=1 USE_CONTEXT_DETAIL_REFINEMENT=0 SEED=2222 bash sh/start_train_synapse.sh
USE_CONTENT_AWARE_ANTIALIAS=0 USE_CONTEXT_DETAIL_REFINEMENT=1 SEED=2222 bash sh/start_train_synapse.sh
USE_CONTENT_AWARE_ANTIALIAS=1 USE_CONTEXT_DETAIL_REFINEMENT=1 SEED=2222 bash sh/start_train_synapse.sh
```

将脚本名换成 `start_train_acdc.sh`、`start_train_polyp.sh`（加
`DATASET_NAME=ClinicDB`）、`start_train_busi.sh`，再重复其他预先选定 seed。
扩展阶段按相同开关和 seed 运行所有 12 个数据集；ISIC、Polyp、Cell 通过
`DATASET_NAME` 选子数据集。每次训练的 `best.pth` 要用对应的 `start_test_*.sh`
和 `CKPT=/absolute/path/to/best.pth` 测试。测试结果仍位于该 checkpoint 目录下
独立的 `test_results/<run_id>`，不会相互覆盖。

逐病例记录 Dice、HD95、前景空掩膜状态，并报告每个 seed 的逐数据集均值及跨 seed
均值/标准差。Synapse/ACDC 是体数据多类指标，Polyp/ISIC/BUSI/Cell 是二分类
图像指标，不能直接把不同统计口径汇成一个单一数字。当前二分类实现对单边空掩膜
HD95 记 NaN；ACDC 对单边空掩膜返回 HD95=0；Synapse 对“预测非空、真值为空”
返回 Dice=1、HD95=0。必须把这些现行规则和发生次数同时报告，不能只比较总体均值。
如需修正指标，应另开统一协议、重新评估全部对照组，不能只修新模块组。

本地验证命令：

```bash
python -m unittest tests.test_context_detail_refinement -v
```

当前本机 Python 缺 PyTorch，上述运行时测试会跳过。已进行 Python AST、
Bash `-n` 和 `git diff --check` 静态验证；真实前反向、AMP、训练指标、
整网复杂度、耗时和显存仍须在服务器运行后填写，不能作为已完成结论。
