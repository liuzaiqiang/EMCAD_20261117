# EMCAD 小结构局部支撑细化

## 启动与数据集

设置 `LOCAL_SUPPORT_REFINEMENT=1` 启用本版，`0` 关闭。Python 参数为 `--local_support_refinement 1`。Synapse 的 shell 默认启用，其余启动器按需显式设置。训练将开关写入实验目录和 `config.json`，测试从训练配置恢复。原版 PointRend 权重的点头结构不同，本版需要重新训练。

| 数据集 | `sh/` 中的训练脚本 | 对应测试脚本 |
| --- | --- | --- |
| Synapse | `start_train_synapse.sh` | `start_test_synapse.sh` |
| ACDC | `start_train_acdc.sh` | `start_test_acdc.sh` |
| ClinicDB、ColonDB、Kvasir、ETIS、BKAI | `start_train_polyp.sh` | `start_test_polyp.sh` |
| ISIC2017、ISIC2018 | `start_train_isic.sh` | `start_test_isic.sh` |
| BUSI | `start_train_busi.sh` | `start_test_busi.sh` |
| DSB18、EM | `start_train_cell.sh` | `start_test_cell.sh` |

```bash
LOCAL_SUPPORT_REFINEMENT=1 SMALL_TARGET_HARD_SUPERVISION=1 DATASET_NAME=BKAI bash sh/start_train_polyp.sh
```

## 方法

第一项小器官困难区域监督使用训练掩膜提升困难前景对密集损失的贡献。本模块在细化阶段利用预测本身的局部支撑，关注小病灶及器官狭窄部位。二分类将 sigmoid 预测展开为背景和前景概率，多类取 softmax，背景不参加结构评分。掩膜只用于训练损失。

1. **局部支撑评分。** 取 EMCAD 末级原生预测概率。在每个前景类别上，计算 3×3 邻域概率上界 $p_{\max}$、下界 $p_{\min}$ 及 5×5 邻域内该类别胜出的比例 $A$。构造 $(p_{\max}-p_{\min})p_{\max}(1-A)^2$，再取前景类别最大值 $S$。用 $2S+0.25U$ 排序，其中 $U=2p_{(2)}/(p_{(1)}+p_{(2)})$。同置信度的边缘中，局部支撑较少的前景得到更高优先级。$A$ 是预测空间上的局部覆盖比例，尺度以末级预测网格计量。
2. **锚点选择。** 在末级预测网格上，每个 2×2 区块只提交一个最高分位置，避免多个相邻候选点都来自同一局部边缘。训练选最多 32 个锚点，通常保留 8 个随机选择的候选锚点。推理选最多 128 个锚点。
3. **窗口残差。** 锚点对应原图上的 5×5 窗口。点头读取末级解码特征、锚点 logits、结构与不确定性分数，以及窗口平均强度和水平、垂直强度差。每类预测三个系数表示窗口内的常数项及横纵方向斜率，通过 `tanh` 将修正幅度限制在 3。训练对窗口的像素做 BCE 或交叉熵。推理在原分辨率对残差加径向衰减，重叠位置用权重和归一后叠加到双线性基线。零初始化使初始推理输出等于基线。

这套设计优先处理带有局部概率变化的小结构。末级预测完全没有目标线索时，只能依赖不确定性进入候选。局部面积也可能为噪声或误检分配点预算。窗口大小和预算是实验默认值。

## 与公开方法的对照

- **PointRend** 采用预测不确定性采样、独立点分类与逐级细分。这里以局部前景支撑为主要排序依据，用锚点共享的三个系数生成窗口残差，并在原分辨率单次合并。 [Kirillov 等，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html) · [公开代码](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)
- **Boundary Patch Refinement** 沿实例边界生成候选框并使用 NMS，裁剪图像及掩膜后由单独的高分辨率分割网络重预测，再合并重叠 logits。本版用 EMCAD 末级语义网格做局部支撑排序，直接从锚点特征预测空间相关的低维残差。 [Tang 等，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html) · [公开代码](https://github.com/chenhang98/BPR)
- **SparseRefine** 用熵筛选像素、用稀疏卷积提取特征、再学习门控融合两套预测。本版使用同类局部覆盖率和概率起伏筛选锚点，使用固定的空间基函数及衰减权重合并修正。 [SparseRefine，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8392_ECCV_2024_paper.php)
- **IRM-UNet** 对 3D 胰腺 CT 按前两类概率差进行 Top-K 采样，再通过隐式函数估计点预测。这里以 2D EMCAD 局部前景支撑为主要依据，窗口像素共享有界仿射残差。 [IRM-UNet](https://link.springer.com/article/10.1007/s10489-026-07162-9)

实验应引用 EMCAD 及相关先行工作。以上对照给出可复核的结构差异，无法作为全网检索完备性、首创性或相似度百分比的证明。 [EMCAD，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html)

## 指标验证

以相同数据划分、种子、训练预算、阈值和权重选择标准比较 EMCAD、仅小器官监督、仅局部支撑细化及两者组合。Synapse 和 ACDC 同时报告逐器官结果，其余数据集报告 Dice 和 HD95 的病例汇总。沿用原有的距离单位、空掩膜规则和 TopN 设置。THOP 可能低估排序、采样与聚合的开销，如需比较部署成本应实测推理时间。Dice 和 HD95 的变化以完整验证结果为准。
