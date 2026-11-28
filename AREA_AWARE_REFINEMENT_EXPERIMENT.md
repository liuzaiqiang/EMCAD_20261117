# EMCAD 面积感知点细化实验

## 开关与数据集

| `AREA_AWARE_REFINEMENT` / `--area_aware_refinement` | 行为 |
| --- | --- |
| `0` | 关闭点细化，可独立启用小器官感知的困难区域监督。 |
| `1` | 启用基于 EMCAD 相邻解码级的面积感知点细化。 |

| 数据集 | 训练脚本 | 测试脚本 |
| --- | --- | --- |
| Synapse | `start_train_synapse.sh` | `start_test_synapse.sh` |
| ACDC | `start_train_acdc.sh` | `start_test_acdc.sh` |
| ClinicDB、ColonDB、Kvasir、ETIS、BKAI | `start_train_polyp.sh` | `start_test_polyp.sh` |
| ISIC2017、ISIC2018 | `start_train_isic.sh` | `start_test_isic.sh` |
| BUSI | `start_train_busi.sh` | `start_test_busi.sh` |
| DSB18、EM | `start_train_cell.sh` | `start_test_cell.sh` |

在 `sh/` 中的对应启动器前设置 `AREA_AWARE_REFINEMENT=1`，同一脚本下的具体数据集用 `DATASET_NAME` 选择。Synapse 启动器默认启用，其余启动器可显式设置 `1`。训练把开关写入实验路径和 `config.json`，测试读取训练配置。直接使用 `train_polyp_SLDGroup.py` 时可传 `--area_aware_refinement 1`，其测试脚本从权重结构识别新点头。由于采样、推理和点头参数命名发生变化，面积感知细化实验应使用本实现重新训练。

```bash
AREA_AWARE_REFINEMENT=1 SMALL_TARGET_HARD_SUPERVISION=1 DATASET_NAME=BKAI bash sh/start_train_polyp.sh
```

## 选点与点预测

1. 取 EMCAD 紧邻的两级原生 logits，`p1` 约为输入的 1/4，`p2` 约为 1/8。单通道任务将 sigmoid 概率及其补数视为前景和背景。多类任务使用 softmax。每一级按图像统计各前景类的预测支持面积，即类别获胜位置的 top-1 与 top-2 概率差之和除以该级像素数。取两级均值 $A_c$，可见前景的权重为 $0.05/(0.05+A_c)$，背景和两级均无支持的类别权重为零。该量估计的是当前切片或图像上的置信度加权面积。
2. 在候选位置，由当前细级预测确定胜出类 $a$ 与竞争类 $b$，并以 $U=2p_b/(p_a+p_b)$ 表示二者的相对不确定性。将较粗一级对这对类别的概率记为 $q$，反向证据为 $D=\max(q_b-q_a,0)$。优先级为 $U+\max(w_a,w_b)(0.5U+D)$。竞争类可以是落后于背景的小目标，胜出类也可以是粗一级不支持的前景。每次先按 $U$ 保留约一半重点名额，再按优先级选取余下点，同一批候选点只选一次。
3. 点头读取 EMCAD 的末级解码特征、原始图像的中心值和四邻域强度差、末级 logits 以及相邻两级的有符号概率差。点头输出零初始化、幅度限制在 3 以内的 logit 修正量。推理在原图分辨率一次选取最多 1568 点，仅在选中的像素中心将残差加到 `p1` 的插值结果上。
4. 训练每张图使用 196 点。将图像划成 24×24 格，每格均匀抽一个候选点，再从中选出 147 个重点点。另外在 7×7 格上每格独立采一个探索点，确保各区域都能提供点监督。增强后的真实掩膜用于点级 BCE 或交叉熵。面积估计和选点在训练与推理时均由预测产生。原有的密集监督和小器官困难区域监督使用各自的开关。

面积权重、重点点数和幅度上限是本次实现采用的实验默认值。跨尺度概率反向证据瞄准可能遗漏的小目标和可能误检的前景，点级图像强度差提供局部边界信息。两级预测都漏掉目标时，面积分配无法专门提高该目标的优先级，该情况仍由通用不确定性和训练中的均匀采样覆盖。

## 与相关工作的关系

- **PointRend** 是点特征插值和按预测不确定性选点的先行工作，训练采用均匀过采样，推理使用逐级细分。本实现采用空间分层的训练点、跨解码级的面积及竞争证据，以及原分辨率单次有界残差修正。 [Kirillov et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)
- **Weakly Supervised Segmentation of Small Buildings With Point Labels** 将预测小实例面积与标签重叠关系用于采集疑似假阳性训练点，并使用标注点记忆库。本实现以同一 EMCAD 解码器两级预测的类别支持面积和反向竞争为依据，训练和推理共享选点依据。 [Lee et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Lee_Weakly_Supervised_Segmentation_of_Small_Buildings_With_Point_Labels_ICCV_2021_paper.html)
- **3D ANU-Net 点采样** 依据粗预测的不确定性选点，将点特征写回中间特征图，并另设假阳性分类模块。这里将小前景类别的跨级竞争证据用于选点，并在最终 logits 上修正。 [3D CT 点采样](https://www.mdpi.com/2076-3417/10/19/6838)
- **IRM-UNet** 使用粗预测的前两类概率差在 3D 体素中做 Top-K 采样，之后进行隐式点预测。这里用 EMCAD 两级前景面积和相邻级对同一竞争类别的反向证据分配部分点预算。 [IRM-UNet](https://link.springer.com/article/10.1007/s10489-026-07162-9)
- **UDBRNet** 由三个解码器的掩膜并集和交集计算分歧，再由边界细化网络生成密集残差。这里用同一个 EMCAD 解码器的相邻层 logits 度量类别竞争，在稀疏像素上预测修正量。 [UDBRNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182520/)
- **Implicit Anatomical Rendering** 使用随机专家参数化多尺度的隐式函数。这里使用单个点头，融合固定相邻级概率差与图像强度差。 [IAR, MICCAI 2023](https://arxiv.org/abs/2304.03209)

以上对照记录的是检索到的机制差异。论文撰写应保留对 PointRend、EMCAD 和相关医学图像点细化工作的引用。文献对照无法量化全网模块相似度，也无法证明首创。 [EMCAD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html)

## Dice 与 HD95 的验证

在同一数据划分、训练轮次、选择标准、阈值和种子下，固定小器官监督为 `1`，比较面积感知点细化关闭和启用。加入两项都关闭的 EMCAD 作为基础对照。成对报告各数据集的 Dice 与 HD95，并对 Synapse 和 ACDC 同时报告逐器官结果。沿用各入口的数据尺度、空掩膜规则和距离单位。保留每个种子的结果以及均值和波动，分析退化病例。源代码和冒烟测试只能验证方法运行，指标收益以真实训练和评估结果为准。

Synapse 使用现有 TopN 设置。ACDC 和共用二分类入口可设置 `TOPN_ENABLED=1 TOPN_SIZE=10`，记录各次验证的 Dice 和 HD95，并按 Dice 保留 checkpoint。THOP 可能低估排序、采样和散射操作的开销，因此复杂度对照还需报告实测推理时间。
