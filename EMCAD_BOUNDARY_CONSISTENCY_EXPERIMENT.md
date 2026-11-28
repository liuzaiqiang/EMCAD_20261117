# EMCAD 局部边界一致性改造

## 改造方法

**均值守恒的跳连引导上采样 `PHASE_UPSAMPLING=1`**。三个 EUCB 对粗特征先做固定双线性插值。每个 2×2 单元内，将下一跳编码器特征的可学习投影去均值并做有界归一，得到细节方向 $q$。把解码器单元内的零均值细节 $d$ 投影到这个方向，再回加 $alpha q(q^\top d)$。其中 $\alpha$ 是限制在 $[0,1]$ 的可训练增益。此项在每个通道、每个单元内的和为零，因此插值结果的单元均值在后续 EUCB 卷积之前保持不变。恒定区域和与跳连方向正交的细节没有此项修正。这里的约束作用于中间特征，最终预测还会经过深度卷积、跳连融合和分割头。

每个通道和单元的四个修正分量乘以同一个限幅系数，使叠加值落在粗网格 3×3 邻域的最小值和最大值之间。共用系数使限幅后仍保持单元均值。训练将限幅系数作为常量，反传经受限残差回到投影分支。

该方法与 FADE、SAPA 都使用编码器信息辅助恢复边界。FADE 生成动态核并门控编码器细节，SAPA 用局部语义相似性决定重组归属。Unsharp Mask Guided Filtering 的单系数细节转移与这里的方向投影更接近。本实现把方向投影限定在每个互不重叠的 2×2 单元内，通过同一限幅系数同时保持插值均值和粗网格邻域值域。超分辨率也有下采样一致性的先例，例如 PULSE。 [FADE](https://arxiv.org/abs/2407.13500) · [SAPA](https://proceedings.neurips.cc/paper_files/paper/2022/hash/83ccb398f3ce9c4d137011f36a03c7d4-Abstract.html) · [Unsharp Mask Guided Filtering](https://pure.uva.nl/ws/files/68440460/Unsharp_Mask_Guided_Filtering.pdf) · [PULSE](https://openaccess.thecvf.com/content_CVPR_2020/html/Menon_PULSE_Self-Supervised_Photo_Upsampling_via_Latent_Space_Exploration_of_Generative_Models_CVPR_2020_paper.html)

**局部梯度一致的跨核尺度校正 `STAGE_ADAPTIVE_KERNELS=1`**。四个 MSCB 先保留 EMCAD 的分支原始求和。对相邻尺度求响应差 $D$。并行模式按核大小确定相邻尺度，串行模式按计算顺序。计算 $D$ 与本级 MSDC 输入在各自通道均值上的水平、垂直差分，取 3×3 邻域的归一化梯度内积 $a$。将 $\frac12\tanh(g)\,a\,(D-\mathrm{AvgPool}_{3\times3}(D))$ 加到分支原和，其中 $g$ 是每级、每相邻尺度对独立学习的增益，初始化为零。常量分支差与缺乏一致梯度的区域不产生校正。所有位置共享每级增益，局部响应是否被修正由空间梯度决定。

多尺度高频增强和局部梯度先验已经用于医学图像融合。本实现的具体对象是同一 EMCAD 解码级的相邻核响应差，以及该差与原输入之间的局部梯度一致性。原论文 EMCAD 采用多尺度深度卷积分支相加。 [局部梯度能量与医学图像融合](https://www.sciencedirect.com/science/article/pii/S0010482520303796) · [EMCAD](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html)

## 运行与评估

两个开关均只接受 `0` 和 `1`。`0` 使用对应的 EMCAD 基础路径，`1` 启用本分支的模块。跨尺度校正以相加为基线，启用时使用默认的加法聚合。

```bash
PHASE_UPSAMPLING=1 STAGE_ADAPTIVE_KERNELS=1 bash sh/start_train_synapse.sh
```

| 启动器 | 覆盖的数据集 |
| --- | --- |
| `start_train_synapse.sh` | Synapse |
| `start_train_acdc.sh` | ACDC |
| `start_train_polyp.sh` | ClinicDB、ColonDB、Kvasir、ETIS、BKAI |
| `start_train_isic.sh` | ISIC2017、ISIC2018 |
| `start_train_busi.sh` | BUSI |
| `start_train_cell.sh` | DSB18、EM |

在其他启动器或 `run_seed_queue.sh` 前可设置同样的变量，并用 `DATASET_NAME` 选择同一入口下的数据集。正常测试使用对应训练目录的配置与 checkpoint。直接运行 SLDGroup 旧版 Polyp 入口时，训练和测试显式使用 `--phase_upsampling 1 --stage_adaptive_kernels 1`。

Windows 上路径过长时，训练输出采用较短的目录层级，完整参数仍记录在 `config.json`。Synapse 测试使用显式传入的 `--test_save_dir`，省略该参数时在 checkpoint 同级生成独立结果目录。

固定划分、种子、输入分辨率、训练预算和 checkpoint 选择准则，比较 `0/0`、`1/0`、`0/1`、`1/1`。Synapse 和 ACDC 同时看逐器官 Dice 和 HD95。两个机制的用途是减少边界平滑和尺度响应干扰。弱边缘若在两路特征中都没有可识别的信号，预期收益有限。是否提高 Dice、降低 HD95，由完整训练结果判断。
