# EMCAD + 困难区域监督 + UACA

## 来源与实现边界

UACA 的三区域权重、上下文汇聚、query/key/value 相关性和残差 logits 更新，对照 [论文](https://arxiv.org/abs/2107.02368) 与 [作者 MIT 许可实现](https://github.com/plemeri/UACANet/blob/main/lib/modules/attention_module.py)。本工程移植的是 UACA 注意力模块，不是完整 UACANet：编码器和 EMCAD 解码器、四级深监督损失保持本工程原样。作者的完整网络在息肉数据集上的成绩不能直接当作此组合的预期成绩。

EMCAD 的 d4 初始预测依次引导 d3、d2、d1 的 UACA。每级把上一层细化特征上采样投影，与本级 EMCAD 特征拼接；沿用原有四个分割头，把本级增量预测加到上一层 logits。所有四级输出仍走原有深监督训练。二分类采用 sigmoid 概率；Synapse、ACDC 等互斥多分类把全部非背景类 softmax 概率相加作为 UACA 的前景图。这是多分类移植所需的明确适配，不能声称与作者二分类实验完全相同。

## 开关与数据集

六个训练脚本使用 `UACA_ENABLED=0/1`，默认 `0`；`SMALL_TARGET_HARD_SUPERVISION=0/1` 独立控制困难区域损失。两者都为 `0` 时不创建 UACA 参数、不进入 UACA 前向，使用原始 EMCAD 和原有损失。开关写入命令参数、启动日志 `PARAM_NAMES`、训练 `config.json` 和实验目录名；测试从训练配置恢复网络结构。

| 入口 | 数据集 |
| --- | --- |
| `start_train_synapse.sh` | Synapse |
| `start_train_acdc.sh` | ACDC |
| `start_train_polyp.sh` | BKAI、ClinicDB、ColonDB、ETIS、Kvasir |
| `start_train_isic.sh` | ISIC2017、ISIC2018 |
| `start_train_busi.sh` | BUSI |
| `start_train_cell.sh` | DSB18、EM |

Synapse 原有 `SYNAPSE_TOPK_ENABLED/SIZE/DICE_THRESHOLD` 保留。ACDC、Polyp、ISIC、BUSI、Cell 可设 `TOPN_ENABLED=1 TOPN_SIZE=10`，按验证 Dice 排名前 N，`topn/trajectory.csv` 逐验证轮记录 Dice 和 HD95。ACDC 在此模式下计算完整病例 HD95；二分类仅对有定义的表面距离取均值，单边空掩膜记录 NaN。TopN 模式会增加验证时间和磁盘占用。最终测试仍应在独立测试集完成。

训练启动时 `model_complexity.txt` 直接记录作者已有的 `CalParams(model, input_tensor)` 输出，即 THOP 的 FLOPs/Params 格式；不把 MACs 再乘二。THOP 对 UACA 中 `torch.bmm` 之类的函数运算可能不计入，故该值是工具口径估计，不应当作完整的硬件操作计数。

## HD95 假设与实验

UACA 同时聚合不确定背景和前景上下文，理论上有机会减少边界外的误检与轮廓偏移，从而降低 HD95；也可能把真实小目标压掉，或保留远端假阳性，导致 Dice/HD95 反向变化。论文没有证明本工程、这 12 个数据集的 HD95 会下降。建议固定数据划分与随机种子，比较四组：EMCAD、仅困难区域监督、仅 UACA、二者同时开启；报告逐病例 Dice、HD95、HD95 未定义比例、各类别和多随机种子置信区间。用户问题中提到的“预测图辅助模块”是另一模块 PMA，本分支实现的是 UACA，没有引入 PMA。
