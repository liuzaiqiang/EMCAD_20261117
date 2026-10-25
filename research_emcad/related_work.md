# 已核验相关工作与创新边界

| 论文 | 来源 | 已有机制及与本方案的关系 |
|---|---|
| Rahman, Munir, Marculescu, *EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation*, CVPR 2024 | [论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.pdf), [官方代码](https://github.com/SLDGroup/EMCAD) | MSCB 多核、LGAG skip、CAB/SAB 已有。新增机制不得仅重复这些操作。 |
| *CCQ: Cross-Class Query Network for Partially Labeled Organ Segmentation*, AAAI 2023 | [官方论文与 DOI](https://ojs.aaai.org/index.php/AAAI/article/view/25264), [代码](https://github.com/SooLab/CCQ) | 器官类别查询已有先例；其目标是部分标注的类间关系，本方案针对 EMCAD 的逐类别解码尺度选择。 |
| Wang et al., *Narrowing the semantic gaps in U-Net with learnable skip connections*, Neural Networks 2024 | [论文](https://www.sciencedirect.com/science/article/pii/S0893608024004702) | 可学习 skip 与编码解码语义差异已有研究；互补跳连需要与普通门控做等预算对照。 |
| Li et al., *FusionU-Net: U-Net with Enhanced Skip Connection for Pathology Image Segmentation*, ACML 2024 | [论文](https://proceedings.mlr.press/v222/li24a.html) | skip 融合改进已有先例，不能仅凭更换融合算子宣称新颖性。 |
| *Enhancing Medical Image Segmentation with TransCeption: A Multi-Scale Feature Fusion Approach*, arXiv 2023 | [预印本](https://arxiv.org/abs/2301.10847) | 多尺度融合及上下文桥已存在；其正式发表 venue 与作者细节尚未在本次检索中核实，不能用来证明新颖性。 |

归类：类别查询、尺度融合及可学习 skip 均属于已有机制；在 EMCAD 的具体接入属于适配/组合。逐类选择三个解码尺度，以及共享/互补 skip 分解能否独立改善边界与 Dice，是待实验验证的具体假设。目前没有完成系统专利/论文查重，不能断言正式新颖性；优于任何已有方法的说法也没有证据。
