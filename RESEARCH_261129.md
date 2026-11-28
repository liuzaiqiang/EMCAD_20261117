# 261129_WTConvxbjj_SFAlignkjdz

基线提交：`977a31c9e4f3d1ce9e5274c5f33a6f223a111ef5`（261119_baseModel）。研究候选，未证明有效或新颖。

## 研究机制与来源

- WTConv：Wavelet Convolutions for Large Receptive Fields (ECCV 2024)，https://github.com/BGU-CS-VIL/WTConv 。保留多级Haar分解与逆变换；增加每通道四子带之间的可学习门控交互，置于深层d4/d3残差路径。验证：原WTConv、静态尺度、跨子带门控。
- SFAlign：Semantic Flow for Fast and Accurate Scene Parsing (ECCV 2020)，https://github.com/lxtGH/SFSegNets 。保留特征位移采样；估计正反向位移，循环误差构造可靠性，与未变形跳接混合；置于EUCB与LGAG之间。双向流不是本研究首创。验证：单向、双向无可靠性、完整方案。

共同实现位于 lib/research_261129.py；本分支 PAIR 仅允许这两个模块构造及参与前向。未直接复制第三方源码；发表前必须引用原论文并核查授权。

## 独立消融

环境变量 WTCONV_ENABLED、SFALIGN_ENABLED 均为0/1，默认1。四组实验00、10、01、11，例如：

```bash
WTCONV_ENABLED=0 SFALIGN_ENABLED=0 bash sh/start_train_synapse.sh
WTCONV_ENABLED=1 SFALIGN_ENABLED=1 bash sh/start_train_synapse.sh
```

训练配置记录开关，测试从对应 checkpoint config 恢复。00走原始模型路径，不构造新模块。保留原损失、数据划分、TopN和指标逻辑。PointRend或GSCNN启用时辅助损失权重0.2；其他模块沿用原分割损失。

## 验证要求

```bash
python -m compileall -q lib utils trainer.py train_*.py test_*.py
python -m unittest discover -s tests -p test_research_261129.py
```

测试覆盖4组开关、二分类/4类/9类、前向、反向、有限数值与严格权重重载。本次提交代码和测试，未执行真实项目训练，不代表测试通过。先小批量试跑，再固定划分、至少3随机种子配对实验，报告Dice、HD95、逐类指标及资源消耗。原模块忠实复现也是必要对照，当前不提供所有原模块的直接实现。THOP与新增非卷积算子近似估算分开记录，不等于精确总FLOPs。不能保证指标改善，也没有公认20%学术创新量指标。
