# 261129_FreqFusionplgz_PointRendkndxh

基线提交：`977a31c9e4f3d1ce9e5274c5f33a6f223a111ef5`（261119_baseModel）。研究候选，未证明有效或新颖。

## 研究机制与来源

- FreqFusion：Frequency-aware Feature Fusion for Dense Image Prediction (TPAMI)，https://github.com/Linwei-Chen/FreqFusion 。保留频率分离及重采样思想；使用自适应低通、高通残差、位置偏移与跨路语义一致性门控，在EUCB后同尺度细化。并非官方完整CARAFE实现。验证：普通融合、低高频分解、语义门控与位移。
- PointRend: Image Segmentation as Rendering (CVPR 2020)，https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend 。保留点选择和点预测头；新增不确定性、相邻预测分歧、预测小前景类别面积联合选点；两级细化预测残差，训练随机探索及点级辅助损失0.2。验证：原始不确定性选择、加分歧、加面积、残差细化。推理不读取GT。

共同实现位于 lib/research_261129.py；本分支 PAIR 仅允许这两个模块构造及参与前向。未直接复制第三方源码；发表前必须引用原论文并核查授权。

## 独立消融

环境变量 FREQFUSION_ENABLED、POINTREND_ENABLED 均为0/1，默认1。四组实验00、10、01、11，例如：

```bash
FREQFUSION_ENABLED=0 POINTREND_ENABLED=0 bash sh/start_train_synapse.sh
FREQFUSION_ENABLED=1 POINTREND_ENABLED=1 bash sh/start_train_synapse.sh
```

训练配置记录开关，测试从对应 checkpoint config 恢复。00走原始模型路径，不构造新模块。保留原损失、数据划分、TopN和指标逻辑。PointRend或GSCNN启用时辅助损失权重0.2；其他模块沿用原分割损失。

## 验证要求

```bash
python -m compileall -q lib utils trainer.py train_*.py test_*.py
python -m unittest discover -s tests -p test_research_261129.py
```

测试覆盖4组开关、二分类/4类/9类、前向、反向、有限数值与严格权重重载。本次提交代码和测试，未执行真实项目训练，不代表测试通过。先小批量试跑，再固定划分、至少3随机种子配对实验，报告Dice、HD95、逐类指标及资源消耗。原模块忠实复现也是必要对照，当前不提供所有原模块的直接实现。THOP与新增非卷积算子近似估算分开记录，不等于精确总FLOPs。不能保证指标改善，也没有公认20%学术创新量指标。
