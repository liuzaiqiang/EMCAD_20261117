# 261129_FRDqfblxj_DySampledtscy

基线提交：`977a31c9e4f3d1ce9e5274c5f33a6f223a111ef5`（261119_baseModel）。研究候选，未证明有效或新颖。

## 研究机制与来源

- FRD：继承全分辨率细节流思想与本仓库旧FRD（261117_FRD_qfblxjzl_MCC_dcdlkyzx/lib/pair_modules/frd.py），参考 https://github.com/papercodeHua/FRD-Net 。不等同完整FRD-Net。改变：RGB/灰度全分辨率局部高频、扩张深度卷积、解码语义及差异门控后注入残差logits。验证：旧FRD、去掉语义门控、完整方案。
- DySample：Learning to Upsample by Learning to Sample (ICCV 2023)，https://github.com/tiny-smart/dysample 。保留分组动态坐标采样；新增语义和局部高频两个偏移预测路径，基于高频强度及可学习门控混合，约束位移幅值。验证：原始DySample、单路偏移、双路门控。

共同实现位于 lib/research_261129.py；本分支 PAIR 仅允许这两个模块构造及参与前向。未直接复制第三方源码；发表前必须引用原论文并核查授权。

## 独立消融

环境变量 FRD_ENABLED、DYSAMPLE_ENABLED 均为0/1，默认1。四组实验00、10、01、11，例如：

```bash
FRD_ENABLED=0 DYSAMPLE_ENABLED=0 bash sh/start_train_synapse.sh
FRD_ENABLED=1 DYSAMPLE_ENABLED=1 bash sh/start_train_synapse.sh
```

训练配置记录开关，测试从对应 checkpoint config 恢复。00走原始模型路径，不构造新模块。保留原损失、数据划分、TopN和指标逻辑。PointRend或GSCNN启用时辅助损失权重0.2；其他模块沿用原分割损失。

## 验证要求

```bash
python -m compileall -q lib utils trainer.py train_*.py test_*.py
python -m unittest discover -s tests -p test_research_261129.py
```

测试覆盖4组开关、二分类/4类/9类、前向、反向、有限数值与严格权重重载。本次提交代码和测试，未执行真实项目训练，不代表测试通过。先小批量试跑，再固定划分、至少3随机种子配对实验，报告Dice、HD95、逐类指标及资源消耗。原模块忠实复现也是必要对照，当前不提供所有原模块的直接实现。THOP与新增非卷积算子近似估算分开记录，不等于精确总FLOPs。不能保证指标改善，也没有公认20%学术创新量指标。
