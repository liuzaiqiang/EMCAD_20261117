# 261129_DySampledtscy_PointRendkndxh

基线提交：`977a31c9e4f3d1ce9e5274c5f33a6f223a111ef5`（261119_baseModel）。此分支为研究候选，不是已经证明有效或新颖的论文成果。

## 研究机制与来源

- DySample：Learning to Upsample by Learning to Sample (ICCV 2023)，https://github.com/tiny-smart/dysample 。保留分组动态坐标采样；新增语义和局部高频两个偏移预测路径，基于高频强度及可学习门控混合，约束位移幅值。验证：原始DySample、单路偏移、双路门控。
- PointRend: Image Segmentation as Rendering (CVPR 2020)，https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend 。保留点选择和点预测头；新增不确定性、相邻预测分歧、预测小前景类别面积联合选点；两级细化预测残差，训练随机探索及点级辅助损失0.2。验证：原始不确定性选择、加分歧、加面积、残差细化。推理不读取GT。

共同实现位于 lib/research_261129.py；本分支配置 PAIR 仅允许这两个模块构造及参与前向。未直接复制上述第三方源码；引用原论文并核查授权仍是发表前的责任。

## 独立消融

环境变量 DYSAMPLE_ENABLED、POINTREND_ENABLED 均为 0/1，默认1。四组实验 00、10、01、11；例如：

```bash
DYSAMPLE_ENABLED=0 POINTREND_ENABLED=0 bash sh/start_train_synapse.sh
DYSAMPLE_ENABLED=1 POINTREND_ENABLED=1 bash sh/start_train_synapse.sh
```

训练配置记录开关，测试从对应 checkpoint config 恢复。两开关关闭时走原始模型路径，不构造新模块。保留原损失及数据划分、TopN和指标逻辑。对于训练辅助项，PointRend 或 GSCNN 启用时权重0.2；其他模块沿用原分割损失。

## 验证要求

```bash
python -m compileall -q lib utils trainer.py train_*.py test_*.py
python -m unittest discover -s tests -p test_research_261129.py
```

测试覆盖4组开关、二分类/4类/9类、前向、反向、有限数值与严格权重重载。本次仅提交代码和测试，未执行真实项目训练，不能把测试文件存在说成测试通过。先以小批量试跑，再进行固定划分、至少3随机种子配对实验，报告 Dice、HD95、逐类指标和资源消耗。原模块忠实复现也是必要对照，当前不提供所有原模块的直接实现。THOP与新增非卷积算子近似估算分开记录，不等于精确总FLOPs。不能保证Dice升高、HD95降低，也没有公认20%学术创新量指标。
