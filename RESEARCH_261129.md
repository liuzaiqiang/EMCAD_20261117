# 261129_FRDqfblxj_UACAbqdxsxw

基线提交：`977a31c9e4f3d1ce9e5274c5f33a6f223a111ef5`（261119_baseModel）。此分支为研究候选，不是已经证明有效或新颖的论文成果。

## 研究机制与来源

- FRD：继承全分辨率细节流思想与本仓库旧FRD（261117_FRD_qfblxjzl_MCC_dcdlkyzx/lib/pair_modules/frd.py），参考 https://github.com/papercodeHua/FRD-Net 。不等同完整FRD-Net。改变：RGB/灰度全分辨率局部高频，扩张深度卷积序列，结合最终解码语义及差异生成门控后才注入残差logits。验证：旧FRD、去掉语义门控、完整方案。
- UACANet: Uncertainty Augmented Context Attention for Polyp Segmentation (ACM MM 2021)，https://github.com/plemeri/UACANet 。保留不确定性上下文聚合；增加按类别软概率归一化原型及不确定性原型，将跨级预测分歧和不确定性用于残差门控。验证：原UACA、归一化类别原型、分歧门控。

共同实现位于 lib/research_261129.py；本分支配置 PAIR 仅允许这两个模块构造及参与前向。未直接复制上述第三方源码；引用原论文并核查授权仍是发表前的责任。

## 独立消融

环境变量 FRD_ENABLED、UACA_ENABLED 均为 0/1，默认1。四组实验 00、10、01、11；例如：

```bash
FRD_ENABLED=0 UACA_ENABLED=0 bash sh/start_train_synapse.sh
FRD_ENABLED=1 UACA_ENABLED=1 bash sh/start_train_synapse.sh
```

训练配置记录开关，测试从对应 checkpoint config 恢复。两开关关闭时走原始模型路径，不构造新模块。保留原损失及数据划分、TopN和指标逻辑。对于训练辅助项，PointRend 或 GSCNN 启用时权重0.2；其他模块沿用原分割损失。

## 验证要求

```bash
python -m compileall -q lib utils trainer.py train_*.py test_*.py
python -m unittest discover -s tests -p test_research_261129.py
```

测试覆盖4组开关、二分类/4类/9类、前向、反向、有限数值与严格权重重载。本次仅提交代码和测试，未执行真实项目训练，不能把测试文件存在说成测试通过。先以小批量试跑，再进行固定划分、至少3随机种子配对实验，报告 Dice、HD95、逐类指标和资源消耗。原模块忠实复现也是必要对照，当前不提供所有原模块的直接实现。THOP与新增非卷积算子近似估算分开记录，不等于精确总FLOPs。不能保证Dice升高、HD95降低，也没有公认20%学术创新量指标。
