# 目标尺度课程监督候选方法：实验设计与可行性报告

日期：2026-09-25  
状态：代码已接入，正式 Dice 效果尚未验证。本文档不预先声称候选方法有效。

## 1. 研究问题与主假设

**研究问题**：在不改变现有 EMCAD 编码器、解码器、四个多尺度输出、mutation 监督、数据划分、优化器、学习率和推理流程的前提下，按训练标签中的前景尺度安排四个输出头的监督强度，能否稳定改善 Synapse 与 ACDC 的患者级总体 Dice？

**主假设（待验证）**：目标前景面积较小时，高分辨率头 P1/P2 的监督相对更重要；面积较大时，粗尺度头 P4/P3 的监督相对更重要；在训练后期把这种偏置线性退火回统一监督，可以减少对单一尺度的过拟合。主结果只把 `target_scale_mode=curriculum,target_scale_factors=area` 视为“目标尺度课程监督候选方法”。

边界复杂度不是主结论的组成部分，而是单独的机制扩展：只有在 `area` 已完成后，才允许比较 `area_boundary`。因此不能把 `area_boundary` 的结果直接归因于面积课程。

## 2. 当前 EMCAD baseline（必须保持不变）

- 网络仍为当前 EMCAD，包括原编码器、解码器和四个全分辨率 logits，顺序为 `[P4, P3, P2, P1]`，从粗到细。
- `supervision=mutation` 仍枚举四头的 15 个非空组合；组合内仍逐元素**求和 logits**，不是求概率或平均概率。
- 每个组合仍使用 `0.3 * CE + 0.7 * Dice`。`target_scale_mode=off` 时训练代码继续走原始损失路径。
- Synapse 与 ACDC 推理仍只取 `P[-1]`，softmax/argmax、空类别处理、Dice/HD95/Jaccard/ASD 实现均未因候选方法改变。
- 现有 Synapse 训练器每个 epoch 使用 `test_vol` 评估并选择 `best.pth`。如果 `test_vol` 是官方测试清单，这不是独立测试选择；旧流程与新流程必须共同承受这一限制，论文中应称为“现有测试清单选择协议”，不能把它包装成严格独立测试证据。

## 3. 候选方法的具体计算

对一个训练切片标签 `Y`：

1. 前景面积比例 `r = #(Y>0)/(H*W)`。
2. 用 3x3 腐蚀得到前景边界，计算 `b = boundary_pixels / foreground_pixels`。空前景样本的边界复杂度定义为 0，并使用全 1 权重。
3. 面积分桶：`small: r < 0.05`，`medium: 0.05 <= r < 0.20`，`large: r >= 0.20`，另保留 `empty`。
4. 四头基础因子（顺序 P4/P3/P2/P1）：

   | 桶 | P4 | P3 | P2 | P1 |
   |---|---:|---:|---:|---:|
   | small | 0.70 | 0.90 | 1.20 | 1.50 |
   | medium | 1.00 | 1.00 | 1.00 | 1.00 |
   | large | 1.50 | 1.20 | 0.90 | 0.70 |

5. `area_boundary` 在上述因子上额外乘以边界调制 `(0.90, 0.95, 1.05, 1.10)` 的复杂度插值；`area` 不使用该调制。
6. `static` 全程使用完整因子；`curriculum` 的强度从第 1 个 epoch 的 1 线性下降到最后一个 epoch 的 0；最后阶段恢复统一监督。
7. 对每个已有 mutation 组合，取组合内头权重的样本级平均，再分别加权原来的 CE 和 Dice。没有增加网络头、改变 logits 求和或增加额外 loss 项。

例如一个 small 样本在单头 P1 组合中的相对因子是 1.50，在单头 P4 组合中是 0.70；组合 `[P4,P1]` 的因子是两者的平均。`curriculum` 第 `e` 个 epoch 的实际权重为 `1 + s(e) * (factor - 1)`，其中 `s(0)=1`、`s(last)=0`。因此训练末期所有头都回到 1，原始 EMCAD 监督恢复。每个样本的四头因子再除以其四头均值，避免把一个样本仅因目标大小整体放大成额外全局 loss 系数。

实现只改变监督权重，不改变输入、标签、增强、优化器、学习率、batch size、epoch、checkpoint 规则或推理。

## 4. 正式实验矩阵

每一行使用同一固定划分、同一预处理、同一训练轮数、同一优化器/学习率、同一 batch size、同一 checkpoint 规则和同一评估实现。每组至少使用相同 seed 配对；正式论文建议先做 seed `2222`，若主结果方向成立，再冻结方案做预先声明的多 seed 复核。

| 组 | `target_scale_mode` | `target_scale_factors` | 只改变的因素 | 允许的归因 |
|---|---|---|---|---|
| B0 baseline | `off` | 任意（off 不使用） | 无 | 当前 EMCAD 基线 |
| M1 面积静态 | `static` | `area` | 是否全程按面积分桶加权 | 只能归因于静态面积加权 |
| M2 **主候选** | `curriculum` | `area` | 把 M1 的全程加权改为线性退火 | 可检验“课程退火”相对 M1 的贡献；M2 相对 B0 是面积课程候选的整体效果 |
| M3 边界扩展 | `curriculum` | `area_boundary` | 在 M2 上增加边界复杂度调制 | 只能归因于边界扩展，不能回溯为面积因素 |
| M4 早期阶段 | `early_only` | `area` | 仅前半程加权，后半程统一 | 课程阶段必要消融 |
| M5 后期阶段 | `late_only` | `area` | 仅后半程加权 | 课程阶段必要消融 |
| M6 统一静态边界 | `static` | `area_boundary` | 面积静态 + 边界调制 | 仅作机制补充，不作为主结论 |

不要把 M2、M3、额外阈值搜索、不同 seed 或不同 checkpoint 混在一次对比中。阈值 `0.05/0.20`、边界参考值 `0.25` 必须在验证阶段冻结；不能用最终测试结果重新选择。

## 5. 必须报告的证据

### 5.1 患者级总体 Dice

- Synapse：每个患者先按 8 个器官得到患者级宏平均 Dice，再在患者维度平均；同时报告各器官 Dice。
- ACDC：每个患者先按 RV/MYO/LV 得到患者级宏平均 Dice，再在患者维度平均；同时报告三类 Dice。
- 主比较使用相同 seed 的患者级配对差值：`Dice_candidate(case) - Dice_baseline(case)`。报告均值、标准差、中位数、正差病例数和 95% bootstrap CI（bootstrap 单位是患者，不是切片）。
- 只有总体 Dice 在预先声明的多个 seed 上方向一致且配对 CI/统计证据支持，才可以写“候选方法带来提升”。单器官、单 seed、训练损失或验证集改善都不能替代这一证据。

### 5.2 small/medium/large 分桶

测试代码会为每个患者输出：`bucket`、`area_ratio`、`boundary_complexity`、各器官 Dice 和 `mean_dice`，并生成 `target_scale_bucket_summary.csv`。必须报告每桶的病例数、总体 Dice、面积比例均值和边界复杂度均值。

注意：训练监督按 2D 切片计算尺度；测试分桶按完整患者体数据的前景比例计算，二者统计单位不同。该表用于机制分层分析，不应宣称它与训练桶一一相同。

### 5.3 课程阶段消融

至少比较 M1、M2、M4、M5：

- 前 25%、50%、75%、100% 训练进度的四头权重均值；
- 每阶段 `curriculum_strength`；
- `P4/P3/P2/P1` 权重均值、最小值、最大值；
- empty/small/medium/large 样本计数；
- small/medium/large 患者级 Dice。

若四头权重长期都接近 1、或某一头长期达到极端边界，应报告“权重塌缩/调制失效”，不能把结果解释为尺度机制证据。日志和 TensorBoard 已记录这些统计。

## 6. 运行参数

### Synapse 训练

在项目根目录运行（正式实验使用新的独立输出目录）：

```bash
python train_synapse.py \
  --root_path /path/to/data/Synapse/train_npz \
  --volume_path /path/to/data/Synapse/test_vol_h5 \
  --list_dir /path/to/data/Synapse/lists/lists_Synapse \
  --dataset Synapse --encoder pvt_v2_b2 --kernel_sizes 1 3 5 \
  --expansion_factor 2 --lgag_ks 3 --activation_mscb relu6 \
  --supervision mutation --img_size 224 --batch_size 16 --max_epochs 300 \
  --base_lr 1e-4 --n_gpu 1 --seed 2222 --deterministic 1 \
  --target_scale_mode curriculum --target_scale_factors area \
  --target_scale_small_threshold 0.05 --target_scale_large_threshold 0.20 \
  --target_scale_boundary_reference 0.25 \
  --output_dir ./model_pth/Synapse/target_scale_runs \
  --run_name synapse_curriculum_area_seed2222
```

`B0` 只把 `--target_scale_mode off`，`M1` 把 mode 改为 `static`，`M3` 把 factors 改为 `area_boundary`，`M4/M5` 把 mode 改为 `early_only/late_only`。其余参数不得变化。

### ACDC 训练

```bash
python train_acdc.py \
  --root_path /path/to/data/ACDC \
  --list_dir /path/to/data/ACDC/lists/lists_ACDC \
  --output_dir ./model_pth/ACDC/target_scale_runs \
  --run_name acdc_curriculum_area_seed2222 \
  --encoder pvt_v2_b2 --kernel_sizes 1 3 5 --expansion_factor 2 \
  --lgag_ks 3 --activation_mscb relu6 --supervision mutation \
  --img_size 224 --batch_size 6 --max_epochs 150 --base_lr 1e-4 \
  --weight_decay 1e-4 --num_workers 4 --n_gpu 1 --seed 2222 \
  --deterministic 1 --target_scale_mode curriculum \
  --target_scale_factors area --target_scale_small_threshold 0.05 \
  --target_scale_large_threshold 0.20 --target_scale_boundary_reference 0.25
```

### 测试

Synapse 直接指定冻结后的 `best.pth`，并指定一个此前不存在的输出目录；ACDC 使用 `test_acdc.py` 的 `--checkpoint/--output_dir/--output_csv`。测试命令不需要 target-scale mode，因为模型推理仍取 `P[-1]`；只需保持分桶阈值与训练配置一致。

```bash
python test_synapse.py --checkpoint /path/to/synapse_run/best.pth \
  --volume_path /path/to/data/Synapse/test_vol_h5 \
  --list_dir /path/to/data/Synapse/lists/lists_Synapse \
  --img_size 224 --seed 2222 \
  --target_scale_small_threshold 0.05 --target_scale_large_threshold 0.20 \
  --output_dir ./eval/Synapse/curriculum_area_seed2222

python test_acdc.py --checkpoint /path/to/acdc_run/best.pth \
  --root_path /path/to/data/ACDC \
  --list_dir /path/to/data/ACDC/lists/lists_ACDC \
  --output_dir ./eval/ACDC/curriculum_area_seed2222 \
  --output_csv ./eval/ACDC/curriculum_area_seed2222/test_metrics.csv \
  --img_size 224 --inference_batch_size 8 --seed 2222 \
  --target_scale_small_threshold 0.05 --target_scale_large_threshold 0.20
```

测试代码拒绝覆盖已有 CSV、日志或非空预测目录；每个候选 run 使用新的 eval 目录。

## 7. `.sh` 手动修改清单（本次没有修改任何 `.sh`）

当前 `sh/start_train_synapse.sh` 仍传入旧的 `--fusion_mode`、`--fusion_loss_weight`、`--reliability_loss_weight`。它们不属于本目标尺度候选方法，且当前训练入口没有这些参数。请手动：

1. 删除这三个旧 fusion 参数及其变量定义；不要把它们与目标尺度监督捆绑。
2. 增加变量：

   ```bash
   TARGET_SCALE_MODE="curriculum"
   TARGET_SCALE_FACTORS="area"
   TARGET_SCALE_SMALL_THRESHOLD="0.05"
   TARGET_SCALE_LARGE_THRESHOLD="0.20"
   TARGET_SCALE_BOUNDARY_REFERENCE="0.25"
   OUTPUT_DIR="${PROJECT_DIR}/model_pth/Synapse/target_scale_runs"
   RUN_NAME="synapse_curriculum_area_seed${SEED}"
   ```

3. 在 `python train_synapse.py` 命令中加入：

   ```bash
   --target_scale_mode "${TARGET_SCALE_MODE}" \
   --target_scale_factors "${TARGET_SCALE_FACTORS}" \
   --target_scale_small_threshold "${TARGET_SCALE_SMALL_THRESHOLD}" \
   --target_scale_large_threshold "${TARGET_SCALE_LARGE_THRESHOLD}" \
   --target_scale_boundary_reference "${TARGET_SCALE_BOUNDARY_REFERENCE}" \
   --output_dir "${OUTPUT_DIR}" \
   --run_name "${RUN_NAME}"
   ```

4. baseline 运行只把 `TARGET_SCALE_MODE="off"`；不要改变其它训练参数。

当前 `sh/start_train_acdc.sh` 的 `set -u` 下直接引用未定义 `${n_gpu}`，请手动在脚本中定义 `n_gpu=1`（或改成 `${n_gpu:-1}`），并加入与 Synapse 相同的五个 target-scale 参数及 `--target_scale_mode`。这只是启动脚本修正，不是方法因素；所有组必须使用相同修正后的脚本协议。

5. 当前 `sh/start_test_synapse.sh` 没有显式 checkpoint，仍依赖旧的目录重建逻辑。请手动增加 `CKPT=/绝对路径/到/本次run/best.pth`、`EVAL_DIR=/绝对路径/到/新的评估目录`，并在 `python test_synapse.py` 命令中加入 `--checkpoint "${CKPT}" --output_dir "${EVAL_DIR}"`。不要让测试脚本根据测试 Dice 自动寻找或替换 checkpoint。

6. 当前 `sh/start_test_acdc.sh` 已有 `CKPT`、`--output_dir` 和 `--output_csv` 参数，但每个候选组仍必须换成新的 `EVAL_DIR/OUTPUT_CSV`；测试代码会拒绝覆盖已有结果。Synapse 和 ACDC 测试均加入 `--target_scale_small_threshold 0.05 --target_scale_large_threshold 0.20`，保证分桶口径固定。

不要把 `nohup`、日志路径、GPU 暴露方式或输出目录命名当作方法变量。`CUDA_VISIBLE_DEVICES` 只决定可见物理 GPU；`n_gpu=1` 只让程序不启用 DataParallel，它们不是候选方法。

## 8. 代码变更清单

### 方法代码

- `utils/target_scale_supervision.py`：GT 面积分桶、边界复杂度、静态/课程/早期/后期权重、加权 CE/Dice 和统计。
- `trainer.py`：Synapse 在 `off` 时保持原始 mutation 损失，在其它 mode 下只替换已有组合的监督权重。
- `utils/acdc_utils.py`：ACDC 公共 `supervised_loss` 增加同样的可开关权重路径；`off` 返回原 baseline 损失。

### 实验协议代码

- `train_synapse.py`：增加 target-scale 参数、独立输出目录和 `config.json`，修复原 `exp_name` tuple 路径错误。
- `train_acdc.py`：增加 target-scale 参数、配置日志和独立实验目录拒绝覆盖。

### 测试代码

- `test_synapse.py`：支持显式 `--checkpoint`/独立 `--output_dir`，输出患者尺度分桶明细和汇总。
- `test_acdc.py`：输出患者尺度分桶列及 `target_scale_bucket_summary.csv`，拒绝覆盖旧结果。

### 文档

- `docs/目标尺度课程监督_实验设计与可行性报告.md`：本实验协议、归因边界、运行参数和手动 `.sh` 修改清单。

## 9. 当前结论

代码路径已接入并通过 Python 语法检查；本地缺少 PyTorch，未执行训练或 GPU smoke test。当前没有 Synapse/ACDC 正式患者级实验结果，因此只能结论为：**目标尺度课程监督是可配置、可独立消融的待验证候选方法，效果尚未验证。**

若 M2 没有在相同 seed 配对的多个正式实验中稳定超过 B0，或 M4/M5 无法支持“训练后期恢复统一监督”的机制，则应报告负结果并停止把该方向表述为 Dice 提升方法。
