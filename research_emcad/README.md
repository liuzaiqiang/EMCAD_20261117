# EMCAD 类别条件尺度融合

本目录包含类别条件尺度融合与[互补跳连融合](complementary_skip_fusion.md)、消融启动器与测试。两个模块默认关闭；均关闭时保留原始 EMCAD 四头输出及参数结构。启动命令见 [ablation_commands.md](ablation_commands.md)。

按原项目 `requirements.txt` 准备 PyTorch/CUDA、数据路径与 PVT 预训练权重。研究启动器经 `sh/run_seed_queue.sh` 顺序运行手动指定的 seed，输出保存到 `research_emcad/runs/<配置>/...`。`OUTPUT_DIR` 可覆盖输出根目录。

运行 `python -m unittest discover research_emcad/tests -v` 检查模型路径。测试入口从训练配置恢复两个模块开关；直接运行测试脚本时也可传入 `--use_class_conditional_scale_fusion` 和 `--use_complementary_skip_fusion`。基础模型均使用默认值 0。
