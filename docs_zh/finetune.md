# 联合微调

[English](../docs/finetune.md) · **简体中文**

从本地原始 Qwen3.5 或 Gemma 4 基座开始，让模型学习类型化决策和每道图像题的单框定位。
联合配方分两阶段。预热先冻结骨干，训练两个头。LoRA 微调从选中的预热检查点出发，创建
新的优化器并加上低秩适配器。 [数据格式](multimodal.md#local-jsonl-data-loading) · [v6 维度与目标](project-readme/multimodal-current.md#training)

<a name="run-the-joint-trainer"></a>

## 准备数据并运行两个阶段

安装 `.[multimodal]`，在仓库根目录执行。准备 `train.jsonl`、`warmup.jsonl`、
用于单框选模的 `dev.jsonl` 和用于决策选模的 `decision_dev.jsonl`。按来源分组和媒体身份
隔离校准集及最终测试集。数据目录要匹配训练使用的处理器与 `max_len`。

```bash
base=models/pretrained/Qwen3.5-4B
data=multimodal-data/example/native
run=multimodal-runs/example

laya-multimodal train --model "$base" --data "$data" --train-split warmup \
  --out "$run/head" --config configs/head.json \
  --visual-only --require-original-base --allow-unadmitted

warmup_checkpoint=$(python -c 'import json, sys; print(json.load(sys.stdin)["checkpoint"])' \
  < "$run/head/training_report.json")
laya-multimodal train --model "$warmup_checkpoint" --data "$data" \
  --out "$run/sft" --config configs/sft.json --visual-only --allow-unadmitted
```

示例允许使用未保存处理器准入记录的本地 JSONL。如果已有准入记录，省略 `--allow-unadmitted`。
使用支持音频的 Gemma 4 时，省略 `--visual-only`，并提供对应的音频或视频记录。通过
`CUDA_VISIBLE_DEVICES` 选择显卡。四卡训练可把两条命令中的 `laya-multimodal train`
替换成 `torchrun --standalone --nproc_per_node=4 --module laya_multimodal.train`，
也可按 v6 的启动示例执行。 [v6 两阶段示例](project-readme/multimodal.md#reproduce)

使用候选读出头和 v2 提示格式的合并检查点内含基座配置，可作为新一轮训练的起点。仅保存
决策头或 LoRA 适配器的检查点还需要原始基座。

<a name="targets-and-losses"></a>

## 训练目标与损失

候选顺序为 `continue, inspect` 时，`expected` 硬标签 `"inspect"` 对应 `[0, 1]`。
`gold[question_id].probabilities` 中的 `{"continue": 0.2, "inspect": 0.8}` 按候选顺序转换为
`[0.2, 0.8]`，优先于 `expected`，为两项交叉熵提供权重。预热配方还加入 Brier 损失，并用
排序概率分数（RPS）监督有序评分。单框定位使用坐标 L1 和广义 IoU（GIoU），并用预测框与标注框
的 IoU 监督质量分数。
[损失计算示例](project-readme/multimodal-current.md#worked-losses) · [单框标签](grounding.md#training-records)

<a name="batch-and-memory-settings"></a>

## 批次与显存设置

| 设置 | 作用 |
| --- | --- |
| `global_batch_size` | 所有 rank 每次优化更新的总样本数，入门配方为 64。 |
| `micro_batch`、`max_micro_tokens` | 限制每个微批的兼容样本数和补齐 token 数，入门配置为 4 和 16,384。 |
| `max_len` | 单条处理后请求的上下文上限。 |
| `length_bucketing`、`balance_ranks` | 将长度相近的请求分组，平衡每个 rank 的已准入负载。 |
| `video_cache_dir`、`video_cache_size`、`prefetch_video` | 控制磁盘缓存、每个 rank 的内存缓存和 CPU 解码预取。 |
| `activation_offload_min_tokens` | 达到 token 门槛的微批将反向传播所需的激活保存到 CPU。 |
| `fused_adamw`、`--qwen-training-delta` | 支持时减少优化器或 Qwen 梯度的临时内存。 |
| `shuffle_options` | 重排允许换序的选择题，将目标对齐原候选键。 |
| `distributed_decision_eval` | 将决策评测分给训练 rank，汇总指标前恢复原顺序。 |

Qwen LoRA 配方冻结视觉编码器，更新语言适配器、视觉 merger 和两个读出头。可在配方中配置
梯度检查点、分组学习率以及预热后的余弦衰减。

<a name="resume-and-monitor"></a>

## 续训与监控

从 `training_report.json` 读取选中的检查点路径，用于导出或最终评测。LoRA 配方结合开发集
的决策准确率、单框 IoU ≥ 0.5 命中率及保留阈值，用调和平均选模。报告区分选中的检查点和最后
保存的权重。

设置 `checkpoint_every` 可保存用于恢复的优化器快照。续训使用 `--resume RUN_DIRECTORY`，
并保持原有模型、数据、配方、显卡数和确定性设置。快照恢复对应步数的优化器、抽样顺序和 rank
随机状态。 [续训教程](project-readme/multimodal.md#resume)

安装 `.[multimodal,swanlab]` 后，用 `--swanlab local` 在 `OUT/swanlog` 记录日志，或在
登录后用 `--swanlab online --swanlab-project PROJECT` 在线记录。rank 0 上报优化器步数、
损失、梯度范数、学习率和开发指标。 [SwanLab](project-readme/multimodal.md#swanlab)

Python 调用使用 `laya_multimodal.train` 中的 `GroundingTrainConfig` 和 `finetune_grounding`。
命令行和 Python API 使用同一训练器。使用保存的检查点进行独立评测和校准。 [训练 API](reference/train.md)
