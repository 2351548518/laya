# 模型导出与发布

## 检查点与版本

| 对象 | 位置 | 用途 |
| --- | --- | --- |
| v6 本地推理检查点 | `multimodal-runs/qwen4b-v6-fresh/sft/best` | 用当前选中的 LoRA 和头部评测、推理或导出 |
| v6 本地优化器快照 | `sft/checkpoint_latest.json` 指向的目录 | 恢复原训练 |
| v1 原生发布 | [jev-multimodal-qwen3.5-4b](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b) | 已合并的 Safetensors |
| v1 GGUF 发布 | [jev-multimodal-qwen3.5-4b-GGUF](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF) | 分开的语言权重和视觉投影文件 |

本文的已发布量化结果对应 v1 final。v6 记录的是本地训练协议与检查点。需要判断 v6 当前能力时，参照同一开发集的[检查点对照](multimodal-current.md#matched-development)。

## 合并 LoRA

导出器加载指定 LoRA 检查点和原始基座，将语言适配器合入骨干，并保留视觉 merger、候选评分头、单框定位头、处理器和校准字段。源目录的 `multimodal_config.json` 会记录架构、基座和模式。合并目录应是新的本地路径。

```bash
checkpoint=multimodal-runs/qwen4b-v6-fresh/sft/best
release=multimodal-runs/qwen4b-v6-export

python scripts/multimodal/export.py \
  --checkpoint "$checkpoint" --out "$release/merged" \
  --base-id Qwen/Qwen3.5-4B --device cpu
```

可以用 `--fixtures` 传入本地 JSON 夹具，每项包含 `id`、`state`、`questions`，逐项检查合并前后的选项概率、坐标和质量分。结构和重载检查通过后，输出目录才能作为可部署模型使用。 [本地导出说明](https://github.com/2351548518/laya/blob/4f7e045725adf8c7c7067dfc404de1e54f68b05b/scripts/multimodal/README.md#export-checkpoints)

## 导出 GGUF 并量化

转换使用兼容 JEV 头部和 Qwen3.5 视觉路径的本地 llama.cpp-jev。文本推理可单独加载语言 GGUF。图像和视频需要同时加载与该模型匹配的 `mmproj`。

```bash
llama_cpp=/path/to/llama.cpp-jev

python scripts/multimodal/export_gguf.py \
  --checkpoint "$release/merged" --llama-cpp "$llama_cpp" \
  --out "$release/gguf" --name jev-multimodal-qwen3.5-4b-v6 --quantize none

python scripts/multimodal/quantize_gguf.py \
  --model "$release/gguf/jev-multimodal-qwen3.5-4b-v6-BF16.gguf" \
  --mmproj "$release/gguf/mmproj-BF16.gguf" --llama-cpp "$llama_cpp" \
  --out "$release/quantized" --name jev-multimodal-qwen3.5-4b-v6 \
  --variants Q4_K_M Q4_0 Q3_K_M
```

语言量化以 BF16 导出为输入。默认保留 FP32 决策和定位张量。显式传 `--head-quantization Q8_0` 等参数可以量化三个头部矩阵，偏置和归一化张量保留 FP32。转换程序检查结构与张量类型。部署验证还需在目标运行时复测概率、目标框、准确率和实际占用。 [量化脚本](https://github.com/2351548518/laya/blob/4f7e045725adf8c7c7067dfc404de1e54f68b05b/scripts/multimodal/quantize_gguf.py)

## v1 的已发布量化对照

大小是语言和视觉投影文件之和，设备实际内存还受上下文、KV 缓存、媒体张量和运行时影响。H 是全体决策准确率与单框 Acc@IoU≥0.5 的调和平均。发布选择采用关闭训练期头部 QAT 的蒸馏版本。修复列使用 900 条决策和 128 个正框的 WebM 修复后复测结果。

| 配置 | 文件大小 | 蒸馏记录 | 初版 H | 选中版 H | WebM 修复复测 H |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 4.811 GiB | 10,496 | 76.80% | 78.22% | — |
| Q4_K_M，FP32 头 | 2.902 GiB | 10,496 | 75.57% | 77.53% | 77.83% |
| Q4_K_M，Q8 头 | 2.897 GiB | 10,496 | 75.57% | 77.19% | — |
| Q4_0 | 2.718 GiB | 10,496 | 77.04% | 77.13% | — |
| Q3_K_M | 2.490 GiB | 744 | 74.37% | 75.77% | — |
| IQ3_XXS / SSM8 | 2.150 GiB | 10,496 | 72.11% | 72.85% | — |
| IQ2_XS / SSM4 | 1.840 GiB | 744 | 71.41% | 75.58% | 75.92% |
| Q1_0，体积实验 | 0.624 GiB | 744 | 2.96% | 2.98% | — |

手机接入应同时确认所用模型、视觉投影文件、运行时版本和当前长度预算。Android 和 iPhone 的应用接入示例及修复重跑记录由已发布模型卡维护。 [正式发布说明](../multimodal-release.md) · [手机接入](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/blob/main/mobile/README.md)
