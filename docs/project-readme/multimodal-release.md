# 从训练检查点到部署模型

先用原生检查点跑通推理，再合并 LoRA。需要 llama.cpp 或移动端部署时，从合并权重生成 GGUF，最后在相同输入上复测。 [教程首页](index.md) · [原生推理](multimodal.md#python-inference)

<a name="native-checkpoint"></a>

## 选择检查点和版本

| 对象 | 位置 | 用途 |
| --- | --- | --- |
| v6 本地推理检查点 | `multimodal-runs/qwen4b-v6-fresh/sft/best` | 用选中的 LoRA 和头部评测、推理或导出。 |
| v6 完成后的校准检查点 | 校准命令的 `--out`，通常为 `final/` | 部署使用经过验证的概率校准参数。 |
| v6 本地优化器快照 | `sft/checkpoint_latest.json` 指向的目录 | 恢复原训练 |
| v1 原生发布 | [jev-multimodal-qwen3.5-4b](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b) | 已合并的 Safetensors |
| v1 GGUF 发布 | [jev-multimodal-qwen3.5-4b-GGUF](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF) | 分开的语言权重和视觉投影文件 |

打开原生检查点的 `multimodal_config.json`，先看 `mode` 和 `backbone`。`lora` 模式从 `backbone` 所指位置读取原始权重，再叠加适配器和任务头。`full` 模式带有完整权重，可直接提供给 Python 推理和 GGUF 转换。已有本地 v1 发布目录 `models/releases/jev-multimodal-qwen3.5-4b/native` 时，可直接拿它做推理。首次下载可使用下面的路径，再把该目录作为第二步推理的 `checkpoint`。

```bash
hf download ZeeoRe/jev-multimodal-qwen3.5-4b --local-dir models/jev-multimodal-qwen3.5-4b
```

量化表对应 v1 final。评估 v6 的实际表现请使用本地检查点和同一评测分片。 [已核验的同集结果](multimodal-current.md#matched-development)

## 合并 LoRA

选定一个固定的 v6 LoRA 检查点。导出器读取原始基座，把语言适配器合入骨干，并保存视觉 merger、候选评分头、单框定位头、处理器和校准参数。合并目录使用新的路径。

```bash
checkpoint=multimodal-runs/qwen4b-v6-fresh/sft/best
release=multimodal-runs/qwen4b-v6-export

python scripts/multimodal/export.py \
  --checkpoint "$checkpoint" --out "$release/merged" \
  --base-id Qwen/Qwen3.5-4B --device cpu

(cd "$release/merged" && sha256sum -c SHA256SUMS)
python verify/multimodal_smoke_test.py "$release/merged" --device cpu
```

合并后先读 `merge-verification.json` 的 `export_mode`、`status` 和 `fixtures`。`export_mode` 为 `full`，`weights_only` 表示通过了保存和重载检查。若需要比较合并前后的预测，准备一份合成输入 JSON，在导出命令中加 `--fixtures path/to/fixtures.json`。可以把第二步生成的测试图用于单框夹具。

```json
[
  {
    "id": "synthetic-square",
    "state": {"image": "_local/project-readme-tutorial/scene.png"},
    "questions": {
      "color": {"type": "noul", "instructions": "Is the square red?"},
      "target": {"type": "box", "instructions": "the red square"}
    }
  }
]
```

提供夹具时，检查 `fixtures[].deltas` 中的概率差、坐标差和质量分差。通过时 `status` 为 `passed`。冒烟测试检查输出结构，完整评测则使用同一组带标签输入。 [本地导出说明](https://github.com/2351548518/laya/blob/4f7e045725adf8c7c7067dfc404de1e54f68b05b/scripts/multimodal/README.md#export-checkpoints) · [评测方法](multimodal.md#evaluate)

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

转换后在 `gguf/` 中确认生成 `*-BF16.gguf` 和 `mmproj-BF16.gguf`，再用这组 BF16 输入执行量化。打开 `quantized/quantization-verification.json` 查看每个输出文件的张量类型。默认保留 FP32 决策和定位张量。显式传 `--head-quantization Q8_0` 等参数可以量化三个头部矩阵，偏置和归一化张量保留 FP32。 [量化脚本](https://github.com/2351548518/laya/blob/4f7e045725adf8c7c7067dfc404de1e54f68b05b/scripts/multimodal/quantize_gguf.py)

部署到目标运行时，先用语言 GGUF 做文本请求，再在加载匹配的 `mmproj` 后做图像定位与视频决策。校验答案字段后，复测概率、准确率、单框 IoU 和设备实际占用。部署性能比较需固定模型对、上下文和输入。 [启动 llama.cpp-jev 服务](../multimodal-release.md#run-the-decision-server)

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

读表时先找准备部署的文件大小，再看同一测试集上的 H。比如 Q4_K_M 的 FP32 头版本是 2.902 GiB，WebM 修复后 H 为 77.83%。IQ2_XS / SSM4 是 1.840 GiB，同一轮复测为 75.92%。这两行展示了减少存储时的准确率变化。空白的复测栏表示该配置没有表内的 WebM 修复后结果。

手机接入应同时确认所用模型、视觉投影文件、运行时版本和当前长度预算。Android 和 iPhone 的应用接入示例及修复重跑记录由已发布模型卡维护。 [正式发布说明](../multimodal-release.md) · [手机接入](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/blob/main/mobile/README.md)
