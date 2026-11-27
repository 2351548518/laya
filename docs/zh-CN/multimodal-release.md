# 已发布模型与 GGUF 部署

[English](../multimodal-release.md) · **简体中文**

发布的原生检查点来自训练完成的 Qwen3.5-4B v1，使用文本、图像及视频画面训练。
权重已合并 LoRA，并包含训练后的视觉 merger、候选评分头及单框头。通过
`laya_multimodal.MultimodalAgent` 加载。 [原生模型与依赖](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b)

<a name="select-a-gguf-pair"></a>

## 选择 GGUF 文件

使用语言 GGUF 处理文本。处理图片或视频时，配套加载含视觉编码器及训练后 merger 的 `mmproj`。
下表的大小是配对文件合计。

| 量化配置 | 大小 | 蒸馏记录数 | 初始 H | 原运行时的选中版 H | 修复后的选中版 H |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 4.811 GiB | 10,496 | 76.80% | 78.22% | — |
| Q4_K_M，FP32 头 | 2.902 GiB | 10,496 | 75.57% | 77.53% | **77.83%** |
| Q4_K_M，Q8 头 | 2.897 GiB | 10,496 | 75.57% | 77.19% | — |
| Q4_0 | 2.718 GiB | 10,496 | 77.04% | 77.13% | — |
| Q3_K_M | 2.489 GiB | 744 | 74.37% | 75.77% | — |
| IQ3_XXS / SSM8 | 2.150 GiB | 10,496 | 72.11% | 72.85% | — |
| IQ2_XS / SSM4 | 1.840 GiB | 744 | 71.41% | 75.58% | **75.92%** |
| Q1_0，体积实验 | 0.624 GiB | 744 | 2.96% | 2.98% | — |

上述选中版采用教师蒸馏并关闭训练期头部 QAT。H 是决策准确率与单框 IoU ≥ 0.5 命中率
的调和平均。Q4_K_M 与 IQ2_XS／SSM4 分别在选模开发集的 900 条决策和 128 个正例框上复测。
运行时修复让两种配置各恢复 9 条 WebM 输入，已正常返回的 1,019 个答案保持一致，框命中率一致。
表中的破折号表示该配置未复测。
[GGUF 模型卡](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF) · [运行时评测](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/blob/main/recovery/runtime-fixed-evaluation.json)

<a name="run-the-decision-server"></a>

## 启动本地决策服务

选择复测使用的 llama.cpp-jev 提交与运行时补丁。安装 C++ 编译器及 CMake。视频解码还需要
在 PATH 中安装 `ffmpeg` 和 `ffprobe`。

```bash
git clone https://github.com/2351548518/llama.cpp-jev.git
git -C llama.cpp-jev checkout --detach c649d0fcca794985df25370f89b2aebffc41b23c
curl -fL https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/resolve/cb71291936b718f2b9e39fe627f000c52c577655/recovery/runtime-fixes.patch -o runtime-fixes.patch
git -C llama.cpp-jev apply --check ../runtime-fixes.patch
git -C llama.cpp-jev apply ../runtime-fixes.patch
cmake -S llama.cpp-jev -B llama.cpp-jev/build-laya -DCMAKE_BUILD_TYPE=Release -DMTMD_VIDEO=ON
cmake --build llama.cpp-jev/build-laya --target llama-server llama-quantize -j 4
```

以下示例下载 IQ2_XS／SSM4 文件并在环回地址启动。Q4_K_M 可使用
`recovery/jev-multimodal-qwen3.5-4b-Q4_K_M.gguf` 配 `mmproj-Q8_0.gguf`。

```bash
hf download ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF \
  recovery/jev-multimodal-qwen3.5-4b-IQ2_XS-SSM4-heads-Q8_0.gguf \
  mmproj-Q4_0.gguf --local-dir models/jev-compact

llama.cpp-jev/build-laya/bin/llama-server \
  -m models/jev-compact/recovery/jev-multimodal-qwen3.5-4b-IQ2_XS-SSM4-heads-Q8_0.gguf \
  --mmproj models/jev-compact/mmproj-Q4_0.gguf \
  --host 127.0.0.1 --port 8080 -ngl 0 -c 2048 -b 256 -ub 128 -np 1
```

```bash
curl -fsS http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice for the same invoice.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {"billing": "invoices and refunds", "technical": "software issues"}
    }
  }
}'
```

图像和无音轨视频可在 `files` 中传入媒体文件的 data URL。运行时指南给出完整格式，模型卡记录
手机接入与桌面内存实测。 [llama.cpp-jev](https://github.com/2351548518/llama.cpp-jev) · [手机接入](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/blob/main/mobile/README.md)

<a name="export-a-local-checkpoint"></a>

## 导出本地检查点

用 `export.py` 合并原生 LoRA，用 `export_gguf.py` 转换合并的 Qwen 检查点，用
`quantize_gguf.py` 生成较小的量化档位。脚本检查文件结构和校验值。转换后在目标运行时上用固定样本
核验行为。 [导出步骤](project-readme/multimodal-release.md) · [命令目录](https://github.com/2351548518/laya/tree/main/scripts/multimodal)
