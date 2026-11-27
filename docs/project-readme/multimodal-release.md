# Export a trained checkpoint

**English** · [简体中文](../zh-CN/project-readme/multimodal-release.md)

Run inference with the native checkpoint, merge the LoRA adapter, then export to GGUF for
llama.cpp or mobile deployment. Test the result on the same inputs.
[Tutorial home](index.md) · [Native inference](multimodal.md#python-inference)

<a name="native-checkpoint"></a>

## Select the checkpoint

| Artifact | Location | Use |
| --- | --- | --- |
| Local v6 checkpoint | `multimodal-runs/qwen4b-v6-fresh/sft/best` | Selected LoRA adapter and readouts for inference, evaluation or export |
| Calibrated checkpoint | Calibration command's `--out`, typically `final/` | Use the validated calibration map |
| Local optimizer snapshot | Directory named in `sft/checkpoint_latest.json` | Continue the original training run |
| Published native v1 | [Merged safetensors](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b) | Native Python inference |
| Published v1 GGUF | [Language weights and visual projector](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF) | Local llama.cpp-jev |

Read `mode` and `backbone` in `multimodal_config.json`. A `lora` checkpoint combines the
original local backbone with the adapter and heads. A `full` checkpoint carries the merged
weights. For the published v1 checkpoint, download it to a local directory.

```bash
hf download ZeeoRe/jev-multimodal-qwen3.5-4b --local-dir models/jev-multimodal-qwen3.5-4b
```

[Matched development results](multimodal-current.md#matched-development)

## Merge LoRA

Export a selected v6 adapter to a new directory. The exporter merges language weights and
preserves the visual merger, learned heads, processor and calibration.

```bash
checkpoint=multimodal-runs/qwen4b-v6-fresh/sft/best
release=multimodal-runs/qwen4b-v6-export

python scripts/multimodal/export.py \
  --checkpoint "$checkpoint" --out "$release/merged" \
  --base-id Qwen/Qwen3.5-4B --device cpu

(cd "$release/merged" && sha256sum -c SHA256SUMS)
python verify/multimodal_smoke_test.py "$release/merged" --device cpu
```

Read `export_mode`, `status` and `fixtures` in `merge-verification.json`. The output mode is
`full`. `weights_only` indicates that saving and reloading completed. Pass
`--fixtures path/to/fixtures.json` to compare model outputs before and after merging. The
synthetic image from the inference exercise can be part of that fixture.

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

For fixture-based verification, inspect probability, coordinate and quality differences in
`fixtures[].deltas` and confirm `status` is `passed`. Use a labelled suite for quality measurements.
[Evaluate a checkpoint](multimodal.md#evaluate)

## Convert and quantize GGUF

Use a local llama.cpp-jev checkout that supports the Qwen3.5 visual path and decision heads.
Text uses the language GGUF. Image and video inputs also load the matching `mmproj`.

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

Confirm the BF16 language and projector files before quantizing. Inspect
`quantized/quantization-verification.json` for tensor types. The default keeps decision and
box tensors in FP32. `--head-quantization Q8_0` can quantize the three head matrices while
retaining FP32 biases and normalization tensors.

On the target runtime, verify text decisions, image boxes and video decisions. Re-evaluate
accuracy, IoU, probabilities and actual memory with a fixed input budget.
[Run llama.cpp-jev](../multimodal-release.md#run-the-decision-server)

## Published v1 quantization

The release guide maintains the projector pairs, combined sizes, distillation sample counts
and harmonic accuracy. Record the runtime revision, context and actual device memory when
choosing a deployment.
[Quantization table](../multimodal-release.md#select-a-gguf-pair) · [Mobile integration](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/blob/main/mobile/README.md)
