# Benchmark index

| Workload | Results and commands |
| --- | --- |
| Native Qwen3.5 decisions and single-box localization | [Native model card](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b) and the evaluator below |
| GGUF size, accuracy and runtime checks | [GGUF release guide](../docs/multimodal-release.md) and [runtime evaluation](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/blob/main/recovery/runtime-fixed-evaluation.json) |
| Classic text checkpoints and CPU performance | [Text benchmark tables](../BENCHMARKS.md) and [reference results](results/cpu/README.md) |

Run native evaluation from the repository root with held-out input and target JSONL files.

```bash
python scripts/multimodal/evaluate.py \
  --model models/jev-multimodal-qwen3.5-4b \
  --decisions multimodal-data/held-out/decisions.jsonl \
  --grounding multimodal-data/held-out/boxes.jsonl \
  --device cuda --out multimodal-runs/evaluation/report.json
```

Use the same split, questions, frame sampling and context when comparing checkpoints. For GGUF,
record the runtime revision and the language and vision files. The classic tables characterize
their named text checkpoints on the hardware recorded in each run.
[Native JSONL format](../docs/multimodal.md#local-jsonl-data-loading) · [Single-box format](../docs/grounding.md#training-records)
