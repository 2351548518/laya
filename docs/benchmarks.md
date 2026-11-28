# Benchmark native decisions

**English** · [简体中文](../docs_zh/benchmarks.md)

Use a frozen labelled split that was held out from training, checkpoint selection and calibration.
Compare models on identical rows, media budgets and targets. Record the checkpoint revision,
suite checksum, frame sampling and context limit with each result.

## Accuracy and calibration

Run the native evaluator with the manifest generated during data preparation.

```bash
model=models/jev-multimodal-qwen3.5-4b
data=multimodal-data/example/native

python scripts/multimodal/evaluate.py --model "$model" \
  --suite-manifest "$data/manifest.json" --suite test \
  --device cuda:0 --out multimodal-runs/eval/test.json

python benchmarks/bench_multimodal.py --help
```

| Output | Read in the report |
| --- | --- |
| `choice` and `noul` | Accuracy and probability metrics such as NLL, Brier and ECE |
| `score` | Accuracy, level error (MAE), normalized MAE and RPS |
| Single positive `box` | Mean IoU and share with IoU ≥ 0.5 |
| Both decisions and positive boxes | Harmonic mean of overall decision accuracy and box Acc@IoU≥0.5 |

Read per-modality, task and source slices where the report includes their labels. Check sample
counts alongside each percentage. For model selection on the development set, record the
configured score and its denominator. Report final test scores separately. Use the text
harness to compare probabilities from a published decision API on the same text cases.
[Text evaluator](evals.md) · [v6 matched development comparison](project-readme/multimodal-current.md#matched-development)

## Latency

Measure text, image, video and audio separately on hardware capable of the selected inputs.
Report device, precision, checkpoint, input length, cold first prediction, median and P95.
Record request counts and whether the decoder caches repeated clips. Use fixed media and
questions when comparing runtimes.
[Latency commands](https://github.com/2351548518/laya/blob/main/benchmarks/README.md) · [Published comparisons](https://github.com/2351548518/laya/blob/main/BENCHMARKS.md)
