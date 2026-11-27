# Benchmarking native decisions

Measure inference latency separately for text, images, video and audio. Use the same checkpoint,
device, media decoding parameters and question set for every comparison.

```bash
python benchmarks/bench_multimodal.py --help
python scripts/multimodal/evaluate.py --help
```

Use frozen development and test splits. Report decision accuracy by modality and question type,
score MAE for ordinal decisions, and IoU and Acc@IoU≥0.5 for single-box decisions. Compare models
only on the same requests and targets. [Native evaluation](multimodal.md#calibration-and-evaluation)
