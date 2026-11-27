# Native benchmarks

Use `benchmarks/bench_multimodal.py` to measure local latency on a trained checkpoint.
Use `benchmarks/plot_results.py` to plot the resulting per-request timings.
Use `scripts/multimodal/evaluate.py` to score decision accuracy and single-box geometry.
[Latency benchmark commands](benchmarks/README.md) · [Quality benchmarks](docs/benchmarks.md)

For checkpoint selection, compare the same examples and input budgets. The public Qwen
quantization comparison reports both decision and grounding accuracy, plus their harmonic
mean. [Qwen quantized releases](docs/multimodal-release.md)

The historical encoder results are available at their original revision.
[Classic benchmark archive](https://github.com/2351548518/laya/blob/21167295a16ec64de8b68999e17c37fb9a5e8706/BENCHMARKS.md)
