# Benchmarks

| Workload | Results and commands |
| --- | --- |
| Native decisions and single-box localization | [Evaluator and metric definitions](../docs/benchmarks.md#native-multimodal-checkpoints) · [Released checkpoint](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b) |
| Native request latency | [Run the latency benchmark](#native-latency) |
| GGUF size, accuracy and runtime checks | [GGUF release guide](../docs/multimodal-release.md) and [runtime evaluation](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/blob/main/recovery/runtime-fixed-evaluation.json) |
| Classic text checkpoints and CPU performance | [Text benchmark tables](../BENCHMARKS.md) and [reference results](results/cpu/README.md) |

## Native latency

Install `.[multimodal]`, download a trained checkpoint and prepare a JSON list of requests.
Each request contains `id`, `state` and `questions`. Labels are only needed for accuracy
evaluation. Local media paths are relative to the working directory.

```json
[
  {"id":"text","state":"I was billed twice.","questions":{"refund":{"type":"noul","instructions":"Is this about billing?"}}},
  {"id":"image","state":{"image":"scene.jpg"},"questions":{"target":{"type":"box","instructions":"the red cup"}}},
  {"id":"video","state":{"video":"clip.mp4","video_audio":false},"questions":{"motion":{"type":"noul","instructions":"Is there visible motion?"}}}
]
```

Use your own `scene.jpg` and `clip.mp4`, or remove those requests for a text-only run.
With a trained Gemma 4 checkpoint, pass `{"audio":"clip.wav"}` as the state for a waveform
request. [Download and setup](../docs/multimodal-release.md) · [Additional input types](../docs/multimodal.md#backbones-and-input-types)

```bash
python benchmarks/bench_multimodal.py --model models/jev-multimodal-qwen3.5-4b \
  --requests requests.json --device cuda --warmup 2 --repeats 20 \
  --out multimodal-runs/evaluation/latency.json
```

The benchmark loads the checkpoint once and records the first call per case separately from
the timed warm repetitions. Timings include media preprocessing, inference and answer decoding.
Compare runs using the same checkpoint, files, request order, device and thread count. The
report includes the request-file hash and actual frame timestamps and token usage.

## Classic text benchmarks

`bench_cpu.py`, `bench_predict_batch.py`, `bench_mps_autocast.py` and the compile/fast-path
scripts exercise classic text checkpoints. The archived measurements name the checkpoint and
hardware used in each run. [CPU runs](results/cpu/README.md) · [Compile runs](results/compile-defaults/README.md)

Pass `--models models` to `bench_predict_batch.py` to use local checkpoints. Its
`--batch-size` controls both timing and consistency checks. Consistency checks compare the
choice, modal score level and Boolean result for each question. For CUDA TileLang checks, use
`bench_fast.py` or `parity_fast.py` with the fast extra and `--device cuda:N`.

With Matplotlib installed, use
`python benchmarks/plot_results.py --out _local/bench-rtx4070.png` to regenerate the RTX 4070 chart.
