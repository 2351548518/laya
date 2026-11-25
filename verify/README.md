# Local runtime checks

## Native checkpoints

Install `.[multimodal]` and run from the repository root using the downloaded checkpoint.
[Model setup](../docs/multimodal-release.md)

```sh
python verify/multimodal_smoke_test.py models/jev-multimodal-qwen3.5-4b --device cpu
```

The check exercises `choice`, `score` and `noul` on text and uses a generated image for
an image decision and, when the checkpoint has a trained grounding head, a `box` answer.
It validates the answer shape, normalized geometry and token accounting. Pass `--video clip.mp4`
to check visual video or `--audio clip.wav` with a trained Gemma 4 checkpoint. Video decoding
requires `ffmpeg` and `ffprobe`. Use labelled splits for accuracy measurements.
[Native benchmarks](../docs/benchmarks.md#native-multimodal-checkpoints) · [HTTP checks](../docs/multimodal-clients.md)

## Classic text checkpoints

With the three classic checkpoints under `models/`, run these checks from the repository root.

```sh
python verify/checkpoints.py --models ./models
python verify/laya_smoke_test.py --models ./models --device cpu
```

`checkpoints.py` checks the pinned weights against this directory's manifest.
`laya_smoke_test.py` exercises text routing and typed answers. `edge_sweep.py`,
`numerics_check.py` and `soak_check.py` cover boundary inputs, numerical parity and repeated
calls. `bench_devices.py` measures classic CPU and available MPS paths.
[Classic local setup](../LOCAL_SETUP.md)
