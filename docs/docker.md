# Deploy native decisions with Docker

**English** · [简体中文](../docs_zh/docker.md)

The same image runs one-shot decisions and serves the System One HTTP API. It includes the
Qwen3.5 and Gemma 4 runtime, FFmpeg and the media dependencies.

## Serve a native multimodal checkpoint

Use a merged checkpoint containing `multimodal_config.json`, the saved processor and the
model weights. Run from the repository root.

```bash
export LAYA_MULTIMODAL_MODEL_DIR="$PWD/models/jev-multimodal-qwen3.5-4b"
docker compose -f compose.yaml -f compose.http.yaml -f compose.multimodal.yaml \
  up --build laya-multimodal-serve
```

The overlay mounts the checkpoint read-only and enables offline loading. To use a Hub
checkpoint, omit `compose.multimodal.yaml` and set `LAYA_MULTIMODAL_MODEL` to its repository ID.
With an empty value, the service uses the published Qwen3.5-4B checkpoint. The named
`model-cache` volume preserves Hub downloads between container replacements.
[Published checkpoints](multimodal-release.md)

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/docker/request.json
```

`/health` reports service liveness. Send a decision request to verify model inference.
Set `LAYA_PRELOAD=1` to load the checkpoint at startup. The host binds to loopback by default.
Use `LAYA_PORT` to change the port and `LAYA_BIND_ADDRESS` to change the published address.
Set `LAYA_API_KEY` for bearer authentication, or mount a readable secret file and set
`LAYA_API_KEY_FILE` to its container path. The container runs as UID 10001.
[HTTP configuration](http-api.md) · [Deployment security](security.md)

Qwen3.5 handles text, images and visual video. Select a trained audio-capable Gemma 4
checkpoint for waveform audio. [Media requests](multimodal-clients.md)

## Platforms

| Host | Overlay | Default wheel index |
| --- | --- | --- |
| CPU on Linux AMD64 or ARM64 | Base files above | `cpu` |
| NVIDIA GPU | Add `-f compose.cuda.yaml` before `up` | `cu126` |
| DGX Spark | Add `-f compose.spark.yaml` before `up` | `cu130` |

For NVIDIA, install a compatible driver and the NVIDIA Container Toolkit on the host. Choose
the physical device using `LAYA_GPU_ID`. The CUDA 12.6 overlay fits the 570 driver series.
The CUDA 13.0 overlay requires a driver from the 580 series or newer.
[NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html) · [PyTorch CUDA 12.6 wheels](https://download.pytorch.org/whl/cu126/torch/)

Use the host setup for Apple silicon with PyTorch MPS.
[Local setup](https://github.com/2351548518/laya/blob/main/LOCAL_SETUP.md)

## Run one local request

```bash
docker compose -f compose.yaml -f compose.http.yaml -f compose.multimodal.yaml \
  run --build --rm laya-multimodal
```

For your own request file, set `LAYA_INPUT_DIR` to the directory containing it and add
`-f compose.example.yml`. The container reads `/inputs/request.json`. A request uses
`state` and `questions` just like the Python API. Put any media beside the JSON and refer
to the container path, for example `"state": {"text": "Localize the red cup", "image": "/inputs/cup.png"}`.
[Native inputs](multimodal.md#local-prediction)
