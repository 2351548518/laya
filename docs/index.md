# laya-multimodal documentation

**English** · [简体中文](zh-CN/index.md)

laya-multimodal runs local System One decisions on text and supported image, video and audio
inputs. Each question returns a typed `choice`, `noul`, `score` or `box` answer with zero
generated tokens. Qwen3.5 handles text and vision. Use an audio-capable Gemma 4 checkpoint for
waveform input.

## Start in the repository root

Install the project and download the published Qwen checkpoint. Run a text decision first, then
use the same Python agent with an image or a visual video. [English README](https://github.com/2351548518/laya#installation) · [Native inference](multimodal.md)

## Read by task

| Goal | Guides |
| --- | --- |
| Design a question and interpret probabilities | [Questions and answers](questions-and-answers.md) · [Structured outputs](structured.md) |
| Localize a referred object in one image | [Single-box grounding](grounding.md) |
| Select a checkpoint or run a terminal command | [Routing](routing.md) · [CLI and MCP](cli-mcp.md) |
| Reproduce v6 or train on your own JSONL | [v6 tutorial](project-readme/index.md) · [Fine-tuning](finetune.md) · [Dataset format](multimodal.md#local-jsonl-data-loading) |
| Compare accuracy, calibration and latency | [Native benchmarks](benchmarks.md) · [Text evaluation](evals.md) |
| Serve to Python, TypeScript or another local process | [System One HTTP](http-api.md) · [Media clients](multimodal-clients.md) |
| Deploy with Docker or run a smaller quantized model | [Docker](docker.md) · [GGUF deployment](multimodal-release.md) · [Checkpoint integrity](security.md) |
| Integrate into an application | [Decision workflows](use-cases.md) · [Prediction hooks](hooks/index.md) · [Staged adoption](staged-adoption.md) |
| Look up a Python call | [Python API](reference/index.md) |

The v6 tutorial includes diagrams with tensor shapes, exact training recipes and original sample
indexes. Its recorded measurements identify the data splits and checkpoints they used.
[Architecture and objectives](project-readme/multimodal-current.md) · [Examples by modality](project-readme/multimodal-current-io-examples.md)
