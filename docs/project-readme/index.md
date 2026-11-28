# laya-multimodal v6 tutorial

**English** · [简体中文](../../docs_zh/project-readme/index.md)

Follow a request from its input and target to the model's answer and an optimizer update.
This walkthrough uses the v6 Qwen3.5-4B recipe with 1.5 million training rows. It covers text,
images, video and single-box localization. The audio examples use a Gemma checkpoint.

## Start with one example

| Step | Read | Check as you go |
| --- | --- | --- |
| Run both heads on CPU | [CPU walkthrough](multimodal.md#cpu-walkthrough) | Candidate logits `[3, 3]`, boxes `[1, 1, 4]` and reproducible losses |
| Read one input record | [Follow an image choice](multimodal-current.md#input-to-model) | Match a file to its asset reference and a target to the question ID |
| Follow the model diagram | [Architecture and dimensions](multimodal-current.md#architecture) | Trace patch rows, vision tokens, the shared backbone and both readouts |
| Follow the training diagram | [Two stages](multimodal-current.md#training) · [One optimizer update](multimodal-current.md#optimizer-step) | Identify the updated weights and how four GPUs process a 64-row window |
| Inspect a real example | [Read a JSONL row by its index](multimodal-current-io-examples.md#read-original-row) | Compare the request, label and source of the same row |

## Choose a workflow

| Goal | Guide | What to prepare |
| --- | --- | --- |
| Predict in Python and draw a box | [Local inference](multimodal.md#python-inference) | A trained native checkpoint. The tutorial creates a small test image. |
| Send a request from a client | [System One](multimodal.md#systemone-http) · [Audio and video](multimodal.md#audio-and-video) | A local service and a checkpoint for the chosen input |
| Read complete input and target examples | [30 v6 rows and Gemma audio](multimodal-current-io-examples.md) | The page includes the requests and supervision |
| Inspect task coverage and source diversity | [64 task types and 192 excerpts](multimodal-current-examples.md) | Browse by modality and task identifier |
| Reproduce training or continue a run | [Four-GPU training](multimodal.md#reproduce) · [Resume](multimodal.md#resume) · [SwanLab](multimodal.md#swanlab) | The original Qwen backbone, v6 data and free GPUs, or the original run's snapshots |
| Compare checkpoints | [Evaluation and calibration](multimodal.md#evaluate) · [Matched development results](multimodal-current.md#matched-development) | The same rows, input budgets and metric definitions |
| Merge LoRA or export GGUF | [Export and deployment](multimodal-release.md) | A trained checkpoint or the published v1 files |

## Read the tutorial locally

Open this file in your editor's Markdown preview, or run the local docs site from the checkout.
The diagram links open larger SVG or PNG versions. Run commands from the directory containing
`laya_multimodal/` and `pyproject.toml`.

```bash
python -m pip install -r requirements-docs.txt
python scripts/docs_site.py serve --dev-addr 127.0.0.1:8001
```

[Open the tutorial in your browser](http://127.0.0.1:8001/project-readme/)

Restart the preview to load edits from `docs/` and `docs_zh/`.

## Workspace paths

| Path | Contents |
| --- | --- |
| `docs/project-readme/assets/` | Diagrams, sample index, processor traces, training recipes and the CPU exercise |
| `multimodal-data/open-jev-omni-qwen-filtered-v6/native/` | Local training, warmup, development and calibration rows |
| `models/pretrained/Qwen3.5-4B/` | Original backbone for training |
| `multimodal-runs/qwen4b-v6-fresh/` | Run logs, checkpoints and optimizer snapshots |

The download commands use the published v1 weights. The tutorial identifies which tasks use a
local v6 checkpoint. [Checkpoints and releases](multimodal-release.md)
