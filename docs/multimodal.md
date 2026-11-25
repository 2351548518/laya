# Native multimodal decisions

`MultimodalAgent` reads native media and returns typed decisions without generating answer text.
It uses the backbone's image, audio and video processing, a shared candidate scorer, and an
optional compact single-box readout. Every question uses one backbone forward pass.

The published Qwen3.5-4B checkpoint is available as merged safetensors and typed-decision GGUF.
[Models and GGUF deployment](multimodal-release.md) ·
[Native multimodal Notebook](https://github.com/2351548518/laya/blob/main/notebooks/laya_native_multimodal.ipynb)

Install the optional runtime and training dependencies:

```bash
pip install -e '.[multimodal]'
```

## Backbones and input types

| Backbone family | Text | Images | Audio | Video |
| --- | --- | --- | --- | --- |
| Gemma 4 E2B | yes | yes | native waveform | ordered frames, with optional soundtrack |
| Qwen3.5 | yes | yes | unsupported | native frames and timestamps; visual-only |

The local aliases include `gemma4`, `qwen3.5-4b` and `qwen3.5-9b`. A local model directory or
trained checkpoint can be supplied directly. The starter recipes target the tested Qwen3.5-4B
profile; resource requirements depend on the backbone and native input length.

A fresh candidate scorer and box head are untrained. Train them before relying on their outputs.
LoRA checkpoints also require the original backbone weights at the location recorded in the
checkpoint. Training data, model weights and local run outputs are not shipped in the source tree.

## Local prediction

Use the Python runtime below or run the same checkpoint from a terminal.
[Native CLI](cli-mcp.md#native-multimodal-checkpoint)

```python
import laya

agent = laya.MultimodalAgent("multimodal-runs/example/final", device="cuda")
questions = {
    "action": {
        "type": "choice", "instructions": "Choose the next action.",
        "criteria": {"continue": "continue normally", "inspect": "inspect the situation"},
    },
    "review": {"type": "noul", "instructions": "Does this scene need human review?"},
}
result = agent.predict("Current scene", questions, image="scene.jpg")
```

`choice` returns a distribution over the supplied candidates. `noul` returns the probability of
true. `score` returns a distribution and its expected ordered level index. Fractional targets
remain distributions during training. Single-box localization is covered in [grounding](grounding.md).

For an audio-capable backbone, pass an audio path or a waveform with `sampling_rate`. Qwen3.5
rejects audio rather than silently dropping it. Mark visual-only video states with
`"video_audio": false`. Native input lengths exceeding `max_len` raise an error; media and
questions are not silently truncated to fit.

The existing hooks, `predict_batch`, `Router(multimodal=...)` and inline-media HTTP endpoint can
use the same runtime. Identical-media features may be reused within an inference request;
training does not reuse detached features across parameter updates.

## From input to answer

`MultimodalAgent` in `laya/multimodal.py` renders the question and passes it with the input
to the backbone's processor. For `choice`, `noul` and `score`, `candidate_positions` maps
the candidate markers to their token positions after media insertion. `CandidateScorer`
in `laya/multimodal_readout.py` compares each candidate's hidden state with the final question
state. A softmax turns the scores into probabilities. For example, probabilities of 0.2 and
0.8 for `continue` and `inspect` produce `choice="inspect"`. The same two probabilities for
levels 0 and 1 produce `score=0.8`.

For a `box` question, `GroundingHead` in `laya/grounding.py` projects the final valid hidden
state to normalized coordinates and an IoU quality estimate. One box targets one referred
object. The quality estimate describes localization confidence.

## Router and HTTP

Configure the native lane with a trained local checkpoint:

```python
router = laya.Router(multimodal="multimodal-runs/example/final")
result = router.predict("Current scene", questions, image="scene.jpg")
```

For the HTTP server, install `.[multimodal,serve]`, set `LAYA_MULTIMODAL_MODEL` to the same
checkpoint, and start `laya-serve`. `POST /v1/multimodal` accepts `state`, `questions`, exactly
one of `image`, `audio` or `video`, and optional `max_len` and `min_confidence` controls. Encode
file bytes as `{"base64": "..."}` in the selected media field. Remote callers cannot provide
server file paths. Existing authentication, request size limits and admission controls apply.
Video audio follows the configured backbone's capabilities. [HTTP API](http-api.md) ·
[Python and TypeScript media examples](multimodal-clients.md)

## Local JSONL data loading

The core loaders live in `laya.multimodal_train` and `laya.grounding_train`. They read one JSON
object per line and preserve Unicode and soft labels. The decision-only loader resolves relative
media paths against the JSONL directory; joint training uses the recorded paths directly, so use
absolute media paths or run from their intended working directory. A joint training record
contains exactly one question.

```json
{"id":"scene-1","group":"episode-1","state":{"image":"images/scene.jpg","text":"Current scene"},"questions":{"action":{"type":"choice","instructions":"Choose the next action.","criteria":{"continue":"continue normally","inspect":"inspect the situation"}}},"expected":{"action":"inspect"}}
```

Alternatively, replace `expected` with a native probability target:

```json
{"gold":{"action":{"probabilities":{"continue":0.2,"inspect":0.8}}}}
```

Preserve the original source and group identifiers. Use distinct training, validation, calibration
and test groups, including their media identities. Store dataset preparation in `_local/`.
The public dataset provides a separate, source-attributed export. See its card for its contents
and how the media are stored. [Open JEV Omni public data](https://huggingface.co/datasets/ZeeoRe/open-jev-omni-public)

For joint training, each data directory contains `train.jsonl`, `dev.jsonl` for single-box
validation, and `decision_dev.jsonl` for decision validation. A head warmup directory can contain
a smaller training subset with the same protected development files.

Prepared Qwen exports can carry `training_admission` receipts recording the processor, native
token count and context budget. Receipts must match the chosen model. For local JSONL without
stored receipts, the standalone trainer accepts `--allow-unadmitted`; native media capability
and context checks still apply during loading and forward passes. This flag never drops audio
or truncates an input.

## Head warmup and LoRA training

The reusable joint trainer is `laya.grounding_train.finetune_grounding`. It supports `head`,
`lora` and `full` modes. The standalone entry point is `scripts/multimodal/train.py`.

The supplied recipes use a global batch of 64, a microbatch cap of 4 and a padded-token budget
of 16,384. Each rank accumulates its part of the same optimizer window. Changing GPU count does
not multiply the global batch size. Short and long inputs are packed within the native budget.

Start from an original local backbone with fresh heads:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/multimodal/train.py --model models/Qwen3.5-4B \
  --data multimodal-data/example/head --out multimodal-runs/example/head \
  --config configs/multimodal/head.json --visual-only \
  --require-original-base --allow-unadmitted
```

Continue from this run's selected head checkpoint, creating new LoRA adapters and an optimizer:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/multimodal/train.py --model multimodal-runs/example/head/best \
  --data multimodal-data/example/full --out multimodal-runs/example/sft \
  --config configs/multimodal/sft.json --visual-only --allow-unadmitted
```

Omit `--visual-only` for a Gemma 4 audio/video run. The native backbone reads the media directly.
The Qwen recipe trains language LoRA adapters, the visual merger and the decision/localization
readouts. It keeps the visual encoder frozen. Main decision supervision uses cross-entropy;
box targets use L1, GIoU and quality losses. Warmup can additionally use Brier and ordinal RPS.
These recipes use supervised targets. In `laya/grounding_train.py`, a hard decision row such
as `inspect` raises the probability of that option through cross-entropy. A soft target such
as `[0.2, 0.8]` weights the two log probabilities by 0.2 and 0.8. For an image annotated with
one box, L1 penalizes coordinate error, GIoU penalizes poor overlap, and the quality loss trains
the confidence estimate from the predicted box's overlap with the target.

Readout and localization calculations use FP32. LoRA training uses gradient checkpointing,
length bucketing, per-module learning rates and a warmup/cosine schedule. Optional activation
CPU offload can help long contexts. It does not change the input evidence or create labels.

The decision-only entry point remains available as `laya-mm-train`; run `laya-mm-train --help`
for its JSONL, independent selection, calibration and resume options.

## Calibration and evaluation

Fit decision temperatures on calibration data and check them on a separate development set:

```bash
python scripts/multimodal/calibrate.py --model multimodal-runs/example/sft/best \
  --calibration multimodal-data/example/full/calibration.jsonl \
  --validation multimodal-data/example/full/decision_dev.jsonl \
  --out multimodal-runs/example/final
```

The acceptance guard checks overall calibration and per-modality regressions. A rejected fit
keeps the original model probabilities. Calibration never consumes the final test labels.

Evaluate the frozen checkpoint on local typed decisions and single-box targets:

```bash
python scripts/multimodal/evaluate.py --model multimodal-runs/example/final \
  --decisions multimodal-data/example/test.jsonl \
  --grounding multimodal-data/example/box_test.jsonl \
  --out multimodal-runs/example/evaluation.json
```

The report includes decision accuracy, NLL, Brier, ECE and score MAE, with modality/type slices.
Grounding reports mean IoU and accuracy at IoU 0.5. These are local evaluations; converted
question formats or custom source splits are not interchangeable with an official leaderboard.

## Source and local workspace layout

Core runtime and training modules are under `laya/`; portable commands are under
`scripts/multimodal/`, with starter settings under `configs/multimodal/`. Core contracts are
registered in `scripts/test_suites.py`, and the native multimodal CI lane runs on CPU without
model downloads.

The ignored `_local/` directory holds dataset-specific preparation code, private research
configs, comparison experiments, logs and their source/processing records. Local datasets,
model weights and training outputs go in `multimodal-data/`, `models/` and `multimodal-runs/`.
