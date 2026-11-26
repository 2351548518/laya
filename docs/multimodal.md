# Native multimodal decisions

`MultimodalAgent` turns text and native media into typed answers. The backbone processes the
input, and the candidate scorer or single-box head reads its hidden states. Each question uses
one backbone forward pass and returns zero generated tokens.

The released Qwen3.5-4B checkpoint is available as merged safetensors and typed-decision GGUF.
[Models and GGUF deployment](multimodal-release.md)

Install the optional runtime and training dependencies from the repository root.

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

Ask two questions about the same image with the downloaded checkpoint.

```python
import laya

agent = laya.MultimodalAgent("models/jev-multimodal-qwen3.5-4b", device="cpu")
questions = {
    "action": {
        "type": "choice", "instructions": "Choose the next action.",
        "criteria": {"continue": "continue normally", "inspect": "inspect the situation"},
    },
    "review": {"type": "noul", "instructions": "Does this scene need human review?"},
}
result = agent.predict("Current scene", questions, image="scene.jpg")
print(result["answers"]["action"]["probabilities"])
print(result["answers"]["review"]["noul"])
```

`choice` returns the selected candidate and a distribution over the supplied candidates. `noul`
returns the probability of true. `score` returns a distribution over ordered levels and the
expected level index. A `box` question localizes one referred object in an image.
[Single-box grounding](grounding.md)

For an audio-capable backbone, pass an audio path or a waveform with `sampling_rate`. Use
visual-only video with Qwen3.5. In the state object, set `video_audio` to `False` in Python or
`false` in JSON. Inputs that exceed the `max_len` token budget raise an error after processing.

The existing hooks, `predict_batch`, `Router(multimodal=...)` and inline-media HTTP endpoint can
use the same runtime. All questions about one state share its decoded images, audio and sampled
video frames. Each question still has its own prompt and forward pass. The Qwen3.5 runtime can
also reuse identical visual features across questions about that state during inference.

For runnable file-based examples and a checkpoint smoke test, use the examples and verification
commands from the repository root. [Native example](https://github.com/2351548518/laya/blob/main/examples/README.md) ·
[Checkpoint smoke test](https://github.com/2351548518/laya/blob/main/verify/README.md) ·
[Native CLI](cli-mcp.md#native-multimodal-checkpoint) · [Python API](reference/multimodal.md)

## From input to answer

`MultimodalAgent` in `laya/multimodal.py` passes the rendered question and its input to the
backbone's processor. For `choice`, `noul` and `score`, `candidate_positions` identifies the
candidate marker tokens after the processor inserts media tokens. `CandidateScorer` in
`laya/multimodal_readout.py` projects and compares each marker's hidden state with the final
question state. A softmax turns the resulting logits into probabilities. Probabilities of 0.2
for `continue` and 0.8 for `inspect` yield `choice="inspect"`. The same probabilities over
levels 0 and 1 yield `score=0.8`.

For a `box` question, `GroundingHead` in `laya/grounding.py` projects the final valid hidden
state to one normalized box. Its quality score estimates the intersection over union (IoU)
between the predicted box and the target.

## Router and HTTP

Pass the trained checkpoint to `Router` to enable the native inference path.

```python
router = laya.Router(multimodal="models/jev-multimodal-qwen3.5-4b", device="cpu")
result = router.predict("Current scene", questions, image="scene.jpg")
```

For the HTTP server, install `.[multimodal,serve]`, set `LAYA_MULTIMODAL_MODEL` to the same
checkpoint and start `laya-serve`. Use `POST /v1/systemone` for text and image questions.
The `assets` map places pictures in the state, instructions or criteria. Set
`"model": "multimodal"` for native text decisions. Use `POST /v1/multimodal` for inline audio
and video file bytes. The configured backbone determines which modalities and video audio
the service can process. [HTTP API](http-api.md) · [Media examples](multimodal-clients.md)

Set `batch_size=4` on a native Qwen3.5 Python text request to batch its questions as
independent rows within a padded token budget.

## Local JSONL data loading

`load_rows` accepts local JSONL with a System One `request` and labels stored next to it.
The joint trainer accepts the same envelope with one question per row. Both readers resolve
local media relative to the JSONL file. Existing flat native JSONL and Kev text records with
a `label` on each question also load directly.

Image placeholders in `request.state` reference `request.assets`. The `media_files` map resolves
each named part on disk. An array of image placeholders followed by a text string uses the
same image-first prompt layout as the original native dataset. For HTTP multipart calls,
attach those files as the named parts and send the image header.

```json
{"id":"scene-1","group":"episode-1","request":{"model":"multimodal","state":["scene","Current scene"],"assets":{"scene":{"type":"image","part":"picture"}},"questions":{"action":{"type":"choice","instructions":"Choose the next action.","criteria":{"continue":"continue normally","inspect":"inspect the situation"}}}},"media_files":{"picture":"images/scene.jpg"},"expected":{"action":"inspect"}}
```

Use `expected` for a choice label, a Boolean answer, a zero-based score, or a typed Box target.
Use `gold` when the source supplies a probability distribution, which takes precedence for
that question. Keep a Box target's `confidence` null because localization quality is learned.

```json
{"gold":{"action":{"probabilities":{"continue":0.2,"inspect":0.8}}}}
```

```json
{"expected":{"target":{"type":"box","box":{"left":0.1,"top":0.2,"right":0.5,"bottom":0.6},"confidence":null}}}
```

The converter stores fixed option orders in `option_orders` outside the HTTP request. A
Choice's criteria appear in the same display order on the wire. Noul and Score keep their
official ordering on the wire and recover the original order when loaded for native training.
Audio, video and rubrics exceeding 10 levels use the local format, which preserves the
annotation and media controls. Prepare a text and image benchmark for external HTTP calls
with `prepare_systemone.py`. [Dataset commands](https://github.com/2351548518/laya/tree/main/scripts/multimodal)

Preserve the original source and group identifiers. Use distinct training, validation, calibration
and test groups, including their media identities. Store dataset preparation in `_local/`.
The public dataset provides a separate, source-attributed export. See its card for its contents
and how the media are stored. [Open JEV Omni public data](https://huggingface.co/datasets/ZeeoRe/open-jev-omni-public)

For joint training, each data directory contains `train.jsonl`, `dev.jsonl` for single-box
validation, and `decision_dev.jsonl` for decision validation. Choose `--train-split warmup` to
use `warmup.jsonl` with the same development files.

Prepared Qwen exports can record the processor, native token count and context budget in
`training_admission`. Match the processor and budget to the run configuration. For local
JSONL without these records, the standalone trainer accepts `--allow-unadmitted` and still
checks the input modality and token limit.

## Head warmup and LoRA training

The reusable joint trainer is `laya.grounding_train.finetune_grounding`. It supports `head`,
`lora` and `full` modes. The standalone entry point is `scripts/multimodal/train.py`.

The supplied recipes use a global batch size of 64 records across all GPUs. Each rank packs up
to four records per microbatch within a budget of 16,384 padded tokens. A microbatch contributes
in proportion to its share of the global batch before the ranks sum their gradients.

Start with the original local Qwen backbone. This stage initializes the heads using the
configured seed and trains the readouts while the backbone stays frozen.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/multimodal/train.py --model models/Qwen3.5-4B \
  --data multimodal-data/example/head --out multimodal-runs/example/head \
  --config configs/multimodal/head.json --visual-only \
  --require-original-base --allow-unadmitted
```

Start the LoRA stage from the selected head checkpoint. This run creates new adapters and a
fresh optimizer.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/multimodal/train.py --model multimodal-runs/example/head/best \
  --data multimodal-data/example/full --out multimodal-runs/example/sft \
  --config configs/multimodal/sft.json --visual-only --allow-unadmitted
```

Omit `--visual-only` and select a Gemma 4 backbone to train on audio and video with a
soundtrack. The Qwen LoRA recipe trains the language adapters, visual merger, and decision
and box readouts while keeping the visual encoder frozen.

The starter recipes use supervised losses in `laya/grounding_train.py`.

- A hard label such as `inspect` gives all cross-entropy weight to that option. A soft target
  of `[0.2, 0.8]` weights the log probabilities of `continue` and `inspect` by 0.2 and 0.8.
  The head warmup adds Brier loss and applies ranked probability score (RPS) to ordinal decisions.
- A box annotation supplies four coordinates. L1 penalizes coordinate errors, generalized IoU
  (GIoU) penalizes poor overlap, and the quality loss uses the predicted box's IoU with the target.

Candidate readouts project only the selected hidden states in FP32. The LoRA recipe uses gradient checkpointing, length bucketing,
per-module learning rates, and cosine decay after warmup. Set `activation_offload_min_tokens`
to offload saved activations to CPU for long inputs. Set `balance_ranks` to group compatible
admitted lengths and distribute their work across GPUs within each optimizer window.
Set `prefetch_video` to overlap video decoding on the CPU with decision updates.
Set `fused_adamw` to use PyTorch's fused AdamW optimizer and reduce optimizer-step temporary storage.
For Qwen3.5, `--qwen-training-delta` reduces the temporary gradients in the PyTorch
chunk scan during training. The LoRA recipe selects by the harmonic
mean of the single-box IoU≥0.5 accuracy and decision accuracy averaged across modalities.
Retention checks can reject a checkpoint that exceeds its allowed development-set regression.
Set `video_cache_dir` to `null` to disable disk caching and use `video_cache_size` to cap the
number of decoded clips retained in memory on each rank.
Set `shuffle_options` in the joint configuration to permute Choice options without a fixed
order or references to option positions. The loss aligns logits to the original labels and replay
distributions. Development inference uses one request per decision forward pass by default.
The recipes allow four compatible image requests per grounding evaluation forward pass.
Set `distributed_decision_eval` for single-request decision evaluation on all training ranks.
Each rank keeps requests for the same video clip together, and the primary rank restores the original
record order before computing metrics.
For large JSONL splits, the primary rank validates the records and shares the offset index
with the other ranks.

For recoverable joint training, set `checkpoint_every` to a positive number of optimizer steps.
The trainer publishes a local optimizer snapshot at that interval, at development evaluation
points and at the final optimizer step. It keeps the latest two complete snapshots. Resume with
`--resume RUN_DIRECTORY` and the original model, dataset, optimizer configuration and rank count.
The snapshot restores AdamW, the current epoch order, the per-rank random states, and the
best checkpoint selected at that step. Use the same `--deterministic` setting for the original
training and its continuation when comparing exact updates. Deterministic kernels can cost throughput.

Use `laya-mm-train --help` for the decision-only trainer's JSONL, checkpoint selection,
calibration and resume options.

## Calibration and evaluation

Fit decision temperatures on calibration data and check them on a separate development set.

```bash
python scripts/multimodal/calibrate.py --model multimodal-runs/example/sft/best \
  --calibration multimodal-data/example/full/calibration.jsonl \
  --validation multimodal-data/example/full/decision_dev.jsonl \
  --out multimodal-runs/example/final
```

The acceptance guard checks overall calibration and per-modality regressions. A rejected fit
keeps the original model probabilities. Calibration never consumes the final test labels.
When ordinal channels are enabled, a score uses the channel for its modality and number of
levels. If the channel is missing, the score uses the modality's fitted temperature or the raw
distribution when the modality has no score fit.

Evaluate the frozen checkpoint on local typed decisions and single-box targets. The benchmark
guide gives the commands, metric denominators and latency protocol. Native evaluation supports a
named suite from a checked manifest. It reports accuracy and calibration by modality, question type,
source and task when provided, and includes normalized score MAE and RPS.
Set `--video-cache-size 24` to reuse decoded clips during serial evaluation. Requests sharing the
same frame selection are evaluated together, with their results returned in the dataset's order.
[Native accuracy and latency](benchmarks.md#native-multimodal-checkpoints)

## Source and local workspace layout

Core runtime and training modules live in `laya/`. Find the portable commands in
`scripts/multimodal/` and starter settings in `configs/multimodal/`. The native multimodal CI
lane runs the relevant suites on CPU using small local fixtures.

The ignored `_local/` directory holds dataset-specific preparation code, private research
configs, comparison experiments, logs and their source/processing records. Local datasets,
model weights and training outputs go in `multimodal-data/`, `models/` and `multimodal-runs/`.
