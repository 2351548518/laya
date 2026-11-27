# Fine-tuning

Use the joint trainer to adapt a local Qwen3.5 or Gemma 4 checkpoint for typed decisions and
image grounding. The same text inputs and `choice`, `noul` and `score` questions used at
inference can be supervised alongside single-box annotations.
[Supported inputs and checkpoints](multimodal.md#backbones-and-input-types)

## Fine-tune with `laya-train`

Install the native training dependencies and run the packaged CLI from the repository root
to use the example configurations.

```bash
pip install -e '.[multimodal]'
laya-train --help
```

`laya train`, `python -m laya.train_cli` and `python scripts/multimodal/train.py` accept the
same arguments. Use `torchrun --module laya.train_cli` for multi-GPU runs.

The data directory contains `train.jsonl`, `dev.jsonl` for box development and
`decision_dev.jsonl` for decision development. An optional `warmup.jsonl` supplies the head
warmup rows. Prepare the inputs with the same processor and context budget as the run.
[Records, media paths and data preparation](multimodal.md#local-jsonl-data-loading)

Start the heads from an original local backbone and use the resulting selected checkpoint
as the input to LoRA SFT. The following examples use a prepared Qwen dataset.

```bash
laya-train --model models/Qwen3.5-4B --data multimodal-data/example \
  --out multimodal-runs/example/head --config configs/multimodal/head.json \
  --train-split warmup --visual-only --require-original-base

laya-train --model multimodal-runs/example/head/best --data multimodal-data/example \
  --out multimodal-runs/example/sft --config configs/multimodal/sft.json --visual-only
```

Use Gemma 4 and omit `--visual-only` for audio or video with a soundtrack. If Qwen rows were
prepared locally without saved processor admission, `--allow-unadmitted` lets the trainer
process them subject to the usual modality and context checks. Use a separate dataset for
final testing. [Training settings and checkpoint selection](multimodal.md#head-warmup-and-lora-training)

## Targets and losses

Each decision question uses `expected` for a hard label or
`gold[question_id].probabilities` for a soft target. For the choice options `continue` and
`inspect`, `"expected": {"action": "inspect"}` produces `[0, 1]` in option order.
A soft label with probabilities `{"continue": 0.2, "inspect": 0.8}` assigns those weights to
the two cross-entropy terms. If both labels are present, `gold` supplies the target.
The same conversion handles Noul answers and ordered Score
levels. Box rows supervise the predicted coordinates and box quality.
[Single-box record example](grounding.md)

## Resume and monitor

With `checkpoint_every` enabled in the JSON configuration, pass `--resume RUN_DIRECTORY`
alongside the original model, data, configuration and GPU count to continue a run.
Install the `swanlab` extra for `--swanlab local` or `--swanlab online`.
[Resume behavior and SwanLab setup](multimodal.md#swanlab-reporting)

For programmatic training, import `GroundingTrainConfig` and `finetune_grounding` from
`laya.train`. [Python API](reference/train.md)
