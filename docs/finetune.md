# Joint fine-tuning

**English** · [简体中文](zh-CN/finetune.md)

Adapt an original local Qwen3.5 or Gemma 4 backbone for typed decisions and one box per image
question. The joint recipe has two stages. Head warmup freezes the backbone. LoRA fine-tuning
starts from the selected warmup checkpoint, creates an optimizer and adds low-rank adapters.
[Data format](multimodal.md#local-jsonl-data-loading) · [v6 shapes and objectives](project-readme/multimodal-current.md#training)

<a name="run-the-joint-trainer"></a>

## Prepare data and run both stages

Install `.[multimodal]` and run from the checkout root. Prepare `train.jsonl`, `warmup.jsonl`,
`dev.jsonl` for box selection and `decision_dev.jsonl` for decision selection. Keep calibration
and final test samples separate by group and media identity. The prepared directory must match
the processor and `max_len` used for training.

```bash
base=models/pretrained/Qwen3.5-4B
data=multimodal-data/example/native
run=multimodal-runs/example

laya-multimodal train --model "$base" --data "$data" --train-split warmup \
  --out "$run/head" --config configs/multimodal/head.json \
  --visual-only --require-original-base --allow-unadmitted

warmup_checkpoint=$(python -c 'import json, sys; print(json.load(sys.stdin)["checkpoint"])' \
  < "$run/head/training_report.json")
laya-multimodal train --model "$warmup_checkpoint" --data "$data" \
  --out "$run/sft" --config configs/multimodal/sft.json --visual-only --allow-unadmitted
```

This example allows ordinary local JSONL without saved processor admission. With processor
receipts, omit `--allow-unadmitted`. For an audio-capable Gemma 4 backbone, omit `--visual-only`
and use the appropriate audio or video records. Select GPUs through `CUDA_VISIBLE_DEVICES`. For
four processes, replace `laya-multimodal train` in each command with
`torchrun --standalone --nproc_per_node=4 --module laya_multimodal.train_cli` or follow the
recorded v6 launch example. [v6 two-stage example](project-readme/multimodal.md#reproduce)

## Targets and losses

A hard `expected` label `"inspect"` becomes `[0, 1]` for the candidate order
`continue, inspect`. A soft `gold[question_id].probabilities` target such as `[0.2, 0.8]`
weights both cross-entropy terms and takes precedence over `expected`. In the head recipe, Brier
loss helps shape probabilities, and ranked probability score (RPS) supervises ordinal
questions. For a box, use coordinate L1, generalized IoU (GIoU) and the quality loss, which
uses the model's observed IoU against the target.
[Worked losses](project-readme/multimodal-current.md#worked-losses) · [Box target](grounding.md#training-records)

## Batch and memory settings

| Setting | Effect |
| --- | --- |
| `global_batch_size` | Records per optimizer step across all ranks. The starter value is 64. |
| `micro_batch`, `max_micro_tokens` | Limit compatible samples and padded tokens per microbatch. Starter values are 4 and 16,384. |
| `max_len` | Context limit for a single processed request. |
| `length_bucketing`, `balance_ranks` | Group similar lengths and spread admitted work across ranks. |
| `video_cache_dir`, `video_cache_size`, `prefetch_video` | Control disk cache, in-memory clips per rank and CPU decode overlap. |
| `activation_offload_min_tokens` | Offload saved activations to CPU for sufficiently large training microbatches. |
| `fused_adamw`, `--qwen-training-delta` | Reduce temporary optimizer or Qwen gradient allocations when supported. |
| `shuffle_options` | Permute order-safe choices and realign targets to the original keys. |
| `distributed_decision_eval` | Shard decision evaluation across the training ranks and restore the original order before reporting. |

For Qwen LoRA, the visual encoder stays frozen. The default recipe trains language adapters,
the visual merger and both readouts. Gradient checkpointing, group learning rates and cosine
decay after warmup are available through the recipe.

## Resume and monitor

Use the selected path in `training_report.json` for export or final evaluation. The LoRA
recipe selects on development decision accuracy and single-box IoU ≥ 0.5 using their harmonic
mean and retention limits. A report distinguishes the selected checkpoint from the last saved
weights.

Enable `checkpoint_every` to save recoverable optimizer snapshots. Resume with
`--resume RUN_DIRECTORY` and the original model, data, configuration, GPU count and deterministic
settings. The snapshot restores the optimizer, sample order and rank RNG state at the saved step.
[Resume tutorial](project-readme/multimodal.md#resume)

Install `.[multimodal,swanlab]` and set `--swanlab local` for runs under `OUT/swanlog`,
or authenticate and use `--swanlab online --swanlab-project PROJECT`. Rank zero records optimizer
steps, losses, gradient norms, learning rates and later development metrics.
[SwanLab](project-readme/multimodal.md#swanlab)

For Python calls use `GroundingTrainConfig` and `finetune_grounding` in `laya_multimodal.train`.
The optional `laya-multimodal-train-decisions` command runs decision-only distillation and
reference-KL experiments. [Training API](reference/train.md)
