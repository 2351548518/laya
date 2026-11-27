# Research

Use the native always-local decision engine for reproducible evaluations. For the v6
training and benchmark protocol, see the [native benchmark guide](../BENCHMARKS.md) and
[project tutorial](../docs/project-readme/multimodal.md#evaluate).

| Purpose | Entry point |
| --- | --- |
| Frozen text and multimodal JSONL | [Evaluation harness](../docs/evals.md) |
| MASSIVE language slices and option presentation checks | [Research evaluators](eval/README.md) |
| Chinese short command decisions | [Frozen command suite](benchmarks/zh_short_commands/README.md) |
| Chinese workplace decisions | [Feishu-style fixture](benchmarks/feishu_zh/README.md) |
| Spanish phone action decisions | [Spanish phone suite](benchmarks/es_phone_turns/README.md) |
| Large-choice elimination | [Tournament benchmark](benchmarks/tournament/README.md) |
| Model memory across idle and reload | [Idle unload benchmark](benchmarks/idle_unload/README.md) |

Pass the same native checkpoint, fixture and configuration when comparing alternatives.
Save new run outputs under `_local/`. The JSON reports in `research/results/` and the benchmark
result directories record historical experiments and their original checkpoints.
