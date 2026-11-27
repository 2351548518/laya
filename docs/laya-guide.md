# Laya guide

Load a trained native checkpoint and use it for text decisions or local media.

```python
from laya import load

agent = load("models/jev-multimodal-qwen3.5-4b", device="cpu")
questions = {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}
print(agent.predict("I was billed twice. Please refund the duplicate.", questions)["answers"])
```

Use `Agent` or `MultimodalAgent` when constructing the agent directly.
[Native Python usage](multimodal.md#local-prediction)

## Routing

Register multiple trained checkpoints on a `Router` and select one by name.
[Routing guide](routing.md)

## Calibration

Fit native calibration on labelled decisions from a held-out split and validate it before relying
on confidence thresholds. [Calibration and evaluation](multimodal.md#calibration-and-evaluation)

## MCP Server

Run the same model locally from MCP or connect an MCP process to the local HTTP service.
For batch prediction, one tool call takes an array of `{state, questions, model?, task?, lang?, lang_guess?, max_len?, head_max_len?}`.
A shortlist reports the retained option labels, their selection scores, `k`, the original option count `n`,
and whether the full choice was passed through.
[CLI and MCP setup](cli-mcp.md)

## Evaluation harness

Use frozen JSONL with the native evaluator or the generic text decision evaluation harness.
[Evaluation](evals.md)
