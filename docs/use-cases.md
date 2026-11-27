# Decision workflows

**English** · [简体中文](zh-CN/use-cases.md)

Use a typed decision to choose from a set of actions the application already knows how to perform.
Keep action execution and its input validation at the application boundary.

## Route a request

Choose a queue using a text decision. The option keys are application identifiers. Define each
criterion so a reviewer can assign the same labels when preparing an evaluation set.

```python
from laya_multimodal import MultimodalAgent

agent = MultimodalAgent("models/jev-multimodal-qwen3.5-4b", device="cpu")
questions = {"queue": {
    "type": "choice",
    "instructions": "Which team should handle this request?",
    "criteria": {"billing": "charges or refunds", "support": "product help"},
}}
answer = agent.predict("I was billed twice.", questions)["answers"]["queue"]
print(answer["choice"], answer["probabilities"])
```

For many requests with the same questions, call `predict_batch` and use the returned order.
[Candidate types and confidence](questions-and-answers.md) · [Batch HTTP](http-api.md)

## Inspect a scene or a clip

Ask a `noul` question about a photo to route it for review. Ask a `box` question in the same
request to find a referred object. Convert the normalized coordinates to pixel positions before
drawing a box. For a visual clip, use an explicit `video_audio: false` with Qwen. For a question
about a sound, load a trained audio-capable Gemma checkpoint. [Input formats](multimodal.md#local-prediction) · [Single-box example](grounding.md#prediction-contract)

## Choose a tool or stop

Supply the available actions as `choice` options, such as `search`, `respond` and `review`.
Decode this result to a tool or handler you registered explicitly. A `noul` question can also
represent a stop condition, such as whether the collected evidence satisfies the task.
[Tool selection](tools.md) · [LangGraph](langchain.md) · [CrewAI](crewai.md) · [Agno](agno.md)

## Route retrieval and escalation

For a local search service, make the options the indexes or query engines you can actually call.
Treat multilingual routing as a test slice and compare results on the same question schema
and reviewed data. Fit confidence thresholds for the accepted traffic.
[Checkpoint routing](routing.md) · [LlamaIndex](llamaindex.md) · [Text evaluation](evals.md)

## Observe a workflow before enabling actions

Record `run_id`, the selected checkpoint, the measured probability and the resulting action
when running a trial. Keep human review and fallbacks at the execution boundary.
[Prediction hooks](hooks/index.md) · [Staged adoption](staged-adoption.md)
