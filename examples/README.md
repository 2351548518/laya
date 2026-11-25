# laya-multimodal examples

## Native decisions

Run from the repository root with a trained local checkpoint and `.[multimodal]` installed.
[Checkpoint download](../docs/multimodal-release.md)

```bash
python examples/native_multimodal.py models/jev-multimodal-qwen3.5-4b --device cpu
python examples/native_multimodal.py models/jev-multimodal-qwen3.5-4b \
  --image scene.jpg --video clip.mp4
```

The example prints a JSON result for text routing and each supplied file. Use an image with
a red cup for the single-box query and a short clip for the visual motion score. To check a
waveform, select a trained Gemma 4 checkpoint and add `--audio clip.wav`. For video, install
`ffmpeg` and `ffprobe`. [Local Python API](../docs/reference/multimodal.md) ·
[Native notebook and data examples](../notebooks/laya_native_multimodal.ipynb) ·
[Python and TypeScript HTTP clients](../docs/multimodal-clients.md)

## Classic text examples

The scripts in `classic/` use the classic text checkpoints. `classic/_common.py` prefers the
checkpoint directories under `models/` and otherwise uses the classic Hub bundle.
[Classic setup](../LOCAL_SETUP.md)

```bash
python examples/classic/first_decision.py models/laya --device cpu
./examples/run_all.sh
./examples/run_all.sh predict_batch
./examples/run_all.sh '*routing*'
```

**Before you start**

* Each example loads its own checkpoint. Without a local copy, the first call may download
  its weights. Warm-up and load time depend on your hardware.
* `device=None` lets Laya choose the available device. `_common.load(name, device="cpu")` selects CPU.
* `_common.py` holds shared states, question sets and printing helpers. Keep it beside
  examples that import it.
* Where a weak spot is visible in an example's output, the example says so rather than hiding it:
  `score` is the weakest primitive, high-cardinality choice sets degrade, and Latin-script
  language detection is a heuristic.

---

`data/routing_labels.csv` is a 60-row labelled set (five teams, three urgency levels) for
`laya.Agent(vegaml=True).fit(...)`. [Vega walkthrough](../docs/laya-guide.md)

---

### Stage 1 — first contact

| | what you'll learn |
|---|---|
| [first_decision.py](classic/first_decision.py) | the smallest possible program: load a checkpoint, ask one question, read the answer. Deliberately standalone, no shared helpers, so you can copy it into your own project. |
| [inspect_prediction.py](classic/inspect_prediction.py) | what `predict()` actually returns: `answers` per question, the three answer shapes, `usage`, and the `action` payload. |

**You can now** load a checkpoint and read a typed answer.

### Stage 2 — the three decision primitives

| | what you'll learn |
|---|---|
| [choice_questions.py](classic/choice_questions.py) | `choice`: a closed label set with descriptions, returning the winner and the full distribution. |
| [design_choice_criteria.py](classic/design_choice_criteria.py) | the criteria text is part of the prompt, so wording is part of the job: vague, specific and confusable rubrics on the same input. |
| [score_questions.py](classic/score_questions.py) | `score`: an ordinal rubric, returning the expected level on the rubric's own scale. |
| [choose_score_levels.py](classic/choose_score_levels.py) | the same judgement at 3, 5 and 7 levels, and what granularity does to the distribution. |
| [noul_questions.py](classic/noul_questions.py) | `noul`: a calibrated P(true) you threshold yourself. |
| [write_noul_criteria.py](classic/write_noul_criteria.py) | spelling out what true and false mean, and how loaded wording moves the probabilities. |

**You can now** ask any of the three question types and read the answers as probabilities.

### Stage 3 — state shapes

| | what you'll learn |
|---|---|
| [json_input.py](classic/json_input.py) | a dict state, serialised to JSON for the model — no schema lookup, the text is the input. |
| [compare_input_formats.py](classic/compare_input_formats.py) | the same facts as a ticket, an email and a JSON record, and which answers move. |
| [clean_email.py](classic/clean_email.py) | `email_state` / `clean_email_body`: stripping quoted replies, signatures and disclaimers, and what that buys. |
| [conversation_input.py](classic/conversation_input.py) | a list of turns as the state, so a whole exchange can be judged at once. |
| [input_truncation.py](classic/input_truncation.py) | what happens when the state does not fit `max_len`, and how to see it in `usage`. |

**You can now** feed Laya text, JSON, email or a conversation, and know what happens when it is too long.

### Stage 4 — cost, confidence and batching

| | what you'll learn |
|---|---|
| [multiple_questions.py](classic/multiple_questions.py) | many questions, one pass: `input_tokens` is a batch total across the questions in the call. |
| [measure_latency.py](classic/measure_latency.py) | measuring honestly: load time vs inference, warm-up, and medians over several calls. |
| [confidence_gating.py](classic/confidence_gating.py) | the production pattern: gate on `answer_confidence`, the calibrated probability of the reported answer, and escalate on low. |
| [predict_batch.py](classic/predict_batch.py) | compare per-ticket prediction with `predict_batch()` on a queue. |

**You can now** size a workload, and branch on confidence instead of always acting.

### Stage 5 — the three checkpoints and routing

| | what you'll learn |
|---|---|
| [compare_checkpoints.py](classic/compare_checkpoints.py) | what `english`, `multilingual` and `typed-decisions` each are good at, on the same input. |
| [route_without_loading.py](classic/route_without_loading.py) | `route()` alone: a checkpoint decision in microseconds, with the reason string, before any forward pass. |
| [multilingual_routing.py](classic/multilingual_routing.py) | `predict()` routing and answering in one call, in several languages. |
| [local_checkpoints.py](classic/local_checkpoints.py) | use the repository's `models/` directory or load from the Hub, and check local weights against the manifest. |
| [checkpoint_residency.py](classic/checkpoint_residency.py) | what stays resident: `preload`, `max_loaded`, LRU eviction, `attach`, `unload`. |

**You can now** let the router choose, and control what is held in memory.

### Stage 6 — the built-in presets

| | what you'll learn |
|---|---|
| [email_preset.py](classic/email_preset.py) | `email_questions()`: phishing, spam, category and urgency from a raw email. |
| [triage_preset.py](classic/triage_preset.py) | `triage_questions()`: intent, urgency, frustration, refund, churn in one pass. |
| [guardrail_preset.py](classic/guardrail_preset.py) | `guard_questions()`: jailbreak, injection and sensitive-data signals. |
| [moderation_preset.py](classic/moderation_preset.py) | `moderation_questions()`: toxicity, harassment, threat and spam. |
| [model_routing_preset.py](classic/model_routing_preset.py) | `router_questions()`: difficulty, domain and sensitivity, and a small-vs-frontier policy. |

**You can now** skip schema design for the common application shapes.

### Stage 7 — design your own schema, then tune it

| | what you'll learn |
|---|---|
| [custom_questions.py](classic/custom_questions.py) | building a question schema from scratch for your own domain, with the design decisions spelled out. |
| [structured_criteria.py](classic/structured_criteria.py) | criteria as dicts and lists — rendered as compact JSON for the model. |
| [typed_decision_workflow.py](classic/typed_decision_workflow.py) | the fine-tuned checkpoint's four real workflows, and `auto_task_detection`. |
| [high_cardinality_choice.py](classic/high_cardinality_choice.py) | 20 and 77 options: the `head_max_len` option budget, and when to raise it. |
| [temperature_scaling.py](classic/temperature_scaling.py) | the shipped temperature buckets, and what they do to the logits. |

**You can now** write a schema for your own problem and know where its limits are.

### Stage 8 — devices, scale and production shape

| | what you'll learn |
|---|---|
| [device_latency.py](classic/device_latency.py) | CPU vs MPS latency, and that both give the same answers. |
| [question_throughput.py](classic/question_throughput.py) | questions per second, and the trade-off between calls and questions per call. |
| [long_documents.py](classic/long_documents.py) | raising the context window at runtime, and what that costs. |
| [multi_checkpoint_pipeline.py](classic/multi_checkpoint_pipeline.py) | composing route → triage → specialist decision in one pipeline. |
| [handle_errors.py](classic/handle_errors.py) | the real behaviour of bad inputs — including the cases that do *not* raise. |
| [cache_and_monitor.py](classic/cache_and_monitor.py) | an in-memory answer cache, and a confidence monitor over a batch. |
| [triage_service.py](classic/triage_service.py) | the capstone: route, triage, apply policy, emit a structured decision record. |

**You can now** run Laya as a component of a real system.

---

[fit_and_calibrate_vega.py](fit_and_calibrate_vega.py) fits and calibrates a Vega reader from the local CSV.
Run it separately with `--reader /path/to/vega-checkpoint` for a local reader.
The [hook guide](../docs/hooks/examples.md) walks through the runnable files under `hooks/`.
