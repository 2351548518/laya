# Questions and answers

**English** · [简体中文](zh-CN/questions-and-answers.md)

A request has a state and questions keyed by ID. Each question defines the output shape. The
native agent reads a question's own prompt and returns the answer under the same ID. Without
question batching it runs one backbone forward pass per question.

```python
from laya_multimodal import MultimodalAgent

agent = MultimodalAgent("models/jev-multimodal-qwen3.5-4b", device="cpu")
questions = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle the request?",
        "criteria": {"billing": "invoices or refunds", "support": "product help"},
    },
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "urgency": {"type": "score", "instructions": "How urgent is the request?",
                "criteria": ["routine", "soon", "immediate"]},
}
result = agent.predict("I was billed twice. Please refund the duplicate charge.", questions)
print(result["answers"], result["usage"])
```

## The three candidate-based types

| Question | Define | Read |
| --- | --- | --- |
| `choice` | An ordered map of candidate keys to descriptions | `choice` is the top candidate. `probabilities` uses the original keys. |
| `noul` | A condition to test | `noul` is the probability of true. |
| `score` | An ordered `criteria` list, from low to high | `score` is the expected level index. `probabilities` and `legend` use zero-based level indexes. |

Use short, distinct option descriptions. A `null` choice description renders the label alone.
Native Python supports relabelling `noul` with a `labels` mapping containing `false` and `true`.
The latter still supplies the answer probability. Compare wording changes on reviewed examples.

A `score` with probabilities `[0.1, 0.2, 0.7]` has expected value `1.6`. To read the most
likely level, take the argmax of the distribution and look it up in `legend`.

## Locate an object with Box

A `box` uses an image and a referring phrase in `instructions`, such as `the red cup`.
The output contains normalized `left`, `top`, `right` and `bottom` coordinates and a quality
score. [Single-box request and response](grounding.md#prediction-contract)

## Probability precision

Native `choice` and `score` probabilities are rounded individually to four decimal places. Three
equal candidates can return `0.3333` each, summing to `0.9999`. Use a tolerance proportional to
the option count or normalize a copy if a downstream calculation requires unit mass.

## Reading confidence

| Field | Native local meaning |
| --- | --- |
| `answer_confidence` | Largest candidate probability. For `noul` this is `max(p_true, 1 - p_true)`. |
| `confidence` for `choice` and `score` | Concentration of the full distribution, `1 - H(p) / log(k)` for `k ≥ 2`. |
| `confidence` for `noul` | The same top-side probability as `answer_confidence`. |
| `confidence` for `box` | Estimated IoU of the predicted box. |

For decisions, use `min_confidence` to flag answers below a threshold fitted on held-out data
for the same checkpoint, modality and option count. Measure calibration and accuracy at the
coverage your application plans to accept. The strict HTTP response uses the System One field
set and confidence formulas. [Calibration](multimodal.md#calibration-and-evaluation) · [HTTP response](http-api.md#response)

## Presets

`triage_questions`, `email_questions`, `guard_questions`, `moderation_questions` and
`router_questions` return question maps. Read the supplied instructions and criteria, then
validate the preset on examples from your workflow.

## Reading the options back

`render_options` accepts the short-key representation. The order of `crit` is the display
order in the prompt. A change to that order or to a description changes the input to the model.

```python
from laya_multimodal import render_options

render_options({"t": "choice", "crit": {"billing": None, "support": "product help"}})
render_options({"t": "score", "crit": ["routine", "soon", "immediate"]})
```

Use a shortlist or a tournament when an application needs to choose among many candidates.
Evaluate the retained-candidate recall alongside final decision accuracy.
[Large choices](reference/helpers.md#large-candidate-sets) · [Benchmarks](benchmarks.md)
