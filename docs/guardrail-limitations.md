# Historical guardrail measurements

**English** · [简体中文](zh-CN/guardrail-limitations.md)

A 53-case synthetic study tested the earlier Laya 0.3.21 and 0.3.22 checkpoints on CPU. It found
misses on document-shaped inputs and false alarms on benign text that quoted attack vocabulary.
The current native checkpoint needs its own evaluation on reviewed deployment examples.
[Original cases and measurements](https://github.com/NandhaKishorM/laya/issues/781)

<a name="what-was-measured"></a>

## Reported results

| Screening rule at threshold 0.5 | Recall | False-positive rate |
| --- | ---: | ---: |
| `jailbreak` or `prompt_injection` | 0.72 | 0.21 |
| `prompt_injection` | 0.60 | 0.18 |
| Full `LayaGuardrail` rule | 0.76 | 0.29 |

In the report, adding an embedded instruction to case c043 produced c042. Both scored 0.000
for `prompt_injection`. Case c038 quoted attack language in a security lesson and scored 0.619.
Retain paired controls like these when building an evaluation.

<a name="what-this-does-and-does-not-say"></a>

## Read scores in their own units

`prompt_injection.noul` is a probability. The four-level `harm_severity.score` is an expected
index on the 0–3 scale. The guard uses the probability mass of the upper half of the rubric
when that distribution is available. A score of 0.82 bounds that mass by 0.41 since each upper
level contributes at least two units to the expected index.

Choose application thresholds from held-out data and review uncertain or high-impact requests
at the point of action. [Probability and confidence](questions-and-answers.md#reading-confidence) · [Staged adoption](staged-adoption.md)
