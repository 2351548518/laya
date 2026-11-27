# Many-option choice: one question, a wider budget, the shortlist, and a tournament

Compare approaches to large text choice sets on the same native checkpoint. Each request supplies
one utterance and a menu of candidate intents.

| Dataset | Labels | Test rows | Development rows |
|---|---:|---:|---|
| BANKING77 (`mteb/banking77`) | 77 | 3,076 | 600 sampled from train |
| CLINC150 (`clinc/clinc_oos`, `plus`, in-scope only) | 150 | 4,500 | 600 sampled from validation |
| MASSIVE intent, en-US (`SetFit/amazon_massive_intent_en-US`) | 60 | 2,974 | 600 sampled from validation |

- `single` submits the whole label set in one request.
- `wide` sets the request's prompt limit to at least 8,192 tokens.
- `shortlist` ranks options with mean-pooled native token embeddings and submits up to 20.
- `tournament` compares groups of 16 before the final question. Use `tournament-<n>` to change the group size.

Set `CHECKPOINT` to a [local native checkpoint](../../../README.md#quickstart) and install
`pyarrow` for datasets stored as Parquet. The runner reads the dataset revisions pinned in
`benchmark.py` and records the checkpoint and device. The first three requests warm up the model.

```bash
mkdir -p _local
python research/benchmarks/tournament/benchmark.py --dataset banking77 --split test \
  --model "$CHECKPOINT" --device cpu --limit 300 --out _local/banking77-native.json
```

Choose strategies and thresholds on a development split before evaluating the test split. Omit
`--limit` for the full split.

## Recorded results

Apple M4 Pro, MPS, Python 3.12, torch 2.14, Laya 0.3.27, English checkpoint. `results.json` holds
every number below.

### Test splits: one question against the tournament

| dataset | labels | rows | one question | tournament | rows gained / lost | ECE, one question → tournament |
|---|---:|---:|---:|---:|---:|---|
| BANKING77 | 77 | 3,076 | 0.430 | **0.610** | 779 / 227 | 0.350 → 0.082 |
| CLINC150 | 150 | 4,500 | 0.625 | **0.876** | 1,302 / 172 | 0.297 → 0.057 |
| MASSIVE intent | 60 | 2,974 | 0.515 | **0.569** | 486 / 326 | 0.370 → 0.136 |

The final call reads only the group winners, so its probabilities are over a few labels and its
confidence is calibrated much closer to its accuracy than the one-question answer over every label.

### The other remedies, on 500 test rows per dataset

Accuracy, with ECE in brackets, every strategy on the same rows:

| dataset | one question | `wide` | `shortlist` | tournament |
|---|---:|---:|---:|---:|
| BANKING77 | 0.416 (0.355) | 0.554 (0.308) | 0.224 (0.455) | **0.584** (0.111) |
| CLINC150 | 0.604 (0.320) | 0.680 (0.266) | 0.188 (0.625) | **0.894** (0.054) |
| MASSIVE intent | 0.542 (0.349) | **0.622** (0.270) | 0.184 (0.636) | 0.590 (0.125) |

Raising the budget is the better choice when the labels fit uncut in the wider head: MASSIVE's 60
short labels do at `head_max_len=512`. It falls behind as the label set outgrows the wider head, and
on CLINC150's 150 labels it is still cut to 4 tokens per label. The shortlist lost to one question on
all three sets with `embed_fn_from_agent`; its docstring already says a dedicated bi-encoder will
usually shortlist better, and these numbers say the same about the built-in one.

### Latency

300 test rows per dataset (seed 1), one benchmark process on the GPU at a time:

| dataset | one question, p50 / p95 | tournament, p50 / p95 | calls |
|---|---:|---:|---:|
| BANKING77 | 69 / 82 ms | 145 / 174 ms | 2 |
| CLINC150 | 103 / 109 ms | 187 / 205 ms | 2 |
| MASSIVE intent | 56 / 61 ms | 100 / 110 ms | 2 |

The test and sample accuracy runs above had up to two benchmark processes sharing the GPU, so their
recorded times are left out of `results.json`.

### Group size, development rows only

| group size | BANKING77 | CLINC150 | MASSIVE intent |
|---|---:|---:|---:|
| one question | 0.370 | 0.632 | 0.543 |
| 8 | 0.570 | | |
| 12 | 0.593 | 0.832 | 0.588 |
| **16** | 0.583 | 0.855 | 0.580 |
| 24 | 0.582 | 0.863 | 0.563 |
| 32 | 0.610 | | 0.568 |

Sizes 12 to 24 are within about a point of one another on average, so the default is 16: the largest
size that settles up to 256 labels in a single round, and below the roughly 20 options at which the
README says option text starts to be cut, so longer labels and descriptions keep their tokens too.
Size 8 needs a second round for 77 labels and scored lower. Advancing the top two of each group of 16
instead of one, measured with a local variant of this script, scored 0.602 on BANKING77 and 0.587
on MASSIVE, within noise of one winner, so the library advances one. Size 8 ran on BANKING77 only, and
size 32 did not finish on CLINC150.
