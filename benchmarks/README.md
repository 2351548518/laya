# Native inference benchmark

Use the same checkpoint, device, prompts and media for comparisons.

`requests.json` is a list of objects with an `id`, a `state`, and a `questions` map.
For example:

```json
[
  {
    "id": "billing",
    "state": "I was charged twice for the same invoice.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {"billing": "Invoices and refunds", "support": "Product help"}
      }
    }
  }
]
```

Write the latency report, then render it with matplotlib installed.

```bash
python benchmarks/bench_multimodal.py --model models/jev-multimodal-qwen3.5-4b \
  --requests requests.json --device cpu --out _local/native-latency.json
python benchmarks/plot_results.py _local/native-latency.json --out _local/native-latency.png
```

The report records checkpoint load time and each request's first call, timed samples,
median and P95. Warmup calls are excluded from the sample distribution.
[Evaluation and model comparison](../BENCHMARKS.md)
