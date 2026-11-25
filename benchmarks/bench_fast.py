"""Compare the classic stock and TileLang paths on a fixed request workload.

    python benchmarks/bench_fast.py --model models/laya --device cuda:0
    python benchmarks/bench_fast.py --eval 1000

The optional accuracy run uses the AG News and Emotion test splits via datasets.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from benchmarks.parity_fast import batch, forward, load_cuda_agent
from laya.common import ece_score

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages",
                                "sales": "pricing", "shipping": "delivery"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
}
SHORT = {"subject": "Duplicate charge on invoice 4411",
         "body": "We were billed twice for March. Please refund the duplicate or we're moving to a competitor."}
LONG = {"subject": "Outage report", "body": (
    "Since yesterday our whole team cannot log in, the dashboard returns 502 errors and our release is blocked. "
) * 40}


def measure_ms(fn, device, iterations):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / iterations


def latency(agent, iterations):
    many = {f"{key}{i}": question for i in range(10) for key, question in QUESTIONS.items()}
    cases = [("short x1", SHORT, {"department": QUESTIONS["department"]}),
             ("short x3", SHORT, QUESTIONS), ("short x30", SHORT, many),
             ("long x3", LONG, QUESTIONS), ("long x30", LONG, many)]
    rows = []
    print("case        tokens   stock ms    fast ms   stock e2e    fast e2e")
    for name, state, questions in cases:
        inputs, _ = batch(agent, state, questions)
        agent.deaccelerate()
        fp32, _ = forward(agent, inputs, amp=False)
        stock, _ = forward(agent, inputs)
        stock_ms = measure_ms(lambda: forward(agent, inputs), agent.device, iterations)
        stock_e2e = measure_ms(lambda: agent.predict(state, questions), agent.device, iterations)

        agent.accelerate(strict=True)
        fast, _ = forward(agent, inputs)
        fast_ms = measure_ms(lambda: forward(agent, inputs), agent.device, iterations)
        fast_e2e = measure_ms(lambda: agent.predict(state, questions), agent.device, iterations)

        reference_p, stock_p, fast_p = [torch.softmax(logits.float(), dim=-1) for logits in (fp32, stock, fast)]
        row = dict(
            case=name, L=inputs["input_ids"].shape[1], stock_ms=stock_ms, fast_ms=fast_ms,
            dp_stock=(stock_p - reference_p).abs().max().item(),
            dp_fast=(fast_p - reference_p).abs().max().item(),
            dp_stock_fast=(stock_p - fast_p).abs().max().item(),
            agree=(fast.argmax(-1) == fp32.argmax(-1)).float().mean().item(),
            e2e_stock_ms=stock_e2e, e2e_fast_ms=fast_e2e,
        )
        rows.append(row)
        print(f"{name:11s} {row['L']:6d} {stock_ms:10.2f} {fast_ms:10.2f} {stock_e2e:11.2f} {fast_e2e:11.2f}")
        print(f"  max probability deltas vs fp32: stock {row['dp_stock']:.4f}, fast {row['dp_fast']:.4f}")
    return rows


def accuracy(agent, sample_count):
    from datasets import load_dataset

    datasets = {
        "ag_news": ("fancyzhx/ag_news",
                    {"world": "international news, politics, conflicts", "sports": "sports, games, athletes",
                     "business": "companies, markets, economy", "sci/tech": "science, technology, software, space"}),
        "emotion": ("dair-ai/emotion",
                    dict.fromkeys(("sadness", "joy", "love", "anger", "fear", "surprise"))),
    }
    result = {}
    for name, (repo, criteria) in datasets.items():
        split = load_dataset(repo, split="test")
        if sample_count > len(split):
            raise ValueError(f"{name} has only {len(split)} test rows")
        split = split.shuffle(seed=0).select(range(sample_count))
        labels = list(criteria)
        questions = {"label": {"type": "choice", "instructions": f"Which category does this {name.replace('_', ' ')} text belong to?",
                               "criteria": criteria}}
        scores, predictions = {}, {}
        for mode in ("stock", "fast"):
            if mode == "stock":
                agent.deaccelerate()
            else:
                agent.accelerate(strict=True)
            choices, confidence, correct = [], [], []
            torch.cuda.synchronize(agent.device)
            start = time.perf_counter()
            for example in split:
                answer = agent.predict({"text": example["text"]}, questions)["answers"]["label"]
                choices.append(answer["choice"])
                confidence.append(max(answer["probabilities"].values()))
                correct.append(labels.index(answer["choice"]) == example["label"])
            torch.cuda.synchronize(agent.device)
            predictions[mode] = choices
            scores[mode] = dict(acc=float(np.mean(correct)), ece=ece_score(np.array(confidence), np.array(correct, dtype=float)),
                                ms=(time.perf_counter() - start) * 1000 / len(split))
        agree = float(np.mean([a == b for a, b in zip(predictions["stock"], predictions["fast"], strict=True)]))
        result[name] = dict(n=len(split), **scores, agreement=agree)
        print(f"{name}: stock accuracy {scores['stock']['acc']:.3f}, fast {scores['fast']['acc']:.3f}, agreement {agree:.3f}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--subfolder")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval", type=int, default=0, help="samples per test split (0 = skip)")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), help="stock and TileLang autocast dtype")
    args = parser.parse_args()
    if args.iters < 1 or args.eval < 0:
        parser.error("iters must be positive and eval must be nonnegative")

    agent = load_cuda_agent(parser, args)
    report = dict(model=args.model, subfolder=args.subfolder, gpu=torch.cuda.get_device_name(agent.device),
                  torch=torch.__version__, dtype=str(agent.dtype), latency=latency(agent, args.iters))
    if args.eval:
        try:
            report["eval"] = accuracy(agent, args.eval)
        except (ImportError, ValueError) as exc:
            parser.error(str(exc))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=1)
            stream.write("\n")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    main()
