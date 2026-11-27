"""Measure local native decisions for a fixed JSON list of state/question requests."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from laya import MultimodalAgent


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_request(agent, case, warmup, repeats):
    def timed():
        synchronize(agent.device)
        start = time.perf_counter()
        result = agent.predict(case["state"], case["questions"])
        synchronize(agent.device)
        return (time.perf_counter() - start) * 1000, result

    first_ms, result = timed()
    for _ in range(warmup):
        timed()
    samples = [timed()[0] for _ in range(repeats)]
    return {
        "id": case["id"], "modality": result["modality"], "usage": result["usage"],
        "media": result["media"], "first_ms": first_ms, "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[math.ceil(0.95 * repeats) - 1], "samples_ms": samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local native checkpoint directory")
    parser.add_argument("--requests", type=Path, required=True, help="JSON list of id, state and questions objects")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1, help="extra calls after timing the first call")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", type=Path, help="save the JSON report")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup must be nonnegative")
    if not (args.model / "multimodal_config.json").is_file():
        parser.error("model must be a directory containing multimodal_config.json")
    if args.out:
        output_path = args.out.resolve()
        model_path = args.model.resolve()
        if output_path in (args.requests.resolve(), model_path) or model_path in output_path.parents:
            parser.error("write the report outside the model directory and use a separate request file")
    try:
        request_bytes = args.requests.read_bytes()
        cases = json.loads(request_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read request JSON: {exc}")
    if not isinstance(cases, list) or not cases or any(
        not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip()
        or "state" not in case or not isinstance(case.get("questions"), dict) or not case["questions"]
        for case in cases
    ):
        parser.error("requests must be a nonempty JSON list with id, state and nonempty questions in each case")
    if len({case["id"] for case in cases}) != len(cases):
        parser.error("request IDs must be unique")

    start = time.perf_counter()
    agent = MultimodalAgent(args.model, device=args.device)
    synchronize(agent.device)
    load_ms = (time.perf_counter() - start) * 1000
    report = {
        "model": str(args.model.resolve()), "backbone": agent.model_type,
        "device": str(agent.device), "dtype": str(agent.dtype), "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(), "max_len": agent.max_len, "default_video_frames": agent.frames,
        "requests_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "load_ms": load_ms, "warmup": args.warmup, "repeats": args.repeats,
        "cases": [benchmark_request(agent, case, args.warmup, args.repeats) for case in cases],
    }
    output = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
