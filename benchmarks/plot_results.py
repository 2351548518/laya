#!/usr/bin/env python3
"""Plot per-request latency from a native bench_multimodal.py report."""

import argparse
import json
import math
from pathlib import Path


TIMINGS = ("first_ms", "median_ms", "p95_ms")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON produced by bench_multimodal.py")
    parser.add_argument("--out", type=Path, required=True, help="figure path, such as latency.png or latency.svg")
    args = parser.parse_args(argv)
    if args.out.resolve() == args.report.resolve():
        parser.error("write the figure to a different file from the JSON report")
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read the benchmark report: {exc}")

    cases = report.get("cases") if isinstance(report, dict) else None
    if not isinstance(cases, list) or not cases or any(
        not isinstance(case, dict) or any(not isinstance(case.get(key), str) for key in ("id", "modality"))
        or any(type(case.get(key)) not in (int, float) or not math.isfinite(case[key]) or case[key] < 0
               for key in TIMINGS)
        for case in cases
    ):
        parser.error("expected per-request modality, ID and latency values from bench_multimodal.py")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error("plotting requires matplotlib")

    cases = sorted(cases, key=lambda case: case["modality"])
    rows = range(len(cases))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 2 + .32 * len(cases))))
    ax.hlines(rows, [case["median_ms"] for case in cases], [case["p95_ms"] for case in cases],
              colors="#748da8", linewidth=1.5)
    ax.scatter([case["first_ms"] for case in cases], rows, marker="x", color="#a54228", label="First call")
    ax.scatter([case["p95_ms"] for case in cases], rows, marker="|", color="#426588", label="P95")
    ax.scatter([case["median_ms"] for case in cases], rows, marker="o", color="#426588", label="Median")
    ax.set_yticks(rows, [f'{case["modality"]} · {case["id"]}' for case in cases])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_title(f'Native decision latency · {report.get("device", "local device")}')
    ax.set_xlabel("Latency (ms)")
    ax.grid(axis="x", alpha=.25)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.12), ncol=3, frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
