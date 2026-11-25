"""Assert decoded parity across benchmark logs that include answers.

python benchmarks/compare_compile_defaults.py reference.log reduced.log restart.log
"""
import argparse
import json
import math
from pathlib import Path


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT "):])
    raise ValueError(f"{path} is missing a RESULT record")


def compare(left, right):
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError("answer fields differ")
        return max((compare(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise ValueError("answer lengths differ")
        return max((compare(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, float):
        if not isinstance(right, (int, float)) or not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("expected finite numerical answers")
        return abs(left - right)
    if type(left) is not type(right) or left != right:
        raise ValueError(f"categorical answers differ: {left!r}, {right!r}")
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        reference = read(args.reference)["answers"]
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    for path in args.logs:
        try:
            delta = compare(reference, read(path)["answers"])
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"{path}: {exc}")
        if delta > 1e-5:
            parser.error(f"{path}: numerical answer delta {delta:.9g} exceeds 1e-5")
        print("%s: categorical parity passed; max numerical delta %.9g" % (path, delta))


if __name__ == "__main__":
    main()
