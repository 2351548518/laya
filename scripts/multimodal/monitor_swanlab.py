"""Backfill legacy joint training logs into SwanLab, with optional live tailing."""

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laya._swanlab import development_metrics, init_dashboard, progress_metrics


_TRAIN = re.compile(r"grounding step=(\d+) loss=(\S+) elapsed=(\S+)s ranks=(\d+)")
_DEV = "GROUNDING DEV "
_RESUME = "Resumed joint training at optimizer step "


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def log_lines(path, follow, report):
    """Replay complete lines and wait for appends while the training stage is active."""
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        while True:
            position = stream.tell()
            line = stream.readline()
            if line.endswith(b"\n") or (line and not follow):
                yield line.decode("utf-8", errors="replace").strip()
                continue
            if not follow or read_json(report).get("status") == "complete":
                return
            current = path.stat()
            if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino) or current.st_size < position:
                raise RuntimeError(f"Training log was replaced or truncated. Reopen the monitor for {path}")
            stream.seek(position)
            time.sleep(2)


def mirror(swanlab, run_dir, stage, logdir, follow, total_steps, *, mode, project):
    launch = read_json(run_dir / "launch.json")
    total_steps = total_steps or launch.get("schedule", {}).get(stage)
    recipe = read_json(run_dir / f"{stage}-config.json")
    phase = run_dir / stage
    hardware = launch.get("physical_gpus")
    config = {**recipe, "stage": stage, "planned_steps": total_steps, "physical_gpus": hardware,
              "code_commit": launch.get("code_commit"), "dataset_sha256": launch.get(
                  "warmup_sha256" if stage == "head" else "dataset_sha256")}

    def experiment(suffix=""):
        return init_dashboard(swanlab, mode=mode, logdir=logdir, project=project,
                              name=f"{run_dir.name}-{stage}{suffix}", group=run_dir.name, config=config)

    tracker = experiment()
    history = deque()
    last_train = last_dev = -1
    try:
        baseline = development_metrics(read_json(phase / "before.json"))
        if baseline:
            tracker.log(baseline, step=0)
            last_dev = 0
        for line in log_lines(run_dir / f"{stage}.log", follow, phase / "training_report.json"):
            if line.startswith(_RESUME):
                resumed_at = int(line.removeprefix(_RESUME))
                tracker.finish()
                tracker = experiment(f"-resume-{resumed_at}")
                history.clear()
                last_train = last_dev = resumed_at - 1
                print(line, flush=True)
                continue
            match = _TRAIN.fullmatch(line)
            if match:
                step, loss, elapsed, ranks = int(match[1]), float(match[2]), float(match[3]), int(match[4])
                if step > last_train and math.isfinite(loss) and math.isfinite(elapsed):
                    tracker.log(progress_metrics(step, loss, elapsed, ranks, history, total_steps,
                                                 recipe.get("global_batch_size")), step=step)
                    last_train = step
                    print(line, flush=True)
            elif line.startswith(_DEV):
                try:
                    report = json.loads(line.removeprefix(_DEV))
                except json.JSONDecodeError as error:
                    print(f"Skipped an incomplete development record: {error.msg}", flush=True)
                    continue
                step = report["step"]
                if step > last_dev:
                    metrics = development_metrics(report)
                    tracker.log(metrics, step=step)
                    last_dev = step
                    decision = metrics.get("dev/decision/accuracy")
                    box = metrics.get("dev/box/acc_iou50")
                    print(f"GROUNDING DEV step={step} decision_accuracy={decision} box_acc_iou50={box}", flush=True)
    except KeyboardInterrupt:
        tracker.finish(state="aborted")
    except Exception:
        tracker.finish(state="crashed")
        raise
    else:
        tracker.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing head.log and sft.log")
    parser.add_argument("--stage", choices=("head", "sft"), default="sft")
    parser.add_argument("--logdir", type=Path, help="SwanLab storage directory (default: RUN/swanlog)")
    parser.add_argument("--mode", choices=("local", "online"), default="local",
                        help="Open a local dashboard or send metrics to an authenticated SwanLab account")
    parser.add_argument("--project", default="laya-multimodal", help="SwanLab project name")
    parser.add_argument("--total-steps", type=int, help="Planned optimizer steps when launch.json has no schedule")
    parser.add_argument("--follow", action="store_true", help="Keep importing while training appends to the log")
    args = parser.parse_args()
    if not (args.run / f"{args.stage}.log").is_file():
        parser.error(f"No {args.stage}.log in {args.run}")
    if args.total_steps is not None and args.total_steps < 1:
        parser.error("--total-steps must be positive")
    try:
        import swanlab
    except ImportError:
        parser.error("Install SwanLab in the monitoring environment: pip install 'swanlab[dashboard]>=0.10.6'")
    mirror(swanlab, args.run.resolve(), args.stage, args.logdir or args.run / "swanlog",
           args.follow, args.total_steps, mode=args.mode, project=args.project)


if __name__ == "__main__":
    main()
