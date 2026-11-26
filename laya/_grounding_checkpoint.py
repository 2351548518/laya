"""Atomic local snapshots for resuming the joint decision and Box trainer."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .multimodal_distributed import _atomic_torch_save


def _atomic_json(path, value):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_checkpoint(path):
    directory = Path(path).resolve()
    if (directory / "progress.json").is_file():
        return directory
    pointer = json.loads((directory / "checkpoint_latest.json").read_text())
    snapshot = (directory / pointer["path"]).resolve()
    if not snapshot.is_relative_to(directory / "optimizer-checkpoints"):
        raise ValueError("checkpoint pointer must reference this run's optimizer-checkpoints")
    return snapshot


def save_checkpoint(output, agent, ctx, optimizer, progress, order, rng, option_rng):
    root = output / "optimizer-checkpoints"
    directory = None
    if ctx.main:
        root.mkdir(exist_ok=True)
        directory = tempfile.mkdtemp(prefix=".partial-", dir=root)
    staged = Path(ctx.broadcast(directory))
    _atomic_torch_save(option_rng.getstate() if option_rng is not None else None,
                       staged / ("option_rng_rank%d.pt" % ctx.rank))
    if ctx.main:
        from safetensors.torch import save_file
        weights = {name: p.detach().to("cpu", copy=True).contiguous()
                   for name, p in agent.model.named_parameters() if p.requires_grad}
        save_file(weights, str(staged / "trainable.safetensors"))
        del weights
        _atomic_torch_save({"order": torch.tensor(order, dtype=torch.int64), "rng": rng.getstate()},
                           staged / "sampler.pt")
        # Selection snapshots are independent of the mutable best and initial
        # directories. A restart can recover the selection at this exact step.
        for name in ("best", "initial"):
            if (output / name).is_dir():
                shutil.copytree(output / name, staged / name)
    ctx.save_training_state(staged, optimizer)
    if ctx.main:
        _atomic_json(staged / "progress.json", progress)
        snapshot = root / ("step-%08d-%s" % (progress["steps"], staged.name.removeprefix(".partial-")))
        staged.replace(snapshot)
        _atomic_json(output / "checkpoint_latest.json", {"path": str(snapshot.relative_to(output))})
        # Keep this snapshot and its most recent predecessor. Incomplete
        # snapshots are never eligible to load or to displace a published one.
        previous = sorted((entry for entry in root.glob("step-*") if entry != snapshot and not entry.is_symlink()
                           and (entry / "progress.json").is_file()),
                          key=lambda entry: entry.stat().st_mtime_ns, reverse=True)
        for expired in previous[1:]:
            shutil.rmtree(expired)
    ctx.barrier()


def load_checkpoint(path, output, agent, ctx, optimizer, identity, rng, option_rng):
    from safetensors.torch import load_file

    directory = resolve_checkpoint(path)
    progress = json.loads((directory / "progress.json").read_text())
    if progress["identity"] != identity:
        raise ValueError("resume checkpoint does not match the data, model, ranks or training configuration")
    sampler = torch.load(directory / "sampler.pt", map_location="cpu", weights_only=True)
    order = sampler["order"].tolist()
    count = identity["train_records"]
    batch = identity["config"]["global_batch_size"]
    steps, epoch, offset = progress["steps"], progress["epoch"], progress["next_offset"]
    expected_step = epoch * math.ceil(count / batch) + math.ceil(offset / batch)
    if (len(order) != count or not 0 < steps <= identity["schedule_steps"]
            or not 0 <= epoch < identity["config"]["epochs"] or not 0 < offset <= count
            or (offset != count and offset % batch) or expected_step != steps):
        raise ValueError("resume checkpoint has an invalid optimizer cursor")
    weights = load_file(str(directory / "trainable.safetensors"))
    params = {name: p for name, p in agent.model.named_parameters() if p.requires_grad}
    if weights.keys() != params.keys() or any(
            tensor.shape != params[name].shape or tensor.dtype != params[name].dtype
            for name, tensor in weights.items()):
        raise ValueError("resume checkpoint does not match the trainable parameters")
    options = torch.load(directory / ("option_rng_rank%d.pt" % ctx.rank), map_location="cpu", weights_only=True)
    if (options is None) != (option_rng is None):
        raise ValueError("resume checkpoint has different option shuffling")
    if not ctx.restore_training_state(directory, optimizer):
        raise ValueError("resume checkpoint is missing optimizer state")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(weights[name])
    rng.setstate(sampler["rng"])
    if option_rng is not None:
        option_rng.setstate(options)
    agent.grounding_trained = progress["grounding_trained"]
    agent.detection_trained = progress["detection_trained"]
    if ctx.main:
        previous_selection = None
        for name in ("initial", "best"):
            source = directory / name
            destination = output / name
            if destination.exists() or destination.is_symlink():
                if previous_selection is None:
                    previous_selection = Path(tempfile.mkdtemp(prefix=".selection-before-resume-", dir=output))
                destination.replace(previous_selection / name)
            if source.is_dir():
                shutil.copytree(source, destination)
    ctx.barrier()
    return progress, order
