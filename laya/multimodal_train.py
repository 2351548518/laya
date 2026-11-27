"""Supervised head, LoRA and full-parameter training for multimodal decisions."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, dataclass
import hashlib
from functools import lru_cache
from itertools import chain
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .calibrate import fit_temperature_map
from .common import QTYPES, ece_score, temp_bucket, unpermute_probs
from .evals import file_fingerprint
from .multimodal import DEFAULT_MULTIMODAL_MODEL, MultimodalAgent, _active_head, _is_adapter_model
from .multimodal_calibration import ordinal_probabilities
from .multimodal_data import can_shuffle_choice, collate_multimodal_inputs, read_records, record_input, record_media, record_modality
from .multimodal_objectives import decision_loss, expand_lora, grow_lora, learning_rate_factor, lora_target_pattern, optimizer_groups
from .multimodal_distributed import distributed_context
from .train import _train_mode, decision_target as _decision_target, target_from_gold


@dataclass
class MultimodalTrainConfig:
    """Training settings; the validation file is reserved for temperature fitting."""
    mode: str = "lora"
    epochs: int = 1
    learning_rate: float = 1e-4
    grad_accum: int = 4
    micro_batch: int = 4
    lora_rank: int = 8
    lora_alpha: int = 16
    shuffle_options: bool = True
    gradient_checkpointing: bool = True
    loss: str = "soft-ce"
    seed: int = 42
    max_steps: int | None = None
    log_every: int = 10
    video_cache_size: int = 128
    video_cache_dir: str | None = None
    max_micro_tokens: int | None = None
    balanced_batches: bool = False
    train_projectors: bool = False
    head_learning_rate: float | None = None
    projector_learning_rate: float | None = None
    lr_schedule: str = "constant"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    ordinal_weight: float = 0.0
    brier_weight: float = 0.0
    distill_weight: float = 0.0
    distill_temperature: float = 2.0
    kl_weight: float = 0.0
    rl_weight: float = 1.0
    rl_sigma_start: float = 0.3
    rl_sigma_end: float = 0.3
    rl_samples: int = 4
    checkpoint_every: int = 0
    eval_every: int = 0
    ordinal_calibration_guard: bool = False
    global_batch_size: int | None = None
    order_micro_batch: int | None = None
    resume_from: str | None = None
    resume_optimizer: bool = True
    schedule_steps: int | None = None
    lora_targets: str = "qv"
    reference_gold_agree: bool = False
    readout: str | None = None
    prompt_version: str | None = None

    def validate(self):
        for key in ("global_batch_size", "order_micro_batch", "schedule_steps"):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(key + " must be a positive integer")
        if self.resume_from is not None and not isinstance(self.resume_from, str):
            raise ValueError("resume_from must be a local checkpoint path")
        if not isinstance(self.resume_optimizer, bool):
            raise ValueError("resume_optimizer must be a boolean")
        if not isinstance(self.reference_gold_agree, bool):
            raise ValueError("reference_gold_agree must be a boolean")
        if self.lora_targets not in {"qv", "attention", "all-linear"}:
            raise ValueError("unsupported lora_targets scope")
        if self.readout not in {None, "letter", "candidate"} or self.prompt_version not in {None, "v1", "v2"}:
            raise ValueError("unsupported readout or prompt version")
        if self.mode not in ("head", "lora", "full"):
            raise ValueError("mode must be head, lora or full")
        if self.loss not in ("soft-ce", "rlcd"):
            raise ValueError("loss must be soft-ce or rlcd")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be constant or cosine")
        for key in ("ordinal_weight", "brier_weight", "distill_weight", "kl_weight", "rl_weight"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(key + " must be finite and nonnegative")
        for key in ("head_learning_rate", "projector_learning_rate", "distill_temperature", "rl_sigma_start", "rl_sigma_end"):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(key + " must be positive and finite")
        for key in ("warmup_ratio", "min_lr_ratio"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(key + " must lie in [0,1]")
        for key in ("checkpoint_every", "eval_every"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(key + " must be a nonnegative integer")
        if isinstance(self.rl_samples, bool) or not isinstance(self.rl_samples, int) or self.rl_samples < 2:
            raise ValueError("rl_samples must be at least two")
        if self.max_micro_tokens is not None and (isinstance(self.max_micro_tokens, bool)
                or not isinstance(self.max_micro_tokens, int) or self.max_micro_tokens < 1):
            raise ValueError("max_micro_tokens must be positive")
        if self.train_projectors and self.mode == "head":
            raise ValueError("train_projectors requires lora or full mode")
        if isinstance(self.video_cache_size, bool) or not isinstance(self.video_cache_size, int) or self.video_cache_size < 0:
            raise ValueError("video_cache_size must be a nonnegative integer")
        for key in ("epochs", "grad_accum", "micro_batch", "lora_rank", "lora_alpha", "log_every"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("%s must be a positive integer" % key)
        if self.max_steps is not None and (isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int)
                                          or self.max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        if isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")


def load_rows(path):
    """Read native decision JSONL or labelled System One and Kev records."""
    rows = list(read_records(path))
    if not rows:
        raise ValueError("empty multimodal dataset: %s" % path)
    if any(not ("expected" in row or "gold" in row) for row in rows):
        raise ValueError("%s needs expected (or gold) labels" % path)
    return rows


@lru_cache(maxsize=524288)
def _media_hash(path, modified_ns, size):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprints(row):
    state = row["state"]
    keys = set()
    if row.get("group") is not None:
        keys.add(("group", row.get("dataset", ""), str(row["group"])))
    paths = tuple(record_media(row))
    for path in paths:
        stat = Path(path).stat()
        keys.add(("media", _media_hash(str(path), stat.st_mtime_ns, stat.st_size)))
    if not paths:
        text = json.dumps(state, ensure_ascii=False, sort_keys=True)
        if state in ("", "{}", "[]", "null", None) or state == {} or state == []:
            # An empty shared context is not the problem text: the question carries it.
            text += json.dumps(row["questions"], ensure_ascii=False, sort_keys=True)
        keys.add(("text", text))
    return keys


def check_splits(**splits):
    """Reject exact-media and source-group overlap before loading any model."""
    seen = {}
    for name, rows in splits.items():
        for row in rows or []:
            for key in _fingerprints(row):
                if key in seen and seen[key] != name:
                    raise ValueError("data leakage between %s and %s: %s" % (seen[key], name, key))
                seen[key] = name


def filter_split_overlaps(splits):
    """Prepare clean splits, preserving test then validation before excluding training overlaps."""
    occupied, dropped = {}, {}
    for split in ("test", "validation", "train"):
        kept = []
        for row in splits.get(split, []):
            fingerprints = _fingerprints(row)
            if any(key in occupied and occupied[key] != split for key in fingerprints):
                continue
            kept.append(row)
            occupied.update({key: split for key in fingerprints})
        dropped[split] = len(splits.get(split, [])) - len(kept)
        splits[split] = kept
    check_splits(**splits)
    return dropped


def _items(agent, rows):
    items = []
    for row in rows:
        for qid, qdef in row["questions"].items():
            agent._check_question(qid, qdef)
            q = agent._to_internal(qdef)
            items.append((row, qid, qdef, _decision_target(row, qid, q)))
    return items


def example_hash(row, qid, qdef):
    content = [row["state"], qid, qdef]
    if row.get("_image_assets"):
        content.append(row["_image_assets"])
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def _calibration_group(row):
    for value in row.get("leakage_groups", []):
        if ":actor:" in value or ":speaker:" in value or "uploader:" in value:
            return value
    return str(row.get("dataset", "custom")) + ":" + str(row.get("group", row.get("id")))


def _evaluation_video_key(agent, row):
    state = row["state"]
    video = state.get("video") if isinstance(state, dict) else None
    if isinstance(video, (str, os.PathLike)) and not os.fspath(video).endswith(".frames.json"):
        return (os.path.realpath(video), state.get("video_frames", agent.frames),
                state.get("video_audio", agent.use_audio_in_video), tuple(state.get("video_bounds") or ()))
    return None


def _serial_evaluation_order(agent, items):
    """Place requests sharing a decoded clip together when an in-memory cache is active."""
    if not getattr(agent, "_video_cache_size", 0):
        return enumerate(items)
    groups = {}
    for index, item in enumerate(items):
        key = _evaluation_video_key(agent, item[0])
        groups.setdefault(index if key is None else key, []).append((index, item))
    return chain.from_iterable(groups.values())


@torch.no_grad()
def collect_records(agent, rows, micro_batch=4, *, max_micro_tokens=None):
    """Collect unrounded option logits for calibration and held-out evaluation."""
    if micro_batch < 1 or (max_micro_tokens is not None and max_micro_tokens < 1):
        raise ValueError("evaluation batch size and token budget must be positive")
    agent.model.eval()
    items = _items(agent, rows)
    records = [None] * len(items)
    prepared, metadata = [], []
    finished = 0

    def collect_batch():
        nonlocal finished
        inputs = prepared[0] if len(prepared) == 1 else collate_multimodal_inputs(
            prepared, getattr(agent.tok, "pad_token_id", 0) or 0)
        logits = agent._logits(inputs, max(len(entry[4]) for entry in metadata)).detach().cpu().numpy()
        for values, (index, row, qid, q, target, modality) in zip(logits, metadata):
            values = unpermute_probs(values[:len(target)], q.get("option_order"))
            records[index] = {"qtype": QTYPES[q["t"]], "logits": values, "target": np.asarray(target),
                            "k": len(target), "modality": modality, "id": row.get("id"), "qid": qid,
                            "example_hash": example_hash(row, qid, row["questions"][qid]), "group": row.get("group"),
                            "calibration_group": _calibration_group(row),
                            "source": row.get("source", row.get("dataset", "custom")),
                            "task": (row.get("quality") or {}).get("decision_task") or row.get("benchmark_task")}
        old = finished
        finished += len(metadata)
        if old // 200 < finished // 200:
            print("evaluation decisions=%d/%d" % (finished, len(items)), flush=True)

    longest = 0
    if micro_batch == 1:
        ordered = _serial_evaluation_order(agent, items)
    else:
        ordered = sorted(enumerate(items), key=lambda item: record_modality(item[1][0]))
    for index, (row, qid, qdef, target) in ordered:
        request, question = record_input(agent, row, qid, qdef)
        inputs, q, _media = agent._prepare(request, question)
        length = inputs["input_ids"].shape[-1]
        padded = max(length, longest) * (len(prepared) + 1)
        if prepared and (len(prepared) >= micro_batch or inputs.keys() != prepared[0].keys()
                         or (max_micro_tokens and padded > max_micro_tokens)):
            collect_batch()
            prepared.clear()
            metadata.clear()
            longest = 0
        prepared.append(inputs)
        longest = max(longest, length)
        modality = record_modality(row)
        metadata.append((index, row, qid, q, target, modality))
    if prepared:
        collect_batch()
    return records


def _save_records(path, records):
    serializable = [{key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in record.items()} for record in records]
    Path(path).write_text(json.dumps(serializable) + "\n")


def metrics(records, temperature=None, temperature_by_options=None, modality_calibration=None,
            ordinal_calibration=None, *, slices=False):
    """Compute exact-level accuracy, NLL, Brier and ECE with modality slices."""
    temperature = temperature or [1.0] * 3
    temperature_by_options = temperature_by_options or {}
    scores = []
    for rec in records:
        qt, k = rec["qtype"], rec["k"]
        calibration = (modality_calibration or {}).get(rec["modality"], {})
        use_ordinal = qt == QTYPES["score"] and bool(ordinal_calibration)
        model = ordinal_calibration["models"].get(f"{rec['modality']}:{k}") if use_ordinal else None
        if use_ordinal and (model is not None or qt not in calibration.get("fitted_types", [])):
            p = ordinal_probabilities(rec["logits"], model)
        else:
            scale = calibration.get("temperature_by_options", temperature_by_options).get(
                temp_bucket(qt, k), calibration.get("temperature", temperature)[qt])
            z = rec["logits"] / scale
            p = np.exp(z - z.max())
            p /= p.sum()
        target = rec["target"]
        correct = int(target[p.argmax()] == target.max())
        confidence = float(p.max())
        ordinal = np.arange(k) @ (p - target) if qt == QTYPES["score"] else None
        scores.append({"correct": correct, "confidence": confidence,
                       "nll": float(-(target * np.log(np.maximum(p, 1e-12))).sum()),
                       "brier": float(((p - target) ** 2).sum()), "modality": rec["modality"],
                       "qtype": qt, "source": rec.get("source"), "task": rec.get("task"),
                       "score_mae": float(abs(ordinal)) if ordinal is not None else None,
                       "score_normalized_mae": float(abs(ordinal) / max(1, k - 1)) if ordinal is not None else None,
                       "score_rps": float(np.square(np.cumsum(p - target)[:-1]).sum() / max(1, k - 1))
                       if ordinal is not None else None})

    def summarize(rows):
        if not rows:
            return {"n": 0}
        confidence = np.asarray([r["confidence"] for r in rows])
        correct = np.asarray([r["correct"] for r in rows])
        def mean(field):
            values = [row[field] for row in rows if row[field] is not None]
            return float(np.mean(values)) if values else None
        return {"n": len(rows), "accuracy": float(correct.mean()),
                "ece": ece_score(confidence, correct), "mean_confidence": float(confidence.mean()),
                "nll": mean("nll"), "brier": mean("brier"), "score_mae": mean("score_mae"),
                "score_normalized_mae": mean("score_normalized_mae"), "score_rps": mean("score_rps")}
    result = {"overall": summarize(scores),
              "by_modality": {m: summarize([r for r in scores if r["modality"] == m])
                              for m in sorted({r["modality"] for r in scores})},
              "by_type": {name: summarize([r for r in scores if r["qtype"] == qt]) for name, qt in QTYPES.items()}}
    if slices:
        for key in ("source", "task"):
            groups = {}
            for row in scores:
                if row[key] is not None:
                    groups.setdefault(row[key], []).append(row)
            result["by_" + key] = {name: summarize(group) for name, group in sorted(groups.items())}
    return result


def calibrate_multimodal(agent, records, *, ordinal_guard=False, ordinal_channels=False):
    """Fit temperatures on calibration data, optionally validating ordinal changes on separate groups.

    The ordinal guard uses only this calibration split. Its fit and check groups never consult
    development checkpoint-selection outcomes or test labels.
    """
    from .calibrate import MIN_TYPE_N
    pairs = lambda values: [(r["qtype"], r["logits"], r["target"], r["k"]) for r in values]
    calibration = fit_temperature_map(pairs(records))
    agent.temperature = calibration["temperature"]
    agent.temperature_by_options = calibration["temperature_by_options"]
    counts = Counter(r["qtype"] for r in records)
    agent.calibrated = bool(counts and all(n >= MIN_TYPE_N for n in counts.values()))
    agent.modality_calibration = {}
    agent.ordinal_calibration = {}
    for kind in sorted({r["modality"] for r in records}):
        subset = [r for r in records if r["modality"] == kind]
        fitted = fit_temperature_map(pairs(subset))
        fitted["fitted_types"] = [qt for qt, count in Counter(r["qtype"] for r in subset).items() if count >= MIN_TYPE_N]
        if ordinal_guard:
            score = [r for r in subset if r["qtype"] == QTYPES["score"]]
            folds = [[], []]
            for record in score:
                identity = str(record.get("calibration_group") or record.get("group") or record.get("example_hash") or record.get("id"))
                fold = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 2
                folds[fold].append(record)
            accepted, candidate, evidence = False, None, {"fit_n": len(folds[0]), "check_n": len(folds[1])}
            if min(map(len, folds)) >= MIN_TYPE_N:
                candidate = fit_temperature_map(pairs(folds[0]))
                plain = metrics(folds[1])["overall"]
                changed = metrics(folds[1], candidate["temperature"], candidate["temperature_by_options"])["overall"]
                accepted = changed["nll"] <= plain["nll"] + 1e-6 and changed["score_mae"] <= plain["score_mae"] + 1e-6
                evidence.update(raw=plain, candidate=changed)
            fitted["temperature_by_options"] = {key: value for key, value in fitted["temperature_by_options"].items()
                                                 if not key.startswith("score:")}
            fitted["temperature"][QTYPES["score"]] = candidate["temperature"][QTYPES["score"]] if accepted else 1.0
            if accepted:
                fitted["temperature_by_options"].update({key: value for key, value in candidate["temperature_by_options"].items()
                                                       if key.startswith("score:")})
            else:
                fitted["fitted_types"] = [qt for qt in fitted["fitted_types"] if qt != QTYPES["score"]]
            fitted["ordinal_guard"] = dict(evidence, accepted=accepted)
        agent.modality_calibration[kind] = fitted
    if ordinal_guard:
        # Unseen modalities do not inherit an unchecked global ordinal temperature.
        agent.temperature[QTYPES["score"]] = 1.0
        agent.temperature_by_options = {key: value for key, value in agent.temperature_by_options.items() if not key.startswith("score:")}
        calibration["temperature_by_options"] = agent.temperature_by_options
    calibration["by_modality"] = agent.modality_calibration
    if ordinal_channels:
        from .multimodal_calibration import fit_ordinal_calibration
        agent.ordinal_calibration = fit_ordinal_calibration(records)
        calibration["ordinal_calibration"] = agent.ordinal_calibration
    return calibration


def configure_training(agent, config):
    """Select trainable parameters without changing the input encoders or vocabulary size."""
    config.validate()
    # Any weight update invalidates previously fitted temperatures and modality certificates.
    agent.calibrated, agent.modality_calibration = False, {}
    agent.ordinal_calibration = {}
    agent.temperature, agent.temperature_by_options = [1.0] * 3, {}
    if config.mode == "full":
        agent._checkpoint_mode = "full"
    candidate = getattr(agent, "readout", "letter") == "candidate"
    if _is_adapter_model(agent.model):
        if config.mode != "lora":
            raise ValueError("an adapter checkpoint must be continued in lora mode, or merged before head/full training")
        grow_lora(agent.model, config.lora_rank, config.lora_alpha)
        expand_lora(agent.model, config.lora_targets)
        for name, parameter in agent.model.named_parameters():
            parameter.requires_grad_("lora_" in name or "modules_to_save." in name or "decision_scorer" in name)
        _configure_projectors(agent, config)
        _active_head(agent.model).float()
        if candidate:
            _active_head(agent.model).requires_grad_(False)
        if config.gradient_checkpointing:
            agent.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return [p for p in agent.model.parameters() if p.requires_grad]
    agent.model.requires_grad_(config.mode == "full")
    if config.mode == "lora":
        from peft import LoraConfig, get_peft_model
        agent.model = get_peft_model(agent.model, LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_alpha,
            target_modules=lora_target_pattern(config.lora_targets),
            modules_to_save=None if candidate else ["lm_head"], bias="none"))
    elif not candidate:
        agent.model.get_output_embeddings().requires_grad_(True)
    base = agent.model.get_base_model() if _is_adapter_model(agent.model) else agent.model
    if candidate:
        base.decision_scorer.requires_grad_(True).float()
        _active_head(agent.model).requires_grad_(False)
    _configure_projectors(agent, config)
    _active_head(agent.model).float()
    if config.gradient_checkpointing and config.mode != "head":
        agent.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in agent.model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("no trainable parameters")
    return parameters


def _configure_projectors(agent, config):
    if not config.train_projectors:
        return
    base = agent.model.get_base_model() if _is_adapter_model(agent.model) else agent.model
    names = [name for name, module in base.named_modules()
             if name in {"model.embed_audio", "model.embed_vision"} and module is not None]
    if not names:
        raise ValueError("native projector training currently requires Gemma 4 embed_audio/embed_vision modules")
    agent.native_trainable_modules = names
    for name in names:
        base.get_submodule(name).float().requires_grad_(True)


def _training_order(items, config, rng):
    if not config.balanced_batches:
        indices = list(range(len(items)))
        rng.shuffle(indices)
        return indices
    groups = {}
    for index, (row, _qid, qdef, _target) in enumerate(items):
        groups.setdefault(record_modality(row), []).append(index)
    for indices in groups.values():
        rng.shuffle(indices)
        # Local length buckets preserve shuffle while reducing padding inside native micro-batches.
        for offset in range(0, len(indices), 256):
            indices[offset:offset + 256] = sorted(indices[offset:offset + 256],
                key=lambda i: len(json.dumps([items[i][0]["state"].get("text", "") if isinstance(items[i][0]["state"], dict)
                                             else items[i][0]["state"], items[i][2]], ensure_ascii=False)))
    order = []
    grain = config.order_micro_batch or config.micro_batch
    for offset in range(0, max(map(len, groups.values())), grain):
        for kind in sorted(groups):
            order.extend(groups[kind][offset:offset + grain])
    return order


def _restore_checkpoint(agent, path, mode):
    from safetensors.torch import load_file, load_model
    if mode == "full":
        load_model(agent.model, str(Path(path) / "model.safetensors"))
    elif mode == "lora":
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(agent.model, load_file(str(Path(path) / "adapter/adapter_model.safetensors")))
    _active_head(agent.model).load_state_dict(load_file(str(Path(path) / "decision_head.safetensors")))
    if getattr(agent, "readout", "letter") == "candidate":
        base = agent.model.get_base_model() if _is_adapter_model(agent.model) else agent.model
        base.decision_scorer.load_state_dict(load_file(str(Path(path) / "candidate_head.safetensors")))
    names = getattr(agent, "native_trainable_modules", [])
    if names and mode != "full":
        base = agent.model.get_base_model() if _is_adapter_model(agent.model) else agent.model
        weights = load_file(str(Path(path) / "native_projectors.safetensors"))
        for name in names:
            prefix = name + "."
            base.get_submodule(name).load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})


def finetune_multimodal(train_data, output_dir, *, model=DEFAULT_MULTIMODAL_MODEL, validation_data=None,
                        test_data=None, config=None, device=None, max_len=4096, frames=8,
                        agent=None, selection_data=None, reference_data=None):
    """Train decisions, calibrate on validation data, and compare on a separate test set."""
    config = config or MultimodalTrainConfig()
    config.validate()
    distributed = distributed_context()
    if distributed.world_size > 1:
        device = distributed.device
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_rows = load_rows(train_data)
    if distributed.main:
        val_rows = load_rows(validation_data) if validation_data else []
        test_rows = load_rows(test_data) if test_data else []
        selection_rows = load_rows(selection_data) if selection_data else []
        check_splits(train=train_rows, validation=val_rows, test=test_rows, selection=selection_rows)
    distributed.barrier()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if distributed.main:
        print("audited rows: train=%d validation=%d test=%d world_size=%d" % (
            len(train_rows), len(val_rows), len(test_rows), distributed.world_size), flush=True)
    if agent is not None and distributed.world_size > 1 and agent.device != distributed.device:
        raise ValueError("agent device must match this rank's local device")
    supplied_agent = agent is not None
    agent = agent or MultimodalAgent(config.resume_from or model, device=device, max_len=max_len, frames=frames,
                                    readout=config.readout, prompt_version=config.prompt_version)
    if config.resume_from and supplied_agent:
        _restore_checkpoint(agent, config.resume_from, config.mode)
    agent._video_cache_size = config.video_cache_size
    agent._video_cache_dir = config.video_cache_dir
    before = None
    if distributed.main:
        if test_rows:
            print("Evaluating the initial model on held-out test decisions", flush=True)
        initial_records = collect_records(agent, test_rows, config.micro_batch,
                                          max_micro_tokens=config.max_micro_tokens) if test_rows else []
        _save_records(output / "test_before.json", initial_records)
        before = metrics(initial_records) if test_rows else None
    distributed.barrier()
    parameters = configure_training(agent, config)
    distributed.synchronize_parameters(parameters)
    torch.manual_seed(config.seed + distributed.rank * 1009)
    optimizer = torch.optim.AdamW(optimizer_groups(agent.model, config.learning_rate,
                                                  config.head_learning_rate, config.projector_learning_rate))
    items = _items(agent, train_rows)
    if not items:
        raise ValueError("no training decisions")
    references = {r["example_hash"]: r for r in json.loads(Path(reference_data).read_text())} if reference_data else {}
    distill = config.distill_weight > 0
    anchor = config.kl_weight > 0
    if anchor and not references:
        raise ValueError("kl_weight requires reference_data collected on the exact training decisions")
    if references:
        for row, qid, qdef, target in items:
            key = example_hash(row, qid, qdef)
            if (key not in references and not config.reference_gold_agree) or (
                    key in references and references[key]["k"] != len(target)):
                raise ValueError("reference cache does not match training input/options: " + str(row.get("id")))
    teacher_count = sum(bool(row.get("teacher_targets", {}).get(qid)) for row, qid, _, _ in items)
    if distill and teacher_count == 0:
        raise ValueError("distillation enabled but training has no separate teacher targets")
    rng = random.Random(config.seed)
    losses, steps, trained_decisions, elapsed = [], 0, 0, time.perf_counter()
    effective_batch = config.global_batch_size or config.grad_accum * config.micro_batch * distributed.world_size
    total_steps = config.schedule_steps or math.ceil(len(items) / effective_batch) * config.epochs
    selection_history, best_key, best_step = [], None, None
    train_sha = file_fingerprint(train_data)
    resume_cursor, optimizer_restored = 0, False
    if config.resume_from:
        progress = json.loads((Path(config.resume_from) / "training_progress.json").read_text())
        previous = progress["config"]
        if progress.get("train_sha256", train_sha) != train_sha:
            raise ValueError("resume checkpoint belongs to a different training manifest")
        for key in ("mode", "seed", "lora_rank", "lora_alpha", "loss", "shuffle_options", "balanced_batches",
                    "train_projectors", "learning_rate", "head_learning_rate", "projector_learning_rate",
                    "ordinal_weight", "brier_weight", "distill_weight", "distill_temperature", "kl_weight",
                    "epochs", "lr_schedule", "warmup_ratio", "min_lr_ratio", "lora_targets",
                    "reference_gold_agree", "readout", "prompt_version", "rl_weight", "rl_samples",
                    "rl_sigma_start", "rl_sigma_end"):
            if previous.get(key, getattr(MultimodalTrainConfig(), key)) != getattr(config, key):
                raise ValueError("resume configuration changed: " + key)
        prior_horizon = progress.get("schedule_total_steps", previous.get("schedule_steps"))
        if prior_horizon is not None and prior_horizon != total_steps:
            raise ValueError("resume requires preserving the learning-rate schedule horizon")
        previous_batch = previous.get("global_batch_size") or (
            previous["micro_batch"] * previous["grad_accum"] * progress.get("world_size", 1))
        if effective_batch != previous_batch:
            raise ValueError("resume requires preserving the global batch size")
        previous_grain = previous.get("order_micro_batch") or previous["micro_batch"]
        if config.balanced_batches and (config.order_micro_batch or config.micro_batch) != previous_grain:
            raise ValueError("resume requires preserving order_micro_batch")
        steps, trained_decisions = progress["optimizer_steps"], progress["trained_decisions"]
        resume_cursor = trained_decisions
        if resume_cursor > len(items) * config.epochs:
            raise ValueError("resume data cursor exceeds the requested training epochs")
        selection_history = progress.get("selection_history", [])
        best_step = progress.get("best_step")
        if best_step is not None:
            chosen = next(r for r in selection_history if r["step"] == best_step)
            best_key = chosen["macro_accuracy"], -chosen["metrics"]["overall"]["nll"]
            source_best = Path(config.resume_from).parent / "checkpoint_best"
            if not (output / "checkpoint_best").is_dir() and not source_best.is_dir():
                raise ValueError("resume checkpoint's best weights are missing: " + str(source_best))
            if not (output / "checkpoint_best").is_dir() and source_best.is_dir() and distributed.main:
                import shutil
                source_progress = json.loads((source_best / "training_progress.json").read_text())
                if source_progress["optimizer_steps"] != best_step:
                    raise ValueError("resume checkpoint's matching best weights are no longer available")
                shutil.copytree(source_best, output / "checkpoint_best")
        if config.resume_optimizer:
            optimizer_restored = distributed.restore_training_state(config.resume_from, optimizer)
        if distributed.main:
            print("Resuming step=%d decisions=%d optimizer_restored=%s" % (
                steps, trained_decisions, optimizer_restored), flush=True)
            if not optimizer_restored:
                print("Weight/data-cursor continuation: previous checkpoint has no restored Adam state", flush=True)
    if distributed.main:
        print("Training %d decisions, mode=%s global_batch=%d ranks=%d local_micro=%d" % (
            len(items), config.mode, effective_batch, distributed.world_size, config.micro_batch), flush=True)
    distributed.barrier()

    def checkpoint(name):
        destination = output / name
        if distributed.main:
            destination.mkdir(parents=True, exist_ok=True)
            agent.save(destination, mode=config.mode)
        distributed.save_training_state(destination, optimizer)
        if distributed.main:
            (destination / "training_progress.json").write_text(json.dumps({"optimizer_steps": steps,
                "trained_decisions": trained_decisions, "config": asdict(config), "epoch": epoch + 1,
                "world_size": distributed.world_size, "train_sha256": train_sha,
                "schedule_total_steps": total_steps,
                "selection_history": selection_history, "best_step": best_step}) + "\n")
        distributed.barrier()

    def select_model():
        nonlocal best_key, best_step
        if not selection_data:
            return
        measured = metrics(collect_records(agent, selection_rows, config.micro_batch,
                                           max_micro_tokens=config.max_micro_tokens)) if distributed.main else None
        measured = distributed.broadcast(measured)
        macro = float(np.mean([values["accuracy"] for values in measured["by_modality"].values()]))
        key = macro, -measured["overall"]["nll"]
        selection_history.append({"step": steps, "macro_accuracy": macro, "metrics": measured})
        if best_key is None or key > best_key:
            best_key, best_step = key, steps
            checkpoint("checkpoint_best")
        if distributed.main:
            print("selection step=%d macro_accuracy=%.5f best_step=%s" % (steps, macro, best_step), flush=True)
        _train_mode(agent, config)

    for epoch in range(config.epochs):
        indices = _training_order(items, config, rng)
        _train_mode(agent, config)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss, epoch_count, log_loss, log_count = 0., 0, 0., 0
        log_parts = Counter()
        for offset in range(0, len(indices), effective_batch):
            window = indices[offset:offset + effective_batch]
            plans = []
            # All ranks advance the same option-permutation RNG, even for other ranks'
            # decisions and replayed windows. Changing GPU count preserves the data cursor.
            for position, index in enumerate(window):
                row, qid, qdef, target = items[index]
                question = copy.deepcopy(qdef)
                order = list(range(len(target)))
                if config.shuffle_options and (can_shuffle_choice(question)
                        or question["type"] == "noul" and question.get("option_order") is None):
                    rng.shuffle(order)
                    question["option_order"] = order
                else:
                    order = question.get("option_order", order)
                if position % distributed.world_size == distributed.rank:
                    plans.append((row, qid, qdef, target, question, order))
            cursor = epoch * len(items) + offset
            if cursor + len(window) <= resume_cursor:
                continue
            if cursor < resume_cursor:
                raise ValueError("resume data cursor is inside an optimizer window")
            if config.max_steps is not None and steps >= config.max_steps:
                break
            for start in range(0, len(plans), config.micro_batch):
                examples = []
                for row, qid, qdef, target, question, order in plans[start:start + config.micro_batch]:
                    request, definition = record_input(agent, row, qid, question)
                    inputs, q, _metadata = agent._prepare(request, definition)
                    teacher = row.get("teacher_targets", {}).get(qid)
                    teacher = target_from_gold(agent._to_internal(qdef), teacher) if teacher else None
                    accepted = teacher is not None and target[int(np.argmax(teacher))] == max(target)
                    reference = references.get(example_hash(row, qid, qdef)) if references else None
                    reference_accepted = reference is not None
                    if reference is not None:
                        z = np.asarray(reference["logits"], dtype=np.float64)
                        reference = np.exp(z - z.max()); reference /= reference.sum()
                        if config.reference_gold_agree:
                            reference_accepted = target[int(reference.argmax())] == max(target)
                    teacher_scores = [teacher[i] for i in order] if distill and teacher is not None else []
                    reference_scores = [float(reference[i]) for i in order] if anchor and reference is not None else []
                    examples.append((inputs, [target[i] for i in order], QTYPES[q["t"]],
                                     teacher_scores, accepted, reference_scores, reference_accepted))
                # Native batches may be split by their actual padded token footprint. Every
                # micro-batch retains the same optimizer-window denominator, including a tail.
                position = 0
                while position < len(examples):
                    end = position + 1
                    while end < len(examples):
                        candidate = examples[position:end + 1]
                        footprint = max(e[0]["input_ids"].shape[-1] for e in candidate) * len(candidate)
                        if config.max_micro_tokens and footprint > config.max_micro_tokens:
                            break
                        end += 1
                    batch = examples[position:end]
                    inputs = collate_multimodal_inputs([e[0] for e in batch], getattr(agent.tok, "pad_token_id", 0) or 0)
                    width = max(len(e[1]) for e in batch)
                    mask = torch.tensor([[i < len(e[1]) for i in range(width)] for e in batch], device=agent.device)
                    logits = agent._logits(inputs, width).masked_fill(~mask, -1e4)
                    tensor = lambda values: torch.tensor([v + [0.] * (width - len(v)) for v in values], device=agent.device, dtype=torch.float32)
                    y = tensor([e[1] for e in batch])
                    progress = steps / max(1, total_steps - 1)
                    sigma = config.rl_sigma_start + (config.rl_sigma_end - config.rl_sigma_start) * progress
                    loss, _parts = decision_loss(logits, y, mask, torch.tensor([e[2] for e in batch], device=agent.device),
                        objective=config.loss, ordinal_weight=config.ordinal_weight, brier_weight=config.brier_weight,
                        teacher=tensor([e[3] for e in batch]) if distill else None,
                        teacher_mask=torch.tensor([e[4] for e in batch], device=agent.device) if distill else None,
                        distill_weight=config.distill_weight, distill_temperature=config.distill_temperature,
                        reference=tensor([e[5] for e in batch]) if anchor else None, kl_weight=config.kl_weight,
                        reference_mask=torch.tensor([e[6] for e in batch], device=agent.device) if anchor else None,
                        sigma=sigma, rl_samples=config.rl_samples, rl_weight=config.rl_weight)
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite training loss at row %r question %r" % (row.get("id"), qid))
                    (loss * len(batch) / len(window)).backward()
                    amount = float(loss.detach()) * len(batch)
                    epoch_loss += amount
                    epoch_count += len(batch)
                    log_loss += amount
                    log_count += len(batch)
                    for name, value in _parts.items():
                        log_parts[name] += value * len(batch)
                    log_parts["teacher_accepted"] += sum(e[4] for e in batch)
                    log_parts["reference_accepted"] += sum(e[6] for e in batch)
                    position = end
            distributed.sum_gradients(parameters)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            factor = learning_rate_factor(steps + 1, total_steps, config.warmup_ratio, config.min_lr_ratio) if config.lr_schedule == "cosine" else 1.0
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            trained_decisions += len(window)
            if steps % config.log_every == 0:
                total_loss, total_count = distributed.sum_values([log_loss, log_count], device=agent.device)
                names = ["ce", "policy_gradient", "ordinal_rps", "brier", "distillation_kl", "reference_kl",
                         "teacher_accepted", "reference_accepted"]
                values = distributed.sum_values([log_parts[name] for name in names], device=agent.device)
                if distributed.main:
                    print("epoch=%d step=%d decisions=%d loss=%.5f lr_factor=%.4f elapsed=%.1fs ranks=%d" % (
                        epoch + 1, steps, trained_decisions, total_loss / max(1, total_count), factor,
                        time.perf_counter() - elapsed, distributed.world_size), flush=True)
                    print("loss_parts=" + json.dumps(dict(zip(names, [v / max(1, total_count) for v in values])))
                          + " grad_norm=%.5f" % float(grad_norm), flush=True)
                log_loss, log_count = 0., 0
                log_parts.clear()
            if config.eval_every and steps % config.eval_every == 0:
                select_model()
            if config.checkpoint_every and steps % config.checkpoint_every == 0:
                checkpoint("checkpoint_latest")
            if config.max_steps is not None and steps >= config.max_steps:
                break
        total_loss, total_count = distributed.sum_values([epoch_loss, epoch_count], device=agent.device)
        if total_count:
            losses.append(total_loss / total_count)
        # An epoch checkpoint is uncalibrated until the validation stage below.
        agent.calibrated = False
        if not selection_history or selection_history[-1]["step"] != steps:
            select_model()
        checkpoint("checkpoint_latest")
        if config.max_steps is not None and steps >= config.max_steps:
            break
    agent.model.eval()
    optimizer.state.clear()
    if best_step is not None:
        _restore_checkpoint(agent, output / "checkpoint_best", config.mode)
    calibration = None
    if distributed.main and val_rows:
        print("Fitting temperatures on validation decisions", flush=True)
        validation_records = collect_records(agent, val_rows, config.micro_batch,
                                             max_micro_tokens=config.max_micro_tokens)
        _save_records(output / "validation.json", validation_records)
        calibration = calibrate_multimodal(agent, validation_records, ordinal_guard=config.ordinal_calibration_guard)
    after_uncalibrated, after = None, None
    if distributed.main:
        if test_rows:
            print("Evaluating trained model on held-out test decisions", flush=True)
        test_records = collect_records(agent, test_rows, config.micro_batch,
                                       max_micro_tokens=config.max_micro_tokens) if test_rows else []
        _save_records(output / "test_after.json", test_records)
        after_uncalibrated = metrics(test_records) if test_rows else None
        after = metrics(test_records, agent.temperature, agent.temperature_by_options,
                        getattr(agent, "modality_calibration", {})) if test_rows else None
        agent.save(output, mode=config.mode)
    distributed.barrier()
    report = None
    if distributed.main:
        report = {
            "model": agent.model_id, "model_type": agent.model_type, "revision": agent.revision,
            "config": asdict(config), "train_rows": len(train_rows), "train_decisions": len(items),
            "validation_rows": len(val_rows), "test_rows": len(test_rows), "optimizer_steps": steps,
            "trained_decisions": trained_decisions,
            "selection_rows": len(selection_rows), "selection_history": selection_history,
            "world_size": distributed.world_size, "global_batch_size": effective_batch,
            "schedule_total_steps": total_steps, "policy_estimator": "per-decision leave-one-out without std normalization",
            "parallelism": "synchronous optimizer-window gradient sum" if distributed.world_size > 1 else "single process",
            "resume": {"checkpoint": config.resume_from, "starting_decisions": resume_cursor,
                       "optimizer_restored": optimizer_restored},
            "selected_step": best_step if best_step is not None else steps,
            "teacher_target_decisions": teacher_count, "reference_cache": str(reference_data) if reference_data else None,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "total_parameters": sum(p.numel() for p in agent.model.parameters()), "epoch_loss": losses,
            "elapsed_seconds": time.perf_counter() - elapsed, "before": before,
            "after_uncalibrated": after_uncalibrated, "after": after,
            "calibration": calibration, "held_out": bool(test_rows),
            "dataset_sha256": {"train": train_sha,
                               "validation": file_fingerprint(validation_data) if validation_data else None,
                               "test": file_fingerprint(test_data) if test_data else None,
                               "selection": file_fingerprint(selection_data) if selection_data else None,
                               "reference": file_fingerprint(reference_data) if reference_data else None},
        }
        (output / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    report = distributed.broadcast(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--test")
    parser.add_argument("--selection", help="Independent development set for checkpoint selection")
    parser.add_argument("--reference", help="Frozen reference logits collected on training inputs")
    parser.add_argument("--config", help="JSON training settings; unspecified CLI settings inherit this file")
    parser.add_argument("--resume", help="Local checkpoint with weights and a training_progress.json data cursor")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MULTIMODAL_MODEL)
    parser.add_argument("--device")
    parser.add_argument("--mode", choices=["head", "lora", "full"])
    parser.add_argument("--loss", choices=["soft-ce", "rlcd"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--threads", type=int, default=4, help="CPU preprocessing threads in this training process")
    parser.add_argument("--video-cache-size", type=int, help="LRU decoded-video cache entries in host RAM; 0 disables caching")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    values = json.loads(Path(args.config).read_text()) if args.config else {}
    for key in ("mode", "loss", "epochs", "learning_rate", "grad_accum", "micro_batch", "lora_rank",
                "lora_alpha", "max_steps", "seed", "video_cache_size"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if args.no_checkpointing:
        values["gradient_checkpointing"] = False
    config = MultimodalTrainConfig(**values)
    if args.resume:
        config.resume_from = args.resume
    report = finetune_multimodal(args.train, args.out, model=args.model, validation_data=args.validation,
                                test_data=args.test, config=config, device=args.device,
                                max_len=args.max_len, frames=args.frames,
                                selection_data=args.selection, reference_data=args.reference)
    if distributed_context().main:
        print(json.dumps(report, indent=2))
    from .multimodal_distributed import close_distributed
    close_distributed()


if __name__ == "__main__":
    main()
