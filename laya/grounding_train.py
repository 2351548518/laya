"""Staged geometric/set supervision with ordinary decision replay on one local VLM."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import distributed as dist
from torch.nn import functional as F

from ._grounding_checkpoint import load_checkpoint, save_checkpoint
from .evals import file_fingerprint
from .grounding import box_iou, crowd_overlap, grounding_loss, validate_boxes
from .multimodal import MultimodalAgent, _active_head, _is_adapter_model
from .multimodal_data import can_shuffle_choice, normalize_record, read_records, record_input, record_modality
from .multimodal_distributed import distributed_context
from .multimodal_train import (
    MultimodalTrainConfig, configure_training, collate_multimodal_inputs, _train_mode, _decision_target,
)
from .multimodal_objectives import learning_rate_factor, optimizer_groups


@dataclass
class GroundingTrainConfig:
    """Head warmup or LoRA/full adaptation; evaluation benchmarks are never selection data."""
    mode: str = "head"
    epochs: int = 1
    max_steps: int | None = None
    global_batch_size: int = 32
    learning_rate: float = 2e-5
    head_learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: str = "attention"
    slots: int = 1
    max_len: int = 4096
    replay_fraction: float = .2
    replay_kl_weight: float = .1
    train_readout_in_warmup: bool = False
    max_decision_regression: float = .05
    eval_every: int = 500
    eval_limit: int = 256
    decision_eval_micro_batch: int = 1
    grounding_eval_micro_batch: int = 1
    seed: int = 42
    micro_batch: int = 4
    max_micro_tokens: int = 4096
    decision_ordinal_weight: float = 0.0
    decision_brier_weight: float = 0.0
    video_cache_dir: str | None = None
    selection_metric: str = "joint"
    include_initial_selection: bool = False
    quality_matched_only: bool = False
    retention_tagged_only: bool = False
    batch_decisions: bool = False
    max_grounding_regression: float | None = None
    projector_learning_rate: float | None = None
    warmup_ratio: float = .05
    min_lr_ratio: float = .1
    length_bucketing: bool = False
    activation_offload_min_tokens: int | None = None
    video_cache_size: int = 0
    shuffle_options: bool = False
    balance_ranks: bool = False
    prefetch_video: bool = False
    distributed_decision_eval: bool = False
    checkpoint_every: int = 0
    fused_adamw: bool = False

    def validate(self):
        if self.mode not in {"head", "lora", "full"}:
            raise ValueError("grounding mode must be head, lora or full")
        if self.lora_targets not in {'qv', 'attention', 'all-linear'}:
            raise ValueError('invalid grounding LoRA target scope')
        for key in ("epochs", "global_batch_size", "micro_batch", "max_micro_tokens", "slots", "max_len",
                    "eval_limit", "lora_rank", "lora_alpha", "decision_eval_micro_batch", "grounding_eval_micro_batch"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(key + " must be a positive integer")
        if self.max_steps is not None and (isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1):
            raise ValueError("max_steps must be positive")
        value = self.activation_offload_min_tokens
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError("activation_offload_min_tokens must be a positive integer or None")
        if not 0 <= self.replay_fraction < 1 or not math.isfinite(self.replay_fraction):
            raise ValueError("replay_fraction must lie in [0,1)")
        if isinstance(self.eval_every, bool) or not isinstance(self.eval_every, int) or self.eval_every < 0:
            raise ValueError("eval_every must be a nonnegative integer")
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be a nonnegative integer")
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in [self.learning_rate, self.head_learning_rate]):
            raise ValueError("learning rates must be finite and positive")
        if self.projector_learning_rate is not None and (isinstance(self.projector_learning_rate, bool)
                or not math.isfinite(self.projector_learning_rate) or self.projector_learning_rate <= 0):
            raise ValueError('projector_learning_rate must be finite and positive')
        for key in ('warmup_ratio', 'min_lr_ratio'):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(key + ' must lie in [0,1]')
        if not math.isfinite(self.replay_kl_weight) or self.replay_kl_weight < 0:
            raise ValueError("replay_kl_weight must be finite and nonnegative")
        for key in ('decision_ordinal_weight','decision_brier_weight'):
            value=getattr(self,key)
            if isinstance(value,bool) or not math.isfinite(value) or value<0:
                raise ValueError(key+' must be finite and nonnegative')
        if self.video_cache_dir is not None and (not isinstance(self.video_cache_dir,str) or not self.video_cache_dir):
            raise ValueError('video_cache_dir must be a nonempty local path or None')
        if isinstance(self.video_cache_size, bool) or not isinstance(self.video_cache_size, int) or self.video_cache_size < 0:
            raise ValueError('video_cache_size must be a nonnegative integer')
        if not isinstance(self.shuffle_options, bool):
            raise ValueError('shuffle_options must be boolean')
        if not isinstance(self.balance_ranks, bool):
            raise ValueError('balance_ranks must be boolean')
        if not isinstance(self.prefetch_video, bool):
            raise ValueError('prefetch_video must be boolean')
        if not isinstance(self.distributed_decision_eval, bool):
            raise ValueError('distributed_decision_eval must be boolean')
        if not isinstance(self.fused_adamw, bool):
            raise ValueError('fused_adamw must be boolean')
        if self.selection_metric not in {'joint', 'decision_macro', 'harmonic'} or not isinstance(self.include_initial_selection,bool) or not isinstance(self.quality_matched_only,bool):
            raise ValueError('invalid checkpoint selection configuration')
        if not isinstance(self.retention_tagged_only,bool):raise ValueError('retention_tagged_only must be boolean')
        if not isinstance(self.batch_decisions,bool):raise ValueError('batch_decisions must be boolean')
        if not isinstance(self.length_bucketing,bool):raise ValueError('length_bucketing must be boolean')
        if self.max_grounding_regression is not None and (isinstance(self.max_grounding_regression,bool) or
                not math.isfinite(self.max_grounding_regression) or not 0<=self.max_grounding_regression<=1):
            raise ValueError('max_grounding_regression must lie in [0,1] or be None')
        if not isinstance(self.train_readout_in_warmup, bool) or not 0 <= self.max_decision_regression <= 1:
            raise ValueError("invalid readout warmup or decision retention guard")


def _validate_grounding_row(row):
    if len(row.get("questions", {})) != 1:
        raise ValueError("grounding training rows require exactly one question")
    for qid, qdef in row["questions"].items():
        MultimodalAgent._check_question(qid, qdef)
    if qdef["type"] != "box":
        _decision_target(row, qid, MultimodalAgent._to_internal(qdef))
        if "grounding" in row:
            raise ValueError("geometric supervision requires a Box question")
        return
    if "grounding" not in row:
        raise ValueError("Box question needs geometric supervision")
    truth = row["grounding"]
    validate_boxes(truth["boxes"])
    validate_boxes(truth.get("ignored", []))
    if truth["task"] not in {"single", "multiple"}:
        raise ValueError("unknown geometric supervision task")
    if truth["task"] == "single" and len(truth["boxes"]) > 1:
        raise ValueError("an official Box target contains at most one box")


def load_grounding_rows(path):
    rows = list(read_records(path))
    for row in rows:
        _validate_grounding_row(row)
    if not rows:
        raise ValueError("empty grounding dataset: " + str(path))
    return rows


class _IndexedGroundingRows:
    def __init__(self, path, validate_split, *, index=None):
        self.path = Path(path).resolve()
        self.offsets = array('Q')
        if index is None:
            version, self.sha256 = self._build_index(validate_split)
        else:
            version, self.sha256, offsets = index
            self.offsets.frombytes(offsets)
        self.stream = self.path.open('rb')
        self.version = version
        if self._version(self.stream) != self.version:
            self.stream.close()
            raise ValueError("training file changed while building or sharing its index: " + str(self.path))

    @staticmethod
    def _version(stream):
        stat = os.fstat(stream.fileno())
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _build_index(self, validate_split):
        offset = 0
        digest = hashlib.sha256()
        with self.path.open('rb') as stream:
            version = self._version(stream)
            for line_number, line in enumerate(stream, 1):
                digest.update(line)
                if line.strip() and not line.lstrip().startswith(b"#"):
                    try:
                        row = normalize_record(json.loads(line), self.path.parent, verify_media=False)
                        _validate_grounding_row(row)
                        validate_split(row)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError("%s:%d: %s" % (self.path, line_number, error)) from error
                    self.offsets.append(offset)
                offset += len(line)
            if self._version(stream) != version:
                raise ValueError("training file changed during validation: " + str(self.path))
        self.offsets.append(offset)
        if len(self.offsets) == 1:
            raise ValueError('empty grounding dataset: ' + str(self.path))
        return version, digest.hexdigest()

    def share_index(self):
        return self.version, self.sha256, self.offsets.tobytes()

    def check_unchanged(self):
        if self._version(self.stream) != self.version:
            raise ValueError("indexed training file changed during training: " + str(self.path))

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        start, end = self.offsets[index], self.offsets[index + 1]
        line = os.pread(self.stream.fileno(), end - start, start).partition(b"\n")[0]
        return normalize_record(json.loads(line),
                                self.path.parent, verify_media=False)

    def __iter__(self):
        yield from read_records(self.path, verify_media=False)


def _shared_indexed_rows(path, validate_split, ctx):
    if ctx.world_size == 1:
        return _IndexedGroundingRows(path, validate_split)
    payload = rows = None
    if ctx.main:
        try:
            rows = _IndexedGroundingRows(path, validate_split)
        except (OSError, ValueError) as exc:
            payload = str(exc), None
        else:
            payload = None, rows.share_index()
    error, index = ctx.broadcast(payload)
    if error is not None:
        raise ValueError(error)
    return rows if ctx.main else _IndexedGroundingRows(path, validate_split, index=index)


def decision_retained(current, baseline, margin):
    """An improving text score must not hide regression in audio, image or video."""
    return all(mode in current['by_modality'] and current['by_modality'][mode]['accuracy'] >= values['accuracy'] - margin
               for mode, values in baseline['by_modality'].items())


def checkpoint_score(metrics, selection_metric="joint"):
    """Score the available geometric tasks and optionally blend in macro decision accuracy."""
    geometric = []
    if metrics.get('single_n', 1):
        geometric.append(metrics['acc_iou50'])
    if metrics.get('set_n', 1):
        geometric.append(metrics['set_f1_50'])
    geometry = sum(geometric) / len(geometric) if geometric else 0.

    decisions = metrics.get('decisions')
    if selection_metric in ('decision_macro', 'harmonic') and (not decisions or not decisions['by_modality']):
        raise ValueError(selection_metric + ' selection requires decision development data')
    if selection_metric not in ('joint', 'decision_macro', 'harmonic'):
        raise ValueError('unknown checkpoint selection metric')
    if not decisions:
        return geometry
    modalities = decisions['by_modality']
    macro_accuracy = sum(values['accuracy'] for values in modalities.values()) / len(modalities)
    if selection_metric == 'harmonic':
        if not geometric:
            raise ValueError('harmonic selection requires geometry development data')
        return 2 * geometry * macro_accuracy / (geometry + macro_accuracy) if geometry + macro_accuracy else 0.
    return macro_accuracy if selection_metric == 'decision_macro' else (geometry + macro_accuracy) / 2


def grounding_retained(current, baseline, margin):
    """Check retention on the geometric tasks actually present in development data."""
    keys = []
    if baseline.get('single_n', 1):
        keys.append('acc_iou50')
    if baseline.get('set_n', 1):
        keys.append('set_f1_50')
    return all(current[key] >= baseline[key] - margin for key in keys)


def _decision_development_metrics(rows, records, tagged_only):
    from .multimodal_train import example_hash, metrics
    result={'decisions':metrics(records)}
    if tagged_only:
        keys={example_hash(row,qid,qdef) for row in rows if row.get('retention_reference')
              for qid,qdef in row['questions'].items()}
        selected=[record for record in records if record['example_hash'] in keys]
        if not selected:raise ValueError('tagged retention requested without reference development decisions')
        result['retention_decisions']=metrics(selected)
    return result


def _decision_eval_shard(agent, rows, rank, world_size):
    """Keep requests for a shared clip together and balance the decoding work."""
    from .multimodal_train import _evaluation_video_key
    groups = {}
    for index, row in enumerate(rows):
        key = _evaluation_video_key(agent, row)
        groups.setdefault(index if key is None else key, []).append((index, row))

    def work(group):
        return sum(len(row["questions"]) * (3 if record_modality(row) == "video" else 1)
                   for _, row in group)

    shards = [[] for _ in range(world_size)]
    totals = [0] * world_size
    for group in sorted(groups.values(), key=work, reverse=True):
        worker = min(range(world_size), key=totals.__getitem__)
        shards[worker].extend(group)
        totals[worker] += work(group)
    return sorted(shards[rank])


def _decision_development_records(agent, rows, config, ctx, sharded):
    from .multimodal_train import collect_records
    if not sharded:
        return collect_records(agent, rows, config.decision_eval_micro_batch, max_micro_tokens=config.max_micro_tokens)
    shard = _decision_eval_shard(agent, rows, ctx.rank, ctx.world_size)
    records = collect_records(agent, [row for _, row in shard], 1, max_micro_tokens=config.max_micro_tokens)
    cursor = 0
    local = []
    for index, row in shard:
        next_cursor = cursor + len(row["questions"])
        local.append((index, records[cursor:next_cursor]))
        cursor = next_cursor
    gathered = [None] * ctx.world_size if ctx.main else None
    dist.gather_object(local, gathered, dst=0)
    if ctx.main:
        ordered = sorted(item for shard_records in gathered for item in shard_records)
        return [record for _, group in ordered for record in group]
    return None


def _development_retained(current,baseline,margin):
    return decision_retained(current.get('retention_decisions',current['decisions']),
        baseline.get('retention_decisions',baseline['decisions']),margin)


def _rank_window(window, rank, world_size, config):
    """Group compatible lengths and distribute estimated work within one optimizer window."""
    if not config.balance_ranks or world_size == 1:
        return window[rank::world_size]
    buckets = {}
    for index, row in enumerate(window):
        admission = row.get("training_admission")
        length = admission.get("tokens") if isinstance(admission, dict) else None
        if type(length) is not int or length < 1:
            return window[rank::world_size]
        modality = record_modality(row)
        task = row.get("grounding", {}).get("task")
        buckets.setdefault((modality, task), []).append((length, index))

    groups = []
    for (modality, task), candidates in buckets.items():
        limit = config.micro_batch if task is not None or config.batch_decisions else 1
        batch = []
        for length, index in sorted(candidates):
            if batch and (len(batch) >= limit or length * (len(batch) + 1) > config.max_micro_tokens):
                groups.append((modality, batch))
                batch = []
            batch.append((length, index))
        groups.append((modality, batch))

    def estimated_work(group):
        modality, batch = group
        length = batch[-1][0]
        # Estimate per-record preparation and sequence compute. The true
        # token limit is checked again after native processing on each rank.
        return len(batch) * (256 + (512 if modality == "video" else 0) + length + length**2 / 8192)

    assigned = [[] for _ in range(world_size)]
    load = [0.] * world_size
    for group in sorted(groups, key=estimated_work, reverse=True):
        worker = min(range(world_size), key=load.__getitem__)
        assigned[worker].extend(window[index] for _, index in group[1])
        load[worker] += estimated_work(group)
    return assigned[rank]


def _geometry_microbatches(agent, rows, config):
    """Batch same-head image requests while preserving every original target and its weight."""
    buckets = {}
    def pack(task, items):
        examples = [item[1] for item in items]
        inputs = examples[0] if len(examples) == 1 else collate_multimodal_inputs(
            examples, getattr(agent.tok, 'pad_token_id', 0) or 0)
        return [item[0] for item in items], inputs, task == 'multiple'
    for row in rows:
        if 'grounding' not in row:
            continue
        qid, qdef = next(iter(row['questions'].items()))
        request, question = record_input(agent, row, qid, qdef)
        inputs, _, _ = agent._prepare(request, question)
        task = row['grounding']['task']
        key = task, tuple(sorted(inputs))
        items = buckets.setdefault(key, [])
        length = inputs['input_ids'].shape[-1]
        longest = max([length] + [entry[1]['input_ids'].shape[-1] for entry in items])
        if items and (len(items) >= config.micro_batch or longest * (len(items) + 1) > config.max_micro_tokens):
            buckets[key] = []
            yield pack(task, items)
        buckets[key].append((row, inputs))
    for (task, _fields), items in buckets.items():
        if items:
            yield pack(task, items)


def _training_question(qdef, option_rng):
    if option_rng is None or not can_shuffle_choice(qdef):
        return qdef
    order = list(range(len(qdef["criteria"])))
    option_rng.shuffle(order)
    return {**qdef, "option_order": order}


def _decision_microbatches(agent, rows, config, *, option_rng=None):
    """Pack native decision inputs by media fields and the actual padded token budget."""
    buckets = {}

    def pack(items):
        inputs = items[0][1] if len(items) == 1 else collate_multimodal_inputs(
            [item[1] for item in items], getattr(agent.tok, 'pad_token_id', 0) or 0)
        return [item[0] for item in items], inputs

    if getattr(config, 'length_bucketing', False):
        rows = sorted(rows, key=lambda row: row.get('training_admission', {}).get('tokens', 0))
    with ThreadPoolExecutor(max_workers=1) if config.prefetch_video else nullcontext() as loader:
        videos = {}
        if loader is not None:
            for row in rows:
                state = row["state"]
                if (id(row) not in videos and 'grounding' not in row and not row.get('_image_assets')
                        and isinstance(state, dict) and state.get('video') is not None):
                    videos[id(row)] = loader.submit(agent._media, agent._input(state))
        for row in rows:
            if 'grounding' in row:
                continue
            qid, qdef = next(iter(row['questions'].items()))
            request, question = record_input(agent, row, qid, _training_question(qdef, option_rng))
            future = videos.get(id(row))
            inputs, q, _ = (agent._prepare(request, question, load_media=future.result) if future is not None
                            else agent._prepare(request, question))
            target = _decision_target(row, qid, q)
            input_fields = tuple(sorted(inputs))
            items = buckets.setdefault(input_fields, [])
            lengths = [inputs['input_ids'].shape[-1]] + [item[1]['input_ids'].shape[-1] for item in items]
            padded_tokens = max(lengths) * (len(items) + 1)
            if items and (len(items) >= config.micro_batch or padded_tokens > config.max_micro_tokens):
                buckets[input_fields] = []
                yield pack(items)
            buckets[input_fields].append(((row, qid, qdef, q, target), inputs))
        for items in buckets.values():
            if items:
                yield pack(items)


def _activation_context(inputs, config):
    threshold = config.activation_offload_min_tokens
    if threshold is None or not torch.is_grad_enabled() or inputs['input_ids'].numel() < threshold:
        return nullcontext()
    return torch.autograd.graph.save_on_cpu(pin_memory=inputs['input_ids'].device.type == 'cuda')


def _decision_batch_loss(agent,chunk,inputs,config):
    """Use the same per-example CE/RPS/Brier/KL objective as the scalar replay path."""
    from .common import QTYPES
    from .multimodal_objectives import decision_loss
    from .multimodal_train import example_hash
    width=max(len(item[4]) for item in chunk);raw=agent._logits(inputs,width);aligned=[];targets=[];references=[];accepted=[]
    for index,(row,qid,qdef,q,values) in enumerate(chunk):
        k=len(values);logits=raw[index,:k]
        if q.get('option_order') is not None:
            logits=logits[torch.argsort(torch.as_tensor(q['option_order'],device=agent.device))]
        aligned.append(F.pad(logits,(0,width-k),value=-1e4));targets.append(list(values)+[0.]*(width-k))
        reference=row.get('replay_reference');valid=False
        if reference is not None:
            if row.get('replay_reference_identity')!=example_hash(row,qid,qdef):raise ValueError('replay reference input/options changed')
            if len(reference)!=k:raise ValueError('replay reference option order/size mismatch')
            valid=int(max(range(k),key=lambda i:reference[i]))==int(max(range(k),key=lambda i:values[i])) and max(reference)>=.5
        references.append((list(reference) if reference is not None else [0.]*k)+[0.]*(width-k));accepted.append(valid)
    mask=torch.tensor([[i<len(item[4]) for i in range(width)] for item in chunk],device=agent.device)
    loss,_=decision_loss(torch.stack(aligned),torch.tensor(targets,device=agent.device,dtype=torch.float32),mask,
        torch.tensor([QTYPES[item[3]['t']] for item in chunk],device=agent.device),
        ordinal_weight=config.decision_ordinal_weight,brier_weight=config.decision_brier_weight,
        reference=torch.tensor(references,device=agent.device,dtype=torch.float32),
        reference_mask=torch.tensor(accepted,device=agent.device),kl_weight=config.replay_kl_weight)
    return loss


def _grounding_train_mode(agent, config):
    if config.mode == 'head':
        agent.model.eval()
        base = agent.model.get_base_model() if _is_adapter_model(agent.model) else agent.model
        base.grounding_head.train()
    else:
        # Transformer checkpointing is conditional on training=True. Freeze stochastic
        # vision/audio towers through the existing native training-mode helper.
        _train_mode(agent, config)


@torch.no_grad()
def grounding_metrics(agent, rows, limit=256, threshold=.25, *, micro_batch=1, max_micro_tokens=4096):
    """Held-out single-box accuracy and score-thresholded set precision/recall, not COCO AP."""
    if micro_batch < 1 or max_micro_tokens < 1:
        raise ValueError("grounding batch size and token budget must be positive")
    rng = random.Random(0)
    chosen = rng.sample(rows, min(len(rows), limit)) if limit else rows
    stats = {"single_n": 0, "single_iou": 0., "single_iou50": 0, "set_tp": 0, "set_pred": 0,
             "set_gold": 0, "set_n": 0, "presence_n": 0, "presence_correct": 0}
    agent.model.eval()
    for row in chosen:
        if "grounding" in row:
            continue
        qid, qdef = next(iter(row["questions"].items()))
        request, question = record_input(agent, row, qid, qdef)
        inputs, q, _ = agent._prepare(request, question)
        target = _decision_target(row, qid, q)
        logits = agent._logits(inputs, len(target))[0]
        order = q.get('option_order')
        if order is not None:
            logits = logits[torch.argsort(torch.as_tensor(order, device=agent.device))]
        selected = int(logits.argmax())
        stats["presence_n"] += 1
        stats["presence_correct"] += int(target[selected] == max(target))
    config = GroundingTrainConfig(micro_batch=micro_batch, max_micro_tokens=max_micro_tokens)
    for chunk, inputs, multiple in _geometry_microbatches(agent, chosen, config):
        out = agent._grounding_outputs(inputs, multiple=multiple)
        if not multiple:
            positive = [i for i, row in enumerate(chunk) if row["grounding"]["boxes"]]
            if positive:
                truth = validate_boxes([chunk[i]["grounding"]["boxes"][0] for i in positive]).to(agent.device)
                ious = box_iou(out["boxes"][positive, 0], truth).diag().tolist()
                stats["single_n"] += len(ious)
                stats["single_iou"] += sum(ious)
                stats["single_iou50"] += sum(iou >= .5 for iou in ious)
            continue
        for index, row in enumerate(chunk):
            spec = row["grounding"]
            truth = validate_boxes(spec["boxes"]).to(agent.device)
            boxes = out["boxes"][index]
            stats['set_n'] += 1
            scores = out["object_logits"][index].sigmoid() * out["quality_logits"][index].sigmoid()
            predicted = boxes[scores >= threshold]
            ignore = validate_boxes(spec.get("ignored", [])).to(agent.device)
            if len(ignore) and len(predicted):
                overlaps = crowd_overlap(predicted, ignore).max(-1).values >= .5
                if len(truth):overlaps &= box_iou(predicted, truth).max(-1).values < .5
                predicted = predicted[~overlaps]
            stats["set_pred"] += len(predicted); stats["set_gold"] += len(truth)
            if len(predicted) and len(truth):
                from scipy.optimize import linear_sum_assignment
                ious = box_iou(predicted, truth)
                reward = (ious >= .5).float() * (min(ious.shape) + 1) + ious
                a, b = linear_sum_assignment(-reward.cpu().numpy())
                stats["set_tp"] += int((ious[a, b] >= .5).sum())
    precision = stats["set_tp"] / max(1, stats["set_pred"])
    recall = stats["set_tp"] / max(1, stats["set_gold"])
    return {**stats, "miou": stats["single_iou"] / max(1, stats["single_n"]),
            "acc_iou50": stats["single_iou50"] / max(1, stats["single_n"]),
            "set_precision50": precision, "set_recall50": recall,
            "set_f1_50": 2 * precision * recall / max(1e-8, precision + recall),
            "presence_accuracy": stats["presence_correct"] / max(1, stats["presence_n"]), "evaluated": len(chosen)}


def _evaluate_development(agent, geometry_rows, decision_rows, config, ctx):
    sharded = (config.distributed_decision_eval and bool(decision_rows)
               and ctx.world_size > 1 and config.decision_eval_micro_batch == 1)
    if not ctx.main and not sharded:
        return None
    result = (grounding_metrics(agent, geometry_rows, config.eval_limit,
                               micro_batch=config.grounding_eval_micro_batch,
                               max_micro_tokens=config.max_micro_tokens) if ctx.main else None)
    if decision_rows:
        records = _decision_development_records(agent, decision_rows, config, ctx, sharded)
        if ctx.main:
            result.update(_decision_development_metrics(decision_rows, records, config.retention_tagged_only))
    return result


def _checkpoint_identity(agent, model, cfg, ctx, total, train_records, train_digest, dev, decisions, replay):
    origin = Path(model)
    backbone = Path(getattr(agent, "model_id", model))
    files = []
    if backbone.is_dir():
        for path in sorted(backbone.glob("*.safetensors")):
            stat = path.stat()
            files.append([path.name, stat.st_size, stat.st_mtime_ns])
    config = asdict(cfg)
    config.pop("checkpoint_every")
    return {"format": 1, "config": config, "world_size": ctx.world_size, "schedule_steps": total,
            "train_records": train_records,
            "origin": str(origin.resolve()) if origin.is_dir() else str(origin),
            "backbone": str(backbone), "backbone_files": files,
            "revision": getattr(agent, "revision", None), "dtype": str(agent.dtype),
            "readout": getattr(agent, "readout", "letter"), "prompt_version": getattr(agent, "prompt_version", "v1"),
            "execution": getattr(agent, "_training_variant", {}),
            "data": {"train": train_digest, "dev": file_fingerprint(str(dev)),
                     "decision_dev": file_fingerprint(str(decisions)) if decisions else None,
                     "replay": file_fingerprint(str(replay)) if replay else None}}


def finetune_grounding(train_data, dev_data, output, *, model="google/gemma-4-E2B-it", replay_data=None,
                       decision_dev_data=None, config=None, device=None, agent=None, row_validator=None,
                       resume_from=None):
    """Train the attached heads and select checkpoints on development data."""
    cfg = config or GroundingTrainConfig()
    cfg.validate()
    ctx = distributed_context()
    if cfg.selection_metric in ('decision_macro', 'harmonic') and not decision_dev_data:
        raise ValueError(cfg.selection_metric + ' selection requires decision_dev_data')
    torch.manual_seed(cfg.seed)
    dev = load_grounding_rows(dev_data)
    from .multimodal_train import load_rows
    decision_dev = load_rows(decision_dev_data) if decision_dev_data else []
    dev_groups = {row["group"] for row in dev if row.get("group") is not None}
    decision_groups = set()
    for row in decision_dev:
        if row_validator is not None:
            row_validator("decision_dev", row)
        group = row.get("group")
        if group is not None:
            source = None if str(group).startswith(("toolace:", "coco:")) else row.get("source", row.get("dataset"))
            decision_groups.add((source, group))

    def check_single_box(row):
        if cfg.slots == 1 and row.get("grounding", {}).get("task") == "multiple":
            raise ValueError("single-box training requires a single-target view; never relabel multi-object annotations as one box")

    def validate_split(row):
        check_single_box(row)
        if row_validator is not None:
            row_validator("train", row)
        group = row.get("group")
        if group in dev_groups:
            raise ValueError("grounding train/dev image groups overlap")
        source = None if str(group).startswith(("toolace:", "coco:")) else row.get("source", row.get("dataset"))
        if (source, group) in decision_groups:
            raise ValueError("grounding train/decision development groups overlap: " + str(group))

    for row in dev:
        check_single_box(row)
        if row_validator is not None:
            row_validator("dev", row)
    # Share the validated index instead of parsing every JSONL record on every rank.
    indexed = ctx.broadcast(Path(train_data).stat().st_size >= 128 * 1024**2 if ctx.main else None)
    if indexed:
        rows = _shared_indexed_rows(train_data, validate_split, ctx)
    else:
        rows = load_grounding_rows(train_data)
        for row in rows:
            validate_split(row)
    agent = agent or MultimodalAgent(model, device=ctx.device if ctx.world_size > 1 else device,
                                     grounding_slots=cfg.slots, max_len=cfg.max_len)
    agent.max_len = cfg.max_len
    agent._video_cache_dir = cfg.video_cache_dir
    agent._video_cache_size = cfg.video_cache_size
    base = agent.model.get_base_model() if _is_adapter_model(agent.model) else agent.model
    if not hasattr(base, "grounding_head"):
        raise ValueError("grounding architecture must be installed before training")
    if cfg.slots == 1 and base.grounding_head.architecture != 'single-v1':
        base.grounding_head = base.grounding_head.to_single()
        agent.detection_trained = False
    if base.grounding_head.slots != cfg.slots:
        raise ValueError('training slots differ from the installed grounding architecture')
    if cfg.mode == "head":
        agent.model.requires_grad_(False)
        readout = base.decision_scorer if getattr(agent, 'readout', 'letter') == 'candidate' else _active_head(agent.model)
        readout.float().requires_grad_(cfg.train_readout_in_warmup)
    else:
        configure_training(agent, MultimodalTrainConfig(mode=cfg.mode, lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha, lora_targets=cfg.lora_targets, train_projectors=agent.model_type == 'gemma4'))
        base = agent.model.get_base_model() if _is_adapter_model(agent.model) else agent.model
        if agent.model_type.startswith('qwen') and cfg.mode == 'lora':
            mergers = [name for name, _ in base.named_modules() if name.endswith('visual.merger')]
            for name in mergers:
                base.get_submodule(name).float().requires_grad_(True)
            agent.native_trainable_modules = sorted(set(getattr(agent, 'native_trainable_modules', [])) | set(mergers))
    base.grounding_head.float().requires_grad_(True)
    params = [p for p in agent.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optimizer_groups(agent.model, cfg.learning_rate,
        cfg.head_learning_rate, cfg.projector_learning_rate), fused=True if cfg.fused_adamw else None)
    optimizer_summary = [{key: value for key, value in group.items() if key != 'params'} |
        {'parameters': sum(parameter.numel() for parameter in group['params'])} for group in optimizer.param_groups]
    ctx.synchronize_parameters(params)
    replay = []
    if replay_data:
        replay = load_rows(replay_data)
        replay = [r for r in replay if r.get("group") not in dev_groups]
        if agent.model_type not in {"gemma4", "qwen2_5_omni", "qwen2_5_omni_thinker", "gemma3n"}:
            replay = [r for r in replay if not (isinstance(r["state"], dict) and r["state"].get("audio") is not None)]
    rng = random.Random(cfg.seed)
    option_rng = random.Random(cfg.seed + 1009 * (ctx.rank + 1)) if cfg.shuffle_options else None
    order = list(range(len(rows)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    total = min(cfg.max_steps or 10**12, math.ceil(len(rows) / cfg.global_batch_size) * cfg.epochs)
    steps, best, history = 0, -1., []
    first_epoch, first_offset = 0, 0
    forward_counts = {'geometry': 0, 'decisions': 0}
    execution = {}
    train_decisions = cfg.mode != 'head' or cfg.train_readout_in_warmup
    identity = None
    if cfg.checkpoint_every or resume_from is not None:
        train_digest = rows.sha256 if indexed else file_fingerprint(str(train_data))
        identity = _checkpoint_identity(agent, model, cfg, ctx, total, len(rows), train_digest,
                                        dev_data, decision_dev_data, replay_data)
    if resume_from is None:
        before = _evaluate_development(agent, dev, decision_dev, cfg, ctx)
        if ctx.main:
            agent.save(output / 'initial')
            if cfg.include_initial_selection:
                best = checkpoint_score(before, cfg.selection_metric)
                agent.save(output / 'best')
    else:
        progress, order = load_checkpoint(resume_from, output, agent, ctx, optimizer, identity, rng, option_rng)
        steps, first_epoch, first_offset = progress["steps"], progress["epoch"], progress["next_offset"]
        best, history, before = progress["best"], progress["history"], progress["before"]
        forward_counts, execution = progress["forward_counts"], progress["execution"]
        if ctx.main:
            print("Resumed joint training at optimizer step %d" % steps, flush=True)
    starting_step = steps
    if ctx.main:
        (output / "before.json").write_text(json.dumps(before, indent=2) + "\n")
    ctx.barrier()
    start = time.perf_counter()
    for epoch in range(first_epoch, cfg.epochs):
        if steps >= total:
            break
        if indexed:
            rows.check_unchanged()
        if resume_from is None or epoch != first_epoch:
            rng.shuffle(order)
        offset_start = first_offset if epoch == first_epoch else 0
        for offset in range(offset_start, len(order), cfg.global_batch_size):
            window = [rows[i] for i in order[offset:offset + cfg.global_batch_size]]
            window_size = len(window)
            if replay:
                replay_count = int(window_size * cfg.replay_fraction)
                if replay_count:
                    window[-replay_count:] = rng.choices(replay, k=replay_count)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            positive_targets = [0, 0]  # Single-box and set targets, in that order.
            geometry_forwards, decision_forwards = 0, 0
            _grounding_train_mode(agent, cfg)
            if not execution:
                execution = {'training_mode_during_updates': agent.model.training,
                    'checkpointed_training_modules': sum(bool(getattr(module, 'gradient_checkpointing', False)
                        or getattr(module, 'activation_checkpointing', False)) and module.training
                        for module in agent.model.modules())}
            local_rows = _rank_window(window, ctx.rank, ctx.world_size, cfg)
            # Each rank scales by the same optimizer window before gradients are summed.
            for chunk, inputs, multiple in _geometry_microbatches(agent, local_rows, cfg):
                with _activation_context(inputs, cfg):
                    result = agent._grounding_outputs(inputs, multiple=multiple)
                    geometry_forwards += 1
                    loss, _ = grounding_loss(result, [row['grounding']['boxes'] for row in chunk], multiple=multiple,
                        ignored=[row['grounding'].get('ignored', []) for row in chunk],
                        quality_matched_only=cfg.quality_matched_only)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite grounding loss")
                (loss * len(chunk) / window_size).backward()
                loss_sum += float(loss.detach()) * len(chunk)
                positive_targets[int(multiple)] += sum(bool(row['grounding']['boxes']) for row in chunk)
            for row in local_rows:
                if cfg.batch_decisions or not train_decisions or 'grounding' in row:
                    continue
                qid, qdef = next(iter(row["questions"].items()))
                request, question = record_input(agent, row, qid, _training_question(qdef, option_rng))
                inputs, q, _ = agent._prepare(request, question)
                values = _decision_target(row, qid, q)
                with _activation_context(inputs, cfg):
                    logits = agent._logits(inputs, len(values))
                    decision_forwards += 1
                if q.get('option_order') is not None:
                    option_order = torch.as_tensor(q['option_order'], device=agent.device)
                    logits = logits[:, torch.argsort(option_order)]
                target = torch.tensor(values, device=agent.device)
                loss = -(target * F.log_softmax(logits[0].float(), -1)).sum()
                if cfg.decision_ordinal_weight or cfg.decision_brier_weight:
                    from .common import QTYPES
                    from .multimodal_objectives import decision_loss
                    loss,_=decision_loss(logits,target.unsqueeze(0),torch.ones_like(logits,dtype=torch.bool),
                        torch.tensor([QTYPES[q['t']]],device=agent.device),objective='soft-ce',
                        ordinal_weight=cfg.decision_ordinal_weight,brier_weight=cfg.decision_brier_weight)
                if row.get('replay_reference') is not None and cfg.replay_kl_weight:
                    from .multimodal_train import example_hash
                    if row.get('replay_reference_identity') != example_hash(row, qid, qdef):
                        raise ValueError('replay reference input/options changed')
                    reference = torch.tensor(row['replay_reference'], device=agent.device, dtype=torch.float32)
                    if reference.shape != target.shape:raise ValueError('replay reference option order/size mismatch')
                    if int(reference.argmax()) == int(target.argmax()) and float(reference.max()) >= .5:
                        loss = loss + cfg.replay_kl_weight * F.kl_div(F.log_softmax(logits[0].float(), -1), reference, reduction='sum')
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite grounding loss")
                (loss / window_size).backward()
                loss_sum += float(loss.detach())
            if cfg.batch_decisions and train_decisions:
                for chunk, inputs in _decision_microbatches(agent, local_rows, cfg, option_rng=option_rng):
                    with _activation_context(inputs, cfg):
                        loss = _decision_batch_loss(agent, chunk, inputs, cfg)
                        decision_forwards += 1
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite decision micro-batch loss')
                    (loss * len(chunk) / window_size).backward()
                    loss_sum += float(loss.detach()) * len(chunk)
            ctx.sum_gradients(params)
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            factor = learning_rate_factor(steps + 1, total, cfg.warmup_ratio, cfg.min_lr_ratio)
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            optimizer.step()
            steps += 1
            reduced = ctx.sum_values([loss_sum, *positive_targets, geometry_forwards, decision_forwards], agent.device)
            total_loss, single_targets, set_targets, geometry_calls, decision_calls = reduced
            forward_counts['geometry'] += int(geometry_calls)
            forward_counts['decisions'] += int(decision_calls)
            agent.grounding_trained |= bool(single_targets)
            agent.detection_trained |= bool(set_targets)
            agent.calibrated = False; agent.modality_calibration = {}; agent.temperature = [1.] * 3; agent.temperature_by_options = {}
            agent.ordinal_calibration = {}
            if ctx.main and (steps == starting_step + 1 or steps % 10 == 0):
                print('grounding step=%d loss=%.5f elapsed=%.1fs ranks=%d' %
                      (steps, total_loss / window_size, time.perf_counter() - start, ctx.world_size), flush=True)
            evaluated = cfg.eval_every and steps % cfg.eval_every == 0
            if evaluated:
                metrics = _evaluate_development(agent, dev, decision_dev, cfg, ctx)
                if ctx.main:
                    record = dict(step=steps, **metrics)
                    history.append(record)
                    score = checkpoint_score(metrics, cfg.selection_metric)
                    decision_ok = True
                    if decision_dev:
                        decision_ok = _development_retained(metrics, before, cfg.max_decision_regression)
                        record['decision_retention_passed'] = decision_ok
                    geometry_ok = cfg.max_grounding_regression is None or grounding_retained(
                        metrics, before, cfg.max_grounding_regression)
                    record['grounding_retention_passed'] = geometry_ok
                    print('GROUNDING DEV', json.dumps(record), flush=True)
                    if decision_ok and geometry_ok and score > best:
                        best = score
                        agent.save(output / 'best')
                ctx.barrier()
            if cfg.checkpoint_every and (steps % cfg.checkpoint_every == 0 or steps == total or evaluated):
                if indexed:
                    rows.check_unchanged()
                progress = {"identity": identity, "steps": steps, "epoch": epoch, "next_offset": offset + window_size,
                            "best": best, "history": history, "before": before, "forward_counts": forward_counts,
                            "execution": execution, "grounding_trained": agent.grounding_trained,
                            "detection_trained": agent.detection_trained}
                save_checkpoint(output, agent, ctx, optimizer, progress, order, rng, option_rng)
            if steps >= total:
                break
        if steps >= total:
            break
    after = _evaluate_development(agent, dev, decision_dev, cfg, ctx)
    if ctx.main:
        agent.save(output / 'last')
        decision_ok = not decision_dev or _development_retained(after, before, cfg.max_decision_regression)
        geometry_ok = cfg.max_grounding_regression is None or grounding_retained(
            after, before, cfg.max_grounding_regression)
        accepted = decision_ok and geometry_ok
        final_score = checkpoint_score(after, cfg.selection_metric)
        if accepted and final_score > best:
            best = final_score
            agent.save(output / 'best')
        if indexed:
            rows.check_unchanged()
        train_sha256 = identity["data"]["train"] if identity is not None else (
            rows.sha256 if indexed else file_fingerprint(str(train_data)))
        report = {'status': 'complete', 'steps': steps, 'config': asdict(cfg), 'train_records': len(rows),
            'dev_records': len(dev), 'before': before, 'after': after, 'selection_history': history,
            'world_size': ctx.world_size, 'train_sha256': train_sha256,
            'training_data_access': 'indexed_jsonl' if isinstance(rows, _IndexedGroundingRows) else 'in_memory',
            'forward_passes': forward_counts, 'execution': execution, 'optimizer_groups': optimizer_summary,
            'checkpoint': str(output / ('best' if best >= 0 or accepted else 'initial')),
            'final_decision_retention_passed': decision_ok,'final_grounding_retention_passed':geometry_ok,
            'note': 'Development selection only; native benchmark results are separate. Failed decision or geometric retention falls back to the unchanged initial checkpoint.'}
        if cfg.checkpoint_every or resume_from is not None:
            report["resume"] = {"starting_step": starting_step,
                                "latest": str(output / "checkpoint_latest.json") if cfg.checkpoint_every else None}
        (output / 'training_report.json').write_text(json.dumps(report, indent=2) + '\n')
    ctx.barrier()
    return agent
