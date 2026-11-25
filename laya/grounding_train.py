"""Staged geometric/set supervision with ordinary decision replay on one local VLM."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
from array import array
from itertools import chain
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .evals import file_fingerprint
from .grounding import box_iou, crowd_overlap, grounding_loss, validate_boxes
from .multimodal import MultimodalAgent, _active_head, _is_adapter_model
from .multimodal_distributed import distributed_context
from .multimodal_train import MultimodalTrainConfig, configure_training, collate_multimodal_inputs, _train_mode
from .multimodal_objectives import learning_rate_factor, optimizer_groups
from .train import target_from_gold, target_from_expected


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

    def validate(self):
        if self.mode not in {"head", "lora", "full"}:
            raise ValueError("grounding mode must be head, lora or full")
        if self.lora_targets not in {'qv', 'attention', 'all-linear'}:
            raise ValueError('invalid grounding LoRA target scope')
        for key in ["epochs", "global_batch_size", "micro_batch", "max_micro_tokens", "slots", "max_len", "eval_limit", "lora_rank", "lora_alpha"]:
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
        if self.selection_metric not in {'joint','decision_macro'} or not isinstance(self.include_initial_selection,bool) or not isinstance(self.quality_matched_only,bool):
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
    if "grounding" in row:
        if qdef['type'] != 'box':
            raise ValueError("geometric supervision requires a Box question")
        truth = row["grounding"]
        validate_boxes(truth["boxes"])
        validate_boxes(truth.get("ignored", []))
        if truth["task"] not in {"single", "multiple"}:
            raise ValueError("unknown geometric supervision task")
        if truth["task"] == "single" and len(truth["boxes"]) > 1:
            raise ValueError("an official Box target contains at most one box")


def load_grounding_rows(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            _validate_grounding_row(row)
            rows.append(row)
    if not rows:
        raise ValueError("empty grounding dataset: " + str(path))
    return rows


class _IndexedGroundingRows:
    def __init__(self, path):
        self.path = Path(path)
        self.offsets = array('Q', [0])
        with self.path.open('rb') as stream:
            for line in stream:
                _validate_grounding_row(json.loads(line))
                self.offsets.append(self.offsets[-1] + len(line))
        if len(self.offsets) == 1:
            raise ValueError('empty grounding dataset: ' + str(path))
        self.stream = self.path.open('rb')

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        start, end = self.offsets[index], self.offsets[index + 1]
        return json.loads(os.pread(self.stream.fileno(), end - start, start))

    def __iter__(self):
        with self.path.open('rb') as stream:
            for line in stream: yield json.loads(line)


def decision_retained(current, baseline, margin):
    """An improving text score must not hide regression in audio, image or video."""
    return all(mode in current['by_modality'] and current['by_modality'][mode]['accuracy'] >= values['accuracy'] - margin
               for mode, values in baseline['by_modality'].items())


def checkpoint_score(metrics, selection_metric="joint"):
    """Keep decision adaptation selection separate from geometric adaptation selection."""
    geometric=[]
    if metrics.get('single_n',1):geometric.append(metrics['acc_iou50'])
    if metrics.get('set_n',1):geometric.append(metrics['set_f1_50'])
    geometry=sum(geometric)/len(geometric) if geometric else 0.
    decisions=metrics.get('decisions')
    if selection_metric=='decision_macro':
        if not decisions or not decisions['by_modality']:
            raise ValueError('decision_macro selection requires decision development data')
        return sum(v['accuracy'] for v in decisions['by_modality'].values())/len(decisions['by_modality'])
    if selection_metric!='joint':raise ValueError('unknown checkpoint selection metric')
    if decisions:
        macro=sum(v['accuracy'] for v in decisions['by_modality'].values())/len(decisions['by_modality'])
        return (geometry+macro)/2
    return geometry


def grounding_retained(current,baseline,margin):
    """Check retention on the geometric tasks actually present in development data."""
    keys=[]
    if baseline.get('single_n',1):keys.append('acc_iou50')
    if baseline.get('set_n',1):keys.append('set_f1_50')
    return all(current[key]>=baseline[key]-margin for key in keys)


def _decision_development(agent,rows,tagged_only=False):
    from .multimodal_train import collect_records,example_hash,metrics
    records=collect_records(agent,rows);result={'decisions':metrics(records)}
    if tagged_only:
        keys={example_hash(row,qid,qdef) for row in rows if row.get('retention_reference')
              for qid,qdef in row['questions'].items()}
        selected=[record for record in records if record['example_hash'] in keys]
        if not selected:raise ValueError('tagged retention requested without reference development decisions')
        result['retention_decisions']=metrics(selected)
    return result


def _development_retained(current,baseline,margin):
    return decision_retained(current.get('retention_decisions',current['decisions']),
        baseline.get('retention_decisions',baseline['decisions']),margin)


def _geometry_microbatches(agent, rows, config):
    """Batch same-head image requests while preserving every original target and its weight."""
    buckets = {'single': [], 'multiple': []}
    def pack(task, items):
        examples = [item[1] for item in items]
        inputs = examples[0] if len(examples) == 1 else collate_multimodal_inputs(
            examples, getattr(agent.tok, 'pad_token_id', 0) or 0)
        return [item[0] for item in items], inputs, task == 'multiple'
    for row in rows:
        if 'grounding' not in row:
            continue
        qdef = next(iter(row['questions'].values()))
        inputs, _, _ = agent._prepare(agent._input(row['state']), qdef)
        task = row['grounding']['task'];items = buckets[task]
        length = inputs['input_ids'].shape[-1]
        longest = max([length] + [entry[1]['input_ids'].shape[-1] for entry in items])
        if items and (len(items) >= config.micro_batch or longest * (len(items) + 1) > config.max_micro_tokens):
            buckets[task] = []
            yield pack(task, items)
        buckets[task].append((row, inputs))
    for task, items in buckets.items():
        if items:
            yield pack(task, items)


def _decision_microbatches(agent,rows,config):
    """Pack native decision inputs by media fields and the actual padded token budget."""
    buckets={}
    def pack(items):
        inputs=items[0][1] if len(items)==1 else collate_multimodal_inputs(
            [item[1] for item in items],getattr(agent.tok,'pad_token_id',0) or 0)
        return [item[0] for item in items],inputs
    if getattr(config,'length_bucketing',False):
        rows=sorted(rows,key=lambda row:row.get('training_admission',{}).get('tokens',0))
    for row in rows:
        if 'grounding' in row:continue
        qid,qdef=next(iter(row['questions'].items()))
        inputs,q,_=agent._prepare(agent._input(row['state']),qdef)
        values=target_from_gold(q,row['gold'][qid]) if 'gold' in row else target_from_expected(q,row['expected'][qid])
        key=tuple(sorted(inputs));items=buckets.setdefault(key,[]);length=inputs['input_ids'].shape[-1]
        longest=max([length]+[item[1]['input_ids'].shape[-1] for item in items])
        if items and (len(items)>=config.micro_batch or longest*(len(items)+1)>config.max_micro_tokens):
            buckets[key]=[];yield pack(items)
        buckets[key].append(((row,qid,qdef,q,values),inputs))
    for items in buckets.values():
        if items:yield pack(items)


def _activation_context(inputs, config):
    threshold = config.activation_offload_min_tokens
    if threshold is None or not torch.is_grad_enabled() or inputs['input_ids'].shape[-1] < threshold:
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
def grounding_metrics(agent, rows, limit=256, threshold=.25):
    """Held-out single-box accuracy and score-thresholded set precision/recall, not COCO AP."""
    rng = random.Random(0)
    chosen = rng.sample(rows, min(len(rows), limit)) if limit else rows
    stats = {"single_n": 0, "single_iou": 0., "single_iou50": 0, "set_tp": 0, "set_pred": 0,
             "set_gold": 0, "set_n": 0, "presence_n": 0, "presence_correct": 0}
    agent.model.eval()
    for row in chosen:
        qid, qdef = next(iter(row["questions"].items()))
        inputs, q, _ = agent._prepare(agent._input(row["state"]), qdef)
        if "grounding" not in row:
            y = target_from_gold(q, row["gold"][qid]) if "gold" in row else target_from_expected(q, row["expected"][qid])
            probabilities = agent._logits(inputs, len(y))[0].softmax(-1)
            if q.get('option_order') is not None:
                probabilities = probabilities[torch.argsort(torch.as_tensor(q['option_order'],device=agent.device))]
            stats["presence_n"] += 1
            stats["presence_correct"] += int(int(probabilities.argmax()) == max(range(len(y)), key=lambda i: y[i]))
            continue
        spec = row["grounding"]; multiple = spec["task"] == "multiple"
        out = agent._grounding_outputs(inputs, multiple=multiple)
        truth = validate_boxes(spec["boxes"]).to(agent.device)
        boxes = out["boxes"][0]
        if not multiple:
            if len(truth):
                iou = float(box_iou(boxes, truth)[0, 0])
                stats["single_n"] += 1; stats["single_iou"] += iou; stats["single_iou50"] += int(iou >= .5)
        else:
            stats['set_n'] += 1
            scores = out["object_logits"][0].sigmoid() * out["quality_logits"][0].sigmoid()
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


def finetune_grounding(train_data, dev_data, output, *, model="google/gemma-4-E2B-it", replay_data=None,
                       decision_dev_data=None, config=None, device=None, agent=None):
    """Train the attached heads/backbone; benchmarks are evaluated by a separate command."""
    cfg = config or GroundingTrainConfig(); cfg.validate(); ctx = distributed_context()
    if cfg.selection_metric=='decision_macro' and not decision_dev_data:
        raise ValueError('decision_macro selection requires decision_dev_data')
    torch.manual_seed(cfg.seed)
    # Keep million-row corpora off the Python heap on each independent GPU rank.
    # The index is eight bytes per row; full schema validation still runs first.
    rows = _IndexedGroundingRows(train_data) if Path(train_data).stat().st_size >= 128 * 1024**2 else load_grounding_rows(train_data)
    dev = load_grounding_rows(dev_data)
    if cfg.slots == 1 and any(row.get('grounding',{}).get('task')=='multiple' for row in chain(rows, dev)):
        raise ValueError('single-box training requires a single-target view; never relabel multi-object annotations as one box')
    dev_groups = {row["group"] for row in dev}
    if any(row["group"] in dev_groups for row in rows):
        raise ValueError("grounding train/dev image groups overlap")
    agent = agent or MultimodalAgent(model, device=ctx.device if ctx.world_size > 1 else device,
                                     grounding_slots=cfg.slots, max_len=cfg.max_len)
    agent.max_len = cfg.max_len
    agent._video_cache_dir = cfg.video_cache_dir or '.cache/multimodal-video'
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
        _active_head(agent.model).float().requires_grad_(cfg.train_readout_in_warmup)
        if hasattr(base, "decision_scorer"):base.decision_scorer.float().requires_grad_(cfg.train_readout_in_warmup)
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
        cfg.head_learning_rate, cfg.projector_learning_rate))
    optimizer_summary = [{key: value for key, value in group.items() if key != 'params'} |
        {'parameters': sum(parameter.numel() for parameter in group['params'])} for group in optimizer.param_groups]
    ctx.synchronize_parameters(params)
    replay = []
    decision_dev = []
    if decision_dev_data:
        from .multimodal_train import load_rows
        decision_dev = load_rows(decision_dev_data)
    if replay_data:
        from .multimodal_train import load_rows
        replay = load_rows(replay_data)
        replay = [r for r in replay if r.get("group") not in dev_groups]
        if agent.model_type not in {"gemma4", "qwen2_5_omni", "qwen2_5_omni_thinker", "gemma3n"}:
            replay = [r for r in replay if not (isinstance(r["state"], dict) and r["state"].get("audio") is not None)]
    rng = random.Random(cfg.seed); order = list(range(len(rows))); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    total = min(cfg.max_steps or 10**12, math.ceil(len(rows) / cfg.global_batch_size) * cfg.epochs)
    steps, best, history = 0, -1., []
    forward_counts = {'geometry': 0, 'decisions': 0}; execution = {}
    if ctx.main:
        before = grounding_metrics(agent, dev, cfg.eval_limit)
        if decision_dev:before.update(_decision_development(agent,decision_dev,cfg.retention_tagged_only))
        (output / "before.json").write_text(json.dumps(before, indent=2) + "\n")
        agent.save(output / 'initial')
        if cfg.include_initial_selection:
            best=checkpoint_score(before,cfg.selection_metric);agent.save(output/'best')
    ctx.barrier(); start = time.perf_counter()
    for epoch in range(cfg.epochs):
        rng.shuffle(order)
        for offset in range(0, len(order), cfg.global_batch_size):
            window = [rows[i] for i in order[offset:offset + cfg.global_batch_size]]
            if replay:
                count = int(len(window) * cfg.replay_fraction)
                if count:window[-count:] = rng.choices(replay, k=count)
            optimizer.zero_grad(set_to_none=True); loss_sum = 0.; flags = [0, 0]; calls = [0, 0]
            _grounding_train_mode(agent, cfg)
            if not execution:
                execution = {'training_mode_during_updates': agent.model.training,
                    'checkpointed_training_modules': sum(bool(getattr(module, 'gradient_checkpointing', False)
                        or getattr(module, 'activation_checkpointing', False)) and module.training
                        for module in agent.model.modules())}
            local_rows = window[ctx.rank::ctx.world_size]
            for chunk, inputs, multiple in _geometry_microbatches(agent, local_rows, cfg):
                with _activation_context(inputs, cfg):
                    result = agent._grounding_outputs(inputs, multiple=multiple); calls[0] += 1
                    loss, _ = grounding_loss(result, [row['grounding']['boxes'] for row in chunk], multiple=multiple,
                        ignored=[row['grounding'].get('ignored', []) for row in chunk],quality_matched_only=cfg.quality_matched_only)
                if not torch.isfinite(loss):raise RuntimeError("nonfinite grounding loss")
                (loss * len(chunk) / len(window)).backward();loss_sum += float(loss.detach()) * len(chunk)
                flags[int(multiple)] += sum(bool(row['grounding']['boxes']) for row in chunk)
            for row in local_rows:
                if 'grounding' in row:
                    continue
                if cfg.batch_decisions:continue
                if cfg.mode == 'head' and not cfg.train_readout_in_warmup and 'grounding' not in row:
                    continue
                qid, qdef = next(iter(row["questions"].items()))
                inputs, q, _ = agent._prepare(agent._input(row["state"]), qdef)
                values = target_from_gold(q, row["gold"][qid]) if "gold" in row else target_from_expected(q, row["expected"][qid])
                with _activation_context(inputs, cfg):
                    logits = agent._logits(inputs, len(values)); calls[1] += 1
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
                if not torch.isfinite(loss):raise RuntimeError("nonfinite grounding loss")
                (loss / len(window)).backward(); loss_sum += float(loss.detach())
            if cfg.batch_decisions and (cfg.mode!='head' or cfg.train_readout_in_warmup):
                for chunk,inputs in _decision_microbatches(agent,local_rows,cfg):
                    with _activation_context(inputs, cfg):
                        loss=_decision_batch_loss(agent,chunk,inputs,cfg);calls[1]+=1
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite decision micro-batch loss')
                    (loss*len(chunk)/len(window)).backward();loss_sum+=float(loss.detach())*len(chunk)
            ctx.sum_gradients(params); torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            factor = learning_rate_factor(steps + 1, total, cfg.warmup_ratio, cfg.min_lr_ratio)
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            optimizer.step(); steps += 1
            reduced = ctx.sum_values([loss_sum, *flags, *calls], agent.device)
            forward_counts['geometry'] += int(reduced[3]); forward_counts['decisions'] += int(reduced[4])
            agent.grounding_trained |= bool(reduced[1]); agent.detection_trained |= bool(reduced[2])
            agent.calibrated = False; agent.modality_calibration = {}; agent.temperature = [1.] * 3; agent.temperature_by_options = {}
            agent.ordinal_calibration = {}
            if ctx.main and (steps == 1 or steps % 10 == 0):
                print('grounding step=%d loss=%.5f elapsed=%.1fs ranks=%d' % (steps, reduced[0] / len(window), time.perf_counter()-start, ctx.world_size), flush=True)
            if cfg.eval_every and steps % cfg.eval_every == 0:
                if ctx.main:
                    metrics = grounding_metrics(agent, dev, cfg.eval_limit)
                    if decision_dev:
                        metrics.update(_decision_development(agent,decision_dev,cfg.retention_tagged_only))
                    history.append(dict(step=steps, **metrics)); score = checkpoint_score(metrics,cfg.selection_metric)
                    decision_ok = True
                    if decision_dev:
                        decision_ok = _development_retained(metrics,before,cfg.max_decision_regression)
                        history[-1]['decision_retention_passed'] = decision_ok
                    geometry_ok = cfg.max_grounding_regression is None or grounding_retained(metrics,before,cfg.max_grounding_regression)
                    history[-1]['grounding_retention_passed']=geometry_ok
                    retained=decision_ok and geometry_ok
                    print('GROUNDING DEV', json.dumps(history[-1]), flush=True)
                    if retained and score > best:best = score; agent.save(output / 'best')
                ctx.barrier()
            if steps >= total:break
        if steps >= total:break
    if ctx.main:
        after = grounding_metrics(agent, dev, cfg.eval_limit)
        if decision_dev:
            after.update(_decision_development(agent,decision_dev,cfg.retention_tagged_only))
        agent.save(output / 'last')
        decision_ok = not decision_dev or _development_retained(after,before,cfg.max_decision_regression)
        geometry_ok = cfg.max_grounding_regression is None or grounding_retained(after,before,cfg.max_grounding_regression)
        accepted=decision_ok and geometry_ok
        final_score = checkpoint_score(after,cfg.selection_metric)
        if accepted and final_score > best:
            best = final_score;agent.save(output / 'best')
        train_sha256 = file_fingerprint(str(train_data))
        report = {'status': 'complete', 'steps': steps, 'config': asdict(cfg), 'train_records': len(rows),
            'dev_records': len(dev), 'before': before, 'after': after, 'selection_history': history,
            'world_size': ctx.world_size, 'train_sha256': train_sha256,
            'training_data_access': 'indexed_jsonl' if isinstance(rows, _IndexedGroundingRows) else 'in_memory',
            'forward_passes': forward_counts, 'execution': execution, 'optimizer_groups': optimizer_summary,
            'checkpoint': str(output / ('best' if best >= 0 or accepted else 'initial')),
            'final_decision_retention_passed': decision_ok,'final_grounding_retention_passed':geometry_ok,
            'note': 'Development selection only; native benchmark results are separate. Failed decision or geometric retention falls back to the unchanged initial checkpoint.'}
        (output / 'training_report.json').write_text(json.dumps(report, indent=2) + '\n')
    ctx.barrier()
    return agent
