"""Decision losses, LoRA setup and optimizer policy for joint training."""
from __future__ import annotations

import math
from typing import Dict

import torch

from ..decisions.questions import QTYPES


def decision_loss(logits, target, mask, qtypes, *, ordinal_weight=0.0, brier_weight=0.0,
                  reference=None, reference_mask=None, kl_weight=0.0, collect_components=True):
    """Return the loss and optional scalar components for one microbatch."""
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e4).float(), -1)
    if ordinal_weight or brier_weight:
        p = logp.exp() * mask
    ce = -(target * logp).sum(-1).mean()
    loss = ce
    parts = {"ce": float(ce.detach())} if collect_components else {}
    if ordinal_weight:
        selected = qtypes == QTYPES["score"]
        widths = mask.sum(-1)
        boundaries = torch.arange(logits.shape[-1], device=logits.device)[None, :] < (widths - 1)[:, None]
        rps = ((p.cumsum(-1) - target.cumsum(-1)).square() * boundaries).sum(-1) / (widths - 1).clamp_min(1)
        # Use the full batch denominator, so token-budget micro-batch splitting preserves
        # the objective and does not amplify a rare ordinal example.
        ordinal = (rps * selected).mean()
        loss = loss + ordinal_weight * ordinal
        if collect_components:
            parts["ordinal_rps"] = float(ordinal.detach())
    if brier_weight:
        brier = ((p - target).square() * mask).sum(-1).mean()
        loss = loss + brier_weight * brier
        if collect_components:
            parts["brier"] = float(brier.detach())
    if reference is not None and kl_weight:
        reference_options = mask if reference_mask is None else mask & reference_mask[:, None]
        ref = reference.detach().masked_fill(~reference_options, 0)
        ref = ref / ref.sum(-1, keepdim=True).clamp_min(1e-12)
        anchor = (ref * (ref.clamp_min(1e-12).log() - logp)).sum(-1)
        anchor = anchor.mean()
        loss = loss + kl_weight * anchor
        if collect_components:
            parts["reference_kl"] = float(anchor.detach())
    return loss, parts


def learning_rate_factor(step, total, warmup_ratio=0.05, min_ratio=0.1):
    """Linear warmup followed by cosine decay; steps are one based optimizer updates."""
    warmup = max(1, int(total * warmup_ratio)) if warmup_ratio else 0
    if step <= warmup:
        return step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def grow_lora(model, rank, alpha):
    """Grow existing adapter matrices while preserving their learned delta before any updates."""
    from peft.tuners.lora import LoraLayer
    for name, config in model.peft_config.items():
        if getattr(config, "use_dora", False) or config.rank_pattern:
            raise ValueError("rank growth requires uniform ordinary LoRA adapters")
        if rank < config.r:
            raise ValueError("rank reduction would discard trained adapter information")
        for module in model.modules():
            if not isinstance(module, LoraLayer) or name not in module.lora_A:
                continue
            old_a, old_b = module.lora_A[name], module.lora_B[name]
            old_rank, old_scale = old_a.out_features, module.scaling[name]
            new_scale = alpha / (math.sqrt(rank) if getattr(config, "use_rslora", False) else rank)
            if rank != old_rank:
                a = torch.nn.Linear(old_a.in_features, rank, bias=False, device=old_a.weight.device, dtype=old_a.weight.dtype)
                b = torch.nn.Linear(rank, old_b.out_features, bias=False, device=old_b.weight.device, dtype=old_b.weight.dtype)
                with torch.no_grad():
                    a.weight[:old_rank].copy_(old_a.weight)
                    b.weight.zero_()
                    b.weight[:, :old_rank].copy_(old_b.weight * (old_scale / new_scale))
                module.lora_A[name], module.lora_B[name] = a, b
            else:
                with torch.no_grad():
                    old_b.weight.mul_(old_scale / new_scale)
            module.r[name], module.lora_alpha[name], module.scaling[name] = rank, alpha, new_scale
        config.r, config.lora_alpha = rank, alpha


def lora_target_pattern(scope):
    # Qwen3.5 uses fused QKV and gated DeltaNet projections in three of every
    # four blocks. Restricting targets to q_proj/v_proj misses those blocks.
    fused = "in_proj_qkv|in_proj_qkvz"
    attention = "q_proj|k_proj|v_proj|o_proj|" + fused + "|in_proj_z|in_proj_b|in_proj_a|in_proj_ba|out_proj"
    names = {"qv": "q_proj|v_proj|" + fused, "attention": attention,
             "all-linear": attention + "|gate_proj|up_proj|down_proj"}[scope]
    return r".*(?:language_model|model\.layers)\..*\.(?:" + names + r")$"


def expand_lora(model, scope):
    """Add zero-delta language adapters without reinitializing any loaded adapters."""
    import re
    from peft.tuners.lora.layer import Linear
    pattern = lora_target_pattern(scope)
    base = model.get_base_model()
    for adapter, config in model.peft_config.items():
        for name, module in base.named_modules():
            if hasattr(module, "lora_A") and adapter in module.lora_A and not re.fullmatch(pattern, name):
                raise ValueError("lora_targets would discard an existing adapter: " + name)
        for name, module in list(base.named_modules()):
            if not re.fullmatch(pattern, name) or not isinstance(module, torch.nn.Linear):
                continue
            parent, _, child = name.rpartition(".")
            wrapped = Linear(module, adapter, r=config.r, lora_alpha=config.lora_alpha,
                             lora_dropout=config.lora_dropout, init_lora_weights=True,
                             use_rslora=config.use_rslora)
            wrapped.lora_A[adapter].float()
            wrapped.lora_B[adapter].float()
            setattr(base.get_submodule(parent), child, wrapped)
        config.target_modules = pattern


def optimizer_groups(model, learning_rate, head_lr=None, projector_lr=None):
    """Separate decision head, language adapters and native media projectors."""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        kind = "head" if any(k in name for k in ("lm_head", "decision_scorer", "grounding_head")) else "projector" if any(k in name for k in ("embed_audio", "embed_vision", "visual.merger")) else "adapter"
        lr = (head_lr or learning_rate) if kind == "head" else (projector_lr or learning_rate) if kind == "projector" else learning_rate
        decay = 0.0 if parameter.ndim < 2 or "norm" in name else 0.01
        key = kind, lr, decay
        groups.setdefault(key, []).append(parameter)
    return [{"params": params, "lr": lr, "base_lr": lr, "weight_decay": decay, "name": kind}
            for (kind, lr, decay), params in groups.items()]


def configure_training(agent, config, *, train_projectors=False, gradient_checkpointing=True):
    """Select trainable parameters without changing the input encoders or vocabulary size."""
    from ..models.multimodal import _active_head, _is_adapter_model

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
        _configure_projectors(agent, train_projectors)
        _active_head(agent.model).float()
        if candidate:
            _active_head(agent.model).requires_grad_(False)
        if gradient_checkpointing:
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
    _configure_projectors(agent, train_projectors)
    _active_head(agent.model).float()
    if gradient_checkpointing and config.mode != "head":
        agent.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in agent.model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("no trainable parameters")
    return parameters


def _configure_projectors(agent, train_projectors):
    if not train_projectors:
        return
    from ..models.multimodal import _is_adapter_model

    base = agent.model.get_base_model() if _is_adapter_model(agent.model) else agent.model
    names = [name for name, module in base.named_modules()
             if name in {"model.embed_audio", "model.embed_vision"} and module is not None]
    if not names:
        raise ValueError("native projector training currently requires Gemma 4 embed_audio/embed_vision modules")
    agent.native_trainable_modules = names
    for name in names:
        base.get_submodule(name).float().requires_grad_(True)


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Historical RLCD reward: clipped log score + spherical score + ranked probability score.

    The finite log floor is a numerical surrogate. Unlike the unclipped scoring rules, this
    combination is not globally strictly proper for arbitrarily small target probabilities.
    Supervised decision losses use stable log-softmax and differentiable Brier/RPS terms.

    q: [..., N, K] reported distributions
    target: [N, K] (one-hot or soft target distributions)
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float = 1.0) -> torch.Tensor:
    """TD(lambda) targets for multi-turn conversation trajectories."""
    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target
