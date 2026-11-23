"""Proper supervised/ordinal, filtered distillation and anchored noisy-policy decision objectives.

The RL term uses centered Gaussian logit actions with a per-decision leave-one-out baseline.
It does not reproduce proprietary Jev RLCD or rationale-generating OpenJev-RLCD.
"""
from __future__ import annotations

import math

import torch

from .common import QTYPES, proper_reward


def _policy_loss(logits, target, mask, qtypes, sigma, samples):
    """Per-decision leave-one-out Gaussian estimator, independent of batch partitioning."""
    if samples < 2 or sigma <= 0:
        raise ValueError("policy sampling requires samples >= 2 and sigma > 0")
    valid = mask.float()
    eps = torch.randn((samples,) + logits.shape, device=logits.device) * sigma * valid
    eps = (eps - eps.sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)) * valid
    actions = logits.detach().unsqueeze(0) + eps
    with torch.no_grad():
        reported = actions.masked_fill(~mask, -1e4).softmax(-1)
        rewards = proper_reward(reported, target.unsqueeze(0), qtypes, valid, w_sph=.75)
        baseline = (rewards.sum(0, keepdim=True) - rewards) / (samples - 1)
        advantage = rewards - baseline
    log_density = -((actions - logits.unsqueeze(0)).square() * valid).sum(-1) / (2 * sigma ** 2)
    return -(advantage * log_density).mean()


def decision_loss(logits, target, mask, qtypes, *, objective="soft-ce", ordinal_weight=0.0,
                  brier_weight=0.0, teacher=None, teacher_mask=None, distill_weight=0.0,
                  distill_temperature=2.0, reference=None, reference_mask=None, kl_weight=0.0, sigma=0.15,
                  rl_samples=8, rl_weight=1.0):
    """Return a differentiable loss and detached component measurements for one micro-batch."""
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e4).float(), -1)
    p = logp.exp() * mask
    ce = -(target * logp).sum(-1).mean()
    loss = ce
    parts = {"ce": float(ce.detach())}
    if objective == "rlcd":
        policy = _policy_loss(logits, target, mask, qtypes, sigma, rl_samples)
        loss = loss + rl_weight * policy
        parts["policy_gradient"] = float(policy.detach())
    elif objective != "soft-ce":
        raise ValueError("unknown decision objective")
    if ordinal_weight:
        selected = qtypes == QTYPES["score"]
        widths = mask.sum(-1)
        boundaries = torch.arange(logits.shape[-1], device=logits.device)[None, :] < (widths - 1)[:, None]
        rps = ((p.cumsum(-1) - target.cumsum(-1)).square() * boundaries).sum(-1) / (widths - 1).clamp_min(1)
        # Use the full batch denominator, so token-budget micro-batch splitting preserves
        # the objective and does not amplify a rare ordinal example.
        ordinal = (rps * selected).mean()
        loss = loss + ordinal_weight * ordinal
        parts["ordinal_rps"] = float(ordinal.detach())
    if brier_weight:
        brier = ((p - target).square() * mask).sum(-1).mean()
        loss = loss + brier_weight * brier
        parts["brier"] = float(brier.detach())
    if teacher is not None and distill_weight:
        accepted = teacher_mask if teacher_mask is not None else torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        if accepted.any():
            temperature = distill_temperature
            t = teacher.clamp_min(1e-12).pow(1 / temperature) * mask
            t = t / t.sum(-1, keepdim=True).clamp_min(1e-12)
            student_logp = torch.log_softmax(logits.float().masked_fill(~mask, -1e4) / temperature, -1)
            kd = (t * (t.clamp_min(1e-12).log() - student_logp)).sum(-1)
            # Weight over the full batch so a rare teacher does not receive an inflated coefficient.
            kd = (kd * accepted).mean() * temperature ** 2
            loss = loss + distill_weight * kd
            parts["distillation_kl"] = float(kd.detach())
    if reference is not None and kl_weight:
        ref = reference.detach() * mask
        ref = ref / ref.sum(-1, keepdim=True).clamp_min(1e-12)
        anchor = (ref * (ref.clamp_min(1e-12).log() - logp)).sum(-1)
        if reference_mask is not None:
            anchor = anchor * reference_mask
        anchor = anchor.mean()
        loss = loss + kl_weight * anchor
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
