"""Chunked gated delta scan for Qwen3.5 training."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F


def _chunk_gated_delta(query, key, value, g, beta, chunk_size=64, initial_state=None,
                       output_final_state=False, use_qk_l2norm_in_kernel=False, **_kwargs):
    dtype = query.dtype
    batch, length, heads, key_size = key.shape
    value_heads, value_size = value.shape[-2:]
    if heads != value_heads:
        query = query.repeat_interleave(value_heads // heads, dim=2)
        key = key.repeat_interleave(value_heads // heads, dim=2)
    query, key, value, beta, decay = (
        item.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for item in (query, key, value, beta, g)
    )
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    query = query * (key_size ** -0.5)
    padding = (-length) % chunk_size
    query, key, value = (F.pad(item, (0, 0, 0, padding)) for item in (query, key, value))
    beta, decay = (F.pad(item, (0, padding)) for item in (beta, decay))
    chunks = (length + padding) // chunk_size
    weighted_keys = key * beta.unsqueeze(-1)
    weighted_values = value * beta.unsqueeze(-1)
    query, key, weighted_keys, weighted_values = (
        item.reshape(batch, value_heads, chunks, chunk_size, item.shape[-1])
        for item in (query, key, weighted_keys, weighted_values)
    )
    log_decay = decay.reshape(batch, value_heads, chunks, chunk_size).cumsum(dim=-1)
    future = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    decay_between = (log_decay.unsqueeze(-1) - log_decay.unsqueeze(-2)).masked_fill(future, -float("inf")).exp()
    system = (weighted_keys @ key.transpose(-1, -2)) * decay_between
    attention = (query @ key.transpose(-1, -2)) * decay_between
    prefix_decay = log_decay.exp().unsqueeze(-1)
    updates = torch.linalg.solve_triangular(system, weighted_values, upper=False, unitriangular=True)
    old_values = torch.linalg.solve_triangular(system, weighted_keys * prefix_decay,
                                              upper=False, unitriangular=True)
    state = (updates.new_zeros((batch, value_heads, key_size, value_size))
             if initial_state is None else initial_state.to(updates))
    query = query * prefix_decay
    last_decay = log_decay[..., -1:]
    key = key * (last_decay - log_decay).exp().unsqueeze(-1)
    state_decay = last_decay.exp().unsqueeze(-1)

    # Unbind and stack collect chunk gradients once. Indexing every input and
    # assigning every output slice independently creates full-size gradients
    # for each chunk, which is particularly costly for long sequences.
    output = []
    scan = (item.unbind(2) for item in (updates, old_values, query, attention, key, state_decay))
    for update, previously_predicted, q, weights, k, retention in zip(*scan):
        corrected = update - previously_predicted @ state
        output.append(q @ state + weights @ corrected)
        state = state * retention + k.transpose(-1, -2) @ corrected
    output = torch.stack(output, dim=2).flatten(2, 3)[:, :, :length]
    return output.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format), state if output_final_state else None


@contextmanager
def qwen_training_delta():
    """Enable the local scan in one trainer process and retain native evaluation dispatch."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    original = modeling_qwen3_5.torch_chunk_gated_delta_rule

    def dispatch(query, key, value, *args, **kwargs):
        if torch.is_grad_enabled() and query.requires_grad and kwargs.get("cu_seqlens") is None:
            return _chunk_gated_delta(query, key, value, *args, **kwargs)
        return original(query, key, value, *args, **kwargs)

    modeling_qwen3_5.torch_chunk_gated_delta_rule = dispatch
    try:
        yield
    finally:
        modeling_qwen3_5.torch_chunk_gated_delta_rule = original
