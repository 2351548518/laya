"""Shared candidate-conditioned scoring without a fixed answer alphabet."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CandidateScorer(nn.Module):
    """Compare every candidate's contextual state with the complete decision state."""
    def __init__(self, hidden_size, dimension=256):
        super().__init__()
        self.query = nn.Linear(hidden_size, dimension, bias=False)
        self.candidate = nn.Linear(hidden_size, dimension, bias=False)
        with torch.no_grad():
            self.candidate.weight.copy_(self.query.weight)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.)))

    def forward(self, hidden, positions):
        mask = positions >= 0
        gathered = hidden.gather(1, positions.clamp_min(0)[..., None].expand(-1, -1, hidden.shape[-1]))
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            query = F.normalize(self.query(hidden[:, -1].float()), dim=-1)
            candidate = F.normalize(self.candidate(gathered.float()), dim=-1)
            logits = (candidate * query[:, None]).sum(-1) * self.log_scale.clamp(max=math.log(100.)).exp()
        return logits.masked_fill(~mask, -1e4)


def candidate_positions(offsets, original_text_length, spans):
    """Resolve marker spans after media expansion, which occurs before the text suffix."""
    offsets = torch.as_tensor(offsets)
    shift = int(offsets[:, 1].max()) - original_text_length
    spans = spans if isinstance(spans, torch.Tensor) else list(spans)
    boundaries = None
    if len(spans) >= 8 and offsets.dtype == torch.long:
        boundaries = torch.as_tensor(spans, device=offsets.device)
    if boundaries is not None and boundaries.dtype == torch.long:
        # Most tokenizers emit ordered offsets; ignore special/padding tokens.
        # Resolve all candidate markers with one scan and batched binary search.
        valid = (offsets[:, 1] > offsets[:, 0]).nonzero().flatten()
        ordered = offsets[valid]
        if len(ordered) and bool((ordered[1:] >= ordered[:-1]).all()):
            boundaries = boundaries + shift
            last = torch.searchsorted(ordered[:, 1].contiguous(), boundaries[:, 1].contiguous(), right=True) - 1
            found = (last >= 0) & (ordered[last.clamp_min(0), 0] >= boundaries[:, 0])
            if not bool(found.all()):
                raise ValueError("candidate marker has no matching tokenizer offsets")
            return valid[last].reshape(1, -1).to(device='cpu', dtype=torch.long)
        # Unusual tokenizers can emit non-monotonic offsets; keep exact semantics.
    positions = []
    for start, end in spans:
        selected = ((offsets[:, 0] >= start + shift) & (offsets[:, 1] <= end + shift)
                    & (offsets[:, 1] > offsets[:, 0])).nonzero().flatten()
        if not len(selected):
            raise ValueError("candidate marker has no matching tokenizer offsets")
        positions.append(int(selected[-1]))
    return torch.tensor([positions], dtype=torch.long)
