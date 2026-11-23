"""Direct single-box grounding on native VLM states, with legacy checkpoint support.

Box follows Jev's public response geometry; set detection is a separate local extension.
The architecture is an open implementation, not a claim about proprietary Jev weights.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_boxes(boxes):
    """Validate normalized xyxy targets without silently repairing annotation errors."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
        raise ValueError("boxes must contain finite normalized xyxy coordinates in [0,1]")
    if ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
        raise ValueError("boxes must have positive width and height")
    return boxes


def box_iou(boxes, targets, generalized=False):
    """Differentiable pairwise IoU/GIoU in normalized xyxy coordinates."""
    area_a = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    area_b = (targets[:, 2:] - targets[:, :2]).clamp_min(0).prod(-1)
    lo = torch.maximum(boxes[:, None, :2], targets[None, :, :2])
    hi = torch.minimum(boxes[:, None, 2:], targets[None, :, 2:])
    intersection = (hi - lo).clamp_min(0).prod(-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    iou = intersection / union.clamp_min(1e-8)
    if not generalized:
        return iou
    cover = (torch.maximum(boxes[:, None, 2:], targets[None, :, 2:])
             - torch.minimum(boxes[:, None, :2], targets[None, :, :2])).clamp_min(0).prod(-1)
    return iou - (cover - union) / cover.clamp_min(1e-8)


def ordered_boxes(raw):
    """Produce strictly ordered in-frame boxes, including at finite-precision extremes."""
    p = raw.float().sigmoid()
    lo = torch.minimum(p[..., :2], p[..., 2:]).clamp(max=1 - 2e-6)
    hi = torch.maximum(p[..., :2], p[..., 2:]).clamp(max=1)
    return torch.cat((lo, torch.maximum(hi, lo + 1e-6)), dim=-1)


def crowd_overlap(boxes, targets):
    """COCO crowd overlap uses intersection over prediction area, not union area."""
    lo = torch.maximum(boxes[:, None, :2], targets[None, :, :2])
    hi = torch.minimum(boxes[:, None, 2:], targets[None, :, 2:])
    intersection = (hi - lo).clamp_min(0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    return intersection / area[:, None].clamp_min(1e-8)


def last_valid_state(hidden, attention_mask=None):
    if attention_mask is None:
        return hidden[:, -1]
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None].expand(hidden.shape[:2])
    last = positions.masked_fill(~attention_mask.bool(), -1).max(-1).values
    if (last < 0).any():
        raise ValueError("grounding input has no valid tokens")
    return hidden[torch.arange(len(hidden), device=hidden.device), last]


class GroundingHead(nn.Module):
    """A direct single-box readout; explicit legacy slots load older set checkpoints."""
    def __init__(self, hidden_size, dimension=256, slots=1, architecture=None):
        super().__init__()
        if slots < 1 or dimension < 8 or dimension % 8:
            raise ValueError("positive slots and a grounding dimension divisible by eight are required")
        self.hidden_size, self.dimension, self.slots = hidden_size, dimension, slots
        architecture = architecture or ("single-v1" if slots == 1 else "set-v2")
        if architecture not in {"single-v1", "set-v1", "set-v2"}:
            raise ValueError("unsupported grounding architecture")
        if architecture == "single-v1" and slots != 1:
            raise ValueError("single-box architecture requires exactly one slot")
        self.architecture = architecture
        self.single = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dimension), nn.GELU(),
                                    nn.Linear(dimension, 5))
        with torch.no_grad():
            self.single[-1].bias[:4].copy_(torch.tensor([-1., -1., 1., 1.]))
        if architecture == "single-v1":
            return
        self.memory = nn.Linear(hidden_size, dimension)
        self.queries = nn.Parameter(torch.randn(slots, dimension) * .02)
        self.attention = nn.MultiheadAttention(dimension, 8, batch_first=True)
        self.norm = nn.LayerNorm(dimension)
        self.objects = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 6))
        if architecture == "set-v2":
            self.memory_norm = nn.LayerNorm(dimension)
            self.query_norm = nn.LayerNorm(dimension)
            layer = nn.TransformerDecoderLayer(dimension, 8, 4 * dimension, dropout=0., batch_first=True, norm_first=True)
            self.decoder = nn.TransformerDecoder(layer, 2)
            for parameter in self.decoder.parameters():
                if parameter.ndim > 1:nn.init.xavier_uniform_(parameter)
            side = math.ceil(math.sqrt(slots))
            centers = torch.tensor([[(i % side + .5) / side, (i // side + .5) / side] for i in range(slots)])
            anchors = torch.cat(((centers - .2).clamp(.01, .99), (centers + .2).clamp(.01, .99)), -1)
            self.anchor_logits = nn.Parameter(torch.logit(anchors))
        with torch.no_grad():
            self.objects[-1].bias[:4].copy_(torch.tensor([-1., -1., 1., 1.]))
            if architecture == "set-v2":self.objects[-1].bias[:4].zero_()
            self.objects[-1].bias[4] = -2.

    def to_single(self):
        """Copy the trained Box readout exactly, leaving all legacy set parameters behind."""
        parameter = next(self.single.parameters())
        result = GroundingHead(self.hidden_size, self.dimension).to(device=parameter.device, dtype=parameter.dtype)
        result.single.load_state_dict(self.single.state_dict())
        for new, old in zip(result.single.parameters(), self.single.parameters()):
            new.requires_grad_(old.requires_grad)
        return result.train(self.training)

    def forward(self, hidden, attention_mask=None, multiple=False):
        if multiple and self.architecture == "single-v1":
            raise ValueError("single-box architecture does not support multi-object detection")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            if not multiple:
                query = last_valid_state(hidden, attention_mask).float()
                raw = self.single(query)[:, None]
                return {"boxes": ordered_boxes(raw[..., :4]), "quality_logits": raw[..., 4]}
            hidden = hidden.float()
            query = last_valid_state(hidden, attention_mask)
            memory = self.memory(hidden)
            padding = None if attention_mask is None else ~attention_mask.bool()
            if self.architecture == "set-v2":
                slots = self.query_norm(self.queries)[None] + self.memory_norm(self.memory(query))[:, None]
                decoded = self.decoder(slots, self.memory_norm(memory), memory_key_padding_mask=padding)
                raw = self.objects(self.norm(decoded))
                raw = torch.cat((raw[..., :4] + self.anchor_logits[None], raw[..., 4:]), -1)
            else:
                slots = self.queries[None] + self.memory(query)[:, None]
                attended, _ = self.attention(slots, memory, memory, key_padding_mask=padding, need_weights=False)
                raw = self.objects(self.norm(slots + attended))
            return {"boxes": ordered_boxes(raw[..., :4]), "object_logits": raw[..., 4], "quality_logits": raw[..., 5]}


def grounding_loss(output, targets, *, multiple=False, l1_weight=5., giou_weight=2., quality_weight=.5,
                   no_object_weight=.1, ignored=None, quality_matched_only=False):
    """Direct single-box supervision; legacy set targets use bipartite matching.

    Empty targets supervise absence only, never invented zero-size regression boxes.
    The quality head estimates IoU; its score is not automatically a calibrated probability.
    """
    if multiple:
        from scipy.optimize import linear_sum_assignment
    if len(targets) != len(output["boxes"]):
        raise ValueError("grounding target batch mismatch")
    loss = output["boxes"].sum() * 0
    totals = {"l1": 0., "giou": 0., "quality": 0., "objectness": 0.}
    for batch, values in enumerate(targets):
        boxes = output["boxes"][batch]
        truth = validate_boxes(values).to(boxes.device)
        if not multiple and len(boxes) != 1:
            raise ValueError("a single Box question requires exactly one prediction")
        if len(truth) > len(boxes):
            raise ValueError("more target objects than slots; increase slots, never truncate targets")
        quality = torch.zeros(len(boxes), device=boxes.device)
        presence = torch.zeros_like(quality)
        if len(truth):
            if multiple:
                cost = l1_weight * torch.cdist(boxes, truth, p=1) - giou_weight * box_iou(boxes, truth, True)
                cost = cost - output["object_logits"][batch].sigmoid()[:, None]
                predicted, actual = linear_sum_assignment(cost.detach().cpu().numpy())
            else:
                predicted, actual = [0], [0]
            predicted = torch.as_tensor(predicted, device=boxes.device)
            actual = torch.as_tensor(actual, device=boxes.device)
            selected, matched = boxes[predicted], truth[actual]
            l1 = (selected - matched).abs().sum(-1).mean()
            giou = (1 - box_iou(selected, matched, True).diag()).mean()
            loss = loss + l1_weight * l1 + giou_weight * giou
            totals["l1"] += float(l1.detach()); totals["giou"] += float(giou.detach())
            quality[predicted] = box_iou(selected, matched).diag().detach()
            presence[predicted] = 1
        valid = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
        if ignored is not None and len(ignored[batch]):
            crowd = validate_boxes(ignored[batch]).to(boxes.device)
            valid = (presence > 0) | (crowd_overlap(boxes.detach(), crowd).max(-1).values < .5)
        qloss = F.binary_cross_entropy_with_logits(output["quality_logits"][batch], quality, reduction="none")
        # With separate objectness, conditional IoU should describe matched objects.
        # Training its unmatched slots toward zero counts absence a second time in
        # objectness * quality and can suppress scores for sparse target sets.
        quality_valid = valid & (presence > 0) if multiple and quality_matched_only else valid
        qloss = (qloss * quality_valid).sum() / quality_valid.sum().clamp_min(1)
        loss = loss + quality_weight * qloss; totals["quality"] += float(qloss.detach())
        if multiple:
            weights = torch.where(presence > 0, 1., no_object_weight) * valid
            objectness = F.binary_cross_entropy_with_logits(output["object_logits"][batch], presence, reduction="none")
            objectness = (objectness * weights).sum() / weights.sum().clamp_min(1)
            loss = loss + objectness; totals["objectness"] += float(objectness.detach())
    return loss / len(targets), {key: value / len(targets) for key, value in totals.items()}


def box_answer(box, confidence):
    coords = [float(x) for x in box]
    if not all(math.isfinite(x) for x in coords + [float(confidence)]):
        raise RuntimeError("nonfinite grounding prediction")
    left, top, right, bottom = coords
    if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
        raise RuntimeError("grounding prediction is not an ordered normalized box")
    return {"type": "box", "box": dict(top=top, bottom=bottom, left=left, right=right),
            "confidence": float(confidence)}
