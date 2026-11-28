"""Collect and score native checkpoint decisions for development and evaluation."""
from __future__ import annotations

from itertools import chain
import os

import numpy as np
import torch

from ..decisions.questions import QTYPES, unpermute_probs
from ..decisions.calibration import ece_score, temp_bucket
from ..training.reporting import log_training_event
from ..models.modality_calibration import ordinal_probabilities
from ..data import (
    collate_multimodal_inputs, decision_target, example_hash, record_input, record_modality,
)


def _items(agent, rows):
    items = []
    for row in rows:
        for qid, qdef in row["questions"].items():
            agent._check_question(qid, qdef)
            q = agent._to_internal(qdef)
            items.append((row, qid, qdef, decision_target(row, qid, q)))
    return items


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
            log_training_event("EVAL", "decisions=%d/%d" % (finished, len(items)))

    longest = 0
    if micro_batch == 1:
        ordered = _serial_evaluation_order(agent, items)
    else:
        ordered = sorted(enumerate(items), key=lambda item: record_modality(item[1][0]))
    for index, (row, qid, qdef, target) in ordered:
        request, question = record_input(agent, row, qid, qdef)
        inputs, q, media = agent._prepare(request, question)
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
        if row.get("_image_assets") and media.get("images") == 0:
            modality = "text"
        metadata.append((index, row, qid, q, target, modality))
    if prepared:
        collect_batch()
    return records


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
