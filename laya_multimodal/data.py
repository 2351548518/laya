"""Local dataset records with System One requests, labels and file references."""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List

from .decisions.questions import render_options
from .decisions.systemone import bind_image_references

MEDIA_FIELDS = ("image", "audio", "video")
MEDIA_CONTROLS = ("sampling_rate", "video_audio", "video_bounds", "video_frames", "video_encoding")
BOX_EDGES = ("left", "top", "right", "bottom")
_POSITIONAL_OPTION = re.compile(
    r"\b(?:all|none|both|neither|any) of (?:the )?(?:above|below|other (?:options|choices))\b"
    r"|\b(?:first|last|previous|next) (?:option|choice|answer)\b"
    r"|\b(?:option|choice|answer|candidate)\s+(?:\([a-z0-9]+\)|[a-z]\b|\d+\b)",
    re.IGNORECASE,
)
_LETTER_PAIR = re.compile(r"\b[A-Z]\s+(?:and|or|&|plus|/)\s+[A-Z]\b")


def _media_path(value, directory, verify):
    path = Path(value)
    path = path if path.is_absolute() else directory / path
    if verify and not path.is_file():
        raise FileNotFoundError("missing media file: " + str(path))
    return str(path)


def _file_part(asset, media_files, used_parts, directory, verify):
    if not isinstance(asset, dict) or not isinstance(asset.get("part"), str) or not asset["part"].strip():
        raise ValueError("local asset needs a named file part")
    name = asset["part"]
    if name not in media_files:
        raise ValueError("file part has no media_files entry: " + name)
    if name in used_parts:
        raise ValueError("each asset needs a distinct file part: " + name)
    used_parts.add(name)
    return _media_path(media_files[name], directory, verify)


def normalize_record(row, directory, *, verify_media=True):
    """Load labelled native, Kev or System One records into the local trainer's layout."""
    if not isinstance(row, dict):
        raise ValueError("each record must be an object")
    row = {**row}
    request = row.pop("request", None)
    if request is not None:
        if not isinstance(request, dict) or "state" in row or "questions" in row:
            raise ValueError("provide either a request envelope or state and questions")
        if "state" not in request or "questions" not in request:
            raise ValueError("request needs state and questions")
        state = row["state"] = request["state"]
        row["questions"] = request["questions"]
        orders = row.pop("option_orders", {})
        if orders:
            if not isinstance(orders, dict) or not isinstance(row["questions"], dict):
                raise ValueError("option_orders must be keyed by question name")
            questions = row["questions"] = {**row["questions"]}
            for qid, order in orders.items():
                question = questions[qid]
                if (not isinstance(order, list) or not all(type(index) is int for index in order)
                        or sorted(order) != list(range(len(order)))):
                    raise ValueError("invalid option order for " + qid)
                question = questions[qid] = {**question, "option_order": order}
                if question["type"] == "choice":
                    criteria = question["criteria"]
                    entries = list(criteria.items()) if isinstance(criteria, dict) else criteria
                    if len(entries) != len(order):
                        raise ValueError("option order and criteria differ in size for " + qid)
                    original = [entries[index] for index in sorted(range(len(order)), key=order.__getitem__)]
                    question["criteria"] = dict(original) if isinstance(criteria, dict) else original
        media_files = row.pop("media_files", {})
        if not isinstance(media_files, dict):
            raise ValueError("media_files must be an object of local paths")
        used_parts = set()
        assets = request.get("assets", {})
        if not isinstance(assets, dict):
            raise ValueError("assets must be an object")
        if assets:
            if any(not ref.strip() for ref in assets):
                raise ValueError("image references must be nonempty")
            if any(not isinstance(asset, dict) or asset.get("type") != "image" for asset in assets.values()):
                raise ValueError("System One dataset assets must declare type image")
            row["_image_assets"] = {ref: _file_part(asset, media_files, used_parts, directory, verify_media)
                                    for ref, asset in assets.items()}
        native_media = {}
        for kind in MEDIA_FIELDS:
            value = request.get(kind)
            if value is None:
                continue
            if isinstance(value, list) and (not value or kind == "video"):
                raise ValueError("each request needs one video part or nonempty image/audio parts")
            parts = value if isinstance(value, list) else [value]
            paths = [_file_part(part, media_files, used_parts, directory, verify_media) for part in parts]
            native_media[kind] = paths if isinstance(value, list) else paths[0]
        if used_parts != media_files.keys():
            raise ValueError("each media_files entry needs a matching asset or media part")
        if native_media:
            if assets:
                raise ValueError("combine media through the native extension or through image assets")
            row["state"] = {"text": state, **native_media,
                            **{key: request[key] for key in MEDIA_CONTROLS if key in request}}
    else:
        state = row.get("state")
        if isinstance(state, dict) and any(state.get(key) is not None for key in MEDIA_FIELDS):
            state = row["state"] = {**state}
            for key in MEDIA_FIELDS:
                value = state.get(key)
                if value is not None:
                    state[key] = ([_media_path(path, directory, verify_media) for path in value]
                                  if isinstance(value, list) else _media_path(value, directory, verify_media))

    questions = row.get("questions")
    if "state" not in row or not isinstance(questions, dict) or not questions:
        raise ValueError("record needs state and a nonempty questions object")
    if any(isinstance(question, dict) and "label" in question for question in questions.values()):
        expected = dict(row.get("expected", {}))
        questions = row["questions"] = {qid: {**question} for qid, question in questions.items()}
        for qid, question in questions.items():
            if "label" in question:
                label = question.pop("label")
                if qid in expected and expected[qid] != label:
                    raise ValueError("disagreeing labels for question " + qid)
                expected[qid] = label
            question.pop("src", None)
        row["expected"] = expected
        metadata = row.get("_meta")
        if isinstance(metadata, dict):
            for name, key in (("id", "id"), ("group", "group_id"), ("source", "source")):
                if key in metadata:
                    row.setdefault(name, metadata[key])
    for key in ("expected", "gold"):
        if key in row and not isinstance(row[key], dict):
            raise ValueError(key + " must be keyed by question name")
    if "grounding" not in row and len(questions) == 1:
        qid, question = next(iter(questions.items()))
        answer = row.get("expected", {}).get(qid)
        if isinstance(question, dict) and question.get("type") == "box" and isinstance(answer, dict) and answer.get("type") == "box":
            coordinates = answer.get("box")
            if not isinstance(coordinates, dict) or set(coordinates) != set(BOX_EDGES):
                raise ValueError("Box supervision needs four named coordinates")
            row["grounding"] = {"task": "single", "boxes": [[coordinates[key] for key in BOX_EDGES]],
                                "ignored": row.get("ignored_boxes", [])}
    return row


def read_records(path, *, verify_media=True):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith(b"#"):
                continue
            try:
                yield normalize_record(json.loads(line), path.parent, verify_media=verify_media)
            except FileNotFoundError as error:
                raise FileNotFoundError("%s:%d: %s" % (path, line_number, error)) from error
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("%s:%d: %s" % (path, line_number, error)) from error


def load_rows(path):
    """Read native decision JSONL or labelled System One and Kev records."""
    rows = list(read_records(path))
    if not rows:
        raise ValueError("empty multimodal dataset: %s" % path)
    if any(not ("expected" in row or "gold" in row) for row in rows):
        raise ValueError("%s needs expected (or gold) labels" % path)
    return rows


def record_modality(row):
    state = row["state"]
    if row.get("_image_assets"):
        return "image"
    return next((key for key in ("video", "audio", "image")
                 if isinstance(state, dict) and state.get(key) is not None), "text")


def record_media(row):
    if row.get("_image_assets"):
        yield from row["_image_assets"].values()
    state = row["state"]
    if isinstance(state, dict):
        for kind in MEDIA_FIELDS:
            value = state.get(kind)
            if value is not None:
                yield from value if isinstance(value, list) else (value,)


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


def example_hash(row, qid, qdef):
    content = [row["state"], qid, qdef]
    if row.get("_image_assets"):
        content.append(row["_image_assets"])
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def target_from_gold(q: Dict[str, Any], gold_q: Dict[str, Any]) -> List[float]:
    """Return the normalized target distribution for an internal question, in option order.

    `gold_q["probabilities"]` is keyed by choice label, by `"false"`/`"true"` for noul, and by
    the level index as a string for score -- the `{state, questions, gold}` schema in
    `docs/finetune.md`. A choice label that is not a string is also looked up as its JSON key.
    A row whose probabilities sum to zero becomes uniform. A negative or
    non-finite probability raises `ValueError`.
    """
    probs = gold_q["probabilities"]
    if not isinstance(probs, Mapping):
        raise ValueError("gold probabilities must be keyed by option")

    def get(key, default):
        return probs.get(key, probs.get(str(key), default))

    t, crit = q["t"], q["crit"]
    if t == "choice":
        target = [float(get(k, 0.0)) for k in crit.keys()]
    elif t == "noul":
        target = [float(get("false", 0.5)), float(get("true", 0.5))]
    else:
        target = [float(get(str(i), 0.0)) for i in range(len(crit))]
    if any(not math.isfinite(v) or v < 0 for v in target):
        raise ValueError("probabilities must be finite and non-negative, got %r" % (target,))
    total = sum(target)
    if math.isinf(total):
        # The weights are finite, but their sum can still overflow.
        scale = max(target)
        target = [value / scale for value in target]
        total = sum(target)
    if total > 0:
        return [v / total for v in target]
    return [1.0 / len(target)] * len(target)


def target_from_expected(q: Dict[str, Any], expected_val: Any,
                         label_smoothing: float = 0.0) -> List[float]:
    """Return the normalized target distribution from one expected answer, in option order.

    `expected_val` is:
    - a choice label (str or value matching a criterion) for choice questions
    - a boolean for noul questions
    - an integer level index or level string for score questions

    When `label_smoothing > 0`, smooths probability mass uniformly:
    target[i] = (1 - eps) * one_hot[i] + eps / K.
    """
    if isinstance(label_smoothing, bool) or not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must be in [0, 1), got %r" % (label_smoothing,))

    t, crit = q["t"], q["crit"]
    k = len(render_options(q))
    idx = None
    target = None

    if t == "choice":
        keys = list(crit.keys())
        if expected_val in crit:
            idx = keys.index(expected_val)
        else:
            str_expected = str(expected_val)
            for i, key in enumerate(keys):
                if str(key) == str_expected:
                    idx = i
                    break
        if idx is None:
            raise ValueError("expected answer %r not found in choice options %r" % (expected_val, keys))
    elif t == "noul":
        if isinstance(expected_val, bool):
            idx = 1 if expected_val else 0
        elif isinstance(expected_val, (int, float)):
            if expected_val in (0, 1) or expected_val in (0.0, 1.0):
                idx = int(expected_val)
            else:
                raise ValueError("numeric noul expected answer must be 0 or 1, got %r" % (expected_val,))
        elif isinstance(expected_val, str):
            lower = expected_val.strip().lower()
            if lower in ("true", "1", "yes"):
                idx = 1
            elif lower in ("false", "0", "no"):
                idx = 0
        if idx is None:
            raise ValueError("expected answer %r cannot be parsed as a noul boolean" % (expected_val,))
    else:  # score
        keys = list(crit.keys()) if isinstance(crit, dict) else [str(i) for i in range(len(crit))]
        K = len(keys)
        target = None
        if isinstance(expected_val, str) and expected_val in keys:
            idx = keys.index(expected_val)
            target = [1.0 if i == idx else 0.0 for i in range(K)]
        else:
            try:
                if not isinstance(expected_val, bool):
                    val = float(expected_val)
                    if 0.0 <= val <= float(K - 1):
                        low = int(math.floor(val))
                        if low == K - 1 or val == float(low):
                            target = [1.0 if i == low else 0.0 for i in range(K)]
                        else:
                            frac = val - float(low)
                            target = [0.0] * K
                            target[low] = 1.0 - frac
                            target[low + 1] = frac
                    else:
                        raise ValueError("expected score %r out of bounds [0, %d]" % (expected_val, K - 1))
            except (ValueError, TypeError) as exc:
                if isinstance(exc, ValueError) and "out of bounds" in str(exc):
                    raise
                pass
        if target is None:
            str_val = str(expected_val)
            for i, key in enumerate(keys):
                if str(key) == str_val:
                    target = [1.0 if j == i else 0.0 for j in range(K)]
                    break
        if target is None:
            raise ValueError("expected answer %r not found in score levels %r" % (expected_val, keys))

    if target is None:
        target = [1.0 if i == idx else 0.0 for i in range(k)]
    if label_smoothing > 0.0:
        smooth_val = label_smoothing / k
        target = [(1.0 - label_smoothing) * v + smooth_val for v in target]
    return target


def decision_target(row, qid, question):
    """Read the target in canonical option order, preferring soft labels when supplied."""
    if qid in row.get("gold", {}):
        return target_from_gold(question, row["gold"][qid])
    return target_from_expected(question, row["expected"][qid])


def record_input(agent, row, qid, question):
    images = row.get("_image_assets")
    if not images:
        return agent._input(row["state"]), question
    # Bind the complete request so a question can use only a subset of request assets.
    questions = {**row["questions"], qid: question}
    request, questions = bind_image_references(row["state"], questions, images)
    return request, questions[qid]


def can_shuffle_choice(question):
    """Keep the supplied order when option text or instructions refer to positions."""
    if question.get("type") != "choice" or question.get("option_order") is not None:
        return False
    criteria = question["criteria"]
    if isinstance(criteria, dict):
        options = [label if value in (None, "") else value for label, value in criteria.items()]
    else:
        options = criteria
    for value in (question.get("instructions", ""), *options):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if _POSITIONAL_OPTION.search(text) or _LETTER_PAIR.search(text):
            return False
    return True


def collate_multimodal_inputs(examples, pad_token_id):
    """Left-pad text and concatenate padded native media tensors in request order.

    Each request was processed independently, so image patches and audio features can have
    different lengths. Position padding is -1, which the Gemma processor uses for absent patches.
    """
    import torch
    from torch.nn import functional as F

    result = {}
    sequence_keys = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
    for key in set().union(*(example.keys() for example in examples)):
        values = [example[key] for example in examples if key in example]
        if key == "_candidate_positions":
            length = max(example["input_ids"].shape[-1] for example in examples)
            values = [torch.where(example[key] < 0, example[key], example[key] + length - example["input_ids"].shape[-1])
                      for example in examples]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise ValueError("cannot batch a non-tensor processor field: " + key)
        if any(value.ndim != values[0].ndim for value in values):
            raise ValueError("inconsistent processor dimensions for " + key)
        sizes = [max(value.shape[axis] for value in values) for axis in range(1, values[0].ndim)]
        padded = []
        for value in values:
            widths = []
            for axis in reversed(range(1, value.ndim)):
                difference = sizes[axis - 1] - value.shape[axis]
                widths.extend([difference, 0] if key in sequence_keys else [0, difference])
            fill = pad_token_id if key == "input_ids" else -1 if key in {"image_position_ids", "video_position_ids", "_candidate_positions"} else 0
            padded.append(F.pad(value, widths, value=fill) if widths else value)
        result[key] = torch.cat(padded, dim=0)
    return result
