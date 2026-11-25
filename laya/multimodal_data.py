"""Local dataset records with System One requests, labels and file references."""
from __future__ import annotations

import json
from pathlib import Path
import re

from .systemone import bind_image_references

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
