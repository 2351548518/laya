"""Convert local labelled JSONL splits to request-shaped training or benchmark files."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laya_multimodal.multimodal import MultimodalAgent
from laya_multimodal.multimodal_data import BOX_EDGES, MEDIA_CONTROLS, MEDIA_FIELDS, can_shuffle_choice, normalize_record
from laya_multimodal.systemone import MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS
from laya_multimodal.train import _validate_grounding_row, decision_target

try:
    from orjson import dumps as encode_json, loads
except ImportError:
    loads = json.loads

    def encode_json(record):
        return json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


@lru_cache(maxsize=131072)
def resolved_media(path, directory, destination):
    local = (directory / path).resolve(strict=True)
    if not local.is_file():
        raise ValueError("media path must be a local file: " + str(local))
    return os.path.relpath(local, destination)


def _structured(value):
    return isinstance(value, (str, list, dict))


def systemone_question(question):
    if set(question) - {"type", "instructions", "criteria"}:
        return False
    if question.get("instructions") is not None and not _structured(question["instructions"]):
        return False
    criteria = question.get("criteria")
    if question["type"] == "choice":
        return (isinstance(criteria, dict) and 1 <= len(criteria) <= MAX_CHOICE_OPTIONS
                and all(_structured(value) or value is None for value in criteria.values()))
    if question["type"] == "score":
        return len(criteria) <= MAX_SCORE_LEVELS and all(_structured(value) for value in criteria)
    return not criteria or all(_structured(value) or value is None for value in criteria.values())


def _hard_label(question, target):
    index = target.index(1.)
    kind = question["type"]
    if kind == "choice":
        return list(question["criteria"])[index]
    return bool(index) if kind == "noul" else index


def _supervision(record, row, questions):
    expected, gold = dict(row.get("expected", {})), dict(row.get("gold", {}))
    for qid, question in questions.items():
        MultimodalAgent._check_question(qid, question)
        if question["type"] == "box":
            truth = row["grounding"]
            for box in truth["boxes"] + truth.get("ignored", []):
                if (len(box) != 4 or not all(math.isfinite(value) for value in box)
                        or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                    raise ValueError("invalid normalized box")
            if truth["task"] == "single" and len(truth["boxes"]) == 1:
                expected[qid] = {"type": "box", "box": dict(zip(BOX_EDGES, truth["boxes"][0])),
                                 "confidence": None}
                record.pop("grounding", None)
                if truth.get("ignored"):
                    record["ignored_boxes"] = truth["ignored"]
            continue
        raw = gold.get(qid)
        if raw is not None:
            probabilities = raw["probabilities"]
            keys = (list(question["criteria"]) if question["type"] == "choice"
                    else ["false", "true"] if question["type"] == "noul"
                    else [str(i) for i in range(len(question["criteria"]))])
            if (not probabilities or not set(probabilities) <= {str(key) for key in keys}
                    or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in probabilities.values())
                    or not sum(probabilities.values())):
                raise ValueError("invalid gold probability vector for " + qid)
        target = decision_target(row, qid, MultimodalAgent._to_internal(question))
        if max(target) == 1.:
            label = _hard_label(question, target)
            if qid not in expected or raw is None or expected[qid] == label:
                expected[qid] = label
                gold.pop(qid, None)
    if expected:
        record["expected"] = expected
    if gold:
        record["gold"] = gold


def export_record(row, directory, destination, model):
    questions = {}
    for qid, question in row["questions"].items():
        question = {**question}
        if (question["type"] == "choice" and isinstance(question.get("criteria"), list)
                and all(isinstance(label, str) for label in question["criteria"])):
            MultimodalAgent._check_question(qid, question)
            question["criteria"] = dict.fromkeys(question["criteria"])
        questions[qid] = question
    state = row["state"]
    request = {"model": model, "state": state}
    record = {key: value for key, value in row.items() if key not in {"state", "questions", "expected", "gold"}}
    record["request"] = request
    _supervision(record, row, questions)
    fixed = {}
    wire_questions = {}
    for qid, question in questions.items():
        wire = wire_questions[qid] = {**question}
        order = wire.pop("option_order", None)
        if order is not None:
            fixed[qid] = order
            if wire["type"] == "choice":
                criteria = wire["criteria"]
                if isinstance(criteria, dict):
                    entries = list(criteria.items())
                    wire["criteria"] = dict(entries[index] for index in order)
                else:
                    wire["criteria"] = [criteria[index] for index in order]
    request["questions"] = wire_questions
    if fixed:
        record["option_orders"] = fixed
    media = {key: state[key] for key in MEDIA_FIELDS if isinstance(state, dict) and state.get(key) is not None}
    if media:
        request["state"] = state.get("text", "")
        parts, files = {}, {}
        for kind, value in media.items():
            values = value if isinstance(value, list) else [value]
            names = ["__laya_media_%s_%d__" % (kind, i) for i in range(len(values))]
            files.update({name: resolved_media(path, directory, destination) for name, path in zip(names, values)})
            specs = [{"part": name} for name in names]
            parts[kind] = specs if isinstance(value, list) else specs[0]
        record["media_files"] = files
        if set(media) == {"image"}:
            request["state"] = [*files, state.get("text", "")]
            request["assets"] = {name: {"type": "image", "part": name} for name in files}
        else:
            request.update(parts)
            request.update({key: state[key] for key in MEDIA_CONTROLS if key in state})
    modality = next((key for key in ("video", "audio", "image") if key in media), "text")
    return record, modality


def build_split(source, target, destination, model, check_training=False):
    counts, compatibility, sources, tasks = Counter(), Counter(), Counter(), Counter()
    checksum, output_checksum = hashlib.sha256(), hashlib.sha256()
    with source.open("rb") as stream, target.open("wb") as output:
        for line_number, line in enumerate(stream, 1):
            checksum.update(line)
            if not line.strip() or line.lstrip().startswith(b"#"):
                continue
            row = {}
            try:
                row = loads(line)
                if isinstance(row, dict) and "_meta" in row:
                    row = normalize_record(row, source.parent, verify_media=False)
                record, modality = export_record(row, source.parent, destination, model)
                normalized = normalize_record(record, destination, verify_media=False)
                if check_training:
                    _validate_grounding_row(normalized)
                request = record["request"]
                if modality in ("video", "audio"):
                    compatibility["native_media"] += 1
                elif not _structured(request["state"]) or not all(map(systemone_question, request["questions"].values())):
                    compatibility["native_question_schema"] += 1
                else:
                    compatibility["systemone_image_parts" if modality == "image" else "systemone_text"] += 1
                for qid, question in request["questions"].items():
                    kind = question["type"]
                    counts[modality + "|" + kind] += 1
                    if kind == "choice" and not can_shuffle_choice(row["questions"][qid]):
                        compatibility["protected_choice_order"] += 1
                    order = record.get("option_orders", {}).get(qid)
                    if kind != "choice" and order is not None and order != list(range(len(order))):
                        compatibility["native_order_in_metadata"] += 1
                    task = (row.get("quality") or {}).get("decision_task") or row.get("benchmark_task") or row.get("skill")
                    if task:
                        tasks[modality + "|" + str(task)] += 1
                sources[str(row.get("source") or row.get("dataset") or "custom")] += 1
                encoded = encode_json(record) + b"\n"
                output.write(encoded)
                output_checksum.update(encoded)
            except (ValueError, TypeError, KeyError, OSError) as error:
                identifier = row.get("id", "?") if isinstance(row, dict) else "?"
                raise ValueError("%s:%d (%s): %s" % (source, line_number, identifier, error)) from error
    if not sources:
        raise ValueError("empty split: " + str(source))
    return {"source": str(source), "source_sha256": checksum.hexdigest(),
            "file": target.name, "sha256": output_checksum.hexdigest(), "records": sum(sources.values()),
            "decisions": sum(counts.values()), "modality_types": dict(counts), "compatibility": dict(compatibility),
            "sources": dict(sources), "tasks": dict(tasks)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="multimodal")
    parser.add_argument("--training", action="store_true", help="check the single-question joint training contract")
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if output.exists():
        parser.error("--out must name a new directory")
    splits = {}
    for argument in args.split:
        name, equals, source = argument.partition("=")
        if not equals or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in splits:
            parser.error("--split needs a unique NAME=JSONL")
        splits[name] = Path(source).resolve(strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".dataset-") as tmp:
        stage = Path(tmp) / "export"
        stage.mkdir()
        report = {}
        for name, source in splits.items():
            result = report[name] = build_split(source, stage / (name + ".jsonl"), output, args.model, args.training)
            print(name, result["records"], result["compatibility"], flush=True)
        (stage / "manifest.json").write_text(json.dumps({"status": "complete", "splits": report},
                                                         ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)


if __name__ == "__main__":
    main()
