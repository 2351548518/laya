"""Export the text and image part of a frozen benchmark for System One and Kev."""
import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laya.agent import Agent
from laya.evals import file_fingerprint
from laya.systemone import MAX_CHOICE_OPTIONS, MAX_IMAGE_BYTES, MAX_IMAGE_REQUEST_BYTES, MAX_IMAGE_SIDE, MIN_IMAGE_SIDE
from laya.train import target_from_expected
from scripts.multimodal.prepare_dataset import systemone_question


@dataclass(frozen=True)
class ImageFile:
    path: str
    resized: tuple
    size: tuple
    offset: tuple
    adapted: bool = False

    def box(self, bounds):
        w, h = self.resized
        W, H = self.size
        x, y = self.offset
        l, t, r, b = bounds
        return [(l * w + x) / W, (t * h + y) / H, (r * w + x) / W, (b * h + y) / H]


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


@lru_cache(maxsize=None)
def prepare_image(path, directory):
    with Image.open(path) as image:
        if getattr(image, "is_animated", False):
            raise ValueError("image is animated")
        if (image.format in {"PNG", "JPEG"} and MIN_IMAGE_SIDE <= min(image.size)
                and max(image.size) <= MAX_IMAGE_SIDE and path.stat().st_size <= MAX_IMAGE_BYTES):
            return ImageFile(str(path), image.size, image.size, (0, 0))

        scale = min(MAX_IMAGE_SIDE / max(image.size), max(1., MIN_IMAGE_SIDE / min(image.size)))
        resized = tuple(max(1, min(MAX_IMAGE_SIDE, round(side * scale))) for side in image.size)
        size = tuple(max(MIN_IMAGE_SIDE, side) for side in resized)
        offset = tuple((canvas - side) // 2 for canvas, side in zip(size, resized))
        filename = hashlib.sha256(str(path).encode()).hexdigest() + ".png"
        if not (directory / filename).exists():
            pixels = image.convert("RGB")
            if pixels.size != resized:
                pixels = pixels.resize(resized, Image.Resampling.LANCZOS)
            if size != resized:
                canvas = Image.new("RGB", size)
                canvas.paste(pixels, offset)
                pixels = canvas
            pixels.save(directory / filename, format="PNG")
    return ImageFile("media/" + filename, resized, size, offset, True)


def request_questions(questions):
    clean = {}
    for qid, question in questions.items():
        if question.get("option_order") is not None:
            raise ValueError("option_order is not part of the System One wire format")
        clean[qid] = {k: question[k] for k in ("type", "instructions", "criteria") if k in question}
        if question["type"] == "choice" and isinstance(question.get("criteria"), list):
            clean[qid]["criteria"] = {label: None for label in question["criteria"]}
    return clean


def fits_choice_options(questions):
    return all(q["type"] != "choice" or len(q["criteria"]) <= MAX_CHOICE_OPTIONS for q in questions.values())


def systemone_record(row, suite, source_dir, media_dir, model):
    state = row["state"]
    questions = request_questions(row["questions"])
    request = dict(model=model, state=state, questions=questions)
    result = {k: row[k] for k in ("id", "source", "group", "benchmark_task", "category", "provenance",
                                "expected", "gold", "grounding", "upstream_id", "official_task_id") if k in row}
    result.update(suite=suite, request=request)
    if not isinstance(state, dict) or not state.get("image"):
        return result, 0
    raw_images = state["image"]
    images = raw_images if isinstance(raw_images, list) else [raw_images]
    asset_ids, assets, media_files = [], {}, {}
    prepared = []
    for index, source in enumerate(images):
        path = (source_dir / source).resolve(strict=True)
        media = prepare_image(path, media_dir)
        asset_id = "image-" + hashlib.sha256(str(path).encode()).hexdigest()[:24]
        if asset_id in assets:
            asset_id += "-" + str(index)
        assets[asset_id] = {"type": "image", "part": asset_id}
        media_files[asset_id] = media.path
        asset_ids.append(asset_id)
        prepared.append(media)
    if sum(3 * media.size[0] * media.size[1] for media in prepared) > MAX_IMAGE_REQUEST_BYTES:
        raise ValueError("expanded images exceed the local request limit")
    request["state"] = [*asset_ids, state.get("text", "")]
    request["assets"] = assets
    result["media_files"] = media_files
    if "grounding" in row and prepared[0].size != prepared[0].resized:
        if len(prepared) != 1:
            raise ValueError("grounding with padding needs a single image")
        result["grounding"] = dict(row["grounding"])
        for key in ("boxes", "ignored"):
            if key in result["grounding"]:
                result["grounding"][key] = [prepared[0].box(box) for box in result["grounding"][key]]
    return result, sum(media.adapted for media in prepared)


def kev_record(record):
    request = record["request"]
    if "assets" in request:
        return None
    questions = {}
    labels = record.get("expected", {})
    for qid, question in request["questions"].items():
        kind = question["type"]
        if kind not in ("choice", "noul", "score") or qid not in labels:
            return None
        target = target_from_expected(Agent._to_internal(question), labels[qid])
        if max(target) != 1.:
            return None
        index = target.index(1.)
        label = (list(question["criteria"])[index] if kind == "choice" else
                 bool(index) if kind == "noul" else index)
        questions[qid] = {**question, "label": label, "src": record.get("benchmark_task") or record["suite"]}
    return {"state": request["state"], "questions": questions,
            "_meta": {"id": record["id"], "group_id": record.get("group", record["id"]),
                      "source": record["suite"], "variant": "clean"}}


def build(source, stage, model):
    manifest = source / "manifest.json"
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    if spec.get("status") != "frozen":
        raise ValueError("the source benchmark manifest must be frozen")
    for name in ("systemone", "kev", "media"):
        (stage / name).mkdir()
    suites, total = {}, Counter()
    for suite, entry in spec["suites"].items():
        data = source / entry["path"]
        if file_fingerprint(data) != entry["sha256"]:
            raise ValueError("benchmark fingerprint changed: " + suite)
        systemone, kev = stage / "systemone" / (suite + ".jsonl"), stage / "kev" / (suite + ".jsonl")
        count = Counter()
        with systemone.open("w", encoding="utf-8") as wire, kev.open("w", encoding="utf-8") as text:
            for row in jsonl(data):
                state = row["state"]
                count["source_records"] += 1
                if isinstance(state, dict) and (state.get("audio") or state.get("video")):
                    count["native_audio_video"] += 1
                    continue
                if not fits_choice_options(row["questions"]):
                    count["outside_systemone_choice_limit"] += 1
                    continue
                if not all(map(systemone_question, request_questions(row["questions"]).values())):
                    count["outside_systemone_question_schema"] += 1
                    continue
                record, adapted = systemone_record(row, suite, data.parent, stage / "media", model)
                write_record(wire, record)
                count["systemone_image" if "media_files" in record else "systemone_text"] += 1
                count["adapted_image_references"] += adapted
                compat = kev_record(record)
                if compat is not None:
                    write_record(text, compat)
                    count["kev_text"] += 1
                elif "media_files" not in record:
                    count["text_without_hard_label"] += 1
        paths = {}
        for name, path in (("systemone", systemone), ("kev", kev)):
            if path.stat().st_size:
                paths[name] = {"path": str(path.relative_to(stage)), "sha256": file_fingerprint(path)}
            else:
                path.unlink()
        if count["source_records"] != entry["records"]:
            raise ValueError("unexpected source record count: " + suite)
        suites[suite] = {"counts": dict(count), "files": paths}
        total.update(count)
        print(suite, dict(count), flush=True)
    return {"status": "ready", "source_manifest": str(manifest),
            "source_manifest_sha256": file_fingerprint(manifest), "model": model,
            "counts": dict(total), "adapted_image_files": len(list((stage / "media").glob("*.png"))),
            "suites": suites}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="frozen benchmark directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="multimodal")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.out.resolve()
    if output.exists():
        parser.error("--out must name a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".systemone-") as tmp:
        stage = Path(tmp) / "export"
        stage.mkdir()
        report = build(source, stage, args.model)
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
        stage.rename(output)


if __name__ == "__main__":
    main()
