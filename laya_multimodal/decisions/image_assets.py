"""Decode and bind named image assets for System One and local datasets."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
import io
import re
from typing import Any
from uuid import uuid4

from .questions import serialize_state
from .systemone import IMAGE_HEADER, MAX_REQUEST_BYTES, SystemOneRequestError

MAX_IMAGE_BYTES = 16 * 1024**2
MIN_IMAGE_SIDE = 280
MAX_IMAGE_SIDE = 1568
IMAGE_FORMATS = {"png", "jpg", "jpeg"}


def image_form_parts(content_type, body):
    """Read the bounded multipart body, keeping file parts as binary data."""
    try:
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body
        )
        if message.defects or not message.is_multipart():
            raise SystemOneRequestError("invalid multipart body")
        parts = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if (part.get_content_disposition() != "form-data" or not name or name in parts
                    or part.is_multipart() or part.defects
                    or part.get("Content-Transfer-Encoding", "binary").lower() not in ("7bit", "8bit", "binary")):
                raise SystemOneRequestError("multipart parts must have unique names and contain raw bytes")
            if not parts:
                if name != "request" or part.get_content_type() != "application/json":
                    raise SystemOneRequestError("the first multipart part must be application/json named 'request'")
            parts[name] = part.get_payload(decode=True)
        if "request" not in parts:
            raise SystemOneRequestError("multipart body needs a 'request' part")
        return parts.pop("request"), parts
    except (RecursionError, UnicodeError) as error:
        raise SystemOneRequestError("invalid multipart body") from error


def _asset_bytes(asset, files, used_parts):
    if not isinstance(asset, dict) or set(asset) - {"type", "format", "part", "data_base64"}:
        raise SystemOneRequestError("an asset must declare image file data")
    if asset.get("type", "image") != "image":
        raise SystemOneRequestError("System One assets must be images")
    fmt = asset.get("format")
    if fmt is not None and (not isinstance(fmt, str) or fmt not in IMAGE_FORMATS):
        raise SystemOneRequestError("image format must be png, jpg or jpeg")
    if "part" in asset:
        if "data_base64" in asset:
            raise SystemOneRequestError("an asset must use either a file part or inline image data")
        name = asset["part"]
        if not isinstance(name, str) or name not in files or name in used_parts:
            raise SystemOneRequestError("each multipart image asset must refer to its own file part")
        used_parts.add(name)
        data = files[name]
    else:
        encoded = asset.get("data_base64")
        if not isinstance(encoded, str) or fmt is None:
            raise SystemOneRequestError("inline images need data_base64 and format")
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError:
            raise SystemOneRequestError("invalid image data_base64") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise SystemOneRequestError("image file exceeds 16 MiB", 413)
    return data


@dataclass(frozen=True)
class ImageReferences:
    images: tuple[Any, ...]
    marker: str
    box_targets: dict[str, int] = field(default_factory=dict)

    @property
    def candidate_marker(self):
        return self.marker + "candidate\ue002"

    @property
    def pattern(self):
        return re.compile(re.escape(self.marker) + r"(\d+)\ue002")

    def box_state(self, text, instructions):
        """Keep the selected search image in state. Query images in instructions stay in place."""
        target = self.box_targets.get(instructions)
        if target is None:
            return text
        return self.pattern.sub(lambda match: match[0] if int(match[1]) == target else "[image]", text)

    def split_prompt(self, prompt):
        """Use the processor's typed image tokens at each reference in the rendered question."""
        content, image_ids = [], []
        offset = 0
        for match in self.pattern.finditer(prompt):
            if match.start() > offset:
                content.append({"type": "text", "text": prompt[offset:match.start()]})
            content.append({"type": "image"})
            image_ids.append(int(match[1]))
            offset = match.end()
        if offset < len(prompt):
            content.append({"type": "text", "text": prompt[offset:]})
        return content, image_ids


def prepare_image_request(state, questions, assets, files, *, images_enabled):
    """Decode uploaded still images and mark references in JSON values for local inference."""
    if not images_enabled:
        raise SystemOneRequestError("set " + IMAGE_HEADER + ": true for assets or box questions")
    if not isinstance(assets, dict) or not assets:
        raise SystemOneRequestError("image and box requests need an assets object")
    if any(not ref.strip() for ref in assets):
        raise SystemOneRequestError("image references must be nonempty")
    lengths = [len(ref.encode("utf-8")) for ref in assets]
    if max(lengths) > 1024 or sum(lengths) > 65536:
        raise SystemOneRequestError("image reference length exceeds the request limit", 413)

    from PIL import Image

    used_parts, images, pixel_sizes = set(), [], []
    decoded_size = 0
    for asset in assets.values():
        data = _asset_bytes(asset, files, used_parts)
        try:
            with Image.open(io.BytesIO(data), formats=("PNG", "JPEG")) as image:
                expected_format = asset.get("format")
                if expected_format == "jpg":
                    expected_format = "jpeg"
                if expected_format and image.format.lower() != expected_format:
                    raise SystemOneRequestError("declared image format does not match the file")
                if getattr(image, "is_animated", False):
                    raise SystemOneRequestError("assets must be still images")
                if max(image.size) > MAX_IMAGE_SIDE:
                    raise SystemOneRequestError("image width or height exceeds 1568 pixels", 413)
                if min(image.size) < MIN_IMAGE_SIDE:
                    raise SystemOneRequestError("image width and height must each be at least 280 pixels")
                size = 3 * image.width * image.height
                decoded_size += size
                if decoded_size > MAX_REQUEST_BYTES:
                    raise SystemOneRequestError("decoded images exceed the request limit", 413)
                images.append(image.convert("RGB"))
                pixel_sizes.append(size)
        except Image.DecompressionBombError:
            raise SystemOneRequestError("image dimensions exceed the request limit", 413) from None
        except SystemOneRequestError:
            raise
        except (OSError, SyntaxError, ValueError):
            raise SystemOneRequestError("asset is not a readable PNG or JPEG") from None
    if used_parts != files.keys():
        raise SystemOneRequestError("every multipart file part must be named by an asset")

    return bind_image_references(state, questions, dict(zip(assets, images)), pixel_sizes, resolve_boxes=True)


def image_reference_prefix(state, references):
    """Identify an array of distinct image placeholders followed by one text value."""
    if not isinstance(state, list) or len(state) != len(references) + 1:
        return None
    names, text = state[:-1], state[-1]
    if (not isinstance(text, str) or not all(isinstance(name, str) for name in names)
            or set(names) != references.keys()):
        return None
    return names, text


def _image_paths(value, pattern, paths, path=""):
    """Index only image-bearing values, including objects and arrays containing images."""
    if isinstance(value, str):
        indices = {int(index) for index in pattern.findall(value)}
    elif isinstance(value, dict):
        indices = set()
        for key, child in value.items():
            indices.update(_image_paths(child, pattern, paths, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        indices = set()
        for index, child in enumerate(value):
            indices.update(_image_paths(child, pattern, paths, f"{path}[{index}]"))
    else:
        return set()
    if indices:
        paths.setdefault(path, set()).update(indices)
    return indices


def _backticked_paths(value):
    if isinstance(value, str):
        yield from re.findall(r"`([^`]+)`", value)
    elif isinstance(value, (list, dict)):
        for child in value.values() if isinstance(value, dict) else value:
            yield from _backticked_paths(child)


def _box_target(state_paths, instructions, bound_instructions, references):
    named = set()
    for path in _backticked_paths(instructions):
        relative = path[5:] if path.startswith("state[") else path.removeprefix("state.")
        named.update(state_paths.get(path) or state_paths.get(relative, ()))
    state_images = state_paths.get("", set())
    mentioned = {int(index) for index in references.pattern.findall(bound_instructions)}
    # A named state path wins. A single search image can have extra visual examples
    # in instructions, while an instructions-only Box can reference the search image there.
    candidates = named or (state_images & mentioned if len(state_images) > 1 else state_images or mentioned)
    if len(candidates) != 1:
        raise SystemOneRequestError("Box requires one search image. Use a backticked state path when several images are present.")
    return next(iter(candidates))


def bind_image_references(state, questions, images, pixel_sizes=None, *, resolve_boxes=False):
    """Bind either checked HTTP images or lazy local paths to the same prompt layout."""
    marker = "\ue000" + uuid4().hex + "\ue001"
    refs = {name: index for index, name in enumerate(images)}
    pattern = re.compile("|".join(re.escape(ref) for ref in sorted(refs, key=len, reverse=True)))
    used_refs = set()
    expanded_size = 0

    def replace(match):
        nonlocal expanded_size
        name = match[0]
        index = refs[name]
        used_refs.add(name)
        if pixel_sizes is not None:
            expanded_size += pixel_sizes[index]
            if expanded_size > MAX_REQUEST_BYTES:
                raise SystemOneRequestError("expanded image references exceed the request limit", 413)
        return marker + str(index) + "\ue002"

    def rewrite(value):
        if isinstance(value, str):
            return pattern.sub(replace, value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    try:
        bound_state = rewrite(state)
        text = serialize_state(bound_state)
        references = ImageReferences(tuple(images.values()), marker)
        state_paths = None
        converted = {}
        for qid, q in questions.items():
            if not isinstance(q, dict):
                raise SystemOneRequestError("questions must be objects")
            question = {**q}
            for name in ("instructions", "criteria"):
                if name in q:
                    question[name] = rewrite(q[name])
            if q.get("type") == "box":
                if isinstance(question.get("instructions"), (dict, list)):
                    question["instructions"] = serialize_state(question["instructions"])
                if resolve_boxes:
                    if state_paths is None:
                        state_paths = {}
                        _image_paths(bound_state, references.pattern, state_paths)
                    instructions = question["instructions"]
                    references.box_targets[instructions] = _box_target(
                        state_paths, q["instructions"], instructions, references)
            converted[qid] = question
    except RecursionError:
        raise SystemOneRequestError("image references are nested too deeply") from None
    if used_refs != refs.keys():
        raise SystemOneRequestError("each image asset must be referenced in state, instructions or criteria")
    prefix = image_reference_prefix(state, images)
    if prefix is not None and not references.box_targets and not pattern.search(prefix[1]) and converted == questions:
        names, context = prefix
        return {"text": context, "image": [images[name] for name in names]}, converted
    return {"text": text, "image": references}, converted
