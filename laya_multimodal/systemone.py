"""TypeSafe System One image uploads and reference placement."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import io
import re
from typing import Any
from uuid import uuid4

from .common import serialize_state

IMAGE_HEADER = "TypeSafe-Danger-Zone-Images"
MAX_IMAGE_REQUEST_BYTES = 32 * 1024**2
MAX_IMAGE_BYTES = 16 * 1024**2
MIN_IMAGE_SIDE = 280
MAX_IMAGE_SIDE = 1568
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
IMAGE_FORMATS = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG"}


class ImageRequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def image_form_parts(content_type, body):
    """Read the bounded multipart body, keeping file parts as binary data."""
    try:
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body
        )
        if message.defects or not message.is_multipart():
            raise ImageRequestError("invalid multipart body")
        parts = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if (part.get_content_disposition() != "form-data" or not name or name in parts
                    or part.is_multipart() or part.defects
                    or part.get("Content-Transfer-Encoding", "binary").lower() not in ("7bit", "8bit", "binary")):
                raise ImageRequestError("multipart parts must have unique names and contain raw bytes")
            if not parts:
                if name != "request" or part.get_content_type() != "application/json":
                    raise ImageRequestError("the first multipart part must be application/json named 'request'")
            elif part.get("Content-Type") and part.get_content_type() not in {
                    "image/png", "image/jpeg", "image/jpg", "application/octet-stream"}:
                raise ImageRequestError("image parts must contain PNG or JPEG")
            parts[name] = part.get_payload(decode=True)
        if "request" not in parts:
            raise ImageRequestError("multipart body needs a 'request' part")
        return parts.pop("request"), parts
    except (RecursionError, UnicodeError) as error:
        raise ImageRequestError("invalid multipart body") from error


def _asset_bytes(asset, files, used_parts):
    if not isinstance(asset, dict) or set(asset) - {"type", "format", "part", "data_base64"}:
        raise ImageRequestError("an asset must declare image file data")
    if asset.get("type", "image") != "image":
        raise ImageRequestError("System One assets must be images")
    fmt = asset.get("format")
    if fmt is not None and (not isinstance(fmt, str) or fmt not in IMAGE_FORMATS):
        raise ImageRequestError("image format must be png, jpg or jpeg")
    if "part" in asset:
        name = asset["part"]
        if not isinstance(name, str) or name not in files or name in used_parts:
            raise ImageRequestError("each multipart image asset must refer to its own file part")
        used_parts.add(name)
        data = files[name]
    else:
        encoded = asset.get("data_base64")
        if not isinstance(encoded, str) or fmt is None:
            raise ImageRequestError("inline images need data_base64 and format")
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError:
            raise ImageRequestError("invalid image data_base64") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageRequestError("image file exceeds 16 MiB", 413)
    return data, IMAGE_FORMATS.get(fmt)


@dataclass(frozen=True)
class ImageReferences:
    images: tuple[Any, ...]
    marker: str

    @property
    def candidate_marker(self):
        return self.marker + "candidate\ue002"

    def split_prompt(self, prompt):
        """Use the processor's typed image tokens at each reference in the rendered question."""
        pattern = re.compile(re.escape(self.marker) + r"(\d+)\ue002")
        content, image_ids = [], []
        offset = 0
        for match in pattern.finditer(prompt):
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
        raise ImageRequestError("set " + IMAGE_HEADER + ": true for assets or box questions")
    if not isinstance(assets, dict) or not assets:
        raise ImageRequestError("image and box requests need an assets object")
    if any(not ref.strip() for ref in assets):
        raise ImageRequestError("image references must be nonempty")
    lengths = [len(ref.encode("utf-8")) for ref in assets]
    if max(lengths) > 1024 or sum(lengths) > 65536:
        raise ImageRequestError("image reference length exceeds the request limit", 413)

    from PIL import Image

    used_parts, images, pixel_sizes = set(), [], []
    decoded_size = 0
    for asset in assets.values():
        data, fmt = _asset_bytes(asset, files, used_parts)
        try:
            with Image.open(io.BytesIO(data), formats=("PNG", "JPEG")) as image:
                if fmt is not None and image.format != fmt:
                    raise ImageRequestError("image bytes do not match the declared format")
                if getattr(image, "is_animated", False):
                    raise ImageRequestError("assets must be still images")
                if max(image.size) > MAX_IMAGE_SIDE:
                    raise ImageRequestError("image width or height exceeds 1568 pixels", 413)
                if min(image.size) < MIN_IMAGE_SIDE:
                    raise ImageRequestError("image width and height must each be at least 280 pixels")
                size = 3 * image.width * image.height
                decoded_size += size
                if decoded_size > MAX_IMAGE_REQUEST_BYTES:
                    raise ImageRequestError("decoded images exceed the request limit", 413)
                images.append(image.convert("RGB"))
                pixel_sizes.append(size)
        except Image.DecompressionBombError:
            raise ImageRequestError("image dimensions exceed the request limit", 413) from None
        except (OSError, SyntaxError, ValueError) as exc:
            if isinstance(exc, ImageRequestError):
                raise
            raise ImageRequestError("asset is not a readable PNG or JPEG") from None
    if used_parts != files.keys():
        raise ImageRequestError("every multipart file part must be named by an asset")

    return bind_image_references(state, questions, dict(zip(assets, images)), pixel_sizes)


def image_reference_prefix(state, references):
    """Identify an array of distinct image placeholders followed by one text value."""
    if not isinstance(state, list) or len(state) != len(references) + 1:
        return None
    names, text = state[:-1], state[-1]
    if (not isinstance(text, str) or not all(isinstance(name, str) for name in names)
            or set(names) != references.keys()):
        return None
    return names, text


def bind_image_references(state, questions, images, pixel_sizes=None):
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
            if expanded_size > MAX_IMAGE_REQUEST_BYTES:
                raise ImageRequestError("expanded image references exceed the request limit", 413)
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
        text = serialize_state(rewrite(state))
        converted = {}
        for qid, q in questions.items():
            if not isinstance(q, dict):
                raise ImageRequestError("questions must be objects")
            question = {**q}
            for name in ("instructions", "criteria"):
                if name in q:
                    question[name] = rewrite(q[name])
            if q.get("type") == "box" and isinstance(question.get("instructions"), (dict, list)):
                question["instructions"] = serialize_state(question["instructions"])
            converted[qid] = question
    except RecursionError:
        raise ImageRequestError("image references are nested too deeply") from None
    if used_refs != refs.keys():
        raise ImageRequestError("each image asset must be referenced in state, instructions or criteria")
    prefix = image_reference_prefix(state, images)
    if prefix is not None and not pattern.search(prefix[1]) and converted == questions:
        names, context = prefix
        return {"text": context, "image": [images[name] for name in names]}, converted
    return {"text": text, "image": ImageReferences(tuple(images.values()), marker)}, converted
