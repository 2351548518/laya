"""TypeSafe System One request validation and context limits."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import gzip
import io
import math
import zlib

IMAGE_HEADER = "TypeSafe-Danger-Zone-Images"
LONG_CONTEXT_HEADER = "TypeSafe-Danger-Zone-Long-Context"
MODEL_IDS = ("jev-mm-qwen-4b", "jev-mm-gemma-e2b")
MODEL_ALIAS = "jev-mm-latest"
TYPESAFE_MODEL_ALIAS = "jev-latest"
MAX_REQUEST_BYTES = 32 * 1024**2
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
_DEFAULT_INSTRUCTIONS = {
    "choice": "Select the option that fits the state.",
    "score": "Assess the state using the ordered criteria.",
    "noul": "Does the state satisfy the true criterion?",
    "box": "Locate the subject in the image.",
}


class SystemOneValidationError(ValueError):
    """A field-specific error using the OpenAPI 422 response envelope."""

    def __init__(self, loc, message, kind="value_error"):
        super().__init__(message)
        self.detail = [{"loc": loc, "msg": message, "type": kind}]


class SystemOneRequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def uses_typesafe_model(model):
    # Older Laya clients send the unversioned jev-1 as an auto-routing hint.
    return isinstance(model, str) and model.lower().startswith("jev-") and model.lower() != "jev-1"


def _validate_json_tree(value, depth):
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise SystemOneRequestError("JSON numbers must be finite")
    if isinstance(value, (dict, list)):
        if depth >= 64:
            raise SystemOneRequestError("JSON values may contain at most 64 nested levels")
        for child in value.values() if isinstance(value, dict) else value:
            _validate_json_tree(child, depth + 1)


def _validate_entry(value, loc, *, nullable=False, depth=0):
    _validate_json_tree(value, depth)
    if not (isinstance(value, (str, dict, list)) or nullable and value is None):
        expected = "Expected a string, object or array"
        raise SystemOneValidationError(loc, expected + (" or null" if nullable else ""))


def adapt_systemone_question(question):
    """Supply native prompt text for omitted instructions and null score levels."""
    if not isinstance(question, dict):
        return question
    kind = question.get("type")
    if not isinstance(kind, str) or kind not in _DEFAULT_INSTRUCTIONS:
        return question
    instructions = question.get("instructions")
    if (instructions is None or isinstance(instructions, str) and not instructions.strip()
            or isinstance(instructions, (list, dict)) and not instructions):
        question = {**question, "instructions": _DEFAULT_INSTRUCTIONS[kind]}
    criteria = question.get("criteria")
    if kind == "score" and isinstance(criteria, list) and None in criteria:
        # Keep each level index, including levels that have an empty description.
        question = {**question, "criteria": ["" if level is None else level for level in criteria]}
    return question


def validate_systemone_question(question, loc):
    """Validate a primitive and adapt optional descriptions for the local readouts."""
    if not isinstance(question, dict):
        raise SystemOneValidationError(loc, "Expected a question object")
    kind = question.get("type")
    if not isinstance(kind, str) or kind not in _DEFAULT_INSTRUCTIONS:
        raise SystemOneValidationError([*loc, "type"], "Expected choice, score, noul or box")
    instructions = question.get("instructions")
    _validate_entry(instructions, [*loc, "instructions"], nullable=True)
    criteria = question.get("criteria")
    criteria_loc = [*loc, "criteria"]
    if kind in ("choice", "score"):
        typ = dict if kind == "choice" else list
        limit = MAX_CHOICE_OPTIONS if kind == "choice" else MAX_SCORE_LEVELS
        if not isinstance(criteria, typ) or not 1 <= len(criteria) <= limit:
            raise SystemOneValidationError(criteria_loc, f"Expected {typ.__name__} with 1 to {limit} entries")
    elif kind == "noul" and criteria is not None:
        if not isinstance(criteria, dict) or not criteria.keys() <= {"true", "false"}:
            raise SystemOneValidationError(criteria_loc, "Expected an object with optional true and false keys")
    elif kind == "box" and "criteria" in question:
        raise SystemOneValidationError(criteria_loc, "Box does not take criteria")
    if criteria is not None:
        for key, description in criteria.items() if isinstance(criteria, dict) else enumerate(criteria):
            _validate_entry(description, [*criteria_loc, key], nullable=True, depth=1)
    result = {"type": kind, "instructions": instructions}
    if criteria is not None:
        result["criteria"] = criteria
    for key in ("option_order", "labels"):
        if key in question:
            result[key] = question[key]
    return adapt_systemone_question(result)


def validate_systemone_request(body):
    """Validate the canonical envelope and return the questions passed to inference."""
    for name in ("model", "state", "questions"):
        if name not in body:
            raise SystemOneValidationError(["body", name], "Field required", "missing")
    if not isinstance(body["model"], str) or not body["model"].strip():
        raise SystemOneValidationError(["body", "model"], "Model name must be nonempty")
    _validate_entry(body["state"], ["body", "state"])
    questions = body["questions"]
    if not isinstance(questions, dict) or not questions:
        raise SystemOneValidationError(["body", "questions"], "At least one question is required", "too_short")
    result = {}
    for name, question in questions.items():
        loc = ["body", "questions", name]
        if not isinstance(name, str) or not name.strip():
            raise SystemOneValidationError(loc, "Expected a nonempty question name")
        if isinstance(question, dict) and question.get("type") == "choice":
            criteria = question.get("criteria")
            # The local SDKs also accept ordered lists of unique string labels.
            if (isinstance(criteria, list) and 1 <= len(criteria) <= MAX_CHOICE_OPTIONS
                    and all(isinstance(label, str) for label in criteria)
                    and len(set(criteria)) == len(criteria)):
                question = {**question, "criteria": dict.fromkeys(criteria)}
        result[name] = validate_systemone_question(question, loc)
    if "assets" in body and (not isinstance(body["assets"], dict) or not body["assets"]):
        raise SystemOneValidationError(["body", "assets"], "Expected a nonempty object of image assets")
    return result


def decode_systemone_body(body, encoding, limit):
    """Decode at most one compression layer within the request size limit."""
    encoding = encoding.strip().lower()
    if encoding in ("", "identity"):
        return body
    if encoding in ("gzip", "x-gzip"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                decoded = stream.read(limit + 1)
        except (OSError, EOFError, zlib.error) as error:
            raise SystemOneRequestError("Invalid gzip request body") from error
    elif encoding == "zstd":
        import zstandard

        try:
            frame = zstandard.get_frame_parameters(body)
            if frame.window_size > 8 * 1024**2:
                raise SystemOneRequestError("The zstd window must be at most 8 MiB")
            if frame.content_size == zstandard.CONTENTSIZE_UNKNOWN:
                decompressor = zstandard.ZstdDecompressor()
                with decompressor.stream_reader(body) as stream:
                    decoded = stream.read(limit + 1)
                if len(decoded) <= limit:
                    # A stream can end without signalling a truncated frame. Verify its end.
                    decompressor.decompress(body, max_output_size=max(1, len(decoded)), allow_extra_data=False)
            elif frame.content_size > limit:
                raise SystemOneRequestError("Request body too large", 413)
            else:
                decoded = zstandard.ZstdDecompressor().decompress(body, allow_extra_data=False)
        except zstandard.ZstdError as error:
            raise SystemOneRequestError("Invalid zstd request body") from error
    else:
        raise SystemOneRequestError("Expected one supported Content-Encoding", 415)
    if len(decoded) > limit:
        raise SystemOneRequestError("Request body too large", 413)
    return decoded


class ContextLengthError(ValueError):
    def __init__(self, tokens, limit, scope="per_question_context"):
        super().__init__(f"multimodal input needs {tokens} tokens, exceeds max_len={limit}; shorten text/media or increase the budget")
        self.tokens, self.limit, self.scope = tokens, limit, scope


@dataclass
class _ContextBudget:
    per_question: int
    per_request: int
    used: int = 0

    def reserve(self, length, checkpoint_limit):
        limit = min(checkpoint_limit, self.per_question)
        if length > limit:
            raise ContextLengthError(length, limit)
        if self.used + length > self.per_request:
            raise ContextLengthError(self.used + length, self.per_request, "per_request_context")
        self.used += length


_CONTEXT_BUDGET = ContextVar("system_one_context_budget", default=None)


def run_systemone(predict, long_context=False):
    per_question, per_request = (262144, 262144) if long_context else (32768, 65536)
    reset = _CONTEXT_BUDGET.set(_ContextBudget(per_question, per_request))
    try:
        return predict()
    finally:
        _CONTEXT_BUDGET.reset(reset)


def check_context_length(length, checkpoint_limit, backbone_limit=None):
    budget = _CONTEXT_BUDGET.get()
    if budget is not None:
        if isinstance(backbone_limit, int) and backbone_limit > 0:
            checkpoint_limit = min(checkpoint_limit, backbone_limit)
        budget.reserve(length, checkpoint_limit)
    elif length > checkpoint_limit:
        raise ContextLengthError(length, checkpoint_limit)
