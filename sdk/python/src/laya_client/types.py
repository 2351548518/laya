"""The HTTP payload's required fields and optional Laya extensions.

Responses remain ordinary dictionaries. Unknown JSON fields are preserved.
"""
from typing import Dict, List, Literal, TypedDict, Union

JsonValue = Union[None, bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]]
State = Union[str, List[JsonValue], Dict[str, JsonValue]]
Questions = Dict[str, Dict[str, JsonValue]]
MinConfidence = Union[float, Dict[str, float]]


class _AnswerExtensions(TypedDict, total=False):
    answer_confidence: float
    x_jev_confidence: float
    low_confidence: bool
    abstention: Literal["passed", "abstained", "unevaluated"]
    abstention_threshold: float
    action: Dict[str, JsonValue]
    x_tournament: Dict[str, JsonValue]


class _ConfidenceAnswer(_AnswerExtensions):
    confidence: float


class ChoiceAnswer(_ConfidenceAnswer):
    type: Literal["choice"]
    choice: str
    probabilities: Dict[str, float]


class ScoreAnswer(_ConfidenceAnswer):
    type: Literal["score"]
    score: float
    probabilities: Dict[str, float]
    legend: Dict[str, JsonValue]


class _NoulExtensions(_AnswerExtensions, total=False):
    confidence: float


class NoulAnswer(_NoulExtensions):
    type: Literal["noul"]
    noul: float


class Box(TypedDict):
    left: float
    top: float
    right: float
    bottom: float


class BoxAnswer(_ConfidenceAnswer):
    """Normalized coordinates and localization quality for a referred object."""

    type: Literal["box"]
    box: Box


class _ImageAssetType(TypedDict, total=False):
    type: Literal["image"]


class ImageAsset(_ImageAssetType):
    format: Literal["png", "jpg", "jpeg"]
    data_base64: str


class SystemOneBoxAnswer(TypedDict):
    type: Literal["box"]
    box: Box
    confidence: Union[float, None]


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class _PredictionExtensions(TypedDict, total=False):
    routing: Dict[str, JsonValue]


class Prediction(_PredictionExtensions):
    model: str
    answers: Dict[str, Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]]
    usage: Usage


class SystemOneImagePrediction(TypedDict):
    model: str
    answers: Dict[str, Union[ChoiceAnswer, ScoreAnswer, NoulAnswer, SystemOneBoxAnswer]]
    usage: Usage


class MultimodalUsage(Usage):
    forward_passes: int
    truncated: bool


class MultimodalPrediction(_PredictionExtensions):
    model: str
    answers: Dict[str, Union[ChoiceAnswer, ScoreAnswer, NoulAnswer, BoxAnswer]]
    usage: MultimodalUsage


class BatchPrediction(TypedDict):
    results: List[Prediction]
    total_usage: Usage


class Health(TypedDict):
    status: Literal["ok"]
