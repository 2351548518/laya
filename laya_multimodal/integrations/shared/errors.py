"""Exceptions shared by the framework integrations."""
from typing import Any, Dict


class LayaLowConfidenceError(ValueError):
    """An integration rejected an unusable or low-confidence decision.

    Adapters use a configured fallback when available. Their gating settings determine when
    to raise this error and expose the confidence, threshold and original response.
    """

    def __init__(self, message: str, confidence: float, threshold: float, raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold
        self.raw_decision = raw_decision
