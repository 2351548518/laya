"""An explicitly configured Vega reader beside the native runtime."""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from laya_multimodal.decisions.routing import VEGA_MODEL, Router, _estimate_tokens  # noqa: E402

HAS_VEGAML = importlib.util.find_spec("vegaml") is not None
Q = {"dept": {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "money", "technical": "bugs"}},
     "urgent": {"type": "noul", "instructions": "Urgent?"}}
ENGLISH = "Please refund the duplicate charge on invoice 4411 today."
LONG = ("The contract between the parties states the following. " * 3000)   # ~40k tokens


class _StubEngine:
    def __init__(self, heads):
        self.heads = heads


class _StubVega:
    """What the Router needs from a vega agent: `.vega.heads`, `predict`, `predict_long`."""
    model_id = "stub/vega"

    def __init__(self, heads=()):
        self.vega = _StubEngine({h: object() for h in heads})
        self.calls = []

    def predict(self, state=None, questions=None, lang=None, image=None, audio=None, video=None, **kw):
        self.calls.append({"state": state, "image": image, "audio": audio, "video": video, "lang": lang, **kw})
        answers = {q: {"choice": "billing", "probabilities": {"billing": 0.9, "technical": 0.1},
                       "answer_confidence": 0.9, "confidence": 0.9, "route": "auto"} if d["type"] == "choice"
                   else {"noul": 0.2, "answer_confidence": 0.8, "confidence": 0.8, "route": "auto"}
                   for q, d in questions.items()}
        return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}, "latency_ms": 1.0,
                "modality": "image" if image is not None else "audio" if audio is not None else
                "video" if video is not None else "text"}

    system_one = predict

    def predict_long(self, state, questions, **kw):
        out = self.predict(state, questions, **kw)
        out["usage"]["windows"] = 1
        return out


# ---------------------------------------------------------------------- off by default without vegaml
@pytest.mark.skipif(HAS_VEGAML, reason="vegaml is installed here, so auto turns the lane on")
def test_native_router_is_the_default():
    router = Router()
    assert not router.vega_enabled
    assert router.route(ENGLISH, Q).model == "multimodal"
    assert router.route(None, Q, media="image").model == "multimodal"


def test_attached_vega_can_take_a_fitted_or_long_text_request():
    vega = _StubVega(heads=["dept"])
    router = Router(vega=vega)
    assert router.vega_enabled
    assert router.route(ENGLISH, Q).model == VEGA_MODEL
    assert router.route(LONG, {}).model == VEGA_MODEL
    assert router.route({"image": "scene.png"}, Q).model == "multimodal"
    assert router.predict(ENGLISH, Q)["routing"]["model"] == VEGA_MODEL


def test_explicit_vega_media_route():
    vega = _StubVega()
    router = Router(vega=vega)
    result = router.predict("context", Q, image="path.png", model="vega")
    assert result["modality"] == "image"
    assert vega.calls[-1]["image"] == "path.png"


def test_lazy_vega_reader_registration_does_not_load_weights():
    router = Router(vega="4b")
    assert router.resolve("vega") == VEGA_MODEL
    assert router.route(ENGLISH, Q, model="vega")["repo"] == "4b"
    assert _estimate_tokens(LONG) > router.vega_over_tokens
    assert router.loaded == []
