"""Production parity and isolation for ordinal calibration and vision reuse."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from laya.grounding_train import GroundingTrainConfig, _activation_context
from laya.multimodal_calibration import (fit_ordinal_calibration, ordinal_probabilities,
                                        validate_ordinal_calibration)
from laya.multimodal_train import metrics
from test_multimodal import tiny_agent


def channel(kind="mix", k=4):
    size = {"identity": 0, "temperature": 1, "mix": 5, "location": 5}[kind]
    return dict(kind=kind, parameters=[math.log(2), .25, -.2, math.log(4), math.log(19)][:size], prior=[1/k]*k)




def test_decoder_canonical_order_no_double_temperature_and_unseen_identity():
    agent = tiny_agent()
    agent.ordinal_calibration = dict(models={"text:4": channel()})
    agent.temperature = [8.] * 3
    agent.modality_calibration = {"text": dict(temperature=[9.] * 3, fitted_types=[0, 1, 2])}
    canonical = torch.tensor([2., -1., 3., 0.])
    q = dict(t="score", crit=["a", "b", "c", "d"])
    expected = agent._decode(canonical, q)
    permutation = [2, 0, 3, 1]
    assert agent._decode(canonical[permutation], dict(q, option_order=permutation)) == expected
    p = ordinal_probabilities(canonical.numpy(), channel())
    assert expected["score"] == round(float(np.arange(4) @ p), 4) and expected["calibrated"]
    unseen = agent._decode(canonical, q, "image")
    assert unseen["score"] == round(float(np.arange(4) @ ordinal_probabilities(canonical.numpy())), 4)
    assert not unseen["calibrated"]
    choice = dict(t="choice", crit={"x": "a", "y": "b"})
    result = agent._decode(canonical[:2], choice)
    agent.ordinal_calibration = {}
    assert agent._decode(canonical[:2], choice) == result


@pytest.mark.parametrize("mutation", ["bucket", "nan", "size", "negative", "sum", "kind"])
def test_invalid_profile_rejected(mutation):
    model = channel()
    key = "text:4"
    if mutation == "bucket": key = "text:04"
    if mutation == "nan": model["parameters"][0] = float("nan")
    if mutation == "size": model["parameters"] = []
    if mutation == "negative": model["prior"] = [-.1, .1, .5, .5]
    if mutation == "sum": model["prior"] = [1.] * 4
    if mutation == "kind": model["kind"] = "unknown"
    with pytest.raises(ValueError): validate_ordinal_calibration(dict(models={key: model}))


def test_fitter_requires_groups_and_falls_back_for_small_buckets():
    row = dict(qtype=1, k=4, modality="text", logits=np.array([0., 1., 2., 3.]), target=np.array([0., .2, .7, .1]))
    with pytest.raises(ValueError, match="groups"): fit_ordinal_calibration([row])
    before = row["target"].copy()
    fitted = fit_ordinal_calibration([dict(row, calibration_group="one-source:one-document")] * 60)
    assert not fitted["models"]  # One independent group cannot populate both folds.
    np.testing.assert_array_equal(row["target"], before)
    assert metrics([row], ordinal_calibration=fitted) == metrics([row])


@pytest.mark.parametrize("threshold", [None, 1, 12])
def test_activation_offload_preserves_cpu_forward_and_gradients(threshold):
    cfg = GroundingTrainConfig(activation_offload_min_tokens=threshold)
    cfg.validate()
    torch.manual_seed(105)
    a = torch.randn(10, 8, requires_grad=True)
    weight = torch.randn(8, 4, requires_grad=True)
    expected = (a @ weight).tanh().square().mean()
    gradients = torch.autograd.grad(expected, (a, weight))
    with _activation_context(dict(input_ids=torch.zeros(1, 10, dtype=torch.long)), cfg):
        actual = (a @ weight).tanh().square().mean()
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for observed, wanted in zip((a.grad, weight.grad), gradients):
        torch.testing.assert_close(observed, wanted, atol=0, rtol=0)


@pytest.mark.parametrize("threshold", [True, 0, -1, 1.5])
def test_activation_offload_validates_threshold(threshold):
    with pytest.raises(ValueError, match="activation_offload"): GroundingTrainConfig(activation_offload_min_tokens=threshold).validate()


class Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, mm_encoder_outputs=None, **kwargs):
        pass

    def get_image_features(self, pixels, geometry, return_dict=True):
        self.calls += 1
        return SimpleNamespace(pooler_output=pixels.sum(-1))

    get_video_features = get_image_features


def test_vision_reuse_checks_pixels_and_grid_and_is_disabled_during_training():
    agent = tiny_agent()
    agent.model.model = Vision()
    agent.model_type, agent.readout = "qwen3_5", "candidate"
    agent.model.eval()
    inputs = dict(input_ids=torch.tensor([[1, 2]]), pixel_values=torch.ones(4, 8), image_grid_thw=torch.tensor([[1, 2, 2]]))
    cache = {}
    with torch.no_grad():
        first = agent._reuse_vision_features(inputs, cache)
        second = agent._reuse_vision_features(copy.deepcopy(inputs), cache)
        assert agent.model.model.calls == 1 and first["mm_encoder_outputs"]["image"] is second["mm_encoder_outputs"]["image"]
        changed = copy.deepcopy(inputs); changed["pixel_values"][0, 0] = 2
        agent._reuse_vision_features(changed, cache)
        assert agent.model.model.calls == 2
        changed["image_grid_thw"] = torch.tensor([[1, 1, 4]])
        agent._reuse_vision_features(changed, cache)
        assert agent.model.model.calls == 3
        assert "pixel_values" in inputs and "mm_encoder_outputs" not in inputs
        assert "pixel_values" not in first and torch.equal(first["image_grid_thw"], inputs["image_grid_thw"])
        agent._reuse_vision_features(inputs, {})
        assert agent.model.model.calls == 4  # A new state/request cannot inherit features.
        agent.model.train()
        assert agent._reuse_vision_features(inputs, cache) is inputs
    agent.model.eval()
    assert agent._reuse_vision_features(inputs, cache) is inputs  # Grad enabled.




def test_public_multiquestion_reuse_preserves_hooks_usage_and_state_isolation(monkeypatch):
    agent = tiny_agent()
    agent.model.model = Vision()
    agent.model_type, agent.readout = "qwen3_5", "candidate"

    def prepare(request, qdef, **kwargs):
        inputs = dict(input_ids=torch.tensor([[1, 2]]), pixel_values=torch.ones(4, 8),
                      image_grid_thw=torch.tensor([[1, 2, 2]]))
        return inputs, dict(t="noul"), {}

    def logits(inputs, count):
        return torch.tensor([[0., float(inputs["mm_encoder_outputs"]["image"].pooler_output.mean())]])

    monkeypatch.setattr(agent, "_prepare", prepare)
    monkeypatch.setattr(agent, "_logits", logits)
    questions = {"a": {"type": "noul", "instructions": "first"}, "b": {"type": "noul", "instructions": "second"}}
    events = []
    results = agent.predict_batch(["same", "same"], questions,
        on_predict_start=lambda ctx: events.append("start"), on_predict_end=lambda ctx: events.append(("end", ctx.usage)))
    assert agent.model.model.calls == 2  # State-local, even for identical states.
    assert len(results) == 2 and all(item["usage"]["forward_passes"] == 2 for item in results)
    assert events[0] == "start" and events[1][0] == "end" and events[1][1]["input_tokens"] == 8
    agent.predict("same", questions)
    assert agent.model.model.calls == 3  # Request-local, too.


@pytest.mark.parametrize("kind", ["image", "video"])
def test_real_native_qwen_vision_reuse_preserves_full_sequence_logits(kind):
    from test_decision_backbones import tiny_qwen_config
    Qwen3_5ForConditionalGeneration = pytest.importorskip(
        "transformers.models.qwen3_5.modeling_qwen3_5").Qwen3_5ForConditionalGeneration
    from laya.multimodal_readout import CandidateScorer
    torch.manual_seed(105)
    torch.set_num_threads(1)
    agent = tiny_agent()
    agent.model = Qwen3_5ForConditionalGeneration(tiny_qwen_config()).eval()
    agent.model.decision_scorer = CandidateScorer(32, 16).eval()
    from laya.grounding import GroundingHead
    agent.model.grounding_head = GroundingHead(32, 16, 1, "single-v1").eval()
    agent.model_type, agent.readout = "qwen3_5", "candidate"
    token = 120 if kind == "image" else 121
    inputs = dict(input_ids=torch.tensor([[2, token, 3, 4, 5, 6]]), attention_mask=torch.ones(1, 6, dtype=torch.long),
                  mm_token_type_ids=torch.tensor([[0, 1 if kind == "image" else 2, 0, 0, 0, 0]]),
                  _candidate_positions=torch.tensor([[3, 4]]))
    vision = agent.model.config.vision_config
    width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    inputs["pixel_values" if kind == "image" else "pixel_values_videos"] = torch.randn(4, width)
    inputs[kind + "_grid_thw"] = torch.tensor([[1, 2, 2]])
    with torch.no_grad():
        expected = agent._logits(inputs, 2)
        cache = {}
        first = agent._reuse_vision_features(inputs, cache)
        repeated = agent._reuse_vision_features(copy.deepcopy(inputs), cache)
        torch.testing.assert_close(agent._logits(first, 2), expected, atol=0, rtol=0)
        torch.testing.assert_close(agent._logits(repeated, 2), expected, atol=0, rtol=0)
        original_box, reused_box = agent._grounding_outputs(inputs), agent._grounding_outputs(first)
        for key in original_box:
            torch.testing.assert_close(original_box[key], reused_box[key], atol=0, rtol=0)
