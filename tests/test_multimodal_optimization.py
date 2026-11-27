"""Production parity and isolation for ordinal calibration and vision reuse."""
import copy
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from laya.train import GroundingTrainConfig, _activation_context, _decision_microbatches, _rank_window
from laya._qwen_training import qwen_training_delta
from laya.multimodal import MultimodalAgent
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


def test_missing_ordinal_bucket_preserves_fitted_modality_temperature():
    agent = tiny_agent()
    agent.modality_calibration = {"text": dict(temperature=[1., 2., 1.], fitted_types=[1])}
    logits = torch.tensor([2., -1., 3.])
    question = dict(t="score", crit=["low", "medium", "high"])
    record = dict(qtype=1, k=3, modality="text", logits=logits.double().numpy(), target=np.array([0., 0., 1.]))
    expected = agent._decode(logits, question)
    agent.ordinal_calibration = dict(models={"text:4": channel()})
    assert expected["calibrated"]
    assert agent._decode(logits, question) == expected
    assert metrics([record], modality_calibration=agent.modality_calibration,
                   ordinal_calibration=agent.ordinal_calibration) == metrics(
                       [record], modality_calibration=agent.modality_calibration)

    agent.ordinal_calibration["models"]["text:3"] = channel("identity", k=3)
    identity = agent._decode(logits, question)
    assert not identity["calibrated"]
    assert identity["score"] == round(float(np.arange(3) @ ordinal_probabilities(record["logits"])), 4)


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


def test_activation_offload_counts_all_padded_tokens(monkeypatch):
    offload = object()
    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", lambda *, pin_memory: offload)
    config = GroundingTrainConfig(activation_offload_min_tokens=4096)
    inputs = {"input_ids": torch.zeros(4, 4064, dtype=torch.long)}
    assert _activation_context(inputs, config) is offload
    assert _activation_context({"input_ids": inputs["input_ids"][:1]}, config) is not offload
    with torch.no_grad():
        assert _activation_context(inputs, config) is not offload


@pytest.mark.parametrize("threshold", [True, 0, -1, 1.5])
def test_activation_offload_validates_threshold(threshold):
    with pytest.raises(ValueError, match="activation_offload"): GroundingTrainConfig(activation_offload_min_tokens=threshold).validate()


def test_rank_balancing_keeps_optimizer_windows_and_handles_missing_admission():
    lengths = [16384 if index % 4 == 0 else 512 + index for index in range(32)]
    rows = [{"id": index, "state": "text", "training_admission": {"tokens": length}}
            for index, length in enumerate(lengths)]
    config = GroundingTrainConfig(balance_ranks=True, batch_decisions=True, max_micro_tokens=16384)
    config.validate()
    ranks = [_rank_window(rows, rank, 4, config) for rank in range(4)]
    assert sorted(row["id"] for assigned in ranks for row in assigned) == list(range(len(rows)))
    assert all(sum(row["training_admission"]["tokens"] == 16384 for row in assigned) == 2 for assigned in ranks)
    assert _rank_window(rows, 0, 1, config) == rows
    rows[-1].pop("training_admission")
    assert [_rank_window(rows, rank, 4, config) for rank in range(4)] == [rows[rank::4] for rank in range(4)]


def test_video_prefetch_reuses_the_same_media_and_preserves_targets():
    seen = []
    owner = threading.get_ident()

    def decode(request):
        seen.append((request["video"], threading.get_ident()))
        return [1, 2, 3]

    def prepare(request, question, *, load_media=None):
        pixels = decode(request) if load_media is None else load_media()
        return {"input_ids": torch.tensor([pixels])}, MultimodalAgent._to_internal(question), {}

    agent = SimpleNamespace(_input=dict, _media=decode, _prepare=prepare)
    record = {"state": {"text": "A clip", "video": "clip.mp4"},
              "questions": {"q": {"type": "noul", "instructions": "Does it move?"}},
              "expected": {"q": True}}
    output = []
    for prefetch in (False, True):
        config = GroundingTrainConfig(prefetch_video=prefetch, micro_batch=1)
        output.append(list(_decision_microbatches(agent, [record, record], config)))
    assert seen[:2] == [("clip.mp4", owner)] * 2
    assert len(seen) == 3 and seen[2][0] == "clip.mp4" and seen[2][1] != owner
    for (serial, left), (parallel, right) in zip(*output, strict=True):
        assert serial == parallel
        torch.testing.assert_close(left["input_ids"], right["input_ids"], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen_training_scan_matches_native_forward_and_gradients(dtype):
    native = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    original = native.torch_chunk_gated_delta_rule
    reference = getattr(original, "__wrapped__", original)
    torch.manual_seed(105)
    query = torch.randn(2, 137, 2, 16, dtype=dtype) * .1
    values = [query, torch.randn_like(query) * .1, torch.randn(2, 137, 4, 16, dtype=dtype) * .1,
              -torch.rand(2, 137, 4), torch.rand(2, 137, 4, dtype=dtype),
              torch.randn(2, 4, 16, 16) * .01]

    def result(operator):
        inputs = [value.clone().requires_grad_() for value in values]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            output, state = operator(*inputs[:5], initial_state=inputs[5], output_final_state=True,
                                     use_qk_l2norm_in_kernel=True)
            loss = output.float().square().mean() + state.float().square().mean()
        return output.detach(), state.detach(), torch.autograd.grad(loss, inputs)

    expected_output, expected_state, expected_grads = result(reference)
    with qwen_training_delta():
        output, state, grads = result(native.torch_chunk_gated_delta_rule)
        with torch.no_grad():
            fallback, _ = native.torch_chunk_gated_delta_rule(*values[:5])
            expected_fallback, _ = reference(*values[:5])
        torch.testing.assert_close(fallback, expected_fallback, rtol=0, atol=0)
    assert native.torch_chunk_gated_delta_rule is original
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    for gradient, expected in zip(grads, expected_grads):
        torch.testing.assert_close(gradient, expected, rtol=.01, atol=1e-7)


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
