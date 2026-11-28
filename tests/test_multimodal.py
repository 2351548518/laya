"""Multimodal decision contracts, media processing, and real gradient updates without weights."""
import copy
import errno
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("PIL", reason="native media tests need the multimodal extra")
pytest.importorskip("scipy", reason="audio resampling tests need the multimodal extra")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laya_multimodal.models.multimodal import MultimodalAgent, install_decision_head, read_audio  # noqa: E402
from laya_multimodal.data import check_splits, collate_multimodal_inputs, load_rows  # noqa: E402
from laya_multimodal.evaluation.native_metrics import collect_records  # noqa: E402
from laya_multimodal.decisions.routing import Router  # noqa: E402

Q = {"category": {"type": "choice", "instructions": "Choose the category", "criteria": ["dark", "bright"]},
     "present": {"type": "noul", "instructions": "Is the event present?"},
     "level": {"type": "score", "instructions": "Rate the level", "criteria": ["low", "high"]}}


class TinyTokenizer:
    def encode(self, text, **kwargs):
        return [ord(text)]


class TinyProcessor:
    tokenizer = TinyTokenizer()

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "A B question"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        value = 0
        if kwargs.get("images"):
            value = int(np.asarray(kwargs["images"][0]).mean() > 100)
        elif kwargs.get("audio"):
            value = int(np.abs(kwargs["audio"][0]).mean() > 0.1)
        return {"input_ids": torch.tensor([[1, value + 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(256, 8)
        self.lm_head = torch.nn.Linear(8, 256, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.calls = 0

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, head):
        self.lm_head = head

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        return SimpleNamespace(logits=self.lm_head(self.embeddings(input_ids)))

    def generate(self, **kwargs):
        raise AssertionError("decisions must never generate tokens")


def tiny_agent():
    torch.manual_seed(1)
    agent = MultimodalAgent.__new__(MultimodalAgent)
    agent.processor = TinyProcessor()
    agent.tok = agent.processor.tokenizer
    agent.model = TinyBackbone()
    agent.symbols = install_decision_head(agent.model, agent.tok)
    agent.device, agent.dtype = torch.device("cpu"), torch.float32
    agent.model_type, agent.model_id, agent.revision = "gemma4", "tiny", None
    agent.max_len, agent.frames, agent.max_audio_seconds = 128, 3, 30.0
    agent.use_audio_in_video = True
    agent.temperature, agent.temperature_by_options, agent.calibrated = [1.0] * 3, {}, False
    agent.hooks, agent._forward_options, agent._lock = [], {}, threading.RLock()
    agent.cfg = {"max_len": 128}
    return agent


def test_head_replaces_vocabulary_without_changing_input_embeddings():
    model, tokenizer = TinyBackbone(), TinyTokenizer()
    original = model.lm_head.weight.detach().clone()
    embeddings = model.embeddings.weight.detach().clone()
    symbols = install_decision_head(model, tokenizer)
    assert model.lm_head.out_features == 62
    torch.testing.assert_close(model.lm_head.weight[0], original[ord(symbols[0])])
    torch.testing.assert_close(model.embeddings.weight, embeddings)
    assert not model.config.tie_word_embeddings


def test_native_image_audio_and_text_produce_typed_distributions():
    agent = tiny_agent()
    image = np.full((8, 8, 3), 255, np.uint8)
    result = agent.predict("visual context", Q, image=image, min_confidence=1.0)
    assert result["usage"]["output_tokens"] == 0 and result["usage"]["forward_passes"] == 3
    assert result["answers"]["category"]["choice"] in ("dark", "bright")
    assert sum(result["answers"]["category"]["probabilities"].values()) == pytest.approx(1, abs=1e-4)
    assert 0 <= result["answers"]["present"]["noul"] <= 1
    assert 0 <= result["answers"]["level"]["score"] <= 1
    assert result["answers"]["category"]["abstention"] == "abstained"
    assert not result["answers"]["category"]["calibrated"]
    assert agent.processor.calls[-1]["images"][0].size == (8, 8)
    agent.predict(audio=np.full(1600, 0.5), sampling_rate=16000, questions={"x": Q["present"]})
    np.testing.assert_allclose(agent.processor.calls[-1]["audio"][0], 0.5)
    assert "transcript" not in result
    agent.predict("text", {"x": Q["present"]})
    assert "audio" not in agent.processor.calls[-1]


def test_input_changes_reach_the_decision_head():
    agent = tiny_agent()
    questions = {"category": Q["category"]}
    low = agent.predict(image=np.zeros((4, 4, 3), np.uint8), questions=questions)
    high = agent.predict(image=np.full((4, 4, 3), 255, np.uint8), questions=questions)
    assert low["answers"]["category"]["probabilities"] != high["answers"]["category"]["probabilities"]


def test_option_permutation_is_restored_and_budget_never_silently_truncates():
    agent = tiny_agent()
    q = copy.deepcopy(Q["category"])
    ordinary = agent.predict("s", {"x": q})["answers"]["x"]["probabilities"]
    q["option_order"] = [1, 0]
    permuted = agent.predict("s", {"x": q})["answers"]["x"]["probabilities"]
    assert ordinary["dark"] == permuted["bright"]
    with pytest.raises(ValueError, match="exceeds max_len"):
        agent.predict("s", {"x": q}, max_len=1)
    with pytest.raises(ValueError, match="encoder-only"):
        agent.predict("s", Q, head_max_len=20)


def test_audio_validation_resampling_and_capability():
    assert len(read_audio(np.ones(800), 8000)) == 1600
    for wave in (np.array([]), np.array([np.nan]), np.ones((2, 2, 2))):
        with pytest.raises(ValueError):
            read_audio(wave, 16000)
    with pytest.raises(ValueError, match="sampling_rate"):
        read_audio(np.ones(100))
    with pytest.raises(ValueError, match="max_audio_seconds"):
        read_audio(np.ones(32000), 16000, max_seconds=1)
    agent = tiny_agent()
    agent.model_type = "qwen3_5"
    with pytest.raises(ValueError, match="no native audio"):
        agent.predict(audio=np.ones(100), sampling_rate=16000, questions=Q)


def test_video_preserves_order_and_audio_together(tmp_path, monkeypatch):
    import laya_multimodal.models.video as video
    from PIL import Image
    path = tmp_path / "clip.avi"
    path.write_bytes(b"test")
    frames = [Image.new("RGB", (4, 4), c) for c in ("red", "green", "blue")]
    monkeypatch.setattr(video, "probe", lambda p: {"duration_s": 3, "has_audio": True})
    monkeypatch.setattr(video, "sample_frames", lambda *args: (frames, [0.5, 1.5, 2.5]))
    monkeypatch.setattr(video, "extract_audio", lambda p: np.ones(1600, np.float32))
    agent = tiny_agent()
    agent._video_cache_size = 1
    result = agent.predict(video=path, questions={"x": Q["category"]})
    call = agent.processor.calls[-1]
    assert len(call["images"]) == 3 and len(call["audio"]) == 1
    assert result["media"]["video"]["frame_times"] == [0.5, 1.5, 2.5]
    assert call["images"][0].getpixel((0, 0)) == (255, 0, 0)
    monkeypatch.setattr(video, "sample_frames", lambda *args: pytest.fail("unchanged cached video was decoded again"))
    agent.predict(video=path, questions={"x": Q["category"]})
    path.write_bytes(b"changed recording")
    with pytest.raises(pytest.fail.Exception, match="decoded again"):
        agent.predict(video=path, questions={"x": Q["category"]})


def test_serial_evaluation_reuses_video_decodes_and_preserves_predictions(tmp_path, monkeypatch):
    import laya_multimodal.models.video as video
    from PIL import Image

    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.touch()
    second.touch()
    decodes = []
    monkeypatch.setattr(video, "probe", lambda path: {"duration_s": 3, "has_audio": False})

    def sample(path, frames, _duration):
        decodes.append((Path(path), frames))
        color = "white" if Path(path) == first and frames == 3 else "black"
        return [Image.new("RGB", (4, 4), color) for _ in range(frames)], list(range(frames))

    monkeypatch.setattr(video, "sample_frames", sample)
    rows = [{"id": str(i), "state": {"text": str(i), "video": str(path), "video_frames": frames,
                                    "video_audio": False},
             "questions": {"x": Q["present"]}, "expected": {"x": i % 2 == 0}}
            for i, (path, frames) in enumerate(((first, 3), (second, 3), (first, 3), (first, 1)))]
    reference = tiny_agent()
    original = collect_records(reference, rows, micro_batch=1)
    assert decodes == [(first, 3), (second, 3), (first, 3), (first, 1)]
    decodes.clear()
    cached = tiny_agent()
    cached._video_cache_size = 1
    optimized = collect_records(cached, rows, micro_batch=1)
    assert decodes == [(first, 3), (second, 3), (first, 1)]
    assert [r["id"] for r in optimized] == [r["id"] for r in original]
    np.testing.assert_array_equal([r["logits"] for r in optimized], [r["logits"] for r in original])
    np.testing.assert_array_equal([r["target"] for r in optimized], [r["target"] for r in original])


def test_native_video_bytes_are_decoded_once_for_each_state(monkeypatch):
    pytest.importorskip("transformers.video_utils")
    from PIL import Image

    agent = tiny_agent()
    agent.model_type = "qwen3_5"
    agent.use_audio_in_video = False
    decoded = []

    def read_clip(path, **kwargs):
        data = Path(path).read_bytes()
        decoded.append((path, data))
        frames = [Image.new("RGB", (4, 4), (data[0], 0, 0)) for _ in range(3)]
        return {"duration_s": 3, "has_audio": False}, frames, [0.5, 1.5, 2.5], None

    monkeypatch.setattr(agent, "_read_video", read_clip)
    states = [{"video": video, "video_encoding": "native"} for video in (b"a", b"z")]
    results = agent.predict_batch(states, Q)
    assert [data for _, data in decoded] == [b"a", b"z"]
    assert all(not Path(path).exists() for path, _ in decoded)
    assert all(result["usage"]["forward_passes"] == len(Q) for result in results)
    assert all(list(result["media"]) == ["video"] for result in results)
    assert [int(call["videos"][0][0, 0, 0, 0]) for call in agent.processor.calls] == [97] * len(Q) + [122] * len(Q)
    assert [part["type"] for part in agent.processor.messages[0]["content"]] == ["video", "text"]


def test_hooks_skip_and_error_lifecycle():
    agent, events = tiny_agent(), []
    def start(ctx):
        events.append("start")
        ctx.skip([{"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}])
    agent.predict("s", Q, on_predict_start=start, on_predict_end=lambda ctx: events.append("end"))
    assert events == ["start", "end"] and agent.model.calls == 0
    class Observer:
        def on_error(self, ctx):
            events.append("error")
        def on_predict_end(self, ctx):
            events.append("failed_end" if ctx.error else "end")
    with pytest.raises(ValueError):
        agent.predict("s", {"x": {"type": "invalid"}}, hooks=[Observer()])
    assert events[-2:] == ["error", "failed_end"]


def test_frame_decode_keeps_seek_precision_and_missing_frame_policy(tmp_path, monkeypatch):
    import io
    from PIL import Image
    import laya_multimodal.models.video as video

    image = io.BytesIO()
    Image.new("RGB", (2, 2), (11, 39, 201)).save(image, format="PNG")
    captured = []

    def read_frame(args, _description):
        captured.append(args[args.index("-ss") + 1])
        return image.getvalue() if len(captured) == 1 else b""

    monkeypatch.setattr(video, "_need_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(video, "_run", read_frame)
    frames, times = video.sample_frames("clip.mp4", 2, 4.)
    assert captured == ["1.000", "3.000"] and times == [1.]
    assert frames[0].getpixel((0, 0)) == (11, 39, 201)
    path = tmp_path / "clip.mp4"
    path.touch()
    monkeypatch.setattr(video, "probe", lambda _path: {"duration_s": 4., "has_audio": False})
    with pytest.raises(ValueError, match="bounded video frame"):
        tiny_agent()._read_video(path, use_audio=False, frames=2, bounds=(.1, 2.1234))
    assert captured[-1] == str(.1 + (2.1234 - .1) * .5 / 2)


def test_video_overrides_and_persistent_cache_preserve_native_evidence(tmp_path, monkeypatch):
    import laya_multimodal.models.video as video
    from PIL import Image
    path = tmp_path / "clip.avi"
    path.write_bytes(b"video")
    monkeypatch.setattr(video, "probe", lambda p: {"duration_s": 60, "has_audio": True})
    def frames(path, count, duration):
        return [Image.new("RGB", (8, 8), "red")] * count, list(range(count))
    monkeypatch.setattr(video, "sample_frames", frames)
    monkeypatch.setattr(video, "extract_audio", lambda p: pytest.fail("visual-only labels must not read audio"))
    agent = tiny_agent()
    agent._video_cache_dir = str(tmp_path / "cache")
    state = {"video": str(path), "video_audio": False, "video_frames": 6}
    before = agent.predict(state, {"x": Q["present"]})
    assert len(agent.processor.calls[-1]["images"]) == 6
    assert "audio" not in agent.processor.calls[-1]
    monkeypatch.setattr(video, "sample_frames", lambda *a: pytest.fail("disk cache was ignored"))
    restored = tiny_agent()
    restored._video_cache_dir = agent._video_cache_dir
    assert restored.predict(state, {"x": Q["present"]})["answers"] == before["answers"]
    with pytest.raises(ValueError, match="video_audio"):
        restored.predict(dict(state, video_audio="false"), Q)
    with pytest.raises(ValueError, match="video_frames"):
        restored.predict(dict(state, video_frames=0), Q)


@pytest.mark.parametrize('error_number', [errno.ENOSPC, errno.EDQUOT])
def test_full_video_cache_preserves_decoded_evidence(tmp_path, monkeypatch, error_number):
    import laya_multimodal.models.video as video
    from PIL import Image
    from types import SimpleNamespace
    import laya_multimodal.models.multimodal as multimodal
    path = tmp_path / 'clip.avi'; path.write_bytes(b'video')
    frames = [Image.new('RGB', (8, 8), 'red')] * 4
    monkeypatch.setattr(video, 'probe', lambda p: dict(duration_s=4., has_audio=False))
    monkeypatch.setattr(video, 'sample_frames', lambda *args: (frames, [0., 1., 2., 3.]))
    monkeypatch.setattr(multimodal.shutil, 'disk_usage', lambda p: SimpleNamespace(free=4 << 30))
    def full_cache(stream, **data):
        stream.write(b'incomplete cache')
        raise OSError(error_number, 'storage exhausted')
    monkeypatch.setattr(multimodal.np, 'savez_compressed', full_cache)
    agent = tiny_agent(); agent._video_cache_dir = str(tmp_path / 'cache')
    answer = agent.predict(video=path, questions={'x': Q['present']})
    assert answer['answers'] and len(agent.processor.calls[-1]['images']) == 4
    assert not list((tmp_path / 'cache').iterdir())


def test_low_video_cache_space_skips_persistence_before_writing(tmp_path, monkeypatch):
    import laya_multimodal.models.video as video
    import laya_multimodal.models.multimodal as multimodal
    from PIL import Image
    from types import SimpleNamespace
    path = tmp_path / 'clip.avi'; path.write_bytes(b'video')
    monkeypatch.setattr(video, 'probe', lambda p: dict(duration_s=1., has_audio=False))
    monkeypatch.setattr(video, 'sample_frames', lambda *args: ([Image.new('RGB', (4, 4))], [0.]))
    monkeypatch.setattr(multimodal.shutil, 'disk_usage', lambda p: SimpleNamespace(free=0))
    monkeypatch.setattr(multimodal.tempfile, 'mkstemp', lambda **kwargs: pytest.fail('full cache must not create a temporary file'))
    agent = tiny_agent(); agent._video_cache_dir = str(tmp_path / 'cache')
    assert agent.predict(video=path, questions={'x': Q['present']})['answers']


def test_ordinal_calibration_guard_rejects_worse_calibration_and_keeps_groups_together(monkeypatch):
    from laya_multimodal.models import modality_calibration
    seen = []
    def fit(values):
        seen.append(len(values))
        return {"temperature": [1., 2., 1.], "temperature_by_options": {"score:3-5": 2.}}
    monkeypatch.setattr(modality_calibration, "fit_temperature_map", fit)
    records = [{"qtype": 1, "k": 3, "modality": "image", "group": "group%d" % i,
                "logits": np.array([2., 0., -1.]), "target": np.array([1., 0., 0.])}
               for i in range(100)]
    agent = tiny_agent()
    calibration = modality_calibration.calibrate_multimodal(agent, records, ordinal_guard=True)
    guard = calibration["by_modality"]["image"]["ordinal_guard"]
    assert guard["fit_n"] + guard["check_n"] == 100
    assert min(guard["fit_n"], guard["check_n"]) >= 10
    assert not guard["accepted"] and agent.modality_calibration["image"]["temperature"][1] == 1.
    assert agent.temperature[1] == 1. and "score:3-5" not in agent.temperature_by_options
    assert len(seen) == 3
    # Distinct files by one uploader remain a single unit in calibration's fit/check split.
    for record in records:
        record["calibration_group"] = "freesound_uploader:one_author"
    calibration = modality_calibration.calibrate_multimodal(agent, records, ordinal_guard=True)
    guard = calibration["by_modality"]["image"]["ordinal_guard"]
    assert min(guard["fit_n"], guard["check_n"]) == 0 and not guard["accepted"]


def test_router_uses_one_native_checkpoint_for_text_and_media():
    agent = tiny_agent()
    router = Router(multimodal=agent, vega=False)
    assert router.route(None, Q, media="audio").model == "multimodal"
    assert router.route({"image": "local.png"}, Q).model == "multimodal"
    assert router.route("Please refund the duplicate charge on my invoice.", Q).model == "multimodal"
    result = router.predict("context", Q, image=np.zeros((4, 4, 3), np.uint8))
    assert result["routing"]["model"] == "multimodal"
    results = router.predict_batch([{"state": {"image": np.zeros((4, 4, 3), np.uint8)}, "questions": Q}])
    assert len(results) == 1 and results[0]["routing"]["model"] == "multimodal"


def test_split_overlap_and_relative_paths(tmp_path):
    media = tmp_path / "sample.wav"
    media.write_bytes(b"same recording")
    row = {"state": {"audio": "sample.wav"}, "questions": {"x": Q["present"]}, "expected": {"x": True}}
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps(row) + "\n")
    rows = load_rows(data)
    assert rows[0]["state"]["audio"] == str(media)
    with pytest.raises(ValueError, match="data leakage"):
        check_splits(train=rows, test=rows)


def test_native_collation_keeps_last_token_and_media_masks():
    examples = [{"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]]),
                 "input_features": torch.ones(1, 3, 8), "input_features_mask": torch.ones(1, 3, dtype=torch.bool)},
                {"input_ids": torch.tensor([[3, 4, 5]]), "attention_mask": torch.tensor([[1, 1, 1]]),
                 "input_features": torch.ones(1, 2, 8), "input_features_mask": torch.ones(1, 2, dtype=torch.bool)}]
    batch = collate_multimodal_inputs(examples, 0)
    assert batch["input_ids"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert batch["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert batch["input_features_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["input_features"].shape == (2, 3, 8)




def test_empty_context_is_distinguished_by_the_decision_question():
    one = {"state": "{}", "questions": {"x": Q["present"]}}
    other = {"state": "{}", "questions": {"x": Q["category"]}}
    check_splits(train=[one], test=[other])
    with pytest.raises(ValueError, match="data leakage"):
        check_splits(train=[one], test=[one])




def test_metrics_accept_all_equally_valid_options():
    from laya_multimodal.evaluation.native_metrics import metrics
    records = [{"qtype": 0, "k": 3, "modality": "text", "logits": np.array([0., 4., -1.]),
                "target": np.array([0.5, 0.5, 0.])}]
    assert metrics(records)["overall"]["accuracy"] == 1.0


def test_inline_media_http_endpoint_rejects_paths_and_shares_auth(monkeypatch):
    import base64
    import io
    from PIL import Image
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from laya_multimodal.serve import create_app
    monkeypatch.setenv("LAYA_API_KEY", "test-key")
    agent = tiny_agent()
    image = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(image, format="PNG")
    body = {"questions": {"x": Q["category"]}, "image": {"base64": base64.b64encode(image.getvalue()).decode()}}
    with TestClient(create_app(Router(multimodal=agent, vega=False))) as client:
        assert client.post("/v1/multimodal", json=body).status_code == 401
        response = client.post("/v1/multimodal", json=body, headers={"Authorization": "Bearer test-key"})
        assert response.status_code == 200, response.text
        assert response.json()["routing"]["model"] == "multimodal"
        body["image"] = {"path": "/etc/passwd"}
        assert client.post("/v1/multimodal", json=body, headers={"Authorization": "Bearer test-key"}).status_code == 422


@pytest.mark.parametrize("multipart", [False, True])
def test_systemone_image_and_box_wire_contract(monkeypatch, multipart):
    import base64
    import io
    from PIL import Image
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    from laya_multimodal.serve import create_app

    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_JEV_STRICT", "0")
    agent = tiny_agent()
    agent.grounding_trained = True
    agent._grounding_outputs = lambda inputs: {
        "boxes": torch.tensor([[[0.25, 0.25, 0.75, 0.75]]]),
        "quality_logits": torch.zeros(1, 1),
    }
    stream = io.BytesIO()
    Image.new("RGB", (280, 280), "white").save(stream, format="PNG")
    payload = stream.getvalue()
    asset = {"type": "image", "part": "photo"} if multipart else {
        "type": "image", "format": "png", "data_base64": base64.b64encode(payload).decode(),
    }
    body = {"model": "jev-latest", "state": {"photo": "image-ref"}, "assets": {"image-ref": asset},
            "questions": {"target": {"type": "box", "instructions": {"find": "the white region"}},
                          "level": {"type": "score", "instructions": "Rate brightness",
                                    "criteria": [{"val": "dark"}, {"val": "as bright as image-ref"}]}}}
    files = [("request", (None, json.dumps(body), "application/json")),
             ("photo", ("photo.png", payload, "image/png"))]
    request = {"files": files} if multipart else {"json": body}
    with TestClient(create_app(Router(multimodal=agent, vega=False))) as client:
        assert client.post("/v1/systemone", **request).status_code == 400
        response = client.post("/v1/systemone", headers={"TypeSafe-Danger-Zone-Images": "true"}, **request)
        assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {"model", "answers", "usage"}
    assert result["answers"]["target"] == {"type": "box", "box": {
        "left": 0.25, "top": 0.25, "right": 0.75, "bottom": 0.75}, "confidence": 0.5}
    assert result["answers"]["level"]["legend"] == {"0": {"val": "dark"}, "1": {"val": "as bright as image-ref"}}
    assert set(result["usage"]) == {"input_tokens", "output_tokens"}
    assert len(agent.processor.calls[0]["images"]) == 1
    assert len(agent.processor.calls[1]["images"]) == 2


def test_systemone_reference_expansion_places_each_candidate_after_its_own_image():
    import base64
    import io
    import re
    from PIL import Image
    from laya_multimodal.decisions.systemone import prepare_image_request

    class ExpandingProcessor(TinyProcessor):
        image_token = "<img>"

        def apply_chat_template(self, messages, **kwargs):
            return "".join(item["text"] if item["type"] == "text" else self.image_token
                           for item in messages[0]["content"])

        def __call__(self, **kwargs):
            text = kwargs["text"]
            edits = []
            parts, before, delta = [], 0, 0
            for i, match in enumerate(re.finditer(self.image_token, text), start=1):
                replacement = self.image_token * (i + 1)
                start = match.start() + delta
                parts.extend((text[before:match.start()], replacement))
                edits.append({"span": match.span(), "new_span": (start, start + len(replacement))})
                delta += len(replacement) - len(self.image_token)
                before = match.end()
            self.expanded = "".join([*parts, text[before:]])
            n = len(self.expanded)
            return {"input_ids": torch.ones(1, n, dtype=torch.long),
                    "offset_mapping": [[(i, i + 1) for i in range(n)]],
                    "text_replacement_offsets": [edits]}

    agent = tiny_agent()
    agent.max_len = 2048
    agent.readout, agent.processor = "candidate", ExpandingProcessor()
    image = io.BytesIO()
    Image.new("RGB", (280, 280)).save(image, format="PNG")
    asset = {"format": "png", "data_base64": base64.b64encode(image.getvalue()).decode()}
    request, questions = prepare_image_request(
        "photo-a-ref", {"match": {"type": "choice", "instructions": "Which matches?",
                                   "criteria": {"A": "first choice", "B": "photo-b-ref", "C": "third choice"}}},
        {"photo-a-ref": asset, "photo-b-ref": asset}, {}, images_enabled=True,
    )
    inputs, _, _ = agent._prepare(request, questions["match"])
    expected = [match.end() - 2 for match in re.finditer(r" ->\n", agent.processor.expanded)]
    assert inputs["_candidate_positions"].tolist() == [expected]


@pytest.mark.parametrize("selection", [{}, {"model": "multimodal"}, {"task": "MuLTiMoDaL"}])
def test_http_text_endpoints_require_inline_bytes_for_native_media(monkeypatch, selection):
    from laya_multimodal.decisions.routing import Router
    from laya_multimodal.serve import create_app

    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    monkeypatch.setenv("LAYA_PRELOAD", "0")
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    agent = tiny_agent()
    state = {"image": "example.png"}
    with TestClient(create_app(Router(multimodal=agent, vega=False))) as client:
        for endpoint, inputs in (("/v1/systemone", {"state": state}),
                                 ("/v1/systemone/batch", {"states": ["Text context", state]})):
            response = client.post(endpoint, json={**selection, **inputs, "questions": Q})
            assert response.status_code == 422
            assert "/v1/multimodal" in response.json()["detail"]
    assert agent.model.calls == 0


@pytest.mark.parametrize("mode", ["head", "full"])
def test_joint_training_updates_weights_and_evaluates_development(tmp_path, mode):
    from laya_multimodal.models.grounding import GroundingHead
    from laya_multimodal.train import GroundingTrainConfig, finetune_grounding

    agent = tiny_agent()
    agent.model_type = "qwen3_5"
    agent.model.grounding_head = GroundingHead(8, 16, 1, "single-v1")
    agent.grounding_trained = agent.detection_trained = False
    agent.model.gradient_checkpointing_enable = lambda **kwargs: None
    original = agent.model.get_output_embeddings().weight.detach().clone()
    question = {"x": Q["category"]}
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"state": "training text", "questions": question, "expected": {"x": "bright"}}) + "\n")
    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps({"state": "held-out presence", "questions": {"x": Q["present"]},
                               "expected": {"x": True}}) + "\n")
    decision_dev = tmp_path / "decision_dev.jsonl"
    decision_dev.write_text(json.dumps({"state": "held-out decision", "questions": question,
                                        "expected": {"x": "dark"}}) + "\n")
    saved = []
    agent.save = lambda path: saved.append(path.name)

    output = tmp_path / "out"
    config = GroundingTrainConfig(mode=mode, epochs=2, learning_rate=.01, head_learning_rate=.01,
                                  train_readout_in_warmup=True)
    finetune_grounding(train, dev, output, decision_dev_data=decision_dev, agent=agent, config=config)
    report = json.loads((output / "training_report.json").read_text())

    assert not torch.equal(original, agent.model.get_output_embeddings().weight)
    assert report["steps"] == 2 and report["train_records"] == 1
    assert report["forward_passes"] == {"geometry": 0, "decisions": 2}
    assert report["before"]["decisions"]["overall"]["n"] == report["after"]["decisions"]["overall"]["n"] == 1
    assert "initial" in saved and "last" in saved


def test_evaluation_respects_padded_token_budget_without_truncating():
    agent = tiny_agent()
    lengths = {"short": 2, "medium": 4, "long": 9, "tail": 3}
    rows = [{"id": name, "state": name, "questions": {"x": Q["present"]},
             "expected": {"x": True}} for name in lengths]
    batches = []

    def prepare(request, q):
        ids = torch.ones(1, lengths[request["text"]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, agent._to_internal(q), {}

    def logits(inputs, count):
        batches.append(tuple(inputs["input_ids"].shape))
        return torch.zeros(inputs["input_ids"].shape[0], count)

    agent._prepare = prepare
    agent._logits = logits
    records = collect_records(agent, rows, micro_batch=4, max_micro_tokens=8)
    assert batches == [(2, 4), (1, 9), (1, 3)]
    assert [item["id"] for item in records] == list(lengths)


@pytest.mark.skipif(not os.environ.get("LAYA_MM_MODEL"), reason="set LAYA_MM_MODEL to a local multimodal checkpoint")
def test_real_backbone_image_audio_and_reload(tmp_path):
    from PIL import Image
    agent = MultimodalAgent(os.environ["LAYA_MM_MODEL"], device=os.environ.get("LAYA_MM_DEVICE", "cpu"))
    q = {"present": Q["present"]}
    before = agent.predict(image=Image.new("RGB", (64, 64), "red"), questions=q)
    agent.predict(audio=np.zeros(16000, np.float32), sampling_rate=16000, questions=q)
    agent.save(tmp_path)
    restored = MultimodalAgent(tmp_path, device=agent.device)
    after = restored.predict(image=Image.new("RGB", (64, 64), "red"), questions=q)
    assert before["answers"] == after["answers"]


def test_joint_training_preserves_best_development_checkpoint(tmp_path, monkeypatch):
    from laya_multimodal import train
    from laya_multimodal.models.grounding import GroundingHead

    agent = tiny_agent()
    agent.model.grounding_head = GroundingHead(8, 16, 1, "single-v1")
    agent.grounding_trained = agent.detection_trained = False
    train_rows, development = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    sample = {"state": "train", "questions": {"x": Q["category"]}, "expected": {"x": "bright"}}
    train_rows.write_text(json.dumps(sample) + "\n")
    sample["state"] = "independent development context"
    development.write_text(json.dumps(sample) + "\n")
    saved, calls = {}, []

    def save(path):
        saved[path.name] = agent.model.lm_head.weight.detach().clone()

    def collect(agent, rows, micro_batch, *, max_micro_tokens):
        calls.append(max_micro_tokens)
        return []

    def development_metrics(records):
        accuracy = [.5, 1., .75, .75][len(calls) - 1]
        return {"by_modality": {"text": {"accuracy": accuracy}}, "overall": {"accuracy": accuracy}}

    agent.save = save
    monkeypatch.setattr(train, "collect_records", collect)
    monkeypatch.setattr(train, "decision_metrics", development_metrics)
    config = train.GroundingTrainConfig(mode="head", epochs=2, head_learning_rate=.01, eval_every=1,
                                        train_readout_in_warmup=True, max_micro_tokens=32,
                                        selection_metric="decision_macro", include_initial_selection=True)
    output = tmp_path / "out"
    train.finetune_grounding(train_rows, development, output, decision_dev_data=development, agent=agent, config=config)
    report = json.loads((output / "training_report.json").read_text())
    assert [record["decisions"]["overall"]["accuracy"] for record in report["selection_history"]] == [1., .75]
    assert calls == [32, 32, 32, 32] and report["checkpoint"] == str(output / "best")
    assert not torch.equal(saved["initial"], saved["best"])
    assert not torch.equal(saved["best"], saved["last"])


@pytest.mark.parametrize("mode", ["head", "lora", "full", "merged"])
def test_transformers_checkpoint_roundtrip_and_adapter_continuation(tmp_path, monkeypatch, mode):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4Config"):
        pytest.skip("Gemma 4 checkpoint tests need the multimodal extra's Transformers version")
    if mode in ("lora", "merged"):
        pytest.importorskip("peft")
    from laya_multimodal.training.objectives import configure_training
    from laya_multimodal.train import GroundingTrainConfig
    text = transformers.Gemma4TextConfig(vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size_per_layer_input=256, hidden_size_per_layer_input=4,
        layer_types=["sliding_attention", "full_attention"], num_kv_shared_layers=0)
    vision = transformers.Gemma4VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, patch_size=2, position_embedding_size=16)
    audio = transformers.Gemma4AudioConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        subsampling_conv_channels=[4, 4], output_proj_dims=16)
    config = transformers.Gemma4Config(text_config=text, vision_config=vision, audio_config=audio,
        image_token_id=250, audio_token_id=251, video_token_id=252)
    backbone = transformers.Gemma4ForConditionalGeneration(config)
    source = tmp_path / "base"
    backbone.save_pretrained(source)
    processor = TinyProcessor()
    processor.save_pretrained = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    agent = MultimodalAgent(source, device="cpu")
    training_config = GroundingTrainConfig(mode="lora" if mode == "merged" else mode,
        lora_rank=8, lora_alpha=16, lora_targets="qv")
    parameters = configure_training(agent, training_config, gradient_checkpointing=False,
                                    train_projectors=mode in ("lora", "merged"))
    if mode == "lora":
        base = agent.model.get_base_model()
        with torch.no_grad():
            for name in agent.native_trainable_modules:
                next(base.get_submodule(name).parameters()).add_(.25)
    inputs, _, _ = agent._prepare(agent._input("example"), Q["category"])
    agent.model.train()
    loss = -torch.log_softmax(agent._logits(inputs, 2), -1)[0, 1]
    loss.backward()
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    optimizer.step()
    agent.model.eval()
    if mode == "merged":
        agent.model = agent.model.merge_and_unload(safe_merge=True)
    before = agent._logits(inputs, 2).detach()
    agent.ordinal_calibration = {"models": {"text:2": {"kind": "temperature", "parameters": [.4], "prior": [.5, .5]}}}
    score_question = {"t": "score", "crit": ["low", "high"]}
    decoded = agent._decode(before[0], score_question)
    saved = tmp_path / "saved"
    agent.save(saved, mode="full" if mode == "merged" else None)
    assert json.loads((saved / "multimodal_config.json").read_text())["mode"] == ("full" if mode == "merged" else mode)
    restored = MultimodalAgent(saved, device="cpu")
    torch.testing.assert_close(restored._logits(inputs, 2), before, atol=0, rtol=0)
    assert restored.ordinal_calibration == agent.ordinal_calibration
    assert restored._decode(before[0], score_question) == decoded
    if mode == "lora":
        assert (saved / "native_projectors.safetensors").is_file()
        for name in agent.native_trainable_modules:
            expected = agent.model.get_base_model().get_submodule(name).state_dict()
            actual = restored.model.get_base_model().get_submodule(name).state_dict()
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        continued = configure_training(restored, training_config, gradient_checkpointing=False)
        assert restored.ordinal_calibration == {}
        assert continued and all(p.requires_grad for p in continued)
        restored._logits(inputs, 2).sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in continued)


def test_jsonl_preserves_unicode_separators(tmp_path):
    row = {"state": "多模态\u2028embedded\u0085line", "questions": {"x": Q["present"]},
           "expected": {"x": True}}
    path = tmp_path / "train.jsonl"
    path.write_bytes(("# comment\r\n" + json.dumps(row, ensure_ascii=False) + "\r\n").encode("utf-8"))
    assert load_rows(path) == [row]
