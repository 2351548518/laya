"""Local multimodal backbones with a compact, non-generative typed decision head.

The language model's vocabulary projection is replaced with a projection onto option slots.
Its image/audio encoders and temporal frame inputs remain part of the trainable backbone.
No captioner, speech recognizer, generation loop or hosted inference service is involved.
"""
from __future__ import annotations

import inspect
from collections import OrderedDict
import errno
from functools import cache, partial
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import string
import tempfile
import threading
import time

import numpy as np
import torch

from .agent import DecisionAgent
from .common import QTYPES, confidence_from_probs, render_criterion, render_options, serialize_state, temp_bucket, unpermute_probs
from .confidence import apply_confidence_gate, check_min_confidence
from .hooks import PredictContext, aggregate_usage, compose_hooks, dispatch, normalise_hooks, validate_timeout
from .multimodal_readout import CandidateScorer, candidate_positions
from .multimodal_calibration import ordinal_probabilities, validate_ordinal_calibration
from .revisions import default_checkpoint, resolve_revision, snapshot_revision, verify_digests
from .systemone import ImageReferences

DEFAULT_MULTIMODAL_MODEL = "google/gemma-4-E2B-it"
MULTIMODAL_MODELS = {
    "gemma4": DEFAULT_MULTIMODAL_MODEL,
    "qwen3.5": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen-omni": "Qwen/Qwen2.5-Omni-3B",
}
_AUDIO_MODELS = {"gemma4", "gemma3n", "qwen2_5_omni", "qwen2_5_omni_thinker"}
_SYMBOLS = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _checkpoint_directory(source, revision, token, expected_sha256):
    path = Path(source).expanduser()
    if path.is_dir():
        verify_digests(str(path), expected_sha256)
        return path
    if path.is_absolute() or source.startswith(("./", "../")) or len(path.parts) > 2:
        raise FileNotFoundError("checkpoint directory does not exist: " + source)

    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import EntryNotFoundError

    options = {"revision": revision, "token": token}
    try:
        manifest = hf_hub_download(source, "multimodal_config.json", **options)
    except EntryNotFoundError:
        if expected_sha256 or os.environ.get("LAYA_SHA256_DIGESTS", "").strip():
            snapshot = Path(snapshot_download(source, **options))
            verify_digests(str(snapshot), expected_sha256)
            return snapshot
        return path

    options["revision"] = snapshot_revision(str(Path(manifest).parent)) or revision
    snapshot = Path(snapshot_download(source, **options, allow_patterns=[
        "multimodal_config.json", "backbone_config/*", "processor/*", "adapter/*", "*.safetensors",
    ]))
    verify_digests(str(snapshot), expected_sha256)
    return snapshot


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _local_file(value, kind):
    path = os.fspath(value)
    if not os.path.isfile(path):
        raise FileNotFoundError("%s must name a local file: %r" % (kind, path))
    return path


def read_image(value):
    """Load pixels locally; callers may also supply a PIL image or an RGB array."""
    from PIL import Image
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")
    source = io.BytesIO(value) if isinstance(value, bytes) else _local_file(value, "image")
    with Image.open(source) as image:
        return image.convert("RGB")


def read_audio(value, sampling_rate=None, max_seconds=30.0):
    """Read and resample audio to 16 kHz without a transcription step."""
    if isinstance(value, dict):
        sampling_rate = value.get("sampling_rate", sampling_rate)
        value = value.get("array", value.get("path"))
    if isinstance(value, (np.ndarray, list, tuple)):
        if sampling_rate is None:
            raise ValueError("a waveform requires sampling_rate")
        wave = np.asarray(value, dtype=np.float32)
    else:
        import soundfile as sf
        source = io.BytesIO(value) if isinstance(value, bytes) else _local_file(value, "audio")
        wave, sampling_rate = sf.read(source, dtype="float32", always_2d=False)
    _positive("sampling_rate", sampling_rate)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError("audio must contain a nonempty, finite mono or channels-last waveform")
    if len(wave) / sampling_rate > max_seconds:
        raise ValueError("audio exceeds max_audio_seconds=%g; split the recording explicitly" % max_seconds)
    if sampling_rate != 16000:
        from scipy.signal import resample_poly
        divisor = math.gcd(sampling_rate, 16000)
        wave = resample_poly(wave, 16000 // divisor, sampling_rate // divisor).astype(np.float32)
    return wave


def _model_class(transformers, model_type):
    if model_type in ("qwen2_5_omni", "qwen2_5_omni_thinker"):
        return transformers.Qwen2_5OmniThinkerForConditionalGeneration
    if model_type not in {"gemma4", "gemma3n", "qwen3_5", "qwen3_vl", "qwen2_5_vl"}:
        raise ValueError("unsupported multimodal backbone model_type=%r" % model_type)
    if model_type == "gemma4" and hasattr(transformers, "AutoModelForMultimodalLM"):
        return transformers.AutoModelForMultimodalLM
    return transformers.AutoModelForImageTextToText


def install_decision_head(model, tokenizer, symbols=None):
    """Replace the vocabulary projection, initializing option slots from single-token symbols."""
    original = model.get_output_embeddings()
    if symbols is None:
        symbols = [s for s in _SYMBOLS if len(tokenizer.encode(s, add_special_tokens=False)) == 1]
    token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in symbols]
    if len(symbols) < 2 or any(len(ids) != 1 for ids in token_ids):
        raise ValueError("decision symbols must each encode as one token, with at least two symbols")
    if len(set(ids[0] for ids in token_ids)) != len(symbols):
        raise ValueError("decision symbols must have distinct token ids")
    head = torch.nn.Linear(original.weight.shape[1], len(symbols), bias=False,
                           device=original.weight.device, dtype=original.weight.dtype)
    with torch.no_grad():
        head.weight.copy_(original.weight[[ids[0] for ids in token_ids]])
    model.set_output_embeddings(head)
    # PEFT also inspects class metadata even when config.tie_word_embeddings is false.
    head_names = [name + "." for name, module in model.named_modules() if module is head]
    for attribute in ("_tied_weights_keys", "all_tied_weights_keys"):
        tied = getattr(model, attribute, None)
        if isinstance(tied, dict):
            setattr(model, attribute, {k: v for k, v in tied.items() if not any(k.startswith(n) for n in head_names)})
        elif isinstance(tied, list):
            setattr(model, attribute, [k for k in tied if not any(k.startswith(n) for n in head_names)])
    # Otherwise adapter loading may tie this small projection back to the vocabulary embedding.
    model.config.tie_word_embeddings = False
    for key in ("text_config", "thinker_config"):
        config = getattr(model.config, key, None)
        if config is not None:
            config.tie_word_embeddings = False
    return list(symbols)


def _active_head(model):
    head = model.get_output_embeddings()
    if hasattr(head, "modules_to_save"):
        active = head.active_adapter
        head = head.modules_to_save[active[0] if isinstance(active, list) else active]
    return head


def _is_adapter_model(model):
    # merge_and_unload() can leave peft_config on the unwrapped Transformers model.
    return hasattr(model, "peft_config") and callable(getattr(model, "get_base_model", None))


class MultimodalAgent(DecisionAgent):
    """A general multimodal reader converted into a local typed decision model.

    Image, sound and sampled video frames are passed as model inputs, including non-speech
    audio. Probabilities are conditional on the supplied options and require held-out calibration.
    """

    def __init__(self, model_id_or_path=None, device=None, *, revision=None,
                 token=None, max_len=4096, frames=8, max_audio_seconds=30.0,
                 use_audio_in_video=True, dtype=None, hooks=None, on_predict_start=None,
                 on_predict_end=None, hooks_raise=True, hooks_concurrent=True, hooks_timeout=None,
                 readout=None, prompt_version=None, grounding_slots=None, expected_sha256=None):
        import transformers
        source = default_checkpoint() if model_id_or_path is None else os.fspath(model_id_or_path)
        source = MULTIMODAL_MODELS.get(source, source)
        revision = resolve_revision(source, revision) if not Path(source).expanduser().is_dir() else revision
        checkpoint = _checkpoint_directory(source, revision, token, expected_sha256)
        model_id_or_path = str(checkpoint) if checkpoint.is_dir() else source
        self.checkpoint_revision = snapshot_revision(str(checkpoint))
        saved = None
        if (checkpoint / "multimodal_config.json").is_file():
            saved = json.loads((checkpoint / "multimodal_config.json").read_text())
            if saved.get("version") != 1:
                raise ValueError("unsupported multimodal checkpoint version")
            model_id_or_path = saved["backbone"]
            revision = saved.get("revision")
            max_len, frames = saved["max_len"], saved["frames"]
            max_audio_seconds = saved["max_audio_seconds"]
            use_audio_in_video = saved["use_audio_in_video"]
        self.max_len = _positive("max_len", max_len)
        self.frames = _positive("frames", frames)
        if isinstance(max_audio_seconds, bool) or not math.isfinite(max_audio_seconds) or max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive and finite")
        self.max_audio_seconds = float(max_audio_seconds)
        self.use_audio_in_video = bool(use_audio_in_video)
        self.model_id = self.model_id_or_path = MULTIMODAL_MODELS.get(model_id_or_path, model_id_or_path)
        if os.path.isdir(self.model_id):
            self.model_id = self.model_id_or_path = str(Path(self.model_id).resolve())
        self.revision = revision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.readout = readout or (saved or {}).get("readout", "letter")
        self.prompt_version = prompt_version or (saved or {}).get("prompt_version", "v1")
        if self.readout not in {"letter", "candidate"} or self.prompt_version not in {"v1", "v2"}:
            raise ValueError("unsupported decision readout or prompt version")
        options = {"revision": revision, "token": token, "trust_remote_code": False}
        config_source = checkpoint / "backbone_config" if saved and saved["mode"] == "full" else self.model_id
        config = transformers.AutoConfig.from_pretrained(config_source, **options)
        self.model_type = config.model_type
        attention = {"": "sdpa", "audio_config": "eager"} if self.model_type == "gemma4" else "sdpa"
        processor_source = checkpoint / "processor" if saved else self.model_id
        self.processor = transformers.AutoProcessor.from_pretrained(processor_source, **options)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        factory = _model_class(transformers, self.model_type)
        model_config = config.thinker_config if self.model_type == "qwen2_5_omni" else config
        if saved and saved["mode"] == "full":
            self.model = factory.from_config(model_config, attn_implementation=attention, torch_dtype=self.dtype)
        else:
            self.model = factory.from_pretrained(self.model_id, config=model_config, torch_dtype=self.dtype,
                                                 attn_implementation=attention, use_safetensors=True, **options)
        forward_options = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(self.model.forward).parameters else {}
        self.symbols = install_decision_head(self.model, self.tok, saved.get("symbols") if saved else None)
        if self.readout == "candidate":
            hidden_size = self.model.get_output_embeddings().weight.shape[-1]
            self.model.decision_scorer = CandidateScorer(hidden_size)
        grounding = (saved or {}).get("grounding")
        self.grounding_trained = bool(grounding and grounding.get("trained"))
        self.detection_trained = bool(grounding and grounding.get("detection_trained"))
        if grounding or grounding_slots is not None:
            from .grounding import GroundingHead
            if grounding_slots is not None:_positive("grounding_slots", grounding_slots)
            slots = grounding["slots"] if grounding else _positive("grounding_slots", grounding_slots)
            compact_grounding = bool(grounding and grounding_slots == 1 and grounding.get("architecture", "set-v1") != "single-v1")
            if grounding and grounding_slots is not None and grounding_slots != slots and not compact_grounding:
                raise ValueError("grounding_slots differs from the saved checkpoint architecture")
            dimension = grounding.get("dimension", 256) if grounding else 256
            architecture = grounding.get("architecture", "set-v1") if grounding else "single-v1" if slots == 1 else "set-v2"
            self.model.grounding_head = GroundingHead(self.model.get_output_embeddings().weight.shape[-1], dimension, slots, architecture)
        self.native_trainable_modules = saved.get("native_trainable_modules", []) if saved else []
        base = self.model
        for name in self.native_trainable_modules:
            base.get_submodule(name).float()
        if saved:
            from safetensors.torch import load_file, load_model
            if saved["mode"] == "full":
                load_model(self.model, str(checkpoint / "model.safetensors"), strict=True)
            elif saved["mode"] == "lora":
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, str(checkpoint / "adapter"))
            saved_head = load_file(str(checkpoint / "decision_head.safetensors"))
            _active_head(self.model).to(dtype=saved_head["weight"].dtype).load_state_dict(saved_head)
            scorer_file = checkpoint / "candidate_head.safetensors"
            if self.readout == "candidate" and saved.get("readout") == "candidate" and not scorer_file.is_file():
                raise FileNotFoundError("candidate checkpoint is missing " + str(scorer_file))
            if self.readout == "candidate" and scorer_file.is_file():
                base.decision_scorer.load_state_dict(load_file(str(scorer_file)))
            if self.native_trainable_modules and saved["mode"] != "full":
                native = load_file(str(checkpoint / "native_projectors.safetensors"))
                for name in self.native_trainable_modules:
                    prefix = name + "."
                    base.get_submodule(name).load_state_dict({k[len(prefix):]: v for k, v in native.items() if k.startswith(prefix)})
            if grounding:
                base.grounding_head.load_state_dict(load_file(str(checkpoint / "grounding_head.safetensors")))
                if compact_grounding:
                    base.grounding_head = base.grounding_head.to_single()
                    self.detection_trained = False
        self.model.to(self.device).eval().requires_grad_(False)
        if self.readout == "candidate":
            base.decision_scorer.float()
        if grounding or grounding_slots is not None:
            base.grounding_head.float()
        self.cfg = {"max_len": self.max_len}
        self.temperature = saved.get("temperature", [1.0] * 3) if saved else [1.0] * 3
        self.temperature_by_options = saved.get("temperature_by_options", {}) if saved else {}
        self.calibrated = bool(saved and saved.get("calibrated"))
        self.modality_calibration = saved.get("modality_calibration", {}) if saved else {}
        self.ordinal_calibration = validate_ordinal_calibration(saved.get("ordinal_calibration", {}) if saved else {})
        self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
        self.hooks_raise = bool(hooks_raise)
        self.hooks_concurrent = bool(hooks_concurrent)
        self._hooks_lock = threading.RLock() if not hooks_concurrent else None
        self.hooks_timeout = None if hooks_timeout is None else validate_timeout(hooks_timeout)
        self._hooks_mutex = threading.Lock()
        self._lock = threading.RLock()
        self._forward_options = forward_options
        self._checkpoint_mode = saved["mode"] if saved else "head"

    @property
    def backend(self):
        return "multimodal"

    def _input(self, state=None, *, image=None, audio=None, video=None, sampling_rate=None):
        if isinstance(state, dict) and any(k in state for k in ("image", "audio", "video")):
            if any(x is not None for x in (image, audio, video)):
                raise ValueError("supply media in state or keyword arguments, not both")
            return dict(state)
        if state is None and all(x is None for x in (image, audio, video)):
            raise ValueError("supply text or at least one image, audio or video input")
        return {"text": "" if state is None else serialize_state(state), "image": image,
                "audio": audio, "video": video, "sampling_rate": sampling_rate}

    def _media(self, request):
        content, images, audios, metadata = [], [], [], {}
        if request.get("image") is not None:
            values = request["image"] if isinstance(request["image"], (list, tuple)) else [request["image"]]
            for value in values:
                images.append(read_image(value))
                content.append({"type": "image"})
            metadata["images"] = len(images)
        if request.get("video") is not None:
            from .video import as_path
            path, cleanup = as_path(request["video"])
            try:
                use_audio = request.get("video_audio", self.use_audio_in_video)
                if not isinstance(use_audio, bool):
                    raise ValueError("video_audio must be a boolean")
                frame_count = _positive("video_frames", request.get("video_frames", self.frames))
                bounds = request.get("video_bounds")
                interval = {"bounds": bounds} if bounds is not None else {}
                info, frames, times, wave = self._read_video(path, use_audio=use_audio, frames=frame_count, **interval)
                if not frames:
                    raise ValueError("video has no decodable frames")
                encoding = request.get("video_encoding", "native" if getattr(self, "readout", "letter") == "candidate" else "images")
                if encoding == "native":
                    if self.model_type not in {"gemma4", "qwen3_5"}:
                        raise ValueError("native video encoding currently requires gemma4 or qwen3_5")
                    content.append({"type": "video"})
                    metadata["_native_video_frames"] = frames
                else:
                    content.append({"type": "text", "text": "Video frames in chronological order:"})
                    for frame, timestamp in zip(frames, times):
                        content.extend([{"type": "text", "text": "Time %.3f seconds:" % timestamp}, {"type": "image"}])
                        images.append(frame)
                metadata["video"] = {"duration_seconds": info["duration_s"], "frame_times": times}
                metadata["video"]["encoding"] = encoding
                if use_audio and info["has_audio"]:
                    if wave is not None:
                        audios.append(read_audio(wave, 16000, self.max_audio_seconds))
                        content.append({"type": "audio"})
                        metadata["video"]["audio"] = True
            finally:
                if cleanup:
                    os.unlink(cleanup)
        if request.get("audio") is not None:
            value = request["audio"]
            multiple = isinstance(value, (list, tuple)) and value and isinstance(value[0], (str, os.PathLike, dict))
            for segment in value if multiple else [value]:
                audios.append(read_audio(segment, request.get("sampling_rate"), self.max_audio_seconds))
                content.append({"type": "audio"})
        if audios and self.model_type not in _AUDIO_MODELS:
            raise ValueError("%s has no native audio encoder; use gemma4 or qwen-omni" % self.model_type)
        if audios:
            metadata["audio_seconds"] = [len(w) / 16000 for w in audios]
        return content, images, audios, metadata

    def _read_video(self, path, *, use_audio=None, frames=None, bounds=None):
        from .video import extract_audio, probe, sample_frames
        use_audio = self.use_audio_in_video if use_audio is None else use_audio
        frames = self.frames if frames is None else frames
        if str(path).endswith(".frames.json"):
            from PIL import Image
            sequence = json.loads(Path(path).read_text())
            if sequence.get("format") != "laya_video_frames_v1" or bounds is not None:
                raise ValueError("invalid predecoded video frame sequence or duplicate bounds")
            if len(sequence["paths"]) != frames or len(sequence["timestamps"]) != frames:
                raise ValueError("predecoded frame count does not match video_frames")
            images = []
            for frame_path in sequence["paths"]:
                frame_path = Path(frame_path)
                if not frame_path.is_absolute():
                    frame_path = Path(path).parent / frame_path
                with Image.open(frame_path) as image:
                    images.append(image.convert("RGB"))
            return ({"duration_s": float(sequence["duration_seconds"]), "has_audio": False, "has_video": True},
                    images, sequence["timestamps"], None)
        cache_size = getattr(self, "_video_cache_size", 0)
        stat = os.stat(path)
        key = (os.path.realpath(path), stat.st_mtime_ns, stat.st_size, frames,
               use_audio, self.max_audio_seconds)
        if bounds is not None:
            if len(bounds) != 2 or not all(math.isfinite(float(x)) for x in bounds):
                raise ValueError("video_bounds must contain two finite timestamps")
            if use_audio:
                raise ValueError("bounded video requires video_audio=False; provide an explicitly bounded soundtrack separately")
            key += (tuple(float(x) for x in bounds),)
        cache = getattr(self, "_video_cache", None)
        if cache is not None and key in cache:
            cache.move_to_end(key)
            return cache[key]
        disk = getattr(self, "_video_cache_dir", None)
        cached_path = Path(disk) / (hashlib.sha256(json.dumps(key).encode()).hexdigest() + ".npz") if disk else None
        if cached_path is not None and cached_path.is_file():
            from PIL import Image
            with np.load(cached_path, allow_pickle=False) as saved:
                info = {"duration_s": float(saved["duration"]), "has_audio": bool(saved["has_audio"]), "has_video": True}
                times = saved["times"].tolist()
                images = [Image.fromarray(saved["frame_%d" % i]) for i in range(len(times))]
                wave = saved["wave"] if saved["wave"].size else None
                value = info, images, times, wave
            if cache_size:
                if cache is None:
                    cache = self._video_cache = OrderedDict()
                cache[key] = value
                while len(cache) > cache_size:
                    cache.popitem(last=False)
            return value
        info = probe(path)
        if use_audio and info["has_audio"]:
            if self.model_type not in _AUDIO_MODELS:
                raise ValueError("this backbone cannot read audio; set use_audio_in_video=False for visual-only video")
            if info["duration_s"] > self.max_audio_seconds:
                raise ValueError("video soundtrack exceeds max_audio_seconds; split the clip explicitly")
        if bounds is None:
            images, times = sample_frames(path, frames, info["duration_s"])
        else:
            from .video import _decode_frame, _need_ffmpeg
            start, end = max(0., float(bounds[0])), min(info["duration_s"], float(bounds[1]))
            if not start < end:
                raise ValueError("video_bounds have no overlap with the video")
            times = [start + (end - start) * (i + .5) / frames for i in range(frames)]
            images = []
            ffmpeg = _need_ffmpeg()
            for timestamp in times:
                frame = _decode_frame(ffmpeg, path, ["-ss", str(timestamp)], "sample a bounded clip")
                if frame is None:
                    raise ValueError("bounded video frame could not be decoded")
                images.append(frame)
        wave = extract_audio(path) if use_audio and info["has_audio"] else None
        value = info, images, times, wave
        if cached_path is not None and images:
            temporary = None
            try:
                cached_path.parent.mkdir(parents=True, exist_ok=True)
                required = sum(frame.width * frame.height * 3 for frame in images)
                required += wave.nbytes if wave is not None else 0
                # Reserve space for other ranks writing to the same volume.
                if shutil.disk_usage(cached_path.parent).free >= (1 << 30) + required:
                    data = {"duration": np.asarray(info["duration_s"]), "has_audio": np.asarray(info["has_audio"]),
                            "times": np.asarray(times), "wave": np.asarray(wave if wave is not None else [], dtype=np.float32)}
                    data.update({"frame_%d" % i: np.asarray(frame) for i, frame in enumerate(images)})
                    fd, temporary = tempfile.mkstemp(dir=cached_path.parent, suffix=".npz")
                    with os.fdopen(fd, "wb") as stream:
                        np.savez_compressed(stream, **data)
                    os.replace(temporary, cached_path)
            except OSError as error:
                if error.errno not in {errno.ENOSPC, errno.EDQUOT}:
                    raise
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
        if cache_size:
            if cache is None:
                cache = self._video_cache = OrderedDict()
            cache[key] = value
            while len(cache) > cache_size:
                cache.popitem(last=False)
        return value

    @staticmethod
    def _check_question(qid, qdef):
        if isinstance(qdef, dict) and qdef.get("type") == "box":
            if set(qdef) - {"type", "instructions"}:
                raise ValueError("Box takes only type and instructions; it has no criteria or option_order")
            if not isinstance(qdef.get("instructions"), str) or not qdef['instructions'].strip():
                raise ValueError("Box instructions must be a nonempty string")
            DecisionAgent._check_question(qid, dict(qdef, type="noul"))
            return
        DecisionAgent._check_question(qid, qdef)

    def _prepare(self, request, qdef, *, max_len=None, load_media=None):
        q = self._to_internal(qdef)
        is_box = q["t"] == "box"
        options = [] if is_box else render_options(q)
        candidate = not is_box and getattr(self, "readout", "letter") == "candidate"
        if candidate and q["t"] == "choice":
            options = [render_criterion(value) if value not in (None, "") and re.fullmatch(r"(?:[A-Za-z]|c\d+|\d+)", str(key))
                       else options[i] for i, (key, value) in enumerate(q["crit"].items())]
        capacity = 256 if candidate else len(self.symbols)
        if len(options) > capacity:
            raise ValueError("multimodal questions support at most %d options; shortlist or split the question" % capacity)
        order = q.get("option_order", list(range(len(options))))
        image_refs = request.get("image")
        image_refs = image_refs if isinstance(image_refs, ImageReferences) else None
        if image_refs is not None:
            content, images, audios, metadata = [], [], [], {}
        else:
            content, images, audios, metadata = self._media(request) if load_media is None else load_media()
        spans = []
        if is_box:
            if image_refs is None and (len(images) != 1 or request.get("video") is not None):
                raise ValueError("Box currently requires exactly one image; select a video frame or image explicitly")
            prompt = "State: %s\nLocate in the image: %s\nReturn the target bounding box." % (request.get("text", ""), q["ins"])
        elif candidate:
            prompt = "State: %s\n%s question: %s\nOptions:\n" % (request.get("text", ""), q["t"], q["ins"])
            for slot, index in enumerate(order):
                prompt += "Candidate %d: %s" % (slot + 1, options[index])
                start = len(prompt)
                prompt += " ->\n"
                spans.append((start, len(prompt) - 1))
            prompt += "Decide using the full state, question and candidates."
        elif getattr(self, "prompt_version", "v1") == "v2":
            if q["t"] == "choice":
                options = [render_criterion(value) if value not in (None, "") else str(key)
                           for key, value in q["crit"].items()]
            prompt = "State: %s\n%s question: %s\n" % (request.get("text", ""), q["t"], q["ins"])
            prompt += "\n".join("%s: %s" % (self.symbols[slot], options[index]) for slot, index in enumerate(order))
            prompt += "\nReturn only one answer label from: " + ", ".join(self.symbols[:len(options)]) + "."
        else:
            prompt = "%s question: %s\n" % (q["t"], q["ins"])
            prompt += "\n".join("%s: %s" % (self.symbols[slot], options[index]) for slot, index in enumerate(order))
            prompt += "\nState: %s\nSelect exactly one option. Answer with its letter or digit only." % request.get("text", "")
        if image_refs is not None:
            # Locate the candidate suffixes after the chat template inserts its image tokens.
            # The markers are removed before the text reaches the processor/tokenizer.
            for start, end in reversed(spans):
                prompt = prompt[:start] + image_refs.candidate_marker + prompt[end:]
            content, image_ids = image_refs.split_prompt(prompt)
            if is_box and len(set(image_ids)) != 1:
                raise ValueError("each box question requires a single referenced image")
            images = [read_image(image_refs.images[i]) if isinstance(image_refs.images[i], (str, os.PathLike))
                      else image_refs.images[i] for i in image_ids]
            metadata["images"] = len(images)
        else:
            content = [*content, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
        if candidate and image_refs is not None:
            parts = text.split(image_refs.candidate_marker)
            if len(parts) != len(spans) + 1:
                raise ValueError("processor template changed the candidate suffix")
            spans = []
            offset = 0
            for part in parts[:-1]:
                start = offset + len(part)
                offset = start + len(" ->")
                spans.append((start, offset))
            text = " ->".join(parts)
        kwargs = {"text": text, "return_tensors": "pt", "padding": False}
        metadata = metadata.copy()
        native_frames = metadata.pop("_native_video_frames", None)
        if native_frames is not None:
            from transformers.video_utils import VideoMetadata
            video = metadata["video"]
            kwargs["videos"] = [np.stack([np.asarray(frame) for frame in native_frames])]
            kwargs["videos_kwargs"] = {"do_sample_frames": False,
                "video_metadata": [VideoMetadata(total_num_frames=max(1, round(video["duration_seconds"] * 1000)),
                    fps=1000., duration=video["duration_seconds"],
                    frames_indices=[round(t * 1000) for t in video["frame_times"]], video_backend="local_ffmpeg")]}
            if self.model_type == 'gemma4':
                kwargs['videos_kwargs']['max_soft_tokens'] = 70
            elif self.model_type == 'qwen3_5':
                kwargs['videos_kwargs']['cap_pixels_per_frame'] = True
        if candidate:
            kwargs["return_offsets_mapping"] = True
            if image_refs is not None and images:
                kwargs["return_text_replacement_offsets"] = True
        if images:
            kwargs["images"] = images
        if audios:
            kwargs.update(audio=audios, sampling_rate=16000)
            if len(audios) > 1:
                kwargs["padding"] = True
        inputs = self.processor(**kwargs)
        if candidate:
            text_length = len(text)
            if image_refs is None:
                start = text.rfind(prompt)
                if start < 0:
                    raise ValueError("processor template changed the candidate suffix")
                spans = [(start + a, start + b) for a, b in spans]
            elif images:
                replacements = inputs.pop("text_replacement_offsets", [])
                if replacements and replacements[0]:
                    ends = [(item["span"][1], item["new_span"][1] - item["span"][1])
                            for item in replacements[0]]
                    text_length += ends[-1][1]
                    shifted, index, shift = [], 0, 0
                    for start, end in spans:
                        while index < len(ends) and ends[index][0] <= start:
                            _, shift = ends[index]
                            index += 1
                        shifted.append((start + shift, end + shift))
                    spans = shifted
                else:
                    token = getattr(self.processor, "image_token", None)
                    if token is None or text.rfind(token) >= spans[0][0]:
                        raise ValueError("inline option images require processor image replacement offsets")
            offsets = inputs.pop("offset_mapping")[0]
            inputs["_candidate_positions"] = candidate_positions(offsets, text_length, spans)
        length = inputs["input_ids"].shape[-1]
        limit = self.max_len if max_len is None else _positive("max_len", max_len)
        if length > limit:
            raise ValueError("multimodal input needs %d tokens, exceeds max_len=%d; shorten text/media or increase the budget" % (length, limit))
        inputs = {k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                  if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        return inputs, q, metadata

    def _grounding_outputs(self, inputs, multiple=False):
        base = self.model.get_base_model() if _is_adapter_model(self.model) else self.model
        if not hasattr(base, "grounding_head"):
            raise ValueError("this checkpoint has no grounding head; train a grounding checkpoint first")
        if multiple and base.grounding_head.architecture == "single-v1":
            raise ValueError("single-box architecture does not support multi-object detection")
        values = {key: value for key, value in inputs.items() if not key.startswith("_")}
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            out = base.model(**values, use_cache=False, return_dict=True)
        result = base.grounding_head(out.last_hidden_state, inputs.get("attention_mask"), multiple=multiple)
        if any(not torch.isfinite(value).all() for value in result.values()):
            raise RuntimeError("multimodal backbone returned nonfinite grounding predictions")
        return result

    @torch.no_grad()
    def detect(self, state=None, labels=None, *, image=None, threshold=.5, max_len=None):
        """Local multi-instance extension; one non-generative forward per text label.

        This is not a fifth official Jev question type. Use predict(type='box') for
        its standard one-box-per-question interface. Scores require domain validation.
        """
        from .grounding import box_answer
        if not getattr(self, "grounding_trained", False):
            raise ValueError("grounding head has not been trained")
        if not getattr(self, "detection_trained", False):
            raise ValueError("multi-object detection head has not been trained")
        if not isinstance(labels, (list, tuple)) or not labels or any(not isinstance(v, str) or not v.strip() for v in labels):
            raise ValueError("labels must be a nonempty list of target descriptions")
        if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and in [0,1]")
        request = self._input(state, image=image)
        load_media = cache(partial(self._media, request))
        objects, tokens = [], 0
        with self._lock:
            self.model.eval()
            for label in labels:
                inputs, _, _ = self._prepare(request, {"type": "box", "instructions": "all instances of " + label},
                                              max_len=max_len, load_media=load_media)
                out = self._grounding_outputs(inputs, multiple=True)
                scores = out["object_logits"][0].sigmoid() * out["quality_logits"][0].sigmoid()
                for box, score in zip(out["boxes"][0], scores):
                    if float(score) >= threshold:
                        answer = box_answer(box, score)
                        objects.append({"label": label, "box": answer["box"], "confidence": answer["confidence"]})
                tokens += inputs["input_ids"].shape[-1]
        return {"model": self.model_id, "objects": sorted(objects, key=lambda value: -value["confidence"]),
                "usage": {"input_tokens": tokens, "output_tokens": 0, "forward_passes": len(labels), "truncated": False}}

    def _logits(self, inputs, count):
        with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                            enabled=self.device.type == "cuda"):
            if getattr(self, "readout", "letter") == "candidate":
                values = {key: value for key, value in inputs.items() if key != "_candidate_positions"}
                base = self.model.get_base_model() if _is_adapter_model(self.model) else self.model
                out = base.model(**values, use_cache=False, return_dict=True)
                logits = base.decision_scorer(out.last_hidden_state, inputs["_candidate_positions"])[:, :count].float()
            else:
                out = self.model(**inputs, use_cache=False, return_dict=True, **self._forward_options)
                logits = out.logits[:, -1, :count].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("multimodal backbone returned nonfinite decision logits")
        return logits

    def _reuse_vision_features(self, inputs, cache):
        if (torch.is_grad_enabled() or self.model.training or getattr(self, "model_type", None) != "qwen3_5"
                or getattr(self, "readout", "letter") != "candidate"):
            return inputs
        fields = [(kind, pixel, grid) for kind, pixel, grid in (
            ("image", "pixel_values", "image_grid_thw"), ("video", "pixel_values_videos", "video_grid_thw"))
            if inputs.get(pixel) is not None]
        if not fields:
            return inputs
        base = self.model.get_base_model() if _is_adapter_model(self.model) else self.model
        encoder = base.model
        if ("mm_encoder_outputs" not in inspect.signature(encoder.forward).parameters
                or any(not callable(getattr(encoder, "get_" + kind + "_features", None)) for kind, _, _ in fields)):
            return inputs
        features = {}
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            for kind, pixel, grid in fields:
                pixels, geometry = inputs[pixel], inputs.get(grid)
                if geometry is None:
                    return inputs
                previous = cache.get(kind)
                if (previous is None or previous[0].shape != pixels.shape or previous[0].dtype != pixels.dtype
                        or previous[0].device != pixels.device or previous[1].device != geometry.device
                        or previous[1].shape != geometry.shape or previous[1].dtype != geometry.dtype
                        or not torch.equal(previous[0], pixels) or not torch.equal(previous[1], geometry)):
                    features[kind] = getattr(encoder, "get_" + kind + "_features")(pixels, geometry, return_dict=True)
                    cache[kind] = (pixels, geometry, features[kind])
                else:
                    features[kind] = previous[2]
        values = {key: value for key, value in inputs.items() if key not in {pixel for _, pixel, _ in fields}}
        values["mm_encoder_outputs"] = features
        return values

    def _decode(self, logits, q, modality="text"):
        k = len(logits)
        qtype = QTYPES[q["t"]]
        calibration = getattr(self, "modality_calibration", {}).get(modality, {})
        ordinal = getattr(self, "ordinal_calibration", {})
        calibrated = qtype in calibration.get("fitted_types", [])
        use_ordinal = q["t"] == "score" and bool(ordinal)
        model = ordinal["models"].get(f"{modality}:{k}") if use_ordinal else None
        if use_ordinal and (model is not None or not calibrated):
            # A fitted channel includes its own temperature. An unfitted modality uses raw logits.
            canonical = unpermute_probs(logits.detach().double().cpu().numpy(), q.get("option_order"))
            p = ordinal_probabilities(canonical, model)
            calibrated = model is not None and model["kind"] != "identity"
        else:
            temperatures = calibration.get("temperature", self.temperature)
            buckets = calibration.get("temperature_by_options", self.temperature_by_options)
            scale = buckets.get(temp_bucket(qtype, k), temperatures[qtype])
            p = torch.softmax(logits / scale, -1).detach().cpu().numpy()
            p = unpermute_probs(p, q.get("option_order"))
        answer = {"type": q["t"], "answer_confidence": round(float(p.max()), 4),
                  "confidence": round(float(p.max()) if q["t"] == "noul" else confidence_from_probs(p, k), 4),
                  "calibrated": calibrated}
        if q["t"] == "choice":
            labels = list(q["crit"])
            answer.update(choice=labels[int(p.argmax())], probabilities={v: round(float(x), 4) for v, x in zip(labels, p)})
        elif q["t"] == "score":
            answer.update(score=round(float(np.arange(k) @ p), 4),
                          probabilities={str(i): round(float(x), 4) for i, x in enumerate(p)},
                          legend={str(i): render_criterion(value) for i, value in enumerate(q["crit"])})
        else:
            answer["noul"] = round(float(p[1]), 4)
        return answer

    @torch.no_grad()
    def predict(self, state=None, questions=None, *, image=None, audio=None, video=None, sampling_rate=None,
                batch_size=None, lang=None, hooks=None, on_predict_start=None, on_predict_end=None,
                hooks_raise=None, hooks_timeout=None, max_len=None, head_max_len=None,
                sort_by_length=False, min_confidence=None):
        return self.predict_batch([self._input(state, image=image, audio=audio, video=video,
                                               sampling_rate=sampling_rate)], questions,
                                  batch_size=batch_size, lang=lang, hooks=hooks,
                                  on_predict_start=on_predict_start, on_predict_end=on_predict_end,
                                  hooks_raise=hooks_raise, hooks_timeout=hooks_timeout, max_len=max_len,
                                  head_max_len=head_max_len, sort_by_length=sort_by_length,
                                  min_confidence=min_confidence)[0]

    system_one = predict

    def _answer_text_questions(self, request, questions, max_len, batch_size):
        from .multimodal_data import collate_multimodal_inputs

        answers, pending = {}, []
        tokens, forwards, longest = 0, 0, 0
        budget = min(self.max_len if max_len is None else max_len, 8192)

        def flush():
            values = [item[1] for item in pending]
            inputs = values[0] if len(values) == 1 else collate_multimodal_inputs(
                values, getattr(self.tok, "pad_token_id", 0) or 0)
            logits = self._logits(inputs, max(item[3] for item in pending))
            for scores, (qid, _, q, count) in zip(logits, pending):
                answers[qid] = self._decode(scores[:count], q)

        for qid, qdef in questions.items():
            inputs, q, _ = self._prepare(request, qdef, max_len=max_len)
            length = inputs["input_ids"].shape[-1]
            if pending and (len(pending) >= batch_size or max(longest, length) * (len(pending) + 1) > budget):
                flush()
                forwards += 1
                pending.clear()
                longest = 0
            pending.append((qid, inputs, q, len(render_options(q))))
            tokens += length
            longest = max(longest, length)
        if pending:
            flush()
            forwards += 1
        return answers, tokens, forwards

    @torch.no_grad()
    def predict_batch(self, states, questions, batch_size=None, lang=None, hooks=None,
                      on_predict_start=None, on_predict_end=None, hooks_raise=None, hooks_timeout=None,
                      max_len=None, head_max_len=None, sort_by_length=False, min_confidence=None):
        """Score states sequentially, with isolated rows for batches of native text questions."""
        from .common import check_batch_size
        check_batch_size(batch_size)
        if isinstance(states, (str, bytes, dict)):
            raise TypeError("predict_batch expects a list of states")
        if head_max_len is not None:
            raise ValueError("head_max_len is an encoder-only control; multimodal inputs use max_len")
        mc = check_min_confidence(min_confidence) if min_confidence is not None else None
        active = compose_hooks(self.hooks, hooks, on_predict_start, on_predict_end)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)
        ctx = PredictContext(states=list(states), questions=questions, model=self.model_id, agent=self,
                             max_len=max_len, head_max_len=head_max_len)
        try:
            dispatch(active, "on_predict_start", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            if ctx.results is None:
                if ctx.head_max_len is not None:
                    raise ValueError("head_max_len is an encoder-only control; multimodal inputs use max_len")
                if not isinstance(ctx.questions, dict):
                    raise TypeError("questions must be a dict")
                for qid, qdef in ctx.questions.items():
                    self._check_question(qid, qdef)
                results = []
                with self._lock:
                    self.model.eval()
                    for state in ctx.states:
                        request = self._input(state)
                        modality = next((k for k in ("video", "audio", "image") if request.get(k) is not None), "text")
                        answers, tokens, metadata = {}, 0, {}
                        if (modality == "text" and self.model_type == "qwen3_5"
                                and getattr(self, "readout", "letter") == "candidate"
                                and batch_size is not None and batch_size > 1 and len(ctx.questions) > 1
                                and all(q.get("type") != "box" for q in ctx.questions.values())):
                            answers, tokens, forwards = self._answer_text_questions(
                                request, ctx.questions, ctx.max_len, min(batch_size, 4))
                            results.append({"model": self.model_id, "answers": answers,
                                            "usage": {"input_tokens": tokens, "output_tokens": 0,
                                                      "forward_passes": forwards, "truncated": False},
                                            "modality": modality, "media": metadata})
                            continue
                        vision_cache = {}
                        load_media = cache(partial(self._media, request))
                        for qid, qdef in ctx.questions.items():
                            inputs, q, metadata = self._prepare(request, qdef, max_len=ctx.max_len,
                                                               load_media=load_media)
                            if q["t"] == "box" and not getattr(self, "grounding_trained", False):
                                raise ValueError("grounding head has not been trained")
                            if len(ctx.questions) > 1:
                                inputs = self._reuse_vision_features(inputs, vision_cache)
                            if q["t"] == "box":
                                from .grounding import box_answer
                                out = self._grounding_outputs(inputs)
                                answers[qid] = box_answer(out["boxes"][0, 0], out["quality_logits"][0, 0].sigmoid())
                            else:
                                logits = self._logits(inputs, len(render_options(q)))[0]
                                question_modality = modality
                                if isinstance(request.get("image"), ImageReferences) and not metadata["images"]:
                                    question_modality = "text"
                                answers[qid] = self._decode(logits, q, question_modality)
                            tokens += inputs["input_ids"].shape[-1]
                        results.append({"model": self.model_id, "answers": answers,
                                        "usage": {"input_tokens": tokens, "output_tokens": 0,
                                                  "forward_passes": len(answers), "truncated": False},
                                        "modality": modality, "media": metadata})
                ctx.results = results
        except BaseException as exc:
            ctx.error = exc
            try:
                dispatch(active, "on_error", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                exc.__context__ = hook_exc
            raise
        finally:
            ctx.elapsed_ms = (time.perf_counter() - ctx.started_at) * 1000
            if ctx.results is not None:
                apply_confidence_gate(ctx.results, mc)
                ctx.usage = aggregate_usage(ctx.results)
            try:
                dispatch(active, "on_predict_end", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                if ctx.error is not None:
                    ctx.error.__context__ = hook_exc
                else:
                    raise
        return ctx.results

    def predict_long(self, state, questions, **kwargs):
        raise ValueError("multimodal models do not use encoder window aggregation; split media explicitly")

    def save(self, path, mode=None):
        """Save the decision projection and either adapters or the complete trained backbone."""
        from safetensors.torch import save_file, save_model
        has_adapter = _is_adapter_model(self.model)
        mode = mode or ("lora" if has_adapter else getattr(self, "_checkpoint_mode", "head"))
        if mode not in ("head", "lora", "full"):
            raise ValueError("mode must be head, lora or full")
        if has_adapter and mode != "lora":
            raise ValueError("save an adapter model with mode='lora'; merge adapters before exporting other modes")
        if mode == "lora" and not has_adapter:
            raise ValueError("mode='lora' requires an adapter model")
        if mode == "head" and getattr(self, "_checkpoint_mode", "head") == "full":
            mode = "full"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        base = self.model.get_base_model() if has_adapter else self.model
        grounding = None
        if hasattr(base, "grounding_head"):
            head = base.grounding_head
            grounding = {"slots": head.slots, "dimension": head.dimension,
                         "architecture": head.architecture,
                         "trained": getattr(self, "grounding_trained", False),
                         "detection_trained": getattr(self, "detection_trained", False)}
            save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()},
                      str(path / "grounding_head.safetensors"))
        if getattr(self, "readout", "letter") == "candidate":
            save_file({k: v.detach().cpu().contiguous() for k, v in base.decision_scorer.state_dict().items()},
                      str(path / "candidate_head.safetensors"))
        base.config.save_pretrained(path / "backbone_config")
        self.processor.save_pretrained(path / "processor")
        head = _active_head(self.model)
        save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()},
                  str(path / "decision_head.safetensors"))
        native_names = getattr(self, "native_trainable_modules", [])
        if native_names and mode != "full":
            native = {name + "." + key: value.detach().cpu().contiguous() for name in native_names
                      for key, value in base.get_submodule(name).state_dict().items()}
            save_file(native, str(path / "native_projectors.safetensors"))
        if mode == "lora":
            self.model.save_pretrained(path / "adapter", safe_serialization=True, save_embedding_layers=False)
        elif mode == "full":
            save_model(self.model, str(path / "model.safetensors"))
        payload = {"version": 1, "backbone": self.model_id, "revision": self.revision,
                   "readout": getattr(self, "readout", "letter"), "prompt_version": getattr(self, "prompt_version", "v1"),
                   "mode": mode, "symbols": self.symbols, "max_len": self.max_len,
                   "grounding": grounding,
                   "native_trainable_modules": native_names,
                   "frames": self.frames, "max_audio_seconds": self.max_audio_seconds,
                   "use_audio_in_video": self.use_audio_in_video, "temperature": self.temperature,
                   "temperature_by_options": self.temperature_by_options, "calibrated": self.calibrated,
                   "modality_calibration": getattr(self, "modality_calibration", {}),
                   "ordinal_calibration": validate_ordinal_calibration(getattr(self, "ordinal_calibration", {}))}
        (path / "multimodal_config.json").write_text(json.dumps(payload, indent=2) + "\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return False
