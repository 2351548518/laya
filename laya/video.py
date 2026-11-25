"""Local FFmpeg helpers for video probing, frame sampling and 16 kHz audio extraction.

The native multimodal agent passes sampled frames and supported audio to the model processor.
Vega uses a speech transcript when available and assembles a contact sheet for its image path.
FFmpeg must be on the PATH. Probing uses ffprobe when it is available.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave as _wave
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ffmpeg_available", "probe", "extract_audio", "sample_frames", "contact_sheet",
           "has_speech", "as_path", "TARGET_SR"]

#: What whisper-family transcribers expect.
TARGET_SR = 16_000

#: Below this root-mean-square level the sound track is treated as silent and never sent to the
#: transcriber, which would otherwise invent words for it.
SILENCE_RMS = 1e-3

#: A transcript with fewer words than this is noise the transcriber dressed up, not speech.
MIN_WORDS = 3

_NO_FFMPEG = ("video needs the ffmpeg binary on the PATH (brew install ffmpeg, apt install "
              "ffmpeg, or https://ffmpeg.org/download.html); laya takes no Python dependency "
              "for it. Images and audio work without it.")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _need_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(_NO_FFMPEG)
    return exe


def _run(args: Sequence[str], what: str) -> bytes:
    proc = subprocess.run(list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg could not %s: %s" % (what, " | ".join(tail) or "no output"))
    return proc.stdout


def as_path(video) -> Tuple[str, Optional[str]]:
    """A readable path for the clip, and the temporary file to delete afterwards, if any.

    ffmpeg needs a seekable input for most containers, so raw bytes are written to a temporary
    file rather than piped. A path or PathLike is used as given.
    """
    if isinstance(video, (bytes, bytearray)):
        fd, path = tempfile.mkstemp(suffix=".video")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(video)
        except BaseException:
            os.unlink(path)
            raise
        return path, path
    if isinstance(video, (str, os.PathLike)):
        path = os.fspath(video)
        if not os.path.isfile(path):
            raise FileNotFoundError("no video file at %r" % path)
        return path, None
    raise TypeError("video must be a path or raw video-file bytes, got %r" % type(video).__name__)


def probe(path: str) -> dict:
    """Duration in seconds (0.0 when the tool reports none) and whether the clip has an audio
    stream, from ffprobe when it is there and from ffmpeg's own banner when it is not."""
    ffmpeg = _need_ffmpeg()
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        out = _run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                    "-show_entries", "stream=codec_type", "-of", "json", path], "probe the clip")
        info = json.loads(out.decode("utf-8", "replace") or "{}")
        try:
            dur = float((info.get("format") or {}).get("duration", 0.0))
        except (TypeError, ValueError):
            dur = 0.0
        if not math.isfinite(dur) or dur < 0:
            dur = 0.0
        kinds = {s.get("codec_type") for s in info.get("streams", [])}
        return {"duration_s": dur, "has_audio": "audio" in kinds, "has_video": "video" in kinds}
    proc = subprocess.run([ffmpeg, "-i", path], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    text = proc.stderr.decode("utf-8", "replace")
    dur = 0.0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            # A stream with no container duration -- an elementary .h264, a live capture --
            # banners "Duration: N/A". Leave 0.0, the same answer an absent `format.duration`
            # gets on the ffprobe path, instead of crashing the unpack or the int() reads.
            fields = line.split(",")[0].split("Duration:")[1].strip().split(":")
            if len(fields) == 3:
                try:
                    dur = int(fields[0]) * 3600 + int(fields[1]) * 60 + float(fields[2])
                except ValueError:
                    pass
    return {"duration_s": dur, "has_audio": "Audio:" in text, "has_video": "Video:" in text}


def extract_audio(path: str) -> Optional[np.ndarray]:
    """The sound track as a 1-D float32 waveform at 16 kHz mono, or None when the clip has
    no audio stream."""
    exe = _need_ffmpeg()
    if not probe(path)["has_audio"]:
        return None
    out = _run([exe, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(TARGET_SR),
                "-f", "wav", "-acodec", "pcm_s16le", "-"], "extract the sound track")
    with _wave.open(io.BytesIO(out)) as w:
        frames = w.readframes(w.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return pcm


def has_speech(wave: Optional[np.ndarray], transcript: Optional[str] = None) -> bool:
    """The gate that chooses the path: a sound track quiet enough to be silence never reaches
    the transcriber, and a transcript shorter than a few words is not treated as speech."""
    if wave is None or wave.size == 0:
        return False
    if float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))) < SILENCE_RMS:
        return False
    if transcript is None:
        return True            # loud enough to be worth transcribing; the caller decides after
    # Words with letters or digits in them. A tone or music comes back from whisper as rows of
    # dots or dashes, which split() counts as words and this does not.
    return len(re.findall(r"[^\W_]+", transcript)) >= MIN_WORDS


def sample_frames(path: str, n: int = 4, duration_s: Optional[float] = None) -> Tuple[List, List[float]]:
    """``n`` frames at evenly spaced times, as PIL RGB images, with the times they came from.

    Times sit at the centres of ``n`` equal slices, so a two-frame sample of a ten-second clip
    reads 2.5 s and 7.5 s rather than the first and last frame, which are often black. A time
    whose frame ffmpeg cannot produce is skipped, and the times returned are the ones the
    frames actually came from.
    """
    exe = _need_ffmpeg()
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("video frames need Pillow: pip install pillow") from e
    if duration_s is None:
        duration_s = probe(path)["duration_s"]
    n = max(1, int(n))
    if duration_s <= 0:
        times = [0.0]
    else:
        times = [duration_s * (i + 0.5) / n for i in range(n)]
    frames, got = [], []
    for t in times:
        # Raw streams may be unable to seek even to zero. Decode their first frame directly.
        seek = ["-ss", "%.3f" % t] if t > 0 else []
        png = _run([exe, "-v", "error", *seek, "-i", path, "-frames:v", "1",
                    "-f", "image2pipe", "-vcodec", "png", "-"], "read a frame at %.1fs" % t)
        if not png:
            continue
        with Image.open(io.BytesIO(png)) as im:
            frames.append(im.convert("RGB"))
        got.append(t)
    if not frames:
        raise RuntimeError("no frame could be read from %r" % path)
    return frames, got


def contact_sheet(frames: Sequence, cell: int = 512):
    """The frames tiled left to right, top to bottom, into one image the vision tower reads
    at once. Each frame is fitted into a ``cell`` square, keeping its aspect ratio."""
    from PIL import Image
    if not frames:
        raise ValueError("no frames to tile")
    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    sheet = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    for i, fr in enumerate(frames):
        fr = fr.copy()
        fr.thumbnail((cell, cell))
        x = (i % cols) * cell + (cell - fr.width) // 2
        y = (i // cols) * cell + (cell - fr.height) // 2
        sheet.paste(fr, (x, y))
    return sheet
