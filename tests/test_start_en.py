"""A real `podcue start` on a short clip. CI sets PODCUE_LIVE_EN=1.

One run uses built-in ASR (the asr extra). One passes `--transcript` and must not need the
words to come from the model. Both must come back with cues.
"""

import json
import os
import shutil
import subprocess
import wave

import pytest
from conftest import run

SENTENCE = "Hello from podcue. This is a short English sentence about cities and homes."


def _require_live() -> None:
    if os.environ.get("PODCUE_LIVE_EN") != "1":
        pytest.skip("set PODCUE_LIVE_EN=1 to run a real start")
    missing = [tool for tool in ("ffmpeg", "espeak-ng") if shutil.which(tool) is None]
    if missing:
        pytest.fail(f"live start needs {', '.join(missing)} on PATH")


def _speak(tmp_path, text: str):
    raw = tmp_path / "espeak.wav"
    wav = tmp_path / "clip.wav"
    subprocess.run(
        ["espeak-ng", "-v", "en-us", "-w", str(raw), text],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
         "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
    )
    with wave.open(str(wav), "rb") as reader:
        duration_ms = reader.getnframes() * 1000 // reader.getframerate()
    return wav, duration_ms


def _cues(capsys, episode_id: str) -> list[dict]:
    page, code = run(capsys, "cues", episode_id, "--limit", "50")
    assert code == 0 and page["state"] == "present", page
    return page["cues"]


def test_builtin_start_produces_cues(tmp_path, monkeypatch, capsys):
    _require_live()
    monkeypatch.setenv("PODCUE_HOME", str(tmp_path / "home"))
    wav, _duration = _speak(tmp_path, SENTENCE)
    started, code = run(capsys, "start", wav, "--language", "en")
    assert code == 0, started
    assert started["kind"] == "started" and started["episode"]["language"] == "en"
    cues = _cues(capsys, started["episode"]["episodeId"])
    assert cues and all(cue["text"].strip() for cue in cues)
    print("BUILTIN_CUES", json.dumps([cue["text"] for cue in cues], ensure_ascii=False))


def test_transcript_start_produces_cues(tmp_path, monkeypatch, capsys):
    _require_live()
    monkeypatch.setenv("PODCUE_HOME", str(tmp_path / "home"))
    text = "Hello from podcue. A transcript stands in for the built-in model."
    wav, duration_ms = _speak(tmp_path, text)
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps({
        "language": "en",
        "segments": [{"startMs": 0, "endMs": duration_ms, "text": text}],
    }), encoding="utf-8")
    started, code = run(capsys, "start", wav, "--transcript", path)
    assert code == 0, started
    assert started["episode"]["language"] == "en"
    cues = _cues(capsys, started["episode"]["episodeId"])
    heard = " ".join(cue["text"] for cue in cues)
    assert "podcue" in heard
    print("TRANSCRIPT_CUES", json.dumps([cue["text"] for cue in cues], ensure_ascii=False))


def test_faster_whisper_decodes_a_wav(tmp_path):
    """PyAV 19 removed av.open(metadata_errors=...). faster-whisper 1.2.1 still passes it."""
    _require_live()
    faster_whisper = pytest.importorskip("faster_whisper")
    from faster_whisper.audio import decode_audio

    wav = tmp_path / "tone.wav"
    with wave.open(str(wav), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x10" * 16_000)
    decoded = decode_audio(str(wav), sampling_rate=16_000)
    assert decoded.shape[0] > 1_000
    assert faster_whisper.audio.decode_audio is decode_audio


def test_real_whisper_skips_a_one_frame_slice_and_keeps_going():
    """10 ms of audio makes faster-whisper align with num_frames=1 and raise."""
    _require_live()
    pytest.importorskip("faster_whisper")
    import numpy as np

    from podcue.pipeline import run_slices
    from podcue.regions import Slice
    from podcue.transcribe import WhisperTranscriber

    transcriber = WhisperTranscriber("en")
    one_frame = np.zeros(160, dtype=np.float32)
    silence = np.zeros(8_000, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * np.arange(16_000) / 16_000).astype(np.float32)
    noise = np.random.default_rng(1).standard_normal(16_000).astype(np.float32) * 0.05
    pieces = [np.zeros(0, dtype=np.float32), one_frame, silence, tone, noise]
    samples = np.concatenate(pieces)
    plan = []
    cursor = 0
    for index, piece in enumerate(pieces):
        start = cursor * 1000 // 16_000
        cursor += len(piece)
        end = cursor * 1000 // 16_000
        plan.append(Slice(index, start, end, start, end, 0))
    results = run_slices(samples, plan, transcriber, None, audio_id="live", language="en")
    assert len(results) == len(plan)
    assert results[0][1] == []
    assert results[1][1] == []
