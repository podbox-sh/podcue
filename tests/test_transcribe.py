"""Word mapping for Whisper and Moonshine. The Whisper engine is a fake; no weights are downloaded."""

import hashlib
import os
import sys
import types
from pathlib import Path

import pytest

from podcue.cues import Word
from podcue.transcribe import (
    WHISPER_EN,
    WHISPER_MULTI,
    WhisperPin,
    _ensure_whisper,
    moonshine_words,
    open_transcriber,
    whisper_pin,
    whisper_words,
)


def test_whisper_silence_has_no_words():
    assert whisper_words([]) == []
    assert whisper_words([{"text": "   ", "words": []}]) == []


def test_whisper_speech_words():
    segments = [{
        "text": "Hi there.",
        "words": [
            {"word": " Hi", "start": 0.1, "end": 0.3},
            {"word": " there.", "start": 0.35, "end": 0.7},
        ],
    }]
    assert whisper_words(segments) == [Word(" Hi", 0.1, 0.3), Word(" there.", 0.35, 0.7)]


def test_whisper_speech_without_words_is_an_error():
    with pytest.raises(ValueError, match="word timestamps"):
        whisper_words([{"text": "Hi", "words": []}])


def test_whisper_rejects_negative_times():
    with pytest.raises(ValueError):
        whisper_words([{"text": "x", "words": [{"word": "x", "start": -1, "end": 0}]}])


def test_english_uses_small_en_and_the_others_use_multilingual_small():
    assert whisper_pin("en") is WHISPER_EN
    assert WHISPER_EN.repo == "Systran/faster-whisper-small.en"
    assert whisper_pin("de") is WHISPER_MULTI
    assert whisper_pin("pl") is WHISPER_MULTI
    assert WHISPER_MULTI.repo == "Systran/faster-whisper-small"
    assert WHISPER_EN.revision != "main"
    assert WHISPER_MULTI.revision != "main"


def test_ensure_whisper_keeps_a_matching_file_and_rejects_a_bad_one(tmp_path):
    blob = b"good"
    pin = WhisperPin(
        repo="Systran/faster-whisper-small.en",
        revision="abc",
        model_bytes=len(blob),
        model_sha256=hashlib.sha256(blob).hexdigest(),
    )
    calls = {"n": 0}

    def download(repo, output_dir=None, revision=None):
        calls["n"] += 1
        assert (repo, revision) == (pin.repo, pin.revision)
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.bin").write_bytes(blob)
        return output_dir

    first = _ensure_whisper(pin, download, directory=tmp_path)
    second = _ensure_whisper(pin, download, directory=tmp_path)
    assert first == second == tmp_path / "abc"
    assert calls["n"] == 1

    bad = WhisperPin(pin.repo, "def", model_bytes=3, model_sha256="0" * 64)

    def wrong(repo, output_dir=None, revision=None):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.bin").write_bytes(b"nope")
        return output_dir

    with pytest.raises(RuntimeError, match="checksum"):
        _ensure_whisper(bad, wrong, directory=tmp_path)


def test_whisper_transcriber_maps_a_fake_engine_and_stays_quiet(monkeypatch, tmp_path):
    blob = b"w"
    pin = WhisperPin(
        repo="Systran/faster-whisper-small.en",
        revision="fake",
        model_bytes=len(blob),
        model_sha256=hashlib.sha256(blob).hexdigest(),
    )
    seen: dict[str, object] = {}

    class Model:
        def __init__(self, path, device, compute_type, cpu_threads, local_files_only):
            seen["init"] = (path, device, compute_type, cpu_threads, local_files_only)

        def transcribe(self, samples, **kwargs):
            seen["samples"] = samples
            seen["kwargs"] = kwargs

            class Word:
                def __init__(self, word, start, end):
                    self.word = word
                    self.start = start
                    self.end = end

            class Segment:
                text = "Hi there."
                words = [Word("Hi", 0.0, 0.2), Word(" there.", 0.2, 0.5)]

            return [Segment()], None

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = Model
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, revision=None, local_dir=None, allow_patterns=None):
        seen["download"] = (repo_id, revision)
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.bin").write_bytes(blob)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr("podcue.transcribe.whisper_pin", lambda _language: pin)
    monkeypatch.setattr("podcue.transcribe.cache_dir", lambda: tmp_path)
    monkeypatch.delenv("HF_HUB_VERBOSITY", raising=False)

    from podcue.transcribe import WhisperTranscriber

    engine = WhisperTranscriber("en")
    words = engine.transcribe(object())
    assert words == [Word("Hi", 0.0, 0.2), Word(" there.", 0.2, 0.5)]
    assert seen["download"] == (pin.repo, pin.revision)
    assert seen["init"][1:] == ("cpu", "int8", 4, True)
    assert seen["kwargs"]["language"] == "en"
    assert seen["kwargs"]["word_timestamps"] is True
    assert seen["kwargs"]["vad_filter"] is False
    assert os.environ["HF_HUB_VERBOSITY"] == "error"


def test_a_missing_whisper_install_names_the_asr_extra(monkeypatch):
    import builtins

    real = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "faster_whisper" or name.startswith("faster_whisper."):
            raise ModuleNotFoundError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    monkeypatch.delitem(sys.modules, "faster_whisper.utils", raising=False)
    from podcue.transcribe import WhisperTranscriber

    with pytest.raises(RuntimeError, match=r"podcue\[asr\]"):
        WhisperTranscriber("de")


def test_moonshine_line_without_word_times_becomes_one_word():
    lines = [
        {"text": "こんにちは。", "start_time": 1.0, "duration": 0.5, "words": []},
        {"text": "", "start_time": 2.0, "duration": 0.1, "words": []},
        {"text": "x", "start_time": 3.0, "duration": 1.0, "words": [{"word": "x", "start": 3.1, "end": 3.4}]},
    ]
    assert moonshine_words(lines) == [Word("こんにちは。", 1.0, 1.5), Word("x", 3.1, 3.4)]


def test_unknown_language_is_refused():
    with pytest.raises(ValueError, match="unsupported language"):
        open_transcriber("xx")
