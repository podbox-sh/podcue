"""Transcript JSON: schema, language agreement, and cue placement without ASR."""

import json
import subprocess
import sys

import pytest
from conftest import make_wav, run

from podcue.cues import Word
from podcue.regions import Slice
from podcue.transcript import TranscriptError, assign_words, parse_transcript, resolve_language


def test_importing_podcue_does_not_load_asr():
    script = (
        "import podcue.cli, podcue.pipeline, sys; "
        "assert 'cactus' not in sys.modules; "
        "assert 'needle' not in sys.modules; "
        "assert 'moonshine_voice' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_parse_spreads_text_when_words_are_omitted_and_keeps_explicit_words():
    loaded = parse_transcript({
        "language": "EN",
        "segments": [
            {"startMs": 0, "endMs": 1000, "text": "ab c"},
            {"startMs": 1000, "endMs": 1600, "text": "Hello there.", "words": [
                {"startMs": 1000, "endMs": 1200, "text": "Hello"},
                {"startMs": 1200, "endMs": 1600, "text": "there."},
            ]},
        ],
    })
    assert loaded.language == "en"
    assert loaded.words == [
        (0, 600, "ab"), (600, 1000, "c"),
        (1000, 1200, "Hello"), (1200, 1600, "there."),
    ]


def test_parse_rejects_a_bad_transcript():
    with pytest.raises(TranscriptError, match="transcript must be"):
        parse_transcript([{"startMs": 0, "endMs": 1, "text": "Hi."}])
    with pytest.raises(TranscriptError, match="language"):
        parse_transcript({"language": "", "segments": [{"startMs": 0, "endMs": 1, "text": "Hi."}]})
    with pytest.raises(TranscriptError, match="endMs must be after"):
        parse_transcript({"segments": [{"startMs": 5, "endMs": 5, "text": "Hi."}]})
    with pytest.raises(TranscriptError, match="outside the segment"):
        parse_transcript({"language": "pt", "segments": [{
            "startMs": 0, "endMs": 100, "text": "Hi.",
            "words": [{"startMs": 0, "endMs": 200, "text": "Hi."}],
        }]})


def test_resolve_language_accepts_any_code_and_rejects_a_disagreement():
    assert resolve_language(None, "pt-BR") == "pt-br"
    assert resolve_language("SV", None) == "sv"
    assert resolve_language("en", "EN") == "en"
    with pytest.raises(TranscriptError, match="does not match"):
        resolve_language("en", "sv")
    with pytest.raises(TranscriptError, match="language is missing"):
        resolve_language(None, None)


def test_words_outside_detected_speech_are_left_out():
    piece = Slice(0, 0, 2_000, 0, 2_000, 0)
    results, dropped = assign_words([piece], [(100, 400, "Hello"), (5_000, 5_500, "later")])
    assert dropped == 1
    assert results == [(piece, [Word("Hello", 0.1, 0.4)])]


def test_transcript_language_can_be_outside_the_builtin_list(engine, root, tmp_path, capsys):
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps({
        "language": "pt",
        "segments": [{"startMs": 0, "endMs": 800, "text": "Ola."}],
    }), encoding="utf-8")
    result, code = run(capsys, "start", make_wav(tmp_path / "ep.wav"), "--transcript", path)
    assert code == 0, result
    assert result["episode"]["language"] == "pt"


def test_flag_supplies_the_language_when_the_file_omits_it(engine, root, tmp_path, capsys):
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps({
        "segments": [{"startMs": 0, "endMs": 800, "text": "Hej."}],
    }), encoding="utf-8")
    result, code = run(capsys, "start", make_wav(tmp_path / "ep.wav", seed=1),
                       "--language", "sv", "--transcript", path)
    assert code == 0, result
    assert result["episode"]["language"] == "sv"


def test_transcript_language_errors_are_one_json_line(tmp_path, capsys):
    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text(json.dumps({
        "language": "sv",
        "segments": [{"startMs": 0, "endMs": 800, "text": "Hej."}],
    }), encoding="utf-8")
    result, code = run(capsys, "start", "ep.wav", "--language", "en", "--transcript", mismatch)
    assert code == 1 and result["kind"] == "error"
    assert result["message"] == "--language en does not match the transcript language sv"
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({
        "segments": [{"startMs": 0, "endMs": 800, "text": "Hi."}],
    }), encoding="utf-8")
    result, code = run(capsys, "start", "ep.wav", "--transcript", missing)
    assert code == 1 and result == {
        "kind": "error",
        "message": 'transcript language is missing; set "language" in the file or pass --language',
    }


def test_transcript_path_does_not_open_asr(monkeypatch):
    import numpy as np

    from podcue import pipeline, weights
    from podcue.diarize import Tracks, Turn

    def fail_open(_language):
        raise AssertionError("built-in ASR must not load")

    class Fake:
        def __init__(self, *_args):
            return None

        def diarize(self, _samples):
            return Tracks([Turn(0, 1_000, 0)], [[1.0, 0.0]])

    monkeypatch.setattr(pipeline, "open_transcriber", fail_open)
    monkeypatch.setattr(pipeline, "Diarizer", Fake)
    monkeypatch.setattr(weights, "ensure", lambda _pin: _pin.file)
    cues, _tracks = pipeline.transcribe(
        np.zeros(16_000, dtype=np.float32), "sv", lambda *_args: None,
        [(0, 400, "Hello"), (400, 900, "there.")],
    )
    assert [cue.text for cue in cues] == ["Hello there."]


def test_builtin_unknown_language_is_still_rejected(engine, root, tmp_path, capsys):
    result, code = run(capsys, "start", make_wav(tmp_path / "ep.wav"), "--language", "pt")
    assert code == 1 and result["reason"] == "unknown-language"
