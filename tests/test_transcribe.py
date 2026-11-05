import pytest

from podcue.cues import Word
from podcue.transcribe import moonshine_words, needle_words, open_transcriber


def test_needle_silence_has_no_words():
    assert needle_words({"text": "", "language": "", "ttft_ms": 1.0, "decode_tps": 0.0}) == []


def test_needle_speech_words():
    result = {
        "text": "Hi there.",
        "language": "en",
        "words": [{"word": "Hi", "start": 0.1, "end": 0.3, "probability": 0.9},
                  {"word": "there.", "start": 0.35, "end": 0.7, "probability": 0.8}],
    }
    assert needle_words(result) == [Word("Hi", 0.1, 0.3), Word("there.", 0.35, 0.7)]


def test_needle_speech_without_words_is_an_error():
    with pytest.raises(ValueError):
        needle_words({"text": "Hi", "language": "en"})


def test_needle_rejects_negative_times():
    with pytest.raises(ValueError):
        needle_words({"text": "x", "words": [{"word": "x", "start": -1, "end": 0}]})


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
