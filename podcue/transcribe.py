"""Speech to timed words. cactus-needle (Whistle) for European languages, Moonshine for ja/ko/zh."""

from __future__ import annotations

import math
import os
from typing import Protocol

import numpy as np

from .cues import Word
from .weights import cache_dir

NEEDLE_LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")
# Moonshine architecture per language: 0 is tiny, 1 is base.
MOONSHINE_ARCH = {"ja": 0, "ko": 0, "zh": 1}
LANGUAGES = NEEDLE_LANGUAGES + tuple(MOONSHINE_ARCH)

MOONSHINE_MISSING = (
    "Japanese, Korean and Chinese use Moonshine, which is an optional extra. "
    "Run: uvx --from 'podcue[moonshine]' podcue start <audio> --language <code>"
)


class Transcriber(Protocol):
    def transcribe(self, samples: np.ndarray) -> list[Word]: ...


def open_transcriber(language: str) -> Transcriber:
    if language in NEEDLE_LANGUAGES:
        return NeedleTranscriber(language)
    if language in MOONSHINE_ARCH:
        return MoonshineTranscriber(language)
    raise ValueError(f"unsupported language {language!r}; use one of {', '.join(LANGUAGES)}")


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"transcriber field {field} must be a non-negative finite number, got {value!r}")
    return float(value)


def needle_words(result: dict) -> list[Word]:
    """Words from one needle.transcribe result. Silence has no `words` key."""
    words = result.get("words")
    if words is None:
        if result.get("text"):
            raise ValueError("cactus-needle returned speech without word timestamps")
        return []
    return [Word(str(w["word"]), _finite(w["start"], "start"), _finite(w["end"], "end")) for w in words]


def moonshine_words(lines: list[dict]) -> list[Word]:
    """Words from Moonshine lines. A line with text and no word times becomes one word."""
    out: list[Word] = []
    for line in lines:
        timed = line.get("words") or []
        if timed:
            out.extend(Word(w["word"], _finite(w["start"], "start"), _finite(w["end"], "end")) for w in timed)
        elif line.get("text"):
            start = _finite(line["start_time"], "start_time")
            out.append(Word(line["text"], start, start + _finite(line["duration"], "duration")))
    return out


class NeedleTranscriber:
    def __init__(self, language: str):
        # cactus-needle sends anonymous usage counts unless NEEDLE_TELEMETRY is "0".
        # Off by default here; a value the user set is left alone.
        os.environ.setdefault("NEEDLE_TELEMETRY", "0")
        import needle

        self._needle = needle
        self._language = language

    def transcribe(self, samples: np.ndarray) -> list[Word]:
        result = self._needle.transcribe(samples, language=self._language, word_timestamps=True)
        return needle_words(result)


class MoonshineTranscriber:
    def __init__(self, language: str):
        try:
            import moonshine_voice
        except ModuleNotFoundError as error:
            raise RuntimeError(MOONSHINE_MISSING) from error
        arch = moonshine_voice.ModelArch(MOONSHINE_ARCH[language])
        path, arch = moonshine_voice.get_model_for_language(
            language, arch, cache_root=cache_dir() / "moonshine", include_word_timestamps=True
        )
        self._transcriber = moonshine_voice.Transcriber(
            model_path=path,
            model_arch=arch,
            options={"word_timestamps": "true", "max_tokens_per_second": "13.0"},
        )

    def transcribe(self, samples: np.ndarray) -> list[Word]:
        transcript = self._transcriber.transcribe_without_streaming(samples.tolist())
        lines = [
            {
                "text": line.text,
                "start_time": float(line.start_time),
                "duration": float(line.duration),
                "words": [
                    {"word": w.word, "start": float(w.start), "end": float(w.end)} for w in (line.words or [])
                ],
            }
            for line in transcript.lines
        ]
        return moonshine_words(lines)
