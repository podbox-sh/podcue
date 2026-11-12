"""Speech to timed words. Whisper small for European languages, Moonshine for ja/ko/zh.

Both engines are imported only when a built-in transcription actually runs.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .cues import Word
from .weights import cache_dir, sha256_file

WHISPER_LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")
# Moonshine architecture per language: 0 is tiny, 1 is base.
MOONSHINE_ARCH = {"ja": 0, "ko": 0, "zh": 1}
LANGUAGES = WHISPER_LANGUAGES + tuple(MOONSHINE_ARCH)

MOONSHINE_MISSING = (
    "Japanese, Korean and Chinese use Moonshine, which is an optional extra. "
    "Run: uvx --from 'podcue[moonshine]' podcue start <audio> --language <code>"
)
ASR_MISSING = (
    "built-in speech-to-text is the asr extra (faster-whisper, Whisper small). "
    "Run: uvx --from 'podcue[asr]' podcue start <audio> "
    "or pass --transcript with timed words from your own model."
)


@dataclass(frozen=True)
class WhisperPin:
    """One CTranslate2 Whisper checkpoint. The revision is a commit, not a branch."""

    repo: str
    revision: str
    model_bytes: int
    model_sha256: str


# English-only small.en, and multilingual small for the other six languages.
# Sizes and SHA-256 are the Git LFS oid of model.bin at that commit.
WHISPER_EN = WhisperPin(
    repo="Systran/faster-whisper-small.en",
    revision="d1d751a5f8271d482d14ca55d9e2deeebbae577f",
    model_bytes=483_545_366,
    model_sha256="62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a",
)
WHISPER_MULTI = WhisperPin(
    repo="Systran/faster-whisper-small",
    revision="536b0662742c02347bc0e980a01041f333bce120",
    model_bytes=483_546_902,
    model_sha256="3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
)


def whisper_pin(language: str) -> WhisperPin:
    return WHISPER_EN if language == "en" else WHISPER_MULTI


def quiet_hub() -> None:
    """Public pinned downloads do not need a Hugging Face token.

    huggingface_hub warns when `HF_TOKEN` is unset. Podcue never sends one. The verbosity
    variable is read when that library is imported, so it is set before the import.
    """
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    warnings.filterwarnings("ignore", message=r".*HF_TOKEN.*")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    hub_logging = sys.modules.get("huggingface_hub.utils.logging")
    if hub_logging is not None and hasattr(hub_logging, "set_verbosity_error"):
        hub_logging.set_verbosity_error()


class Transcriber(Protocol):
    def transcribe(self, samples: np.ndarray) -> list[Word]: ...


def open_transcriber(language: str) -> Transcriber:
    if language in WHISPER_LANGUAGES:
        return WhisperTranscriber(language)
    if language in MOONSHINE_ARCH:
        return MoonshineTranscriber(language)
    raise ValueError(f"unsupported language {language!r}; use one of {', '.join(LANGUAGES)}")


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"transcriber field {field} must be a non-negative finite number, got {value!r}")
    return float(value)


def whisper_words(segments: list[dict]) -> list[Word]:
    """Words from faster-whisper segments. Silence has no segments.

    A segment that has text and no word times is an error: cues need a start and an end.
    """
    out: list[Word] = []
    for segment in segments:
        words = segment.get("words") or []
        if words:
            out.extend(
                Word(str(word["word"]), _finite(word["start"], "start"), _finite(word["end"], "end"))
                for word in words
            )
        elif str(segment.get("text") or "").strip():
            raise ValueError("Whisper returned speech without word timestamps")
    return out


def _model_matches(path: Path, pin: WhisperPin) -> bool:
    return path.is_file() and path.stat().st_size == pin.model_bytes and sha256_file(path) == pin.model_sha256


def _ensure_whisper(pin: WhisperPin, download: Callable[..., object], directory: Path | None = None) -> Path:
    """Download the pinned checkpoint into the cache and check model.bin."""
    dest = (directory or cache_dir() / "whisper") / pin.revision
    model = dest / "model.bin"
    if _model_matches(model, pin):
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    download(pin.repo, output_dir=str(dest), revision=pin.revision)
    if not _model_matches(model, pin):
        raise RuntimeError(
            f"Whisper weights for {pin.repo}@{pin.revision} failed the checksum "
            f"(expected {pin.model_bytes} bytes, sha256 {pin.model_sha256})"
        )
    return dest


class WhisperTranscriber:
    """Whisper small on CPU. English uses small.en; the other six languages use multilingual small."""

    def __init__(self, language: str):
        quiet_hub()
        try:
            from faster_whisper import WhisperModel
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as error:
            raise RuntimeError(ASR_MISSING) from error
        pin = whisper_pin(language)

        def download(repo: str, output_dir: str | None = None, revision: str | None = None) -> None:
            snapshot_download(
                repo_id=repo,
                revision=revision,
                local_dir=output_dir,
                allow_patterns=[
                    "config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*",
                ],
            )

        path = _ensure_whisper(pin, download)
        # int8 is the CPU quantization. A GPU, if present, is not used.
        self._model = WhisperModel(
            str(path), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True,
        )
        self._language = language

    def transcribe(self, samples: np.ndarray) -> list[Word]:
        segments, _info = self._model.transcribe(
            samples,
            language=self._language,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        packed = [
            {
                "text": segment.text,
                "words": [
                    {"word": word.word, "start": word.start, "end": word.end}
                    for word in (segment.words or [])
                ],
            }
            for segment in segments
        ]
        return whisper_words(packed)


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
