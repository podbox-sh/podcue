"""A transcript an agent already has, instead of built-in speech-to-text.

The file is one JSON object:

    {"language": "<code>", "segments": [{"startMs", "endMs", "text", "words"?}]}

`language` may be omitted when `podcue start --language` supplies it. Any code is accepted.
`words`, when present, are `{"startMs", "endMs", "text"}` inside the segment. Without them the
segment text is split on whitespace and spread across the segment by character length, so cue
building can still close on sentence punctuation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .cues import Word
from .regions import Slice

SHAPE = (
    'transcript must be {"language": "<code>", "segments": [{startMs, endMs, text, words?}]}'
)


class TranscriptError(Exception):
    """The transcript file cannot be used. The CLI prints it as `{"kind": "error"}`."""


@dataclass(frozen=True)
class Transcript:
    language: str | None
    words: list[tuple[int, int, str]]


def _code(value: str) -> str:
    return value.strip().casefold()


def resolve_language(flag: str | None, file_language: str | None) -> str:
    """The episode language. The flag and the file must agree when both are set.

    Comparison is case-insensitive. The stored code is the casefolded form.
    """
    flag_code = _code(flag) if flag is not None else None
    file_code = _code(file_language) if file_language is not None else None
    if flag_code == "":
        flag_code = None
    if file_code == "":
        file_code = None
    if flag_code and file_code and flag_code != file_code:
        raise TranscriptError(
            f"--language {flag} does not match the transcript language {file_language}"
        )
    chosen = file_code or flag_code
    if not chosen:
        raise TranscriptError(
            'transcript language is missing; set "language" in the file or pass --language'
        )
    return chosen


def _millis(value: object, index: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TranscriptError(f"segment {index}: {field} must be a non-negative integer millisecond")
    return value


def _word_millis(value: object, index: int, word_index: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TranscriptError(
            f"segment {index}: word {word_index}: {field} must be a non-negative integer millisecond"
        )
    return value


def _spread(text: str, start: int, end: int, index: int) -> list[tuple[int, int, str]]:
    tokens = text.split()
    if not tokens:
        raise TranscriptError(f"segment {index}: text must contain a word")
    span = end - start
    weight = sum(len(token) + 1 for token in tokens) or 1
    cursor = start
    words: list[tuple[int, int, str]] = []
    for token_index, token in enumerate(tokens):
        if token_index == len(tokens) - 1:
            word_end = end
        else:
            word_end = int(cursor + span * (len(token) + 1) / weight)
            if word_end <= cursor:
                word_end = cursor + 1
            if word_end > end:
                word_end = end
        words.append((cursor, word_end, token))
        cursor = word_end
    return words


def _explicit_words(raw: object, index: int, start: int, end: int) -> list[tuple[int, int, str]]:
    if not isinstance(raw, list) or not raw:
        raise TranscriptError(f"segment {index}: words must be a non-empty list")
    words: list[tuple[int, int, str]] = []
    previous = -1
    for word_index, word in enumerate(raw, 1):
        if not isinstance(word, dict):
            raise TranscriptError(f"segment {index}: word {word_index} must be an object")
        word_start = _word_millis(word.get("startMs"), index, word_index, "startMs")
        word_end = _word_millis(word.get("endMs"), index, word_index, "endMs")
        text = word.get("text")
        if not isinstance(text, str) or not text.strip():
            raise TranscriptError(f"segment {index}: word {word_index}: text must be a non-empty string")
        if word_end <= word_start:
            raise TranscriptError(f"segment {index}: word {word_index}: endMs must be after startMs")
        if word_start < start or word_end > end:
            raise TranscriptError(f"segment {index}: word {word_index} falls outside the segment")
        if word_start < previous:
            raise TranscriptError(f"segment {index}: words must be in time order")
        previous = word_start
        words.append((word_start, word_end, text.strip()))
    return words


def parse_transcript(value: object) -> Transcript:
    """Read one transcript object. `language` may be absent; the CLI can supply it."""
    if not isinstance(value, dict):
        raise TranscriptError(SHAPE)
    language: str | None
    if "language" not in value or value["language"] is None:
        language = None
    elif isinstance(value["language"], str) and value["language"].strip():
        language = _code(value["language"])
    else:
        raise TranscriptError('transcript "language" must be a non-empty string')
    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise TranscriptError(SHAPE)
    words: list[tuple[int, int, str]] = []
    previous = -1
    for index, segment in enumerate(raw_segments, 1):
        if not isinstance(segment, dict):
            raise TranscriptError(f"segment {index}: must be an object")
        start = _millis(segment.get("startMs"), index, "startMs")
        end = _millis(segment.get("endMs"), index, "endMs")
        text = segment.get("text")
        if not isinstance(text, str) or not text.strip():
            raise TranscriptError(f"segment {index}: text must be a non-empty string")
        if end <= start:
            raise TranscriptError(f"segment {index}: endMs must be after startMs")
        if start < previous:
            raise TranscriptError("segments must be in time order")
        previous = start
        if "words" in segment and segment["words"] is not None:
            segment_words = _explicit_words(segment["words"], index, start, end)
        else:
            segment_words = _spread(text.strip(), start, end, index)
        words.extend(segment_words)
    return Transcript(language, words)


def load_transcript(path: Path) -> Transcript:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise TranscriptError(f"{path} is not a file") from None
    except OSError as error:
        raise TranscriptError(f"{path} could not be read: {error}") from None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise TranscriptError(f"{path} is not JSON: {error}") from None
    return parse_transcript(value)


def assign_words(slices: list[Slice], words: list[tuple[int, int, str]]) -> tuple[list[tuple[Slice, list[Word]]], int]:
    """Words whose start falls in a slice's keep window, as seconds from that slice's start.

    Returns the slice pairs and how many words were left out. A word that starts outside every
    detected speech region is left out, the same as speech the diarizer misses.
    """
    results: list[tuple[Slice, list[Word]]] = []
    used = 0
    for piece in slices:
        chunk: list[Word] = []
        for start_ms, end_ms, text in words:
            if piece.keep_start_ms <= start_ms < piece.keep_end_ms:
                start_s = (start_ms - piece.start_ms) / 1000.0
                end_s = (end_ms - piece.start_ms) / 1000.0
                if end_s <= start_s:
                    end_s = start_s + 0.001
                chunk.append(Word(text, start_s, end_s))
                used += 1
        results.append((piece, chunk))
    return results, len(words) - used
