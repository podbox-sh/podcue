"""Build cues from timed words, attach diarization speakers, and render JSON and WebVTT.

A cue time is only ever a slice start plus a transcriber word time, so a rerun on the same
words gives the same cues.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple

from .diarize import Tracks, Turn
from .regions import Slice

MAX_CUE_MS = 12_000  # a cue closes even when the transcriber left out punctuation
_SENTENCE_END = re.compile(r"[.!?。！？]$")  # CJK sentences end on 。！？


class Word(NamedTuple):
    """One word as the transcriber returned it, in seconds from the slice start."""

    text: str
    start: float
    end: float


class Cue(NamedTuple):
    id: str
    start_ms: int
    end_ms: int
    text: str


def shift(slice_start_ms: int, seconds: float) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"word time must be a non-negative finite second, got {seconds}")
    return slice_start_ms + math.floor(seconds * 1000 + 0.5)


def cue_id(index: int) -> str:
    return f"c{index + 1:04d}"


def build_cues(results: list[tuple[Slice, list[Word]]]) -> list[Cue]:
    """Keep each word that starts inside its slice's keep window, then group words into cues.

    A cue closes on sentence punctuation, after MAX_CUE_MS, and where one speaker region ends.
    """
    cues: list[Cue] = []
    bucket: list[tuple[int, int, str]] = []
    bucket_region = -1

    def close() -> None:
        nonlocal bucket
        if bucket:
            text = " ".join(word for _, _, word in bucket)
            cues.append(Cue(cue_id(len(cues)), bucket[0][0], bucket[-1][1], text))
            bucket = []

    for piece, words in results:
        for word in words:
            text = word.text.strip()
            if not text:
                continue
            start = shift(piece.start_ms, word.start)
            end = shift(piece.start_ms, word.end)
            if end <= start:
                end = start + 1
            if not piece.keep_start_ms <= start < piece.keep_end_ms:
                continue
            if bucket and (piece.region != bucket_region or start - bucket[0][0] >= MAX_CUE_MS):
                close()
            bucket_region = piece.region
            bucket.append((start, end, text))
            if _SENTENCE_END.search(text):
                close()
    close()
    return cues


def overlapping_speaker(turns: list[Turn], start_ms: int, end_ms: int) -> int | None:
    best, best_overlap = None, 0
    for turn in turns:
        overlap = min(end_ms, turn.end_ms) - max(start_ms, turn.start_ms)
        if overlap > best_overlap:
            best, best_overlap = turn.speaker, overlap
    return best


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 0.0 if norm == 0 else dot / norm


def cue_document(
    *, source: str, duration_ms: int, language: str, cues: list[Cue], tracks: Tracks
) -> dict:
    """The `.cues.json` shape. Speakers are numbered S1, S2, ... in order of their first cue."""
    raw_speakers = [overlapping_speaker(tracks.turns, cue.start_ms, cue.end_ms) for cue in cues]
    ids: dict[int, str] = {}
    for raw in raw_speakers:
        if raw is not None and raw not in ids:
            ids[raw] = f"S{len(ids) + 1}"

    speakers = []
    for raw, sid in ids.items():
        nearest_id, nearest_score = None, None
        for other, other_id in ids.items():
            if other == raw:
                continue
            score = cosine(tracks.centroids[raw], tracks.centroids[other])
            if nearest_score is None or score > nearest_score:
                nearest_id, nearest_score = other_id, score
        speakers.append({
            "id": sid,
            "speakingMs": sum(t.end_ms - t.start_ms for t in tracks.turns if t.speaker == raw),
            "cueCount": sum(1 for r in raw_speakers if r == raw),
            "nearestSpeaker": nearest_id,
            "nearestCosine": None if nearest_score is None else round(nearest_score, 4),
        })

    return {
        "version": 1,
        "source": source,
        "durationMs": duration_ms,
        "language": language,
        "speakers": speakers,
        "cues": [
            {
                "id": cue.id,
                "text": cue.text,
                "startMs": cue.start_ms,
                "endMs": cue.end_ms,
                "speaker": None if raw is None else ids[raw],
            }
            for cue, raw in zip(cues, raw_speakers)
        ],
    }


def stamp(ms: int) -> str:
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_vtt(document: dict) -> str:
    blocks = ["WEBVTT"]
    for cue in document["cues"]:
        text = _escape(cue["text"])
        if cue["speaker"] is not None:
            text = f"<v {cue['speaker']}>{text}"
        blocks.append(f"{cue['id']}\n{stamp(cue['startMs'])} --> {stamp(cue['endMs'])}\n{text}")
    return "\n\n".join(blocks) + "\n"
