"""Build cues from timed words, attach diarization speaker groups, and render WebVTT.

A cue time is only ever a slice start plus a transcriber word time, so a rerun on the same
words gives the same cues.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple

from .diarize import Tracks, Turn
from .regions import Slice

UNGROUPED = "ungrouped"
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


def cue_rows(cues: list[Cue], tracks: Tracks) -> list[dict]:
    """The `cues.json` rows. A voice's group id is `g` + its first cue id; no voice is `ungrouped`.

    `nearestGroupId`/`nearestCosine` name the other group whose ECAPA centroid is most similar to
    this cue's group. Ties go to the smaller group id. Both are null for `ungrouped` and when
    there is only one group.
    """
    speakers = [overlapping_speaker(tracks.turns, cue.start_ms, cue.end_ms) for cue in cues]
    group_of: dict[int, str] = {}
    for cue, speaker in zip(cues, speakers):
        if speaker is not None and speaker not in group_of:
            group_of[speaker] = f"g{cue.id}"

    nearest: dict[int, tuple[str | None, float | None]] = {}
    for speaker, group_id in group_of.items():
        best_id, best_score = None, None
        for other, other_id in group_of.items():
            if other == speaker:
                continue
            score = round(cosine(tracks.centroids[speaker], tracks.centroids[other]), 4)
            if best_score is None or score > best_score or (score == best_score and other_id < best_id):
                best_id, best_score = other_id, score
        nearest[speaker] = (best_id, best_score)

    rows = []
    for cue, speaker in zip(cues, speakers):
        group_id = UNGROUPED if speaker is None else group_of[speaker]
        near_id, near_score = (None, None) if speaker is None else nearest[speaker]
        rows.append({
            "id": cue.id,
            "startMs": cue.start_ms,
            "endMs": cue.end_ms,
            "text": cue.text,
            "groupId": group_id,
            "nearestGroupId": near_id,
            "nearestCosine": near_score,
        })
    return rows


def group_table(rows: list[dict]) -> list[dict]:
    """The `groups.json` rows, in order of each group's first cue.

    `speakingMs` is the sum of the group's cue durations (endMs - startMs), the same measure
    `labeledShare` uses.
    """
    groups: dict[str, dict] = {}
    for row in rows:
        group = groups.setdefault(row["groupId"], {
            "groupId": row["groupId"],
            "speakingMs": 0,
            "cueCount": 0,
            "nearestGroupId": row["nearestGroupId"],
            "nearestCosine": row["nearestCosine"],
        })
        group["speakingMs"] += row["endMs"] - row["startMs"]
        group["cueCount"] += 1
    return list(groups.values())


def stamp(ms: int) -> str:
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_vtt(rows: list[dict]) -> str:
    blocks = ["WEBVTT"]
    for cue in rows:
        text = _escape(cue["text"])
        if cue["groupId"] != UNGROUPED:
            text = f"<v {cue['groupId']}>{text}"
        blocks.append(f"{cue['id']}\n{stamp(cue['startMs'])} --> {stamp(cue['endMs'])}\n{text}")
    return "\n\n".join(blocks) + "\n"
