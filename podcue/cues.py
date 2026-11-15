"""Build cues from timed words, attach diarization speaker groups, and render WebVTT.

A cue time is only ever a slice start plus a transcriber word time, so a rerun on the same
words gives the same cues. Built-in cues also close at pauses in the audio, so a clip cut from
whole cues ends in a pause. Cues from an agent's transcript close only on punctuation, speaker
regions, and the 12 s limit.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple

from .diarize import Tracks, Turn
from .pauses import PAUSE_CLOSE_MS, Quiet
from .regions import Slice

UNGROUPED = "ungrouped"
UNCERTAIN = "uncertain"
MIXED = "mixed"
# A cue with one of these speakers keeps it when a group is labeled.
HELD_SPEAKERS = (UNCERTAIN, MIXED)
MAX_CUE_MS = 12_000  # a cue closes even when the transcriber left out punctuation
SLIP_MS = 1_500  # a cue this short at a turn boundary does not join the surrounding speaker
_CJK = {"ja", "zh", "ko"}
_LATIN_END = re.compile(r"[.!?]$")
_CJK_END = re.compile(r"[.!?。！？]$")  # ja, zh, and ko also close on 。！？


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


def sentence_end(language: str) -> re.Pattern[str]:
    """The punctuation that closes a cue for `language`.

    `ja`, `zh`, and `ko` (and codes that start with them, such as `zh-Hans`) also close on 。！？.
    Every other code closes on `.`, `!`, and `?`.
    """
    code = language.casefold().split("-", 1)[0]
    if code in _CJK:
        return _CJK_END
    return _LATIN_END


def join_cue_text(tokens: list[str]) -> str:
    """Join tokens, keeping spaces between words and gluing number punctuation.

    A token that starts with `,` `.` `%` or similar attaches to the previous token, and a
    digit token attaches when the previous token already ends in `,` or `.`. `400` + `,000`
    becomes `400,000`, and `2` + `.4%` becomes `2.4%`. Ordinary words stay separated.
    """
    text = ""
    for token in tokens:
        if not token:
            continue
        if not text:
            text = token
            continue
        if token[0] in ",.;:!?%)]}'\"\u2019\u201d" or (token[0].isdigit() and text[-1] in ",."):
            text += token
        else:
            text += " " + token
    return text


def build_cues(
    results: list[tuple[Slice, list[Word]]], language: str = "en", quiet: Quiet | None = None,
) -> list[Cue]:
    """Keep each word that starts inside its slice, then group words into cues.

    A cue closes on sentence punctuation for `language` and where one speaker region ends. With
    `quiet`, it also closes after a word followed by a pause of PAUSE_CLOSE_MS or more. A word that
    starts MAX_CUE_MS after the cue start closes the cue. With `quiet` the cue then ends after its
    word with the longest pause, and the words after that one start the next cue.
    """
    closes = sentence_end(language)
    words: list[tuple[int, int, str, int]] = []
    for piece, raw in results:
        for word in raw:
            text = word.text.strip()
            if not text:
                continue
            start = shift(piece.start_ms, word.start)
            end = shift(piece.start_ms, word.end)
            if end <= start:
                end = start + 1
            if piece.start_ms <= start < piece.end_ms:
                words.append((start, end, text, piece.region))
    pauses = [0] * len(words)
    if quiet is not None:
        for index in range(len(words) - 1):
            pauses[index] = quiet.pause_ms(words[index][1], words[index + 1][0])

    cues: list[Cue] = []

    def emit(indices: list[int]) -> None:
        text = join_cue_text([words[index][2] for index in indices])
        cues.append(Cue(cue_id(len(cues)), words[indices[0]][0], words[indices[-1]][1], text))

    bucket: list[int] = []
    for index, (start, _end, text, region) in enumerate(words):
        if bucket and words[bucket[-1]][3] != region:
            emit(bucket)
            bucket = []
        while bucket and start - words[bucket[0]][0] >= MAX_CUE_MS:
            # Ties go to the later word, so with no pause the cue closes where it would without `quiet`.
            cut = max(range(len(bucket)), key=lambda at: (pauses[bucket[at]], at)) + 1
            emit(bucket[:cut])
            bucket = bucket[cut:]
        bucket.append(index)
        if closes.search(text) or pauses[index] >= PAUSE_CLOSE_MS:
            emit(bucket)
            bucket = []
    if bucket:
        emit(bucket)
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


def _touching_speakers(turns: list[Turn], start_ms: int, end_ms: int) -> set[int]:
    found: set[int] = set()
    for turn in turns:
        if min(end_ms, turn.end_ms) - max(start_ms, turn.start_ms) > 0:
            found.add(turn.speaker)
    return found


def settle_boundary_slips(rows: list[dict], turns: list[Turn]) -> None:
    """Move a short cue off a neighbor's group when another voice's turn is inside it.

    The cue is under 1.5 s, the cues on both sides share one group, and this cue was given that
    same group. If a different diarization speaker also overlaps the cue, it stays `uncertain`
    instead of counting as that neighbor.
    """
    if len(rows) < 3:
        return
    for index in range(1, len(rows) - 1):
        row = rows[index]
        if row["endMs"] - row["startMs"] >= SLIP_MS:
            continue
        left, right = rows[index - 1], rows[index + 1]
        if left["groupId"] != right["groupId"] or row["groupId"] != left["groupId"]:
            continue
        if left["groupId"] in (UNGROUPED, UNCERTAIN):
            continue
        neighbor = overlapping_speaker(turns, left["startMs"], left["endMs"])
        if neighbor is None:
            continue
        others = _touching_speakers(turns, row["startMs"], row["endMs"]) - {neighbor}
        if not others:
            continue
        row["groupId"] = UNCERTAIN
        row["speaker"] = UNCERTAIN
        row["nearestGroupId"] = None
        row["nearestCosine"] = None


def cue_rows(cues: list[Cue], tracks: Tracks) -> list[dict]:
    """The `cues.json` rows. A voice's group id is `g` + its first cue id; no voice is `ungrouped`.

    `nearestGroupId`/`nearestCosine` name the other group whose ECAPA centroid is most similar to
    this cue's group. Ties go to the smaller group id. Both are null for `ungrouped`, `uncertain`,
    and when there is only one group. A cue under 1.5 s between one speaker, with another voice's
    turn inside it, is `uncertain` instead of that neighbor's group.
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
            "speaker": group_id,
            "nearestGroupId": near_id,
            "nearestCosine": near_score,
        })
    settle_boundary_slips(rows, tracks.turns)
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
