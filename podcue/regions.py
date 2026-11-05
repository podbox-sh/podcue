"""Plan transcription slices from speaker turns.

Same-speaker turns a short gap apart are joined, turns too short to transcribe alone are
absorbed into a neighbor, and a region longer than one slice is cut on a fixed grid with
a one-second overlap. Only words that start inside a slice's keep window become cues.
"""

from __future__ import annotations

from typing import NamedTuple

from .diarize import Turn

SLICE_MS = 28_000  # under the transcriber's 30 second window
OVERLAP_MS = 1_000  # a word cut at a slice edge is heard whole in the next slice
STEP_MS = SLICE_MS - OVERLAP_MS
JOIN_GAP_MS = 500
SHORT_TURN_MS = 500


class Slice(NamedTuple):
    index: int
    start_ms: int
    end_ms: int
    keep_start_ms: int
    keep_end_ms: int
    region: int


def plan_regions(turns: list[Turn]) -> list[Slice]:
    slices: list[Slice] = []
    for region, (start, end) in enumerate(_absorb_short(_join_same_speaker(turns))):
        _append_span(slices, start, end, region)
    return slices


def _join_same_speaker(turns: list[Turn]) -> list[list[int]]:
    regions: list[list[int]] = []
    speaker = -1
    for turn in turns:
        if regions and turn.speaker == speaker and turn.start_ms - regions[-1][1] <= JOIN_GAP_MS:
            regions[-1][1] = turn.end_ms
            continue
        regions.append([turn.start_ms, turn.end_ms])
        speaker = turn.speaker
    return regions


def _absorb_short(regions: list[list[int]]) -> list[list[int]]:
    pending = [list(region) for region in regions]
    kept: list[list[int]] = []
    for index, region in enumerate(pending):
        if region[1] - region[0] >= SHORT_TURN_MS:
            kept.append(region)
            continue
        if kept and region[0] - kept[-1][1] <= JOIN_GAP_MS:
            kept[-1][1] = region[1]
            continue
        following = pending[index + 1] if index + 1 < len(pending) else None
        if following is not None and following[0] - region[1] <= JOIN_GAP_MS:
            following[0] = region[0]
            continue
        kept.append(region)
    return kept


def _append_span(slices: list[Slice], start: int, end: int, region: int) -> None:
    if end - start <= SLICE_MS:
        slices.append(Slice(len(slices), start, end, start, end, region))
        return
    cursor = start
    while cursor < end:
        stop = min(cursor + SLICE_MS, end)
        keep_end = end if stop == end else cursor + STEP_MS
        slices.append(Slice(len(slices), cursor, stop, cursor, keep_end, region))
        if stop == end:
            break
        cursor += STEP_MS
