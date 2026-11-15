"""Plan transcription slices from speaker turns.

Same-speaker turns a short gap apart are joined, and turns too short to transcribe alone are
absorbed into a neighbor. A region longer than one slice is cut at the quietest 100 ms in the
last 6 seconds of each slice, so the edge falls in a pause instead of inside a word. The cut stays
at least 2 seconds before the region's end, so the last slice is never a sliver. Slices do
not overlap, and a word belongs to the slice where it starts.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

from .diarize import Turn
from .pauses import CUT_SEARCH_MS, MIN_LAST_SLICE_MS, Quiet

SLICE_MS = 28_000  # under the transcriber's 30 second window
JOIN_GAP_MS = 500
SHORT_TURN_MS = 500


class Slice(NamedTuple):
    index: int
    start_ms: int
    end_ms: int
    region: int


def plan_regions(turns: list[Turn], quiet: Quiet | None = None) -> list[Slice]:
    """Slices of each speaker region. Without `quiet`, a long region is cut every SLICE_MS."""
    slices: list[Slice] = []
    for region, (start, end) in enumerate(_absorb_short(_join_same_speaker(turns))):
        _append_span(slices, start, end, region, quiet)
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


def _append_span(slices: list[Slice], start: int, end: int, region: int, quiet: Quiet | None) -> None:
    cursor = start
    while end - cursor > SLICE_MS:
        stop = cursor + SLICE_MS
        cut = stop
        if quiet is not None:
            cut = quiet.quietest_ms(stop - CUT_SEARCH_MS, min(stop, end - MIN_LAST_SLICE_MS))
            if not quiet.is_quiet_ms(cut):
                print(f"podcue: no pause near {cut / 1000:.2f} s, so the slice cut there may split a word",
                      file=sys.stderr, flush=True)
        slices.append(Slice(len(slices), cursor, cut, region))
        cursor = cut
    slices.append(Slice(len(slices), cursor, end, region))
