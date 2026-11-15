"""Quiet stretches in 16 kHz samples, measured from 5 ms block energies.

A long speech region is cut at its quietest point so a slice edge falls between words. A built-in
cue closes at a pause between two words, so a clip cut from whole cues ends in that pause.
"""

from __future__ import annotations

import numpy as np

from .audio import RATE

HOP_MS = 5
WINDOW_MS = 20
QUIET_DB = -25.0
LOCAL_MS = 3_000  # a window is quiet against the level of the speech around it, not the episode
REACH_MS = 150  # a transcriber word edge can be this far off the pause it borders
PAUSE_CLOSE_MS = 250  # the low point between hesitations and phrase breaks on All-In and Huberman Lab
QUIETEST_MS = 100
CUT_SEARCH_MS = 6_000
_HOP = RATE * HOP_MS // 1000


class Quiet:
    """The energy of one episode's samples. Times are milliseconds from the start of the audio."""

    def __init__(self, samples: np.ndarray):
        whole = len(samples) // _HOP * _HOP
        blocks = np.asarray(samples[:whole], dtype=np.float32).reshape(-1, _HOP)
        energy = np.einsum("ij,ij->i", blocks, blocks).astype(np.float64)
        self._total = np.concatenate(([0.0], np.cumsum(energy)))
        width = WINDOW_MS // HOP_MS
        reach = LOCAL_MS // HOP_MS
        power = (self._total[width:] - self._total[:-width]) / width
        centre = np.arange(len(power)) + width // 2
        lo = np.maximum(centre - reach, 0)
        hi = np.minimum(centre + reach, len(energy))
        level = (self._total[hi] - self._total[lo]) / np.maximum(hi - lo, 1)
        self._quiet = power < level * 10 ** (QUIET_DB / 10)

    def pause_ms(self, end_ms: int, next_start_ms: int) -> int:
        """The longest run of quiet 20 ms windows, in ms, inside [end_ms - 150, next_start_ms + 150]."""
        first = max(0, -(-(end_ms - REACH_MS) // HOP_MS))
        last = min((next_start_ms + REACH_MS - WINDOW_MS) // HOP_MS, len(self._quiet) - 1)
        if last < first:
            return 0
        flags = np.concatenate(([0], self._quiet[first : last + 1].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(flags))
        if len(edges) == 0:
            return 0
        longest = int((edges[1::2] - edges[::2]).max())
        return (longest - 1) * HOP_MS + WINDOW_MS

    def quietest_ms(self, lo_ms: int, hi_ms: int) -> int:
        """The centre of the quietest 100 ms window inside [lo_ms, hi_ms], or `hi_ms` when none fits."""
        width = QUIETEST_MS // HOP_MS
        first = max(0, -(-lo_ms // HOP_MS))
        last = min((hi_ms - QUIETEST_MS) // HOP_MS, len(self._total) - 1 - width)
        if last < first:
            return hi_ms
        sums = self._total[first + width : last + width + 1] - self._total[first : last + 1]
        return (first + int(np.argmin(sums))) * HOP_MS + QUIETEST_MS // 2
