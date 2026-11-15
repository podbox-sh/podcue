"""Quiet stretches in 16 kHz samples, measured from 5 ms block energies.

A long speech region is cut at its quietest point so a slice edge falls between words.
"""

from __future__ import annotations

import numpy as np

from .audio import RATE

HOP_MS = 5
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

    def quietest_ms(self, lo_ms: int, hi_ms: int) -> int:
        """The centre of the quietest 100 ms window inside [lo_ms, hi_ms], or `hi_ms` when none fits."""
        width = QUIETEST_MS // HOP_MS
        first = max(0, -(-lo_ms // HOP_MS))
        last = min((hi_ms - QUIETEST_MS) // HOP_MS, len(self._total) - 1 - width)
        if last < first:
            return hi_ms
        sums = self._total[first + width : last + width + 1] - self._total[first : last + 1]
        return (first + int(np.argmin(sums))) * HOP_MS + QUIETEST_MS // 2
