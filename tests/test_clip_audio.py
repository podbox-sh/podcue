"""Real ffmpeg: decode a span, pick the clip edges on it, cut it, and read the cut back."""

import shutil
import wave

import numpy as np
import pytest

from podcue.recap import RATE, clip_edges, cut_clip, read_span

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH")

TONE_AMPLITUDE = 0.5
TONE_RMS = TONE_AMPLITUDE / np.sqrt(2)
WORDS = [(1.00, 1.50, " four"), (1.50, 2.00, " score"), (3.50, 3.90, " and")]


def tone_wav(path, tone_end):
    samples = np.zeros(4 * RATE, dtype=np.float32)
    lo, hi = RATE, round(tone_end * RATE)
    samples[lo:hi] = TONE_AMPLITUDE * np.sin(2 * np.pi * 220 * np.arange(hi - lo) / RATE)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes((samples * 32767).astype("<i2").tobytes())
    return path


def edges(source):
    span = read_span(source, 0.95, 2.00 + 0.20 + 0.40 + 0.02)
    return clip_edges(WORDS, 0, 1, 0.20, span)


def rms(samples):
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def test_the_cut_keeps_the_sound_after_the_last_word(tmp_path):
    source = tone_wav(tmp_path / "tone.wav", 2.15)
    start, end = edges(source)
    assert start == 0.95
    assert end >= 2.15, f"the cut at {end} drops the tone that runs to 2.15 s"
    clip = tmp_path / "clip.wav"
    cut_clip(source, start, end, clip)
    out = read_span(clip, 0.0, end - start + 1.0)
    tail = out.between(2.15 - start - 0.02, 2.15 - start)
    assert rms(tail) == pytest.approx(TONE_RMS, rel=0.1), "the tone's last 20 ms are in the clip at full level"


def test_the_cut_follows_sound_past_the_tail_pad(tmp_path):
    source = tone_wav(tmp_path / "long.wav", 2.35)
    _, end = edges(source)
    assert 2.35 <= end < 2.35 + 0.05, f"the cut at {end} is not where the tone stops"
