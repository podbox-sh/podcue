"""A few minutes of tones and silence through a real start. CI sets PODCUE_LIVE_EN=1.

Marked slow. The unit job ignores this file. The stability job runs it.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import run

pytestmark = pytest.mark.slow


def _require_live() -> None:
    if os.environ.get("PODCUE_LIVE_EN") != "1":
        pytest.skip("set PODCUE_LIVE_EN=1 to run a long start")
    if shutil.which("ffmpeg") is None:
        pytest.fail("long start needs ffmpeg on PATH")


def _lavfi(path: Path, source: str, seconds: int) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", source, "-t", str(seconds),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )


def _speech(path: Path, seconds: int) -> None:
    source = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "gettysburg-75s.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(source), "-t", str(seconds),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )


def _long_wav(directory: Path) -> Path:
    """About three minutes: speech, noise, and tones, with silent gaps between them."""
    parts = []
    pieces = [
        ("speech", 30),
        ("anullsrc=r=16000:cl=mono", 12),
        ("anoisesrc=color=white:sample_rate=16000:amplitude=0.05", 15),
        ("anullsrc=r=16000:cl=mono", 12),
        ("speech", 30),
        ("anullsrc=r=16000:cl=mono", 12),
        ("sine=frequency=440:sample_rate=16000", 15),
        ("anullsrc=r=16000:cl=mono", 12),
        ("speech", 30),
        ("anullsrc=r=16000:cl=mono", 12),
        ("speech", 30),
        ("anullsrc=r=16000:cl=mono", 12),
    ]
    for index, (source, seconds) in enumerate(pieces):
        part = directory / f"part{index}.wav"
        if source == "speech":
            _speech(part, seconds)
        else:
            _lavfi(part, source, seconds)
        parts.append(part)
    listing = directory / "concat.txt"
    listing.write_text("".join(f"file '{part.resolve().as_posix()}'\n" for part in parts), encoding="utf-8")
    wav = directory / "long.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )
    return wav


def test_a_long_file_with_silent_gaps_starts(tmp_path, monkeypatch, capsys):
    _require_live()
    monkeypatch.setenv("PODCUE_HOME", str(tmp_path / "home"))
    wav = _long_wav(tmp_path)
    started, code = run(capsys, "start", wav, "--language", "en")
    assert code == 0, started
    assert started["kind"] == "started"
    assert started["episode"]["durationMs"] >= 200_000
    assert started["episode"]["asr"]["engine"] == "faster-whisper"
    episode = Path(started["episode"]["dir"])
    assert not (episode / "start.lock").exists()
    assert not (episode / "write.lock").exists()
    assert not (episode / "slices.json").exists()
