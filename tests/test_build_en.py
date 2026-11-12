"""A real `podcue build`: Kokoro narration plus one clip, then a file ffprobe can play.

CI sets PODCUE_LIVE_EN=1 and installs ffmpeg. Model files may be cached; the build still runs.
"""

import json
import math
import os
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from podcue.cli import main as cli_main

EPISODE_ID = "ep_" + "0123456789abcdef"


def _require_live() -> None:
    if os.environ.get("PODCUE_LIVE_EN") != "1":
        pytest.skip("set PODCUE_LIVE_EN=1 to run a real build")
    missing = [tool for tool in ("ffmpeg", "ffprobe", "espeak-ng") if shutil.which(tool) is None]
    if missing:
        pytest.fail(f"live build needs {', '.join(missing)} on PATH")


def _tone(path: Path, seconds: float = 5.0) -> None:
    """A tone, not silence. Loudness measurement of a silent clip is non-finite."""
    rate = 16_000
    frames = int(rate * seconds)
    samples = bytearray()
    for i in range(frames):
        value = int(8000 * math.sin(2 * math.pi * 220 * i / rate))
        samples += int(value).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples)


def _episode(root: Path, source: Path) -> None:
    folder = root / "episodes" / EPISODE_ID
    folder.mkdir(parents=True)
    (folder / "episode.json").write_text(json.dumps({
        "episodeId": EPISODE_ID,
        "source": str(source.resolve()),
        "durationMs": 5_000,
        "language": "en",
        "createdAt": "2026-11-12T00:00:00Z",
        "asr": {"engine": "external", "model": None, "revision": None},
    }), encoding="utf-8")
    (folder / "cues.json").write_text(json.dumps([{
        "id": "c0001",
        "startMs": 0,
        "endMs": 5_000,
        "text": "hello from the clip",
        "groupId": "gc0001",
        "speaker": "gc0001",
    }]), encoding="utf-8")


def _torch_warning_lines(stderr: str) -> list[str]:
    hits = []
    for line in stderr.splitlines():
        lowered = line.lower()
        if "/torch/" in line or "weight_norm" in line or "autocast" in lowered:
            hits.append(line)
        elif "warning" in lowered and "torch" in lowered:
            hits.append(line)
    return hits


def test_build_narrates_with_kokoro_and_writes_a_playable_file(tmp_path, monkeypatch, capsys):
    _require_live()
    root = tmp_path / "home"
    monkeypatch.setenv("PODCUE_HOME", str(root))
    source = tmp_path / "source.wav"
    _tone(source)
    _episode(root, source)
    script = tmp_path / "script.json"
    script.write_text(json.dumps({
        "segments": [
            {"type": "narration", "text": "In 1996 the city added 400,000 homes."},
            {"type": "clip", "start": "0", "end": "5", "speaker": "the host"},
        ],
    }), encoding="utf-8")
    output = tmp_path / "recap.mp3"
    code = cli_main(["build", EPISODE_ID, str(script), "--out", str(output)])
    captured = capsys.readouterr()
    result = json.loads(captured.out.strip())
    assert code == 0, captured.err
    assert result["kind"] == "built"
    assert result["segments"][0]["type"] == "narration"
    assert result["segments"][1]["type"] == "clip"
    assert output.is_file() and output.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert float(probe.stdout.strip()) > 1.0
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    assert decoded.returncode == 0, decoded.stderr
    folder = root / "episodes" / EPISODE_ID
    assert not (folder / "start.lock").exists()
    assert not (folder / "write.lock").exists()
    assert _torch_warning_lines(captured.err) == []
    narration = list((folder / "recap-cache").glob("tts_*.wav"))
    assert len(narration) == 1 and narration[0].read_bytes()[:4] == b"RIFF"
