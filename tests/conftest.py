"""A fake engine for command tests: WAV inputs are "converted" by copying, and the models are
replaced by a fixed set of 120 cues in three groups (gc0001, gc0002, ungrouped)."""

import json
import shutil
import wave

import numpy as np
import pytest

from podcue import audio, cli, pipeline
from podcue.cues import Cue
from podcue.diarize import Tracks, Turn

COUNT = 120
CUES = [Cue(f"c{i + 1:04d}", i * 1_000, i * 1_000 + 900, f"line {i + 1}.") for i in range(COUNT)]
# Cue i belongs to speaker i % 3; speaker 2 has no turn, so its cues are ungrouped.
TRACKS = Tracks([Turn(i * 1_000, i * 1_000 + 900, i % 3) for i in range(COUNT) if i % 3 < 2], [[1.0, 0.0], [0.6, 0.8]])


def make_wav(path, seed=0):
    samples = (np.random.default_rng(seed).standard_normal(16_000) * 3_000).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(samples.tobytes())
    return path


def run(capsys, *argv):
    """Run one command and return (parsed stdout line, exit code). Asserts exactly one line."""
    code = cli.main([str(arg) for arg in argv])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1, lines
    return json.loads(lines[0]), code


class Engine:
    def __init__(self):
        self.calls = 0
        self.during = None  # called inside a fake transcribe, after the first progress report
        self.error = None

    def transcribe(self, samples, language, report, transcript_words=None):
        self.calls += 1
        report("diarize", 0)
        if self.during is not None:
            self.during()
        if self.error is not None:
            raise self.error
        report("transcribe", 50)
        return list(CUES), TRACKS


@pytest.fixture
def engine(monkeypatch):
    fake = Engine()
    monkeypatch.setattr(audio, "to_wav", lambda source, target: shutil.copyfile(source, target))
    monkeypatch.setattr(pipeline, "transcribe", fake.transcribe)
    return fake


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("PODCUE_HOME", str(tmp_path / "home"))
    return (tmp_path / "home").resolve()


@pytest.fixture
def episode_id(engine, root, tmp_path, capsys):
    result, code = run(capsys, "start", make_wav(tmp_path / "ep.wav"))
    assert code == 0, result
    return result["episode"]["episodeId"]
