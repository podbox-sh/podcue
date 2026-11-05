"""End to end: synthesize two voices, run the real pipeline, then check a summary against it.

Needs ffmpeg, a speech synthesizer (espeak-ng, or `say` on macOS) and network access the
first time, for the diarization and cactus-needle weights. Skips with the reason otherwise.
"""

import json
import shutil
import subprocess
import sys
import urllib.error

import pytest

from podcue import weights

LINES = [
    ("A", "Welcome back to the show. Today we are talking about why cities build so few new homes near jobs."),
    ("B", "Thanks for having me. The short answer is zoning. Most land near the center only allows single family houses."),
    ("A", "So what would you change first, if you could change one rule in the city plan tomorrow?"),
]


def synthesizer():
    # `say` voices are recorded speakers, distinct enough for ECAPA to keep apart.
    # espeak-ng voices share one formant synthesizer and usually cluster as one speaker.
    # Returns (command builder, speakers diarization must find).
    if shutil.which("say"):
        voices = {"A": "Daniel", "B": "Samantha"}
        return (lambda voice, text, path: ["say", "-v", voices[voice], "-o", str(path),
                                           "--data-format=LEI16@16000", text]), 2
    if shutil.which("espeak-ng"):
        voices = {"A": ["-v", "en-us", "-p", "30"], "B": ["-v", "en-gb+f4", "-p", "75"]}
        return (lambda voice, text, path: ["espeak-ng", *voices[voice], "-w", str(path), text]), 1
    return None


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    missing = [tool for tool in ("ffmpeg",) if shutil.which(tool) is None]
    found = synthesizer()
    if found is None:
        missing.append("espeak-ng or say")
    if missing:
        pytest.skip(f"smoke test skipped: {', '.join(missing)} not installed")
    try:
        weights.ensure(weights.ECAPA)
        weights.ensure(weights.SEGMENTATION)
    except (urllib.error.URLError, OSError) as error:
        pytest.skip(f"smoke test skipped: could not download diarization weights ({error})")

    speak, speakers = found
    directory = tmp_path_factory.mktemp("smoke")
    parts = []
    for index, (voice, text) in enumerate(LINES):
        part = directory / f"part{index}.wav"
        subprocess.run(speak(voice, text, part), check=True, capture_output=True)
        parts.append(part)
    inputs = [arg for part in parts for arg in ("-i", str(part))]
    streams = "".join(f"[{i}:a]aresample=16000,aformat=channel_layouts=mono[a{i}];" for i in range(len(parts)))
    joined = "".join(f"[a{i}]" for i in range(len(parts)))
    out = directory / "two voices.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex",
         f"{streams}{joined}concat=n={len(parts)}:v=0:a=1", str(out)],
        check=True,
    )
    return out, speakers


def test_pipeline_end_to_end(clip, tmp_path):
    clip, expected_speakers = clip
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "podcue", str(clip), "-o", str(out_dir)],
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr
    json_path, vtt_path = proc.stdout.split("\n")[:2]
    assert json_path == str(out_dir / "two voices.cues.json")
    assert vtt_path == str(out_dir / "two voices.vtt")

    doc = json.loads((out_dir / "two voices.cues.json").read_text())
    assert doc["language"] == "en" and doc["source"] == str(clip)
    assert 15_000 < doc["durationMs"] < 40_000
    assert len(doc["speakers"]) >= expected_speakers, doc["speakers"]
    ids = {speaker["id"] for speaker in doc["speakers"]}
    assert all(cue["speaker"] in ids | {None} for cue in doc["cues"])
    text = " ".join(cue["text"] for cue in doc["cues"]).lower()
    assert "zoning" in text and "homes" in text, text
    starts = [cue["startMs"] for cue in doc["cues"]]
    assert starts == sorted(starts)

    vtt = (out_dir / "two voices.vtt").read_text()
    assert vtt.startswith("WEBVTT\n\nc0001\n") and "<v S1>" in vtt
    print(f"\nsmoke: {len(doc['cues'])} cues, speakers {[(s['id'], s['speakingMs']) for s in doc['speakers']]}")
    for cue in doc["cues"]:
        print(f"  {cue['id']} {cue['speaker']} {cue['startMs']}-{cue['endMs']} {cue['text']}")

    first, last = doc["cues"][0], doc["cues"][-1]
    summary = {
        "summary": "A host and a guest discuss housing.",
        "speakers": {speaker["id"]: f"Person {speaker['id']}" for speaker in doc["speakers"]},
        "paragraphs": [{
            "text": "They discuss zoning.",
            "cues": [first["id"], last["id"]],
            "startMs": first["startMs"],
            "endMs": last["endMs"],
        }],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    check = subprocess.run(
        [sys.executable, "-m", "podcue", "check", str(summary_path), "--cues", str(out_dir / "two voices.cues.json")],
        capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stderr


def test_moonshine_japanese(tmp_path):
    pytest.importorskip("moonshine_voice", reason="moonshine smoke test skipped: install the moonshine extra")
    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        pytest.skip("moonshine smoke test skipped: needs macOS `say` with the Kyoko voice, and ffmpeg")
    clip = tmp_path / "ja.aiff"
    spoken = subprocess.run(["say", "-v", "Kyoko", "-o", str(clip), "こんにちは。今日はとても良い天気ですね。散歩に行きましょう。"],
                            capture_output=True, text=True)
    if spoken.returncode != 0:
        pytest.skip(f"moonshine smoke test skipped: `say -v Kyoko` failed ({spoken.stderr.strip()})")
    proc = subprocess.run(
        [sys.executable, "-m", "podcue", str(clip), "-o", str(tmp_path), "--language", "ja"],
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads((tmp_path / "ja.cues.json").read_text())
    assert doc["language"] == "ja" and doc["cues"], doc
    print("\nmoonshine:", [cue["text"] for cue in doc["cues"]])
