"""End to end on a synthesized two-voice clip, through the real CLI and models:
start -> cues -> label-groups -> start-summary -> post-paragraph -> set-summary -> summary -> check.

Needs ffmpeg, a speech synthesizer (`say` on macOS, or espeak-ng) and network access the first
time, for the diarization weights and, with the asr extra, the cactus-compute model. Skips with the reason when a tool is missing.
"""

import json
import shutil
import subprocess
import sys

import pytest

LINES = [
    ("A", "Welcome back to the show. Today we are talking about why cities build so few new homes near jobs."),
    ("B", "Thanks for having me. The short answer is zoning. Most land near the center only allows single family houses."),
    ("A", "So what would you change first, if you could change one rule in the city plan tomorrow?"),
]


def synthesizer():
    # `say` voices are recorded speakers, distinct enough for ECAPA to keep apart.
    # espeak-ng voices share one formant synthesizer and usually cluster as one speaker.
    # Returns (command builder, groups diarization must find).
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
    found = synthesizer()
    missing = [tool for tool in ("ffmpeg",) if shutil.which(tool) is None]
    if found is None:
        missing.append("espeak-ng or say")
    if missing:
        pytest.skip(f"smoke test skipped: {', '.join(missing)} not installed")
    speak, groups = found
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
    return out, groups


def podcue(root, *args, timeout=60):
    proc = subprocess.run([sys.executable, "-m", "podcue", *map(str, args), "--root", str(root)],
                          capture_output=True, text=True, timeout=timeout)
    lines = proc.stdout.splitlines()
    assert len(lines) == 1, (proc.stdout, proc.stderr)
    return json.loads(lines[0]), proc.returncode


def test_whole_flow(clip, tmp_path):
    clip, expected_groups = clip
    root = tmp_path / "root"

    started, code = podcue(root, "start", clip, timeout=900)
    assert code == 0, started
    episode = started["episode"]
    episode_id = episode["episodeId"]
    assert started["created"] is True and episode["language"] == "en"
    assert 15_000 < episode["durationMs"] < 40_000
    voices = [g["groupId"] for g in episode["groups"] if g["groupId"] != "ungrouped"]
    assert len(voices) >= expected_groups, episode["groups"]

    again, code = podcue(root, "start", clip, timeout=120)
    assert code == 0 and again["created"] is False and again["episode"]["episodeId"] == episode_id

    page, code = podcue(root, "cues", episode_id, "--limit", "200")
    assert code == 0
    cues = page["cues"]
    text = " ".join(cue["text"] for cue in cues).lower()
    assert "zoning" in text and "homes" in text, text
    assert [c["startMs"] for c in cues] == sorted(c["startMs"] for c in cues)
    print(f"\nsmoke: {len(cues)} cues, groups {[(g['groupId'], g['speakingMs']) for g in episode['groups']]}")
    for cue in cues:
        print(f"  {cue['id']} {cue['groupId']} ~{cue['nearestGroupId']} {cue['startMs']}-{cue['endMs']} {cue['text']}")

    first_voice, code = podcue(root, "cues", episode_id, "--group-id", voices[0], "--limit", "1")
    assert code == 0 and first_voice["cues"][0]["groupId"] == voices[0]

    names = [{"groupId": group_id, "label": f"Person {n}"} for n, group_id in enumerate(voices, 1)]
    labeled, code = podcue(root, "label-groups", episode_id, json.dumps(names))
    assert code == 0 and labeled["labeledShare"] > 0.9, labeled

    summary, code = podcue(root, "start-summary", episode_id)
    assert code == 0
    summary_id = summary["summaryId"]
    paragraphs = [
        {"text": "The host opens on housing.", "cues": [cues[0]["id"]], "part": "discussion"},
        {"text": "They discuss zoning.", "cues": [cues[-1]["id"], cues[0]["id"]]},
    ]
    posted, code = podcue(root, "post-paragraph", episode_id, summary_id, json.dumps(paragraphs))
    assert code == 0, posted
    assert posted["paragraphs"][1]["cues"] == [cues[0]["id"], cues[-1]["id"]]
    assert (posted["paragraphs"][1]["startMs"], posted["paragraphs"][1]["endMs"]) == (cues[0]["startMs"], cues[-1]["endMs"])

    done, code = podcue(root, "set-summary", episode_id, summary_id, '{"summary": "A host and a guest discuss housing."}')
    assert code == 0, done
    read, code = podcue(root, "summary", episode_id, summary_id)
    assert code == 0 and read == done

    checked, code = podcue(root, "check", episode_id, summary_id)
    assert code == 0 and checked["state"] == "valid", checked

    view, code = podcue(root, "episode", episode_id)
    assert view["summaries"] == [{"summaryId": summary_id}] and view["labeledShare"] == labeled["labeledShare"]


def test_moonshine_japanese(tmp_path):
    pytest.importorskip("moonshine_voice", reason="moonshine smoke test skipped: install the moonshine extra")
    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        pytest.skip("moonshine smoke test skipped: needs macOS `say` with the Kyoko voice, and ffmpeg")
    clip = tmp_path / "ja.aiff"
    spoken = subprocess.run(["say", "-v", "Kyoko", "-o", str(clip), "こんにちは。今日はとても良い天気ですね。散歩に行きましょう。"],
                            capture_output=True, text=True)
    if spoken.returncode != 0:
        pytest.skip(f"moonshine smoke test skipped: `say -v Kyoko` failed ({spoken.stderr.strip()})")
    started, code = podcue(tmp_path / "root", "start", clip, "--language", "ja", timeout=900)
    assert code == 0, started
    page, _ = podcue(tmp_path / "root", "cues", started["episode"]["episodeId"])
    assert started["episode"]["language"] == "ja" and page["cues"], page
    print("\nmoonshine:", [cue["text"] for cue in page["cues"]])
