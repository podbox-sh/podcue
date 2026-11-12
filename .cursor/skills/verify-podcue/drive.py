#!/usr/bin/env python3
"""Drive podcue the way the verify-podcue skill describes.

Launch, doctor, then start -> cues -> label-groups -> build, then start --fresh.
JSON, recap.mp3, ffprobe duration, the LUFS reading, and report.md are copied to
the evidence directory before the isolated --root is removed.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "gettysburg-75s.mp3"
EVIDENCE = Path(os.environ.get("VERIFY_PODCUE_EVIDENCE", REPO / "verify-podcue-evidence")).resolve()
LUFS_TARGET = -16.0
LUFS_TOLERANCE = 2.0


def fail(message: str) -> None:
    raise SystemExit(message)


def run_checked(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        fail(f"{' '.join(args)} exited {proc.returncode}: {detail[-1000:]}")
    return proc


def cache_dirs() -> list[Path]:
    home = Path.home()
    huggingface = Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else home / ".cache" / "huggingface"
    xdg = Path(os.environ["XDG_CACHE_HOME"]) if os.environ.get("XDG_CACHE_HOME") else home / ".cache"
    return [huggingface, xdg / "podcue", home / "Library" / "Caches" / "podcue"]


def launch() -> list[str]:
    preset = os.environ.get("PODCUE", "").strip()
    if preset:
        command = shlex.split(preset)
    else:
        if shutil.which("uv") is None:
            fail("uv is not on PATH")
        shutil.rmtree(REPO / "dist", ignore_errors=True)
        run_checked(["uv", "build"], cwd=REPO)
        wheels = sorted((REPO / "dist").glob("podcue-*-py3-none-any.whl"))
        if len(wheels) != 1:
            fail(f"expected one wheel in dist/, found {[path.name for path in wheels]}")
        command = ["uvx", "--python", "3.12", "--from", f"{wheels[0]}[recap]", "podcue"]
    (EVIDENCE / "launch.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    return command


def doctor(podcue: list[str]) -> str:
    missing = [name for name in ("uv", "ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        fail("not on PATH: " + ", ".join(missing))
    version = subprocess.run([*podcue, "--version"], capture_output=True, text=True, check=False)
    if version.returncode != 0 or not version.stdout.strip().startswith("podcue "):
        fail(f"podcue --version failed: {(version.stderr or version.stdout).strip()}")
    lines = [
        f"version: {version.stdout.strip()}",
        f"uv: {shutil.which('uv')}",
        f"ffmpeg: {shutil.which('ffmpeg')}",
        f"ffprobe: {shutil.which('ffprobe')}",
        "caches:",
    ]
    for path in cache_dirs():
        if path.is_dir():
            names = sorted(child.name for child in path.iterdir())[:8]
            lines.append(f"  {path} present ({len(list(path.iterdir()))} entries) {names}")
        else:
            lines.append(f"  {path} absent")
    text = "\n".join(lines) + "\n"
    (EVIDENCE / "doctor.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    return version.stdout.strip()


def no_locks(root: Path) -> None:
    found = sorted(path for path in root.rglob("*") if path.name.endswith(".lock"))
    if found:
        fail("lock files still present: " + ", ".join(str(path) for path in found))


def podcue_json(podcue: list[str], args: list[str], root: Path, name: str) -> dict:
    command = [*podcue, "--root", str(root), *args]
    print("+ " + " ".join(command), flush=True)
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        (EVIDENCE / f"{name}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    stdout = proc.stdout.strip()
    print(stdout, flush=True)
    (EVIDENCE / f"{name}.stdout.txt").write_text(stdout + "\n", encoding="utf-8")
    if proc.returncode != 0:
        fail(f"{name} exit {proc.returncode}: {stdout or '(no stdout)'}")
    if not stdout:
        fail(f"{name} printed no JSON")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        fail(f"{name} stdout was not JSON: {error}")
    if not isinstance(payload, dict) or payload.get("kind") == "error":
        fail(f"{name} error result: {stdout}")
    (EVIDENCE / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    no_locks(root)
    return payload


def duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        fail(f"ffprobe failed for {path.name}: {proc.stderr.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError:
        fail(f"ffprobe returned no duration for {path.name}: {proc.stdout!r}")


def clip_window(cues: list[dict]) -> tuple[float, float]:
    speech = [cue for cue in cues if cue["endMs"] > cue["startMs"] and (cue.get("text") or "").strip()]
    if not speech:
        fail("start produced no cue text")
    start = speech[0]["startMs"] / 1000.0
    end = min(speech[-1]["endMs"] / 1000.0, start + 12.0)
    if end - start < 8.0:
        end = min(speech[-1]["endMs"] / 1000.0, start + 8.0)
    if end - start < 4.0:
        fail(f"could not find a clip of at least 4s (window {start:.2f}-{end:.2f})")
    return start, end


def speaker_group(started: dict) -> str:
    groups = started["episode"]["groups"]
    skipped = {"ungrouped", "uncertain", "mixed"}
    for group in groups:
        if group["groupId"] not in skipped:
            return group["groupId"]
    if not groups:
        fail("start returned no groups")
    return groups[0]["groupId"]


def drive(podcue: list[str], root: Path) -> None:
    source_seconds = duration(FIXTURE)
    if not 60.0 <= source_seconds <= 90.0:
        fail(f"speech clip is {source_seconds:.1f}s, expected 60-90s")
    (EVIDENCE / "source-duration.txt").write_text(f"{source_seconds:.3f}\n", encoding="utf-8")

    started = podcue_json(podcue, ["start", str(FIXTURE), "--language", "en"], root, "start")
    if started.get("kind") != "started" or started.get("created") is not True:
        fail(f"start did not create the episode: {started}")
    episode_id = started["episode"]["episodeId"]
    if started["episode"]["asr"]["engine"] != "faster-whisper":
        fail(f"start did not use faster-whisper: {started['episode']['asr']}")

    cues = podcue_json(podcue, ["cues", episode_id, "--limit", "200"], root, "cues")
    if cues.get("state") != "present" or not cues.get("cues"):
        fail(f"cues were not present: {cues}")
    start_s, end_s = clip_window(cues["cues"])
    group_id = speaker_group(started)

    labeled = podcue_json(podcue, [
        "label-groups", episode_id,
        json.dumps([{"groupId": group_id, "label": "John Greenman"}]),
    ], root, "label-groups")
    if labeled.get("state") != "present":
        fail(f"label-groups did not apply: {labeled}")

    script = root / "script.json"
    output = root / "recap.mp3"
    script.write_text(json.dumps({
        "segments": [
            {"type": "narration", "text": "About 400,000 people know this address."},
            {"type": "clip", "start": round(start_s, 2), "end": round(end_s, 2), "speaker": "John Greenman"},
            {"type": "narration", "text": "The excerpt is 2.4% of a full hour."},
        ],
    }), encoding="utf-8")
    built = podcue_json(podcue, ["build", episode_id, str(script), "--out", str(output)], root, "build")
    if built.get("kind") != "built":
        fail(f"build did not return kind built: {built}")
    if not output.is_file() or output.stat().st_size == 0:
        fail(f"recap.mp3 is missing: {output}")
    recap_seconds = duration(output)
    (EVIDENCE / "ffprobe-duration.txt").write_text(f"{recap_seconds:.3f}\n", encoding="utf-8")
    if not 8.0 <= recap_seconds <= 90.0:
        fail(f"recap.mp3 duration {recap_seconds:.1f}s is not plausible")
    shutil.copy2(output, EVIDENCE / "recap.mp3")
    report = output.parent / "report.md"
    if not report.is_file():
        fail(f"report.md is missing next to {output}")
    shutil.copy2(report, EVIDENCE / "report.md")
    shutil.copy2(script, EVIDENCE / "script.json")

    narrations = [segment for segment in built["segments"] if segment["type"] == "narration"]
    if len(narrations) != 2:
        fail(f"expected two narration segments, got {len(narrations)}")
    lufs_lines = []
    for segment in narrations:
        lufs = float(segment["lufsOut"])
        lufs_lines.append(f"segment {segment['index']}: {lufs:.2f} LUFS")
        if abs(lufs - LUFS_TARGET) > LUFS_TOLERANCE:
            fail(f"narration {segment['index']} is {lufs:.2f} LUFS, not about {LUFS_TARGET:.0f}")
    (EVIDENCE / "lufs.txt").write_text("\n".join(lufs_lines) + "\n", encoding="utf-8")
    clips = [segment for segment in built["segments"] if segment["type"] == "clip"]
    if len(clips) != 1:
        fail(f"expected one clip, got {len(clips)}")

    fresh = podcue_json(
        podcue, ["start", str(FIXTURE), "--language", "en", "--fresh"], root, "start-fresh",
    )
    if fresh.get("kind") != "started" or fresh.get("created") is not True:
        fail(f"start --fresh did not recompute the episode: {fresh}")
    if fresh["episode"]["episodeId"] != episode_id:
        fail("start --fresh wrote a different episode")
    no_locks(root)
    print(
        f"verify-podcue ok: episode {episode_id}, recap {recap_seconds:.1f}s, "
        + ", ".join(lufs_lines),
        flush=True,
    )


def main() -> None:
    if not FIXTURE.is_file():
        fail(f"missing speech clip {FIXTURE}")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    root: Path | None = None
    try:
        podcue = launch()
        doctor(podcue)
        root = Path(tempfile.mkdtemp(prefix="podcue-verify-"))
        (EVIDENCE / "root.txt").write_text(str(root) + "\n", encoding="utf-8")
        drive(podcue, root)
    finally:
        if root is not None and root.exists():
            shutil.rmtree(root)
            print(f"removed root {root}", flush=True)
        print(f"evidence {EVIDENCE}", flush=True)


if __name__ == "__main__":
    main()
