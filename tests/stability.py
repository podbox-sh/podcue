"""Full-pipeline stability test against an installed podcue, not the source tree.

The clip is a 75-second public-domain excerpt of the LibriVox Gettysburg Address
read by John Greenman (Internet Archive gettysburg_johng_librivox). The test
serves that file and runs fetch on the local URL, then start with faster-whisper
small.en, cues, label-groups, build (Kokoro narration plus one clip), and
start --fresh.

    PODCUE="uvx --python 3.12 --from dist/podcue-0.2.6-py3-none-any.whl[recap] podcue" \
      python3 tests/stability.py
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "gettysburg-75s.mp3"
LUFS_TARGET = -16.0
LUFS_TOLERANCE = 2.0


class _QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FIXTURE.parent), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return None


def _fail(message: str) -> None:
    raise SystemExit(message)


def _command() -> list[str]:
    raw = os.environ.get("PODCUE", "").strip()
    if not raw:
        _fail(
            "Set PODCUE to the installed command, for example "
            "uvx --python 3.12 --from 'dist/podcue-0.2.6-py3-none-any.whl[recap]' podcue"
        )
    return shlex.split(raw)


def _no_locks(home: Path) -> None:
    found = sorted(path for path in home.rglob("*") if path.name.endswith(".lock"))
    if found:
        _fail("lock files still present: " + ", ".join(str(path) for path in found))


def _run(podcue: list[str], args: list[str], home: Path) -> dict:
    printable = " ".join(podcue + ["--root", str(home), *args])
    print(f"+ {printable}", flush=True)
    proc = subprocess.run(
        [*podcue, "--root", str(home), *args],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        check=False,
    )
    stdout = proc.stdout.strip()
    print(stdout, flush=True)
    if proc.returncode != 0:
        _fail(f"exit {proc.returncode}: {stdout or '(no stdout)'}")
    if not stdout:
        _fail("command printed no JSON")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        _fail(f"stdout was not one JSON object: {error}")
    if not isinstance(payload, dict):
        _fail("stdout JSON was not an object")
    if payload.get("kind") == "error" or '"kind": "error"' in stdout or '"kind":"error"' in stdout:
        _fail(f"error result: {stdout}")
    _no_locks(home)
    return payload


def _serve() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _QuietHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}/{FIXTURE.name}"


def _duration(path: Path) -> float:
    if shutil.which("ffprobe") is None:
        _fail("ffprobe is not on PATH")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        _fail(f"ffprobe failed: {proc.stderr.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError:
        _fail(f"ffprobe returned no duration: {proc.stdout!r}")


def _clip_window(cues: list[dict]) -> tuple[float, float]:
    speech = [cue for cue in cues if cue["endMs"] > cue["startMs"] and (cue.get("text") or "").strip()]
    if not speech:
        _fail("start produced no cue text")
    start = speech[0]["startMs"] / 1000.0
    end = min(speech[-1]["endMs"] / 1000.0, start + 12.0)
    if end - start < 8.0:
        end = min(speech[-1]["endMs"] / 1000.0, start + 8.0)
    if end - start < 4.0:
        _fail(f"could not find a clip of at least 4s (window {start:.2f}-{end:.2f})")
    return start, end


def _speaker_group(started: dict) -> str:
    groups = started["episode"]["groups"]
    skipped = {"ungrouped", "uncertain", "mixed"}
    for group in groups:
        if group["groupId"] not in skipped:
            return group["groupId"]
    if not groups:
        _fail("start returned no groups")
    return groups[0]["groupId"]


def main() -> None:
    if not FIXTURE.is_file():
        _fail(f"missing speech clip {FIXTURE}")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            _fail(f"{tool} is not on PATH")
    podcue = _command()
    home = Path(tempfile.mkdtemp(prefix="podcue-stability-"))
    os.environ["PODCUE_HOME"] = str(home)
    server, url = _serve()
    try:
        fetched = _run(podcue, ["fetch", url, "--out", str(home / "fetch")], home)
        if fetched.get("kind") != "fetched":
            _fail(f"fetch did not return kind fetched: {fetched}")
        source = Path(fetched["path"])
        seconds = _duration(source)
        if not 60.0 <= seconds <= 90.0:
            _fail(f"fetched clip is {seconds:.1f}s, expected 60-90s")

        started = _run(podcue, ["start", str(source), "--language", "en"], home)
        if started.get("kind") != "started" or started.get("created") is not True:
            _fail(f"start did not create the episode: {started}")
        episode_id = started["episode"]["episodeId"]

        cues = _run(podcue, ["cues", episode_id, "--limit", "200"], home)
        if cues.get("state") != "present" or not cues.get("cues"):
            _fail(f"cues were not present: {cues}")
        start_s, end_s = _clip_window(cues["cues"])
        group_id = _speaker_group(started)
        labeled = _run(podcue, [
            "label-groups", episode_id,
            json.dumps([{"groupId": group_id, "label": "John Greenman"}]),
        ], home)
        if labeled.get("state") != "present":
            _fail(f"label-groups did not apply: {labeled}")

        script = home / "script.json"
        output = home / "recap.mp3"
        script.write_text(json.dumps({
            "segments": [
                {"type": "narration", "text": "About 400,000 people know this address."},
                {"type": "clip", "start": round(start_s, 2), "end": round(end_s, 2), "speaker": "John Greenman"},
                {"type": "narration", "text": "The excerpt is 2.4% of a full hour."},
            ],
        }), encoding="utf-8")
        built = _run(podcue, ["build", episode_id, str(script), "--out", str(output)], home)
        if built.get("kind") != "built":
            _fail(f"build did not return kind built: {built}")
        if not output.is_file() or output.stat().st_size == 0:
            _fail(f"recap.mp3 is missing: {output}")
        duration = _duration(output)
        if not 8.0 <= duration <= 90.0:
            _fail(f"recap.mp3 duration {duration:.1f}s is not plausible")
        narrations = [segment for segment in built["segments"] if segment["type"] == "narration"]
        if len(narrations) != 2:
            _fail(f"expected two narration segments, got {len(narrations)}")
        for segment in narrations:
            lufs = float(segment["lufsOut"])
            if abs(lufs - LUFS_TARGET) > LUFS_TOLERANCE:
                _fail(f"narration {segment['index']} is {lufs:.2f} LUFS, not about {LUFS_TARGET:.0f}")
        clips = [segment for segment in built["segments"] if segment["type"] == "clip"]
        if len(clips) != 1:
            _fail(f"expected one clip, got {len(clips)}")
        fresh = _run(podcue, ["start", str(source), "--language", "en", "--fresh"], home)
        if fresh.get("kind") != "started" or fresh.get("created") is not True:
            _fail(f"start --fresh did not recompute the episode: {fresh}")
        if fresh["episode"]["episodeId"] != episode_id:
            _fail("start --fresh wrote a different episode")
        _no_locks(home)
        print(
            f"stability ok: episode {episode_id}, recap {duration:.1f}s, "
            f"narration LUFS {[round(float(segment['lufsOut']), 2) for segment in narrations]}",
            flush=True,
        )
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
