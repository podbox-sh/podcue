"""The episode folder. Every piece of state lives in a file here; there is no other process.

Writes go through a temp file plus rename, so a reader never sees half a file. `start.lock`
holds the pid of the running start under an flock, so the OS releases it when that process
dies and a later start replaces it. Read-modify-write commands (labels, summaries) hold
`write.lock` for the length of one call.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import secrets
from pathlib import Path

import yaml

EPISODE_ID = re.compile(r"^ep_[0-9a-f]{16}$")
SUMMARY_ID = re.compile(r"^sm_[0-9a-f]{16}$")
INTERRUPTED = "start was interrupted; run start again"


def resolve_root(flag: str | None) -> Path:
    """`--root`, else `$PODCUE_HOME`, else `~/podcue`."""
    chosen = flag or os.environ.get("PODCUE_HOME") or "~/podcue"
    return Path(chosen).expanduser().resolve()


class Episode:
    def __init__(self, root: Path, episode_id: str):
        self.id = episode_id
        self.dir = root / "episodes" / episode_id
        self.manifest = self.dir / "episode.json"
        self.transcript = self.dir / "transcript.vtt"
        self.cues = self.dir / "cues.json"
        self.groups = self.dir / "groups.json"
        self.labels = self.dir / "labels.json"
        self.status = self.dir / "status.json"
        self.failed = self.dir / "failed.json"
        self.start_lock = self.dir / "start.lock"
        self.write_lock = self.dir / "write.lock"
        self.summaries = self.dir / "summaries"

    def summary_file(self, summary_id: str) -> Path:
        return self.summaries / f"{summary_id}.yaml"

    def known(self) -> bool:
        return EPISODE_ID.match(self.id) is not None and self.manifest.is_file()

    def ready(self) -> bool:
        return self.cues.is_file()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def try_lock(path: Path) -> int | None:
    """Take the start lock and write our pid into it, or return None while another process holds it."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def release(fd: int) -> None:
    os.ftruncate(fd, 0)
    os.close(fd)


def lock_pid(path: Path) -> int | None:
    """The pid written in a lock file, or None when the file is missing or empty."""
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return None
    return int(text) if text.isdigit() else None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def writing(episode: Episode):
    """Serialize read-modify-write calls on one episode."""
    fd = os.open(episode.write_lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def summary_ids(episode: Episode) -> list[str]:
    if not episode.summaries.is_dir():
        return []
    ids = (path.stem for path in episode.summaries.glob("sm_*.yaml"))
    return sorted(i for i in ids if SUMMARY_ID.match(i))


def load_rows(episode: Episode) -> list[dict]:
    return read_json(episode.cues)


def load_labels(episode: Episode) -> list[dict]:
    stored = read_json(episode.labels)
    return [] if stored is None else stored["groups"]


def dump_yaml(value: object) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=1 << 30)


def start_state(episode: Episode) -> dict:
    """`ready`, `running` (with stage and progress), or `failed` (with message)."""
    if episode.ready():
        return {"state": "ready", "progress": 100}
    status = read_json(episode.status)
    if status is not None and pid_alive(status["pid"]) and lock_pid(episode.start_lock) == status["pid"]:
        return {"state": "running", "stage": status["stage"], "progress": status["progress"]}
    failed = read_json(episode.failed)
    return {"state": "failed", "message": INTERRUPTED if failed is None else failed["message"]}
