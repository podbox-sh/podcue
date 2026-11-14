"""The episode folder. Every piece of state lives in a file here; there is no other process.

Writes go through a temp file plus rename, so a reader never sees half a file. `start.lock`
holds the pid of the running start under an flock and is removed when that command finishes,
including when it fails. A crash can leave the file. The next start takes it when the
flock is free, or when `lock_is_stale` says the holder is gone. Read-modify-write
commands hold `write.lock` for one call and remove it the same way.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

import yaml

EPISODE_ID = re.compile(r"^ep_[0-9a-f]{16}$")
SUMMARY_ID = re.compile(r"^sm_[0-9a-f]{16}$")
INTERRUPTED = "start was interrupted; run start again"
LOCK_HELD_LONGER_THAN_A_LIVE_START_S = 12 * 60 * 60
EMPTY_LOCK_BEFORE_PID_S = 30


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
        self.labels_prev = self.dir / "labels.prev.json"
        self.status = self.dir / "status.json"
        self.failed = self.dir / "failed.json"
        self.start_lock = self.dir / "start.lock"
        self.write_lock = self.dir / "write.lock"
        self.slices = self.dir / "slices.json"
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


def _note(message: str) -> None:
    print(f"podcue: {message}", file=sys.stderr, flush=True)


def _write_pid(fd: int) -> None:
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())


def _lock_nb(path: Path) -> int | None:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def lock_is_stale(path: Path) -> str | None:
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    age = time.time() - info.st_mtime
    pid = lock_pid(path)
    if info.st_size == 0 and age >= EMPTY_LOCK_BEFORE_PID_S:
        return f"empty for {int(age)}s"
    if pid is not None and not pid_alive(pid):
        return f"pid {pid} is not running"
    if age >= LOCK_HELD_LONGER_THAN_A_LIVE_START_S:
        return f"older than {LOCK_HELD_LONGER_THAN_A_LIVE_START_S // 3600} hours"
    return None


def _take_over(path: Path, reason: str) -> int | None:
    _note(f"taking over stale {path.name} ({reason})")
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    fd = _lock_nb(path)
    if fd is None:
        return None
    _write_pid(fd)
    return fd


def try_lock(path: Path) -> int | None:
    """Take the start lock and write our pid into it, or return None while another process holds it."""
    fd = _lock_nb(path)
    if fd is not None:
        _write_pid(fd)
        return fd
    reason = lock_is_stale(path)
    if reason is None:
        return None
    return _take_over(path, reason)


def release(fd: int, path: Path) -> None:
    """Drop a lock this process holds and remove its file."""
    _remove_lock(fd, path)


def _remove_lock(fd: int, path: Path) -> None:
    """Unlink `path` while `fd` still holds it, then close `fd`.

    The inode check leaves a newer lock file alone if one replaced this one.
    """
    try:
        held = os.fstat(fd)
        current = os.stat(path)
    except OSError:
        os.close(fd)
        return
    try:
        if held.st_ino == current.st_ino and held.st_dev == current.st_dev:
            os.unlink(path)
    except FileNotFoundError:
        pass
    finally:
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
    """Serialize read-modify-write calls on one episode.

    Wait by trying again. A blocking flock on the old inode stays held after that
    file is unlinked, and the next writer can lock a new file at the same time.
    """
    while True:
        fd = try_lock(episode.write_lock)
        if fd is not None:
            break
        time.sleep(0.05)
    try:
        yield
    finally:
        _remove_lock(fd, episode.write_lock)


def summary_ids(episode: Episode) -> list[str]:
    if not episode.summaries.is_dir():
        return []
    ids = (path.stem for path in episode.summaries.glob("sm_*.yaml"))
    return sorted(i for i in ids if SUMMARY_ID.match(i))


def load_rows(episode: Episode) -> list[dict]:
    return read_json(episode.cues)


def retire_labels(episode: Episode) -> None:
    """Move labels.json onto labels.prev.json so a fresh start labels from scratch.

    `os.replace` overwrites an older prev in one step. Clip and narration files in
    recap-cache are keyed on the source audio and the text, not on these labels, so
    they stay; `build` reads labels at run time and will not reuse the retired names.
    """
    if not episode.labels.is_file():
        return
    os.replace(episode.labels, episode.labels_prev)


def load_labels(episode: Episode) -> list[dict]:
    stored = read_json(episode.labels)
    return [] if stored is None else stored["groups"]


def load_cue_labels(episode: Episode) -> list[dict]:
    stored = read_json(episode.labels)
    if not isinstance(stored, dict):
        return []
    cues = stored.get("cues") or []
    return [item for item in cues if isinstance(item, dict) and "id" in item and "label" in item]


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
