"""The lock scripts from the review of PR 13, plus the inherited-fd case."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent / "lock_repro"


def _run(name: str, timeout: int, *args: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPTS / name), *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return proc.stdout


@pytest.mark.slow
def test_a_live_lock_is_not_stolen_after_thirteen_hours():
    lines = _run("live_lock_stolen.py", 30).strip().splitlines()
    assert lines[-1] == "ok"
    assert "second start after 13 h of mtime age: None" in lines[-2]


@pytest.mark.slow
def test_eight_starts_cannot_share_one_held_lock():
    lines = _run("double_takeover.py", 180).strip().splitlines()
    assert lines[-1] == "ok"
    assert len(lines) == 21
    assert all(line.endswith("0 of 8 starts took the stale lock") for line in lines[:-1])


@pytest.mark.slow
def test_twelve_writers_do_not_overlap():
    assert _run("writers_overlap.py", 120).strip() == "0 times a writer entered while another was inside"


@pytest.mark.slow
def test_a_held_lock_names_the_write_lock_inode():
    lines = _run("inode_check.py", 120).strip().splitlines()
    assert lines[-1] == "ok"
    assert lines[-2].startswith("0 ")


def test_a_forked_child_that_inherited_the_lock_fd_is_a_live_holder(tmp_path):
    out = _run("fork_inherited.py", 15, str(tmp_path / "start.lock"))
    assert "while child lives none" in out
    assert "after child exits taken" in out
