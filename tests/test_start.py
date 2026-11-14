import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
import warnings

import pytest
from conftest import make_wav, run

from podcue import audio, pipeline, transcribe, weights
from podcue.cli import execute, main as cli_main
from podcue.store import Episode, release, try_lock, writing


def expected_id(path):
    return "ep_" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_start_writes_the_folder_and_is_idempotent(engine, root, tmp_path, capsys):
    wav = make_wav(tmp_path / "ep.wav")
    first, code = run(capsys, "start", wav)
    assert code == 0 and first["kind"] == "started" and first["created"] is True
    view = first["episode"]
    assert view["episodeId"] == expected_id(wav)
    assert view["dir"] == str(root / "episodes" / view["episodeId"])
    assert (view["source"], view["durationMs"], view["language"]) == (str(wav.resolve()), 1_000, "en")
    from podcue.transcribe import WHISPER_EN
    assert view["asr"] == {
        "engine": "faster-whisper", "model": WHISPER_EN.repo, "revision": WHISPER_EN.revision,
    }
    assert [(a["artifact"], a["state"], a["progress"]) for a in view["artifacts"]] == [
        ("transcript", "ready", 100), ("cues", "ready", 100)]
    assert [g["groupId"] for g in view["groups"]] == ["gc0001", "gc0002", "ungrouped"]
    assert view["groups"][0] == {"groupId": "gc0001", "speakingMs": 36_000, "cueCount": 40,
                                 "nearestGroupId": "gc0002", "nearestCosine": 0.6, "label": None}
    assert view["labeledShare"] == 0.0 and view["undeterminedShare"] == 0.0 and view["summaries"] == []

    folder = root / "episodes" / view["episodeId"]
    assert {p.name for p in folder.iterdir()} == {
        "episode.json", "transcript.vtt", "cues.json", "groups.json"}
    assert not (folder / "start.lock").exists()
    assert not (folder / "write.lock").exists()
    assert "<v gc0001>line 1." in (folder / "transcript.vtt").read_text()

    again, code = run(capsys, "start", wav, "--language", "de")
    assert code == 0 and again["created"] is False
    assert again["episode"]["episodeId"] == view["episodeId"] and again["episode"]["language"] == "en"
    assert again["episode"]["createdAt"] == view["createdAt"]
    assert engine.calls == 1


def test_a_moved_file_refreshes_the_stored_source(engine, root, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    wav = make_wav(tmp_path / "src" / "audio.wav")
    first, code = run(capsys, "start", "src/audio.wav")
    assert code == 0 and first["episode"]["source"] == str(wav.resolve())
    moved = tmp_path / "later" / "audio.wav"
    moved.parent.mkdir()
    wav.replace(moved)
    again, code = run(capsys, "start", "later/audio.wav")
    assert code == 0 and again["created"] is False
    assert again["episode"]["source"] == str(moved.resolve())
    assert again["episode"]["createdAt"] == first["episode"]["createdAt"]
    assert engine.calls == 1


def test_fresh_recomputes_one_episode_and_keeps_the_rest(engine, root, tmp_path, monkeypatch, capsys):
    from podcue.store import Episode
    from podcue.weights import cache_dir

    marker = cache_dir() / "weights" / "keep.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("keep")
    first_wav = make_wav(tmp_path / "a.wav", seed=1)
    other_wav = make_wav(tmp_path / "b.wav", seed=2)
    first, _ = run(capsys, "start", first_wav)
    other, _ = run(capsys, "start", other_wav)
    episode = Episode(root, first["episode"]["episodeId"])
    other_episode = Episode(root, other["episode"]["episodeId"])
    other_episode.labels.write_text('{"groups": [{"groupId": "gc0001", "label": "Other"}]}')
    (episode.dir / "labels.prev.json").write_text("OLDER")
    labeled, code = run(capsys, "label-groups", first["episode"]["episodeId"],
                        '[{"groupId": "gc0001", "label": "Ada"}]')
    assert code == 0 and labeled["labeledShare"] > 0
    cache = episode.dir / "recap-cache"
    cache.mkdir()
    (cache / "tts_keep.wav").write_bytes(b"RIFFkeep")
    page, _ = run(capsys, "cues", first["episode"]["episodeId"], "--limit", "1")
    assert page["cues"][0]["speaker"] == "Ada"
    other_cues = (other_episode.dir / "cues.json").read_bytes()
    monkeypatch.setattr(pipeline, "now", lambda: "2026-11-12T05:31:00Z")
    fresh, code = run(capsys, "start", first_wav, "--fresh")
    assert code == 0 and fresh["created"] is True
    assert fresh["episode"]["createdAt"] == "2026-11-12T05:31:00Z"
    assert fresh["episode"]["source"] == str(first_wav.resolve())
    assert fresh["episode"]["labeledShare"] == 0.0
    assert fresh["episode"]["groups"][0]["label"] is None
    assert engine.calls == 3
    assert not episode.labels.exists()
    assert "Ada" in episode.labels_prev.read_text()
    assert "OLDER" not in episode.labels_prev.read_text()
    assert (cache / "tts_keep.wav").read_bytes() == b"RIFFkeep"
    page, _ = run(capsys, "cues", first["episode"]["episodeId"], "--limit", "1")
    assert page["cues"][0]["speaker"] == page["cues"][0]["groupId"]
    assert other_episode.labels.read_text() == '{"groups": [{"groupId": "gc0001", "label": "Other"}]}'
    assert (other_episode.dir / "cues.json").read_bytes() == other_cues
    assert marker.read_text() == "keep"
    assert episode.cues.is_file() and episode.transcript.is_file()

    episode.labels.write_text('{"groups": [{"groupId": "gc0002", "label": "Ben"}]}')
    again, code = run(capsys, "start", first_wav, "--fresh")
    assert code == 0 and again["created"] is True
    assert not episode.labels.exists()
    assert "Ben" in episode.labels_prev.read_text()
    assert "Ada" not in episode.labels_prev.read_text()


def test_root_flag_beats_podcue_home(engine, root, tmp_path, capsys):
    other = tmp_path / "other"
    result, _ = run(capsys, "start", make_wav(tmp_path / "ep.wav"), "--root", other)
    assert result["episode"]["dir"].startswith(str(other.resolve()))
    assert not root.exists()
    view, _ = run(capsys, "--root", other, "episode", result["episode"]["episodeId"])
    assert view["state"] == "present"


def test_a_running_start_refuses_a_second_and_episode_reports_progress(engine, root, tmp_path, capsys):
    wav = make_wav(tmp_path / "ep.wav")
    seen = {}

    def during():
        seen["episode"] = execute(["episode", expected_id(wav)])[0]
        seen["second"] = execute(["start", str(wav)])

    engine.during = during
    run(capsys, "start", wav)
    assert seen["episode"]["artifacts"][0] == {"artifact": "transcript", "state": "running", "stage": "diarize", "progress": 0}
    second, code = seen["second"]
    assert code == 1 and second["kind"] == "error" and second["reason"] == "running"


def test_a_held_lock_refuses_start_before_any_model_runs(engine, root, tmp_path, capsys):
    wav = make_wav(tmp_path / "ep.wav")
    folder = Episode(root, expected_id(wav)).dir
    folder.mkdir(parents=True)
    fd = try_lock(folder / "start.lock")
    try:
        result, code = run(capsys, "start", wav)
    finally:
        os.close(fd)
    assert code == 1 and result["reason"] == "running"
    assert "is running" in result["message"] and engine.calls == 0


def test_a_lock_left_by_a_dead_process_is_stale_and_replaced(engine, root, tmp_path, capsys, monkeypatch):
    wav = make_wav(tmp_path / "ep.wav")
    episode = Episode(root, expected_id(wav))
    episode.dir.mkdir(parents=True)
    pid = dead_pid()
    episode.start_lock.write_text(f"{pid}\n")
    episode.status.write_text(json.dumps({"pid": pid, "stage": "transcribe", "progress": 40}))
    episode.manifest.write_text(json.dumps({"episodeId": episode.id, "source": str(wav), "durationMs": 1_000,
                                            "language": "en", "createdAt": "2026-11-01T00:00:00Z"}))

    view, _ = run(capsys, "episode", episode.id)
    assert view["artifacts"][0] == {"artifact": "transcript", "state": "failed",
                                    "message": "start was interrupted; run start again"}

    monkeypatch.setattr(pipeline, "now", lambda: "2026-11-11T23:53:00Z")
    result, code = run(capsys, "start", wav)
    assert code == 0 and result["created"] is True
    assert result["episode"]["createdAt"] == "2026-11-11T23:53:00Z"
    assert not episode.status.exists() and not episode.start_lock.exists()


def test_a_failed_start_writes_failed_json_and_a_later_start_retries(engine, root, tmp_path, capsys, monkeypatch):
    wav = make_wav(tmp_path / "ep.wav")
    monkeypatch.setattr(pipeline, "now", lambda: "2026-11-11T19:22:00Z")
    engine.error = RuntimeError("model exploded")
    result, code = run(capsys, "start", wav)
    assert code == 1 and result == {"kind": "error", "message": "model exploded"}
    episode = Episode(root, expected_id(wav))
    failed = json.loads(episode.failed.read_text())
    assert failed["message"] == "model exploded" and failed["stage"] == "diarize"
    assert not episode.status.exists() and not episode.cues.exists()
    assert not episode.start_lock.exists() and not episode.write_lock.exists()

    view, _ = run(capsys, "episode", episode.id)
    assert view["artifacts"][1] == {"artifact": "cues", "state": "failed", "message": "model exploded"}
    assert "groups" not in view
    cues, code = run(capsys, "cues", episode.id)
    assert code == 0 and cues == {"state": "not-ready", "episodeId": episode.id, "on": "cues"}

    monkeypatch.setattr(pipeline, "now", lambda: "2026-11-11T23:53:00Z")
    engine.error = None
    result, code = run(capsys, "start", wav)
    assert code == 0 and result["created"] is True and not episode.failed.exists()
    assert result["episode"]["createdAt"] == "2026-11-11T23:53:00Z"


def test_a_missing_file_is_rejected_before_any_model_or_download(monkeypatch, root, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not be called")

    for module, name in ((transcribe, "open_transcriber"), (pipeline, "open_transcriber"), (weights, "ensure"),
                         (audio, "to_wav"), (audio, "download"), (pipeline, "transcribe")):
        monkeypatch.setattr(module, name, forbidden)
    result, code = run(capsys, "start", tmp_path / "missing.mp3")
    assert code == 1 and result["reason"] == "not-found"
    result, code = run(capsys, "start", make_wav(tmp_path / "ep.wav"), "--language", "xx")
    assert code == 1 and result["reason"] == "unknown-language"


def test_start_silences_an_hf_token_warning(engine, root, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_VERBOSITY", raising=False)
    seen: dict[str, list[str]] = {}

    def during():
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("HF_TOKEN is unset; public downloads do not need one", UserWarning)
            seen["caught"] = [str(item.message) for item in caught]

    engine.during = during
    code = cli_main(["start", str(make_wav(tmp_path / "quiet.wav"))])
    captured = capsys.readouterr()
    assert code == 0
    assert seen["caught"] == []
    assert "HF_TOKEN" not in captured.out and "HF_TOKEN" not in captured.err
    assert os.environ["HF_HUB_VERBOSITY"] == "error"


def test_any_command_silences_the_hf_token_warning(capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_VERBOSITY", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        code = cli_main(["nope"])
        warnings.warn("HF_TOKEN is unset", UserWarning)
        recorded = [str(item.message) for item in caught]
    captured = capsys.readouterr()
    assert code == 1
    assert recorded == []
    assert "HF_TOKEN" not in captured.out and "HF_TOKEN" not in captured.err
    assert os.environ["HF_HUB_VERBOSITY"] == "error"


def test_quiet_hub_keeps_a_verbosity_the_user_already_set(monkeypatch):
    monkeypatch.setenv("HF_HUB_VERBOSITY", "debug")
    transcribe.quiet_hub()
    assert os.environ["HF_HUB_VERBOSITY"] == "debug"


def test_usage_errors_are_one_json_line(capsys):
    result, code = run(capsys, "cues")
    assert code == 1 and result["kind"] == "error" and "episodeId" in result["message"]
    result, code = run(capsys, "nope")
    assert code == 1 and result["kind"] == "error"


def test_a_young_empty_lock_still_blocks(engine, root, tmp_path, capsys):
    wav = make_wav(tmp_path / "ep.wav")
    folder = Episode(root, expected_id(wav)).dir
    folder.mkdir(parents=True)
    fd = os.open(folder / "start.lock", os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.ftruncate(fd, 0)
    try:
        result, code = run(capsys, "start", wav)
    finally:
        os.close(fd)
    assert code == 1 and result["reason"] == "running"
    assert engine.calls == 0


def test_an_old_lock_held_by_a_live_process_still_blocks(engine, root, tmp_path, capsys):
    wav = make_wav(tmp_path / "ep.wav")
    folder = Episode(root, expected_id(wav)).dir
    folder.mkdir(parents=True)
    lock = folder / "start.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    old = time.time() - 13 * 3600
    os.utime(lock, (old, old))
    try:
        result, code = run(capsys, "start", wav)
    finally:
        os.close(fd)
    assert code == 1 and result["reason"] == "running"
    assert engine.calls == 0


def test_a_held_lock_with_a_dead_pid_still_blocks(engine, root, tmp_path, capsys):
    wav = make_wav(tmp_path / "ep.wav")
    folder = Episode(root, expected_id(wav)).dir
    folder.mkdir(parents=True)
    lock = folder / "start.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.ftruncate(fd, 0)
    os.write(fd, f"{dead_pid()}\n".encode())
    try:
        result, code = run(capsys, "start", wav)
    finally:
        os.close(fd)
    assert code == 1 and result["reason"] == "running"
    assert engine.calls == 0


def test_a_lock_file_left_by_a_dead_process_does_not_block(engine, root, tmp_path, capsys):
    wav = make_wav(tmp_path / "ep.wav")
    folder = Episode(root, expected_id(wav)).dir
    folder.mkdir(parents=True)
    lock = folder / "start.lock"
    pid = dead_pid()
    lock.write_text(f"{pid}\n")
    code = cli_main(["start", str(wav)])
    captured = capsys.readouterr()
    assert code == 0, captured.out
    assert f"replacing start.lock left by pid {pid}" in captured.err
    assert engine.calls == 1
    assert not lock.exists()


def test_writers_do_not_overlap_after_the_lock_file_is_removed(tmp_path):
    episode = Episode(tmp_path, "ep_" + "ef" * 8)
    episode.dir.mkdir(parents=True)
    overlap = {"cur": 0, "max": 0}
    guard = threading.Lock()

    def writer(delay, hold):
        time.sleep(delay)
        with writing(episode):
            with guard:
                overlap["cur"] += 1
                overlap["max"] = max(overlap["max"], overlap["cur"])
            time.sleep(hold)
            with guard:
                overlap["cur"] -= 1

    threads = [
        threading.Thread(target=writer, args=(0.0, 0.4)),
        threading.Thread(target=writer, args=(0.05, 0.6)),
        threading.Thread(target=writer, args=(0.5, 0.2)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert overlap["max"] == 1


def test_a_write_lock_left_by_a_dead_process_is_replaced(tmp_path, capsys):
    episode = Episode(tmp_path, "ep_" + "ab" * 8)
    episode.dir.mkdir(parents=True)
    pid = dead_pid()
    episode.write_lock.write_text(f"{pid}\n")
    with writing(episode):
        assert episode.write_lock.read_text().strip() == str(os.getpid())
    assert f"replacing write.lock left by pid {pid}" in capsys.readouterr().err
    assert not episode.write_lock.exists()


def _hold_lock(path, ready):
    fd = try_lock(path)
    assert fd is not None
    ready.set()
    time.sleep(30)
    release(fd, path)


def _contender(path, barrier, out):
    barrier.wait()
    fd = try_lock(path)
    out.put(fd is not None)
    if fd is not None:
        time.sleep(0.2)
        release(fd, path)


def _orphan(path, ready):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.ftruncate(fd, 0)
    os.write(fd, b"999999\n")
    ready.set()
    time.sleep(30)
    os.close(fd)


def _exclusive_writer(root, iterations, out):
    episode = Episode(root, "ep_0123456789abcdef")
    marker = episode.dir / "inside"
    overlaps = 0
    for _ in range(iterations):
        with writing(episode):
            try:
                fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                overlaps += 1
                continue
            time.sleep(0.001)
            os.close(fd)
            os.unlink(marker)
    out.put(overlaps)


def test_a_live_lock_is_not_stolen_after_twelve_hours(tmp_path):
    ctx = mp.get_context("spawn")
    path = tmp_path / "start.lock"
    ready = ctx.Event()
    holder = ctx.Process(target=_hold_lock, args=(path, ready))
    holder.start()
    assert ready.wait(10)
    assert try_lock(path) is None
    old = time.time() - 13 * 3600
    os.utime(path, (old, old))
    stolen = try_lock(path)
    alive = holder.is_alive()
    holder.terminate()
    holder.join(5)
    if stolen is not None:
        release(stolen, path)
    assert stolen is None and alive


def test_eight_starts_do_not_share_one_held_lock(tmp_path):
    ctx = mp.get_context("spawn")
    path = tmp_path / "start.lock"
    ready = ctx.Event()
    orphan = ctx.Process(target=_orphan, args=(path, ready))
    orphan.start()
    assert ready.wait(10)
    barrier = ctx.Barrier(8)
    out = ctx.Queue()
    contenders = [ctx.Process(target=_contender, args=(path, barrier, out)) for _ in range(8)]
    for contender in contenders:
        contender.start()
    winners = sum(out.get(timeout=10) for _ in contenders)
    for contender in contenders:
        contender.join(10)
    orphan.terminate()
    orphan.join(5)
    assert winners == 0


def test_eight_starts_take_one_free_lock(tmp_path):
    ctx = mp.get_context("spawn")
    path = tmp_path / "start.lock"
    path.write_text("999999\n")
    barrier = ctx.Barrier(8)
    out = ctx.Queue()
    contenders = [ctx.Process(target=_contender, args=(path, barrier, out)) for _ in range(8)]
    for contender in contenders:
        contender.start()
    winners = sum(out.get(timeout=10) for _ in contenders)
    for contender in contenders:
        contender.join(10)
    assert winners == 1


def test_writers_do_not_share_the_lock_across_processes(tmp_path):
    ctx = mp.get_context("spawn")
    episode = Episode(tmp_path, "ep_0123456789abcdef")
    episode.dir.mkdir(parents=True)
    out = ctx.Queue()
    writers = [ctx.Process(target=_exclusive_writer, args=(tmp_path, 40, out)) for _ in range(8)]
    for writer in writers:
        writer.start()
    overlaps = sum(out.get(timeout=30) for _ in writers)
    for writer in writers:
        writer.join(30)
    assert overlaps == 0


def test_fresh_discards_a_slice_checkpoint(engine, root, tmp_path, capsys):
    wav = make_wav(tmp_path / "ep.wav")
    started, code = run(capsys, "start", wav)
    assert code == 0
    episode = Episode(root, started["episode"]["episodeId"])
    episode.slices.write_text('{"version": 1, "slices": [{"index": 0}]}\n', encoding="utf-8")
    again, code = run(capsys, "start", wav, "--fresh")
    assert code == 0 and again["created"] is True
    assert not episode.slices.exists()


def test_unknown_episode(root, capsys):
    assert run(capsys, "episode", "ep_0000000000000000") == ({"state": "unknown", "episodeId": "ep_0000000000000000"}, 0)
    assert run(capsys, "cues", "../../etc")[0]["state"] == "unknown"


def test_write_lock_is_removed_when_the_command_finishes(tmp_path):
    episode = Episode(tmp_path, "ep_" + "ab" * 8)
    episode.dir.mkdir(parents=True)
    with writing(episode):
        assert episode.write_lock.is_file()
    assert not episode.write_lock.exists()
    with pytest.raises(RuntimeError, match="failed write"):
        with writing(episode):
            raise RuntimeError("failed write")
    assert not episode.write_lock.exists()
