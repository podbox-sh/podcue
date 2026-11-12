import hashlib
import json
import os
import subprocess
import sys
import warnings

import pytest
from conftest import make_wav, run

from podcue import audio, pipeline, transcribe, weights
from podcue.cli import execute, main as cli_main
from podcue.store import Episode, try_lock


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
    assert [(a["artifact"], a["state"], a["progress"]) for a in view["artifacts"]] == [
        ("transcript", "ready", 100), ("cues", "ready", 100)]
    assert [g["groupId"] for g in view["groups"]] == ["gc0001", "gc0002", "ungrouped"]
    assert view["groups"][0] == {"groupId": "gc0001", "speakingMs": 36_000, "cueCount": 40,
                                 "nearestGroupId": "gc0002", "nearestCosine": 0.6, "label": None}
    assert view["labeledShare"] == 0.0 and view["undeterminedShare"] == 0.0 and view["summaries"] == []

    folder = root / "episodes" / view["episodeId"]
    assert {p.name for p in folder.iterdir()} == {
        "episode.json", "transcript.vtt", "cues.json", "groups.json", "start.lock"}
    assert (folder / "start.lock").read_text() == ""
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
    episode.labels.write_text('{"groups": [{"groupId": "gc0001", "label": "Ada"}]}')
    other_cues = (root / "episodes" / other["episode"]["episodeId"] / "cues.json").read_bytes()
    monkeypatch.setattr(pipeline, "now", lambda: "2026-11-12T05:31:00Z")
    fresh, code = run(capsys, "start", first_wav, "--fresh")
    assert code == 0 and fresh["created"] is True
    assert fresh["episode"]["createdAt"] == "2026-11-12T05:31:00Z"
    assert fresh["episode"]["source"] == str(first_wav.resolve())
    assert engine.calls == 3
    assert "Ada" in episode.labels.read_text()
    assert (root / "episodes" / other["episode"]["episodeId"] / "cues.json").read_bytes() == other_cues
    assert marker.read_text() == "keep"
    assert episode.cues.is_file() and episode.transcript.is_file()


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
    assert not episode.status.exists() and episode.start_lock.read_text() == ""


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


def test_unknown_episode(root, capsys):
    assert run(capsys, "episode", "ep_0000000000000000") == ({"state": "unknown", "episodeId": "ep_0000000000000000"}, 0)
    assert run(capsys, "cues", "../../etc")[0]["state"] == "unknown"
