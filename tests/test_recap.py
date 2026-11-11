"""Recap fetch and build. Kokoro and yt-dlp are stubbed; the unit job does not install the extra."""

import json
from pathlib import Path

import pytest
from conftest import make_wav, run

from podcue import __version__, audio
from podcue.fetch import download_with_ytdlp, is_direct_media
from podcue.recap import (
    RecapError,
    clip_flags,
    clip_speakers,
    narrate,
    parse_script,
    parse_ts,
    resolve_voice,
    snap,
    words_from_cues,
)
from podcue.store import Episode


def test_version_is_0_2_0():
    assert __version__ == "0.2.1"


def test_importing_the_cli_does_not_load_the_recap_extra():
    import sys

    import podcue.cli

    assert podcue.cli.__version__
    assert "kokoro" not in sys.modules
    assert "yt_dlp" not in sys.modules
    assert "soundfile" not in sys.modules


def test_readme_does_not_name_the_podbox_recap_skill():
    readme = Path("README.md").read_text(encoding="utf-8").lower()
    assert "recap skill" not in readme
    assert "podbox://skill/recap" not in readme
    assert "podsum" not in readme


# script.json


def test_parse_keeps_the_existing_script_shape():
    script = parse_script({
        "voice": "am_puck",
        "output": "produced_summary.mp3",
        "segments": [
            {"type": "narration", "text": "Intro."},
            {"type": "clip", "start": "40:43.5", "end": "40:58.0", "speaker": "the guest"},
            {"type": "narration", "text": "Bridge.", "voice": "american-female-heart"},
        ],
    })
    assert script.output == "produced_summary.mp3"
    assert resolve_voice(script.voice, script.voice_preset, script.segments[0], 1) == "am_puck"
    assert resolve_voice(script.voice, script.voice_preset, script.segments[2], 3) == "af_heart"
    assert parse_ts("40:43.5") == 2443.5
    assert parse_ts("40:58.0") == 2458.0
    assert parse_ts(90.5) == 90.5
    assert parse_ts("1:02:03.5") == 3723.5


def test_voice_preset_and_a_bare_segment_array():
    preset = parse_script({
        "voice_preset": "american-male-puck",
        "segments": [{"type": "narration", "text": "Hi.", "voice_preset": "american-female-heart"}],
    })
    assert resolve_voice(preset.voice, preset.voice_preset, preset.segments[0], 1) == "af_heart"
    assert resolve_voice(preset.voice, preset.voice_preset, None, 1) == "am_puck"
    bare = parse_script([{"type": "narration", "text": "Hi."}])
    assert resolve_voice(bare.voice, bare.voice_preset, bare.segments[0], 1) == "am_michael"


@pytest.mark.parametrize("body, message", [
    ({"segments": []}, "script.json must be"),
    ({"segments": [{"type": "song", "text": "x"}]}, "segment 1: type must be narration or clip"),
    ({"segments": [{"type": "narration", "text": "  "}]}, "segment 1: narration text must be a non-empty string"),
    ({"segments": [{"type": "clip", "start": "nope", "end": "1:00"}]}, "segment 1: start must be seconds"),
    ({"segments": [{"type": "clip", "start": "1:00", "end": 30}]}, "segment 1: end must be after start"),
    ({"segments": [{"type": "clip", "start": 0, "end": 1, "speaker": 3}]}, "segment 1: speaker must be a string"),
    ([{"type": "narration", "text": "Hi.", "voice": 1}], "segment 1: voice must be a string"),
    ("nope", "script.json must be"),
])
def test_parse_rejects_a_bad_script(body, message):
    with pytest.raises(RecapError, match=message):
        parse_script(body)


def test_words_are_spread_across_the_cue_and_edges_snap_to_them():
    words = words_from_cues([
        {"startMs": 0, "endMs": 1_000, "text": "ab c"},
        {"startMs": 10_000, "endMs": 11_000, "text": "next bit."},
    ])
    assert words[0] == (0.0, 0.6, " ab")
    assert words[1] == (0.6, 1.0, " c")
    start, end, first, last = snap(
        [(0.0, 0.4, " hello"), (0.5, 0.9, " there."), (10.0, 10.4, " next"), (10.5, 11.0, " bit.")],
        0.5,
        10.4,
    )
    assert (start, end, first, last) == (0.45, 10.45, 1, 2)


def test_one_group_is_quiet_and_a_second_group_or_a_short_clip_is_flagged():
    one = [{"startMs": 0, "endMs": 20_000, "groupId": "gc0001", "text": "all of it."}]
    labels = [{"groupId": "gc0001", "label": "Ada"}]
    quiet = clip_speakers(one, labels, 0.0, 15.0)
    assert quiet["speakers"] == 1 and clip_flags(15.0, quiet) == []
    two = [
        {"startMs": 0, "endMs": 10_000, "groupId": "gc0001", "text": "a"},
        {"startMs": 10_000, "endMs": 20_000, "groupId": "gc0002", "text": "b"},
    ]
    both = clip_speakers(two, [], 0.0, 20.0)
    assert both["speakers"] == 2
    assert clip_flags(20.0, both) == ["2 speakers in clip (other speaker 10.0s)"]
    blip = [
        {"startMs": 0, "endMs": 15_000, "groupId": "gc0001", "text": "a"},
        {"startMs": 15_000, "endMs": 15_400, "groupId": "gc0002", "text": "b"},
    ]
    assert clip_speakers(blip, [], 0.0, 15.4)["speakers"] == 1
    assert clip_flags(9.9, quiet) == ["length 9.9s outside 10-20s"]
    assert clip_flags(20.1, quiet) == ["length 20.1s outside 10-20s"]
    assert clip_flags(12.0, {"speakers": 0, "otherSeconds": 0.0}) == ["no speech in clip"]


def test_narrate_names_the_recap_extra_when_kokoro_is_absent(monkeypatch, tmp_path):
    import builtins

    real = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"kokoro", "soundfile"} or name.startswith("kokoro."):
            raise ModuleNotFoundError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RecapError, match=rf"podcue\[recap\]=={__version__}"):
        narrate("Hello.", "am_michael", tmp_path / "out.wav")


# commands


@pytest.fixture
def audio_ops(monkeypatch):
    import podcue.recap as recap

    calls = {"narrate": [], "cut": [], "stitch": 0}

    def narrate(text, voice, dest):
        calls["narrate"].append((text, voice))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"RIFFnarr")

    def cut(source, start, end, dest):
        calls["cut"].append((round(start, 2), round(end, 2)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"RIFFclip")

    def loudnorm(source, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"RIFFnorm")
        return -20.0

    def measure(_path):
        return {"input_i": "-16.0", "input_tp": "-1.5", "input_lra": "11.0",
                "input_thresh": "-26.0", "target_offset": "0.0"}

    def stitch(paths, dest, _work):
        calls["stitch"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ID3fake")

    monkeypatch.setattr(recap, "narrate", narrate)
    monkeypatch.setattr(recap, "cut_clip", cut)
    monkeypatch.setattr(recap, "loudnorm_to", loudnorm)
    monkeypatch.setattr(recap, "measure_loudness", measure)
    monkeypatch.setattr(recap, "stitch", stitch)
    monkeypatch.setattr(recap, "probe_duration_s", lambda _path: 30.0)
    return calls


def write_script(path: Path, body: object) -> Path:
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_build_stitches_a_script_and_flags_a_second_speaker(episode_id, audio_ops, tmp_path, capsys):
    script = write_script(tmp_path / "script.json", {
        "voice": "american-male-puck",
        "segments": [
            {"type": "narration", "text": "Intro."},
            {"type": "clip", "start": "0:01.0", "end": "0:16.0", "speaker": "the guest"},
            {"type": "narration", "text": "Bridge.", "voice": "af_heart"},
        ],
    })
    out = tmp_path / "recap.mp3"
    result, code = run(capsys, "build", episode_id, script, "--out", out)
    assert code == 0 and result["kind"] == "built"
    assert result["episodeId"] == episode_id
    assert result["output"] == str(out.resolve())
    assert result["report"] == str(out.resolve().parent / "report.md")
    assert result["durationMs"] == 30_000
    assert result["loudness"] == {"lufs": -16.0, "truePeak": -1.5}
    assert audio_ops["narrate"] == [("Intro.", "am_puck"), ("Bridge.", "af_heart")]
    assert len(audio_ops["cut"]) == 1
    assert any("speakers in clip" in flag for flag in result["flags"])
    assert all("outside 10-20s" not in flag for flag in result["flags"])
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert "podcue recap" in report and "the guest" in report and episode_id in report
    clip = result["segments"][1]
    assert clip["type"] == "clip" and clip["speaker"] == "the guest"
    assert clip["requestedStart"] == "0:01.0" and clip["speakers"] > 1
    assert out.read_bytes() == b"ID3fake"

    run(capsys, "build", episode_id, script, "--out", out)
    assert audio_ops["narrate"] == [("Intro.", "am_puck"), ("Bridge.", "af_heart")]
    assert len(audio_ops["cut"]) == 1


def test_build_flags_a_clip_outside_10_to_20_seconds(episode_id, audio_ops, tmp_path, capsys):
    script = write_script(tmp_path / "script.json", {
        "segments": [{"type": "clip", "start": 0, "end": 3, "speaker": "Ada"}],
    })
    result, code = run(capsys, "build", episode_id, script, "--out", tmp_path / "short.mp3")
    assert code == 0
    assert any("outside 10-20s" in flag for flag in result["flags"])
    assert audio_ops["narrate"] == []


def test_build_output_field_is_used_when_out_is_omitted(episode_id, audio_ops, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    script = write_script(tmp_path / "script.json", {
        "output": "produced_summary.mp3",
        "segments": [{"type": "narration", "text": "Intro."}],
    })
    result, code = run(capsys, "build", episode_id, script)
    assert code == 0
    assert result["output"] == str((tmp_path / "produced_summary.mp3").resolve())
    assert result["flags"] == []
    assert result["segments"][0]["voice"] == "am_michael"
    assert audio_ops["narrate"] == [("Intro.", "am_michael")]
    assert audio_ops["stitch"] == 1


def test_build_errors_are_one_json_line(episode_id, tmp_path, capsys):
    missing, code = run(capsys, "build", episode_id, tmp_path / "missing.json")
    assert code == 1 and missing["kind"] == "error" and "not a file" in missing["message"]
    bad = write_script(tmp_path / "bad.json", {"segments": [{"type": "narration", "text": ""}]})
    result, code = run(capsys, "build", episode_id, bad)
    assert code == 1 and result["kind"] == "error" and "narration text" in result["message"]
    assert run(capsys, "build", "ep_0123456789abcdef", bad)[0]["state"] == "unknown"


def test_build_before_cues_is_not_ready(root, tmp_path, capsys):
    episode = Episode(root, "ep_0123456789abcdef")
    episode.dir.mkdir(parents=True)
    episode.manifest.write_text(json.dumps({
        "episodeId": episode.id, "source": "ep.wav", "durationMs": 1_000,
        "language": "en", "createdAt": "2026-11-01T00:00:00Z",
    }), encoding="utf-8")
    script = write_script(tmp_path / "script.json", {"segments": [{"type": "narration", "text": "Hi."}]})
    assert run(capsys, "build", episode.id, script) == (
        {"state": "not-ready", "episodeId": episode.id, "on": "cues"}, 0)


def test_build_refuses_a_missing_source_file(episode_id, root, tmp_path, capsys):
    view, _ = run(capsys, "episode", episode_id)
    Path(view["source"]).unlink()
    script = write_script(tmp_path / "script.json", {"segments": [{"type": "narration", "text": "Hi."}]})
    result, code = run(capsys, "build", episode_id, script, "--out", tmp_path / "out.mp3")
    assert code == 1 and result["kind"] == "error" and result["reason"] == "not-found"
    assert not Episode(root, episode_id).dir.joinpath("recap-cache").exists()


# fetch


def test_direct_media_is_a_file_suffix_and_a_page_is_not():
    assert is_direct_media("https://cdn.example.com/ep.mp3?token=1")
    assert is_direct_media("https://cdn.example.com/show/EP.M4A")
    assert not is_direct_media("https://www.youtube.com/watch?v=abc")
    assert not is_direct_media("https://youtu.be/abc")


def test_fetch_direct_url_writes_a_wav_without_yt_dlp(tmp_path, monkeypatch, capsys):
    def download(url, directory, opener=None):
        assert url == "https://cdn.example.com/ep.mp3"
        path = directory / "download.mp3"
        path.write_bytes(b"audio")
        return path

    def forbidden(url, directory):
        raise AssertionError("yt-dlp should not run for a direct media URL")

    monkeypatch.setattr(audio, "download", download)
    monkeypatch.setattr(audio, "to_wav", lambda source, target: make_wav(target))
    monkeypatch.setattr("podcue.fetch.download_with_ytdlp", forbidden)
    result, code = run(capsys, "fetch", "https://cdn.example.com/ep.mp3", "--out", tmp_path)
    assert code == 0
    assert result == {
        "kind": "fetched",
        "url": "https://cdn.example.com/ep.mp3",
        "path": str((tmp_path / "audio.wav").resolve()),
        "durationMs": 1_000,
        "via": "direct",
    }
    assert Path(result["path"]).is_file()


def test_fetch_page_uses_yt_dlp(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake(url, directory):
        seen["url"] = url
        path = directory / "download.webm"
        path.write_bytes(b"webm")
        return path

    monkeypatch.setattr("podcue.fetch.download_with_ytdlp", fake)
    monkeypatch.setattr(audio, "to_wav", lambda source, target: make_wav(target))
    result, code = run(capsys, "fetch", "https://www.youtube.com/watch?v=abc", "--out", tmp_path)
    assert code == 0 and result["via"] == "yt-dlp" and seen["url"].endswith("v=abc")
    assert result["path"].endswith("audio.wav")


def test_fetch_page_without_yt_dlp_is_an_error(tmp_path, monkeypatch, capsys):
    import builtins

    real = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "yt_dlp" or name.startswith("yt_dlp."):
            raise ModuleNotFoundError("yt_dlp")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result, code = run(capsys, "fetch", "https://www.youtube.com/watch?v=abc", "--out", tmp_path)
    assert code == 1 and result["kind"] == "error"
    assert f"podcue[recap]=={__version__}" in result["message"]
    assert not (tmp_path / "audio.wav").exists()


def test_fetch_rejects_a_non_url(tmp_path, capsys):
    result, code = run(capsys, "fetch", "/tmp/ep.mp3", "--out", tmp_path)
    assert code == 1 and result == {"kind": "error", "message": "/tmp/ep.mp3 is not an http(s) URL"}


def test_download_with_ytdlp_wraps_a_failure(tmp_path, monkeypatch):
    class Boom:
        def __init__(self, options):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def download(self, urls):
            raise RuntimeError("network down")

    monkeypatch.setattr("podcue.fetch.load_ytdlp", lambda: Boom)
    with pytest.raises(RecapError, match="yt-dlp could not download"):
        download_with_ytdlp("https://www.youtube.com/watch?v=abc", tmp_path)
