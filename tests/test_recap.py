"""Recap fetch and build. Kokoro and yt-dlp are stubbed; the unit job does not install the extra."""

import json
import os
import warnings
from pathlib import Path

import pytest
from conftest import make_wav, run

from podcue import __version__, audio
from podcue.cli import main as cli_main
from podcue.fetch import download_with_ytdlp, is_direct_media
from podcue.recap import (
    RecapError,
    SPACY_MODEL,
    clip_flags,
    clip_speakers,
    ensure_spacy_model,
    narrate,
    narration_loudness_filter,
    parse_script,
    parse_ts,
    quiet_torch,
    resolve_voice,
    snap,
    words_from_cues,
)
from podcue.store import Episode


def test_version_is_0_2_0():
    assert __version__ == "0.2.7"


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
    assert words[0][:3] == (0.0, 0.6, " ab")
    assert words[1][:3] == (0.6, 1.0, " c")
    start, end, first, last = snap(
        [(0.0, 0.4, " hello"), (0.5, 0.9, " there."), (10.0, 10.4, " next"), (10.5, 11.0, " bit.")],
        0.5,
        10.4,
    )
    assert (start, end, first, last) == (0.45, 10.45, 1, 2)


def test_snap_stays_inside_the_cue_and_out_of_another_speaker():
    # Without the clamp the leading pad is 10.00 - 0.08 = 9.92, inside the previous cue,
    # and the trailing pad is 19.90 + 0.35 = 20.25, inside the next speaker's cue.
    words = [
        (9.50, 9.84, " earlier", 0.0, 10.0, "Ada"),
        (10.00, 10.50, " clip", 10.0, 20.0, "Ben"),
        (19.50, 19.90, " tail", 10.0, 20.0, "Ben"),
        (21.00, 21.40, " next", 20.0, 30.0, "Ada"),
    ]
    start, end, first, last = snap(words, 10.0, 19.9)
    assert (first, last) == (1, 2)
    assert start == 10.0
    assert end == 20.0


def test_length_warns_only_under_4s_or_over_45s_and_an_uncertain_cue_is_flagged():
    one = [{"startMs": 0, "endMs": 20_000, "groupId": "gc0001", "speaker": "gc0001", "text": "all of it."}]
    labels = [{"groupId": "gc0001", "label": "Ada"}]
    quiet = clip_speakers(one, labels, 0.0, 15.0)
    assert quiet["speakers"] == 1 and clip_flags(15.0, quiet, one) == []
    assert clip_flags(4.0, quiet) == [] and clip_flags(45.0, quiet) == []
    assert clip_flags(3.9, quiet) == ["length 3.9s under 4s"]
    assert clip_flags(45.1, quiet) == ["length 45.1s over 45s"]
    two = [
        {"startMs": 0, "endMs": 10_000, "groupId": "gc0001", "speaker": "Ada", "text": "a"},
        {"startMs": 10_000, "endMs": 20_000, "groupId": "gc0002", "speaker": "Ben", "text": "b"},
    ]
    both = clip_speakers(two, [], 0.0, 20.0)
    assert both["speakers"] == 2
    assert clip_flags(20.0, both) == ["2 speakers in clip (other speaker 10.0s)"]
    blip = [
        {"startMs": 0, "endMs": 15_000, "groupId": "gc0001", "speaker": "gc0001", "text": "a"},
        {"startMs": 15_000, "endMs": 15_400, "groupId": "gc0002", "speaker": "gc0002", "text": "b"},
    ]
    assert clip_speakers(blip, [], 0.0, 15.4)["speakers"] == 1
    slipped = [
        {"startMs": 0, "endMs": 8_000, "groupId": "gc0001", "speaker": "gc0001", "text": "guest"},
        {"startMs": 8_000, "endMs": 8_400, "groupId": "gc0001", "speaker": "uncertain", "text": "mm"},
        {"startMs": 8_400, "endMs": 16_000, "groupId": "gc0001", "speaker": "gc0001", "text": "guest"},
    ]
    cover = clip_speakers(slipped, [], 0.0, 16.0)
    assert cover["speakers"] == 1
    assert clip_flags(16.0, cover, slipped) == ["uncertain, mixed, or undetermined speaker in clip"]
    mixed = [{"startMs": 0, "endMs": 12_000, "groupId": "gc0001", "speaker": "mixed", "text": "both"}]
    assert clip_flags(12.0, clip_speakers(mixed, [], 0.0, 12.0), mixed) == [
        "uncertain, mixed, or undetermined speaker in clip"]
    same_label = [
        {"startMs": 0, "endMs": 8_000, "groupId": "gc0001", "speaker": "Ada", "text": "a"},
        {"startMs": 8_000, "endMs": 16_000, "groupId": "gc0002", "speaker": "Ada", "text": "b"},
    ]
    assert clip_speakers(same_label, [], 0.0, 16.0)["speakers"] == 1
    assert clip_flags(16.0, clip_speakers(same_label, [], 0.0, 16.0), same_label) == []
    quotes = [
        {"startMs": 0, "endMs": 3_000, "groupId": "gq1", "speaker": "quote", "text": "3"},
        {"startMs": 3_000, "endMs": 6_000, "groupId": "gq2", "speaker": "quote", "text": "2"},
        {"startMs": 6_000, "endMs": 9_000, "groupId": "gq3", "speaker": "quote", "text": "1"},
    ]
    assert clip_speakers(quotes, [], 0.0, 9.0)["speakers"] == 1
    undetermined = [
        {"startMs": 0, "endMs": 8_000, "groupId": "gc0001", "speaker": "Ada", "text": "guest"},
        {"startMs": 8_000, "endMs": 8_400, "groupId": "gc0009", "speaker": "undetermined", "text": "um"},
        {"startMs": 8_400, "endMs": 16_000, "groupId": "gc0001", "speaker": "Ada", "text": "guest"},
    ]
    quiet_blip = clip_speakers(undetermined, [], 0.0, 16.0)
    assert quiet_blip["speakers"] == 1
    assert clip_flags(16.0, quiet_blip, undetermined) == ["uncertain, mixed, or undetermined speaker in clip"]
    old_file = [
        {"startMs": 0, "endMs": 8_000, "groupId": "gc0001", "speaker": "gc0001", "text": "a"},
        {"startMs": 8_000, "endMs": 8_300, "groupId": "gc0002", "speaker": "gc0002", "text": "b"},
    ]
    old_labels = [{"groupId": "gc0001", "label": "Ada"}, {"groupId": "gc0002", "label": "undetermined"}]
    assert clip_flags(8.3, clip_speakers(old_file, old_labels, 0.0, 8.3), old_file, old_labels) == [
        "uncertain, mixed, or undetermined speaker in clip"]
    same_person = [
        {"startMs": 0, "endMs": 8_000, "groupId": "gc0001", "speaker": "gc0001", "text": "a"},
        {"startMs": 8_000, "endMs": 16_000, "groupId": "gc0002", "speaker": "gc0002", "text": "b"},
    ]
    labeled = [{"groupId": "gc0001", "label": "Ada"}, {"groupId": "gc0002", "label": "Ada"}]
    assert clip_speakers(same_person, labeled, 0.0, 16.0)["speakers"] == 1
    quotes = clip_speakers([
        {"startMs": 0, "endMs": 12_000, "groupId": "gc0001", "speaker": "quote", "text": "played"},
    ], [{"groupId": "gc0001", "label": "Ada"}], 0.0, 12.0)
    assert quotes["dominant"] == "quote" and quotes["speakers"] == 1
    assert clip_flags(12.0, {"speakers": 0, "otherSeconds": 0.0}) == ["no speech in clip"]


def test_narration_gain_targets_the_mix_and_limits_the_peak():
    text = narration_loudness_filter(-27.5)
    assert "volume=11.50dB" in text
    assert "alimiter=limit=0.8414" in text
    assert "level=false" in text


def test_quiet_torch_ignores_torch_warnings_only():
    quiet_torch()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet_torch()
        warnings.warn_explicit(
            "`torch.nn.utils.weight_norm` is deprecated",
            FutureWarning,
            filename="kokoro/model.py",
            lineno=1,
            module="kokoro.model",
        )
        warnings.warn_explicit(
            "dropout option adds dropout after all but last recurrent layer",
            UserWarning,
            filename="torch/nn/modules/rnn.py",
            lineno=1,
            module="torch.nn.modules.rnn",
        )
        warnings.warn_explicit(
            "something else is deprecated",
            DeprecationWarning,
            filename="podcue/recap.py",
            lineno=1,
            module="podcue.recap",
        )
    messages = [str(item.message) for item in caught]
    assert not any("weight_norm" in message for message in messages)
    assert not any("dropout" in message for message in messages)
    assert any("something else" in message for message in messages)


def test_a_missing_spacy_model_is_announced_and_unpacked(monkeypatch, tmp_path, capsys):
    import podcue.recap as recap

    wheel = tmp_path / "model.whl"
    import zipfile
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("en_core_web_sm-3.8.0.dist-info/METADATA", "Name: en-core-web-sm\nVersion: 3.8.0\n")
        archive.writestr("en_core_web_sm/__init__.py", "")
    calls = {"n": 0}

    def installed():
        calls["n"] += 1
        return calls["n"] >= 3

    monkeypatch.setattr(recap, "_spacy_installed", installed)
    monkeypatch.setattr(recap, "ensure", lambda pin: wheel)
    dest = tmp_path / "site"
    monkeypatch.setattr(recap.sysconfig, "get_path", lambda _name: str(dest))
    ensure_spacy_model()
    err = capsys.readouterr().err
    assert "installing spaCy model en_core_web_sm 3.8.0" in err
    assert (dest / "en_core_web_sm" / "__init__.py").is_file()
    assert "en_core_web_sm 3.8.0" in SPACY_MODEL.label
    assert SPACY_MODEL.sha256 == "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
    assert "github.com/explosion/spacy-models" in SPACY_MODEL.url
    capsys.readouterr()
    ensure_spacy_model()
    assert capsys.readouterr().err == ""


def test_narrate_names_the_recap_extra_when_kokoro_is_absent(monkeypatch, tmp_path):
    import builtins

    real = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"kokoro", "soundfile"} or name.startswith("kokoro."):
            raise ModuleNotFoundError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.delenv("HF_HUB_VERBOSITY", raising=False)
    with pytest.raises(RecapError, match=rf"podcue\[recap\]=={__version__}"):
        narrate("Hello.", "am_michael", tmp_path / "out.wav")
    assert os.environ["HF_HUB_VERBOSITY"] == "error"


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


def test_build_silences_an_hf_token_warning(episode_id, audio_ops, tmp_path, capsys, monkeypatch):
    import podcue.recap as recap

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_VERBOSITY", raising=False)
    seen: dict[str, list[str]] = {}
    stub = recap.narrate

    def narrate(text, voice, dest):
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("HF_TOKEN is unset", UserWarning)
            seen["caught"] = [str(item.message) for item in caught]
        stub(text, voice, dest)

    monkeypatch.setattr(recap, "narrate", narrate)
    script = write_script(tmp_path / "script.json", {
        "segments": [{"type": "narration", "text": "Intro."}],
    })
    code = cli_main(["build", episode_id, str(script), "--out", str(tmp_path / "out.mp3")])
    captured = capsys.readouterr()
    assert code == 0
    assert seen["caught"] == []
    assert "HF_TOKEN" not in captured.out and "HF_TOKEN" not in captured.err
    assert os.environ["HF_HUB_VERBOSITY"] == "error"


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
    assert all("under 4s" not in flag and "over 45s" not in flag for flag in result["flags"])
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert "podcue recap" in report and "the guest" in report and episode_id in report
    clip = result["segments"][1]
    assert clip["type"] == "clip" and clip["speaker"] == "the guest"
    assert clip["requestedStart"] == "0:01.0" and clip["speakers"] > 1
    assert out.read_bytes() == b"ID3fake"

    run(capsys, "build", episode_id, script, "--out", out)
    assert audio_ops["narrate"] == [("Intro.", "am_puck"), ("Bridge.", "af_heart")]
    assert len(audio_ops["cut"]) == 1


def test_build_flags_a_clip_under_4_seconds(episode_id, audio_ops, tmp_path, capsys):
    script = write_script(tmp_path / "script.json", {
        "segments": [{"type": "clip", "start": 0, "end": 3, "speaker": "Ada"}],
    })
    result, code = run(capsys, "build", episode_id, script, "--out", tmp_path / "short.mp3")
    assert code == 0
    assert any("under 4s" in flag for flag in result["flags"])
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
