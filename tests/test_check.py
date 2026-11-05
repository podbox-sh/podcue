import json

import pytest

from podcue.check import check
from podcue.cli import main

CUES = {
    "version": 1,
    "source": "ep.mp3",
    "durationMs": 9_000,
    "language": "en",
    "speakers": [{"id": "S1"}, {"id": "S2"}],
    "cues": [
        {"id": "c0001", "text": "a", "startMs": 0, "endMs": 1_000, "speaker": "S1"},
        {"id": "c0002", "text": "b", "startMs": 1_000, "endMs": 2_000, "speaker": "S2"},
        {"id": "c0003", "text": "c", "startMs": 2_500, "endMs": 4_000, "speaker": None},
        {"id": "c0004", "text": "d", "startMs": 5_000, "endMs": 9_000, "speaker": "S1"},
    ],
}


def summary(**overrides):
    doc = {
        "summary": "Two people talk.",
        "speakers": {"S1": "Ada", "S2": "Grace"},
        "paragraphs": [
            {"text": "Ada opens.", "cues": ["c0004", "c0001"], "startMs": 0, "endMs": 9_000},
            {"text": "Grace replies.", "cues": ["c0002", "c0003"], "startMs": 1_000, "endMs": 4_000},
        ],
    }
    doc.update(overrides)
    return doc


def test_a_valid_summary_passes():
    assert check(summary(), CUES) == []


def test_speakers_may_be_a_list():
    labels = [{"id": "S1", "label": "Ada"}, {"id": "S2", "label": "Grace"}]
    assert check(summary(speakers=labels), CUES) == []


def test_unknown_and_repeated_cues_are_reported():
    paragraphs = [{"text": "x", "cues": ["c0001", "c0001", "c0099"], "startMs": 0, "endMs": 1_000}]
    assert check(summary(paragraphs=paragraphs), CUES) == [
        "paragraph 1: cue c0001 is listed twice",
        "paragraph 1: cue c0099 is not in the cues file",
    ]


def test_times_must_match_the_first_and_last_cited_cue():
    paragraphs = [{"text": "x", "cues": ["c0001", "c0002"], "startMs": 10, "endMs": 1_000}]
    assert check(summary(paragraphs=paragraphs), CUES) == [
        "paragraph 1: startMs 10 does not match c0001 startMs 0",
        "paragraph 1: endMs 1000 does not match c0002 endMs 2000",
    ]


def test_times_are_required():
    paragraphs = [{"text": "x", "cues": ["c0001"]}]
    assert check(summary(paragraphs=paragraphs), CUES) == [
        "paragraph 1: startMs must be a number",
        "paragraph 1: endMs must be a number",
    ]


@pytest.mark.parametrize("label", [None, "", "  ", "undetermined", "S2"])
def test_every_cited_speaker_needs_a_human_label(label):
    labels = {"S1": "Ada"} if label is None else {"S1": "Ada", "S2": label}
    assert check(summary(speakers=labels), CUES) == [
        "speaker S2 (first cited in paragraph 2) has no human label in speakers"
    ]


def test_speakers_listed_on_a_paragraph_count_as_referenced():
    paragraphs = [{"text": "x", "cues": ["c0003"], "startMs": 2_500, "endMs": 4_000, "speakers": ["S2", "S9"]}]
    assert check(summary(speakers={"S1": "Ada"}, paragraphs=paragraphs), CUES) == [
        "paragraph 1: speaker S9 is not in the cues file",
        "speaker S2 (first cited in paragraph 1) has no human label in speakers",
    ]


def test_labels_for_unknown_speakers_are_reported():
    assert check(summary(speakers={"S1": "Ada", "S2": "Grace", "S7": "Bob"}), CUES) == [
        "speakers: S7 is not a speaker in the cues file"
    ]


def test_shape_errors():
    assert check([], CUES) == ["summary must be a mapping with a paragraphs list"]
    assert check(summary(paragraphs=[]), CUES) == ["paragraphs must be a non-empty list"]
    assert check(summary(summary=""), CUES) == ["summary must be a non-empty string when present"]
    bad = [{"text": "", "cues": [], "startMs": 0, "endMs": 1}]
    assert check(summary(paragraphs=bad), CUES) == [
        "paragraph 1: text must be a non-empty string",
        "paragraph 1: cues must be a non-empty list of cue ids",
    ]


def test_cli_reads_yaml_and_sets_the_exit_code(tmp_path, capsys):
    cues_path = tmp_path / "ep.cues.json"
    cues_path.write_text(json.dumps(CUES))
    good = tmp_path / "good.yaml"
    good.write_text(
        "summary: Two people talk.\n"
        "speakers:\n  S1: Ada\n  S2: Grace\n"
        "paragraphs:\n"
        "  - text: Ada opens.\n    cues: [c0001]\n    startMs: 0\n    endMs: 1000\n"
    )
    assert main(["check", str(good), "--cues", str(cues_path)]) == 0
    assert "ok, 1 paragraphs" in capsys.readouterr().out

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(summary(speakers={})))
    assert main(["check", str(bad), "--cues", str(cues_path)]) == 1
    err = capsys.readouterr().err
    assert "speaker S1 (first cited in paragraph 1) has no human label" in err
    assert "2 problem(s)" in err

    broken = tmp_path / "broken.yaml"
    broken.write_text("paragraphs: [\n")
    assert main(["check", str(broken), "--cues", str(cues_path)]) == 1
