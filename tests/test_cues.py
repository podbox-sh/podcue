import pytest

from podcue.cues import Cue, Word, build_cues, cue_document, render_vtt, shift, stamp
from podcue.diarize import Tracks, Turn
from podcue.regions import Slice


def whole(start, end, region=0, index=0):
    return Slice(index, start, end, start, end, region)


def test_shift_rounds_half_up_to_the_millisecond():
    assert shift(1_000, 0.0005) == 1_001
    assert shift(1_000, 0.0004) == 1_000
    assert shift(0, 2.5) == 2_500


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_shift_rejects_bad_word_times(bad):
    with pytest.raises(ValueError):
        shift(0, bad)


def test_sentence_punctuation_closes_a_cue():
    words = [Word("Hello", 0.0, 0.4), Word("there.", 0.5, 0.9), Word("Next", 1.0, 1.3), Word("one?", 1.4, 1.8)]
    assert build_cues([(whole(10_000, 20_000), words)]) == [
        Cue("c0001", 10_000, 10_900, "Hello there."),
        Cue("c0002", 11_000, 11_800, "Next one?"),
    ]


def test_cjk_full_stop_closes_a_cue():
    words = [Word("今日は", 0.0, 0.5), Word("晴れ。", 0.5, 1.0), Word("明日", 1.1, 1.5)]
    assert [c.text for c in build_cues([(whole(0, 5_000), words)])] == ["今日は 晴れ。", "明日"]


def test_a_cue_closes_after_twelve_seconds_without_punctuation():
    words = [Word(f"w{i}", float(i), i + 0.5) for i in range(14)]
    cues = build_cues([(whole(0, 20_000), words)])
    assert [(c.start_ms, c.end_ms) for c in cues] == [(0, 11_500), (12_000, 13_500)]


def test_a_cue_closes_where_a_speaker_region_ends():
    first = (whole(0, 2_000, region=0, index=0), [Word("so", 0.0, 0.5), Word("anyway", 0.6, 1.0)])
    second = (whole(2_000, 4_000, region=1, index=1), [Word("right", 0.1, 0.5)])
    assert [c.text for c in build_cues([first, second])] == ["so anyway", "right"]


def test_words_outside_the_keep_window_are_dropped():
    early = Slice(0, 0, 28_000, 0, 27_000, 0)
    late = Slice(1, 27_000, 40_000, 27_000, 40_000, 0)
    words_early = [Word("kept.", 26.0, 26.5), Word("repeated.", 27.2, 27.6)]
    words_late = [Word("repeated.", 0.2, 0.6), Word("end.", 1.0, 1.5)]
    cues = build_cues([(early, words_early), (late, words_late)])
    assert [(c.text, c.start_ms) for c in cues] == [("kept.", 26_000), ("repeated.", 27_200), ("end.", 28_000)]


def test_zero_length_words_get_one_millisecond_and_blank_words_are_skipped():
    cues = build_cues([(whole(0, 5_000), [Word(" ", 0.0, 0.1), Word("Hi.", 1.0, 1.0)])])
    assert cues == [Cue("c0001", 1_000, 1_001, "Hi.")]


def make_document():
    cues = [
        Cue("c0001", 0, 1_000, "Hi."),
        Cue("c0002", 1_000, 2_000, "Hello."),
        Cue("c0003", 2_000, 3_000, "Silence?"),
        Cue("c0004", 5_000, 6_000, "Back."),
    ]
    tracks = Tracks(
        turns=[Turn(0, 1_100, 1), Turn(1_100, 2_000, 0), Turn(4_900, 6_000, 1)],
        centroids=[[1.0, 0.0], [0.6, 0.8]],
    )
    return cue_document(source="ep.mp3", duration_ms=6_000, language="en", cues=cues, tracks=tracks)


def test_speakers_are_numbered_by_first_cue_and_unmatched_cues_have_no_speaker():
    doc = make_document()
    assert [c["speaker"] for c in doc["cues"]] == ["S1", "S2", None, "S1"]
    assert doc["speakers"] == [
        {"id": "S1", "speakingMs": 2_200, "cueCount": 2, "nearestSpeaker": "S2", "nearestCosine": 0.6},
        {"id": "S2", "speakingMs": 900, "cueCount": 1, "nearestSpeaker": "S1", "nearestCosine": 0.6},
    ]
    assert doc["cues"][0] == {"id": "c0001", "text": "Hi.", "startMs": 0, "endMs": 1_000, "speaker": "S1"}
    assert (doc["version"], doc["source"], doc["durationMs"], doc["language"]) == (1, "ep.mp3", 6_000, "en")


def test_one_speaker_has_no_nearest():
    tracks = Tracks([Turn(0, 1_000, 0)], [[1.0, 0.0]])
    doc = cue_document(source="x", duration_ms=1_000, language="en", cues=[Cue("c0001", 0, 900, "a")], tracks=tracks)
    assert doc["speakers"][0]["nearestSpeaker"] is None
    assert doc["speakers"][0]["nearestCosine"] is None


def test_vtt_has_ids_voice_tags_and_escaped_text():
    doc = make_document()
    doc["cues"][1]["text"] = "a < b & c"
    assert render_vtt(doc) == (
        "WEBVTT\n\n"
        "c0001\n00:00:00.000 --> 00:00:01.000\n<v S1>Hi.\n\n"
        "c0002\n00:00:01.000 --> 00:00:02.000\n<v S2>a &lt; b &amp; c\n\n"
        "c0003\n00:00:02.000 --> 00:00:03.000\nSilence?\n\n"
        "c0004\n00:00:05.000 --> 00:00:06.000\n<v S1>Back.\n"
    )


def test_stamp_formats_hours():
    assert stamp(3_723_004) == "01:02:03.004"
