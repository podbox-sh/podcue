import numpy as np
import pytest

from podcue.cues import Cue, Word, build_cues, cue_rows, group_table, join_cue_text, render_vtt, shift, stamp
from podcue.diarize import Tracks, Turn
from podcue.pauses import Quiet
from podcue.regions import Slice


def whole(start, end, region=0, index=0):
    return Slice(index, start, end, region)


def speech(total_ms, spans):
    """Noise at one steady level inside each (startMs, endMs) span and digital silence elsewhere."""
    samples = np.zeros(total_ms * 16, dtype=np.float32)
    rng = np.random.default_rng(0)
    for start, end in spans:
        samples[start * 16 : end * 16] = rng.standard_normal((end - start) * 16) * 0.1
    return samples


def timed(spans):
    return [Word(f"w{index}", start / 1000, end / 1000) for index, (start, end) in enumerate(spans)]


def test_shift_rounds_half_up_to_the_millisecond():
    assert shift(1_000, 0.0005) == 1_001
    assert shift(1_000, 0.0004) == 1_000
    assert shift(0, 2.5) == 2_500


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_shift_rejects_bad_word_times(bad):
    with pytest.raises(ValueError):
        shift(0, bad)


def test_number_punctuation_joins_without_a_space():
    assert join_cue_text(["400", ",000"]) == "400,000"
    assert join_cue_text(["2", ".4%"]) == "2.4%"
    assert join_cue_text(["400", ",", "000"]) == "400,000"
    assert join_cue_text(["Hello", "there."]) == "Hello there."
    words = [Word("400", 0.0, 0.2), Word(",000", 0.2, 0.4), Word("and", 0.5, 0.7), Word("2", 0.8, 0.9), Word(".4%", 0.9, 1.1)]
    assert build_cues([(whole(0, 5_000), words)])[0].text == "400,000 and 2.4%"
    words = [Word("Hello", 0.0, 0.4), Word("there.", 0.5, 0.9), Word("Next", 1.0, 1.3), Word("one?", 1.4, 1.8)]
    assert build_cues([(whole(10_000, 20_000), words)]) == [
        Cue("c0001", 10_000, 10_900, "Hello there."),
        Cue("c0002", 11_000, 11_800, "Next one?"),
    ]


def test_cjk_full_stop_closes_a_cue():
    words = [Word("今日は", 0.0, 0.5), Word("晴れ。", 0.5, 1.0), Word("明日", 1.1, 1.5)]
    assert [c.text for c in build_cues([(whole(0, 5_000), words)], language="ja")] == ["今日は 晴れ。", "明日"]


def test_english_does_not_close_on_an_ideographic_full_stop():
    words = [Word("晴れ。", 0.0, 0.5), Word("next", 0.6, 1.0)]
    assert [c.text for c in build_cues([(whole(0, 5_000), words)])] == ["晴れ。 next"]


def test_a_cue_closes_after_twelve_seconds_without_punctuation():
    words = [Word(f"w{i}", float(i), i + 0.5) for i in range(14)]
    cues = build_cues([(whole(0, 20_000), words)])
    assert [(c.start_ms, c.end_ms) for c in cues] == [(0, 11_500), (12_000, 13_500)]


def test_a_pause_of_300_ms_inside_a_sentence_closes_a_built_in_cue():
    spans = [(0, 500), (800, 1_300)]
    cues = build_cues([(whole(0, 2_000), timed(spans))], quiet=Quiet(speech(2_000, spans)))
    assert cues == [Cue("c0001", 0, 500, "w0"), Cue("c0002", 800, 1_300, "w1")]


def test_a_pause_of_100_ms_keeps_one_cue():
    spans = [(0, 500), (600, 1_100)]
    cues = build_cues([(whole(0, 2_000), timed(spans))], quiet=Quiet(speech(2_000, spans)))
    assert cues == [Cue("c0001", 0, 1_100, "w0 w1")]


def test_the_twelve_second_limit_closes_after_the_longest_pause():
    before = [(400 * i, 400 * i + 400) for i in range(13)]
    after = [(5_400 + 400 * i, 5_800 + 400 * i) for i in range(22)]
    spans = before + after
    cues = build_cues([(whole(0, 15_000), timed(spans))], quiet=Quiet(speech(15_000, spans)))
    assert [(c.start_ms, c.end_ms) for c in cues] == [(0, 5_200), (5_400, 14_200)], (
        "the first cue ends at the word before the 200 ms pause, not at the word that crossed 12 s"
    )


def test_a_cue_closes_where_a_speaker_region_ends():
    first = (whole(0, 2_000, region=0, index=0), [Word("so", 0.0, 0.5), Word("anyway", 0.6, 1.0)])
    second = (whole(2_000, 4_000, region=1, index=1), [Word("right", 0.1, 0.5)])
    assert [c.text for c in build_cues([first, second])] == ["so anyway", "right"]


def test_a_word_belongs_to_the_slice_where_it_starts():
    early = Slice(0, 0, 27_000, 0)
    late = Slice(1, 27_000, 40_000, 0)
    words_early = [Word("kept.", 26.0, 26.5), Word("repeated.", 27.2, 27.6)]
    words_late = [Word("repeated.", 0.2, 0.6), Word("end.", 1.0, 1.5)]
    cues = build_cues([(early, words_early), (late, words_late)])
    assert [(c.text, c.start_ms) for c in cues] == [("kept.", 26_000), ("repeated.", 27_200), ("end.", 28_000)]


def test_zero_length_words_get_one_millisecond_and_blank_words_are_skipped():
    cues = build_cues([(whole(0, 5_000), [Word(" ", 0.0, 0.1), Word("Hi.", 1.0, 1.0)])])
    assert cues == [Cue("c0001", 1_000, 1_001, "Hi.")]


def make_rows():
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
    return cue_rows(cues, tracks)


def test_group_ids_come_from_the_first_cue_and_every_row_has_its_nearest_group():
    rows = make_rows()
    assert [r["groupId"] for r in rows] == ["gc0001", "gc0002", "ungrouped", "gc0001"]
    assert rows[0] == {
        "id": "c0001", "startMs": 0, "endMs": 1_000, "text": "Hi.",
        "groupId": "gc0001", "speaker": "gc0001", "nearestGroupId": "gc0002", "nearestCosine": 0.6,
    }
    assert (rows[1]["nearestGroupId"], rows[1]["nearestCosine"]) == ("gc0001", 0.6)
    assert (rows[2]["nearestGroupId"], rows[2]["nearestCosine"]) == (None, None)
    assert rows[3]["nearestGroupId"] == "gc0002"


def test_group_table_sums_cue_durations():
    assert group_table(make_rows()) == [
        {"groupId": "gc0001", "speakingMs": 2_000, "cueCount": 2, "nearestGroupId": "gc0002", "nearestCosine": 0.6},
        {"groupId": "gc0002", "speakingMs": 1_000, "cueCount": 1, "nearestGroupId": "gc0001", "nearestCosine": 0.6},
        {"groupId": "ungrouped", "speakingMs": 1_000, "cueCount": 1, "nearestGroupId": None, "nearestCosine": None},
    ]


def test_one_group_has_no_nearest():
    rows = cue_rows([Cue("c0001", 0, 900, "a")], Tracks([Turn(0, 1_000, 0)], [[1.0, 0.0]]))
    assert (rows[0]["groupId"], rows[0]["nearestGroupId"], rows[0]["nearestCosine"]) == ("gc0001", None, None)


def test_vtt_has_ids_voice_tags_and_escaped_text():
    rows = make_rows()
    rows[1]["text"] = "a < b & c"
    assert render_vtt(rows) == (
        "WEBVTT\n\n"
        "c0001\n00:00:00.000 --> 00:00:01.000\n<v gc0001>Hi.\n\n"
        "c0002\n00:00:01.000 --> 00:00:02.000\n<v gc0002>a &lt; b &amp; c\n\n"
        "c0003\n00:00:02.000 --> 00:00:03.000\nSilence?\n\n"
        "c0004\n00:00:05.000 --> 00:00:06.000\n<v gc0001>Back.\n"
    )


def test_stamp_formats_hours():
    assert stamp(3_723_004) == "01:02:03.004"


def test_a_short_cue_between_one_speaker_and_another_voice_is_uncertain():
    """The guest turn covers the whole short cue, so it would join the guest. The host turn is also in it."""
    cues = [
        Cue("c0001", 0, 4_000, "Guest starts."),
        Cue("c0002", 4_000, 4_800, "Right."),
        Cue("c0003", 4_800, 9_000, "Guest continues."),
    ]
    tracks = Tracks(
        turns=[Turn(0, 9_000, 0), Turn(4_000, 4_800, 1)],
        centroids=[[1.0, 0.0], [0.0, 1.0]],
    )
    rows = cue_rows(cues, tracks)
    assert [row["groupId"] for row in rows] == ["gc0001", "uncertain", "gc0001"]
    assert rows[1]["speaker"] == "uncertain"
    assert rows[1]["nearestGroupId"] is None and rows[1]["nearestCosine"] is None
    assert rows[0]["speaker"] == "gc0001" and rows[2]["speaker"] == "gc0001"


def test_a_short_cue_of_the_same_voice_stays_with_that_group():
    cues = [
        Cue("c0001", 0, 4_000, "A."),
        Cue("c0002", 4_000, 5_500, "B."),
        Cue("c0003", 5_500, 9_000, "C."),
    ]
    tracks = Tracks(turns=[Turn(0, 9_000, 0)], centroids=[[1.0, 0.0]])
    rows = cue_rows(cues, tracks)
    assert [row["speaker"] for row in rows] == ["gc0001", "gc0001", "gc0001"]
    assert [row["groupId"] for row in rows] == ["gc0001", "gc0001", "gc0001"]
