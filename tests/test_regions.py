from podcue.diarize import Turn
from podcue.regions import SLICE_MS, STEP_MS, Slice, plan_regions


def spans(slices):
    return [(s.start_ms, s.end_ms) for s in slices]


def test_joins_same_speaker_across_a_short_gap():
    turns = [Turn(0, 2_000, 0), Turn(2_400, 5_000, 0)]
    assert spans(plan_regions(turns)) == [(0, 5_000)]


def test_keeps_same_speaker_apart_across_a_long_gap():
    turns = [Turn(0, 2_000, 0), Turn(2_600, 5_000, 0)]
    assert spans(plan_regions(turns)) == [(0, 2_000), (2_600, 5_000)]


def test_does_not_join_different_speakers():
    turns = [Turn(0, 2_000, 0), Turn(2_100, 5_000, 1)]
    plan = plan_regions(turns)
    assert spans(plan) == [(0, 2_000), (2_100, 5_000)]
    assert [s.region for s in plan] == [0, 1]


def test_short_turn_joins_the_previous_region():
    turns = [Turn(0, 2_000, 0), Turn(2_100, 2_400, 1), Turn(4_000, 6_000, 0)]
    assert spans(plan_regions(turns)) == [(0, 2_400), (4_000, 6_000)]


def test_short_turn_joins_the_next_region_when_the_previous_is_far():
    turns = [Turn(0, 2_000, 0), Turn(3_000, 3_300, 1), Turn(3_500, 6_000, 0)]
    assert spans(plan_regions(turns)) == [(0, 2_000), (3_000, 6_000)]


def test_isolated_short_turn_stays():
    turns = [Turn(0, 300, 0)]
    assert spans(plan_regions(turns)) == [(0, 300)]


def test_long_region_is_cut_with_overlap_and_keep_windows():
    turns = [Turn(1_000, 61_000, 0)]
    plan = plan_regions(turns)
    assert plan == [
        Slice(0, 1_000, 1_000 + SLICE_MS, 1_000, 1_000 + STEP_MS, 0),
        Slice(1, 1_000 + STEP_MS, 1_000 + STEP_MS + SLICE_MS, 1_000 + STEP_MS, 1_000 + 2 * STEP_MS, 0),
        Slice(2, 1_000 + 2 * STEP_MS, 61_000, 1_000 + 2 * STEP_MS, 61_000, 0),
    ]


def test_region_of_exactly_one_slice_is_not_cut():
    plan = plan_regions([Turn(0, SLICE_MS, 0)])
    assert plan == [Slice(0, 0, SLICE_MS, 0, SLICE_MS, 0)]


def test_no_turns_no_slices():
    assert plan_regions([]) == []
