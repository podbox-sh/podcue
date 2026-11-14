"""Slice skips and the transcription checkpoint. No model weights."""

import numpy as np
import pytest

from podcue.cues import Word
from podcue.diarize import take_mask
from podcue.pipeline import run_slices
from podcue.regions import Slice
from podcue.transcribe import transcribe_slice


def _plan():
    return [
        Slice(0, 0, 1_000, 0, 1_000, 0),
        Slice(1, 1_000, 2_000, 1_000, 2_000, 0),
        Slice(2, 2_000, 2_400, 2_000, 2_400, 0),
    ]


def _samples():
    return np.zeros(16_000 * 3, dtype=np.float32)


class _Scripted:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def transcribe(self, samples):
        self.calls.append(samples.size)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _alignment_crash():
    """The faster-whisper expression: pad an empty diff, then mask an empty time array."""
    time_indices = np.array([], dtype=np.int64)
    jumps = np.pad(np.diff(np.array([], dtype=np.int64)), (1, 0), constant_values=1).astype(bool)
    return time_indices[jumps]


def test_the_empty_alignment_mask_is_the_reported_crash():
    time_indices = np.array([], dtype=np.int64)
    jumps = np.pad(np.diff(np.array([], dtype=np.int64)), (1, 0), constant_values=1).astype(bool)
    assert time_indices.shape == (0,) and jumps.shape == (1,)
    with pytest.raises(IndexError, match="boolean index did not match"):
        time_indices[jumps]


def test_a_slice_with_that_mismatch_is_skipped():
    class Boom:
        def transcribe(self, samples):
            return _alignment_crash()

    outcome = transcribe_slice(Boom(), np.zeros(8_000, dtype=np.float32))
    assert outcome.state == "skipped-empty" and outcome.words == ()


def test_empty_short_and_noise_slices_are_not_fatal():
    seen = []

    class Record:
        def transcribe(self, samples):
            seen.append(samples.size)
            return []

    empty = np.zeros(0, dtype=np.float32)
    short = np.zeros(8_000, dtype=np.float32)  # 0.5 s
    tone = np.sin(2 * np.pi * 440 * np.arange(16_000) / 16_000).astype(np.float32)
    noise = np.random.default_rng(0).standard_normal(16_000).astype(np.float32) * 0.1
    assert transcribe_slice(Record(), empty).state == "skipped-empty"
    assert transcribe_slice(Record(), short).state == "done"
    assert transcribe_slice(Record(), tone).state == "done"
    assert transcribe_slice(Record(), noise).state == "done"
    assert seen == [8_000, 16_000, 16_000]


def test_speech_without_word_times_skips_the_slice():
    class Bare:
        def transcribe(self, samples):
            from podcue.transcribe import whisper_words
            return whisper_words([{"text": "hello", "words": []}])

    outcome = transcribe_slice(Bare(), np.zeros(16_000, dtype=np.float32))
    assert outcome.state == "skipped-empty" and outcome.words == ()


def test_run_slices_skips_an_empty_span_and_the_alignment_crash(tmp_path, capsys):
    plan = [
        Slice(0, 0, 0, 0, 0, 0),
        Slice(1, 0, 1_000, 0, 1_000, 0),
    ]

    class Boom:
        def transcribe(self, samples):
            return _alignment_crash()

    results = run_slices(np.zeros(16_000, dtype=np.float32), plan, Boom(), tmp_path / "slices.json")
    assert results == [(plan[0], []), (plan[1], [])]
    err = capsys.readouterr().err
    assert "skipping slice 1/2 (no speech to align)" in err
    assert "skipping slice 2/2 (no speech to align)" in err


def test_other_errors_still_fail_the_slice():
    class Boom:
        def transcribe(self, samples):
            raise RuntimeError("killed")

    with pytest.raises(RuntimeError, match="killed"):
        transcribe_slice(Boom(), np.zeros(16_000, dtype=np.float32))


def test_a_mask_of_the_wrong_length_is_empty_instead_of_fatal():
    values = np.array([], dtype=np.float32)
    mask = np.array([True])
    assert take_mask(values, mask).shape == (0,)
    assert take_mask(np.array([1.0, 2.0]), np.array([True, False])).tolist() == [1.0]


def test_a_crash_mid_run_resumes_without_redoing_finished_slices(tmp_path, capsys):
    plan = _plan()
    samples = _samples()
    checkpoint = tmp_path / "slices.json"
    first = _Scripted([
        [Word("one", 0.1, 0.2)],
        RuntimeError("killed"),
        [Word("three", 0.0, 0.1)],
    ])
    with pytest.raises(RuntimeError, match="killed"):
        run_slices(samples, plan, first, checkpoint)
    assert first.calls == [16_000, 16_000]
    stored = checkpoint.read_text()
    assert '"index": 0' in stored and '"index": 1' not in stored

    second = _Scripted([
        [Word("two", 0.0, 0.2)],
        [Word("three", 0.0, 0.1)],
    ])
    results = run_slices(samples, plan, second, checkpoint)
    assert second.calls == [16_000, 6_400]
    assert results[0][1] == [Word("one", 0.1, 0.2)]
    assert results[1][1] == [Word("two", 0.0, 0.2)]
    assert "already finished" in capsys.readouterr().err


def test_a_skipped_slice_is_checkpointed_and_not_retried(tmp_path, capsys):
    plan = _plan()[:2]
    checkpoint = tmp_path / "slices.json"
    crash = IndexError(
        "boolean index did not match indexed array along axis 0; "
        "size of axis is 0 but size of corresponding boolean axis is 1"
    )
    first = _Scripted([crash, [Word("two", 0.0, 0.1)]])
    results = run_slices(_samples(), plan, first, checkpoint, audio_id="ep_test", language="en")
    assert results[0][1] == []
    assert "skipping slice 1/2" in capsys.readouterr().err
    second = _Scripted([[Word("nope", 0.0, 0.1)]])
    again = run_slices(_samples(), plan, second, checkpoint, audio_id="ep_test", language="en")
    assert second.calls == []
    assert again[0][1] == []
    assert again[1][1] == [Word("two", 0.0, 0.1)]


def test_a_checkpoint_for_other_audio_is_not_resumed(tmp_path):
    plan = _plan()[:1]
    checkpoint = tmp_path / "slices.json"
    run_slices(_samples(), plan, _Scripted([[Word("one", 0.1, 0.2)]]), checkpoint, audio_id="ep_a", language="en")
    again = _Scripted([[Word("two", 0.0, 0.1)]])
    results = run_slices(_samples(), plan, again, checkpoint, audio_id="ep_b", language="en")
    assert again.calls == [16_000]
    assert results[0][1] == [Word("two", 0.0, 0.1)]


def test_degenerate_slices_run_through_without_aborting(tmp_path):
    """Empty, silence, sub-second, tone, noise, clipped, and an end-of-file boundary."""
    rate = 16_000
    rng = np.random.default_rng(0)
    silence = np.zeros(rate // 2, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * np.arange(rate) / rate).astype(np.float32)
    noise = (rng.standard_normal(rate) * 0.1).astype(np.float32)
    clipped = np.clip(tone * 4, -1, 1)
    pieces = [np.zeros(0, dtype=np.float32), silence, tone[:800], tone, noise, clipped]
    samples = np.concatenate(pieces)
    plan = []
    cursor = 0
    for index, piece in enumerate(pieces):
        start = cursor * 1000 // rate
        cursor += len(piece)
        end = cursor * 1000 // rate
        plan.append(Slice(index, start, end, start, end, 0))
    # A slice whose end is the last sample.
    plan.append(Slice(len(plan), (len(samples) - 160) * 1000 // rate, len(samples) * 1000 // rate,
                      (len(samples) - 160) * 1000 // rate, len(samples) * 1000 // rate, 0))

    class Touchy:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio):
            self.calls += 1
            if audio.size < 200:
                raise IndexError(
                    "boolean index did not match indexed array along axis 0; "
                    "size of axis is 0 but size of corresponding boolean axis is 1"
                )
            return [Word("ok", 0.0, 0.01)]

    model = Touchy()
    results = run_slices(samples, plan, model, tmp_path / "slices.json", audio_id="ep_fuzz", language="en")
    assert len(results) == len(plan)
    assert model.calls == len(plan) - 1  # the empty piece never reaches the model
    again = Touchy()
    rerun = run_slices(samples, plan, again, tmp_path / "slices.json", audio_id="ep_fuzz", language="en")
    assert again.calls == 0
    assert [word.text for _, words in rerun for word in words] == [word.text for _, words in results for word in words]
