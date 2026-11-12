"""Narration numerals become English words. Cue text is not passed through this."""

import pytest

from podcue.numbers import say_numbers


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("400,000", "four hundred thousand"),
        ("2.4%", "two point four percent"),
        ("$1.5 billion", "one point five billion dollars"),
        ("1996", "nineteen ninety-six"),
        ("3rd", "third"),
        ("10x", "ten times"),
        ("-12", "minus twelve"),
        ("-2.4%", "minus two point four percent"),
        ("2-3", "two to three"),
        ("6:30", "six thirty"),
        ("Hello there.", "Hello there."),
        ("cities and homes", "cities and homes"),
        ("COVID-19", "COVID-19"),
    ],
)
def test_narration_numbers_are_spoken_and_words_stay(written, spoken):
    assert say_numbers(written) == spoken


def test_a_sentence_keeps_its_words_and_reads_the_figures():
    written = "In 1996 the city added 400,000 homes, up 2.4%."
    assert say_numbers(written) == (
        "In nineteen ninety-six the city added four hundred thousand homes, up two point four percent."
    )


def test_other_languages_are_left_unchanged():
    assert say_numbers("1996年に2.4%", language="ja") == "1996年に2.4%"
