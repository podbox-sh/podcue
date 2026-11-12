"""Read numerals as English words before narration.

Cue text and transcripts are not passed through here. Kokoro narration is English
(`lang_code="a"`), so other languages are left unchanged.
"""

from __future__ import annotations

import re

from num2words import num2words

_SCALE = re.compile(r"\s+(thousand|million|billion|trillion)\b", re.IGNORECASE)
_TOKEN = re.compile(
    r"""
    (?P<timerange>\b\d{1,2}:\d{2}\s*[-–—]\s*\d{1,2}:\d{2}\b)
    |(?P<time>\b\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?)
    |(?P<money>-?\$\s*-?\d[\d,]*(?:\.\d+)?|\$\s*-?\d[\d,]*(?:\.\d+)?)
    |(?P<percent>-?\d[\d,]*(?:\.\d+)?\s*%)
    |(?P<ordinal>\b\d+(?:st|nd|rd|th)\b)
    |(?P<mult>\b\d+(?:\.\d+)?x\b)
    |(?P<range>\b\d[\d,]*(?:\.\d+)?\s*[-–—]\s*\d[\d,]*(?:\.\d+)?\b)
    |(?P<number>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def say_numbers(text: str, language: str = "en") -> str:
    """Replace numerals with spoken English. A non-English `language` is returned as given."""
    if not language.lower().startswith("en"):
        return text
    parts: list[str] = []
    cursor = 0
    for match in _TOKEN.finditer(text):
        if _stuck_to_a_word(text, match.start()):
            continue
        end = match.end()
        scale = None
        if match.lastgroup == "money":
            found = _SCALE.match(text, end)
            if found is not None:
                scale = found.group(1)
                end = found.end()
        spoken = _speak(match, scale)
        if spoken is None:
            continue
        parts.append(text[cursor:match.start()])
        parts.append(spoken)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _stuck_to_a_word(text: str, start: int) -> bool:
    """A number glued to letters, as in COVID-19, stays a word."""
    if start <= 0:
        return False
    if text[start - 1].isalpha():
        return True
    return start >= 2 and text[start - 1] == "-" and text[start - 2].isalpha()


def _speak(match: re.Match[str], scale: str | None) -> str | None:
    kind = match.lastgroup
    raw = match.group(0)
    if kind == "timerange":
        left, right = re.split(r"\s*[-–—]\s*", raw, maxsplit=1)
        start, stop = _clock(left), _clock(right)
        if start is None or stop is None:
            return None
        return f"{start} to {stop}"
    if kind == "time":
        return _clock(raw)
    if kind == "money":
        return _money(raw, scale)
    if kind == "percent":
        return f"{_amount(raw[:-1].strip())} percent"
    if kind == "ordinal":
        return num2words(int(raw[:-2]), to="ordinal", lang="en")
    if kind == "mult":
        return f"{_amount(raw[:-1])} times"
    if kind == "range":
        left, right = re.split(r"\s*[-–—]\s*", raw, maxsplit=1)
        return f"{_amount(left)} to {_amount(right)}"
    if kind == "number":
        return _amount(raw)
    return None


def _amount(raw: str) -> str:
    negative = raw.startswith("-")
    body = raw[1:] if negative else raw
    body = body.replace(",", "")
    if not negative and re.fullmatch(r"\d{4}", body):
        year = int(body)
        if 1000 <= year <= 2099:
            return num2words(year, to="year", lang="en")
    if "." in body:
        value = float(body)
    else:
        value = int(body)
    if negative:
        value = -value
    return num2words(value, lang="en")


def _clock(raw: str) -> str | None:
    found = re.fullmatch(r"(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)?", raw, re.IGNORECASE)
    if found is None:
        return None
    hour, minute = int(found.group(1)), int(found.group(2))
    if hour > 23 or minute > 59:
        return None
    hour_words = num2words(hour, lang="en")
    if minute == 0:
        spoken = f"{hour_words} o'clock"
    elif minute < 10:
        spoken = f"{hour_words} oh {num2words(minute, lang='en')}"
    else:
        spoken = f"{hour_words} {num2words(minute, lang='en')}"
    suffix = found.group(3)
    if suffix is None:
        return spoken
    marker = "a.m." if suffix.lower().startswith("a") else "p.m."
    return f"{spoken} {marker}"


def _money(raw: str, scale: str | None) -> str:
    negative = "-" in raw
    body = raw.replace("$", "").replace(",", "").replace("-", "").strip()
    value = float(body) if "." in body else int(body)
    if negative:
        value = -value
    words = num2words(value, lang="en")
    if scale is not None:
        words = f"{words} {scale.lower()}"
    return f"{words} dollars"
