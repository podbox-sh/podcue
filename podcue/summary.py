"""Summary files: `summaries/sm_<16 hex>.yaml`, written by the summary commands and validated by `check`.

The commands fill `startMs`/`endMs` from the cited cues and store `cues` in transcript order,
so a file they write always passes the cue and time checks. `check` applies the same rules,
plus the label rule, to any file, including hand-written ones.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

import yaml

from .cues import UNGROUPED
from .groups import is_labeled
from .store import SUMMARY_ID

PARTS = ("news", "explanation", "advice", "claim", "discussion", "story", "items")
FILE_FIELDS = ("version", "episodeId", "summaryId", "summary", "paragraphs")
PARAGRAPH_FIELDS = ("index", "part", "text", "cues", "startMs", "endMs")
POST_SHAPE = "postParagraph expects { text, cues } or an array of them"
PUT_SHAPE = "putParagraph expects { index, text, cues }"


class Invalid(Exception):
    """A request the summary commands refuse. Nothing is written."""


def new_summary_id() -> str:
    return f"sm_{secrets.token_hex(8)}"


def empty(episode_id: str, summary_id: str) -> dict:
    return {"version": 1, "episodeId": episode_id, "summaryId": summary_id, "summary": "", "paragraphs": []}


def _is_ms(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _paragraph(index: int, part: str | None, text: str, cues: list[str], start: int, end: int) -> dict:
    out: dict = {"index": index}
    if part is not None:
        out["part"] = part
    out.update(text=text, cues=cues, startMs=start, endMs=end)
    return out


def order_cues(ids: list[str], start: object, end: object, order: dict[str, int], rows: list[dict]):
    """The cited ids in transcript order, with the first cue's startMs and the last cue's endMs.

    A time the caller sent must equal the one computed here.
    """
    seen: set[str] = set()
    for cue_id in ids:
        if cue_id in seen:
            raise Invalid(f"a cue is listed twice: {cue_id}")
        seen.add(cue_id)
        if cue_id not in order:
            raise Invalid(f"a cue is not in the transcript: {cue_id}")
    ordered = sorted(ids, key=order.__getitem__)
    first, last = rows[order[ordered[0]]], rows[order[ordered[-1]]]
    if start is not None and start != first["startMs"]:
        raise Invalid(f"startMs does not match the cue: {first['id']} starts at {first['startMs']}")
    if end is not None and end != last["endMs"]:
        raise Invalid(f"endMs does not match the cue: {last['id']} ends at {last['endMs']}")
    return ordered, first["startMs"], last["endMs"]


def read_body(body: object, *, put: bool) -> dict:
    """One paragraph request: text, cues, optional part/startMs/endMs, and index for put only."""
    shape = PUT_SHAPE if put else POST_SHAPE
    if not isinstance(body, dict):
        raise Invalid(shape)
    if put:
        index = body.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise Invalid("putParagraph requires index")
    elif "index" in body:
        raise Invalid("postParagraph rejects index")
    allowed = {"index", "part", "text", "cues", "startMs", "endMs"} if put else {"part", "text", "cues", "startMs", "endMs"}
    if set(body) - allowed:
        raise Invalid(shape)
    text, cues = body.get("text"), body.get("cues")
    if not isinstance(text, str) or not text:
        raise Invalid(shape)
    if not isinstance(cues, list) or not cues or not all(isinstance(c, str) and c for c in cues):
        raise Invalid(shape)
    for key in ("startMs", "endMs"):
        if key in body and not _is_ms(body[key]):
            raise Invalid(f"{key} must be a number")
    part = body.get("part")
    if "part" in body and part not in PARTS:
        raise Invalid(f"part must be one of {', '.join(PARTS)}")
    return body


def build_paragraph(body: dict, index: int, rows: list[dict], order: dict[str, int]) -> dict:
    cues, start, end = order_cues(body["cues"], body.get("startMs"), body.get("endMs"), order, rows)
    return _paragraph(index, body.get("part"), body["text"], cues, start, end)


def post(document: dict, bodies: object, rows: list[dict]) -> dict:
    """Append one paragraph, or an array of them in order. All are checked before any is kept."""
    items = bodies if isinstance(bodies, list) else [bodies]
    if not items:
        raise Invalid(POST_SHAPE)
    order = {row["id"]: i for i, row in enumerate(rows)}
    paragraphs = list(document["paragraphs"])
    for position, body in enumerate(items, 1):
        try:
            paragraphs.append(build_paragraph(read_body(body, put=False), len(paragraphs) + 1, rows, order))
        except Invalid as error:
            raise Invalid(f"paragraph {position}: {error}" if len(items) > 1 else str(error)) from None
    return {**document, "paragraphs": paragraphs}


def put(document: dict, body: object, rows: list[dict]) -> dict:
    body = read_body(body, put=True)
    index = body["index"]
    if not any(p["index"] == index for p in document["paragraphs"]):
        raise Invalid("putParagraph index does not exist")
    order = {row["id"]: i for i, row in enumerate(rows)}
    replaced = build_paragraph(body, index, rows, order)
    return {**document, "paragraphs": [replaced if p["index"] == index else p for p in document["paragraphs"]]}


def set_summary(document: dict, summary: object) -> dict:
    if not isinstance(summary, str) or not summary:
        raise Invalid("summary must be a non-empty string")
    if not document["paragraphs"]:
        raise Invalid("write the paragraphs before the episode summary")
    return {**document, "summary": summary}


def load(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def check(
    document: object,
    episode_id: str,
    rows: list[dict],
    labels: list[dict],
    summary_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """(problems, warnings). Any problem fails the file. A warning names a cited `undetermined` group."""
    if not isinstance(document, dict):
        return ["the summary must be a mapping with version, episodeId, summary and paragraphs"], []
    problems: list[str] = []
    extra = sorted(set(document) - set(FILE_FIELDS))
    if extra:
        problems.append(f"unknown keys: {', '.join(extra)}")
    if document.get("version") != 1:
        problems.append("version must be 1")
    if document.get("episodeId") != episode_id:
        problems.append(f"episodeId must be {episode_id}")
    found_id = document.get("summaryId")
    if summary_id is not None and found_id != summary_id:
        problems.append(f"summaryId must be {summary_id}")
    elif found_id is not None and (not isinstance(found_id, str) or not SUMMARY_ID.match(found_id)):
        problems.append("summaryId must be sm_ followed by 16 hex digits")
    summary = document.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        problems.append("summary must be a non-empty string")
    paragraphs = document.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        problems.append("paragraphs must be a non-empty list")
        return problems, []

    order = {row["id"]: i for i, row in enumerate(rows)}
    label_of = {label["groupId"]: label["label"] for label in labels}
    cited_groups: dict[str, int] = {}
    for number, paragraph in enumerate(paragraphs, 1):
        where = f"paragraph {number}"
        if not isinstance(paragraph, dict):
            problems.append(f"{where}: must be a mapping")
            continue
        extra = sorted(set(paragraph) - set(PARAGRAPH_FIELDS))
        if extra:
            problems.append(f"{where}: unknown keys: {', '.join(extra)}")
        if paragraph.get("index") != number:
            problems.append(f"{where}: index must be {number}")
        if "part" in paragraph and paragraph["part"] not in PARTS:
            problems.append(f"{where}: part must be one of {', '.join(PARTS)}")
        text = paragraph.get("text")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"{where}: text must be a non-empty string")
        cues = paragraph.get("cues")
        if not isinstance(cues, list) or not cues or not all(isinstance(c, str) for c in cues):
            problems.append(f"{where}: cues must be a non-empty list of cue ids")
            cues = []
        known: list[str] = []
        for cue_id in cues:
            if cue_id in known:
                problems.append(f"{where}: cue {cue_id} is listed twice")
            elif cue_id not in order:
                problems.append(f"{where}: cue {cue_id} is not in the transcript")
            else:
                known.append(cue_id)
                group_id = rows[order[cue_id]]["groupId"]
                if group_id != UNGROUPED:
                    cited_groups.setdefault(group_id, number)
        if known != sorted(known, key=order.__getitem__):
            problems.append(f"{where}: cues must be in transcript order")
        for key in ("startMs", "endMs"):
            if not _is_ms(paragraph.get(key)):
                problems.append(f"{where}: {key} must be a number")
        if known and len(known) == len(cues):
            first = rows[min(order[c] for c in known)]
            last = rows[max(order[c] for c in known)]
            start, end = paragraph.get("startMs"), paragraph.get("endMs")
            if _is_ms(start) and start != first["startMs"]:
                problems.append(f"{where}: startMs {start} does not match {first['id']} startMs {first['startMs']}")
            if _is_ms(end) and end != last["endMs"]:
                problems.append(f"{where}: endMs {end} does not match {last['id']} endMs {last['endMs']}")

    warnings: list[str] = []
    for group_id, number in sorted(cited_groups.items(), key=lambda item: (item[1], item[0])):
        label = label_of.get(group_id)
        if label is None:
            problems.append(f"group {group_id} (first cited in paragraph {number}) has no label in labels.json")
        elif not is_labeled(label):
            warnings.append(f"group {group_id} (first cited in paragraph {number}) is undetermined")
    return problems, warnings
