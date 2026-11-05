"""Validate an agent-written summary against the cues it cites."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

UNSET_LABELS = {"", "undetermined"}


def load(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _is_ms(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _labels(value: object, errors: list[str]) -> dict[str, object]:
    """`speakers` as {id: label}. Accepts a mapping or a list of {id, label}."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): label for key, label in value.items()}
    if isinstance(value, list):
        labels: dict[str, object] = {}
        for position, item in enumerate(value, 1):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                errors.append(f"speakers[{position}] must be an object with id and label")
                continue
            labels[item["id"]] = item.get("label")
        return labels
    errors.append("speakers must be a mapping of speaker id to label, or a list of {id, label}")
    return {}


def check(summary: object, cues_doc: object) -> list[str]:
    """Every problem found, as one message each. An empty list means the summary is valid."""
    if not isinstance(cues_doc, dict) or not isinstance(cues_doc.get("cues"), list):
        return ["cues file has no cues list; pass the .cues.json that podcue wrote"]
    order = {cue["id"]: index for index, cue in enumerate(cues_doc["cues"])}
    cues = cues_doc["cues"]
    known_speakers = {s["id"] for s in cues_doc.get("speakers", [])}

    if not isinstance(summary, dict):
        return ["summary must be a mapping with a paragraphs list"]
    errors: list[str] = []
    if "summary" in summary and (not isinstance(summary["summary"], str) or not summary["summary"].strip()):
        errors.append("summary must be a non-empty string when present")
    labels = _labels(summary.get("speakers"), errors)
    for sid in labels:
        if sid not in known_speakers:
            errors.append(f"speakers: {sid} is not a speaker in the cues file")

    paragraphs = summary.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        errors.append("paragraphs must be a non-empty list")
        return errors

    referenced: dict[str, int] = {}
    for number, paragraph in enumerate(paragraphs, 1):
        where = f"paragraph {number}"
        if not isinstance(paragraph, dict):
            errors.append(f"{where}: must be a mapping")
            continue
        text = paragraph.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{where}: text must be a non-empty string")

        cited = paragraph.get("cues")
        if not isinstance(cited, list) or not cited or not all(isinstance(c, str) for c in cited):
            errors.append(f"{where}: cues must be a non-empty list of cue ids")
            cited = []
        seen: set[str] = set()
        found: list[int] = []
        for cue_id in cited:
            if cue_id in seen:
                errors.append(f"{where}: cue {cue_id} is listed twice")
                continue
            seen.add(cue_id)
            if cue_id not in order:
                errors.append(f"{where}: cue {cue_id} is not in the cues file")
                continue
            found.append(order[cue_id])

        for key in ("startMs", "endMs"):
            if not _is_ms(paragraph.get(key)):
                errors.append(f"{where}: {key} must be a number")
        if found and len(found) == len(seen):
            first, last = cues[min(found)], cues[max(found)]
            start, end = paragraph.get("startMs"), paragraph.get("endMs")
            if _is_ms(start) and start != first["startMs"]:
                errors.append(f"{where}: startMs {start} does not match {first['id']} startMs {first['startMs']}")
            if _is_ms(end) and end != last["endMs"]:
                errors.append(f"{where}: endMs {end} does not match {last['id']} endMs {last['endMs']}")

        for index in found:
            speaker = cues[index].get("speaker")
            if speaker is not None:
                referenced.setdefault(speaker, number)
        listed = paragraph.get("speakers", [])
        if not isinstance(listed, list) or not all(isinstance(s, str) for s in listed):
            errors.append(f"{where}: speakers must be a list of speaker ids")
            listed = []
        for speaker in listed:
            if speaker not in known_speakers:
                errors.append(f"{where}: speaker {speaker} is not in the cues file")
            else:
                referenced.setdefault(speaker, number)

    for speaker, number in sorted(referenced.items(), key=lambda item: item[1]):
        label = labels.get(speaker)
        if not isinstance(label, str) or label.strip().lower() in UNSET_LABELS or label.strip() == speaker:
            errors.append(f"speaker {speaker} (first cited in paragraph {number}) has no human label in speakers")
    return errors
