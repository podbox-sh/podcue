"""Speaker-group labels and the paged cue view.

A label is a person's name, `undetermined`, `quote` or `ending`. Only a person's name counts
toward `labeledShare`. `undeterminedShare` is the cue time still marked `undetermined`.

`label-groups` can name one group, or every group whose cues sit entirely inside a cue id
range. Afterward each cue's `speaker` is that label. Before any label it is the group id.
"""

from __future__ import annotations

from collections.abc import Callable

from .cues import HELD_SPEAKERS, UNGROUPED

MARKS = ("undetermined", "quote", "ending")
DEFAULT_LIMIT = 50
MAX_LIMIT = 200
LABEL_SHAPE = (
    "label-groups expects a JSON array of { groupId, label } "
    "or { fromCue, toCue, label, scope? } where scope is groups (default) or cues"
)


class Unknown(Exception):
    """An id the episode does not have. Reported as `{"state": "unknown", ...}`, not as an error."""

    def __init__(self, **fields: str):
        super().__init__(fields)
        self.fields = fields


class BadRange(Exception):
    """`fromCue` sits after `toCue` in the episode. Nothing is written."""

    def __init__(self, from_cue: str, to_cue: str):
        super().__init__(f"fromCue {from_cue} is after toCue {to_cue}")
        self.from_cue = from_cue
        self.to_cue = to_cue


def _label_text(item: dict) -> str | None:
    label = item.get("label")
    if not isinstance(label, str) or not label:
        return None
    return label


def parse_label_updates(value: object) -> list[dict] | None:
    """Each entry names one group, or a cue range that expands to the groups inside it."""
    if not isinstance(value, list):
        return None
    updates = []
    for item in value:
        if not isinstance(item, dict):
            return None
        label = _label_text(item)
        if label is None:
            return None
        has_group = "groupId" in item
        has_range = "fromCue" in item or "toCue" in item
        if has_group == has_range:
            return None
        if has_group:
            if "scope" in item:
                return None
            group_id = item.get("groupId")
            if not isinstance(group_id, str) or not group_id:
                return None
            updates.append({"groupId": group_id, "label": label})
            continue
        from_cue, to_cue = item.get("fromCue"), item.get("toCue")
        if not isinstance(from_cue, str) or not from_cue or not isinstance(to_cue, str) or not to_cue:
            return None
        scope = item.get("scope", "groups")
        if scope not in ("groups", "cues"):
            return None
        updates.append({"fromCue": from_cue, "toCue": to_cue, "label": label, "scope": scope})
    return updates


def groups_wholly_inside(rows: list[dict], from_cue: str, to_cue: str) -> list[str]:
    """Groups whose every cue id lies in the inclusive range.

    A group with one cue outside the range is left alone. `uncertain` and `mixed` are not
    labels you can paint on with a range; those cues keep their speaker.
    """
    order = [row["id"] for row in rows]
    if from_cue not in order:
        raise Unknown(cueId=from_cue)
    if to_cue not in order:
        raise Unknown(cueId=to_cue)
    start, end = order.index(from_cue), order.index(to_cue)
    if start > end:
        raise BadRange(from_cue, to_cue)
    window = set(order[start : end + 1])
    members: dict[str, set[str]] = {}
    for row in rows:
        members.setdefault(row["groupId"], set()).add(row["id"])
    return sorted(
        group_id
        for group_id, cue_ids in members.items()
        if group_id not in HELD_SPEAKERS and cue_ids <= window
    )


def cue_ids_between(rows: list[dict], from_cue: str, to_cue: str) -> list[str]:
    """Cue ids from `from_cue` through `to_cue`, inclusive, in cue order."""
    order = [row["id"] for row in rows]
    if from_cue not in order:
        raise Unknown(cueId=from_cue)
    if to_cue not in order:
        raise Unknown(cueId=to_cue)
    start, end = order.index(from_cue), order.index(to_cue)
    if start > end:
        raise BadRange(from_cue, to_cue)
    return order[start : end + 1]


def resolve_label_updates(rows: list[dict], updates: list[dict]) -> list[dict]:
    """Expand ranges to `{groupId, label}` entries. Later entries still win in `merge_labels`."""
    resolved: list[dict] = []
    for update in updates:
        if "groupId" in update:
            resolved.append({"groupId": update["groupId"], "label": update["label"]})
            continue
        if update.get("scope", "groups") == "cues":
            continue
        for group_id in groups_wholly_inside(rows, update["fromCue"], update["toCue"]):
            resolved.append({"groupId": group_id, "label": update["label"]})
    return resolved


def _held(row: dict) -> bool:
    speaker = row.get("speaker") or row["groupId"]
    return speaker in HELD_SPEAKERS or row["groupId"] in HELD_SPEAKERS


def apply_label_updates(
    rows: list[dict], groups: list[dict], cue_labels: list[dict], updates: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Apply entries in order. A later entry wins, which is the existing override rule.

    A cue-scoped range sets `speaker` on the cues inside the window and does not change the
    group's label, so the same group keeps its label outside the window. A later per-group
    entry, or a later range whose scope is `groups`, sets that group label and clears cue
    overrides on the group's cues. `uncertain` and `mixed` are never changed.
    """
    group_map = {item["groupId"]: item["label"] for item in groups}
    cue_map = {item["id"]: item["label"] for item in cue_labels}
    by_id = {row["id"]: row for row in rows}
    members: dict[str, list[str]] = {}
    for row in rows:
        members.setdefault(row["groupId"], []).append(row["id"])

    def paint_group(group_id: str, label: str) -> None:
        group_map[group_id] = label
        for cue_id in members.get(group_id, []):
            cue_map.pop(cue_id, None)

    for update in updates:
        if "groupId" in update:
            paint_group(update["groupId"], update["label"])
            continue
        if update.get("scope", "groups") == "cues":
            for cue_id in cue_ids_between(rows, update["fromCue"], update["toCue"]):
                row = by_id[cue_id]
                if _held(row):
                    continue
                cue_map[cue_id] = update["label"]
            continue
        for group_id in groups_wholly_inside(rows, update["fromCue"], update["toCue"]):
            paint_group(group_id, update["label"])
    group_out = [{"groupId": group_id, "label": group_map[group_id]} for group_id in sorted(group_map)]
    cue_out = [{"id": cue_id, "label": cue_map[cue_id]} for cue_id in sorted(cue_map)]
    return group_out, cue_out


def assign_speakers(rows: list[dict], labels: list[dict], cue_labels: list[dict] | None = None) -> None:
    """Put each group's label on `speaker`, then any per-cue label on top.

    Before any label, `speaker` stays the group id. `uncertain` and `mixed` are left alone,
    including when the group they sit in is labeled or a cue-scoped range covers them.
    """
    by_group = {item["groupId"]: item["label"] for item in labels}
    by_cue = {item["id"]: item["label"] for item in (cue_labels or [])}
    for row in rows:
        speaker = row.get("speaker") or row["groupId"]
        if speaker in HELD_SPEAKERS or row["groupId"] in HELD_SPEAKERS:
            row["speaker"] = speaker if speaker in HELD_SPEAKERS else row["groupId"]
            continue
        if row["id"] in by_cue:
            row["speaker"] = by_cue[row["id"]]
        else:
            row["speaker"] = by_group.get(row["groupId"], row["groupId"])


def known_groups(rows: list[dict]) -> set[str]:
    return {UNGROUPED} | {row["groupId"] for row in rows}


def merge_labels(current: list[dict], updates: list[dict]) -> list[dict]:
    """Later entries win. Sorted by group id."""
    merged = {label["groupId"]: label["label"] for label in current}
    merged.update((label["groupId"], label["label"]) for label in updates)
    return [{"groupId": group_id, "label": merged[group_id]} for group_id in sorted(merged)]


def is_labeled(label: str | None) -> bool:
    """A real speaker name. `undetermined`, `quote`, and `ending` are not one."""
    return label is not None and label not in MARKS


def _share(rows: list[dict], labels: list[dict], wanted: Callable[[str | None], bool]) -> float:
    by_group = {label["groupId"]: label["label"] for label in labels}
    total = sum(row["endMs"] - row["startMs"] for row in rows)
    if total == 0:
        return 0.0
    matched = sum(row["endMs"] - row["startMs"] for row in rows if wanted(by_group.get(row["groupId"])))
    return round(matched / total, 4)


def labeled_share(rows: list[dict], labels: list[dict]) -> float:
    """Share of cue time whose group has a person's name. Marks do not count."""
    return _share(rows, labels, is_labeled)


def undetermined_share(rows: list[dict], labels: list[dict]) -> float:
    """Share of cue time whose group is labeled `undetermined`."""
    return _share(rows, labels, lambda label: label == "undetermined")


def host_names(labels: list[dict]) -> list[str]:
    """Person names only: `quote`, `ending` and `undetermined` are left out."""
    return [label["label"] for label in labels if label["label"] not in MARKS]


def page(rows: list[dict], limit: int, after: str | None, group_id: str | None) -> list[dict]:
    if group_id is not None:
        if group_id not in known_groups(rows):
            raise Unknown(groupId=group_id)
        rows = [row for row in rows if row["groupId"] == group_id]
    start = 0
    if after is not None:
        index = next((i for i, row in enumerate(rows) if row["id"] == after), None)
        if index is None:
            raise Unknown(cueId=after)
        start = index + 1
    return rows[start : start + limit]
