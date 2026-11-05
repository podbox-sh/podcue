"""Speaker-group labels and the paged cue view.

A label is a person's name, `undetermined`, `quote` or `ending`. A group counts as labeled
when its label is anything but `undetermined`.
"""

from __future__ import annotations

from .cues import UNGROUPED

MARKS = ("undetermined", "quote", "ending")
DEFAULT_LIMIT = 50
MAX_LIMIT = 200
LABEL_SHAPE = "label-groups expects a JSON array of { groupId, label }"


class Unknown(Exception):
    """An id the episode does not have. Reported as `{"state": "unknown", ...}`, not as an error."""

    def __init__(self, **fields: str):
        super().__init__(fields)
        self.fields = fields


def parse_label_updates(value: object) -> list[dict] | None:
    if not isinstance(value, list):
        return None
    updates = []
    for item in value:
        if not isinstance(item, dict):
            return None
        group_id, label = item.get("groupId"), item.get("label")
        if not isinstance(group_id, str) or not group_id or not isinstance(label, str) or not label:
            return None
        updates.append({"groupId": group_id, "label": label})
    return updates


def known_groups(rows: list[dict]) -> set[str]:
    return {UNGROUPED} | {row["groupId"] for row in rows}


def merge_labels(current: list[dict], updates: list[dict]) -> list[dict]:
    """Later entries win. Sorted by group id."""
    merged = {label["groupId"]: label["label"] for label in current}
    merged.update((label["groupId"], label["label"]) for label in updates)
    return [{"groupId": group_id, "label": merged[group_id]} for group_id in sorted(merged)]


def is_labeled(label: str | None) -> bool:
    return label is not None and label != "undetermined"


def labeled_share(rows: list[dict], labels: list[dict]) -> float:
    """Share of total cue time (sum of endMs - startMs) whose group has a name, `quote` or `ending`."""
    by_group = {label["groupId"]: label["label"] for label in labels}
    total = sum(row["endMs"] - row["startMs"] for row in rows)
    labeled = sum(row["endMs"] - row["startMs"] for row in rows if is_labeled(by_group.get(row["groupId"])))
    return 0.0 if total == 0 else round(labeled / total, 4)


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
