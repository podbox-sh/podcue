"""Cue-range labels and the speaker field, with a countdown split into tiny groups."""

from podcue.groups import apply_label_updates, assign_speakers, groups_wholly_inside, resolve_label_updates


def countdown():
    """A host line, eight quote groups of one to three cues, then the host again."""
    rows = [{"id": "c0001", "groupId": "ghost", "speaker": "ghost", "startMs": 0, "endMs": 1_000, "text": "and"}]
    cue = 2
    for index in range(8):
        group_id = f"gq{index}"
        for _ in range(1 + (index % 3)):
            rows.append({
                "id": f"c{cue:04d}",
                "groupId": group_id,
                "speaker": group_id,
                "startMs": cue * 1_000,
                "endMs": cue * 1_000 + 400,
                "text": str(8 - index),
            })
            cue += 1
    rows.append({
        "id": f"c{cue:04d}",
        "groupId": "ghost",
        "speaker": "ghost",
        "startMs": cue * 1_000,
        "endMs": cue * 1_000 + 1_000,
        "text": "liftoff",
    })
    rows.append({
        "id": "c0099",
        "groupId": "uncertain",
        "speaker": "uncertain",
        "startMs": 90_000,
        "endMs": 90_400,
        "text": "mm",
    })
    return rows, f"c{cue:04d}"


def test_a_range_labels_each_tiny_quote_group_and_leaves_the_host():
    rows, last_host = countdown()
    quote_ids = [row["id"] for row in rows if row["groupId"].startswith("gq")]
    labeled = groups_wholly_inside(rows, quote_ids[0], quote_ids[-1])
    assert labeled == [f"gq{index}" for index in range(8)]
    assert "ghost" not in labeled and "uncertain" not in labeled

    # The host's later cue sits past the countdown, so that group is not wholly inside.
    opened = groups_wholly_inside(rows, "c0001", quote_ids[-1])
    assert opened == [f"gq{index}" for index in range(8)]
    assert "ghost" not in opened
    # One cue of a two-cue quote group is not the whole group.
    assert "gq2" not in groups_wholly_inside(rows, quote_ids[0], quote_ids[2])

    updates = resolve_label_updates(rows, [
        {"fromCue": quote_ids[0], "toCue": quote_ids[-1], "label": "quote"},
        {"groupId": "ghost", "label": "Nora"},
    ])
    assert {item["groupId"] for item in updates if item["label"] == "quote"} == {f"gq{index}" for index in range(8)}
    assign_speakers(rows, [
        {"groupId": group_id, "label": "quote"} for group_id in labeled
    ] + [{"groupId": "ghost", "label": "Nora"}])
    assert {row["speaker"] for row in rows if row["groupId"].startswith("gq")} == {"quote"}
    assert {row["speaker"] for row in rows if row["groupId"] == "ghost"} == {"Nora"}
    assert rows[-1]["speaker"] == "uncertain" and rows[-1]["groupId"] == "uncertain"
    assert last_host.startswith("c")


def test_a_label_does_not_replace_an_uncertain_cue_inside_a_named_group():
    rows = [
        {"id": "c0001", "groupId": "gc0001", "speaker": "gc0001"},
        {"id": "c0002", "groupId": "gc0001", "speaker": "uncertain"},
        {"id": "c0003", "groupId": "gc0001", "speaker": "gc0001"},
    ]
    assign_speakers(rows, [{"groupId": "gc0001", "label": "Ada"}])
    assert [row["speaker"] for row in rows] == ["Ada", "uncertain", "Ada"]
    assign_speakers(rows, [{"groupId": "gc0001", "label": "Ben"}])
    assert [row["speaker"] for row in rows] == ["Ben", "uncertain", "Ben"]


def test_cue_scope_paints_only_the_window_and_a_later_group_wins():
    rows = [
        {"id": "c0001", "groupId": "ghost", "speaker": "ghost"},
        {"id": "c0002", "groupId": "gq", "speaker": "gq"},
        {"id": "c0003", "groupId": "ghost", "speaker": "ghost"},
        {"id": "c0004", "groupId": "uncertain", "speaker": "uncertain"},
    ]
    # The host group recurs after the montage, so a group-scoped range does not label it.
    assert groups_wholly_inside(rows, "c0001", "c0002") == ["gq"]
    groups, cues = apply_label_updates(rows, [], [], [
        {"groupId": "ghost", "label": "Nora"},
        {"fromCue": "c0001", "toCue": "c0002", "label": "quote", "scope": "cues"},
    ])
    assert groups == [{"groupId": "ghost", "label": "Nora"}]
    assert cues == [{"id": "c0001", "label": "quote"}, {"id": "c0002", "label": "quote"}]
    assign_speakers(rows, groups, cues)
    assert [row["speaker"] for row in rows] == ["quote", "quote", "Nora", "uncertain"]

    groups, cues = apply_label_updates(rows, groups, cues, [
        {"groupId": "ghost", "label": "Ada"},
    ])
    assign_speakers(rows, groups, cues)
    assert groups == [{"groupId": "ghost", "label": "Ada"}]
    assert {item["id"] for item in cues} == {"c0002"}
    assert rows[0]["speaker"] == "Ada" and rows[2]["speaker"] == "Ada"
    assert rows[1]["speaker"] == "quote" and rows[3]["speaker"] == "uncertain"
