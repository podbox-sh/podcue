import json

import pytest
import yaml
from conftest import run

from podcue.store import Episode


def ids(rows):
    return [row["id"] for row in rows]


# cues


def test_cues_pages_fifty_by_default_and_continues_after_a_cue(episode_id, capsys):
    page, code = run(capsys, "cues", episode_id)
    assert code == 0 and page["state"] == "present" and ids(page["cues"]) == [f"c{i:04d}" for i in range(1, 51)]
    assert set(page["cues"][0]) == {
        "id", "startMs", "endMs", "text", "groupId", "speaker", "nearestGroupId", "nearestCosine",
    }
    assert page["cues"][0]["speaker"] == page["cues"][0]["groupId"]
    assert "hostNames" not in page
    page, _ = run(capsys, "cues", episode_id, "--after", "c0050")
    assert ids(page["cues"])[0] == "c0051" and len(page["cues"]) == 50
    page, _ = run(capsys, "cues", episode_id, "--after", "c0100")
    assert len(page["cues"]) == 20
    page, _ = run(capsys, "cues", episode_id, "--limit", "200")
    assert len(page["cues"]) == 120


@pytest.mark.parametrize("limit", ["0", "201", "-3"])
def test_cues_limit_is_1_through_200(episode_id, capsys, limit):
    result, code = run(capsys, "cues", episode_id, "--limit", limit)
    assert code == 1 and result == {"kind": "error", "message": "limit must be an integer from 1 through 200",
                                     "reason": "limit"}


def test_cues_filters_one_group_and_pages_inside_it(episode_id, capsys):
    page, _ = run(capsys, "cues", episode_id, "--group-id", "gc0002", "--limit", "3")
    assert ids(page["cues"]) == ["c0002", "c0005", "c0008"]
    assert {row["groupId"] for row in page["cues"]} == {"gc0002"}
    page, _ = run(capsys, "cues", episode_id, "--group-id", "gc0002", "--limit", "2", "--after", "c0005")
    assert ids(page["cues"]) == ["c0008", "c0011"]
    page, _ = run(capsys, "cues", episode_id, "--group-id", "ungrouped", "--limit", "1")
    assert page["cues"][0]["id"] == "c0003" and page["cues"][0]["nearestGroupId"] is None


def test_cues_reports_unknown_groups_and_cues(episode_id, capsys):
    assert run(capsys, "cues", episode_id, "--group-id", "gc0003") == (
        {"state": "unknown", "episodeId": episode_id, "groupId": "gc0003"}, 0)
    assert run(capsys, "cues", episode_id, "--after", "c9999") == (
        {"state": "unknown", "episodeId": episode_id, "cueId": "c9999"}, 0)
    # --after must name a cue inside the filtered group
    assert run(capsys, "cues", episode_id, "--group-id", "gc0002", "--after", "c0001")[0]["cueId"] == "c0001"


def test_cues_hosts_lists_person_names_only(episode_id, capsys):
    run(capsys, "label-groups", episode_id, json.dumps([
        {"groupId": "gc0001", "label": "Ana Ruiz"}, {"groupId": "gc0002", "label": "quote"},
        {"groupId": "ungrouped", "label": "undetermined"}]))
    page, _ = run(capsys, "cues", episode_id, "--hosts", "--limit", "1")
    assert page["hostNames"] == ["Ana Ruiz"]


# label-groups


def test_label_groups_merges_and_reports_labeled_share(episode_id, root, capsys):
    result, code = run(capsys, "label-groups", episode_id, '[{"groupId": "gc0002", "label": "Ben"}]')
    assert code == 0 and result == {"state": "present", "episodeId": episode_id,
                                    "labels": [{"groupId": "gc0002", "label": "Ben"}],
                                    "labeledShare": 0.3333, "undeterminedShare": 0.0}
    result, _ = run(capsys, "label-groups", episode_id, '[{"groupId": "gc0001", "label": "Ana"}]')
    assert result["labels"] == [{"groupId": "gc0001", "label": "Ana"}, {"groupId": "gc0002", "label": "Ben"}]
    assert result["labeledShare"] == 0.6667 and result["undeterminedShare"] == 0.0
    result, _ = run(capsys, "label-groups", episode_id,
                    '[{"groupId": "gc0002", "label": "undetermined"}, {"groupId": "ungrouped", "label": "ending"}]')
    assert result["labels"][1:] == [{"groupId": "gc0002", "label": "undetermined"},
                                    {"groupId": "ungrouped", "label": "ending"}]
    # `ending` and `undetermined` are not speaker names. Only Ana's third of the cue time counts.
    assert result["labeledShare"] == 0.3333 and result["undeterminedShare"] == 0.3333
    stored = json.loads(Episode(root.resolve(), episode_id).labels.read_text())
    assert stored == {"groups": result["labels"]}
    view, _ = run(capsys, "episode", episode_id)
    assert [g["label"] for g in view["groups"]] == ["Ana", "undetermined", "ending"]
    assert view["labeledShare"] == 0.3333 and view["undeterminedShare"] == 0.3333
    page, _ = run(capsys, "cues", episode_id, "--limit", "200")
    by_id = {cue["id"]: cue for cue in page["cues"]}
    assert by_id["c0001"]["groupId"] == "gc0001" and by_id["c0001"]["speaker"] == "Ana"
    assert by_id["c0002"]["groupId"] == "gc0002" and by_id["c0002"]["speaker"] == "undetermined"
    assert by_id["c0003"]["groupId"] == "ungrouped" and by_id["c0003"]["speaker"] == "ending"


def test_quote_does_not_count_as_a_labeled_speaker(episode_id, capsys):
    result, code = run(capsys, "label-groups", episode_id, '[{"groupId": "gc0001", "label": "quote"}]')
    assert code == 0 and result["labeledShare"] == 0.0 and result["undeterminedShare"] == 0.0


@pytest.mark.parametrize("body", ['{"groupId": "gc0001", "label": "A"}', '[{"groupId": "gc0001", "label": ""}]',
                                  '[{"groupId": "gc0001"}]', "not json"])
def test_label_groups_rejects_a_bad_body(episode_id, capsys, body):
    assert run(capsys, "label-groups", episode_id, body) == (
        {"kind": "error", "message":
         "label-groups expects a JSON array of { groupId, label } "
         "or { fromCue, toCue, label, scope? } where scope is groups (default) or cues"}, 1)


def test_label_groups_reports_an_unknown_group_and_writes_nothing(episode_id, root, capsys):
    result, code = run(capsys, "label-groups", episode_id,
                       '[{"groupId": "gc0001", "label": "A"}, {"groupId": "gx", "label": "B"}]')
    assert (result, code) == ({"state": "unknown", "episodeId": episode_id, "groupId": "gx"}, 0)
    assert not Episode(root.resolve(), episode_id).labels.exists()


def test_a_cue_range_labels_every_group_that_sits_inside_it(episode_id, capsys):
    # Each fixture group spans the episode, so a short window contains none of them wholly.
    result, code = run(capsys, "label-groups", episode_id,
                       '[{"fromCue": "c0004", "toCue": "c0006", "label": "quote"}]')
    assert code == 0 and result["labels"] == []
    page, _ = run(capsys, "cues", episode_id, "--limit", "1")
    assert page["cues"][0]["speaker"] == "gc0001"

    result, code = run(capsys, "label-groups", episode_id,
                       '[{"fromCue": "c0001", "toCue": "c0120", "label": "quote"}]')
    assert code == 0 and result["labels"] == [
        {"groupId": "gc0001", "label": "quote"},
        {"groupId": "gc0002", "label": "quote"},
        {"groupId": "ungrouped", "label": "quote"},
    ]
    assert result["labeledShare"] == 0.0 and result["undeterminedShare"] == 0.0
    page, _ = run(capsys, "cues", episode_id, "--limit", "3")
    assert [cue["speaker"] for cue in page["cues"]] == ["quote", "quote", "quote"]
    assert [cue["groupId"] for cue in page["cues"]] == ["gc0001", "gc0002", "ungrouped"]

    result, _ = run(capsys, "label-groups", episode_id, '[{"groupId": "gc0001", "label": "Ada"}]')
    assert {"groupId": "gc0001", "label": "Ada"} in result["labels"]
    page, _ = run(capsys, "cues", episode_id, "--limit", "1")
    assert page["cues"][0]["speaker"] == "Ada" and page["cues"][0]["groupId"] == "gc0001"


def test_a_range_with_an_unknown_cue_writes_nothing(episode_id, root, capsys):
    result, code = run(capsys, "label-groups", episode_id,
                       '[{"fromCue": "c9999", "toCue": "c0003", "label": "quote"}]')
    assert (result, code) == ({"state": "unknown", "episodeId": episode_id, "cueId": "c9999"}, 0)
    assert not Episode(root.resolve(), episode_id).labels.exists()
    result, code = run(capsys, "label-groups", episode_id,
                       '[{"fromCue": "c0001", "toCue": "c9999", "label": "quote"}]')
    assert result["cueId"] == "c9999" and code == 0
    assert not Episode(root.resolve(), episode_id).labels.exists()


def test_a_backwards_cue_range_is_rejected(episode_id, root, capsys):
    result, code = run(capsys, "label-groups", episode_id,
                       '[{"fromCue": "c0006", "toCue": "c0004", "label": "quote"}]')
    assert code == 1 and result["kind"] == "error" and "c0006" in result["message"] and "c0004" in result["message"]
    assert not Episode(root.resolve(), episode_id).labels.exists()


def test_an_entry_cannot_be_a_group_and_a_range(episode_id, capsys):
    body = '[{"groupId": "gc0001", "fromCue": "c0001", "toCue": "c0002", "label": "quote"}]'
    result, code = run(capsys, "label-groups", episode_id, body)
    assert code == 1 and result["kind"] == "error" and "fromCue" in result["message"]


# summaries


@pytest.fixture
def summary_id(episode_id, capsys):
    result, code = run(capsys, "start-summary", episode_id)
    assert code == 0
    return result["summaryId"]


def test_start_summary_writes_an_empty_file(episode_id, root, capsys):
    result, _ = run(capsys, "start-summary", episode_id)
    assert result == {"state": "present", "episodeId": episode_id, "version": 1,
                      "summaryId": result["summaryId"], "summary": "", "paragraphs": []}
    assert result["summaryId"].startswith("sm_") and len(result["summaryId"]) == 19
    path = Episode(root.resolve(), episode_id).summary_file(result["summaryId"])
    assert yaml.safe_load(path.read_text()) == {"version": 1, "episodeId": episode_id,
                                                 "summaryId": result["summaryId"], "summary": "", "paragraphs": []}
    view, _ = run(capsys, "episode", episode_id)
    assert view["summaries"] == [{"summaryId": result["summaryId"]}]


def test_post_paragraph_orders_cues_and_fills_times(episode_id, summary_id, capsys):
    result, code = run(capsys, "post-paragraph", episode_id, summary_id,
                       '{"text": "They open.", "cues": ["c0004", "c0001"], "part": "claim"}')
    assert code == 0
    assert result["paragraphs"] == [{"index": 1, "part": "claim", "text": "They open.",
                                     "cues": ["c0001", "c0004"], "startMs": 0, "endMs": 3_900}]
    result, _ = run(capsys, "post-paragraph", episode_id, summary_id,
                    '{"text": "Then.", "cues": ["c0010"], "startMs": 9000, "endMs": 9900}')
    assert result["paragraphs"][1] == {"index": 2, "text": "Then.", "cues": ["c0010"], "startMs": 9_000, "endMs": 9_900}


def test_post_paragraph_takes_an_array_and_appends_in_order(episode_id, summary_id, root, capsys):
    run(capsys, "post-paragraph", episode_id, summary_id, '{"text": "First.", "cues": ["c0001"]}')
    result, code = run(capsys, "post-paragraph", episode_id, summary_id, json.dumps([
        {"text": "Second.", "cues": ["c0007", "c0005"], "part": "story"},
        {"text": "Third.", "cues": ["c0002"]},
    ]))
    assert code == 0
    assert [(p["index"], p["text"], p["cues"]) for p in result["paragraphs"]] == [
        (1, "First.", ["c0001"]), (2, "Second.", ["c0005", "c0007"]), (3, "Third.", ["c0002"])]
    assert result["paragraphs"][1]["startMs"] == 4_000 and result["paragraphs"][1]["endMs"] == 6_900
    stored = yaml.safe_load(Episode(root.resolve(), episode_id).summary_file(summary_id).read_text())
    assert stored["paragraphs"] == result["paragraphs"]


@pytest.mark.parametrize("body, message", [
    ('{"text": "x", "cues": ["c9999"]}', "a cue is not in the transcript: c9999"),
    ('{"text": "x", "cues": ["c0001", "c0001"]}', "a cue is listed twice: c0001"),
    ('{"text": "x", "cues": ["c0001"], "startMs": 5}', "startMs does not match the cue: c0001 starts at 0"),
    ('{"text": "x", "cues": ["c0001", "c0002"], "endMs": 900}', "endMs does not match the cue: c0002 ends at 1900"),
    ('{"text": "x", "cues": ["c0001"], "part": "rant"}',
     "part must be one of news, explanation, advice, claim, discussion, story, items"),
    ('{"index": 1, "text": "x", "cues": ["c0001"]}', "postParagraph rejects index"),
    ('{"text": "", "cues": ["c0001"]}', "postParagraph expects { text, cues } or an array of them"),
    ('{"text": "x", "cues": []}', "postParagraph expects { text, cues } or an array of them"),
    ('{"text": "x", "cues": ["c0001"], "speakers": []}', "postParagraph expects { text, cues } or an array of them"),
    ("[]", "postParagraph expects { text, cues } or an array of them"),
    ('[{"text": "ok", "cues": ["c0001"]}, {"text": "bad", "cues": ["c9999"]}]',
     "paragraph 2: a cue is not in the transcript: c9999"),
])
def test_post_paragraph_rejects_and_writes_nothing(episode_id, summary_id, capsys, body, message):
    assert run(capsys, "post-paragraph", episode_id, summary_id, body) == ({"kind": "error", "message": message}, 1)
    assert run(capsys, "summary", episode_id, summary_id)[0]["paragraphs"] == []


def test_put_paragraph_replaces_an_existing_index(episode_id, summary_id, capsys):
    run(capsys, "post-paragraph", episode_id, summary_id, '[{"text": "A.", "cues": ["c0001"]}, {"text": "B.", "cues": ["c0002"]}]')
    result, code = run(capsys, "put-paragraph", episode_id, summary_id, '{"index": 1, "text": "A2.", "cues": ["c0006", "c0003"]}')
    assert code == 0 and result["paragraphs"][0] == {"index": 1, "text": "A2.", "cues": ["c0003", "c0006"],
                                                      "startMs": 2_000, "endMs": 5_900}
    assert result["paragraphs"][1]["text"] == "B."
    assert run(capsys, "put-paragraph", episode_id, summary_id, '{"index": 3, "text": "C.", "cues": ["c0001"]}') == (
        {"kind": "error", "message": "putParagraph index does not exist"}, 1)
    assert run(capsys, "put-paragraph", episode_id, summary_id, '{"text": "C.", "cues": ["c0001"]}')[0]["message"] == (
        "putParagraph requires index")


def test_set_summary_needs_paragraphs_and_text(episode_id, summary_id, capsys):
    assert run(capsys, "set-summary", episode_id, summary_id, '{"summary": "All of it."}') == (
        {"kind": "error", "message": "write the paragraphs before the episode summary"}, 1)
    run(capsys, "post-paragraph", episode_id, summary_id, '{"text": "A.", "cues": ["c0001"]}')
    assert run(capsys, "set-summary", episode_id, summary_id, '{"summary": ""}') == (
        {"kind": "error", "message": "summary must be a non-empty string"}, 1)
    result, code = run(capsys, "set-summary", episode_id, summary_id, '{"summary": "All of it."}')
    assert code == 0 and result["summary"] == "All of it."
    assert run(capsys, "summary", episode_id, summary_id) == (result, 0)


def test_unknown_summary(episode_id, capsys):
    assert run(capsys, "summary", episode_id, "sm_0000000000000000") == (
        {"state": "unknown", "episodeId": episode_id, "summaryId": "sm_0000000000000000"}, 0)
    assert run(capsys, "post-paragraph", episode_id, "nope", '{"text": "A.", "cues": ["c0001"]}')[0]["state"] == "unknown"


# check


def label_all(episode_id, capsys, gc0002="Ben"):
    run(capsys, "label-groups", episode_id, json.dumps([
        {"groupId": "gc0001", "label": "Ana"}, {"groupId": "gc0002", "label": gc0002}]))


def valid(episode_id):
    return {
        "version": 1,
        "episodeId": episode_id,
        "summary": "Ana and Ben talk.",
        "paragraphs": [
            {"index": 1, "part": "discussion", "text": "Ana opens.", "cues": ["c0001", "c0003", "c0004"],
             "startMs": 0, "endMs": 3_900},
            {"index": 2, "text": "Ben replies.", "cues": ["c0002"], "startMs": 1_000, "endMs": 1_900},
        ],
    }


def write(tmp_path, doc, name="summary.yaml"):
    path = tmp_path / name
    path.write_text(json.dumps(doc) if name.endswith(".json") else yaml.safe_dump(doc))
    return path


def test_check_passes_a_summary_written_by_the_commands(episode_id, summary_id, capsys):
    label_all(episode_id, capsys)
    run(capsys, "post-paragraph", episode_id, summary_id, '[{"text": "A.", "cues": ["c0004", "c0001"]}, {"text": "B.", "cues": ["c0002", "c0003"]}]')
    run(capsys, "set-summary", episode_id, summary_id, '{"summary": "All."}')
    result, code = run(capsys, "check", episode_id, summary_id)
    assert code == 0 and result["state"] == "valid" and result["summaryId"] == summary_id
    assert result["paragraphs"] == 2 and result["warnings"] == []


@pytest.mark.parametrize("name", ["summary.yaml", "summary.json"])
def test_check_passes_a_hand_written_file(episode_id, capsys, tmp_path, name):
    label_all(episode_id, capsys)
    result, code = run(capsys, "check", episode_id, write(tmp_path, valid(episode_id), name))
    assert code == 0 and result["state"] == "valid" and "summaryId" not in result


def test_check_warns_on_an_undetermined_group_but_passes(episode_id, capsys, tmp_path):
    label_all(episode_id, capsys, gc0002="undetermined")
    result, code = run(capsys, "check", episode_id, write(tmp_path, valid(episode_id)))
    assert code == 0 and result["warnings"] == ["group gc0002 (first cited in paragraph 2) is undetermined"]


def broken(change):
    def apply(doc):
        change(doc)
        return doc
    return apply


FAILURES = [
    (broken(lambda d: d["paragraphs"][0]["cues"].append("c9999")), "paragraph 1: cue c9999 is not in the transcript"),
    (broken(lambda d: d["paragraphs"][0].update(startMs=5)), "paragraph 1: startMs 5 does not match c0001 startMs 0"),
    (broken(lambda d: d["paragraphs"][1].update(endMs=2_000)), "paragraph 2: endMs 2000 does not match c0002 endMs 1900"),
    (broken(lambda d: d["paragraphs"][1].pop("endMs")), "paragraph 2: endMs must be a number"),
    (broken(lambda d: d.update(summary="")), "summary must be a non-empty string"),
    (broken(lambda d: d.update(paragraphs=[])), "paragraphs must be a non-empty list"),
    (broken(lambda d: d["paragraphs"][1].update(text=" ")), "paragraph 2: text must be a non-empty string"),
    (broken(lambda d: d["paragraphs"][1].update(cues=[])), "paragraph 2: cues must be a non-empty list of cue ids"),
    (broken(lambda d: d["paragraphs"][0].update(cues=["c0004", "c0001", "c0003"])),
     "paragraph 1: cues must be in transcript order"),
    (broken(lambda d: d["paragraphs"][1].update(cues=["c0002", "c0002"])), "paragraph 2: cue c0002 is listed twice"),
    (broken(lambda d: d["paragraphs"][0].update(part="rant")),
     "paragraph 1: part must be one of news, explanation, advice, claim, discussion, story, items"),
    (broken(lambda d: d["paragraphs"][1].update(index=3)), "paragraph 2: index must be 2"),
    (broken(lambda d: d["paragraphs"][1].update(speakers=["Ben"])), "paragraph 2: unknown keys: speakers"),
    (broken(lambda d: d.update(speakers={"gc0001": "Ana"})), "unknown keys: speakers"),
    (broken(lambda d: d.update(version=2)), "version must be 1"),
    (broken(lambda d: d.update(episodeId="ep_0000000000000000")), "episodeId must be "),
    (broken(lambda d: d.update(summaryId="sm_x")), "summaryId must be sm_ followed by 16 hex digits"),
]


@pytest.mark.parametrize("change, problem", FAILURES)
def test_check_failures(episode_id, capsys, tmp_path, change, problem):
    label_all(episode_id, capsys)
    result, code = run(capsys, "check", episode_id, write(tmp_path, change(valid(episode_id))))
    assert code == 1 and result["kind"] == "error"
    assert any(p.startswith(problem) for p in result["problems"]), result["problems"]
    assert result["message"].startswith(f"{len(result['problems'])} problem(s): ")


def test_check_fails_a_cited_group_with_no_label(episode_id, capsys, tmp_path):
    run(capsys, "label-groups", episode_id, '[{"groupId": "gc0001", "label": "Ana"}]')
    result, code = run(capsys, "check", episode_id, write(tmp_path, valid(episode_id)))
    assert code == 1 and result["problems"] == ["group gc0002 (first cited in paragraph 2) has no label in labels.json"]


def test_check_lets_ungrouped_cues_through_without_a_label(episode_id, capsys, tmp_path):
    doc = valid(episode_id)
    doc["paragraphs"] = [{"index": 1, "text": "Music.", "cues": ["c0003"], "startMs": 2_000, "endMs": 2_900}]
    assert run(capsys, "check", episode_id, write(tmp_path, doc))[1] == 0


def test_check_refuses_what_it_cannot_read(episode_id, root, capsys, tmp_path):
    assert run(capsys, "check", episode_id, tmp_path / "none.yaml")[1] == 1
    assert run(capsys, "check", episode_id, "sm_0000000000000000")[1] == 1
    assert run(capsys, "check", "ep_0000000000000000", "x.yaml")[0]["message"] == "unknown episode ep_0000000000000000"
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [")
    assert "could not be parsed" in run(capsys, "check", episode_id, bad)[0]["message"]
