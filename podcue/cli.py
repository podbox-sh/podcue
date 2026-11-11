"""The podcue commands. Each prints exactly one JSON line on stdout; progress goes to stderr.

A non-zero exit always comes with `{"kind": "error", "message": ...}`. An id the episode does
not have is `{"state": "unknown", ...}` with exit 0, and a command that needs cues before
`start` has written them is `{"state": "not-ready", ...}` with exit 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import __version__
from .groups import (
    DEFAULT_LIMIT,
    LABEL_SHAPE,
    MAX_LIMIT,
    Unknown,
    host_names,
    known_groups,
    labeled_share,
    merge_labels,
    page,
    parse_label_updates,
)
from .store import (
    SUMMARY_ID,
    Episode,
    dump_yaml,
    load_labels,
    load_rows,
    read_json,
    resolve_root,
    start_state,
    summary_ids,
    write_json,
    write_text,
    writing,
)


class Failure(Exception):
    def __init__(self, message: str, **fields: object):
        super().__init__(message)
        self.fields = fields


class NotReady(Exception):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str):
        raise Failure(f"{self.prog}: {message}")


def build_parser() -> Parser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS, metavar="DIR",
                        help="episode root (default: $PODCUE_HOME, else ~/podcue)")
    parser = Parser(
        prog="podcue",
        parents=[common],
        description="Turn podcast audio into timed, speaker-grouped cues, label the voices, and store "
        "summaries that cite the cues. State lives in <root>/episodes/<episodeId>/. "
        "Every command prints one JSON line.",
    )
    parser.add_argument("--version", action="version", version=f"podcue {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=Parser, metavar="COMMAND")

    def command(name: str, help: str) -> Parser:
        return sub.add_parser(name, parents=[common], help=help, description=help)

    start = command("start", "transcribe and group an audio file or URL; returns when the cues are ready")
    start.add_argument("source", help="audio or video file, or an http(s) URL to one")
    start.add_argument("--language", default="en", metavar="CODE",
                       help="en de fr es it nl pl, or ja ko zh with the moonshine extra (default: en)")

    command("episode", "status, groups, labels and summaries of an episode").add_argument("episodeId")

    cues = command("cues", "one page of cue rows")
    cues.add_argument("episodeId")
    cues.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"1 through {MAX_LIMIT} (default {DEFAULT_LIMIT})")
    cues.add_argument("--after", metavar="CUE_ID", help="start after this cue")
    cues.add_argument("--group-id", metavar="GROUP_ID", help="only this group's cues")
    cues.add_argument("--hosts", action="store_true", help="also return the person names in labels.json")

    labels = command("label-groups", "merge labels into labels.json: '[{\"groupId\":...,\"label\":...}]'")
    labels.add_argument("episodeId")
    labels.add_argument("json")

    command("start-summary", "create an empty summary and return its id").add_argument("episodeId")
    for name, help in (
        ("post-paragraph", "append a paragraph '{\"text\":...,\"cues\":[...]}', or an array of them"),
        ("put-paragraph", "replace paragraph '{\"index\":n,\"text\":...,\"cues\":[...]}'"),
        ("set-summary", "set the episode summary '{\"summary\":...}' after the paragraphs"),
    ):
        body = command(name, help)
        body.add_argument("episodeId")
        body.add_argument("summaryId")
        body.add_argument("json")
    read = command("summary", "read a summary back")
    read.add_argument("episodeId")
    read.add_argument("summaryId")

    check = command("check", "strictly validate a summary file (by id, or a YAML/JSON path) against the episode")
    check.add_argument("episodeId")
    check.add_argument("summary", help="a summary id, or a path to a .yaml/.json file")

    command("fetch-weights", "optional: download the pinned diarization weights now")

    fetch = command("fetch", "download episode audio to a 16 kHz mono wav that start can take")
    fetch.add_argument("url", help="direct audio URL, or a page URL yt-dlp can read (recap extra)")
    fetch.add_argument("--out", default=".", metavar="DIR", help="directory for audio.wav (default: .)")

    build = command("build", "narrate script.json, cut clips from the episode, and stitch one audio file")
    build.add_argument("episodeId")
    build.add_argument("script", help="script.json: {voice?, output?, segments:[...]}")
    build.add_argument("--out", metavar="FILE",
                       help="output audio (default: the script's output, else produced_summary.mp3)")
    return parser


def _episode(root: Path, episode_id: str) -> Episode:
    episode = Episode(root, episode_id)
    if not episode.known():
        raise Unknown()
    return episode


def _ready(episode: Episode) -> list[dict]:
    if not episode.ready():
        raise NotReady()
    return load_rows(episode)


def _json(text: str, message: str) -> object:
    try:
        return json.loads(text)
    except ValueError:
        raise Failure(message) from None


def episode_view(episode: Episode) -> dict:
    manifest = read_json(episode.manifest)
    state = start_state(episode)
    artifacts = []
    for name, path in (("transcript", episode.transcript), ("cues", episode.cues)):
        artifact = {"artifact": name, **state}
        if state["state"] == "ready":
            artifact["path"] = str(path)
        artifacts.append(artifact)
    view = {"state": "present", **manifest, "dir": str(episode.dir), "artifacts": artifacts}
    if episode.ready():
        labels = load_labels(episode)
        label_of = {label["groupId"]: label["label"] for label in labels}
        view["groups"] = [{**group, "label": label_of.get(group["groupId"])} for group in read_json(episode.groups)]
        view["labeledShare"] = labeled_share(load_rows(episode), labels)
    view["summaries"] = [{"summaryId": summary_id} for summary_id in summary_ids(episode)]
    return view


def _present_summary(episode: Episode, document: dict) -> dict:
    return {"state": "present", "episodeId": episode.id, **{k: document[k] for k in ("version", "summaryId", "summary", "paragraphs")}}


def _read_summary(episode: Episode, summary_id: str) -> dict:
    path = episode.summary_file(summary_id)
    if not SUMMARY_ID.match(summary_id) or not path.is_file():
        raise Unknown(summaryId=summary_id)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        document = None
    if not (isinstance(document, dict) and document.get("summaryId") == summary_id
            and isinstance(document.get("paragraphs"), list) and isinstance(document.get("summary"), str)):
        raise Failure("the summary could not be read")
    return document


def _store_summary(episode: Episode, document: dict) -> dict:
    write_text(episode.summary_file(document["summaryId"]), dump_yaml(document))
    return _present_summary(episode, document)


def run_start(args, root):
    from .pipeline import start

    episode, created = start(args.source, args.language, root)
    return {"kind": "started", "created": created, "episode": episode_view(episode)}


def run_cues(args, root):
    if not 1 <= args.limit <= MAX_LIMIT:
        raise Failure(f"limit must be an integer from 1 through {MAX_LIMIT}", reason="limit")
    episode = _episode(root, args.episodeId)
    rows = page(_ready(episode), args.limit, args.after, args.group_id)
    result = {"state": "present", "episodeId": episode.id, "cues": rows}
    if args.hosts:
        result["hostNames"] = host_names(load_labels(episode))
    return result


def run_label_groups(args, root):
    updates = parse_label_updates(_json(args.json, LABEL_SHAPE))
    if updates is None:
        raise Failure(LABEL_SHAPE)
    episode = _episode(root, args.episodeId)
    rows = _ready(episode)
    known = known_groups(rows)
    for update in updates:
        if update["groupId"] not in known:
            raise Unknown(groupId=update["groupId"])
    with writing(episode):
        labels = merge_labels(load_labels(episode), updates)
        write_json(episode.labels, {"groups": labels})
    return {"state": "present", "episodeId": episode.id, "labels": labels, "labeledShare": labeled_share(rows, labels)}


def run_start_summary(args, root):
    from .summary import empty, new_summary_id

    episode = _episode(root, args.episodeId)
    with writing(episode):
        summary_id = new_summary_id()
        while episode.summary_file(summary_id).exists():
            summary_id = new_summary_id()
        return _store_summary(episode, empty(episode.id, summary_id))


def _change_summary(args, root, change, message):
    episode = _episode(root, args.episodeId)
    body = _json(args.json, message)
    with writing(episode):
        document = _read_summary(episode, args.summaryId)
        return _store_summary(episode, change(episode, document, body))


def run_post_paragraph(args, root):
    from .summary import POST_SHAPE, post

    return _change_summary(args, root, lambda ep, doc, body: post(doc, body, _ready(ep)), POST_SHAPE)


def run_put_paragraph(args, root):
    from .summary import PUT_SHAPE, put

    return _change_summary(args, root, lambda ep, doc, body: put(doc, body, _ready(ep)), PUT_SHAPE)


def run_set_summary(args, root):
    from .summary import Invalid, set_summary

    def change(_episode, document, body):
        if not isinstance(body, dict):
            raise Invalid("set-summary expects JSON { summary }")
        return set_summary(document, body.get("summary"))

    return _change_summary(args, root, change, "set-summary expects JSON { summary }")


def run_summary(args, root):
    episode = _episode(root, args.episodeId)
    return _present_summary(episode, _read_summary(episode, args.summaryId))


def run_check(args, root):
    from .summary import check, load

    episode = Episode(root, args.episodeId)
    if not episode.known():
        raise Failure(f"unknown episode {args.episodeId}")
    if not episode.ready():
        raise Failure(f"{episode.id} has no cues yet; run start first")
    summary_id = args.summary if SUMMARY_ID.match(args.summary) else None
    path = episode.summary_file(summary_id) if summary_id else Path(args.summary)
    if not path.is_file():
        raise Failure(f"{args.summary} is not a summary of {episode.id} or a file")
    try:
        document = load(path)
    except (ValueError, yaml.YAMLError) as error:
        raise Failure(f"{path} could not be parsed: {error}") from None
    problems, warnings = check(document, episode.id, load_rows(episode), load_labels(episode), summary_id)
    if problems:
        raise Failure(f"{len(problems)} problem(s): " + "; ".join(problems), problems=problems, warnings=warnings)
    result = {"state": "valid", "episodeId": episode.id, "path": str(path), "paragraphs": len(document["paragraphs"])}
    if summary_id or isinstance(document.get("summaryId"), str):
        result["summaryId"] = document["summaryId"]
    result["warnings"] = warnings
    return result


def run_fetch(args, _root):
    from .fetch import fetch

    return fetch(args.url, Path(args.out).expanduser())


def run_build(args, root):
    from .recap import build

    episode = _episode(root, args.episodeId)
    return build(episode, _ready(episode), Path(args.script), getattr(args, "out", None))


def run_fetch_weights(_args, _root):
    from . import weights

    return {"kind": "ready", "weights": [
        {"file": pin.file, "path": str(weights.ensure(pin)), "sha256": pin.sha256, "license": pin.license}
        for pin in (weights.ECAPA, weights.SEGMENTATION)
    ]}


COMMANDS = {
    "start": run_start,
    "episode": lambda args, root: episode_view(_episode(root, args.episodeId)),
    "cues": run_cues,
    "label-groups": run_label_groups,
    "start-summary": run_start_summary,
    "post-paragraph": run_post_paragraph,
    "put-paragraph": run_put_paragraph,
    "set-summary": run_set_summary,
    "summary": run_summary,
    "check": run_check,
    "fetch-weights": run_fetch_weights,
    "fetch": run_fetch,
    "build": run_build,
}


def execute(argv: list[str]) -> tuple[dict, int]:
    from .pipeline import Rejected
    from .recap import RecapError
    from .summary import Invalid

    episode_id = None
    try:
        args = build_parser().parse_args(argv)
        episode_id = getattr(args, "episodeId", None)
        return COMMANDS[args.command](args, resolve_root(getattr(args, "root", None))), 0
    except Unknown as unknown:
        return {"state": "unknown", "episodeId": episode_id, **unknown.fields}, 0
    except NotReady:
        return {"state": "not-ready", "episodeId": episode_id, "on": "cues"}, 0
    except Failure as error:
        return {"kind": "error", "message": str(error), **error.fields}, 1
    except Rejected as error:
        return {"kind": "error", "reason": error.reason, "message": str(error)}, 1
    except Invalid as error:
        return {"kind": "error", "message": str(error)}, 1
    except RecapError as error:
        return {"kind": "error", "message": str(error), **error.fields}, 1
    except KeyboardInterrupt:
        return {"kind": "error", "message": "interrupted"}, 130
    except Exception as error:  # one JSON line instead of a traceback
        return {"kind": "error", "message": str(error) or type(error).__name__}, 1


def main(argv: list[str] | None = None) -> int:
    result, code = execute(sys.argv[1:] if argv is None else argv)
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return code
