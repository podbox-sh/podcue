"""`podcue <audio> [-o DIR] [--language CODE]` and `podcue check <summary> --cues <cues.json>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .transcribe import LANGUAGES


def _transcribe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcue",
        description="Turn podcast audio into timed, speaker-grouped cues. "
        "Writes <name>.cues.json and <name>.vtt and prints their paths. "
        "Run `podcue check -h` to validate a summary.",
    )
    parser.add_argument("source", help="audio or video file, or an http(s) URL to one")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("."), help="output directory (default: .)")
    parser.add_argument("--language", default="en", choices=LANGUAGES, metavar="CODE",
                        help=f"spoken language, one of {' '.join(LANGUAGES)} (default: en)")
    parser.add_argument("--version", action="version", version=f"podcue {__version__}")
    return parser


def _check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcue check",
        description="Validate a summary (JSON or YAML) against the cues it cites. Exits 1 on any problem.",
    )
    parser.add_argument("summary", type=Path, help="summary file, .json or .yaml")
    parser.add_argument("--cues", type=Path, required=True, help="the .cues.json podcue wrote")
    return parser


def _check(argv: list[str]) -> int:
    import json

    import yaml

    from .check import check, load

    args = _check_parser().parse_args(argv)
    try:
        summary = load(args.summary)
        cues_doc = json.loads(args.cues.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"podcue check: {error}", file=sys.stderr)
        return 1
    errors = check(summary, cues_doc)
    for message in errors:
        print(f"{args.summary}: {message}", file=sys.stderr)
    if errors:
        print(f"podcue check: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    print(f"{args.summary}: ok, {len(summary['paragraphs'])} paragraphs")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["check"]:
        return _check(argv[1:])
    args = _transcribe_parser().parse_args(argv)

    from .pipeline import run

    try:
        paths = run(args.source, args.out_dir, args.language)
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # one clear line instead of a traceback
        print(f"podcue: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0
