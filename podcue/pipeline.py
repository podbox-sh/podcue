"""One foreground run: fetch, convert, diarize, transcribe, write cues JSON and WebVTT."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from . import audio, weights
from .cues import build_cues, cue_document, render_vtt
from .diarize import Diarizer
from .regions import plan_regions
from .transcribe import open_transcriber


def log(message: str) -> None:
    print(f"podcue: {message}", file=sys.stderr, flush=True)


def run(source: str, out_dir: Path, language: str) -> tuple[Path, Path]:
    transcriber = open_transcriber(language)
    ecapa = weights.ensure(weights.ECAPA)
    segmentation = weights.ensure(weights.SEGMENTATION)

    with tempfile.TemporaryDirectory(prefix="podcue-") as scratch:
        scratch_dir = Path(scratch)
        if audio.is_url(source):
            local = audio.download(source, scratch_dir)
        else:
            local = Path(source)
            if not local.is_file():
                raise FileNotFoundError(f"{source} is not a file")
        wav = scratch_dir / "audio.wav"
        log("converting to 16 kHz mono WAV")
        audio.to_wav(local, wav)
        samples = audio.read_wav(wav)

    duration_ms = audio.duration_ms(samples)
    log(f"diarizing {duration_ms / 1000:.1f} s of audio")
    tracks = Diarizer(ecapa, segmentation).diarize(samples)

    plan = plan_regions(tracks.turns)
    results = []
    for piece in plan:
        log(f"transcribing slice {piece.index + 1}/{len(plan)}")
        results.append((piece, transcriber.transcribe(audio.span(samples, piece.start_ms, piece.end_ms))))

    cues = build_cues(results)
    document = cue_document(
        source=source, duration_ms=duration_ms, language=language, cues=cues, tracks=tracks
    )
    log(f"{len(cues)} cues, {len(document['speakers'])} speakers")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio.source_name(source)
    json_path = out_dir / f"{stem}.cues.json"
    vtt_path = out_dir / f"{stem}.vtt"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    vtt_path.write_text(render_vtt(document), encoding="utf-8")
    return json_path, vtt_path
