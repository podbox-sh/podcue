"""`podcue start`: fetch, convert, hash, then diarize and transcribe into the episode folder.

The episode id is `ep_` + the first 16 hex digits of the sha256 of the converted WAV, so the
same audio always lands in the same folder. When that folder already has cues.json, start
returns it without loading a model, and updates `source` to the path it was given this time.
`--fresh` recomputes that one episode and leaves every other episode and the weight cache.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import audio, weights
from .cues import Word, build_cues, cue_rows, group_table, render_vtt
from .diarize import Diarizer, Tracks
from .regions import Slice, plan_regions
from .store import Episode, lock_pid, read_json, release, retire_labels, try_lock, write_json, write_text
from .transcribe import LANGUAGES, asr_record, open_transcriber, quiet_hub, transcribe_slice
from .transcript import assign_words


class Rejected(Exception):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def log(message: str) -> None:
    print(f"podcue: {message}", file=sys.stderr, flush=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _word_row(word: Word) -> dict:
    return {"text": word.text, "start": word.start, "end": word.end}


def _words_from_saved(saved: dict) -> list[Word] | None:
    try:
        return [
            Word(str(row["text"]), float(row["start"]), float(row["end"]))
            for row in saved.get("words") or []
        ]
    except (KeyError, TypeError, ValueError):
        return None


def _slice_matches(saved: dict, piece: Slice) -> bool:
    return (
        saved.get("startMs") == piece.start_ms
        and saved.get("endMs") == piece.end_ms
        and saved.get("keepStartMs") == piece.keep_start_ms
        and saved.get("keepEndMs") == piece.keep_end_ms
        and saved.get("region") == piece.region
    )


def load_slice_checkpoint(path: Path, audio_id: str, language: str) -> dict[int, dict]:
    stored = read_json(path)
    if not isinstance(stored, dict):
        return {}
    if stored.get("version") != 1 or stored.get("audioSha256") != audio_id or stored.get("language") != language:
        log("slice checkpoint does not match this audio; starting over")
        return {}
    found: dict[int, dict] = {}
    for item in stored.get("slices") or []:
        if not isinstance(item, dict) or not isinstance(item.get("index"), int):
            continue
        if item.get("state") not in ("done", "skipped-empty"):
            continue
        found[item["index"]] = item
    return found


def save_slice_checkpoint(path: Path, audio_id: str, language: str, slices: list[dict]) -> None:
    write_json(path, {
        "version": 1,
        "audioSha256": audio_id,
        "language": language,
        "slices": slices,
    })


def run_slices(
    samples, plan: list[Slice], transcriber, checkpoint: Path | None, report=None,
    audio_id: str = "", language: str = "",
) -> list[tuple[Slice, list[Word]]]:
    saved = load_slice_checkpoint(checkpoint, audio_id, language) if checkpoint is not None else {}
    reused = sum(
        1 for piece in plan
        if (prior := saved.get(piece.index)) is not None and _slice_matches(prior, piece)
    )
    if reused:
        log(f"resuming transcription, {reused} of {len(plan)} slices already finished")
    results: list[tuple[Slice, list[Word]]] = []
    finished: list[dict] = []
    total = len(plan)
    for piece in plan:
        if report is not None:
            report("transcribe", piece.index * 100 // total if total else 99)
        prior = saved.get(piece.index)
        if prior is not None and _slice_matches(prior, piece):
            restored = _words_from_saved(prior)
            if restored is not None:
                results.append((piece, restored))
                finished.append(prior)
                continue
        span = audio.span(samples, piece.start_ms, piece.end_ms)
        log(f"transcribing slice {piece.index + 1}/{len(plan)}")
        outcome = transcribe_slice(transcriber, span)
        if outcome.state == "skipped-empty":
            log(f"skipping slice {piece.index + 1}/{len(plan)} (no speech to align)")
        words = list(outcome.words)
        row = {
            "index": piece.index,
            "startMs": piece.start_ms,
            "endMs": piece.end_ms,
            "keepStartMs": piece.keep_start_ms,
            "keepEndMs": piece.keep_end_ms,
            "region": piece.region,
            "state": outcome.state,
            "words": [_word_row(word) for word in words],
        }
        finished.append(row)
        if checkpoint is not None:
            save_slice_checkpoint(checkpoint, audio_id, language, finished)
        results.append((piece, words))
    return results


def transcribe(
    samples, language: str, report, transcript_words=None,
    checkpoint: Path | None = None, audio_id: str = "",
) -> tuple[list, Tracks]:
    """Diarize, then either transcribe each slice or place `transcript_words` into the slices.

    `transcript_words` is absolute `(startMs, endMs, text)`. When it is set, no ASR model loads.
    """
    report("weights", 0)
    transcriber = None if transcript_words is not None else open_transcriber(language)
    ecapa = weights.ensure(weights.ECAPA)
    segmentation = weights.ensure(weights.SEGMENTATION)

    report("diarize", 0)
    log(f"diarizing {audio.duration_ms(samples) / 1000:.1f} s of audio")
    tracks = Diarizer(ecapa, segmentation).diarize(samples)

    plan = plan_regions(tracks.turns)
    if transcript_words is not None:
        results, dropped = assign_words(plan, transcript_words)
        if dropped:
            log(f"{dropped} transcript words started outside detected speech and were left out")
        report("transcribe", 99)
    else:
        results = run_slices(
            samples, plan, transcriber, checkpoint, report,
            audio_id=audio_id, language=language,
        )
    return build_cues(results, language), tracks


def recorded_source(source: str) -> str:
    """A local file is stored as an absolute path. A URL is stored as given."""
    if audio.is_url(source):
        return source
    return str(Path(source).expanduser().resolve())


def refresh_source(episode: Episode, recorded: str) -> None:
    """Point an existing episode at the file this start was given. Other fields stay."""
    manifest = read_json(episode.manifest)
    if not isinstance(manifest, dict) or manifest.get("source") == recorded:
        return
    manifest["source"] = recorded
    write_json(episode.manifest, manifest)


def start(source: str, language: str, root: Path, transcript_words=None, fresh: bool = False) -> tuple[Episode, bool]:
    """Returns the episode and whether this call wrote its cues. Raises Rejected or the failure.

    `transcript_words` skips built-in ASR. `language` is then any code, not only the built-in list.
    `fresh` ignores the saved transcript, cues, and groups for this episode only,
    and moves labels.json to labels.prev.json.
    """
    quiet_hub()
    if transcript_words is None and language not in LANGUAGES:
        raise Rejected("unknown-language", f"unknown language {language}; use one of {' '.join(LANGUAGES)}")
    if not language or not isinstance(language, str):
        raise Rejected("unknown-language", "language must be a non-empty code")
    local = None
    if not audio.is_url(source):
        local = Path(source).expanduser()
        if not local.is_file():
            raise Rejected("not-found", f"{source} is not a file")
    recorded = recorded_source(source)

    with tempfile.TemporaryDirectory(prefix="podcue-") as scratch:
        scratch_dir = Path(scratch)
        if local is None:
            local = audio.download(source, scratch_dir)
        wav = scratch_dir / "audio.wav"
        log("converting to 16 kHz mono WAV")
        audio.to_wav(local, wav)
        episode = Episode(root, "ep_" + sha256_file(wav)[:16])
        if episode.ready() and not fresh:
            refresh_source(episode, recorded)
            return episode, False
        samples = audio.read_wav(wav)
    if len(samples) == 0:
        raise Rejected("not-audio", f"{source} has no decodable audio")

    episode.dir.mkdir(parents=True, exist_ok=True)
    fd = try_lock(episode.start_lock)
    if fd is None:
        raise Rejected("running", f"another start of {episode.id} is running (pid {lock_pid(episode.start_lock)})")
    try:
        if episode.ready() and not fresh:
            refresh_source(episode, recorded)
            return episode, False
        state = {"stage": ""}

        def report(stage: str, progress: int) -> None:
            state["stage"] = stage
            write_json(episode.status, {"pid": os.getpid(), "stage": stage, "progress": progress})

        try:
            report("weights", 0)  # before episode.json, so `episode` never sees a folder with neither
            episode.failed.unlink(missing_ok=True)
            if fresh:
                episode.slices.unlink(missing_ok=True)
            # createdAt is this run. An earlier failed start of the same audio leaves episode.json
            # behind; reusing that stamp made a later job look hours older than it was.
            started_at = now()
            write_json(episode.manifest, {
                "episodeId": episode.id,
                "source": recorded,
                "durationMs": audio.duration_ms(samples),
                "language": language,
                "createdAt": started_at,
                "asr": asr_record(language, transcript_words is not None),
            })
            cues, tracks = transcribe(
                samples, language, report, transcript_words,
                checkpoint=episode.slices, audio_id=episode.id,
            )
            if fresh:
                retire_labels(episode)
            report("write", 99)
            rows = cue_rows(cues, tracks)
            write_text(episode.transcript, render_vtt(rows))
            write_json(episode.groups, group_table(rows))
            write_json(episode.cues, rows)  # last: its presence means the episode is ready
            episode.slices.unlink(missing_ok=True)
            log(f"{len(rows)} cues, {len({row['groupId'] for row in rows})} groups")
        except BaseException as error:
            message = str(error) or type(error).__name__
            write_json(episode.failed, {"message": message, "stage": state["stage"], "at": now()})
            raise
        finally:
            episode.status.unlink(missing_ok=True)
    finally:
        release(fd, episode.start_lock)
    return episode, True
