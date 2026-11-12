"""Build a recap: snap clips to cue times, narrate with Kokoro, stitch, and check speakers.

`script.json` is the plan `podcue build` reads. A clip's `start` and `end` are seconds or
`[h:]mm:ss[.s]`. Each edge snaps to a word time spread across the cue that contains it, the same
way a clip of a few cues covers consecutive lines of at most 12 s. Kokoro, soundfile, and yt-dlp stay
in the `recap` extra and are imported only when a narration or a page download needs them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .audio import is_url
from .store import Episode, load_labels, write_text

# Mix rate. Kokoro's voices are 24 kHz, and every segment is resampled to match before the join.
SAMPLE_RATE = 24_000
# EBU R128 targets. Two passes: measure, then loudnorm with the measured values.
TARGET_I = -16.0
TARGET_TP = -1.5
TARGET_LRA = 11.0
# Silence inserted between segments, in seconds.
GAP_S = 0.4
DEFAULT_VOICE = "am_michael"
# PodBox preset names the script may already use, mapped onto Kokoro voice ids.
VOICE_ALIASES = {
    "american-male-puck": "am_puck",
    "american-female-heart": "af_heart",
}
# A second voice counts when it covers this much of the clip. A short interjection does not.
OTHER_MIN_S = 1.0
OTHER_MIN_FRAC = 0.08
CLIP_WARN_UNDER_S = 4.0
CLIP_WARN_OVER_S = 45.0
KOKORO_REPO = "hexgrad/Kokoro-82M"

SCRIPT_SHAPE = (
    "script.json must be { voice?, voice_preset?, output?, segments: [ "
    "{type: narration, text, voice?, voice_preset?} or {type: clip, start, end, speaker?} ] } "
    "or an array of those segments"
)


class RecapError(Exception):
    """A fetch or build failure. The CLI prints it as one `{"kind": "error"}` line."""

    def __init__(self, message: str, **fields: object):
        super().__init__(message)
        self.fields = fields


def _kokoro_missing() -> str:
    """Install hint. Kokoro 0.9.4 declares Requires-Python <3.13, so 3.13 and 3.14 cannot import it."""
    newer = ""
    if sys.version_info >= (3, 13):
        newer = " Kokoro 0.9.4 supports Python 3.12, not 3.13 or 3.14."
    return (
        f"narration needs the recap extra.{newer} "
        f'Run: uvx --python 3.12 --from "podcue[recap]=={__version__}" podcue build <episodeId> <script.json>'
    )


def log(message: str) -> None:
    print(f"podcue: {message}", file=sys.stderr, flush=True)


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def parse_ts(value: object) -> float:
    """Seconds, or `[h:]mm:ss[.s]`. Raises ValueError when the value is not a time."""
    if isinstance(value, bool):
        raise ValueError("not a time")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("not a time")
        try:
            parts = [float(part) for part in text.split(":")]
        except ValueError:
            raise ValueError("not a time") from None
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60.0 + part
    else:
        raise ValueError("not a time")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("not a time")
    return seconds


def _voice_field(segment: dict, field: str, index: int) -> str | None:
    if field not in segment or segment[field] is None:
        return None
    value = segment[field]
    if not isinstance(value, str):
        raise RecapError(f"segment {index}: {field} must be a string")
    return value.strip() or None


def resolve_voice(script_voice: str | None, script_preset: str | None, segment: dict | None, index: int) -> str:
    """Kokoro id for one narration. A segment voice wins, then the script voice, then `am_michael`."""
    raw = None
    if segment is not None:
        raw = _voice_field(segment, "voice", index) or _voice_field(segment, "voice_preset", index)
    if not raw:
        raw = script_voice or script_preset
    if not raw:
        return DEFAULT_VOICE
    return VOICE_ALIASES.get(raw, raw)


@dataclass(frozen=True)
class Script:
    voice: str | None
    voice_preset: str | None
    output: str | None
    segments: list[dict]


def _script_voice(body: dict, field: str) -> str | None:
    if field not in body or body[field] is None:
        return None
    value = body[field]
    if not isinstance(value, str):
        raise RecapError(f"{field} must be a string")
    return value.strip() or None


def parse_script(value: object) -> Script:
    """Accept the object form and a bare segment array. Unknown keys are ignored."""
    if isinstance(value, list):
        body: dict = {"segments": value}
        voice = voice_preset = output = None
    elif isinstance(value, dict):
        body = value
        voice = _script_voice(body, "voice")
        voice_preset = _script_voice(body, "voice_preset")
        output_value = body.get("output")
        if output_value is None:
            output = None
        elif isinstance(output_value, str) and output_value.strip():
            output = output_value.strip()
        else:
            raise RecapError("output must be a file name")
    else:
        raise RecapError(SCRIPT_SHAPE)
    raw_segments = body.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise RecapError(SCRIPT_SHAPE)
    segments = []
    for index, segment in enumerate(raw_segments, 1):
        if not isinstance(segment, dict):
            raise RecapError(f"segment {index}: must be an object")
        kind = segment.get("type")
        if kind == "narration":
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RecapError(f"segment {index}: narration text must be a non-empty string")
            resolve_voice(voice, voice_preset, segment, index)  # reject a non-string voice now
        elif kind == "clip":
            start = _time(segment.get("start"), index, "start")
            end = _time(segment.get("end"), index, "end")
            if end <= start:
                raise RecapError(f"segment {index}: end must be after start")
            speaker = segment.get("speaker", "?")
            if speaker is None:
                speaker = "?"
            if not isinstance(speaker, str) or not speaker.strip():
                raise RecapError(f"segment {index}: speaker must be a string")
        else:
            raise RecapError(f"segment {index}: type must be narration or clip")
        segments.append(segment)
    return Script(voice, voice_preset, output, segments)


def _time(value: object, index: int, field: str) -> float:
    try:
        return parse_ts(value)
    except ValueError:
        raise RecapError(f"segment {index}: {field} must be seconds or [h:]mm:ss[.s]") from None


def load_script(path: Path) -> Script:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RecapError(f"{path} is not a file") from None
    except OSError as error:
        raise RecapError(f"{path} could not be read: {error}") from None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise RecapError(f"{path} is not JSON: {error}") from None
    return parse_script(value)


def words_from_cues(rows: list[dict]) -> list[tuple[float, float, str]]:
    """Spread each cue's words across its start and end.

    Cues carry no per-word timestamps. The spacing is by character length, so a clip edge still
    lands on a word boundary inside the cue instead of mid-word.
    """
    words: list[tuple[float, float, str]] = []
    for row in rows:
        start = row["startMs"] / 1000.0
        end = row["endMs"] / 1000.0
        tokens = (row.get("text") or "").split()
        if not tokens or end <= start:
            continue
        weight = sum(len(token) + 1 for token in tokens) or 1
        cursor = start
        for token in tokens:
            span = (end - start) * (len(token) + 1) / weight
            words.append((round(cursor, 2), round(min(end, cursor + span), 2), f" {token}"))
            cursor += span
    return words


def snap(words: list[tuple[float, float, str]], start: float, end: float) -> tuple[float, float, int, int]:
    """Snap to the nearest word edges, then pad halfway into the neighbouring silence.

    Returns `(start, end, first_word, last_word)`. The pad is at most 0.25 s before the first
    word and 0.35 s after the last, and never crosses the next word.
    """
    if not words:
        raise RecapError("the episode has no cue text to snap a clip to")
    first = min(range(len(words)), key=lambda i: abs(words[i][0] - start))
    last = min(range(len(words)), key=lambda i: abs(words[i][1] - end))
    if last < first:
        last = first
    previous_end = words[first - 1][1] if first else words[first][0] - 0.5
    next_start = words[last + 1][0] if last + 1 < len(words) else words[last][1] + 0.5
    snapped_start = words[first][0] - min(0.25, max(0.0, (words[first][0] - previous_end) / 2))
    snapped_end = words[last][1] + min(0.35, max(0.05, (next_start - words[last][1]) / 2))
    return round(snapped_start, 2), round(snapped_end, 2), first, last


def speaker_cover(rows: list[dict], labels: list[dict], start: float, end: float) -> dict[str, float]:
    """Seconds of each labeled voice inside `[start, end]`. An unlabeled group keeps its group id."""
    names = {item["groupId"]: item["label"] for item in labels}
    cover: dict[str, float] = {}
    for row in rows:
        overlap_start = max(row["startMs"] / 1000.0, start)
        overlap_end = min(row["endMs"] / 1000.0, end)
        if overlap_end <= overlap_start:
            continue
        raw = row.get("speaker") or row["groupId"]
        if raw in ("uncertain", "mixed"):
            name = raw
        else:
            name = names.get(raw) or names.get(row["groupId"]) or raw
        cover[name] = cover.get(name, 0.0) + (overlap_end - overlap_start)
    return cover


def group_ids_in(rows: list[dict], start: float, end: float) -> list[str]:
    found: list[str] = []
    for row in rows:
        overlap_start = max(row["startMs"] / 1000.0, start)
        overlap_end = min(row["endMs"] / 1000.0, end)
        if overlap_end > overlap_start and row["groupId"] not in found:
            found.append(row["groupId"])
    return found


def clip_speakers(rows: list[dict], labels: list[dict], start: float, end: float) -> dict:
    """How many voices are in the clip. A second voice must cover at least 1 s and 8% of the span."""
    cover = speaker_cover(rows, labels, start, end)
    if not cover:
        return {"speakers": 0, "otherSeconds": 0.0, "dominant": None}
    dominant = max(cover, key=cover.get)
    other = sum(seconds for name, seconds in cover.items() if name != dominant)
    span = max(end - start, 0.001)
    significant = [
        name for name, seconds in cover.items()
        if name != dominant and seconds >= OTHER_MIN_S and seconds >= OTHER_MIN_FRAC * span
    ]
    return {"speakers": 1 + len(significant), "otherSeconds": round(other, 1), "dominant": dominant}


def clip_flags(length: float, speakers: dict, cues: list[dict] | None = None) -> list[str]:
    """Flags for one clip. Length is a warning only under 4 s or over 45 s.

    Speaker identity is each cue's `speaker` (an uncertain or mixed cue counts even when it
    shares a group id). A second voice still has to cover 1 s and 8% of the clip.
    """
    flags = []
    present = cues or []
    if any((cue.get("speaker") or cue.get("groupId")) in ("uncertain", "mixed") for cue in present):
        flags.append("uncertain or mixed speaker in clip")
    count = speakers["speakers"]
    if count == 0:
        flags.append("no speech in clip")
    elif count > 1:
        flags.append(f"{count} speakers in clip (other speaker {speakers['otherSeconds']}s)")
    if length < CLIP_WARN_UNDER_S:
        flags.append(f"length {length}s under 4s")
    elif length > CLIP_WARN_OVER_S:
        flags.append(f"length {length}s over 45s")
    return flags


def fmt_ts(seconds: float) -> str:
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    if shutil.which("ffmpeg") is None:
        raise RecapError("ffmpeg is not installed or not on PATH")
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip() or f"exit code {proc.returncode}"
        raise RecapError(detail[-2000:])
    return proc


def measure_loudness(path: Path) -> dict:
    """One loudnorm measurement. The JSON block is the last object ffmpeg prints on stderr."""
    if shutil.which("ffmpeg") is None:
        raise RecapError("ffmpeg is not installed or not on PATH")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    text = proc.stderr or ""
    start = text.rfind("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        detail = text.strip()[-500:] or f"exit code {proc.returncode}"
        raise RecapError(f"ffmpeg loudnorm produced no measurement: {detail}")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise RecapError(f"ffmpeg loudnorm measurement was not JSON: {error}") from None
    if not isinstance(data, dict) or "input_i" not in data:
        raise RecapError("ffmpeg loudnorm measurement was incomplete")
    return data


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def loudnorm_to(source: Path, dest: Path) -> float:
    """Two-pass loudnorm to -16 LUFS / -1.5 dBTP. Returns the source integrated loudness."""
    measured = measure_loudness(source)
    audio_filter = (
        f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:"
        f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true,aresample={SAMPLE_RATE}"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-af", audio_filter,
        "-ac", "1", "-ar", str(SAMPLE_RATE), str(dest),
    ])
    lufs = _finite(measured["input_i"])
    if lufs is None:
        raise RecapError(f"ffmpeg loudnorm measured a non-finite loudness for {source.name}")
    return lufs


def cut_clip(source: Path, start: float, end: float, dest: Path) -> None:
    """Cut `[start, end]` to 24 kHz mono with a short fade in and out."""
    duration = end - start
    fade_out = max(0.05, min(0.12, duration / 4))
    fade = f"afade=t=in:d=0.06,afade=t=out:st={max(0.0, duration - fade_out):.2f}:d={fade_out:.2f}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(source),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-af", fade, str(dest),
    ])


_pipeline = None


def _samples(audio: object):
    """Kokoro yields a torch tensor. soundfile wants a numpy array."""
    detach = getattr(audio, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return audio


def narrate(text: str, voice: str, dest: Path) -> None:
    """Synthesize `text` with Kokoro at 24 kHz into `dest`. The pipeline is kept for the process."""
    global _pipeline
    # Hugging Face counts toolkit usage unless this is set. Leave a value the user set alone.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Imported here so `import podcue.recap` works without the recap extra (Kokoro pulls torch).
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ModuleNotFoundError:
        raise RecapError(_kokoro_missing()) from None
    if _pipeline is None:
        log(f"loading Kokoro voice {voice}")
        _pipeline = KPipeline(lang_code="a", repo_id=KOKORO_REPO, device="cpu")
    pieces = []
    for result in _pipeline(text.strip(), voice=voice, split_pattern=r"\n+"):
        audio = result[2]
        if audio is None:
            continue
        pieces.append(_samples(audio))
    if not pieces:
        raise RecapError("Kokoro returned no audio")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), np.concatenate(pieces), SAMPLE_RATE)


def _ensure(path: Path, produce) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        produce(path)


def probe_duration_s(path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise RecapError("ffprobe is not installed or not on PATH")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip() or f"exit code {proc.returncode}"
        raise RecapError(f"ffprobe could not read {path.name}: {detail[-500:]}")
    try:
        seconds = float(proc.stdout.strip())
    except ValueError:
        raise RecapError(f"ffprobe returned no duration for {path.name}") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise RecapError(f"ffprobe returned no duration for {path.name}")
    return seconds


def _concat_line(path: Path) -> str:
    escaped = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped}'"


def stitch(paths: list[Path], dest: Path, work: Path) -> None:
    """Join normalized segments with 0.4 s gaps, loudnorm the mix, and encode it."""
    if not paths:
        raise RecapError("the script has no segments")
    work.mkdir(parents=True, exist_ok=True)
    gap = work / f"gap-{GAP_S}.wav"
    _ensure(gap, lambda target: run_ffmpeg([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
        "-t", str(GAP_S), str(target),
    ]))
    lines = []
    for index, path in enumerate(paths):
        lines.append(_concat_line(path))
        if index < len(paths) - 1:
            lines.append(_concat_line(gap))
    listing = work / "concat.txt"
    body = "\n".join(lines)
    listing.write_text(f"{body}\n", encoding="utf-8")
    joined = work / "concat.wav"
    run_ffmpeg([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c", "copy", str(joined),
    ])
    normalized = work / "final.wav"
    loudnorm_to(joined, normalized)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() == ".wav":
        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(normalized),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dest),
        ])
    else:
        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(normalized),
            "-codec:a", "libmp3lame", "-b:a", "128k", str(dest),
        ])


def episode_audio(episode: Episode) -> Path:
    """The file `start` was given. A URL is not kept after start, so the recap cannot cut it."""
    try:
        manifest = json.loads(episode.manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise RecapError("episode.json could not be read") from None
    source = manifest.get("source") if isinstance(manifest, dict) else None
    if not isinstance(source, str) or not source:
        raise RecapError("episode.json has no source")
    if is_url(source):
        raise RecapError(
            "this episode was started from a URL, and that download is not kept. "
            "Run fetch, then start on the wav it writes."
        )
    path = Path(source).expanduser()
    if not path.is_file():
        raise RecapError(f"{source} is not a file", reason="not-found")
    return path


def output_path(script: Script, flag: str | None) -> Path:
    if flag:
        return Path(flag).expanduser().resolve()
    return Path(script.output or "produced_summary.mp3").expanduser().resolve()


def _normalize(raw: Path, dest: Path) -> tuple[float, float]:
    if not dest.is_file() or dest.stat().st_size == 0:
        lufs_in = loudnorm_to(raw, dest)
    else:
        lufs_in = _finite(measure_loudness(raw)["input_i"])
        if lufs_in is None:
            raise RecapError(f"ffmpeg loudnorm measured a non-finite loudness for {raw.name}")
    lufs_out = _finite(measure_loudness(dest)["input_i"])
    if lufs_out is None:
        raise RecapError(f"ffmpeg loudnorm measured a non-finite loudness for {dest.name}")
    return lufs_in, lufs_out


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(episode_id: str, source: Path, dest: Path, duration_s: float, lufs: float, true_peak: float,
                  segments: list[dict]) -> str:
    flags = [flag for segment in segments for flag in segment["flags"]]
    clip_seconds = round(sum(segment.get("length", 0.0) for segment in segments if segment["type"] == "clip"), 1)
    lines = [
        "# podcue recap",
        "",
        f"- Episode: `{episode_id}`",
        f"- Source: `{source}`",
        f"- Output: `{dest}`",
        f"- Duration: {duration_s:.1f}s ({fmt_ts(duration_s)}), clips {clip_seconds}s",
        f"- Final loudness: {lufs:.1f} LUFS, true peak {true_peak:.1f} dBTP",
        f"- Flags: {len(flags)}",
        "",
        "Clip edges are the cue word times the requested range snapped to. "
        "A clip under 4 seconds or over 45 seconds is flagged, as is an uncertain or mixed cue.",
        "",
        "| # | type | speaker | voice | range | length | LUFS in→out | flags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for segment in segments:
        if segment["type"] == "clip":
            span = f"{segment['startTs']}–{segment['endTs']}"
            voice = f"{segment['speakers']} spk, other {segment['otherSeconds']}s"
            speaker = segment["speaker"]
            length = segment["length"]
        else:
            span = ""
            voice = segment["voice"]
            speaker = ""
            length = ""
        lufs_cell = f"{segment['lufsIn']:.1f}→{segment['lufsOut']:.2f}"
        body = " | ".join([
            str(segment["index"]),
            segment["type"],
            _cell(speaker),
            _cell(voice),
            span,
            str(length),
            lufs_cell,
            _cell("; ".join(segment["flags"])),
        ])
        lines.append(f"| {body} |")
    lines.extend(["", "## Clip text", ""])
    clips = [segment for segment in segments if segment["type"] == "clip"]
    if not clips:
        lines.append("No clips.")
    for segment in clips:
        lines.append(
            f"- **{segment['index']}. {segment['speaker']}, {segment['startTs']}–{segment['endTs']}**: "
            f"\"{segment['text']}\""
        )
    text = "\n".join(lines)
    return f"{text}\n"


def _public_segment(segment: dict) -> dict:
    """The JSON segment. Drop the fields that only the markdown table needs."""
    hidden = {"startTs", "endTs"}
    return {key: value for key, value in segment.items() if key not in hidden}


def build(episode: Episode, rows: list[dict], script_path: Path, out_flag: str | None) -> dict:
    """Cut, narrate, stitch, and write `report.md` next to the audio. Flags do not fail the build."""
    script = load_script(script_path)
    source = episode_audio(episode)
    dest = output_path(script, out_flag)
    labels = load_labels(episode)
    words = words_from_cues(rows)
    cache = episode.dir / "recap-cache"
    cache.mkdir(parents=True, exist_ok=True)
    segments: list[dict] = []
    normalized: list[Path] = []
    for index, segment in enumerate(script.segments, 1):
        kind = segment["type"]
        if kind == "narration":
            voice = resolve_voice(script.voice, script.voice_preset, segment, index)
            text = segment["text"].strip()
            log(f"narration {index} ({voice})")
            raw = cache / f"tts_{digest('tts', voice, text)}.wav"
            _ensure(raw, lambda target, text=text, voice=voice: narrate(text, voice, target))
            row = {"index": index, "type": "narration", "text": text, "voice": voice, "flags": []}
        elif kind == "clip":
            requested_start = parse_ts(segment["start"])
            requested_end = parse_ts(segment["end"])
            start, end, first, last = snap(words, requested_start, requested_end)
            if end <= start:
                raise RecapError(f"segment {index}: snapped clip is empty")
            length = round(end - start, 1)
            text = "".join(word[2] for word in words[first:last + 1]).strip()
            speakers = clip_speakers(rows, labels, start, end)
            overlapping = [
                row for row in rows
                if row["endMs"] / 1000.0 > start and row["startMs"] / 1000.0 < end
            ]
            flags = clip_flags(length, speakers, overlapping)
            log(f"clip {index} {fmt_ts(start)}–{fmt_ts(end)} ({length}s)")
            stat = source.stat()
            raw = cache / (
                "clip_"
                f"{digest('clip', str(source.resolve()), str(stat.st_size), str(stat.st_mtime_ns), f'{start:.2f}', f'{end:.2f}')}"
                ".wav"
            )
            _ensure(raw, lambda target, start=start, end=end: cut_clip(source, start, end, target))
            speaker = segment.get("speaker") or "?"
            row = {
                "index": index,
                "type": "clip",
                "speaker": speaker if isinstance(speaker, str) else "?",
                "requestedStart": segment["start"],
                "requestedEnd": segment["end"],
                "start": start,
                "end": end,
                "startMs": int(round(start * 1000)),
                "endMs": int(round(end * 1000)),
                "startTs": fmt_ts(start),
                "endTs": fmt_ts(end),
                "length": length,
                "text": text,
                "groupIds": group_ids_in(rows, start, end),
                "speakers": speakers["speakers"],
                "otherSeconds": speakers["otherSeconds"],
                "flags": flags,
            }
        else:
            raise RecapError(f"segment {index}: type must be narration or clip")
        norm = cache / f"norm_{digest(str(raw), str(raw.stat().st_size), str(TARGET_I))}.wav"
        lufs_in, lufs_out = _normalize(raw, norm)
        row["lufsIn"] = round(lufs_in, 2)
        row["lufsOut"] = round(lufs_out, 2)
        segments.append(row)
        normalized.append(norm)
    log(f"stitching {len(normalized)} segments")
    stitch(normalized, dest, cache)
    duration_s = probe_duration_s(dest)
    final = measure_loudness(dest)
    lufs = _finite(final["input_i"])
    true_peak = _finite(final["input_tp"])
    if lufs is None or true_peak is None:
        raise RecapError("ffmpeg loudnorm measured a non-finite loudness for the mix")
    report_path = dest.parent / "report.md"
    write_text(report_path, render_report(episode.id, source, dest, duration_s, lufs, true_peak, segments))
    flags = [f"segment {segment['index']}: {flag}" for segment in segments for flag in segment["flags"]]
    return {
        "kind": "built",
        "episodeId": episode.id,
        "output": str(dest),
        "report": str(report_path),
        "durationMs": int(round(duration_s * 1000)),
        "loudness": {"lufs": round(lufs, 2), "truePeak": round(true_peak, 2)},
        "flags": flags,
        "segments": [_public_segment(segment) for segment in segments],
    }
