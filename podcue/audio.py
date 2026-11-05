"""Fetch a source, convert it to 16-bit mono 16 kHz WAV with ffmpeg, and read the samples."""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import numpy as np

RATE = 16_000


def is_url(source: str) -> bool:
    return urllib.parse.urlsplit(source).scheme in ("http", "https")


def source_name(source: str) -> str:
    """The output stem: the file name without its extension, for a path or a URL."""
    path = urllib.parse.urlsplit(source).path if is_url(source) else source
    stem = Path(urllib.parse.unquote(path)).stem
    return stem or "episode"


def download(url: str, directory: Path, *, opener=urllib.request.urlopen) -> Path:
    suffix = Path(urllib.parse.urlsplit(url).path).suffix
    target = directory / f"download{suffix}"
    print(f"podcue: downloading {url}", file=sys.stderr)
    request = urllib.request.Request(url, headers={"User-Agent": "podcue"})
    with opener(request) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    return target


def to_wav(source: Path, target: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH")
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", "-f", "wav", str(target)]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"ffmpeg could not convert {source}: {detail}")


def read_wav(path: Path) -> np.ndarray:
    """Float32 samples in [-1, 1). Only the 16-bit mono 16 kHz WAV that `to_wav` writes."""
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2 or reader.getframerate() != RATE:
            raise ValueError(f"{path} is not 16-bit mono 16 kHz WAV")
        pcm = reader.readframes(reader.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768.0)


def duration_ms(samples: np.ndarray) -> int:
    return len(samples) * 1000 // RATE


def span(samples: np.ndarray, start_ms: int, end_ms: int) -> np.ndarray:
    return samples[start_ms * RATE // 1000 : end_ms * RATE // 1000]
