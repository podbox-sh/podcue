"""Download episode audio to a 16 kHz mono WAV that `podcue start` can take.

A URL whose path ends in an audio or video suffix is fetched directly. Anything else (a watch
page, a youtu.be link) goes through yt-dlp, which is part of the recap extra.
"""

from __future__ import annotations

import os
import secrets
import sys
import tempfile
import urllib.parse
from pathlib import Path

from . import __version__, audio
from .recap import RecapError

MEDIA_SUFFIXES = {
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".mpga", ".wma", ".aiff", ".caf",
    ".mp4", ".m4v", ".mkv", ".webm", ".mov", ".mpeg", ".mpg",
}


def is_direct_media(url: str) -> bool:
    """True when the URL path names a media file. The query string is not part of the suffix."""
    suffix = Path(urllib.parse.urlsplit(url).path).suffix.lower()
    return suffix in MEDIA_SUFFIXES


def _ytdlp_missing() -> str:
    return (
        "downloading this page needs yt-dlp from the recap extra. "
        f'Run: uvx --python 3.12 --from "podcue[recap]=={__version__}" podcue fetch <URL> --out DIR'
    )


def load_ytdlp():
    """Import yt-dlp. Kept off the module import so a direct URL works without the extra."""
    try:
        from yt_dlp import YoutubeDL
    except ModuleNotFoundError:
        raise RecapError(_ytdlp_missing()) from None
    return YoutubeDL


class _Logger:
    """yt-dlp logger that stays off stdout. stdout is the one JSON line."""

    def debug(self, message: str) -> None:
        return None

    def info(self, message: str) -> None:
        print(f"podcue: {message}", file=sys.stderr, flush=True)

    def warning(self, message: str) -> None:
        print(f"podcue: {message}", file=sys.stderr, flush=True)

    def error(self, message: str) -> None:
        print(f"podcue: {message}", file=sys.stderr, flush=True)


def download_with_ytdlp(url: str, directory: Path) -> Path:
    """Best audio stream for `url`, written into `directory`. Raises RecapError on failure."""
    youtube_dl = load_ytdlp()
    before = {path.resolve() for path in directory.iterdir()}
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(directory / "download.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "restrictfilenames": True,
        "logger": _Logger(),
    }
    try:
        with youtube_dl(options) as downloader:
            downloader.download([url])
    except RecapError:
        raise
    except Exception as error:
        raise RecapError(f"yt-dlp could not download {url}: {error}") from None
    created = [path for path in directory.iterdir() if path.resolve() not in before and path.is_file()]
    if len(created) != 1:
        raise RecapError(f"yt-dlp did not write one audio file for {url}")
    return created[0]


def fetch(url: str, out_dir: Path) -> dict:
    """Write `<out_dir>/audio.wav` and return the fetch result. `out_dir` is created."""
    if not audio.is_url(url):
        raise RecapError(f"{url} is not an http(s) URL")
    out_dir = out_dir.expanduser().resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RecapError(f"could not create {out_dir}: {error}") from None
    wav = out_dir / "audio.wav"
    try:
        with tempfile.TemporaryDirectory(prefix="podcue-fetch-") as scratch:
            scratch_dir = Path(scratch)
            if is_direct_media(url):
                source = audio.download(url, scratch_dir)
                via = "direct"
            else:
                print(f"podcue: downloading {url}", file=sys.stderr, flush=True)
                source = download_with_ytdlp(url, scratch_dir)
                via = "yt-dlp"
            partial = out_dir / f".audio.{secrets.token_hex(4)}.wav"
            try:
                audio.to_wav(source, partial)
                os.replace(partial, wav)
            finally:
                partial.unlink(missing_ok=True)
    except RecapError:
        raise
    except Exception as error:
        message = str(error) or type(error).__name__
        raise RecapError(message) from None
    samples = audio.read_wav(wav)
    if len(samples) == 0:
        raise RecapError(f"{url} has no decodable audio", reason="not-audio")
    return {
        "kind": "fetched",
        "url": url,
        "path": str(wav),
        "durationMs": audio.duration_ms(samples),
        "via": via,
    }
