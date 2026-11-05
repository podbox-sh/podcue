"""Pinned diarization weights: download once into the user cache, verify size and SHA-256.

The ECAPA model card has no license field. WeSpeaker says its pretrained models follow the
dataset license, and VoxCeleb is CC BY 4.0, so the attribution travels with the file.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import platformdirs


@dataclass(frozen=True)
class Pin:
    label: str
    file: str
    url: str
    bytes: int
    sha256: str
    license: str
    attribution: str


ECAPA = Pin(
    label="ECAPA",
    file="voxceleb_ECAPA512.onnx",
    url="https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512/resolve/89426e111abc9ffcc40279baf808e087693c6e9b/voxceleb_ECAPA512.onnx",
    bytes=24_861_931,
    sha256="f901ae3571912bad4e11e157744cb8b7d77a0b46923b83c410bdbb6c77d69736",
    license="CC BY 4.0",
    attribution="WeSpeaker ECAPA-TDNN512 pretrained on VoxCeleb. VoxCeleb is used under CC BY 4.0.",
)

SEGMENTATION = Pin(
    label="segmentation",
    file="segmentation.onnx",
    url="https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/733a93b6473d019a773298e08cefa686894b1854/onnx/model.onnx",
    bytes=5_986_908,
    sha256="057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25",
    license="MIT",
    attribution="pyannote segmentation-3.0 powerset model. MIT license.",
)


def cache_dir() -> Path:
    return Path(platformdirs.user_cache_dir("podcue"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def status(path: Path, pin: Pin) -> str:
    """'missing', 'match' or 'mismatch'. The hash is only read when the size already matches."""
    if not path.exists():
        return "missing"
    if not path.is_file() or path.stat().st_size != pin.bytes:
        return "mismatch"
    return "match" if sha256_file(path) == pin.sha256 else "mismatch"


def ensure(pin: Pin, directory: Path | None = None, *, opener=urllib.request.urlopen) -> Path:
    """Return the verified weight file, downloading it when it is missing or wrong."""
    target = (directory or cache_dir() / "weights") / pin.file
    if status(target, pin) == "match":
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    print(f"podcue: downloading {pin.label} weights ({pin.bytes / 1e6:.1f} MB, {pin.license})", file=sys.stderr)
    try:
        request = urllib.request.Request(pin.url, headers={"User-Agent": "podcue"})
        with opener(request) as response, partial.open("wb") as out:
            for block in iter(lambda: response.read(1 << 20), b""):
                out.write(block)
        size = partial.stat().st_size
        if size != pin.bytes:
            raise RuntimeError(f"{pin.label} download expected {pin.bytes} bytes, got {size}")
        digest = sha256_file(partial)
        if digest != pin.sha256:
            raise RuntimeError(f"{pin.label} download sha256 {digest} does not match {pin.sha256}")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target
