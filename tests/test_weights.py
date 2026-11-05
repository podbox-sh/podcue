import hashlib
import io

import pytest

from podcue import weights
from podcue.weights import Pin, ensure, status

BODY = b"podcue test weights"
PIN = Pin("test", "w.onnx", "https://example.invalid/w.onnx", len(BODY), hashlib.sha256(BODY).hexdigest(), "MIT", "")


class Opener:
    def __init__(self, body):
        self.body = body
        self.calls = 0

    def __call__(self, request, timeout):
        self.calls += 1
        assert timeout == weights.TIMEOUT_S
        assert request.get_header("User-agent") == "podcue"
        return io.BytesIO(self.body)


def test_status(tmp_path):
    path = tmp_path / "w.onnx"
    assert status(path, PIN) == "missing"
    path.write_bytes(BODY[:-1])
    assert status(path, PIN) == "mismatch"
    path.write_bytes(BODY[:-1] + b"X")
    assert status(path, PIN) == "mismatch"
    path.write_bytes(BODY)
    assert status(path, PIN) == "match"


def test_ensure_downloads_once(tmp_path):
    opener = Opener(BODY)
    path = ensure(PIN, tmp_path, opener=opener)
    assert path.read_bytes() == BODY
    assert ensure(PIN, tmp_path, opener=opener) == path
    assert opener.calls == 1


def test_ensure_replaces_a_corrupt_file(tmp_path):
    (tmp_path / "w.onnx").write_bytes(b"x" * len(BODY))
    assert ensure(PIN, tmp_path, opener=Opener(BODY)).read_bytes() == BODY


@pytest.mark.parametrize("body, message", [(BODY + b"!", "expected"), (BODY[:-1] + b"X", "sha256")])
def test_ensure_rejects_a_bad_download_and_leaves_nothing(tmp_path, body, message):
    with pytest.raises(RuntimeError, match=message):
        ensure(PIN, tmp_path, opener=Opener(body))
    assert list(tmp_path.iterdir()) == []


def test_pins_are_the_reviewed_values():
    assert weights.ECAPA.bytes == 24_861_931
    assert weights.ECAPA.sha256 == "f901ae3571912bad4e11e157744cb8b7d77a0b46923b83c410bdbb6c77d69736"
    assert weights.ECAPA.license == "CC BY 4.0"
    assert weights.SEGMENTATION.bytes == 5_986_908
    assert weights.SEGMENTATION.sha256 == "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25"
    assert weights.SEGMENTATION.license == "MIT"
