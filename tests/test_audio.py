import io
import shutil
import wave

import numpy as np
import pytest

from podcue import audio


def test_is_url():
    assert audio.is_url("https://example.com/a.mp3")
    assert not audio.is_url("/tmp/a.mp3")
    assert not audio.is_url("C:/a.mp3")


def test_download_names_the_file_after_the_url_suffix_and_sets_a_timeout(tmp_path):
    def opener(request, timeout):
        assert request.full_url == "https://example.com/a.m4a"
        assert timeout == audio.TIMEOUT_S
        return io.BytesIO(b"bytes")

    path = audio.download("https://example.com/a.m4a", tmp_path, opener=opener)
    assert path.name == "download.m4a" and path.read_bytes() == b"bytes"


def test_span_cuts_by_millisecond():
    samples = np.arange(32_000, dtype=np.float32)
    assert len(audio.span(samples, 500, 1_500)) == 16_000
    assert audio.span(samples, 500, 1_500)[0] == 8_000
    assert audio.duration_ms(samples) == 2_000


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_to_wav_converts_stereo_44k_to_mono_16k(tmp_path):
    source = tmp_path / "tone.wav"
    tone = (np.sin(np.linspace(0, 440 * 2 * np.pi, 44_100)) * 10_000).astype("<i2")
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(44_100)
        writer.writeframes(np.repeat(tone, 2).tobytes())
    target = tmp_path / "out.wav"
    audio.to_wav(source, target)
    samples = audio.read_wav(target)
    assert abs(audio.duration_ms(samples) - 1_000) <= 5
    assert samples.dtype == np.float32 and np.abs(samples).max() > 0.1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_to_wav_reports_undecodable_input(tmp_path):
    source = tmp_path / "junk.mp3"
    source.write_bytes(b"not audio")
    with pytest.raises(RuntimeError, match="ffmpeg could not convert"):
        audio.to_wav(source, tmp_path / "out.wav")


def test_read_wav_rejects_other_formats(tmp_path):
    path = tmp_path / "8k.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8_000)
        writer.writeframes(b"\0\0" * 10)
    with pytest.raises(ValueError):
        audio.read_wav(path)
