"""
Tests for asr/audio.py — audio normalisation via ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from asr.audio import create_temp_audio_path, normalize_audio


def _create_synthetic_wav(path: Path, channels: int = 2, rate: int = 44100) -> Path:
    """Create a minimal synthetic WAV file for testing."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "sine=d=0.5:f=440",
        "-ac", str(channels),
        "-ar", str(rate),
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return path


class TestNormalizeAudio:
    def test_normalizes_stereo_44100_to_mono_16k(self, tmp_path):
        src = tmp_path / "input_stereo.wav"
        dst = tmp_path / "output.wav"
        _create_synthetic_wav(src, channels=2, rate=44100)
        result = normalize_audio(src, dst)
        assert result == dst.resolve()
        assert dst.exists()
        assert dst.stat().st_size > 0

    def test_normalizes_mono_48k_to_mono_16k(self, tmp_path):
        src = tmp_path / "input_mono_48k.wav"
        dst = tmp_path / "output.wav"
        _create_synthetic_wav(src, channels=1, rate=48000)
        result = normalize_audio(src, dst)
        assert result == dst.resolve()

    def test_raises_on_nonexistent_source(self, tmp_path):
        src = tmp_path / "nonexistent.mp4"
        dst = tmp_path / "output.wav"
        with pytest.raises(RuntimeError, match="Source file not found"):
            normalize_audio(src, dst)


class TestCreateTempAudioPath:
    def test_creates_temp_path_with_suffix(self):
        path = create_temp_audio_path(".wav")
        assert path.suffix == ".wav"
        # Clean up
        path.unlink(missing_ok=True)
