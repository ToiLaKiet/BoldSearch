"""Extract and normalise media audio with ffmpeg."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def normalize_audio(source_path: str | Path, output_path: str | Path) -> Path:
    """Write the full audio timeline as PCM mono 16 kHz WAV."""
    src = Path(source_path)
    dst = Path(output_path)

    if not src.exists():
        raise RuntimeError(f"Source file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-f",
        "wav",
        str(dst),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out on {src}") from None

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src}: {result.stderr.strip()}")

    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"Output file missing or empty after ffmpeg: {dst}")

    return dst.resolve()


def create_temp_audio_path(suffix: str = ".wav") -> Path:
    """Create an audio path that the caller must remove."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return Path(path)
