"""Serve only the result frames requested by the browser from source MP4s.

This is intentionally an on-demand cache, not another indexing pipeline.  A
Milvus result already identifies ``video_id`` and the source ``frame_id``; the
gateway asks this module to decode that one image the first time it is needed.
Subsequent requests reuse the immutable WebP cache in Kaggle working storage.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Mapping


_VIDEO_ID_RE = re.compile(r"^L\d{2}_V\d{2,3}$")


class FrameExtractionError(RuntimeError):
    """A requested source frame could not be decoded from its MP4."""


def _parse_fps(raw: str, *, video_id: str) -> float:
    value = raw.strip().splitlines()[0] if raw.strip() else ""
    try:
        fps = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise FrameExtractionError(f"{video_id}: ffprobe returned invalid fps {value!r}") from exc
    if fps <= 0:
        raise FrameExtractionError(f"{video_id}: video fps must be positive")
    return fps


class VideoFrameProvider:
    """Decode a source-indexed MP4 frame once and retain a WebP cache."""

    def __init__(
        self,
        videos: Mapping[str, Path],
        *,
        cache_root: Path,
        max_width: int = 960,
        webp_quality: int = 82,
    ) -> None:
        if max_width <= 0 or not 1 <= webp_quality <= 100:
            raise ValueError("invalid WebP output settings")
        self._videos: dict[str, Path] = {}
        for video_id, path in videos.items():
            if not _VIDEO_ID_RE.fullmatch(video_id):
                raise ValueError(f"invalid video_id in manifest: {video_id!r}")
            resolved = Path(path).expanduser().resolve()
            if resolved.suffix.casefold() != ".mp4" or not resolved.is_file():
                raise ValueError(f"{video_id}: source MP4 is unavailable: {resolved}")
            self._videos[video_id] = resolved
        if not self._videos:
            raise ValueError("video manifest is empty")
        self._cache_root = cache_root.expanduser().resolve()
        self._max_width = max_width
        self._webp_quality = webp_quality
        self._fps: dict[str, float] = {}
        self._locks: dict[tuple[str, int], threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @classmethod
    def from_json_file(
        cls,
        manifest_path: Path,
        *,
        cache_root: Path,
        max_width: int = 960,
        webp_quality: int = 82,
    ) -> "VideoFrameProvider":
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read video manifest: {manifest_path}") from exc
        if not isinstance(document, dict):
            raise ValueError("video manifest must be an object mapping video_id to MP4 path")
        videos: dict[str, Path] = {}
        for video_id, raw_path in document.items():
            if not isinstance(video_id, str) or not isinstance(raw_path, str):
                raise ValueError("video manifest values must be MP4 path strings")
            videos[video_id] = Path(raw_path)
        return cls(
            videos, cache_root=cache_root, max_width=max_width,
            webp_quality=webp_quality,
        )

    def _destination(self, video_id: str, frame_id: int) -> Path:
        if video_id not in self._videos:
            raise FileNotFoundError(f"source MP4 for {video_id} is not mounted")
        if frame_id < 0:
            raise FileNotFoundError("frame_id must be non-negative")
        root = (self._cache_root / "keyframes").resolve()
        destination = (root / video_id / f"{frame_id}.webp").resolve()
        if not destination.is_relative_to(root):
            raise FileNotFoundError("invalid frame cache path")
        return destination

    def _frame_lock(self, video_id: str, frame_id: int) -> threading.Lock:
        key = (video_id, frame_id)
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _fps_for(self, video_id: str) -> float:
        cached = self._fps.get(video_id)
        if cached is not None:
            return cached
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(self._videos[video_id]),
            ],
            check=False, text=True, capture_output=True,
        )
        if result.returncode != 0:
            raise FrameExtractionError(
                f"{video_id}: ffprobe failed: {result.stderr.strip()[-500:]}"
            )
        fps = _parse_fps(result.stdout, video_id=video_id)
        self._fps[video_id] = fps
        return fps

    def resolve(self, video_id: str, frame_id: int) -> Path:
        """Return a cached WebP, decoding exactly one source frame if absent."""
        destination = self._destination(video_id, frame_id)
        if destination.is_file():
            return destination
        with self._frame_lock(video_id, frame_id):
            if destination.is_file():
                return destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            timestamp_seconds = frame_id / self._fps_for(video_id)
            temporary = destination.with_name(
                f".{destination.stem}.{uuid.uuid4().hex}.webp"
            )
            try:
                result = subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                        "-ss", f"{timestamp_seconds:.6f}", "-i", str(self._videos[video_id]),
                        "-frames:v", "1",
                        "-vf", f"scale='min({self._max_width},iw)':-2",
                        "-c:v", "libwebp", "-q:v", str(self._webp_quality), "-y", str(temporary),
                    ],
                    check=False, text=True, capture_output=True,
                )
                if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                    raise FrameExtractionError(
                        f"{video_id}/{frame_id}: ffmpeg failed: {result.stderr.strip()[-500:]}"
                    )
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination
