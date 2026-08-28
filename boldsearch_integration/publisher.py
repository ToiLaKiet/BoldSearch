from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


_SHOT_ID_RE = re.compile(r"^shot_(?P<number>\d+)$")
_VIDEO_ID_RE = re.compile(r"^L\d{2}_V\d{2,3}$")


@dataclass(frozen=True)
class KeptFrame:
    video_id: str
    frame_id: int
    shot_id: int
    png_path: Path
    vector_path: Path


@dataclass(frozen=True)
class PublishReport:
    release_id: str
    release_root: Path
    video_ids: tuple[str, ...]
    row_count: int
    image_bytes: int


def _resolve_artifact_path(
    data_root: Path,
    value: object,
    *,
    video_id: str,
    frame_id: int,
    suffix: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{video_id}/{frame_id}: missing {suffix} artifact path")
    candidate = Path(value).expanduser()
    path = candidate if candidate.is_absolute() else data_root / candidate
    path = path.resolve()
    if path.suffix.casefold() != suffix or path.stem != str(frame_id):
        raise ValueError(f"{video_id}/{frame_id}: invalid {suffix} artifact path: {path}")
    if path.parent.name != video_id:
        raise ValueError(f"{video_id}/{frame_id}: artifact is outside video directory: {path}")
    if not path.is_file():
        raise ValueError(f"{video_id}/{frame_id}: missing artifact: {path}")
    return path


def _shot_number(video_id: str, raw: object) -> int:
    match = _SHOT_ID_RE.fullmatch(str(raw or "").strip())
    if not match:
        raise ValueError(f"{video_id}: invalid shot_id {raw!r}; expected shot_NNNNNN")
    return int(match.group("number"))


def load_kept_frames(
    data_root: Path,
    video_id: str,
    *,
    expected_vector_dim: int = 1024,
) -> list[KeptFrame]:
    """Load and validate only FINAL/KEPT records from one V1 output."""
    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError(f"invalid video_id: {video_id}")
    data_root = data_root.expanduser().resolve()
    document_path = data_root / "metadata" / video_id / "Frame.json"
    if not document_path.is_file():
        raise ValueError(f"missing Frame.json: {document_path}")
    try:
        document = json.loads(document_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Frame.json: {document_path}") from exc
    if (
        document.get("schema_version") != "2.0"
        or document.get("video_id") != video_id
        or document.get("stage") != "FINAL"
        or not isinstance(document.get("frames"), list)
    ):
        raise ValueError(f"invalid final Frame.json: {document_path}")

    result: list[KeptFrame] = []
    seen: set[int] = set()
    for item in document["frames"]:
        if not isinstance(item, dict):
            raise ValueError(f"{video_id}: Frame.json record is not an object")
        if str(item.get("final_status", "")).upper() != "KEPT":
            continue
        try:
            frame_id = int(str(item["frame_id"]).strip())
            frame_index = int(item["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{video_id}: invalid frame identity") from exc
        if frame_id < 0 or frame_index != frame_id or frame_id in seen:
            raise ValueError(f"{video_id}: invalid or duplicate frame_id {frame_id}")
        seen.add(frame_id)
        shot_id = _shot_number(video_id, item.get("shot_id"))
        png_path = _resolve_artifact_path(
            data_root, item.get("frame_path"), video_id=video_id,
            frame_id=frame_id, suffix=".png",
        )
        vector_path = _resolve_artifact_path(
            data_root, item.get("vector_path"), video_id=video_id,
            frame_id=frame_id, suffix=".npy",
        )
        try:
            vector = np.load(vector_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{video_id}/{frame_id}: cannot read vector") from exc
        if vector.dtype != np.float32 or vector.ndim != 1 or vector.shape[0] != expected_vector_dim:
            raise ValueError(
                f"{video_id}/{frame_id}: vector dimension/dtype mismatch; "
                f"expected float32[{expected_vector_dim}], got {vector.dtype}{vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"{video_id}/{frame_id}: vector contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"{video_id}/{frame_id}: vector is not L2-normalized (norm={norm:.6f})")
        result.append(KeptFrame(video_id, frame_id, shot_id, png_path, vector_path))
    return sorted(result, key=lambda item: item.frame_id)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
    os.replace(temporary, path)


def _write_manifest(path: Path, rows: Iterable[KeptFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=["video_id", "frame_id", "shot_id"])
        writer.writeheader()
        for frame in rows:
            writer.writerow({
                "video_id": frame.video_id,
                "frame_id": frame.frame_id,
                "shot_id": frame.shot_id,
            })
        handle.flush()
    os.replace(temporary, path)


def _derive_webp(source: Path, destination: Path, *, max_width: int, quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            if image.width > max_width:
                height = max(1, round(image.height * max_width / image.width))
                image = image.resize((max_width, height), Image.Resampling.LANCZOS)
            image.save(temporary, format="WEBP", quality=quality, method=4)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_active_release(output_root: Path) -> Path:
    output_root = output_root.expanduser().resolve()
    pointer = output_root / "active.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"active release pointer does not exist: {pointer}")
    document = json.loads(pointer.read_text(encoding="utf-8"))
    release_id = document.get("release_id")
    if not isinstance(release_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", release_id):
        raise ValueError(f"invalid active release pointer: {pointer}")
    release = (output_root / "releases" / release_id).resolve()
    if not release.is_relative_to((output_root / "releases").resolve()) or not (release / "Frames.csv").is_file():
        raise ValueError(f"active release is invalid: {release}")
    return release


def publish_manifest(
    *,
    data_root: Path,
    video_ids: Iterable[str],
    output_root: Path,
    expected_vector_dim: int = 1024,
    thumbnail_width: int = 960,
    webp_quality: int = 82,
) -> PublishReport:
    """Atomically publish validated V1 outputs as a new runtime release."""
    if expected_vector_dim <= 0 or thumbnail_width <= 0 or not 1 <= webp_quality <= 100:
        raise ValueError("invalid publisher settings")
    unique_video_ids = tuple(sorted(set(video_ids)))
    if not unique_video_ids:
        raise ValueError("at least one video_id is required")
    frames: list[KeptFrame] = []
    for video_id in unique_video_ids:
        frames.extend(load_kept_frames(
            data_root, video_id, expected_vector_dim=expected_vector_dim,
        ))
    if not frames:
        raise ValueError("no KEPT frames to publish")

    output_root = output_root.expanduser().resolve()
    staging_root = output_root / ".staging" / uuid.uuid4().hex
    release_id = f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    release_root = output_root / "releases" / release_id
    try:
        staging_root.mkdir(parents=True, exist_ok=False)
        for frame in frames:
            destination = staging_root / "keyframes" / frame.video_id / f"{frame.frame_id}.webp"
            _derive_webp(
                frame.png_path, destination,
                max_width=thumbnail_width, quality=webp_quality,
            )
        _write_manifest(staging_root / "Frames.csv", frames)
        image_bytes = sum(
            (staging_root / "keyframes" / frame.video_id / f"{frame.frame_id}.webp").stat().st_size
            for frame in frames
        )
        _atomic_json(staging_root / "corpus-manifest.json", {
            "schema_version": "1.0",
            "release_id": release_id,
            "video_ids": list(unique_video_ids),
            "row_count": len(frames),
            "thumbnail_width": thumbnail_width,
            "webp_quality": webp_quality,
        })
        release_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_root, release_root)
        _atomic_json(output_root / "active.json", {
            "schema_version": "1.0",
            "release_id": release_id,
        })
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return PublishReport(
        release_id=release_id,
        release_root=release_root,
        video_ids=unique_video_ids,
        row_count=len(frames),
        image_bytes=image_bytes,
    )
