from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .publisher import PublishReport, publish_manifest


_VIDEO_ID_RE = re.compile(r"^L\d{2}_V\d{2,3}$")


def _pipeline_provenance(pipeline_root: Path, config: Path) -> dict[str, str]:
    """Return reproducibility metadata without changing the pipeline tree."""
    config = config.expanduser().resolve()
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    revision = "unknown"
    try:
        completed = subprocess.run(
            ["git", "-C", str(pipeline_root.expanduser().resolve()), "rev-parse", "HEAD"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        revision = completed.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "pipeline": "aic_video_pipeline_v1",
        "config_name": config.name,
        "config_sha256": f"sha256:{digest}",
        "pipeline_revision": revision,
    }


def normalize_video_inputs(video_paths: Iterable[Path]) -> tuple[Path, ...]:
    """Validate direct MP4 inputs and reject duplicate video IDs."""
    result: list[Path] = []
    seen: set[str] = set()
    for raw in video_paths:
        path = Path(raw).expanduser().resolve()
        if path.suffix.casefold() != ".mp4" or not path.is_file():
            raise ValueError(f"input must be an existing .mp4 file: {path}")
        video_id = path.stem
        if _VIDEO_ID_RE.fullmatch(video_id) is None:
            raise ValueError(f"video filename must use Lxx_Vxx[x].mp4: {path.name}")
        if video_id in seen:
            raise ValueError(f"duplicate video_id: {video_id}")
        seen.add(video_id)
        result.append(path)
    if not result:
        raise ValueError("at least one MP4 input is required")
    return tuple(result)


def run_v1_and_publish(
    *,
    pipeline_root: Path,
    config: Path,
    video_paths: Iterable[Path],
    data_root: Path,
    output_root: Path,
    model_path: Path | None = None,
    autoshot_root: Path | None = None,
    autoshot_checkpoint: Path | None = None,
    corpus_version: str,
    expected_vector_dim: int = 1024,
    thumbnail_width: int = 960,
    webp_quality: int = 82,
    fresh: bool = False,
) -> tuple[list[dict[str, Any]], PublishReport]:
    """Run one resident V1 pipeline on direct MP4s, then publish atomically."""
    videos = normalize_video_inputs(video_paths)
    source_root = (pipeline_root / "src").expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError(f"pipeline source directory does not exist: {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from aic_video_pipeline_v1.orchestrator import VideoPipelineV1

    pipeline = VideoPipelineV1.from_yaml(
        config.expanduser().resolve(),
        data_root=data_root.expanduser().resolve(),
        model_path=model_path,
        autoshot_root=autoshot_root,
        autoshot_checkpoint=autoshot_checkpoint,
    )
    results: list[dict[str, Any]] = []
    for video in videos:
        results.append(pipeline.run_streaming(video, video.stem, fresh=fresh))
    report = publish_manifest(
        data_root=data_root,
        video_ids=[video.stem for video in videos],
        output_root=output_root,
        expected_vector_dim=expected_vector_dim,
        thumbnail_width=thumbnail_width,
        webp_quality=webp_quality,
        pipeline_provenance=_pipeline_provenance(pipeline_root, config),
        corpus_version=corpus_version,
    )
    return results, report
