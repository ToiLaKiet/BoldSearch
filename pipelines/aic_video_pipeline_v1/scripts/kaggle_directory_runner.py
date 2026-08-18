from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_archive_runner import (  # noqa: E402
    atomic_json,
    free_bytes,
    human_size,
    package_video,
    previous_result_exists,
    remove_materialized_video,
    result_exists,
)


VIDEO_ID_PATTERN = re.compile(r"^L\d{2}_V\d{2,3}$")
STATE_SCHEMA = "1.0"


def natural_video_sort_key(path: Path) -> tuple[str, ...]:
    """Sort L28_V2 before L28_V10 even when filenames have no zero padding."""
    return tuple(
        part.zfill(20) if part.isdecimal() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def discover_videos(video_root: Path, level: str) -> list[Path]:
    """Read only direct MP4 children; never recurse into unrelated inputs."""
    if not video_root.is_dir():
        raise FileNotFoundError(f"video root does not exist: {video_root}")
    expected = re.compile(rf"^{re.escape(level)}_V\d{{2,3}}$")
    videos = sorted(
        (
            path for path in video_root.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".mp4"
            and expected.fullmatch(path.stem)
            and VIDEO_ID_PATTERN.fullmatch(path.stem)
        ),
        key=natural_video_sort_key,
    )
    if not videos:
        raise ValueError(
            f"no {level}_Vxx or {level}_Vxxx MP4 files directly under: {video_root}"
        )
    return videos


def discover_videos_from_roots(video_roots: list[Path],
                               level: str) -> list[Path]:
    """Merge direct MP4 roots and reject duplicate video IDs across parts."""
    if not video_roots:
        raise ValueError("at least one video root is required")
    videos: list[Path] = []
    by_id: dict[str, Path] = {}
    for root in video_roots:
        for video in discover_videos(root, level):
            previous = by_id.get(video.stem)
            if previous is not None:
                raise ValueError(
                    f"duplicate video_id {video.stem} in {previous} and {video}"
                )
            by_id[video.stem] = video
            videos.append(video)
    return sorted(videos, key=natural_video_sort_key)


def new_state(video_roots: list[Path], level: str) -> dict:
    return {
        "schema_version": STATE_SCHEMA,
        "input_roots": [str(root) for root in video_roots],
        "level": level,
        "completed_video_ids": [],
    }


def load_state(path: Path, video_roots: list[Path], level: str) -> dict:
    if not path.is_file():
        return new_state(video_roots, level)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return new_state(video_roots, level)
    if (value.get("schema_version") != STATE_SCHEMA
            or value.get("level") != level
            or not isinstance(value.get("completed_video_ids"), list)):
        return new_state(video_roots, level)
    value["input_roots"] = [str(root) for root in video_roots]
    value.pop("input_root", None)
    return value


def save_state(work_path: Path, result_root: Path, state: dict) -> None:
    atomic_json(work_path, state)
    atomic_json(result_root / "directory_runner_state.json", state)


def mark_completed(state: dict, video_id: str) -> None:
    completed = state["completed_video_ids"]
    if video_id not in completed:
        completed.append(video_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential Kaggle runner for MP4 files already mounted in a "
            "Kaggle Input directory"
        )
    )
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--autoshot-root", type=Path, required=True)
    parser.add_argument("--autoshot-checkpoint", type=Path, required=True)
    parser.add_argument("--video-root", dest="video_roots", type=Path,
                        action="append", required=True,
                        help="repeat for multipart levels such as L26 a-e")
    parser.add_argument("--level", required=True,
                        help="expected video prefix, for example L28")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--previous-result-root", type=Path, action="append", default=[])
    args = parser.parse_args()

    if not re.fullmatch(r"L\d{2}", args.level):
        raise ValueError("level must follow Lxx, for example L28")
    args.pipeline_root = args.pipeline_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.autoshot_root = args.autoshot_root.expanduser().resolve()
    args.autoshot_checkpoint = args.autoshot_checkpoint.expanduser().resolve()
    args.video_roots = [path.expanduser().resolve()
                        for path in args.video_roots]
    args.work_root = args.work_root.expanduser().resolve()
    args.result_root = args.result_root.expanduser().resolve()
    args.previous_result_root = [path.expanduser().resolve()
                                 for path in args.previous_result_root]

    for required in (args.pipeline_root / "src", args.config, args.model_path,
                     args.autoshot_root, args.autoshot_checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)

    videos = discover_videos_from_roots(args.video_roots, args.level)
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)
    data_root = args.work_root / "pipeline_data"
    state_path = args.work_root / "directory_runner_state.json"
    state = load_state(state_path, args.video_roots, args.level)

    source_root = args.pipeline_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from aic_video_pipeline_v1.orchestrator import VideoPipelineV1

    pipeline = VideoPipelineV1.from_yaml(
        args.config,
        data_root=data_root,
        model_path=args.model_path,
        autoshot_root=args.autoshot_root,
        autoshot_checkpoint=args.autoshot_checkpoint,
    )

    print(
        f"[SEQUENTIAL] {len(videos)} video(s) from "
        f"{len(args.video_roots)} input root(s); "
        "one model process will be reused for the full directory.",
        flush=True,
    )
    for number, video_path in enumerate(videos, 1):
        video_id = video_path.stem
        if (result_exists(args.result_root, video_id)
                or previous_result_exists(args.previous_result_root, video_id)):
            mark_completed(state, video_id)
            save_state(state_path, args.result_root, state)
            print(f"[SKIP {number}/{len(videos)}] {video_id}", flush=True)
            continue

        print(f"[START {number}/{len(videos)}] {video_id}", flush=True)
        pipeline.run_streaming(video_path, video_id)
        output = package_video(data_root, args.result_root, video_id)
        remove_materialized_video(data_root, video_id)
        if not result_exists(args.result_root, video_id):
            raise RuntimeError(f"completed video has no valid result TAR: {video_id}")
        mark_completed(state, video_id)
        save_state(state_path, args.result_root, state)
        print(
            f"[DONE {number}/{len(videos)}] {output.name}; "
            f"free={human_size(free_bytes(args.work_root))}",
            flush=True,
        )

    save_state(state_path, args.result_root, state)
    print(f"All {args.level} videos completed. Results: {args.result_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
