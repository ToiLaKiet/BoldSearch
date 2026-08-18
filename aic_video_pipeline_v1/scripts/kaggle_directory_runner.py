from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_archive_runner import (  # noqa: E402
    atomic_json,
    free_bytes,
    gpu_worker_main,
    human_size,
    package_video,
    remove_materialized_video,
    result_exists,
    stop_workers,
    wait_for_worker_result,
)


VIDEO_ID_PATTERN = re.compile(r"^L\d{2}_V\d{2,3}$")
STATE_SCHEMA = "1.0"


def natural_video_sort_key(path: Path) -> tuple[str, ...]:
    """Sort video numbers numerically even if a dataset omits zero padding."""
    return tuple(
        part.zfill(20) if part.isdecimal() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def discover_videos(video_root: Path, level: str) -> list[Path]:
    """Use direct MP4 children only; mounted Kaggle Input remains read-only."""
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


def discover_videos_from_roots(video_roots: list[Path], level: str) -> list[Path]:
    """Merge direct input folders and reject duplicate video IDs across parts."""
    if not video_roots:
        raise ValueError("at least one video root is required")
    by_id: dict[str, Path] = {}
    for root in video_roots:
        for video in discover_videos(root, level):
            previous = by_id.get(video.stem)
            if previous is not None:
                raise ValueError(
                    f"duplicate video_id {video.stem} in {previous} and {video}"
                )
            by_id[video.stem] = video
    return sorted(by_id.values(), key=natural_video_sort_key)


def select_video_range(videos: list[Path], start_at_video: str | None,
                       end_at_video: str | None) -> list[Path]:
    """Select an inclusive, exact video-ID range from already sorted inputs."""
    if start_at_video is None and end_at_video is None:
        return videos
    ids = [path.stem for path in videos]
    start_index = 0
    end_index = len(videos) - 1
    if start_at_video is not None:
        try:
            start_index = ids.index(start_at_video)
        except ValueError as error:
            raise ValueError(f"start video not found: {start_at_video}") from error
    if end_at_video is not None:
        try:
            end_index = ids.index(end_at_video)
        except ValueError as error:
            raise ValueError(f"end video not found: {end_at_video}") from error
    if start_index > end_index:
        raise ValueError("start video must not be after end video")
    return videos[start_index:end_index + 1]


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
    if (
        value.get("schema_version") != STATE_SCHEMA
        or value.get("level") != level
        or not isinstance(value.get("completed_video_ids"), list)
    ):
        return new_state(video_roots, level)
    value["input_roots"] = [str(root) for root in video_roots]
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
            "Disk-bounded runner for direct MP4 files already mounted as "
            "Kaggle Input; one persistent worker per GPU."
        )
    )
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--autoshot-root", type=Path, required=True)
    parser.add_argument("--autoshot-checkpoint", type=Path, required=True)
    parser.add_argument("--video-root", dest="video_roots", type=Path,
                        action="append", required=True,
                        help="repeat once for each direct MP4 input directory")
    parser.add_argument("--level", required=True, help="for example L26")
    parser.add_argument("--start-at-video",
                        help="inclusive first video ID, for example L26_V175")
    parser.add_argument("--end-at-video",
                        help="inclusive last video ID, for example L26_V199")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--gpu-workers", type=int, default=1)
    args = parser.parse_args()

    if not re.fullmatch(r"L\d{2}", args.level):
        raise ValueError("level must follow Lxx, for example L26")
    if not 1 <= args.gpu_workers <= 2:
        raise ValueError("gpu-workers must be 1 or 2")

    args.pipeline_root = args.pipeline_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.autoshot_root = args.autoshot_root.expanduser().resolve()
    args.autoshot_checkpoint = args.autoshot_checkpoint.expanduser().resolve()
    args.video_roots = [path.expanduser().resolve() for path in args.video_roots]
    args.work_root = args.work_root.expanduser().resolve()
    args.result_root = args.result_root.expanduser().resolve()

    for required in (
        args.pipeline_root / "src",
        args.config,
        args.model_path,
        args.autoshot_root,
        args.autoshot_checkpoint,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    videos = discover_videos_from_roots(args.video_roots, args.level)
    videos = select_video_range(
        videos, args.start_at_video, args.end_at_video
    )
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)
    data_root = args.work_root / "pipeline_data"
    state_path = args.work_root / "directory_runner_state.json"
    state = load_state(state_path, args.video_roots, args.level)
    save_state(state_path, args.result_root, state)

    jobs = [
        {"number": number, "video_path": video, "video_id": video.stem}
        for number, video in enumerate(videos, 1)
        if not result_exists(args.result_root, video.stem)
    ]
    for video in videos:
        if result_exists(args.result_root, video.stem):
            mark_completed(state, video.stem)
    save_state(state_path, args.result_root, state)

    print(
        f"[DIRECT INPUT] {len(videos)} video(s), {len(jobs)} to process, "
        f"{args.gpu_workers} persistent GPU worker(s)",
        flush=True,
    )
    if not jobs:
        print(f"All {args.level} direct videos already completed.", flush=True)
        return 0

    context = mp.get_context("spawn")
    task_queues = [context.Queue(maxsize=1) for _ in range(args.gpu_workers)]
    result_queue = context.Queue()
    settings = {
        "pipeline_root": str(args.pipeline_root),
        "config": str(args.config),
        "model_path": str(args.model_path),
        "autoshot_root": str(args.autoshot_root),
        "autoshot_checkpoint": str(args.autoshot_checkpoint),
        "work_root": str(args.work_root),
        "data_root": str(data_root),
    }
    processes = [
        context.Process(
            target=gpu_worker_main,
            args=(worker_id, settings, task_queues[worker_id], result_queue),
            name=f"gpu-worker-{worker_id}",
        )
        for worker_id in range(args.gpu_workers)
    ]
    for process in processes:
        process.start()

    try:
        available = list(range(args.gpu_workers))
        active: dict[int, dict] = {}
        next_job = 0
        while next_job < len(jobs) or active:
            while available and next_job < len(jobs):
                worker_id = available.pop(0)
                job = jobs[next_job]
                next_job += 1
                active[worker_id] = job
                task_queues[worker_id].put({
                    "video_path": str(job["video_path"]),
                    "video_id": job["video_id"],
                })
                print(
                    f"[DISPATCH GPU {worker_id}] "
                    f"{job['number']}/{len(videos)} {job['video_id']}",
                    flush=True,
                )

            message = wait_for_worker_result(result_queue, processes)
            worker_id = int(message["worker_id"])
            job = active.pop(worker_id, None)
            if job is None:
                raise RuntimeError(f"unexpected result from GPU worker {worker_id}")
            if not message.get("ok"):
                raise RuntimeError(
                    f"GPU {worker_id} failed for {job['video_id']}: "
                    f"{message.get('error')}\n{message.get('traceback', '')}"
                )

            video_id = job["video_id"]
            print(f"[PACKAGE] {video_id}.tar", flush=True)
            output = package_video(data_root, args.result_root, video_id)
            remove_materialized_video(data_root, video_id)
            if not result_exists(args.result_root, video_id):
                raise RuntimeError(f"completed video has no valid result TAR: {video_id}")
            mark_completed(state, video_id)
            save_state(state_path, args.result_root, state)
            print(
                f"[DONE GPU {worker_id}] {output.name}; "
                f"free={human_size(free_bytes(args.work_root))}",
                flush=True,
            )
            available.append(worker_id)

        save_state(state_path, args.result_root, state)
        print(f"All {args.level} videos completed. Results: {args.result_root}", flush=True)
        return 0
    finally:
        stop_workers(processes, task_queues)


if __name__ == "__main__":
    raise SystemExit(main())
