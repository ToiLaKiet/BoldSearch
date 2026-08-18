from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import AutoShotEngine, run_video  # noqa: E402


def read_video_list(path: Path) -> list[Path]:
    """Read one video path per line; blank lines and # comments are ignored."""
    base = path.expanduser().resolve().parent
    videos = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        video = Path(value).expanduser()
        if not video.is_absolute():
            video = base / video
        if not video.is_file():
            raise FileNotFoundError(f"video list line {line_number}: {video}")
        videos.append(video.resolve())
    if not videos:
        raise ValueError(f"video list is empty: {path}")
    return videos


def build_tasks(videos: list[Path], output_dir: Path) -> list[tuple[Path, Path]]:
    output_dir = output_dir.expanduser().resolve()
    tasks = [(video, output_dir / f"{video.stem}_Shot.json") for video in videos]
    outputs = [output for _video, output in tasks]
    if len(set(outputs)) != len(outputs):
        raise ValueError("video stems must be unique when writing one output directory")
    return tasks


_WORKER_ENGINE: AutoShotEngine | None = None


def _init_worker(autoshot_root: str, checkpoint: str, device: str) -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = AutoShotEngine(Path(autoshot_root), Path(checkpoint), device)


def _run_worker(task: tuple[Path, Path, float, bool]) -> tuple[str, str, dict]:
    if _WORKER_ENGINE is None:
        raise RuntimeError("worker AutoShot engine was not initialized")
    video, output, threshold, allow_fallback = task
    result = run_video(video, output, _WORKER_ENGINE, threshold, allow_fallback)
    return str(video), str(output), result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AutoShot for every video in a TXT file")
    parser.add_argument("--video-list", type=Path, required=True,
                        help="UTF-8 text file with one video path per line")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--autoshot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.296)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel processes; use 1 for a single GPU")
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    try:
        videos = read_video_list(args.video_list)
        tasks = build_tasks(videos, args.output_dir)
        args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        work = [(video, output, args.threshold, args.allow_fallback)
                for video, output in tasks]
        completed = 0
        if args.workers == 1:
            engine = AutoShotEngine(args.autoshot_root, args.checkpoint, args.device)
            for video, output, threshold, allow_fallback in work:
                result = run_video(video, output, engine, threshold, allow_fallback)
                completed += 1
                print(f"[{completed}/{len(work)}] {video.name}: "
                      f"{len(result['shots'])} shots -> {output}")
        else:
            max_workers = min(args.workers, len(work), os.cpu_count() or 1)
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_worker,
                    initargs=(str(args.autoshot_root), str(args.checkpoint), args.device)) as pool:
                futures = [pool.submit(_run_worker, task) for task in work]
                for future in concurrent.futures.as_completed(futures):
                    video, output, result = future.result()
                    completed += 1
                    print(f"[{completed}/{len(work)}] {Path(video).name}: "
                          f"{len(result['shots'])} shots -> {output}")
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
