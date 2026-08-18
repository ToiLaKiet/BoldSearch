from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import AutoShotEngine, run_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AutoShot independently for one video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--autoshot-root", type=Path, required=True,
                        help="folder containing AutoShot model Python files")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="Shot.json output; defaults to <video_stem>_Shot.json")
    parser.add_argument("--threshold", type=float, default=0.296)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="write one full-video shot if AutoShot fails")
    args = parser.parse_args()

    output = args.output or args.video.with_name(f"{args.video.stem}_Shot.json")
    engine = AutoShotEngine(args.autoshot_root, args.checkpoint, args.device)

    def progress(current: int, total: int) -> None:
        text = f"AutoShot inference: {current}/{total} windows"
        if sys.stderr.isatty():
            print(f"\r{text}", end="", file=sys.stderr, flush=True)
        else:
            print(text, file=sys.stderr)

    try:
        result = run_video(args.video, output, engine, args.threshold,
                           args.allow_fallback, progress)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    if sys.stderr.isatty():
        print(file=sys.stderr)
    print(f"video_id={result['video_id']}")
    print(f"frames={result['total_frames']}")
    print(f"shots={len(result['shots'])}")
    print(f"detector={result['detector']}")
    print(f"output={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
