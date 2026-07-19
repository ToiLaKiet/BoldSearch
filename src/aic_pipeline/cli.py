from __future__ import annotations

import argparse
from pathlib import Path

from .orchestrator import VideoPipelineOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(prog="aic-pipeline")
    parser.add_argument("command", choices=["run", "validate"])
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-id")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--embedding-provider", choices=["fgclip", "histogram"])
    args = parser.parse_args()
    pipeline = VideoPipelineOrchestrator.from_yaml(args.config)
    if args.command == "run":
        if not args.video: parser.error("run requires --video")
        pipeline.run(args.video, args.embedding_provider)
    else:
        if not args.video_id: parser.error("validate requires --video-id")
        pipeline.validate(args.video_id)


if __name__ == "__main__":
    main()
