from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestrator import VideoPipelineV1

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(prog="aic-video-pipeline-v1")
    parser.add_argument("command", choices=["run", "validate"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-id")
    parser.add_argument("--embedding-provider", choices=["fgclip", "histogram"])
    args = parser.parse_args()
    pipeline = VideoPipelineV1.from_yaml(args.config)
    if args.command == "run":
        if not args.video:
            parser.error("run requires --video")
        result = pipeline.run(args.video, args.embedding_provider, args.video_id)
    else:
        if not args.video_id:
            parser.error("validate requires --video-id")
        pipeline.validate(args.video_id)
        result = {"video_id": args.video_id, "valid": True}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
