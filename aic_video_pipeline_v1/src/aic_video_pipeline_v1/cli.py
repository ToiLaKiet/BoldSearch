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
    parser.add_argument("--data-root", type=Path,
                        help="override paths.data_root (useful on Kaggle)")
    parser.add_argument("--model-path", type=Path,
                        help="override embedding.model_path")
    parser.add_argument("--autoshot-root", type=Path,
                        help="override autoshot.root")
    parser.add_argument("--autoshot-checkpoint", type=Path,
                        help="override autoshot.checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="delete this video_id's checkpoint/artifacts before run")
    parser.add_argument("--streaming", action="store_true",
                        help="batch in RAM and materialize KEPT frames only")
    args = parser.parse_args()
    pipeline = VideoPipelineV1.from_yaml(
        args.config,
        data_root=args.data_root,
        model_path=args.model_path,
        autoshot_root=args.autoshot_root,
        autoshot_checkpoint=args.autoshot_checkpoint,
    )
    if args.command == "run":
        if not args.video:
            parser.error("run requires --video")
        runner = pipeline.run_streaming if args.streaming else pipeline.run
        result = runner(args.video, args.video_id, fresh=args.fresh)
    else:
        if not args.video_id:
            parser.error("validate requires --video-id")
        pipeline.validate(args.video_id)
        result = {"video_id": args.video_id, "valid": True}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
