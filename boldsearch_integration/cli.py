from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .milvus_ingest import build_milvus_rows, ingest_rows
from .publisher import publish_manifest
from .runner import run_v1_and_publish


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish aic_video_pipeline_v1 MP4 artifacts for BoldSearch"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, required=True)
    common.add_argument("--video-id", action="append", dest="video_ids", required=True)
    common.add_argument("--corpus-version", required=True)
    common.add_argument("--expected-vector-dim", type=int, default=1024)

    publish = subparsers.add_parser(
        "publish", parents=[common],
        help="validate V1 output and atomically publish Frames.csv/WebP",
    )
    publish.add_argument("--output-root", type=Path, required=True)
    publish.add_argument("--thumbnail-width", type=int, default=960)
    publish.add_argument("--webp-quality", type=int, default=82)

    ingest = subparsers.add_parser(
        "ingest", parents=[common],
        help="batch upsert validated visual rows into Milvus/Zilliz",
    )
    ingest.add_argument("--collection", required=True)
    ingest.add_argument("--thumbnail-base", default="/keyframes")
    ingest.add_argument("--batch-size", type=int, default=256)
    ingest.add_argument("--uri", default=os.environ.get("ZILLIZ_URI", ""))
    ingest.add_argument("--token", default=os.environ.get("ZILLIZ_TOKEN", ""))
    ingest.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser(
        "run", help="run V1 directly on MP4 files, then publish the release"
    )
    run.add_argument("--pipeline-root", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--video", type=Path, action="append", required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--model-path", type=Path)
    run.add_argument("--autoshot-root", type=Path)
    run.add_argument("--autoshot-checkpoint", type=Path)
    run.add_argument("--corpus-version", required=True)
    run.add_argument("--expected-vector-dim", type=int, default=1024)
    run.add_argument("--thumbnail-width", type=int, default=960)
    run.add_argument("--webp-quality", type=int, default=82)
    run.add_argument("--fresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        results, report = run_v1_and_publish(
            pipeline_root=args.pipeline_root,
            config=args.config,
            video_paths=args.video,
            data_root=args.data_root,
            output_root=args.output_root,
            model_path=args.model_path,
            autoshot_root=args.autoshot_root,
            autoshot_checkpoint=args.autoshot_checkpoint,
            corpus_version=args.corpus_version,
            expected_vector_dim=args.expected_vector_dim,
            thumbnail_width=args.thumbnail_width,
            webp_quality=args.webp_quality,
            fresh=args.fresh,
        )
        print(json.dumps({
            "videos": results,
            "release_id": report.release_id,
            "release_root": str(report.release_root),
            "row_count": report.row_count,
        }, ensure_ascii=False))
        return 0
    if args.command == "publish":
        report = publish_manifest(
            data_root=args.data_root,
            video_ids=args.video_ids,
            output_root=args.output_root,
            expected_vector_dim=args.expected_vector_dim,
            thumbnail_width=args.thumbnail_width,
            webp_quality=args.webp_quality,
        )
        print(json.dumps({
            "release_id": report.release_id,
            "release_root": str(report.release_root),
            "video_ids": report.video_ids,
            "row_count": report.row_count,
            "image_bytes": report.image_bytes,
        }, ensure_ascii=False))
        return 0

    rows = build_milvus_rows(
        data_root=args.data_root,
        video_ids=args.video_ids,
        corpus_version=args.corpus_version,
        expected_vector_dim=args.expected_vector_dim,
        thumbnail_base=args.thumbnail_base,
    )
    if args.dry_run:
        print(json.dumps({
            "collection": args.collection,
            "corpus_version": args.corpus_version,
            "row_count": len(rows),
            "first_id": rows[0]["id"],
            "last_id": rows[-1]["id"],
        }, ensure_ascii=False))
        return 0
    if not args.uri:
        raise SystemExit("--uri or ZILLIZ_URI is required unless --dry-run is used")
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise SystemExit("pymilvus is required for non-dry-run ingest") from exc
    client = MilvusClient(uri=args.uri, token=args.token or None)
    count = ingest_rows(client, args.collection, rows, batch_size=args.batch_size)
    print(json.dumps({
        "collection": args.collection,
        "corpus_version": args.corpus_version,
        "upserted": count,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
