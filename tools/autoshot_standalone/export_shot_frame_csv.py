from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


CSV_FIELDS = (
    "video_id",
    "frame_id",
    "shot_id",
    "frame_start",
    "frame_end",
    "start_ms",
    "end_ms",
)


def export_mapping(shot_dir: Path, output: Path, frame_id_width: int = 8) -> dict[str, int]:
    shot_dir = shot_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    files = sorted(shot_dir.glob("*_Shot.json"))
    if not files:
        raise FileNotFoundError(f"no *_Shot.json files found in {shot_dir}")
    if frame_id_width <= 0:
        raise ValueError("frame_id_width must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    total_rows = 0
    total_shots = 0
    seen_video_ids: set[str] = set()

    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for path in files:
                document = json.loads(path.read_text(encoding="utf-8"))
                video_id = str(document["video_id"])
                expected_video_id = path.name.removesuffix("_Shot.json")
                if video_id != expected_video_id:
                    raise ValueError(
                        f"video_id does not match filename: {path.name} -> {video_id}"
                    )
                if video_id in seen_video_ids:
                    raise ValueError(f"duplicate video_id: {video_id}")
                seen_video_ids.add(video_id)

                total_frames = int(document["total_frames"])
                expected_start = 0
                shots = document.get("shots")
                if not isinstance(shots, list) or not shots:
                    raise ValueError(f"shots must be a non-empty list: {path}")

                for shot in shots:
                    frame_start = int(shot["frame_start"])
                    frame_end = int(shot["frame_end"])
                    if frame_start != expected_start or frame_end < frame_start:
                        raise ValueError(
                            f"shots are not gap-free and ordered in {path.name}: "
                            f"expected {expected_start}, got {frame_start}-{frame_end}"
                        )
                    if frame_end >= total_frames:
                        raise ValueError(f"shot exceeds total_frames in {path.name}")

                    common = {
                        "video_id": video_id,
                        "shot_id": str(shot["shot_id"]),
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "start_ms": int(shot["start_ms"]),
                        "end_ms": int(shot["end_ms"]),
                    }
                    for frame_index in range(frame_start, frame_end + 1):
                        writer.writerow(
                            {**common, "frame_id": f"{frame_index:0{frame_id_width}d}"}
                        )
                    total_rows += frame_end - frame_start + 1
                    total_shots += 1
                    expected_start = frame_end + 1

                if expected_start != total_frames:
                    raise ValueError(
                        f"shots do not cover every frame in {path.name}: "
                        f"covered={expected_start}, total_frames={total_frames}"
                    )
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {"json_files": len(files), "shots": total_shots, "rows": total_rows}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expand AutoShot Shot.json files into one frame-to-shot CSV"
    )
    parser.add_argument("--shot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-id-width", type=int, default=8)
    args = parser.parse_args()

    summary = export_mapping(args.shot_dir, args.output, args.frame_id_width)
    print(f"json_files={summary['json_files']}")
    print(f"shots={summary['shots']}")
    print(f"rows={summary['rows']}")
    print(f"output={args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
