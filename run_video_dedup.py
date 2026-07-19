"""Chạy thử processframe trên một video bằng visual embedding nhẹ từ OpenCV.

Đây là harness kiểm chứng end-to-end, không phải encoder production. Pipeline
processframe nhận embedding + shot_id từ hệ thống phía trước; ở đây hai đầu vào
đó được tạo bằng cách lấy mẫu video và so sánh histogram/ảnh xám thu nhỏ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from processframe import build_deduplicated_index


def visual_embedding(frame: np.ndarray) -> np.ndarray:
    """Embedding nhẹ: cấu trúc ảnh xám 32x18 + HSV histogram 8x8x8."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = (small - small.mean()) / (small.std() + 1e-6)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    vector = np.concatenate((small.flatten(), hist))
    return vector / (np.linalg.norm(vector) + 1e-12)


def sharpness_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_samples(video: Path, sample_every_seconds: float, shot_cut_distance: float):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError("Không đọc được FPS video")
    interval = max(1, round(fps * sample_every_seconds))

    sampled_frames: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    sharpness: list[float] = []
    timestamps: list[float] = []
    shot_ids: list[int] = []
    frame_no = 0
    shot_id = 0
    previous_embedding: np.ndarray | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_no % interval == 0:
            embedding = visual_embedding(frame)
            if previous_embedding is not None:
                distance = 1.0 - float(np.clip(previous_embedding @ embedding, -1, 1))
                if distance >= shot_cut_distance:
                    shot_id += 1
            sampled_frames.append(frame)
            embeddings.append(embedding)
            sharpness.append(sharpness_score(frame))
            timestamps.append(frame_no / fps)
            shot_ids.append(shot_id)
            previous_embedding = embedding
        frame_no += 1
    capture.release()
    return sampled_frames, np.asarray(embeddings), sharpness, timestamps, shot_ids, fps


def write_preview(
    output: Path,
    frames: list[np.ndarray],
    timestamps: list[float],
    representative_ids: list[int],
    output_fps: float,
) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
    )
    for index in representative_ids:
        frame = frames[index].copy()
        label = f"representative #{index} | t={timestamps[index]:.1f}s"
        cv2.rectangle(frame, (20, 18), (610, 66), (0, 0, 0), -1)
        cv2.putText(frame, label, (30, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("video_dedup_output"))
    parser.add_argument("--sample-every", type=float, default=2.0)
    parser.add_argument("--shot-cut-distance", type=float, default=0.18)
    parser.add_argument("--tau-cluster", type=float, default=0.12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, embeddings, sharpness, timestamps, shot_ids, source_fps = read_samples(
        args.video, args.sample_every, args.shot_cut_distance
    )
    result = build_deduplicated_index(
        frames=list(range(len(frames))),
        embeddings=embeddings,
        shot_ids=shot_ids,
        timestamps=timestamps,
        sharpness=sharpness,
        tau_cluster=args.tau_cluster,
        shot_budget=None,
    )
    representatives = result.dedup_representatives
    preview_path = args.output_dir / "deduplicated_representatives.mp4"
    write_preview(preview_path, frames, timestamps, representatives, output_fps=1.0)
    report = {
        "source_video": str(args.video),
        "source_fps": source_fps,
        "sample_every_seconds": args.sample_every,
        "sample_count": len(frames),
        "shot_count": len(set(shot_ids)),
        "deduplicated_frame_count": len(representatives),
        "reduction_percent": round(100 * (1 - len(representatives) / len(frames)), 2),
        "preview_video": str(preview_path),
        "representatives": [
            {"sample_index": index, "timestamp_seconds": round(timestamps[index], 3)}
            for index in representatives
        ],
        "clusters": result.clusters,
        "shot_thresholds": result.shot_thresholds,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "clusters"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
