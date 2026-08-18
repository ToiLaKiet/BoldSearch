from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


ProgressCallback = Callable[[int, int], None]


def read_video_metadata(video_path: Path) -> dict:
    """Read lightweight container metadata without decoding all frames."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    metadata = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "container_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": "".join(
            chr((int(capture.get(cv2.CAP_PROP_FOURCC)) >> (8 * i)) & 255)
            for i in range(4)
        ).rstrip("\0") or None,
    }
    capture.release()
    if metadata["fps"] <= 0 or metadata["width"] <= 0 or metadata["height"] <= 0:
        raise ValueError(f"video has invalid metadata: {video_path}")
    return metadata


def decode_small_frames(video_path: Path, width: int = 48, height: int = 27) -> np.ndarray:
    """Decode the real source frames used by AutoShot.

    ``-vsync 0`` prevents FFmpeg from inserting a synthetic frame to match a
    container duration. This makes the frame count used by shots authoritative.
    """
    command = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", f"scale={width}:{height}", "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FFmpeg failed to decode {video_path}: {detail}")
    frame_bytes = width * height * 3
    if not result.stdout or len(result.stdout) % frame_bytes:
        raise ValueError(f"FFmpeg returned incomplete frames for {video_path}")
    return np.frombuffer(result.stdout, np.uint8).reshape((-1, height, width, 3))


def make_temporal_batches(frames: np.ndarray):
    """Reproduce AutoShot's 100-frame temporal windows with 50-frame stride."""
    if len(frames) == 0:
        return
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0
    padded = np.concatenate(
        [frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), axis=0
    )
    for start in range(0, len(padded) - 50, 50):
        yield padded[start:start + 100]


def predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
    """Convert binary transition predictions into raw scene ranges."""
    values = np.asarray(predictions, dtype=np.uint8).reshape(-1)
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.int32)
    scenes: list[list[int]] = []
    previous, start = 0, 0
    for index, current in enumerate(values):
        if previous == 1 and current == 0:
            start = index
        if previous == 0 and current == 1 and index != 0:
            scenes.append([start, index])
        previous = int(current)
    if previous == 0:
        scenes.append([start, len(values) - 1])
    if not scenes:
        scenes = [[0, len(values) - 1]]
    return np.asarray(scenes, dtype=np.int32)


def normalize_scenes(scenes: np.ndarray, total_frames: int) -> list[tuple[int, int]]:
    """Make non-overlapping, gap-free shot ranges covering every frame once."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    raw = np.asarray(scenes, dtype=np.int64).reshape(-1, 2) if len(scenes) else np.empty((0, 2))
    if len(raw) == 0:
        return [(0, total_frames - 1)]
    starts = [0]
    for start, _end in raw[1:]:
        boundary = max(1, min(int(start), total_frames - 1))
        if boundary > starts[-1]:
            starts.append(boundary)
    return [
        (start, starts[index + 1] - 1 if index + 1 < len(starts) else total_frames - 1)
        for index, start in enumerate(starts)
    ]


class AutoShotEngine:
    """Lazy AutoShot model loader that can be reused for multiple videos."""

    def __init__(self, autoshot_root: Path, checkpoint: Path,
                 device: str = "auto") -> None:
        self.autoshot_root = autoshot_root.expanduser().resolve()
        self.checkpoint = checkpoint.expanduser().resolve()
        self.device_name = device
        self._torch = None
        self._model = None
        self._device = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.autoshot_root.is_dir():
            raise FileNotFoundError(f"AutoShot directory not found: {self.autoshot_root}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"AutoShot checkpoint not found: {self.checkpoint}")
        if str(self.autoshot_root) not in sys.path:
            sys.path.insert(0, str(self.autoshot_root))
        import torch
        from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet

        self._torch = torch
        device_name = self.device_name
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self._device = torch.device(device_name)
        model = TransNetV2Supernet().eval()
        checkpoint_data = torch.load(self.checkpoint, map_location=self._device)
        state = checkpoint_data.get("net", checkpoint_data)
        current = model.state_dict()
        current.update({key: value for key, value in state.items() if key in current})
        model.load_state_dict(current)
        self._model = model.to(self._device).eval()
        # AutoShot's custom temporal layers keep a plain ``device`` attribute
        # instead of a registered buffer. ``Module.to`` cannot update it, so
        # explicitly align every such submodule with the requested device.
        for module in self._model.modules():
            if hasattr(module, "device"):
                module.device = self._device

    @property
    def device(self) -> str:
        if self._device is not None:
            return str(self._device)
        if self.device_name == "auto":
            return "cuda" if self._torch is not None and self._torch.cuda.is_available() else "auto"
        return self.device_name

    def detect(self, video_path: Path, threshold: float = 0.296,
               progress_callback: ProgressCallback | None = None) -> dict:
        self._load_model()
        metadata = read_video_metadata(video_path)
        small_frames = decode_small_frames(video_path)
        total_frames = len(small_frames)
        if total_frames == 0:
            raise ValueError(f"AutoShot decoded no frames: {video_path}")

        predictions = []
        batches = make_temporal_batches(small_frames)
        total_batches = max(1, (total_frames + 49) // 50)
        with warnings.catch_warnings(), self._torch.inference_mode():
            warnings.simplefilter("ignore", UserWarning)
            for batch_index, batch in enumerate(batches, start=1):
                tensor = self._torch.from_numpy(
                    batch.transpose((3, 0, 1, 2))[None]
                ).to(self._device)
                output = self._model(tensor)
                output = output[0] if isinstance(output, tuple) else output
                predictions.append(
                    self._torch.sigmoid(output[0]).detach().cpu().numpy()[25:75]
                )
                if progress_callback is not None:
                    progress_callback(batch_index, total_batches)

        values = np.concatenate(predictions)[:total_frames]
        raw_scenes = predictions_to_scenes((values > float(threshold)).astype(np.uint8))
        scenes = normalize_scenes(raw_scenes, total_frames)
        fps = metadata["fps"]
        shots = [
            {
                "shot_id": f"shot_{index:06d}",
                "frame_start": int(start),
                "frame_end": int(end),
                "start_ms": round(start / fps * 1000),
                "end_ms": round(end / fps * 1000),
            }
            for index, (start, end) in enumerate(scenes, start=1)
        ]
        return {
            "video_id": video_path.stem,
            "source_video_path": str(video_path.resolve()),
            "fps": fps,
            "total_frames": total_frames,
            "container_frame_count": metadata["container_frame_count"],
            "width": metadata["width"],
            "height": metadata["height"],
            "codec": metadata["codec"],
            "detector": "autoshot_v1",
            "threshold": float(threshold),
            "device": str(self._device),
            "shots": shots,
        }


def run_video(video_path: Path, output_path: Path, engine: AutoShotEngine,
              threshold: float = 0.296, allow_fallback: bool = False,
              progress_callback: ProgressCallback | None = None) -> dict:
    """Run AutoShot for one video and atomically write its Shot.json."""
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    try:
        result = engine.detect(video_path, threshold, progress_callback)
    except Exception as error:
        if not allow_fallback:
            raise
        metadata = read_video_metadata(video_path)
        total = max(1, metadata["container_frame_count"])
        result = {
            "video_id": video_path.stem,
            "source_video_path": str(video_path),
            "fps": metadata["fps"],
            "total_frames": total,
            "container_frame_count": metadata["container_frame_count"],
            "width": metadata["width"],
            "height": metadata["height"],
            "codec": metadata["codec"],
            "detector": "fallback_full_video_v1",
            "threshold": float(threshold),
            "device": engine.device,
            "error": f"{type(error).__name__}: {error}",
            "shots": [{
                "shot_id": "shot_000001",
                "frame_start": 0,
                "frame_end": total - 1,
                "start_ms": 0,
                "end_ms": round((total - 1) / metadata["fps"] * 1000),
            }],
        }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    return result
