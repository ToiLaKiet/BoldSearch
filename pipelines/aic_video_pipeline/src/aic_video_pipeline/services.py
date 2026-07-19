from __future__ import annotations

import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .models import FrameRecord
from .storage import NpyFileVectorStore, sha256


def probe_video(video_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    if fps <= 0 or total <= 0:
        raise ValueError("video has invalid FPS or no frames")
    codec = "".join(chr((fourcc >> (8 * offset)) & 0xFF) for offset in range(4)).rstrip("\x00")
    return {"fps": fps, "total_frames": total, "duration_ms": round(total / fps * 1000), "width": width, "height": height,
            "codec": codec or None, "source_video_checksum": sha256(video_path)}


def _fallback_shots(video_id: str, video_path: Path, metadata: dict) -> dict:
    return {"schema_version": "1.0", "video_id": video_id, "source_video_path": str(video_path), **metadata,
            "detector": "fallback_full_video_v1", "shots": [{"shot_id": "shot_000001", "frame_start": 0,
            "frame_end": metadata["total_frames"] - 1, "start_ms": 0, "end_ms": metadata["duration_ms"]}]}


def validate_shots(manifest: dict) -> None:
    total = int(manifest["total_frames"])
    previous_start = -1
    for shot in manifest.get("shots", []):
        start, end = int(shot["frame_start"]), int(shot["frame_end"])
        if not 0 <= start <= end < total:
            raise ValueError(f"invalid shot bounds: {start}-{end}")
        if start <= previous_start:
            raise ValueError("shots must be ordered by frame_start")
        previous_start = start


def detect_shots(video_id: str, video_path: Path, metadata: dict, config: dict) -> dict:
    """Run the locally-installed AutoShot model and convert scenes to shot.json."""
    root, checkpoint = Path(config["root"]), Path(config["checkpoint"])
    try:
        if not root.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError("AutoShot source root or checkpoint is missing")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # AutoShot imports matplotlib.  Keep its cache out of the project and
        # prevent dependency warnings from becoming a pipeline console log.
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aic_video_pipeline_matplotlib"))
        import torch
        from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet
        from utils import get_batches, get_frames, predictions_to_scenes

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing.*", category=UserWarning)
            frames = get_frames(str(video_path), width=48, height=27)
            model = TransNetV2Supernet().eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint_data = torch.load(checkpoint, map_location=device)
            state = checkpoint_data.get("net", checkpoint_data)
            model_state = model.state_dict()
            model_state.update({key: value for key, value in state.items() if key in model_state})
            model.load_state_dict(model_state)
            model.to(device).eval()
            predictions = []
            with torch.inference_mode():
                for batch in get_batches(frames):
                    tensor = torch.from_numpy(batch.transpose((3, 0, 1, 2))[None]).to(device)
                    output = model(tensor)
                    output = output[0] if isinstance(output, tuple) else output
                    predictions.append(torch.sigmoid(output[0]).detach().cpu().numpy()[25:75])
        values = np.concatenate(predictions)[:len(frames)]
        scenes = predictions_to_scenes((values > float(config.get("threshold", 0.296))).astype(np.uint8))
        shots = [{"shot_id": f"shot_{number:06d}", "frame_start": int(start), "frame_end": int(end),
                  "start_ms": round(int(start) / metadata["fps"] * 1000), "end_ms": round(int(end) / metadata["fps"] * 1000)}
                 for number, (start, end) in enumerate(scenes, start=1)]
        manifest = {"schema_version": "1.0", "video_id": video_id, "source_video_path": str(video_path), **metadata,
                    "detector": "autoshot_v1", "autoshot_checkpoint_checksum": sha256(checkpoint), "shots": shots}
    except Exception:
        if not config.get("allow_fallback", False):
            raise
        manifest = _fallback_shots(video_id, video_path, metadata)
    validate_shots(manifest)
    return manifest


def index_frames(video_id: str, shots: dict, interval: int, batch_size: int, width: int) -> list[FrameRecord]:
    frames: list[FrameRecord] = []
    fps = float(shots["fps"])
    for shot in shots["shots"]:
        candidates = list(range(int(shot["frame_start"]), int(shot["frame_end"]) + 1, interval))
        if candidates[-1] != int(shot["frame_end"]):
            candidates.append(int(shot["frame_end"]))
        for index in candidates:
            position = len(frames)
            frames.append(FrameRecord(video_id, f"{index:0{width}d}", index, round(index / fps * 1000), shot["shot_id"],
                                      f"batch_{position // batch_size + 1:06d}", position % batch_size + 1))
    ordered = sorted({frame.frame_index: frame for frame in frames}.values(), key=lambda item: item.frame_index)
    for position, frame in enumerate(ordered):
        frame.batch_id, frame.batch_position = f"batch_{position // batch_size + 1:06d}", position % batch_size + 1
    return ordered


class PreliminaryDedupStrategy(Protocol):
    def select(self, frames: list[FrameRecord]) -> "PreliminaryDedupResult": ...


@dataclass(frozen=True)
class PreliminaryDedupResult:
    frames: list[FrameRecord]


class NoopPreliminaryDedup:
    def select(self, frames: list[FrameRecord]) -> PreliminaryDedupResult:
        return PreliminaryDedupResult(frames)


def preliminary_dedup_from_config(config: dict) -> PreliminaryDedupStrategy:
    """Resolve the configured pre-filter without changing immutable frame ids.

    The first production configuration deliberately uses ``none``.  Additional
    strategies must implement ``PreliminaryDedupStrategy`` and return existing
    records (never renumber or rewrite ``frame_id``).
    """
    strategy = str(config.get("strategy", "none")).lower()
    if strategy == "none":
        return NoopPreliminaryDedup()
    raise ValueError(f"unsupported preliminary_dedup strategy: {strategy}")


@dataclass(frozen=True)
class FrameExtractionRequest:
    video_id: str
    source_video_path: Path
    frame_id: str
    frame_index: int
    timestamp_ms: int
    output_path: Path


@dataclass(frozen=True)
class FrameExtractionResult:
    frame_key: str
    frame_path: Path | None
    width: int | None
    height: int | None
    checksum: str | None
    status: str
    error: str | None


def _extract_one(capture: cv2.VideoCapture, request: FrameExtractionRequest, *, force_overwrite: bool) -> FrameExtractionResult:
    frame_key = f"{request.video_id}::{request.frame_id}"
    try:
        if not force_overwrite and request.output_path.exists():
            image = cv2.imread(str(request.output_path))
            if image is not None:
                height, width = image.shape[:2]
                return FrameExtractionResult(frame_key, request.output_path, width, height, sha256(request.output_path), "EXTRACTED", None)
        capture.set(cv2.CAP_PROP_POS_FRAMES, request.frame_index)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError("decoder did not return frame")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = request.output_path.with_suffix(".tmp.png")
        if not cv2.imwrite(str(temporary), image) or cv2.imread(str(temporary)) is None:
            raise RuntimeError("PNG validation failed")
        os.replace(temporary, request.output_path)
        height, width = image.shape[:2]
        return FrameExtractionResult(frame_key, request.output_path, width, height, sha256(request.output_path), "EXTRACTED", None)
    except Exception as exc:
        return FrameExtractionResult(frame_key, None, None, None, None, "FAILED", str(exc))


def extract_frames(video_path: Path, frames: list[FrameRecord], frame_root: Path, *, force_overwrite: bool = False) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    for frame in frames:
        if frame.preliminary_status != "KEPT":
            continue
        request = FrameExtractionRequest(frame.video_id, video_path, frame.frame_id, frame.frame_index, frame.timestamp_ms,
                                         frame_root / frame.video_id / f"{frame.frame_id}.png")
        result = _extract_one(capture, request, force_overwrite=force_overwrite)
        if result.status == "EXTRACTED":
            frame.mapping_status, frame.frame_path, frame.image_checksum = result.status, str(result.frame_path), result.checksum
        else:
            frame.mapping_status = frame.embedding_status = frame.final_status = "FAILED"
            frame.is_active, frame.error = False, result.error
    capture.release()


class HistogramEmbedder:
    model_version = "histogram_v1"
    metadata = {"model_id": "opencv_hsv_histogram", "model_revision": None,
                "preprocessing_version": "opencv_hsv_8x8x8_v1"}

    def embed_batch(self, paths: list[Path]) -> list[np.ndarray]:
        vectors = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"cannot read image: {path}")
            histogram = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
            vectors.append(histogram.astype(np.float32))
        return vectors


class FGClipEmbedder:
    def __init__(self, model_id: str, model_version: str) -> None:
        self.model_id, self.model_version, self._loaded = model_id, model_version, False

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForCausalLM
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.metadata = {
            "model_id": self.model_id,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "preprocessing_version": self.processor.__class__.__name__,
        }
        self._loaded = True

    def embed_batch(self, paths: list[Path]) -> list[np.ndarray]:
        self._load()
        images = [Image.open(path).convert("RGB") for path in paths]
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        with self.torch.inference_mode():
            features = self.model.get_image_features(pixels)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return [item.detach().float().cpu().numpy() for item in features]


def embed_frames(frames: list[FrameRecord], embedder, store: NpyFileVectorStore, batch_size: int, *, force_overwrite: bool = False) -> int:
    embedded = [frame for frame in frames if frame.mapping_status == "EXTRACTED" and frame.frame_path]
    dimension = 0
    for start in range(0, len(embedded), batch_size):
        group = embedded[start:start + batch_size]
        todo = []
        for frame in group:
            expected = store.path_for(frame.video_id, frame.frame_id)
            if not force_overwrite and store.exists(frame.video_id, frame.frame_id):
                vector = store.get(expected)
                frame.vector_path, frame.embedding_status, frame.embedding_model_version, dimension = str(expected), "EMBEDDED", store.model_version, len(vector)
            else:
                todo.append(frame)
        try:
            vectors = embedder.embed_batch([Path(item.frame_path) for item in todo]) if todo else []
            if len(vectors) != len(todo):
                raise ValueError("embedder returned a vector count different from the input frame count")
            for frame, vector in zip(todo, vectors):
                frame.vector_path = str(store.put(frame.video_id, frame.frame_id, vector))
                frame.embedding_status, frame.embedding_model_version, dimension = "EMBEDDED", store.model_version, len(vector)
        except Exception as exc:
            for frame in todo:
                frame.embedding_status = frame.final_status = "FAILED"
                frame.is_active, frame.error = False, str(exc)
    return dimension
