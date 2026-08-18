from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
from PIL import Image

from .models import FrameRecord
from .storage import NpyVectorStore, sha256


_AUTOSHOT_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


def probe_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    if fps <= 0 or total <= 0:
        raise ValueError("video has invalid FPS or no frames")
    codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).rstrip("\0") or None
    return {"fps": fps, "total_frames": total, "duration_ms": round(total / fps * 1000),
            "width": width, "height": height, "codec": codec,
            "source_video_checksum": sha256(path)}


def _validate_shots(manifest: dict) -> None:
    total = int(manifest["total_frames"])
    previous = -1
    for shot in manifest["shots"]:
        start, end = int(shot["frame_start"]), int(shot["frame_end"])
        if not 0 <= start <= end < total or start <= previous:
            raise ValueError(f"invalid or unordered shot: {start}-{end}")
        previous = start


def _normalize_scenes(scenes: np.ndarray, total_frames: int) -> list[tuple[int, int]]:
    """Return ordered, gap-free shot ranges covering the source video."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    raw = (np.asarray(scenes, dtype=np.int64).reshape(-1, 2)
           if len(scenes) else np.empty((0, 2), dtype=np.int64))
    if len(raw) == 0:
        return [(0, total_frames - 1)]
    starts = [0]
    for start, _end in raw[1:]:
        boundary = max(1, min(int(start), total_frames - 1))
        if boundary > starts[-1]:
            starts.append(boundary)
    return [
        (start, starts[index + 1] - 1 if index + 1 < len(starts)
         else total_frames - 1)
        for index, start in enumerate(starts)
    ]


def detect_shots(video_id: str, path: Path, metadata: dict, cfg: dict) -> dict:
    """Use AutoShot when available; fallback is explicit in config for tests."""
    root, checkpoint = Path(cfg["root"]), Path(cfg["checkpoint"])
    try:
        if not root.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError("AutoShot root/checkpoint is missing")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aic_video_pipeline_v1_mpl"))
        import torch
        from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet
        from utils import get_batches, get_frames, predictions_to_scenes

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            small = get_frames(str(path), width=48, height=27)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cache_key = (str(root.resolve()), str(checkpoint.resolve()), device)
            model = _AUTOSHOT_MODEL_CACHE.get(cache_key)
            if model is None:
                model = TransNetV2Supernet().eval()
                checkpoint_data = torch.load(checkpoint, map_location=device)
                state = checkpoint_data.get("net", checkpoint_data)
                current = model.state_dict()
                current.update({key: value for key, value in state.items() if key in current})
                model.load_state_dict(current)
                model.to(device).eval()
                for module in model.modules():
                    if hasattr(module, "device"):
                        module.device = torch.device(device)
                _AUTOSHOT_MODEL_CACHE[cache_key] = model
            predictions = []
            with torch.inference_mode():
                for batch in get_batches(small):
                    tensor = torch.from_numpy(batch.transpose((3, 0, 1, 2))[None]).to(device)
                    output = model(tensor)
                    output = output[0] if isinstance(output, tuple) else output
                    predictions.append(torch.sigmoid(output[0]).detach().cpu().numpy()[25:75])
        values = np.concatenate(predictions)[:len(small)]
        raw_scenes = predictions_to_scenes(
            (values > float(cfg.get("threshold", 0.296))).astype(np.uint8)
        )
        scenes = _normalize_scenes(raw_scenes, metadata["total_frames"])
        shots = [{"shot_id": f"shot_{i:06d}", "frame_start": int(start), "frame_end": int(end),
                  "start_ms": round(start / metadata["fps"] * 1000),
                  "end_ms": round(end / metadata["fps"] * 1000)}
                 for i, (start, end) in enumerate(scenes, 1)]
        detector = "autoshot_v1"
    except Exception:
        if not cfg.get("allow_fallback", False):
            raise
        shots = [{"shot_id": "shot_000001", "frame_start": 0,
                  "frame_end": metadata["total_frames"] - 1, "start_ms": 0,
                  "end_ms": metadata["duration_ms"]}]
        detector = "fallback_full_video_v1"
    result = {"schema_version": "1.0", "video_id": video_id,
              "source_video_path": str(path), **metadata, "detector": detector, "shots": shots}
    _validate_shots(result)
    return result


def index_frames(video_id: str, metadata: dict, interval: int) -> list[FrameRecord]:
    """Sample the global source timeline independently from AutoShot.

    Frame zero is always selected. With interval 10 the sequence is
    ``0, 10, 20, ...``; shot ends are not injected because shot detection is a
    separate parallel branch.
    """
    if interval <= 0:
        raise ValueError("sample_every_frames must be positive")
    total = int(metadata["total_frames"])
    fps = float(metadata["fps"])
    return [
        FrameRecord(video_id=video_id, frame_id=str(index), frame_index=index,
                    timestamp_ms=round(index / fps * 1000))
        for index in range(0, total, interval)
    ]


def map_frames_to_shots(frames: list[FrameRecord], shots: dict) -> None:
    """Assign every independently sampled frame to exactly one AutoShot shot."""
    ordered_shots = sorted(shots["shots"], key=lambda item: int(item["frame_start"]))
    shot_index = 0
    for frame in sorted(frames, key=lambda item: item.frame_index):
        while (shot_index + 1 < len(ordered_shots)
               and frame.frame_index > int(ordered_shots[shot_index]["frame_end"])):
            shot_index += 1
        shot = ordered_shots[shot_index]
        if not int(shot["frame_start"]) <= frame.frame_index <= int(shot["frame_end"]):
            raise ValueError(f"frame {frame.frame_index} is not covered by any shot")
        frame.shot_id = str(shot["shot_id"])
        frame.mapping_status = "MAPPED"


def extract_frames(path: Path, frames: list[FrameRecord], root: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    decoded = 0
    for frame in sorted(frames, key=lambda item: item.frame_index):
        output = root / frame.video_id / f"{frame.frame_id}.png"
        image = None
        while decoded <= frame.frame_index:
            ok, image = capture.read()
            if not ok:
                image = None
                break
            decoded += 1
        if image is None:
            break
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.is_file() or output.stat().st_size == 0:
            temporary = output.with_suffix(".tmp.png")
            temporary.unlink(missing_ok=True)
            if not cv2.imwrite(str(temporary), image):
                temporary.unlink(missing_ok=True)
                raise ValueError(f"cannot write frame: {output}")
            os.replace(temporary, output)
        frame.mapping_status, frame.frame_path = "EXTRACTED", str(output)
    capture.release()
    frames[:] = [frame for frame in frames if frame.mapping_status == "EXTRACTED"]


def iter_sampled_images(path: Path, total_frames: int,
                        interval: int) -> Iterator[tuple[int, Image.Image]]:
    """Decode once and yield sampled RGB images without writing temporary PNGs."""
    if interval <= 0:
        raise ValueError("sample_every_frames must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        frame_index = 0
        while frame_index < total_frames:
            ok = capture.grab()
            if not ok:
                break
            if frame_index % interval == 0:
                ok, image = capture.retrieve()
                if not ok:
                    break
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield frame_index, Image.fromarray(rgb)
            frame_index += 1
    finally:
        capture.release()


def write_png(image: Image.Image, path: Path) -> Path:
    """Atomically persist a KEPT RGB frame only after similarity classification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    temporary.unlink(missing_ok=True)
    image.save(temporary, format="PNG", compress_level=3)
    if temporary.stat().st_size == 0:
        raise ValueError(f"cannot write frame: {path}")
    os.replace(temporary, path)
    return path


class FGClip2Embedder:
    def __init__(self, model_id: str, model_version: str, model_path: str | None,
                 revision: str | None = None) -> None:
        self.model_id, self.model_version = model_id, model_version
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.revision = revision
        self.loaded = False
        self.max_working_batch: int | None = None

    def _load(self) -> None:
        if self.loaded:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForCausalLM
        source = self.model_path if self.model_path and self.model_path.is_dir() else self.model_id
        local_only = isinstance(source, Path)
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        remote_revision = None if local_only else self.revision
        self.model = AutoModelForCausalLM.from_pretrained(
            source, trust_remote_code=True, local_files_only=local_only,
            revision=remote_revision,
        ).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(
            source, local_files_only=local_only, revision=remote_revision,
        )
        self.loaded = True

    @staticmethod
    def _max_num_patches(image: Image.Image) -> int:
        width, height = image.size
        patch_count = (width // 16) * (height // 16)
        if patch_count > 784:
            return 1024
        if patch_count > 576:
            return 784
        if patch_count > 256:
            return 576
        if patch_count > 128:
            return 256
        return 128

    def _embed_images_once(self, images: list[Image.Image]) -> np.ndarray:
        self._load()
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        converted = [image.convert("RGB") for image in images]
        patch_limits = {self._max_num_patches(image) for image in converted}
        if len(patch_limits) != 1:
            raise ValueError("one FG-CLIP2 batch must use a common patch limit")
        inputs = self.processor(images=converted,
                                max_num_patches=patch_limits.pop(),
                                return_tensors="pt")
        use_cuda = self.device.type == "cuda"
        if use_cuda:
            inputs = {key: value.pin_memory().to(self.device, non_blocking=True)
                      for key, value in inputs.items()}
        else:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.detach().float().cpu().numpy()

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a dynamic-resolution batch and halve it automatically on OOM."""
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        if self.max_working_batch is not None and len(images) > self.max_working_batch:
            parts = [self.embed_images(images[index:index + self.max_working_batch])
                     for index in range(0, len(images), self.max_working_batch)]
            return np.concatenate(parts, axis=0)
        try:
            return self._embed_images_once(images)
        except RuntimeError as error:
            message = str(error).lower()
            if "out of memory" not in message or len(images) == 1:
                raise
            if hasattr(self, "torch") and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            split = max(1, len(images) // 2)
            self.max_working_batch = split
            left = self.embed_images(images[:split])
            right = self.embed_images(images[split:])
            return np.concatenate((left, right), axis=0)

    def embed_frame(self, path: Path) -> np.ndarray:
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.embed_images([image])[0]


def embed_frames(frames: list[FrameRecord], embedder: FGClip2Embedder,
                 store: NpyVectorStore,
                 checkpoint_callback: Callable[[int, int], None] | None = None) -> None:
    total = len(frames)
    for position, frame in enumerate(frames, start=1):
        if not frame.frame_path:
            raise ValueError("frame must have a PNG path before embedding")
        expected = store.path_for(frame.video_id, frame.frame_id)
        if expected.is_file():
            try:
                store.get(expected)
                frame.vector_path = str(expected)
                frame.embedding_status = "EMBEDDED"
                if checkpoint_callback is not None:
                    checkpoint_callback(position, total)
                continue
            except (OSError, ValueError):
                expected.unlink(missing_ok=True)
        vector = embedder.embed_frame(Path(frame.frame_path))
        frame.vector_path = str(store.put(frame.video_id, frame.frame_id, vector))
        frame.embedding_status = "EMBEDDED"
        if checkpoint_callback is not None:
            checkpoint_callback(position, total)
