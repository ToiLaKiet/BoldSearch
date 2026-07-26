from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import FrameRecord
from .storage import NpyVectorStore, sha256


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
            model = TransNetV2Supernet().eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint_data = torch.load(checkpoint, map_location=device)
            state = checkpoint_data.get("net", checkpoint_data)
            current = model.state_dict()
            current.update({key: value for key, value in state.items() if key in current})
            model.load_state_dict(current)
            model.to(device).eval()
            predictions = []
            with torch.inference_mode():
                for batch in get_batches(small):
                    tensor = torch.from_numpy(batch.transpose((3, 0, 1, 2))[None]).to(device)
                    output = model(tensor)
                    output = output[0] if isinstance(output, tuple) else output
                    predictions.append(torch.sigmoid(output[0]).detach().cpu().numpy()[25:75])
        values = np.concatenate(predictions)[:len(small)]
        scenes = predictions_to_scenes((values > float(cfg.get("threshold", 0.296))).astype(np.uint8))
        if len(scenes) == 0:
            scenes = [(0, metadata["total_frames"] - 1)]
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


def index_frames(video_id: str, shots: dict, interval: int,
                 width: int) -> list[FrameRecord]:
    frames: list[FrameRecord] = []
    for shot in shots["shots"]:
        start, end = int(shot["frame_start"]), int(shot["frame_end"])
        candidates = list(range(start, end + 1, interval))
        if not candidates or candidates[-1] != end:
            candidates.append(end)
        for index in candidates:
            frames.append(FrameRecord(video_id, f"{index:0{width}d}", index,
                                      round(index / shots["fps"] * 1000), shot["shot_id"]))
    return sorted({frame.frame_index: frame for frame in frames}.values(),
                  key=lambda item: item.frame_index)


def extract_frames(path: Path, frames: list[FrameRecord], root: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    decoded = 0
    for frame in sorted(frames, key=lambda item: item.frame_index):
        image = None
        while decoded <= frame.frame_index:
            ok, image = capture.read()
            if not ok:
                image = None
                break
            decoded += 1
        if image is None:
            continue
        output = root / frame.video_id / f"{frame.frame_id}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.png")
        if not cv2.imwrite(str(temporary), image):
            raise ValueError(f"cannot write frame: {output}")
        os.replace(temporary, output)
        frame.mapping_status, frame.frame_path = "EXTRACTED", str(output)
    capture.release()
    frames[:] = [frame for frame in frames if frame.mapping_status == "EXTRACTED"]


class FGClipEmbedder:
    def __init__(self, model_id: str, model_version: str, model_path: str | None) -> None:
        self.model_id, self.model_version = model_id, model_version
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForCausalLM
        source = self.model_path if self.model_path and self.model_path.is_dir() else self.model_id
        local_only = isinstance(source, Path)
        if self.model_path and not self.model_path.is_dir():
            raise FileNotFoundError(f"FGCLIP model directory is missing: {self.model_path}")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True,
                                                          local_files_only=local_only).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(source, local_files_only=local_only)
        self.loaded = True

    def embed_frame(self, path: Path) -> np.ndarray:
        self._load()
        with Image.open(path) as source:
            image = source.convert("RGB")
        pixels = self.processor(images=[image], return_tensors="pt")["pixel_values"].to(self.device)
        with self.torch.inference_mode():
            features = self.model.get_image_features(pixels)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features[0].detach().float().cpu().numpy()


def embed_frames(frames: list[FrameRecord], embedder: FGClipEmbedder,
                 store: NpyVectorStore) -> None:
    for frame in frames:
        if not frame.frame_path:
            raise ValueError("frame must have a PNG path before embedding")
        vector = embedder.embed_frame(Path(frame.frame_path))
        frame.vector_path = str(store.put(frame.video_id, frame.frame_id, vector))
        frame.embedding_status = "EMBEDDED"
