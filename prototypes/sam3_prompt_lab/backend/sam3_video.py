"""SAM3 text-prompt inference over sampled video frames for the local test UI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Sam3Match:
    score: float
    box: tuple[int, int, int, int]
    mask: np.ndarray


class LocalSam3:
    """Lazy singleton around the local Meta SAM3 image model."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.model: Any | None = None
        self.processor: Any | None = None
        self.torch: Any | None = None

    def status(self) -> dict[str, object]:
        if not self.checkpoint_path.is_file():
            return {"ready": False, "reason": f"Missing checkpoint: {self.checkpoint_path}"}
        try:
            import torch
            import sam3  # noqa: F401
        except ImportError:
            return {
                "ready": False,
                "reason": "SAM3 package is missing. Start with /home/long/.venvs/sam3/bin/python.",
            }
        if not torch.cuda.is_available():
            return {"ready": False, "reason": "CUDA GPU is required by this SAM3 backend."}
        return {"ready": True, "checkpoint": str(self.checkpoint_path)}

    def _load(self) -> None:
        if self.model is not None:
            return
        current_status = self.status()
        if not current_status["ready"]:
            raise RuntimeError(str(current_status["reason"]))

        import torch
        from PIL import Image
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.torch = torch
        self.image_factory = Image
        self.model = build_sam3_image_model(
            checkpoint_path=str(self.checkpoint_path), load_from_HF=False, device="cuda"
        )
        self.processor = Sam3Processor(self.model, device="cuda")

    def detect(self, frame: np.ndarray, prompt: str, confidence: float) -> list[Sam3Match]:
        self._load()
        assert self.processor is not None and self.torch is not None
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            state = self.processor.set_image(self.image_factory.fromarray(rgb))
            self.processor.reset_all_prompts(state)
            output = self.processor.set_text_prompt(prompt=prompt, state=state)

        scores = output["scores"].detach().float().cpu().numpy().reshape(-1)
        boxes = output["boxes"].detach().float().cpu().numpy()
        masks = output["masks"].detach().cpu().numpy()
        matches: list[Sam3Match] = []
        for index, score in enumerate(scores):
            if float(score) < confidence or index >= len(boxes) or index >= len(masks):
                continue
            mask = masks[index]
            if mask.ndim == 3:
                mask = mask[0]
            mask = mask.astype(bool)
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not mask.any():
                continue
            x1, y1, x2, y2 = boxes[index].tolist()
            box = (
                max(0, min(width - 1, int(x1))),
                max(0, min(height - 1, int(y1))),
                max(0, min(width, int(x2))),
                max(0, min(height, int(y2))),
            )
            matches.append(Sam3Match(float(score), box, mask))
        self.torch.cuda.empty_cache()
        return matches


def draw_matches(frame: np.ndarray, prompt: str, matches: list[Sam3Match]) -> np.ndarray:
    """Overlay masks and boxes onto a BGR frame for browser review."""
    overlay = frame.copy()
    for match in matches:
        overlay[match.mask] = (45, 205, 75)
    rendered = cv2.addWeighted(overlay, 0.34, frame, 0.66, 0)
    for number, match in enumerate(matches, start=1):
        x1, y1, x2, y2 = match.box
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (45, 205, 75), 2)
        cv2.putText(
            rendered, f"{prompt} #{number} {match.score:.2f}", (x1, max(28, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return rendered


def process_video(
    model: LocalSam3,
    video_path: Path,
    output_dir: Path,
    prompt: str,
    sample_every_seconds: float,
    confidence: float,
    max_samples: int,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Cannot open the uploaded video.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        capture.release()
        raise ValueError("Cannot read video frame rate or length.")

    requested_interval = max(1, round(fps * sample_every_seconds))
    budget_interval = max(1, int(np.ceil(total_frames / max_samples)))
    interval = max(requested_interval, budget_interval)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_count = 0
    matches: list[dict[str, object]] = []
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_number % interval == 0:
            sampled_count += 1
            detections = model.detect(frame, prompt, confidence)
            if detections:
                timestamp = frame_number / fps
                filename = f"frame_{sampled_count:04d}_{timestamp:09.2f}s.jpg"
                cv2.imwrite(str(output_dir / filename), draw_matches(frame, prompt, detections))
                matches.append(
                    {
                        "timestampSeconds": round(timestamp, 3),
                        "frame": filename,
                        "detectionCount": len(detections),
                        "bestScore": round(max(item.score for item in detections), 4),
                    }
                )
        frame_number += 1
    capture.release()
    return {
        "durationSeconds": round(total_frames / fps, 3),
        "sampledFrameCount": sampled_count,
        "effectiveSampleEverySeconds": round(interval / fps, 3),
        "matches": matches,
    }
