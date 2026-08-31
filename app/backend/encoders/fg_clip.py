"""FG-CLIP image/text encoder backed by qihoo360/fg-clip2-large."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "qihoo360/fg-clip2-large"
MODEL_REVISION = "4d1d5dc35c716902f07c172dbfc23b82a7bc6bf3"
EMBEDDING_DIM = 1024


class FGClipEncoder:
    """Owns FG-CLIP model state across requests."""

    model_id = MODEL_ID
    embedding_dim = EMBEDDING_DIM

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(device or _default_device())
        print(self.device)

        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        ).to(self.device).eval()

        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )

        self.embedding_dim = self._infer_embedding_dim()
        print(f"Model loaded. embedding_dim = {self.embedding_dim}")

    def _infer_embedding_dim(self) -> int:
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        feats = self.encode_images([dummy])
        return int(feats.shape[-1])

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features.cpu().float().numpy()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self._processor(
            images=rgb_images,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features.cpu().float().numpy()


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
