"""FG-CLIP image/text encoder backed by qihoo360/fg-clip-large."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

from encoders.normalization import l2_normalize

_MODEL_ID = "qihoo360/fg-clip-large"
# Pinned to avoid pulling untrusted remote code from a moving HEAD.
_MODEL_REVISION = "5a8f0f23b5a06dc92310e907599b2a0c2d58fe6f"
_IMAGE_SIZE = 336
_EMBEDDING_DIM = 768
_MAX_TEXT_LENGTH = 77


class FGClipEncoder:
    """Encodes images and texts using FG-CLIP (qihoo360/fg-clip-large).

    Owns model, tokenizer, image-processor, and device state across calls.
    Output vectors are L2-normalized float32 with shape (n, 768).
    """

    MODEL_ID = _MODEL_ID
    EMBEDDING_DIM = _EMBEDDING_DIM

    def __init__(self, device: str | None = None) -> None:
        """Load model, tokenizer, and image processor onto *device*.

        Args:
            device: ``"cuda"``, ``"mps"``, or ``"cpu"``. Auto-detected when None.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self._model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, revision=_MODEL_REVISION, trust_remote_code=True
        ).to(self.device)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_ID, revision=_MODEL_REVISION
        )
        self._image_processor = AutoImageProcessor.from_pretrained(
            _MODEL_ID, revision=_MODEL_REVISION
        )

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode PIL images to L2-normalized float32 embeddings.

        Args:
            images: List of PIL images; each resized to 336×336 before encoding.

        Returns:
            ndarray of shape (n, 768), dtype float32, each row has L2 norm ≈ 1.
        """
        resized = [img.resize((_IMAGE_SIZE, _IMAGE_SIZE)) for img in images]
        pixel_values = self._image_processor.preprocess(
            resized, return_tensors="pt"
        )["pixel_values"].to(self.device)

        with torch.no_grad():
            features = self._model.get_image_features(pixel_values)
            features = l2_normalize(features)

        return features.cpu().float().numpy()

    def encode_texts(
        self, texts: list[str], *, walk_short_pos: bool = True
    ) -> np.ndarray:
        """Encode strings to L2-normalized float32 embeddings.

        Args:
            texts: List of strings; max 77 tokens per entry in short-caption mode.
            walk_short_pos: FG-CLIP positional-walk flag; True for ≤77-token captions.

        Returns:
            ndarray of shape (n, 768), dtype float32, each row has L2 norm ≈ 1.
        """
        input_ids = torch.tensor(
            self._tokenizer(
                texts,
                max_length=_MAX_TEXT_LENGTH,
                padding="max_length",
                truncation=True,
            ).input_ids,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            features = self._model.get_text_features(
                input_ids, walk_short_pos=walk_short_pos
            )
            features = l2_normalize(features)

        return features.cpu().float().numpy()
