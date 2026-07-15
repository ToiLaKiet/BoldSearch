"""BEiT-3 image/text retrieval encoder."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchscale.architecture.config import EncoderConfig
from torchscale.model.BEiT3 import BEiT3
from torchvision import transforms
from transformers import XLMRobertaTokenizer

from encoders.normalization import l2_normalize

_MODEL_ID = "microsoft/beit3-base-itc-patch16-224"
_IMAGE_SIZE = 224
_EMBEDDING_DIM = 768
_MAX_TEXT_LENGTH = 64
_VOCAB_SIZE = 64010


def _beit3_config() -> EncoderConfig:
    """Build the official BEiT-3 base configuration used by the retrieval checkpoint."""
    return EncoderConfig(
        img_size=_IMAGE_SIZE,
        patch_size=16,
        vocab_size=_VOCAB_SIZE,
        multiway=True,
        layernorm_embedding=False,
        normalize_output=True,
        no_output_layer=True,
        drop_path_rate=0,
        encoder_embed_dim=_EMBEDDING_DIM,
        encoder_attention_heads=12,
        encoder_ffn_embed_dim=3072,
        encoder_layers=12,
    )


class _BEiT3RetrievalModel(nn.Module):
    """Own the official BEiT-3 backbone and retrieval projection heads."""

    def __init__(self) -> None:
        super().__init__()
        self.beit3 = BEiT3(_beit3_config())
        self.language_head = nn.Linear(_EMBEDDING_DIM, _EMBEDDING_DIM, bias=False)
        self.vision_head = nn.Linear(_EMBEDDING_DIM, _EMBEDDING_DIM, bias=False)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Return projected image features before final normalization."""
        outputs = self.beit3(
            textual_tokens=None,
            visual_tokens=image,
            text_padding_position=None,
        )
        return self.vision_head(outputs["encoder_out"][:, 0, :])

    def encode_text(
        self, text: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return projected text features before final normalization."""
        outputs = self.beit3(
            textual_tokens=text,
            visual_tokens=None,
            text_padding_position=padding_mask,
        )
        return self.language_head(outputs["encoder_out"][:, 0, :])


def _checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a plain state dict from common PyTorch checkpoint wrappers."""
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("BEiT-3 checkpoint does not contain a state dict")

    state: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        while key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    return state


class Beit3Encoder:
    """Encode images and text with the official BEiT3-base ITC checkpoint.

    The checkpoint is the official image-text retrieval variant, so both
    modalities share the 768-dimensional normalized embedding space.
    """

    MODEL_ID = _MODEL_ID
    EMBEDDING_DIM = _EMBEDDING_DIM

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: str | None = None,
    ) -> None:
        """Load local DVC-managed BEiT-3 assets and the retrieval tokenizer."""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self._model = _BEiT3RetrievalModel()
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        state = _checkpoint_state(checkpoint)
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        unexpected = [key for key in unexpected if key != "logit_scale"]
        if missing or unexpected:
            raise RuntimeError(
                "BEiT-3 checkpoint does not match the retrieval model: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self._model.to(self.device).eval()
        self._tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
        self._image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (_IMAGE_SIZE, _IMAGE_SIZE),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode PIL images to normalized float32 vectors of shape ``(n, 768)``."""
        pixels = torch.stack([self._image_transform(image) for image in images]).to(
            self.device
        )
        with torch.no_grad():
            features = l2_normalize(self._model.encode_image(pixels))
        return features.cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode strings to normalized float32 vectors of shape ``(n, 768)``."""
        encoded = self._tokenizer(
            texts,
            max_length=_MAX_TEXT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        padding_mask = (encoded["attention_mask"] == 0).to(self.device)
        with torch.no_grad():
            features = l2_normalize(self._model.encode_text(input_ids, padding_mask))
        return features.cpu().numpy()
