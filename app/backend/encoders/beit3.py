"""BEiT-3 visual retrieval encoder with 768-dimensional embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchscale.architecture.config import EncoderConfig
from torchscale.model.BEiT3 import BEiT3
from torchvision import transforms
from transformers import XLMRobertaTokenizer

CHECKPOINT_URL = (
    "https://github.com/addf400/files/releases/download/beit3/"
    "beit3_base_patch16_384_coco_retrieval.pth"
)
SENTENCEPIECE_URL = (
    "https://huggingface.co/linhuixiao/OneRef/resolve/main/"
    "beit3_checkpoints/beit3.spm"
)
EMBEDDING_DIM = 768
IMAGE_SIZE = 384
MAX_TEXT_LENGTH = 64


class BEiT3Retrieval(nn.Module):
    """Minimal BEiT-3 retrieval wrapper matching the COCO checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        cfg = EncoderConfig(
            img_size=IMAGE_SIZE,
            patch_size=16,
            vocab_size=64010,
            multiway=True,
            layernorm_embedding=False,
            normalize_output=True,
            no_output_layer=True,
            encoder_embed_dim=EMBEDDING_DIM,
            encoder_attention_heads=12,
            encoder_ffn_embed_dim=3072,
            encoder_layers=12,
        )
        self.beit3 = BEiT3(cfg)
        self.language_head = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.vision_head = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        encoder_out = self.beit3(
            textual_tokens=None,
            visual_tokens=image,
            text_padding_position=None,
        )["encoder_out"]
        cls_embedding = encoder_out[:, 0]
        return F.normalize(self.vision_head(cls_embedding), dim=-1)

    def encode_text(
        self,
        text_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_out = self.beit3(
            textual_tokens=text_tokens,
            visual_tokens=None,
            text_padding_position=padding_mask,
        )["encoder_out"]
        cls_embedding = encoder_out[:, 0]
        return F.normalize(self.language_head(cls_embedding), dim=-1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encode_image(image)


class BEiT3Encoder:
    """Owns BEiT-3 model state across requests."""

    model_id = "beit3_base_patch16_384_coco_retrieval"
    checkpoint_url = CHECKPOINT_URL
    embedding_dim = EMBEDDING_DIM

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(device or _default_device())
        self._model = BEiT3Retrieval()
        checkpoint = torch.hub.load_state_dict_from_url(
            CHECKPOINT_URL,
            map_location="cpu",
        )
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"BEiT-3 checkpoint missing weights: {missing}")
        if unexpected:
            print(f"BEiT-3 checkpoint ignored weights: {unexpected}")

        self._model = self._model.eval().to(self.device)
        self._tokenizer = XLMRobertaTokenizer(_sentencepiece_path())
        self._preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )
        print(f"BEiT-3 loaded on {self.device}. embedding_dim = {self.embedding_dim}")

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        tokens = self._tokenizer(
            list(texts),
            max_length=MAX_TEXT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_tokens = tokens["input_ids"].to(self.device)
        padding_mask = tokens["attention_mask"].eq(0).to(self.device)

        with torch.inference_mode():
            features = self._model.encode_text(text_tokens, padding_mask)

        return features.cpu().float().numpy()

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        batch = torch.stack(
            [self._preprocess(image.convert("RGB")) for image in images]
        ).to(self.device)

        with torch.inference_mode():
            features = self._model.encode_image(batch)

        return features.cpu().float().numpy()


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sentencepiece_path() -> str:
    model_dir = Path(torch.hub.get_dir()) / "checkpoints"
    path = model_dir / "beit3.spm"
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(SENTENCEPIECE_URL, str(path), progress=True)
    return str(path)
