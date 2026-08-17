"""FG-CLIP image/text encoder backed by qihoo360/fg-clip2-large."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

from encoders.normalization import l2_normalize

MODEL_ID = "qihoo360/fg-clip2-large"
MODEL_REVISION = "4d1d5dc35c716902f07c172dbfc23b82a7bc6bf3"
EMBEDDING_DIM = 1024
MAX_TEXT_LENGTH = 64
TEXT_WALK_TYPE = "short"


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
        ).to(self.device)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True
        )
        self._image_processor = AutoImageProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True
        )
        # Suy ra embedding dim động bằng 1 forward pass giả (an toàn cho mọi variant base/large/so400m/huge)
        self.embedding_dim = self._infer_embedding_dim()
        print(f"Model loaded. embedding_dim = {self.embedding_dim}")
        
    def _infer_embedding_dim(self) -> int:
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        with torch.no_grad():
            feats = self.encode_images([dummy])
        
        return int(feats.shape[-1])
    
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        text_input = self._tokenizer(
            texts,
            max_length=MAX_TEXT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            features = self._model.get_text_features(
                **text_input,
                walk_type=TEXT_WALK_TYPE,
            )
            features = l2_normalize(features)
        print('hello')
        return features.cpu().float().numpy()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        rgb_images = [image.convert("RGB") for image in images]
        max_num_patches = max(_determine_max_num_patches(image) for image in rgb_images)
        image_input = self._image_processor(
            images=rgb_images,
            max_num_patches=max_num_patches,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            features = self._model.get_image_features(**image_input)
            features = l2_normalize(features)

        return features.cpu().float().numpy()


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _determine_max_num_patches(image: Image.Image) -> int:
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
# """FG-CLIP v1 image/text encoder backed by qihoo360/fg-clip-large."""

# from __future__ import annotations

# from typing import Optional, Sequence

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from transformers import (
#     AutoImageProcessor,
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )

# MODEL_PROVIDER = "fgclip"
# MODEL_ID = "qihoo360/fg-clip-large"
# MODEL_VERSION = "fgclip_v1"

# # Để None sẽ dùng revision hiện tại trên Hugging Face.
# # Khi production ổn định, nên thay bằng commit hash của fg-clip-large.
# MODEL_REVISION: Optional[str] = None

# IMAGE_SIZE = 336
# MAX_TEXT_LENGTH = 77
# WALK_SHORT_POS = True


# class FGClipEncoder:
#     """Keeps FG-CLIP v1 model state loaded across requests."""

#     provider = MODEL_PROVIDER
#     model_id = MODEL_ID
#     model_version = MODEL_VERSION

#     def __init__(
#         self,
#         device: Optional[str] = None,
#         revision: Optional[str] = MODEL_REVISION,
#     ) -> None:
#         self.device = torch.device(device or _default_device())
#         self.revision = revision

#         load_kwargs: dict[str, object] = {
#             "trust_remote_code": True,
#         }

#         if revision is not None:
#             load_kwargs["revision"] = revision

#         self._model = AutoModelForCausalLM.from_pretrained(
#             MODEL_ID,
#             **load_kwargs,
#         ).to(self.device)

#         self._model.eval()

#         tokenizer_kwargs: dict[str, object] = {}
#         processor_kwargs: dict[str, object] = {}

#         if revision is not None:
#             tokenizer_kwargs["revision"] = revision
#             processor_kwargs["revision"] = revision

#         self._tokenizer = AutoTokenizer.from_pretrained(
#             MODEL_ID,
#             **tokenizer_kwargs,
#         )

#         self._image_processor = AutoImageProcessor.from_pretrained(
#             MODEL_ID,
#             **processor_kwargs,
#         )

#         self.embedding_dim = self._infer_embedding_dim()

#     def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
#         """Encode text into normalized FG-CLIP embeddings."""

#         if not texts:
#             return np.empty((0, self.embedding_dim), dtype=np.float32)

#         encoded = self._tokenizer(
#             list(texts),
#             max_length=MAX_TEXT_LENGTH,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         )

#         # FG-CLIP v1's custom implementation expects input_ids directly.
#         input_ids = encoded["input_ids"].to(self.device)

#         with torch.inference_mode():
#             features = self._model.get_text_features(
#                 input_ids,
#                 walk_short_pos=WALK_SHORT_POS,
#             )
#             features = F.normalize(features.float(), p=2, dim=-1)

#         return features.cpu().numpy()

#     def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
#         """Encode images into normalized FG-CLIP embeddings."""
#         print("run encode image", len(images))
#         if not images:
#             return np.empty((0, self.embedding_dim), dtype=np.float32)

#         prepared_images = [image.convert("RGB") for image in images]
#         processed = self._image_processor.preprocess(images=prepared_images, return_tensors="pt")

#         pixel_values = processed["pixel_values"].to(self.device)

#         with torch.inference_mode():
#             features = self._model.get_image_features(pixel_values)
#             features = F.normalize(features.float(), p=2, dim=-1)
#         print(features.shape)
#         return features.cpu().numpy()

#     def _infer_embedding_dim(self) -> int:
#         """Determine projected embedding dimension from model configuration."""

#         config = self._model.config

#         for attribute in (
#             "projection_dim",
#             "embed_dim",
#             "embedding_dim",
#         ):
#             value = getattr(config, attribute, None)
#             if isinstance(value, int) and value > 0:
#                 return value

#         # Fallback đảm bảo lấy đúng kích thước từ chính checkpoint.
#         dummy_text = self._tokenizer(
#             [""],
#             max_length=MAX_TEXT_LENGTH,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         )["input_ids"].to(self.device)

#         with torch.inference_mode():
#             features = self._model.get_text_features(
#                 dummy_text,
#                 walk_short_pos=WALK_SHORT_POS,
#             )

#         return int(features.shape[-1])


# def _default_device() -> str:
#     if torch.cuda.is_available():
#         return "cuda"

#     if torch.backends.mps.is_available():
#         return "mps"

#     return "cpu"
