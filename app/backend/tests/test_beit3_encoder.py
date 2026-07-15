import numpy as np
import torch
from PIL import Image
from pathlib import Path
from unittest.mock import MagicMock, patch

from encoders.beit3 import Beit3Encoder, _checkpoint_state


DIM = 768


def _encoder() -> tuple[Beit3Encoder, MagicMock, MagicMock]:
    encoder = object.__new__(Beit3Encoder)
    encoder.device = torch.device("cpu")
    encoder._model = MagicMock()
    encoder._tokenizer = MagicMock()
    encoder._image_transform = lambda image: torch.ones(3, 224, 224)
    return encoder, encoder._model, encoder._tokenizer


def test_checkpoint_state_unwraps_module_prefix():
    state = _checkpoint_state({"model": {"module.layer.weight": torch.ones(2)}})
    assert list(state) == ["layer.weight"]


@patch("encoders.beit3.XLMRobertaTokenizer")
@patch("encoders.beit3.torch.load")
@patch("encoders.beit3._BEiT3RetrievalModel")
def test_constructor_uses_local_model_assets(model_cls, load, tokenizer_cls):
    model = model_cls.return_value
    model.load_state_dict.return_value = ([], [])
    load.return_value = {"model": {}}

    encoder = Beit3Encoder("/models/beit3.pth", "/models/beit3.spm", device="cpu")

    load.assert_called_once_with(
        Path("/models/beit3.pth"), map_location="cpu"
    )
    tokenizer_cls.assert_called_once_with("/models/beit3.spm")
    assert encoder.device == torch.device("cpu")


def test_encode_images_returns_normalized_float32_features():
    encoder, model, _ = _encoder()
    model.encode_image.return_value = torch.randn(2, DIM) * 4

    output = encoder.encode_images(
        [Image.new("RGB", (640, 480)), Image.new("RGB", (320, 240))]
    )

    assert output.shape == (2, DIM)
    assert output.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(output, axis=-1), np.ones(2), atol=1e-5)


def test_encode_texts_returns_normalized_float32_features():
    encoder, model, tokenizer = _encoder()
    tokenizer.return_value = {
        "input_ids": torch.zeros(2, 64, dtype=torch.long),
        "attention_mask": torch.ones(2, 64, dtype=torch.long),
    }
    model.encode_text.return_value = torch.randn(2, DIM) * 4

    output = encoder.encode_texts(["one", "two"])

    assert output.shape == (2, DIM)
    assert output.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(output, axis=-1), np.ones(2), atol=1e-5)
    tokenizer.assert_called_once_with(
        ["one", "two"],
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
