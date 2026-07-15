"""Tests for FGClipEncoder.

Fakes replace AutoModelForCausalLM / AutoTokenizer / AutoImageProcessor at the
transformers seam — the real model is never downloaded. Normalization and
batching are exercised with real torch ops.

TDD cycle applied to l2_normalize (core normalization kernel):
  Red  — tests written first; function not yet present → ImportError/NameError
  Green — minimal implementation added to fg_clip.py
  Refactor — docstring and clamp constant clarified; tests stay green
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from encoders.fg_clip import FGClipEncoder, _MODEL_REVISION
from encoders.normalization import l2_normalize

DIM = 768


# ---------------------------------------------------------------------------
# l2_normalize — pure-function TDD cycle
# ---------------------------------------------------------------------------

class TestL2Normalize:
    def test_unit_vector_passes_through(self):
        v = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(l2_normalize(v), v)

    def test_scaled_vector_becomes_unit(self):
        # 3-4-5 right triangle: norm = 5
        v = torch.tensor([[3.0, 4.0]])
        out = l2_normalize(v)
        assert abs(out.norm(p=2, dim=-1).item() - 1.0) < 1e-6

    def test_zero_vector_stays_zero(self):
        v = torch.zeros(1, 4, dtype=torch.float16)
        out = l2_normalize(v)
        assert torch.isfinite(out).all()
        assert torch.all(out == 0.0)

    def test_batch_all_rows_unit_norm(self):
        torch.manual_seed(0)
        v = torch.randn(16, DIM) * 7.3
        out = l2_normalize(v)
        norms = out.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(16), atol=1e-5)

    def test_float16_input_returns_finite_float32_unit_vectors(self):
        v = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
        out = l2_normalize(v)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert torch.allclose(out.norm(p=2, dim=-1), torch.ones(1))


# ---------------------------------------------------------------------------
# Fixture — FGClipEncoder with faked external seam
# ---------------------------------------------------------------------------

@pytest.fixture
def mocked_enc():
    """Yield (encoder, fake_model, fake_tokenizer, fake_image_processor).

    Patches are held open for the duration of each test so the encoder's
    own calls to the fakes go through the mocked attributes.
    """
    with (
        patch("encoders.fg_clip.AutoModelForCausalLM") as mock_causal,
        patch("encoders.fg_clip.AutoTokenizer") as mock_tok,
        patch("encoders.fg_clip.AutoImageProcessor") as mock_proc,
    ):
        fake_model = MagicMock()
        # .to(device) must return the same object, not a new MagicMock
        fake_model.to.return_value = fake_model
        mock_causal.from_pretrained.return_value = fake_model

        fake_tokenizer = MagicMock()
        # Default: 1 text, 77 token ids
        fake_tokenizer.return_value.input_ids = [[0] * 77]
        mock_tok.from_pretrained.return_value = fake_tokenizer

        fake_proc = MagicMock()
        mock_proc.from_pretrained.return_value = fake_proc

        enc = FGClipEncoder(device="cpu")
        yield enc, fake_model, fake_tokenizer, fake_proc


# ---------------------------------------------------------------------------
# Constructor — revision and trust_remote_code are wired correctly
# ---------------------------------------------------------------------------

class TestConstructor:
    """Verify that revision and trust_remote_code reach each from_pretrained call."""

    def _build(self):
        """Construct FGClipEncoder with full seam patching; return the three mocks."""
        with (
            patch("encoders.fg_clip.AutoModelForCausalLM") as mock_causal,
            patch("encoders.fg_clip.AutoTokenizer") as mock_tok,
            patch("encoders.fg_clip.AutoImageProcessor") as mock_proc,
        ):
            fake = MagicMock()
            fake.to.return_value = fake
            mock_causal.from_pretrained.return_value = fake
            mock_tok.from_pretrained.return_value = MagicMock()
            mock_proc.from_pretrained.return_value = MagicMock()
            FGClipEncoder(device="cpu")
            return (
                mock_causal.from_pretrained.call_args.kwargs,
                mock_tok.from_pretrained.call_args.kwargs,
                mock_proc.from_pretrained.call_args.kwargs,
            )

    def test_model_receives_pinned_revision(self):
        model_kwargs, _, _ = self._build()
        assert model_kwargs["revision"] == _MODEL_REVISION

    def test_model_receives_trust_remote_code(self):
        model_kwargs, _, _ = self._build()
        assert model_kwargs["trust_remote_code"] is True

    def test_tokenizer_receives_pinned_revision(self):
        _, tok_kwargs, _ = self._build()
        assert tok_kwargs["revision"] == _MODEL_REVISION

    def test_image_processor_receives_pinned_revision(self):
        _, _, proc_kwargs = self._build()
        assert proc_kwargs["revision"] == _MODEL_REVISION


def _pil_images(n: int):
    from PIL import Image
    return [Image.new("RGB", (100, 100)) for _ in range(n)]


# ---------------------------------------------------------------------------
# encode_images
# ---------------------------------------------------------------------------

class TestEncodeImages:
    def test_shape_single(self, mocked_enc):
        enc, fake_model, _, fake_proc = mocked_enc
        fake_proc.preprocess.return_value = {"pixel_values": torch.randn(1, 3, 336, 336)}
        fake_model.get_image_features.return_value = torch.randn(1, DIM) * 5
        out = enc.encode_images(_pil_images(1))
        assert out.shape == (1, DIM)

    def test_shape_batch(self, mocked_enc):
        enc, fake_model, _, fake_proc = mocked_enc
        n = 4
        fake_proc.preprocess.return_value = {"pixel_values": torch.randn(n, 3, 336, 336)}
        fake_model.get_image_features.return_value = torch.randn(n, DIM) * 3
        out = enc.encode_images(_pil_images(n))
        assert out.shape == (n, DIM)

    def test_dtype_float32(self, mocked_enc):
        enc, fake_model, _, fake_proc = mocked_enc
        fake_proc.preprocess.return_value = {"pixel_values": torch.randn(2, 3, 336, 336)}
        fake_model.get_image_features.return_value = torch.randn(2, DIM)
        out = enc.encode_images(_pil_images(2))
        assert out.dtype == np.float32

    def test_output_l2_normalized(self, mocked_enc):
        enc, fake_model, _, fake_proc = mocked_enc
        n = 3
        fake_proc.preprocess.return_value = {"pixel_values": torch.randn(n, 3, 336, 336)}
        # Un-normalized raw features — encoder must normalize them
        fake_model.get_image_features.return_value = torch.randn(n, DIM) * 7
        out = enc.encode_images(_pil_images(n))
        norms = np.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(norms, np.ones(n), atol=1e-5)

    def test_images_resized_before_preprocess(self, mocked_enc):
        enc, _, _, fake_proc = mocked_enc
        fake_proc.preprocess.return_value = {"pixel_values": torch.randn(1, 3, 336, 336)}
        enc._model.get_image_features.return_value = torch.randn(1, DIM)
        from PIL import Image
        img = Image.new("RGB", (640, 480))
        enc.encode_images([img])
        # The image passed to preprocess must be 336×336
        call_args = fake_proc.preprocess.call_args
        passed_images = call_args.args[0]
        assert passed_images[0].size == (336, 336)


# ---------------------------------------------------------------------------
# encode_texts
# ---------------------------------------------------------------------------

class TestEncodeTexts:
    def test_shape_single(self, mocked_enc):
        enc, fake_model, fake_tok, _ = mocked_enc
        fake_tok.return_value.input_ids = [[0] * 77]
        fake_model.get_text_features.return_value = torch.randn(1, DIM) * 5
        out = enc.encode_texts(["hello world"])
        assert out.shape == (1, DIM)

    def test_shape_batch(self, mocked_enc):
        enc, fake_model, fake_tok, _ = mocked_enc
        n = 5
        fake_tok.return_value.input_ids = [[0] * 77] * n
        fake_model.get_text_features.return_value = torch.randn(n, DIM) * 3
        out = enc.encode_texts(["a", "b", "c", "d", "e"])
        assert out.shape == (n, DIM)

    def test_dtype_float32(self, mocked_enc):
        enc, fake_model, fake_tok, _ = mocked_enc
        fake_tok.return_value.input_ids = [[0] * 77] * 2
        fake_model.get_text_features.return_value = torch.randn(2, DIM)
        out = enc.encode_texts(["foo", "bar"])
        assert out.dtype == np.float32

    def test_output_l2_normalized(self, mocked_enc):
        enc, fake_model, fake_tok, _ = mocked_enc
        n = 4
        fake_tok.return_value.input_ids = [[0] * 77] * n
        fake_model.get_text_features.return_value = torch.randn(n, DIM) * 10
        out = enc.encode_texts(["q1", "q2", "q3", "q4"])
        norms = np.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(norms, np.ones(n), atol=1e-5)

    def test_walk_short_pos_forwarded(self, mocked_enc):
        enc, fake_model, fake_tok, _ = mocked_enc
        fake_tok.return_value.input_ids = [[0] * 77]
        fake_model.get_text_features.return_value = torch.randn(1, DIM)
        enc.encode_texts(["test"], walk_short_pos=False)
        kwargs = fake_model.get_text_features.call_args.kwargs
        assert kwargs.get("walk_short_pos") is False

    def test_tokenizer_called_with_correct_params(self, mocked_enc):
        enc, fake_model, fake_tok, _ = mocked_enc
        fake_tok.return_value.input_ids = [[0] * 77]
        fake_model.get_text_features.return_value = torch.randn(1, DIM)
        enc.encode_texts(["sample text"])
        fake_tok.assert_called_once_with(
            ["sample text"],
            max_length=77,
            padding="max_length",
            truncation=True,
        )
