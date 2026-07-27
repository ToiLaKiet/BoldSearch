"""Shared embedding normalization helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Return finite, float32 L2-normalized features."""
    return F.normalize(tensor.float(), p=2, dim=-1, eps=1e-12)
