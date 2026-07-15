"""Tests for model-rank runtime setup."""

from __future__ import annotations

import torch

from server.foundation.result import Ok
from server.training.runtime.model_rank.process import (
    resolve_model_rank_device,
)


def test_resolve_mps_device_uses_tensor_device_identity() -> None:
    if not torch.backends.mps.is_available():
        return

    result = resolve_model_rank_device(
        model_rank_kind="mps",
        model_rank_device="mps",
    )

    assert isinstance(result, Ok)
    actual_device = torch.empty((), device=result.value).device
    assert result.value == torch.device("mps:0")
    assert actual_device == result.value
