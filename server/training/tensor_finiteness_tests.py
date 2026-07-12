"""Black-box tests for device tensor finiteness validation."""

from __future__ import annotations

import torch

from server.foundation.result import Ok, Rejected
from server.training.tensor_finiteness import (
    NamedTensorCheck,
    reject_if_non_finite,
)


def test_reject_if_non_finite_accepts_finite_tensors() -> None:
    result = reject_if_non_finite(
        (
            NamedTensorCheck(
                tensor=torch.tensor([1.0, 2.0]),
                reason="first tensor must be finite",
            ),
            NamedTensorCheck(
                tensor=torch.tensor([3.0]),
                reason="second tensor must be finite",
            ),
        )
    )

    assert isinstance(result, Ok)


def test_reject_if_non_finite_returns_failed_tensor_reason() -> None:
    result = reject_if_non_finite(
        (
            NamedTensorCheck(
                tensor=torch.tensor([1.0]),
                reason="first tensor must be finite",
            ),
            NamedTensorCheck(
                tensor=torch.tensor([torch.inf]),
                reason="second tensor must be finite",
            ),
        )
    )

    assert isinstance(result, Rejected)
    assert result.reason == "second tensor must be finite"


def test_reject_if_non_finite_accepts_empty_tensor() -> None:
    result = reject_if_non_finite(
        (
            NamedTensorCheck(
                tensor=torch.empty((0,), dtype=torch.float32),
                reason="empty tensor must be finite",
            ),
        )
    )

    assert isinstance(result, Ok)
