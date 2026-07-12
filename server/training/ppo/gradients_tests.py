"""Black-box tests for PPO gradient validation and clipping."""

from __future__ import annotations

import torch
from torch import Tensor

from server.foundation.result import Ok, Rejected
from server.training.ppo.gradients import (
    clip_grad_norm_on_device,
    reject_if_gradients_non_finite,
)


def test_reject_if_gradients_non_finite_accepts_finite_gradients() -> (
    None
):
    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = torch.tensor([0.25, -0.5])

    result = reject_if_gradients_non_finite((parameter,))

    assert isinstance(result, Ok)


def test_reject_if_gradients_non_finite_rejects_nan_gradient() -> None:
    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = torch.tensor([torch.nan, 0.5])

    result = reject_if_gradients_non_finite((parameter,))

    assert isinstance(result, Rejected)
    assert result.reason == "PPO gradients must be finite"


def test_clip_grad_norm_on_device_preserves_small_norm() -> None:
    first = _parameter_with_gradient(
        torch.tensor([1.0]), torch.tensor([3.0])
    )
    second = _parameter_with_gradient(
        torch.tensor([2.0]), torch.tensor([4.0])
    )

    clip_grad_norm_on_device((first, second), max_norm=10.0)

    assert first.grad is not None
    assert second.grad is not None
    assert torch.equal(first.grad, torch.tensor([3.0]))
    assert torch.equal(second.grad, torch.tensor([4.0]))


def test_clip_grad_norm_on_device_scales_global_norm() -> None:
    first = _parameter_with_gradient(
        torch.tensor([1.0]), torch.tensor([3.0])
    )
    second = _parameter_with_gradient(
        torch.tensor([2.0]), torch.tensor([4.0])
    )

    clip_grad_norm_on_device((first, second), max_norm=2.0)

    expected_scale = 2.0 / (5.0 + 0.000001)
    assert first.grad is not None
    assert second.grad is not None
    assert torch.allclose(
        first.grad, torch.tensor([3.0 * expected_scale])
    )
    assert torch.allclose(
        second.grad, torch.tensor([4.0 * expected_scale])
    )


def test_clip_grad_norm_on_device_ignores_missing_gradients() -> None:
    first = _parameter_with_gradient(
        torch.tensor([1.0]), torch.tensor([3.0])
    )
    second = torch.tensor([2.0], requires_grad=True)

    clip_grad_norm_on_device((first, second), max_norm=1.0)

    assert first.grad is not None
    expected_scale = 1.0 / (3.0 + 0.000001)
    assert torch.allclose(
        first.grad, torch.tensor([3.0 * expected_scale])
    )
    assert second.grad is None


def _parameter_with_gradient(value: Tensor, gradient: Tensor) -> Tensor:
    parameter = value.detach().clone().requires_grad_(True)
    parameter.grad = gradient.detach().clone()
    return parameter
