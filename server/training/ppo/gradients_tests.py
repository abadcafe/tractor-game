"""Black-box tests for PPO gradient validation and clipping."""

from __future__ import annotations

import torch
from torch import Tensor

from server.training.ppo.gradients import (
    clip_gradient_group,
    non_finite_gradient_reason,
)


def test_gradient_reason_reports_parameter_and_values() -> None:
    first = torch.tensor([1.0], requires_grad=True)
    first.grad = torch.tensor([0.5])
    second = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    second.grad = torch.tensor([torch.nan, torch.inf, -torch.inf])

    reason = non_finite_gradient_reason(
        (("first", first), ("second", second)),
        stage="before clipping",
    )

    assert reason is not None
    assert "before clipping" in reason
    assert "parameter=second" in reason
    assert "nan_count=1" in reason
    assert "positive_inf_count=1" in reason
    assert "negative_inf_count=1" in reason
    assert "parameter_state=finite" in reason


def test_gradient_reason_reports_non_finite_parameter() -> None:
    parameter = torch.tensor([torch.nan, 2.0], requires_grad=True)
    parameter.grad = torch.tensor([torch.nan, 0.5])

    reason = non_finite_gradient_reason(
        (("weight", parameter),),
        stage="before clipping",
    )

    assert reason is not None
    assert (
        "parameter_state=nonfinite(nan_count=1,positive_inf_count=0,"
        + "negative_inf_count=0)"
        in reason
    )


def test_clip_grad_norm_on_device_preserves_small_norm() -> None:
    first = _parameter_with_gradient(
        torch.tensor([1.0]), torch.tensor([3.0])
    )
    second = _parameter_with_gradient(
        torch.tensor([2.0]), torch.tensor([4.0])
    )

    result = clip_gradient_group((first, second), max_norm=10.0)

    assert first.grad is not None
    assert second.grad is not None
    assert bool(result.gradients_are_finite.cpu().item())
    assert torch.equal(result.total_norm, torch.tensor(5.0))
    assert torch.equal(first.grad, torch.tensor([3.0]))
    assert torch.equal(second.grad, torch.tensor([4.0]))


def test_clip_grad_norm_on_device_preserves_tiny_finite_norm() -> None:
    parameter = _parameter_with_gradient(
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0e-40, 0.0], dtype=torch.float32),
    )

    result = clip_gradient_group((parameter,), max_norm=0.5)

    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert torch.equal(
        parameter.grad,
        torch.tensor([1.0e-40, 0.0], dtype=torch.float32),
    )
    assert bool(result.gradients_are_finite.cpu().item())


def test_clip_grad_norm_on_device_preserves_zero_norm() -> None:
    parameter = _parameter_with_gradient(
        torch.tensor([1.0, 2.0]), torch.zeros((2,), dtype=torch.float32)
    )

    result = clip_gradient_group((parameter,), max_norm=0.5)

    assert parameter.grad is not None
    assert torch.equal(parameter.grad, torch.zeros((2,)))
    assert torch.equal(result.total_norm, torch.tensor(0.0))


def test_clip_grad_norm_on_device_scales_global_norm() -> None:
    first = _parameter_with_gradient(
        torch.tensor([1.0]), torch.tensor([3.0])
    )
    second = _parameter_with_gradient(
        torch.tensor([2.0]), torch.tensor([4.0])
    )

    result = clip_gradient_group((first, second), max_norm=2.0)

    expected_scale = 2.0 / (5.0 + 0.000001)
    assert first.grad is not None
    assert second.grad is not None
    assert torch.allclose(
        first.grad, torch.tensor([3.0 * expected_scale])
    )
    assert torch.allclose(
        second.grad, torch.tensor([4.0 * expected_scale])
    )
    assert torch.equal(result.total_norm, torch.tensor(5.0))


def test_clip_grad_norm_on_device_ignores_missing_gradients() -> None:
    first = _parameter_with_gradient(
        torch.tensor([1.0]), torch.tensor([3.0])
    )
    second = torch.tensor([2.0], requires_grad=True)

    result = clip_gradient_group((first, second), max_norm=1.0)

    assert first.grad is not None
    expected_scale = 1.0 / (3.0 + 0.000001)
    assert torch.allclose(
        first.grad, torch.tensor([3.0 * expected_scale])
    )
    assert second.grad is None
    assert torch.equal(result.total_norm, torch.tensor(3.0))


def test_clip_grad_norm_on_device_scales_huge_finite_gradient() -> None:
    parameter = _parameter_with_gradient(
        torch.tensor([1.0]), torch.tensor([1.0e20])
    )

    result = clip_gradient_group((parameter,), max_norm=0.5)

    assert parameter.grad is not None
    assert torch.allclose(parameter.grad, torch.tensor([0.5]))
    assert bool(result.gradients_are_finite.cpu().item())


def test_clip_grad_norm_on_device_scales_mixed_extreme_gradients() -> (
    None
):
    parameter = _parameter_with_gradient(
        torch.tensor([1.0, 2.0]),
        torch.tensor([torch.finfo(torch.float32).max, 1.0e-40]),
    )

    result = clip_gradient_group((parameter,), max_norm=0.5)

    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert torch.allclose(parameter.grad, torch.tensor([0.5, 0.0]))
    assert bool(result.gradients_are_finite.cpu().item())


def test_clip_grad_norm_on_device_preserves_non_finite_gradient() -> (
    None
):
    finite = _parameter_with_gradient(
        torch.tensor([1.0]), torch.tensor([2.0])
    )
    non_finite = _parameter_with_gradient(
        torch.tensor([2.0]), torch.tensor([torch.inf])
    )

    result = clip_gradient_group((finite, non_finite), max_norm=0.5)

    assert finite.grad is not None
    assert non_finite.grad is not None
    assert torch.equal(finite.grad, torch.tensor([2.0]))
    assert torch.equal(non_finite.grad, torch.tensor([torch.inf]))
    assert not bool(result.gradients_are_finite.cpu().item())


def test_clip_gradient_group_reports_norm_without_clipping() -> None:
    parameter = _parameter_with_gradient(
        torch.tensor([1.0]), torch.tensor([3.0])
    )

    result = clip_gradient_group((parameter,), max_norm=0.0)

    assert parameter.grad is not None
    assert torch.equal(parameter.grad, torch.tensor([3.0]))
    assert torch.equal(result.total_norm, torch.tensor(3.0))


def _parameter_with_gradient(value: Tensor, gradient: Tensor) -> Tensor:
    parameter = value.detach().clone().requires_grad_(True)
    parameter.grad = gradient.detach().clone()
    return parameter
