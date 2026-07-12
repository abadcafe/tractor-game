"""PPO gradient validation and clipping on the parameter device."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from server.foundation import result as _result
from server.training.tensor_finiteness import (
    NamedTensorCheck,
    reject_if_non_finite,
)


def reject_if_gradients_non_finite(
    parameters: tuple[Tensor, ...],
) -> _result.Ok[None] | _result.Rejected:
    """Reject when any parameter gradient contains NaN or infinity."""
    gradients = _gradients(parameters)
    if not gradients:
        return _result.Ok(value=None)
    return reject_if_non_finite(
        tuple(
            NamedTensorCheck(
                tensor=gradient,
                reason="PPO gradients must be finite",
            )
            for gradient in gradients
        )
    )


def clip_grad_norm_on_device(
    parameters: tuple[Tensor, ...], *, max_norm: float
) -> None:
    """Clip gradients by global norm without synchronizing to CPU."""
    assert math.isfinite(max_norm)
    if max_norm <= 0.0:
        return
    gradients = _gradients(parameters)
    if not gradients:
        return
    device = gradients[0].device
    assert all(gradient.device == device for gradient in gradients)
    accumulation_dtype = _norm_accumulation_dtype(gradients)
    total_squared_norm = torch.zeros(
        (), dtype=accumulation_dtype, device=device
    )
    for gradient in gradients:
        detached = gradient.detach().to(dtype=accumulation_dtype)
        total_squared_norm = (
            total_squared_norm + (detached * detached).sum()
        )
    max_norm_tensor = torch.tensor(
        max_norm, dtype=accumulation_dtype, device=device
    )
    clip_coef = max_norm_tensor / (
        torch.sqrt(total_squared_norm) + 0.000001
    )
    bounded_clip_coef = torch.clamp(clip_coef, max=1.0)
    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(bounded_clip_coef.to(dtype=gradient.dtype))


def _gradients(parameters: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(
        gradient
        for parameter in parameters
        if (gradient := parameter.grad) is not None
    )


def _norm_accumulation_dtype(
    gradients: tuple[Tensor, ...],
) -> torch.dtype:
    if any(gradient.dtype == torch.float64 for gradient in gradients):
        return torch.float64
    return torch.float32
