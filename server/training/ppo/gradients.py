"""PPO gradient validation and clipping on the parameter device."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from server.foundation import result as _result
from server.training.rollout_inference.tensor_validation import (
    NamedTensorCheck,
    reject_if_non_finite,
)

type GradientValidationStage = Literal[
    "before clipping", "after clipping"
]
type NamedParameter = tuple[str, Tensor]


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


def non_finite_gradient_reason(
    parameters: tuple[NamedParameter, ...],
    *,
    stage: GradientValidationStage,
) -> str | None:
    """Describe the first non-finite parameter gradient."""
    for name, parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if bool(finite.all().detach().cpu().item()):
            continue
        detached = gradient.detach()
        nan_count = int(torch.isnan(detached).sum().cpu().item())
        positive_inf_count = int(
            torch.isposinf(detached).sum().cpu().item()
        )
        negative_inf_count = int(
            torch.isneginf(detached).sum().cpu().item()
        )
        finite_values = detached[finite]
        max_abs_finite = (
            0.0
            if finite_values.numel() == 0
            else float(finite_values.abs().max().cpu().item())
        )
        return (
            f"PPO gradients must be finite {stage}: parameter={name}, "
            f"nan_count={nan_count}, "
            f"positive_inf_count={positive_inf_count}, "
            f"negative_inf_count={negative_inf_count}, "
            f"max_abs_finite={max_abs_finite}, "
            f"parameter_state={_parameter_state(parameter)}"
        )
    return None


def _parameter_state(parameter: Tensor) -> str:
    detached = parameter.detach()
    finite = torch.isfinite(detached)
    if bool(finite.all().cpu().item()):
        return "finite"
    return (
        "nonfinite("
        f"nan_count={int(torch.isnan(detached).sum().cpu().item())},"
        f"positive_inf_count="
        f"{int(torch.isposinf(detached).sum().cpu().item())},"
        f"negative_inf_count="
        f"{int(torch.isneginf(detached).sum().cpu().item())}"
        ")"
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
    max_abs = torch.zeros((), dtype=accumulation_dtype, device=device)
    for gradient in gradients:
        detached = gradient.detach().to(dtype=accumulation_dtype)
        max_abs = torch.maximum(
            max_abs,
            detached.abs().max(),
        )
    finite_max = torch.isfinite(max_abs)
    scale = torch.where(
        finite_max & (max_abs > 0.0),
        max_abs,
        torch.ones_like(max_abs),
    )
    scaled_squared_norm = torch.zeros_like(max_abs)
    for gradient in gradients:
        detached = gradient.detach().to(dtype=accumulation_dtype)
        scaled = detached / scale
        scaled_squared_norm = (
            scaled_squared_norm + (scaled * scaled).sum()
        )
    scaled_norm = torch.sqrt(scaled_squared_norm)
    finite_positive_norm = torch.isfinite(scaled_norm) & (
        scaled_norm > 0.0
    )
    safe_scaled_norm = torch.where(
        finite_positive_norm,
        scaled_norm,
        torch.ones_like(scaled_norm),
    )
    max_norm_tensor = torch.tensor(
        max_norm, dtype=accumulation_dtype, device=device
    )
    clip_coef = (max_norm_tensor / safe_scaled_norm) / (
        scale + 0.000001 / safe_scaled_norm
    )
    bounded_clip_coef = torch.where(
        finite_max & (max_abs > 0.0) & finite_positive_norm,
        torch.clamp(clip_coef, max=1.0),
        torch.ones_like(clip_coef),
    )
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
