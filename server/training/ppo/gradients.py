"""PPO gradient validation and clipping on the parameter device."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import torch
from torch import Tensor

type GradientValidationStage = Literal["before clipping"]
type NamedParameter = tuple[str, Tensor]


class _ForeachNorm(Protocol):
    def __call__(
        self,
        tensors: tuple[Tensor, ...],
        ord: float,
    ) -> tuple[Tensor, ...]: ...


class _ForeachDiv(Protocol):
    def __call__(
        self, tensors: tuple[Tensor, ...], scalar: Tensor
    ) -> tuple[Tensor, ...]: ...


class _ForeachMulInPlace(Protocol):
    def __call__(
        self, tensors: tuple[Tensor, ...], scalar: Tensor
    ) -> tuple[Tensor, ...]: ...


_foreach_norm = cast(
    _ForeachNorm,
    cast(object, getattr(torch, "_foreach_norm")),
)
_foreach_div = cast(
    _ForeachDiv,
    cast(object, getattr(torch, "_foreach_div")),
)
_foreach_mul_in_place = cast(
    _ForeachMulInPlace,
    cast(object, getattr(torch, "_foreach_mul_")),
)


@dataclass(frozen=True, slots=True)
class GradientGroupResult:
    """Device-resident norm and finiteness for one gradient group."""

    total_norm: Tensor
    gradients_are_finite: Tensor

    def __post_init__(self) -> None:
        assert self.total_norm.shape == ()
        assert self.gradients_are_finite.shape == ()
        assert self.gradients_are_finite.dtype == torch.bool


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


def clip_gradient_group(
    parameters: tuple[Tensor, ...], *, max_norm: float
) -> GradientGroupResult:
    """Measure, validate, and clip through fused foreach kernels."""
    assert parameters
    assert math.isfinite(max_norm)
    gradients = _gradients(parameters)
    device = parameters[0].device
    if not gradients:
        return GradientGroupResult(
            total_norm=torch.zeros(
                (), dtype=torch.float32, device=device
            ),
            gradients_are_finite=torch.ones(
                (), dtype=torch.bool, device=device
            ),
        )
    assert all(gradient.device == device for gradient in gradients)
    component_maxima = _foreach_norm(gradients, math.inf)
    max_abs = torch.stack(component_maxima).max()
    finite_max = torch.isfinite(max_abs)
    safe_scale = torch.where(
        finite_max & (max_abs > 0.0),
        max_abs,
        torch.ones_like(max_abs),
    )
    scaled_gradients = _foreach_div(gradients, safe_scale)
    scaled_component_norms = _foreach_norm(scaled_gradients, 2.0)
    scaled_norm = torch.linalg.vector_norm(
        torch.stack(scaled_component_norms), ord=2
    )
    gradients_are_finite = finite_max & torch.isfinite(scaled_norm)
    if max_norm > 0.0:
        finite_positive_norm = gradients_are_finite & (
            scaled_norm > 0.0
        )
        safe_scaled_norm = torch.where(
            finite_positive_norm,
            scaled_norm,
            torch.ones_like(scaled_norm),
        )
        clip_coefficient = (
            torch.tensor(
                max_norm,
                dtype=torch.float32,
                device=device,
            )
            / safe_scaled_norm
            / (safe_scale + 0.000001 / safe_scaled_norm)
        )
        bounded_coefficient = torch.where(
            finite_positive_norm & (max_abs > 0.0),
            torch.clamp(clip_coefficient, max=1.0),
            torch.ones_like(clip_coefficient),
        )
        with torch.no_grad():
            _ = _foreach_mul_in_place(gradients, bounded_coefficient)
    max_float32 = torch.tensor(
        torch.finfo(torch.float32).max,
        dtype=torch.float32,
        device=device,
    )
    finite_total_norm = torch.minimum(
        max_abs * scaled_norm, max_float32
    )
    reported_norm = torch.where(
        gradients_are_finite,
        finite_total_norm,
        torch.zeros_like(finite_total_norm),
    )
    return GradientGroupResult(
        total_norm=reported_norm,
        gradients_are_finite=gradients_are_finite,
    )


def _gradients(parameters: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(
        gradient
        for parameter in parameters
        if (gradient := parameter.grad) is not None
    )
