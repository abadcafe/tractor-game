"""Device-side PPO validation codes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PPO_VALIDATION_OK = 0
PPO_TRACE_EVALUATION_FAILED = 1
PPO_POLICY_LOSS_NONFINITE = 2
PPO_VALUE_LOSS_NONFINITE = 3
PPO_ENTROPY_NONFINITE = 4
PPO_TOTAL_LOSS_NONFINITE = 5
PPO_APPROX_KL_NONFINITE = 6
PPO_CLIP_FRACTION_NONFINITE = 7
PPO_GRADIENTS_NONFINITE = 8


@dataclass(frozen=True, slots=True)
class TensorValidationCheck:
    """One device tensor validation check."""

    tensor: Tensor
    code: int

    def __post_init__(self) -> None:
        assert self.code > PPO_VALIDATION_OK


def validation_ok(device: torch.device) -> Tensor:
    """Return a device scalar representing a valid state."""
    return torch.zeros((), dtype=torch.long, device=device)


def non_finite_validation_code(
    checks: tuple[TensorValidationCheck, ...],
) -> Tensor:
    """Return the first non-finite validation code."""
    assert checks
    code = validation_ok(checks[0].tensor.device)
    for check in checks:
        failed = ~torch.isfinite(check.tensor).all()
        code = torch.where(
            (code == PPO_VALIDATION_OK) & failed,
            torch.full(
                (),
                check.code,
                dtype=torch.long,
                device=code.device,
            ),
            code,
        )
    return code


def gradient_validation_code(
    parameters: tuple[Tensor, ...],
) -> Tensor:
    """Return a device scalar describing gradient finiteness."""
    assert parameters
    device = parameters[0].device
    code = validation_ok(device)
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        failed = ~torch.isfinite(gradient).all()
        code = torch.where(
            (code == PPO_VALIDATION_OK) & failed,
            torch.full(
                (),
                PPO_GRADIENTS_NONFINITE,
                dtype=torch.long,
                device=device,
            ),
            code,
        )
    return code


def combine_validation_codes(first: Tensor, second: Tensor) -> Tensor:
    """Return the first non-zero validation code."""
    assert first.shape == ()
    assert second.shape == ()
    return torch.where(first != PPO_VALIDATION_OK, first, second)


def validation_rejection_reason(code: Tensor) -> str | None:
    """Return the rejection reason for one validation code."""
    assert code.shape == ()
    value = int(code.detach().cpu().item())
    if value == PPO_VALIDATION_OK:
        return None
    if value == PPO_TRACE_EVALUATION_FAILED:
        return "PPO trace evaluation failed"
    if value == PPO_POLICY_LOSS_NONFINITE:
        return "PPO policy_loss must be finite"
    if value == PPO_VALUE_LOSS_NONFINITE:
        return "PPO value_loss must be finite"
    if value == PPO_ENTROPY_NONFINITE:
        return "PPO entropy must be finite"
    if value == PPO_TOTAL_LOSS_NONFINITE:
        return "PPO total_loss must be finite"
    if value == PPO_APPROX_KL_NONFINITE:
        return "PPO approx_kl must be finite"
    if value == PPO_CLIP_FRACTION_NONFINITE:
        return "PPO clip_fraction must be finite"
    if value == PPO_GRADIENTS_NONFINITE:
        return "PPO gradients must be finite"
    return "PPO validation failed"
