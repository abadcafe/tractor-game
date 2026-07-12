"""Device tensor finiteness validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.foundation import result as _result


@dataclass(frozen=True, slots=True)
class NamedTensorCheck:
    """One tensor finiteness check with its rejection reason."""

    tensor: Tensor
    reason: str

    def __post_init__(self) -> None:
        assert self.reason


@dataclass(frozen=True, slots=True)
class TensorRangeCheck:
    """One inclusive/exclusive tensor range check."""

    tensor: Tensor
    min_inclusive: float
    max_exclusive: float
    reason: str

    def __post_init__(self) -> None:
        assert self.min_inclusive < self.max_exclusive
        assert self.reason


def reject_if_non_finite(
    checks: tuple[NamedTensorCheck, ...],
) -> _result.Ok[None] | _result.Rejected:
    """Reject when any named tensor contains NaN or infinity."""
    assert checks
    return reject_if_invalid_tensors(
        finite_checks=checks,
        range_checks=(),
    )


def reject_if_invalid_tensors(
    *,
    finite_checks: tuple[NamedTensorCheck, ...],
    range_checks: tuple[TensorRangeCheck, ...],
) -> _result.Ok[None] | _result.Rejected:
    """Reject when any tensor fails finite or range checks."""
    assert finite_checks or range_checks
    if _all_checks_pass(finite_checks, range_checks):
        return _result.Ok(value=None)
    for check in finite_checks:
        if not _tensor_is_finite(check.tensor):
            return _result.Rejected(reason=check.reason)
    for check in range_checks:
        if not _tensor_in_range(check):
            return _result.Rejected(reason=check.reason)
    return _result.Rejected(reason="tensor values must be finite")


def _all_checks_pass(
    finite_checks: tuple[NamedTensorCheck, ...],
    range_checks: tuple[TensorRangeCheck, ...],
) -> bool:
    first = (
        finite_checks[0].tensor
        if finite_checks
        else range_checks[0].tensor
    )
    device = first.device
    assert all(check.tensor.device == device for check in finite_checks)
    assert all(check.tensor.device == device for check in range_checks)
    finite_flags: list[Tensor] = []
    for check in finite_checks:
        finite_flags.append(torch.isfinite(check.tensor).all())
    for check in range_checks:
        finite_flags.append(_range_flag(check))
    return _tensor_bool(torch.stack(finite_flags).all())


def _tensor_is_finite(tensor: Tensor) -> bool:
    return _tensor_bool(torch.isfinite(tensor).all())


def _tensor_in_range(check: TensorRangeCheck) -> bool:
    return _tensor_bool(_range_flag(check))


def _range_flag(check: TensorRangeCheck) -> Tensor:
    return (
        (check.tensor >= check.min_inclusive)
        & (check.tensor < check.max_exclusive)
    ).all()


def _tensor_bool(value: Tensor) -> bool:
    return bool(value.detach().cpu().item())
