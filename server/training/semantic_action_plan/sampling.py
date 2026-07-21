"""Sampling directly from the fixed 110-choice action vocabulary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_action_plan.choices import (
    DeviceLegalChoiceBatch,
)
from server.training.semantic_actions.choices import PASS_CHOICE_ID


@dataclass(frozen=True, slots=True)
class SampledActionChoices:
    """Sampled choice ids and device-side diagnostics."""

    choice_ids: Tensor
    selected_log_probabilities: Tensor
    entropies: Tensor
    error_code: Tensor


def sample_legal_choices(
    *,
    choice_logits: Tensor,
    legal_choices: DeviceLegalChoiceBatch,
    thresholds: Tensor,
    active_rows: Tensor,
) -> SampledActionChoices:
    """Sample one legal fixed-vocabulary choice per active row."""
    assert choice_logits.ndim == 2
    assert legal_choices.masks.shape == choice_logits.shape
    batch_size = int(choice_logits.shape[0])
    assert thresholds.shape == (batch_size,)
    assert active_rows.shape == (batch_size,)
    assert active_rows.dtype == torch.bool
    error_code = torch.zeros(
        (), dtype=torch.long, device=choice_logits.device
    )
    checked = thresholds.to(device=choice_logits.device)
    error_code = _set_error_if(
        error_code,
        (~torch.isfinite(checked)).any(),
        _ERROR_NONFINITE_THRESHOLD,
    )
    error_code = _set_error_if(
        error_code,
        ((checked < 0.0) | (checked >= 1.0)).any(),
        _ERROR_THRESHOLD_RANGE,
    )
    has_legal = legal_choices.choice_counts > 0
    error_code = _set_error_if(
        error_code,
        (active_rows & ~has_legal).any(),
        _ERROR_EMPTY_LEGAL_MASK,
    )
    safe_masks = _safe_masks(
        masks=legal_choices.masks, has_legal=has_legal
    )
    valid_logits = choice_logits[
        legal_choices.masks & active_rows.unsqueeze(1)
    ]
    masked_logits = choice_logits.masked_fill(~safe_masks, -torch.inf)
    probabilities = torch.softmax(masked_logits, dim=1).masked_fill(
        ~legal_choices.masks, 0.0
    )
    log_probabilities = torch.log_softmax(
        masked_logits, dim=1
    ).masked_fill(~legal_choices.masks, 0.0)
    choice_ids = _sample_ids(
        probabilities=probabilities,
        masks=legal_choices.masks,
        thresholds=_thresholds_for_logits(
            thresholds=checked, logits=choice_logits
        ),
    )
    selected_log_probabilities = log_probabilities.gather(
        1, choice_ids.unsqueeze(1)
    ).squeeze(1)
    entropies = -(probabilities * log_probabilities).sum(dim=1)
    error_code = _set_error_if(
        error_code,
        (~torch.isfinite(valid_logits)).any(),
        _ERROR_NONFINITE_LOGITS,
    )
    error_code = _set_error_if(
        error_code,
        (
            (~torch.isfinite(probabilities[active_rows])).any()
            | (~torch.isfinite(log_probabilities[active_rows])).any()
            | (
                ~torch.isfinite(selected_log_probabilities[active_rows])
            ).any()
            | (~torch.isfinite(entropies[active_rows])).any()
        ),
        _ERROR_NONFINITE_DISTRIBUTION,
    )
    return SampledActionChoices(
        choice_ids=torch.where(
            has_legal,
            choice_ids,
            torch.full_like(choice_ids, PASS_CHOICE_ID),
        ),
        selected_log_probabilities=torch.where(
            has_legal,
            selected_log_probabilities,
            torch.zeros_like(selected_log_probabilities),
        ),
        entropies=torch.where(
            has_legal, entropies, torch.zeros_like(entropies)
        ),
        error_code=error_code,
    )


def action_sampling_error_reason(error_code: int) -> str | None:
    """Return a stable rejection reason for a sampling error code."""
    if error_code == 0:
        return None
    if error_code == _ERROR_EMPTY_LEGAL_MASK:
        return "policy action has no legal choice"
    if error_code == _ERROR_NONFINITE_THRESHOLD:
        return "policy sampling thresholds must be finite"
    if error_code == _ERROR_THRESHOLD_RANGE:
        return "policy sampling thresholds must be in [0, 1)"
    if error_code == _ERROR_NONFINITE_LOGITS:
        return "policy choice logits must be finite"
    if error_code == _ERROR_NONFINITE_DISTRIBUTION:
        return "policy choice distribution must be finite"
    return None


def _sample_ids(
    *, probabilities: Tensor, masks: Tensor, thresholds: Tensor
) -> Tensor:
    columns = torch.arange(
        int(probabilities.shape[1]),
        dtype=torch.long,
        device=probabilities.device,
    ).unsqueeze(0)
    cumulative = probabilities.cumsum(dim=1)
    crossed = (cumulative > thresholds.unsqueeze(1)) & masks
    sentinel = torch.full_like(
        columns.expand_as(masks), int(masks.shape[1])
    )
    first = torch.where(crossed, columns, sentinel).min(dim=1).values
    last = (
        torch.where(
            masks,
            columns.expand_as(masks),
            torch.full_like(masks, -1, dtype=torch.long),
        )
        .max(dim=1)
        .values.clamp(min=0)
    )
    return torch.where(first <= last, first, last)


def _safe_masks(*, masks: Tensor, has_legal: Tensor) -> Tensor:
    fallback = torch.zeros_like(masks)
    fallback[:, PASS_CHOICE_ID] = True
    return torch.where(has_legal.unsqueeze(1), masks, fallback)


def _thresholds_for_logits(
    *, thresholds: Tensor, logits: Tensor
) -> Tensor:
    result = thresholds.to(dtype=logits.dtype, device=logits.device)
    one = torch.ones((), dtype=result.dtype, device=result.device)
    zero = torch.zeros((), dtype=result.dtype, device=result.device)
    return torch.minimum(result, torch.nextafter(one, zero))


def _set_error_if(
    error_code: Tensor, condition: Tensor, code: int
) -> Tensor:
    assert error_code.shape == ()
    assert condition.shape == ()
    return torch.where(
        (error_code == 0) & condition,
        torch.full(
            (), code, dtype=torch.long, device=error_code.device
        ),
        error_code,
    )


_ERROR_EMPTY_LEGAL_MASK = 1
_ERROR_NONFINITE_THRESHOLD = 10
_ERROR_THRESHOLD_RANGE = 11
_ERROR_NONFINITE_LOGITS = 20
_ERROR_NONFINITE_DISTRIBUTION = 21


__all__ = (
    "SampledActionChoices",
    "action_sampling_error_reason",
    "sample_legal_choices",
)
