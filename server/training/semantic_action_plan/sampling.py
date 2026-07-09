"""Fixed-width legal-candidate sampling for compiled action plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_action_plan.choices import (
    DeviceLegalCandidateBatch,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC


@dataclass(frozen=True, slots=True)
class SampledActionTokens:
    """Sampled semantic token ids and device-side diagnostics."""

    token_ids: Tensor
    selected_choice_offsets: Tensor
    selected_log_probabilities: Tensor
    entropies: Tensor
    error_code: Tensor


def sample_legal_candidates(
    *,
    argument_logits: Tensor,
    legal_candidates: DeviceLegalCandidateBatch,
    thresholds: Tensor,
    active_rows: Tensor,
) -> SampledActionTokens:
    """Sample one legal semantic token per row."""
    assert argument_logits.ndim == 2
    assert legal_candidates.batch_size() == int(
        argument_logits.shape[0]
    )
    assert thresholds.ndim == 1
    assert int(thresholds.shape[0]) == int(argument_logits.shape[0])
    assert active_rows.ndim == 1
    assert int(active_rows.shape[0]) == int(argument_logits.shape[0])
    assert active_rows.dtype == torch.bool
    error_code = torch.zeros(
        (), dtype=torch.long, device=argument_logits.device
    )
    high_precision_thresholds = thresholds.to(
        dtype=torch.float64, device=argument_logits.device
    )
    error_code = _set_error_if(
        error_code,
        (~torch.isfinite(high_precision_thresholds)).any(),
        _ERROR_NONFINITE_THRESHOLD,
    )
    error_code = _set_error_if(
        error_code,
        (
            (high_precision_thresholds < 0.0)
            | (high_precision_thresholds >= 1.0)
        ).any(),
        _ERROR_THRESHOLD_RANGE,
    )
    has_legal_token = legal_candidates.choice_counts > 0
    error_code = _set_error_if(
        error_code,
        (active_rows & ~has_legal_token).any(),
        _ERROR_EMPTY_LEGAL_MASK,
    )
    selected = _sample_candidate_rows(
        argument_logits=argument_logits,
        legal_candidates=legal_candidates,
        thresholds=_sampling_thresholds_for_logits(
            thresholds=high_precision_thresholds,
            argument_logits=argument_logits,
        ),
        has_legal_token=has_legal_token,
        active_rows=active_rows,
    )
    error_code = _set_error_if(
        error_code,
        selected.nonfinite_logits,
        _ERROR_NONFINITE_LOGITS,
    )
    error_code = _set_error_if(
        error_code,
        selected.nonfinite_distribution,
        _ERROR_NONFINITE_DISTRIBUTION,
    )
    return SampledActionTokens(
        token_ids=selected.token_ids,
        selected_choice_offsets=selected.choice_offsets,
        selected_log_probabilities=selected.log_probabilities,
        entropies=selected.entropies,
        error_code=error_code,
    )


def sample_legal_token_error_reason(error_code: int) -> str | None:
    """Return the policy rejection reason for a sampling error code."""
    if error_code == 0:
        return None
    if error_code == _ERROR_EMPTY_LEGAL_MASK:
        return "policy action has no legal semantic token"
    if error_code == _ERROR_NONFINITE_THRESHOLD:
        return "policy sampling thresholds must be finite"
    if error_code == _ERROR_THRESHOLD_RANGE:
        return "policy sampling thresholds must be in [0, 1)"
    if error_code == _ERROR_NONFINITE_LOGITS:
        return "policy argument logits must be finite"
    if error_code == _ERROR_NONFINITE_DISTRIBUTION:
        return "policy argument distribution must be finite"
    return None


@dataclass(frozen=True, slots=True)
class _ChoiceSelection:
    token_ids: Tensor
    choice_offsets: Tensor
    log_probabilities: Tensor
    entropies: Tensor
    nonfinite_logits: Tensor
    nonfinite_distribution: Tensor


def _sample_candidate_rows(
    *,
    argument_logits: Tensor,
    legal_candidates: DeviceLegalCandidateBatch,
    thresholds: Tensor,
    has_legal_token: Tensor,
    active_rows: Tensor,
) -> _ChoiceSelection:
    token_ids = legal_candidates.token_ids
    masks = legal_candidates.masks
    safe_masks = _safe_candidate_masks(
        masks=masks, has_legal_token=has_legal_token
    )
    legal_logits = argument_logits.gather(dim=1, index=token_ids)
    active_masks = masks & active_rows.unsqueeze(1)
    valid_logits = legal_logits[active_masks]
    safe_logits = legal_logits.masked_fill(~safe_masks, -torch.inf)
    probabilities = torch.softmax(safe_logits, dim=1).masked_fill(
        ~masks, 0.0
    )
    log_probabilities = torch.log_softmax(
        safe_logits, dim=1
    ).masked_fill(~masks, 0.0)
    selected_offsets = _selected_choice_offsets(
        probabilities=probabilities,
        masks=masks,
        thresholds=thresholds,
    )
    selected_token_ids = token_ids.gather(
        dim=1, index=selected_offsets.unsqueeze(1)
    ).squeeze(1)
    selected_log_probabilities = log_probabilities.gather(
        dim=1, index=selected_offsets.unsqueeze(1)
    ).squeeze(1)
    entropies = -(probabilities * log_probabilities).sum(dim=1)
    empty_replacement = torch.full_like(
        selected_token_ids, SEMANTIC_CODEC.argument_pass_id
    )
    return _ChoiceSelection(
        token_ids=torch.where(
            has_legal_token, selected_token_ids, empty_replacement
        ),
        choice_offsets=torch.where(
            has_legal_token,
            selected_offsets,
            torch.zeros_like(selected_offsets),
        ),
        log_probabilities=torch.where(
            has_legal_token,
            selected_log_probabilities,
            torch.zeros_like(selected_log_probabilities),
        ),
        entropies=torch.where(
            has_legal_token, entropies, torch.zeros_like(entropies)
        ),
        nonfinite_logits=(~torch.isfinite(valid_logits)).any(),
        nonfinite_distribution=(
            (~torch.isfinite(probabilities[active_rows])).any()
            | (~torch.isfinite(log_probabilities[active_rows])).any()
            | (
                ~torch.isfinite(selected_log_probabilities[active_rows])
            ).any()
            | (~torch.isfinite(entropies[active_rows])).any()
        ),
    )


def _selected_choice_offsets(
    *,
    probabilities: Tensor,
    masks: Tensor,
    thresholds: Tensor,
) -> Tensor:
    columns = torch.arange(
        int(probabilities.shape[1]),
        dtype=torch.long,
        device=probabilities.device,
    ).unsqueeze(0)
    cumulative = probabilities.cumsum(dim=1)
    crossed = (cumulative > thresholds.unsqueeze(1)) & masks
    sentinel = torch.full(
        masks.shape,
        int(probabilities.shape[1]),
        dtype=torch.long,
        device=probabilities.device,
    )
    first_offsets = (
        torch.where(crossed, columns, sentinel).min(dim=1).values
    )
    last_offsets = (
        torch.where(
            masks,
            columns.expand_as(masks),
            torch.full_like(masks, -1, dtype=torch.long),
        )
        .max(dim=1)
        .values
    )
    safe_last_offsets = last_offsets.clamp(min=0)
    return torch.where(
        first_offsets <= safe_last_offsets,
        first_offsets,
        safe_last_offsets,
    )


def _safe_candidate_masks(
    *, masks: Tensor, has_legal_token: Tensor
) -> Tensor:
    if int(masks.shape[1]) == 0:
        return torch.ones_like(masks)
    first_column = torch.zeros_like(masks)
    first_column[:, 0] = True
    return torch.where(
        has_legal_token.unsqueeze(1),
        masks,
        first_column,
    )


def _sampling_thresholds_for_logits(
    *, thresholds: Tensor, argument_logits: Tensor
) -> Tensor:
    cast_thresholds = thresholds.to(
        dtype=argument_logits.dtype, device=argument_logits.device
    )
    one = torch.ones(
        (), dtype=cast_thresholds.dtype, device=cast_thresholds.device
    )
    zero = torch.zeros(
        (), dtype=cast_thresholds.dtype, device=cast_thresholds.device
    )
    upper_bound = torch.nextafter(one, zero)
    return torch.minimum(cast_thresholds, upper_bound)


_ERROR_EMPTY_LEGAL_MASK = 1
_ERROR_NONFINITE_THRESHOLD = 10
_ERROR_THRESHOLD_RANGE = 11
_ERROR_NONFINITE_LOGITS = 12
_ERROR_NONFINITE_DISTRIBUTION = 13


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
