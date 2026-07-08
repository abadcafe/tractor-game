"""Compact legal-choice sampling for compiled action plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_action_plan.choices import (
    DeviceLegalChoices,
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


def sample_legal_choices(
    *,
    argument_logits: Tensor,
    legal_choices: DeviceLegalChoices,
    thresholds: Tensor,
) -> SampledActionTokens:
    """Sample one legal semantic token per row."""
    assert argument_logits.ndim == 2
    assert legal_choices.batch_size() == int(argument_logits.shape[0])
    assert thresholds.ndim == 1
    assert int(thresholds.shape[0]) == int(argument_logits.shape[0])
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
    has_legal_token = legal_choices.choice_counts > 0
    error_code = _set_error_if(
        error_code,
        (~has_legal_token).any(),
        _ERROR_EMPTY_LEGAL_MASK,
    )
    if int(legal_choices.token_ids.shape[0]) == 0:
        return _empty_choice_sample(
            argument_logits=argument_logits, error_code=error_code
        )
    selected = _sample_non_empty_choice_rows(
        argument_logits=argument_logits,
        legal_choices=legal_choices,
        thresholds=_sampling_thresholds_for_logits(
            thresholds=high_precision_thresholds,
            argument_logits=argument_logits,
        ),
        has_legal_token=has_legal_token,
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


def _sample_non_empty_choice_rows(
    *,
    argument_logits: Tensor,
    legal_choices: DeviceLegalChoices,
    thresholds: Tensor,
    has_legal_token: Tensor,
) -> _ChoiceSelection:
    row_indices = legal_choices.row_indices
    token_ids = legal_choices.token_ids
    choice_logits = argument_logits[row_indices, token_ids]
    row_max = _scatter_row_max(
        values=choice_logits,
        row_indices=row_indices,
        row_count=legal_choices.batch_size(),
    )
    safe_row_max = torch.where(
        has_legal_token, row_max, torch.zeros_like(row_max)
    )
    shifted = choice_logits - safe_row_max.index_select(0, row_indices)
    exp_logits = torch.exp(shifted)
    row_sums = _scatter_row_sum(
        values=exp_logits,
        row_indices=row_indices,
        row_count=legal_choices.batch_size(),
    )
    log_row_sums = torch.log(row_sums.index_select(0, row_indices))
    log_probabilities = shifted - log_row_sums
    probabilities = torch.exp(log_probabilities)
    selected_offsets = _selected_choice_offsets(
        probabilities=probabilities,
        legal_choices=legal_choices,
        thresholds=thresholds,
    )
    selected_flat_indices = (
        legal_choices.choice_offsets[:-1] + selected_offsets
    )
    safe_selected_flat_indices = selected_flat_indices.clamp(
        min=0, max=int(token_ids.shape[0]) - 1
    )
    selected_token_ids = token_ids.index_select(
        dim=0, index=safe_selected_flat_indices
    )
    selected_log_probabilities = log_probabilities.index_select(
        dim=0, index=safe_selected_flat_indices
    )
    entropy_terms = -(probabilities * log_probabilities)
    entropies = _scatter_row_sum(
        values=entropy_terms,
        row_indices=row_indices,
        row_count=legal_choices.batch_size(),
    )
    return _ChoiceSelection(
        token_ids=torch.where(
            has_legal_token,
            selected_token_ids,
            torch.full_like(
                selected_token_ids, SEMANTIC_CODEC.argument_pass_id
            ),
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
        nonfinite_logits=(~torch.isfinite(choice_logits)).any(),
        nonfinite_distribution=(
            (~torch.isfinite(probabilities)).any()
            | (~torch.isfinite(log_probabilities)).any()
            | (~torch.isfinite(selected_log_probabilities)).any()
            | (~torch.isfinite(entropies)).any()
        ),
    )


def _selected_choice_offsets(
    *,
    probabilities: Tensor,
    legal_choices: DeviceLegalChoices,
    thresholds: Tensor,
) -> Tensor:
    row_indices = legal_choices.row_indices
    flat_positions = torch.arange(
        int(probabilities.shape[0]),
        dtype=torch.long,
        device=probabilities.device,
    )
    segment_starts = legal_choices.choice_offsets.index_select(
        dim=0, index=row_indices
    )
    local_offsets = flat_positions - segment_starts
    cumulative = probabilities.cumsum(dim=0)
    previous_indices = (segment_starts - 1).clamp(min=0)
    previous_cumulative = cumulative.index_select(
        dim=0, index=previous_indices
    )
    previous_cumulative = torch.where(
        segment_starts > 0,
        previous_cumulative,
        torch.zeros_like(previous_cumulative),
    )
    segmented_cumulative = cumulative - previous_cumulative
    crossed = segmented_cumulative > thresholds.index_select(
        dim=0, index=row_indices
    )
    sentinel = torch.full_like(
        local_offsets, int(probabilities.shape[0])
    )
    crossed_offsets = torch.where(crossed, local_offsets, sentinel)
    first_offsets = torch.full(
        legal_choices.choice_counts.shape,
        int(probabilities.shape[0]),
        dtype=torch.long,
        device=probabilities.device,
    ).scatter_reduce(
        dim=0,
        index=row_indices,
        src=crossed_offsets,
        reduce="amin",
        include_self=True,
    )
    fallback_offsets = (legal_choices.choice_counts - 1).clamp(min=0)
    return torch.where(
        first_offsets <= fallback_offsets,
        first_offsets,
        fallback_offsets,
    )


def _scatter_row_max(
    *, values: Tensor, row_indices: Tensor, row_count: int
) -> Tensor:
    return torch.full(
        (row_count,),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    ).scatter_reduce(
        dim=0,
        index=row_indices,
        src=values,
        reduce="amax",
        include_self=True,
    )


def _scatter_row_sum(
    *, values: Tensor, row_indices: Tensor, row_count: int
) -> Tensor:
    return torch.zeros(
        (row_count,), dtype=values.dtype, device=values.device
    ).scatter_add(dim=0, index=row_indices, src=values)


def _empty_choice_sample(
    *, argument_logits: Tensor, error_code: Tensor
) -> SampledActionTokens:
    batch_size = int(argument_logits.shape[0])
    return SampledActionTokens(
        token_ids=torch.full(
            (batch_size,),
            SEMANTIC_CODEC.argument_pass_id,
            dtype=torch.long,
            device=argument_logits.device,
        ),
        selected_choice_offsets=torch.zeros(
            (batch_size,),
            dtype=torch.long,
            device=argument_logits.device,
        ),
        selected_log_probabilities=torch.zeros(
            (batch_size,),
            dtype=argument_logits.dtype,
            device=argument_logits.device,
        ),
        entropies=torch.zeros(
            (batch_size,),
            dtype=argument_logits.dtype,
            device=argument_logits.device,
        ),
        error_code=error_code,
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
