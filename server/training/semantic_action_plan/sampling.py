"""Full-vocabulary masked sampling for compiled action plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from server.training.semantic_actions.codec import SEMANTIC_CODEC


@dataclass(frozen=True, slots=True)
class SampledActionTokens:
    """Sampled semantic token ids and device-side diagnostics."""

    token_ids: Tensor
    selected_log_probabilities: Tensor
    entropies: Tensor
    error_code: Tensor


def sample_legal_tokens(
    *,
    argument_logits: Tensor,
    legal_token_mask: Tensor,
    thresholds: Tensor,
) -> SampledActionTokens:
    """Sample one legal semantic token per row."""
    assert argument_logits.ndim == 2
    assert legal_token_mask.shape == argument_logits.shape
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
    has_legal_token = legal_token_mask.any(dim=1)
    error_code = _set_error_if(
        error_code,
        (~has_legal_token).any(),
        _ERROR_EMPTY_LEGAL_MASK,
    )
    valid_logits = argument_logits[legal_token_mask]
    error_code = _set_error_if(
        error_code,
        (~torch.isfinite(valid_logits)).any(),
        _ERROR_NONFINITE_LOGITS,
    )
    safe_legal_mask = _safe_legal_token_mask(
        legal_token_mask=legal_token_mask,
        has_legal_token=has_legal_token,
    )
    masked_logits = _safe_masked_logits(
        argument_logits=argument_logits,
        safe_legal_mask=safe_legal_mask,
        has_legal_token=has_legal_token,
    )
    probabilities = torch.softmax(masked_logits, dim=1).masked_fill(
        ~safe_legal_mask, 0.0
    )
    log_probabilities = torch.log_softmax(
        masked_logits, dim=1
    ).masked_fill(~safe_legal_mask, 0.0)
    sampling_thresholds = _sampling_thresholds_for_logits(
        thresholds=thresholds.to(
            dtype=torch.float64, device=argument_logits.device
        ),
        argument_logits=argument_logits,
    )
    token_ids = _sample_token_ids(
        probabilities=probabilities,
        legal_token_mask=safe_legal_mask,
        thresholds=sampling_thresholds,
    )
    selected_log_probabilities = log_probabilities.gather(
        dim=1, index=token_ids.unsqueeze(1)
    ).squeeze(1)
    entropies = -(probabilities * log_probabilities).sum(dim=1)
    error_code = _set_error_if(
        error_code,
        (
            (~torch.isfinite(probabilities)).any()
            | (~torch.isfinite(log_probabilities)).any()
            | (~torch.isfinite(selected_log_probabilities)).any()
            | (~torch.isfinite(entropies)).any()
        ),
        _ERROR_NONFINITE_DISTRIBUTION,
    )
    return SampledActionTokens(
        token_ids=token_ids,
        selected_log_probabilities=selected_log_probabilities,
        entropies=entropies,
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


def _safe_legal_token_mask(
    *, legal_token_mask: Tensor, has_legal_token: Tensor
) -> Tensor:
    safe_token_mask = legal_token_mask.clone()
    safe_token_mask[:, SEMANTIC_CODEC.argument_pass_id] = torch.where(
        has_legal_token,
        safe_token_mask[:, SEMANTIC_CODEC.argument_pass_id],
        torch.ones_like(has_legal_token),
    )
    return safe_token_mask


def _safe_masked_logits(
    *,
    argument_logits: Tensor,
    safe_legal_mask: Tensor,
    has_legal_token: Tensor,
) -> Tensor:
    masked_logits = argument_logits.masked_fill(
        ~safe_legal_mask, -torch.inf
    )
    fallback_logits = torch.full_like(argument_logits, -torch.inf)
    fallback_logits[:, SEMANTIC_CODEC.argument_pass_id] = 0.0
    return torch.where(
        has_legal_token.unsqueeze(1), masked_logits, fallback_logits
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


def _sample_token_ids(
    *,
    probabilities: Tensor,
    legal_token_mask: Tensor,
    thresholds: Tensor,
) -> Tensor:
    threshold_tensor = thresholds.to(
        dtype=probabilities.dtype, device=probabilities.device
    ).unsqueeze(1)
    cumulative = probabilities.cumsum(dim=1)
    crossed = (cumulative > threshold_tensor) & legal_token_mask
    has_crossed = crossed.any(dim=1)
    first_crossed = crossed.to(dtype=torch.long).argmax(dim=1)
    vocab_positions = torch.arange(
        int(legal_token_mask.shape[1]),
        dtype=torch.long,
        device=legal_token_mask.device,
    ).unsqueeze(0)
    last_valid = (
        torch.where(
            legal_token_mask,
            vocab_positions,
            torch.full_like(vocab_positions, -1),
        )
        .max(dim=1)
        .values
    )
    fallback = torch.where(has_crossed, first_crossed, last_valid)
    return fallback


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
