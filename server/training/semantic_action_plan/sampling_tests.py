"""Black-box tests for semantic action token sampling."""

from __future__ import annotations

import torch
from torch import Tensor

from server.training.semantic_action_plan import (
    sample_legal_token_error_reason,
    sample_legal_tokens,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC


def test_sample_legal_tokens_zero_threshold_masks_illegal_prefix() -> (
    None
):
    token_id = SEMANTIC_CODEC.argument_stop_id
    logits = _zero_logits(row_count=1)
    legal_mask = _empty_mask(row_count=1)
    legal_mask[0, token_id] = True

    sampled = sample_legal_tokens(
        argument_logits=logits,
        legal_token_mask=legal_mask,
        thresholds=torch.tensor((0.0,), dtype=torch.float64),
    )

    assert _error_code(sampled.error_code) == 0
    assert _token_ids(sampled.token_ids) == (token_id,)


def test_sample_legal_tokens_returns_only_masked_token() -> None:
    logits = _zero_logits(row_count=2)
    logits[:, 0] = 100.0
    legal_mask = _empty_mask(row_count=2)
    legal_mask[0, SEMANTIC_CODEC.argument_pass_id] = True
    legal_mask[0, SEMANTIC_CODEC.argument_stop_id] = True
    legal_mask[1, SEMANTIC_CODEC.argument_select_base_id + 4] = True

    sampled = sample_legal_tokens(
        argument_logits=logits,
        legal_token_mask=legal_mask,
        thresholds=torch.tensor((0.25, 0.99), dtype=torch.float64),
    )

    assert _error_code(sampled.error_code) == 0
    assert _selected_tokens_are_legal(
        legal_mask, sampled.token_ids
    ) == (
        True,
        True,
    )


def test_sample_legal_tokens_empty_mask_returns_error() -> None:
    sampled = sample_legal_tokens(
        argument_logits=_zero_logits(row_count=1),
        legal_token_mask=_empty_mask(row_count=1),
        thresholds=torch.tensor((0.5,), dtype=torch.float64),
    )

    error_code = _error_code(sampled.error_code)
    assert sample_legal_token_error_reason(error_code) == (
        "policy action has no legal semantic token"
    )
    assert _token_ids(sampled.token_ids) == (
        SEMANTIC_CODEC.argument_pass_id,
    )
    assert _all_finite(sampled.selected_log_probabilities)
    assert _all_finite(sampled.entropies)


def test_sample_legal_tokens_empty_row_does_not_corrupt_valid_row() -> (
    None
):
    legal_mask = _empty_mask(row_count=2)
    legal_mask[1, SEMANTIC_CODEC.argument_stop_id] = True

    sampled = sample_legal_tokens(
        argument_logits=_zero_logits(row_count=2),
        legal_token_mask=legal_mask,
        thresholds=torch.tensor((0.5, 0.0), dtype=torch.float64),
    )

    assert sample_legal_token_error_reason(
        _error_code(sampled.error_code)
    )
    assert _token_ids(sampled.token_ids) == (
        SEMANTIC_CODEC.argument_pass_id,
        SEMANTIC_CODEC.argument_stop_id,
    )
    assert _all_finite(sampled.selected_log_probabilities)
    assert _all_finite(sampled.entropies)


def test_sample_legal_tokens_cdf_boundary_is_half_open() -> None:
    legal_mask = _empty_mask(row_count=1)
    legal_mask[0, SEMANTIC_CODEC.argument_pass_id] = True
    legal_mask[0, SEMANTIC_CODEC.argument_stop_id] = True

    sampled = sample_legal_tokens(
        argument_logits=_zero_logits(row_count=1),
        legal_token_mask=legal_mask,
        thresholds=torch.tensor((0.5,), dtype=torch.float64),
    )

    assert _error_code(sampled.error_code) == 0
    assert _token_ids(sampled.token_ids) == (
        SEMANTIC_CODEC.argument_stop_id,
    )


def _zero_logits(*, row_count: int) -> Tensor:
    return torch.zeros(
        (row_count, SEMANTIC_CODEC.argument_vocab_size),
        dtype=torch.float32,
    )


def _empty_mask(*, row_count: int) -> Tensor:
    return torch.zeros(
        (row_count, SEMANTIC_CODEC.argument_vocab_size),
        dtype=torch.bool,
    )


def _token_ids(tokens: Tensor) -> tuple[int, ...]:
    cpu_tokens = tokens.detach().cpu()
    return tuple(
        int(cpu_tokens[index].item())
        for index in range(int(cpu_tokens.shape[0]))
    )


def _error_code(error_code: Tensor) -> int:
    return int(error_code.detach().cpu().item())


def _selected_tokens_are_legal(
    legal_mask: Tensor, token_ids: Tensor
) -> tuple[bool, ...]:
    cpu_mask = legal_mask.detach().cpu()
    cpu_tokens = token_ids.detach().cpu()
    return tuple(
        bool(
            cpu_mask[
                row_index, int(cpu_tokens[row_index].item())
            ].item()
        )
        for row_index in range(int(cpu_tokens.shape[0]))
    )


def _all_finite(values: Tensor) -> bool:
    return bool(torch.isfinite(values).all().detach().cpu().item())
