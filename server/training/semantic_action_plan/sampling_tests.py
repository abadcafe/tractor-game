"""Black-box tests for compact semantic action token sampling."""

from __future__ import annotations

import torch
from torch import Tensor

from server.training.semantic_action_plan import (
    sample_legal_choices,
    sample_legal_token_error_reason,
)
from server.training.semantic_action_plan.choices import (
    DeviceLegalChoices,
    compact_legal_choices,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC


def test_sample_legal_choices_zero_threshold_selects_first_choice() -> (
    None
):
    token_id = SEMANTIC_CODEC.argument_stop_id

    sampled = sample_legal_choices(
        argument_logits=_zero_logits(row_count=1),
        legal_choices=_choices(((token_id,),)),
        thresholds=torch.tensor((0.0,), dtype=torch.float64),
    )

    assert _error_code(sampled.error_code) == 0
    assert _token_ids(sampled.token_ids) == (token_id,)
    assert _token_ids(sampled.selected_choice_offsets) == (0,)


def test_sample_legal_choices_returns_only_compact_choices() -> None:
    pass_id = SEMANTIC_CODEC.argument_pass_id
    stop_id = SEMANTIC_CODEC.argument_stop_id
    select_id = SEMANTIC_CODEC.argument_select_base_id + 4
    logits = _zero_logits(row_count=2)
    logits[:, 0] = 100.0

    sampled = sample_legal_choices(
        argument_logits=logits,
        legal_choices=_choices(((pass_id, stop_id), (select_id,))),
        thresholds=torch.tensor((0.25, 0.99), dtype=torch.float64),
    )

    assert _error_code(sampled.error_code) == 0
    assert _token_ids(sampled.token_ids)[0] in (pass_id, stop_id)
    assert _token_ids(sampled.token_ids)[1] == select_id


def test_sample_legal_choices_empty_choices_return_error() -> None:
    sampled = sample_legal_choices(
        argument_logits=_zero_logits(row_count=1),
        legal_choices=_choices(((),)),
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


def test_sample_legal_choices_empty_row_keeps_valid_row() -> None:
    sampled = sample_legal_choices(
        argument_logits=_zero_logits(row_count=2),
        legal_choices=_choices(
            ((), (SEMANTIC_CODEC.argument_stop_id,))
        ),
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


def test_sample_legal_choices_cdf_boundary_is_half_open() -> None:
    pass_id = SEMANTIC_CODEC.argument_pass_id
    stop_id = SEMANTIC_CODEC.argument_stop_id

    sampled = sample_legal_choices(
        argument_logits=_zero_logits(row_count=1),
        legal_choices=_choices(((pass_id, stop_id),)),
        thresholds=torch.tensor((0.5,), dtype=torch.float64),
    )

    assert _error_code(sampled.error_code) == 0
    assert _token_ids(sampled.token_ids) == (stop_id,)
    assert _token_ids(sampled.selected_choice_offsets) == (1,)


def _zero_logits(*, row_count: int) -> Tensor:
    return torch.zeros(
        (row_count, SEMANTIC_CODEC.argument_vocab_size),
        dtype=torch.float32,
    )


def _choices(
    token_rows: tuple[tuple[int, ...], ...],
) -> DeviceLegalChoices:
    max_width = max((len(row) for row in token_rows), default=0)
    safe_width = max(max_width, 1)
    token_ids = torch.zeros(
        (len(token_rows), safe_width), dtype=torch.long
    )
    token_mask = torch.zeros(
        (len(token_rows), safe_width), dtype=torch.bool
    )
    for row_index, row in enumerate(token_rows):
        for column_index, token_id in enumerate(row):
            token_ids[row_index, column_index] = token_id
            token_mask[row_index, column_index] = True
    return compact_legal_choices(
        candidate_token_ids=token_ids, candidate_mask=token_mask
    )


def _token_ids(tokens: Tensor) -> tuple[int, ...]:
    cpu_tokens = tokens.detach().cpu()
    return tuple(
        int(cpu_tokens[index].item())
        for index in range(int(cpu_tokens.shape[0]))
    )


def _error_code(error_code: Tensor) -> int:
    return int(error_code.detach().cpu().item())


def _all_finite(values: Tensor) -> bool:
    return bool(torch.isfinite(values).all().detach().cpu().item())
