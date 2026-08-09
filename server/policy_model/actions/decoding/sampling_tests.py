"""Tests for numerically stable fixed-vocabulary sampling."""

from __future__ import annotations

import torch

from server.policy_model._schema.actions import ACTION_CHOICE_COUNT
from server.policy_model.actions.decoding.choices import (
    legal_choice_batch,
)
from server.policy_model.actions.decoding.sampling import (
    sample_legal_choices,
)


def test_sample_legal_choices_computes_distribution_in_float32() -> (
    None
):
    masks = torch.zeros((1, ACTION_CHOICE_COUNT), dtype=torch.bool)
    masks[0, :2] = True

    sampled = sample_legal_choices(
        choice_logits=torch.zeros(
            (1, ACTION_CHOICE_COUNT), dtype=torch.bfloat16
        ),
        legal_choices=legal_choice_batch(masks=masks),
        thresholds=torch.tensor((0.25,), dtype=torch.float32),
        active_rows=torch.ones((1,), dtype=torch.bool),
    )

    assert sampled.selected_log_probabilities.dtype == torch.float32
    assert sampled.entropies.dtype == torch.float32
    assert int(sampled.choice_ids[0]) == 0
