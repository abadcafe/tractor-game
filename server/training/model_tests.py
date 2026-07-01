"""Tests for torch policy model forward pass."""

from __future__ import annotations

import torch

from server.player.test_helpers import card, make_snapshot
from server.training.model import TractorPolicyModel
from server.training.observation import build_observation
from server.training.selection_actions import SelectionState
from server.training.tensorize import (
    tensorize_observation,
    tensorize_selection_state,
)


def test_tractor_policy_model_forward_head_shapes() -> None:
    device = torch.device("cpu")
    model = TractorPolicyModel(
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
    )
    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action="play",
            player_hand=[card("spades", "A", 1)],
        ),
        history=(),
    )
    observation_batch = tensorize_observation(
        observation=observation,
        max_observation_tokens=64,
        device=device,
    )
    selection_batch = tensorize_selection_state(
        query=observation.action_query,
        state=SelectionState(selected_slots=()),
        device=device,
    )

    output = model.forward_lead_play(
        observation_batch,
        selection_batch,
    )

    assert output.card_logits.shape == (1, 33)
    assert output.pass_logits is None
    assert output.stop_logits is not None
    assert output.stop_logits.shape == (1,)
    assert output.values.shape == (1,)
