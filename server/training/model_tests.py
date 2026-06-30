"""Tests for torch policy model forward pass."""

from __future__ import annotations

import torch

from server.player.test_helpers import card, make_snapshot
from server.training.action_tokens import (
    ACTION_PLAY_TOKEN_ID,
    ACTION_TOKEN_VOCAB_SIZE,
    BEGIN_TOKEN_ID,
)
from server.training.model import UpgradePolicyModel
from server.training.observation import build_observation
from server.training.tensorize import (
    tensorize_action_prefix,
    tensorize_observation,
)


def test_upgrade_policy_model_forward_action_shapes() -> None:
    device = torch.device("cpu")
    model = UpgradePolicyModel(
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
    action_prefix_ids = tensorize_action_prefix(
        prefix=(BEGIN_TOKEN_ID, ACTION_PLAY_TOKEN_ID),
        device=device,
    )

    logits, values = model.forward_action(
        observation_batch, action_prefix_ids
    )

    assert logits.shape == (1, ACTION_TOKEN_VOCAB_SIZE)
    assert values.shape == (1,)
