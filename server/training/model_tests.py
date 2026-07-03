"""Tests for torch policy model forward pass."""

from __future__ import annotations

import torch

from server.player.test_helpers import card, make_snapshot
from server.training.model import TractorPolicyModel
from server.training.observation import build_observation
from server.training.semantic_actions import SemanticArgumentPrefix
from server.training.tensorize import (
    tensorize_argument_prefix,
    tensorize_observation,
)


def test_tractor_policy_model_forward_argument_shapes() -> None:
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
    prefix_batch = tensorize_argument_prefix(
        prefix=SemanticArgumentPrefix(arguments=()),
        device=device,
    )

    output = model.forward_argument(
        observation_batch,
        prefix_batch,
    )

    assert output.argument_logits.shape[0] == 1
    assert output.values.shape == (1,)
