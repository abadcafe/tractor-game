"""Tests for torch tensorization of training observations."""

from __future__ import annotations

import torch

from server.player.test_helpers import make_snapshot
from server.training.action_tokens import (
    BEGIN_TOKEN_ID,
    MAX_ACTION_TOKENS,
)
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.observation import build_observation
from server.training.tensorize import (
    tensorize_action_prefix,
    tensorize_observation,
)
from server.training.tokens import GlobalFieldToken


def test_tensorize_observation_outputs_numeric_tensors() -> None:
    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(defender_points=0),
        history=(),
    )

    batch = tensorize_observation(
        observation=observation,
        max_observation_tokens=128,
        device=torch.device("cpu"),
    )

    assert batch.numeric_values.shape == (
        1,
        128,
        NUMERIC_FEATURE_COUNT,
    )
    assert batch.numeric_masks.shape == (
        1,
        128,
        NUMERIC_FEATURE_COUNT,
    )
    assert batch.numeric_values.dtype == torch.float32
    assert batch.numeric_masks.dtype == torch.float32
    assert batch.numeric_masks.sum().item() > 0.0
    assert batch.numeric_values[0, -1].sum().item() == 0.0
    assert batch.numeric_masks[0, -1].sum().item() == 0.0


def test_tensorize_observation_rejects_oversized_observation() -> None:
    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(),
        history=(),
    )
    oversized = type(observation)(
        player_index=observation.player_index,
        tokens=(
            GlobalFieldToken("rules_version", "base-A"),
            GlobalFieldToken("final_target", "WIN"),
        ),
        hand_card_ids=observation.hand_card_ids,
        action_query=observation.action_query,
    )

    try:
        tensorize_observation(
            observation=oversized,
            max_observation_tokens=1,
            device=torch.device("cpu"),
        )
    except AssertionError:
        return

    assert False


def test_tensorize_action_prefix_rejects_oversized_prefix() -> None:
    oversized_prefix = tuple(
        BEGIN_TOKEN_ID for _ in range(MAX_ACTION_TOKENS + 1)
    )

    try:
        tensorize_action_prefix(
            prefix=oversized_prefix,
            device=torch.device("cpu"),
        )
    except AssertionError:
        return

    assert False
