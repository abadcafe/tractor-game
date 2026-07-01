"""Tests for torch tensorization of training observations."""

from __future__ import annotations

import torch

from server.player.test_helpers import card, make_snapshot
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.observation import build_observation
from server.training.selection_actions import SelectionState
from server.training.tensorize import (
    tensorize_observation,
    tensorize_selection_state,
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


def test_tensorize_observation_tracks_self_hand_slots() -> None:
    first = card("spades", "A", 1)
    second = card("hearts", "5", 1)
    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(player_hand=[first, second]),
        history=(),
    )

    batch = tensorize_observation(
        observation=observation,
        max_observation_tokens=128,
        device=torch.device("cpu"),
    )

    assert batch.hand_token_indices.shape == (1, 33)
    assert batch.hand_card_masks.shape == (1, 33)
    assert batch.hand_card_masks[0, 0].item()
    assert batch.hand_card_masks[0, 1].item()
    assert not batch.hand_card_masks[0, 2].item()


def test_tensorize_selection_state_outputs_masks_and_features() -> None:
    first = card("spades", "A", 1)
    second = card("hearts", "5", 1)
    observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action="play",
            player_hand=[first, second],
        ),
        history=(),
    )

    batch = tensorize_selection_state(
        query=observation.action_query,
        state=SelectionState(selected_slots=(1,)),
        device=torch.device("cpu"),
    )

    assert batch.selected_slot_masks.shape == (1, 33)
    assert batch.feature_values.shape == (1, 6)
    assert batch.selected_slot_masks[0, 1].item() == 1.0
    assert batch.selected_slot_masks[0, 0].item() == 0.0
