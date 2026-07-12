"""Tests for torch tensorization of training observations."""

from __future__ import annotations

import subprocess
import sys

import torch

from server.game.players.test_helpers import card, make_snapshot
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.observation import build_observation
from server.training.tensorize import (
    OBSERVATION_COMPONENT_COUNT,
    observation_component_tensors,
    tensorize_observation,
    tensorize_observations,
)


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
        len(observation.tokens),
        NUMERIC_FEATURE_COUNT,
    )
    assert batch.numeric_masks.shape == (
        1,
        len(observation.tokens),
        NUMERIC_FEATURE_COUNT,
    )
    assert batch.component_ids.shape == (
        1,
        len(observation.tokens),
        OBSERVATION_COMPONENT_COUNT,
    )
    assert batch.component_ids.dtype == torch.long
    assert batch.numeric_values.dtype == torch.float32
    assert batch.numeric_masks.dtype == torch.float32
    assert batch.numeric_masks.sum().item() > 0.0


def test_tensorize_observations_use_batch_max_length() -> None:
    short_observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(player_hand=[card("spades", "A", 1)]),
        history=(),
    )
    long_observation = build_observation(
        player_index=0,
        snapshot=make_snapshot(
            player_hand=[
                card("spades", "A", 1),
                card("hearts", "K", 2),
                card("clubs", "3", 1),
            ]
        ),
        history=(),
    )

    batch = tensorize_observations(
        observations=(short_observation, long_observation),
        max_observation_tokens=128,
        device=torch.device("cpu"),
    )

    assert len(short_observation.tokens) < len(long_observation.tokens)
    expected_width = max(
        len(short_observation.tokens), len(long_observation.tokens)
    )
    assert batch.component_ids.shape == (
        2,
        expected_width,
        OBSERVATION_COMPONENT_COUNT,
    )
    assert batch.numeric_values.shape == (
        2,
        expected_width,
        NUMERIC_FEATURE_COUNT,
    )
    assert (
        batch.component_ids[0, len(short_observation.tokens) :, :]
        .sum()
        .item()
        == 0
    )
    assert (
        batch.numeric_masks[0, len(short_observation.tokens) :, :]
        .sum()
        .item()
        == 0.0
    )


def test_tensorize_observation_crashes_on_oversized_observation() -> (
    None
):
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch\n"
                "from server.game.players.test_helpers import "
                "make_snapshot\n"
                "from server.training.observation import "
                "build_observation\n"
                "from server.training.tensorize import "
                "tensorize_observation\n"
                "observation = build_observation(\n"
                "    player_index=0,\n"
                "    snapshot=make_snapshot(),\n"
                "    history=(),\n"
                ")\n"
                "tensorize_observation(\n"
                "    observation=observation,\n"
                "    max_observation_tokens=1,\n"
                "    device=torch.device('cpu'),\n"
                ")\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "AssertionError" in completed.stderr


def test_tensorize_observation_exposes_face_count_component() -> None:
    first = card("spades", "A", 1)
    second = card("spades", "A", 2)
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

    components = observation_component_tensors(batch)

    assert components.count_ids.shape == (1, len(observation.tokens))
    assert components.count_ids.max().item() > 0
