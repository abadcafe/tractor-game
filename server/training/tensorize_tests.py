"""Tests for torch tensorization of training observations."""

from __future__ import annotations

import subprocess
import sys

import torch

from server.player.test_helpers import card, make_snapshot
from server.rules.card_faces import CardFace, FaceCount
from server.training.feature_schema import NUMERIC_FEATURE_COUNT
from server.training.observation import build_observation
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentPrefix,
)
from server.training.semantic_actions.codec import SEMANTIC_CODEC
from server.training.tensorize import (
    OBSERVATION_COMPONENT_COUNT,
    observation_component_tensors,
    stack_observation_batches,
    tensorize_argument_prefix,
    tensorize_argument_prefixes,
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


def test_stack_observation_batches_pads_cached_batches() -> None:
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
    short_batch = tensorize_observation(
        observation=short_observation,
        max_observation_tokens=128,
        device=torch.device("cpu"),
    )
    long_batch = tensorize_observation(
        observation=long_observation,
        max_observation_tokens=128,
        device=torch.device("cpu"),
    )

    batch = stack_observation_batches(
        batches=(short_batch, long_batch),
        device=torch.device("cpu"),
    )

    assert len(short_observation.tokens) < len(long_observation.tokens)
    assert int(short_batch.component_ids.shape[1]) == len(
        short_observation.tokens
    )
    assert int(long_batch.component_ids.shape[1]) == len(
        long_observation.tokens
    )
    assert batch.component_ids.shape == (
        2,
        len(long_observation.tokens),
        OBSERVATION_COMPONENT_COUNT,
    )
    assert (
        batch.component_ids[0, len(short_observation.tokens) :, :]
        .sum()
        .item()
        == 0
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
                "from server.player.test_helpers import make_snapshot\n"
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


def test_tensorize_argument_prefix_outputs_bos_and_arguments() -> None:
    test_card = card("clubs", "3", 1)
    prefix = SemanticArgumentPrefix(
        arguments=(
            SemanticArgument(
                "select_face_count",
                FaceCount(CardFace(test_card.suit, test_card.rank), 1),
            ),
        )
    )

    batch = tensorize_argument_prefix(
        prefix=prefix,
        device=torch.device("cpu"),
    )

    assert batch.argument_ids.shape[0] == 1
    assert batch.argument_ids.shape[1] == 2
    assert (
        batch.argument_ids[0, 0].item()
        == SEMANTIC_CODEC.argument_bos_id
    )
    assert (
        batch.argument_ids[0, 1].item() > SEMANTIC_CODEC.argument_bos_id
    )
    assert batch.argument_masks[0, 0].item()
    assert batch.argument_masks[0, 1].item()


def test_tensorize_argument_prefixes_use_batch_max_length() -> None:
    test_card = card("clubs", "3", 1)
    selected = SemanticArgument(
        "select_face_count",
        FaceCount(CardFace(test_card.suit, test_card.rank), 1),
    )

    batch = tensorize_argument_prefixes(
        prefixes=(
            SemanticArgumentPrefix(arguments=()),
            SemanticArgumentPrefix(arguments=(selected,)),
        ),
        device=torch.device("cpu"),
    )

    assert batch.argument_ids.shape == (2, 2)
    assert bool(batch.argument_masks[0, 0].item())
    assert not bool(batch.argument_masks[0, 1].item())
    assert bool(batch.argument_masks[1, 0].item())
    assert bool(batch.argument_masks[1, 1].item())
