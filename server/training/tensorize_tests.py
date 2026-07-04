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
    tensorize_argument_prefix,
    tensorize_observation,
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

    assert batch.count_ids.shape == (1, 128)
    assert batch.count_ids.max().item() > 0


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
    assert (
        batch.argument_ids[0, 0].item()
        == SEMANTIC_CODEC.argument_bos_id
    )
    assert (
        batch.argument_ids[0, 1].item() > SEMANTIC_CODEC.argument_bos_id
    )
    assert batch.argument_masks[0, 0].item()
    assert batch.argument_masks[0, 1].item()
