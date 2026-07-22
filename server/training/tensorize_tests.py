"""Black-box tests for lossless typed observation tensorization."""

from __future__ import annotations

import torch

from server.game.players.test_helpers import card, make_snapshot
from server.game.rules.cards import Card
from server.training.observation import Observation, build_observation
from server.training.observation_memory import ObservationMemoryView
from server.training.observation_structure import StructureAxis
from server.training.packed_observation import (
    MAX_LOSSLESS_OBSERVATION_TOKENS,
)
from server.training.semantic_actions.choices import CARD_CHOICE_COUNT
from server.training.tensorize import (
    tensorize_observation,
    tensorize_observations,
)
from server.training.tokenization.encoding_schema import CATEGORY_COUNT


def test_tensorize_observation_exposes_every_typed_column() -> None:
    observation = _observation(
        player_hand=[
            card("spades", "A", 1),
            card("spades", "A", 2),
        ]
    )

    batch = tensorize_observation(
        observation=observation,
        device=torch.device("cpu"),
    )

    token_count = len(observation.tokens)
    assert batch.category_ids.shape == (1, token_count, CATEGORY_COUNT)
    assert batch.scalar_values.shape == (1, token_count)
    assert batch.card_rule_values.shape == (1, token_count, 2)
    assert batch.encoded_structure_coordinates.shape == (
        1,
        token_count,
        3,
    )
    assert batch.candidate_category_ids.shape == (
        1,
        CARD_CHOICE_COUNT,
        3,
    )
    assert batch.candidate_counts.shape == (1, CARD_CHOICE_COUNT)
    assert batch.candidate_card_rule_values.shape == (
        1,
        CARD_CHOICE_COUNT,
        2,
    )
    assert batch.query_indices.shape == (1,)
    assert batch.category_ids.dtype == torch.long
    assert batch.scalar_values.dtype == torch.float32
    assert batch.encoded_structure_coordinates.dtype == torch.long
    query_row = batch.encoded_structure_coordinates[
        0, observation.token_sequence.query_index
    ]
    assert int(query_row[int(StructureAxis.TRICK)].item()) == 1


def test_tensorize_observations_pads_only_to_batch_maximum() -> None:
    short = _observation(player_hand=[card("spades", "A", 1)])
    long = _observation(
        player_hand=[
            card("spades", "A", 1),
            card("hearts", "K", 2),
            card("clubs", "3", 1),
        ]
    )

    batch = tensorize_observations(
        observations=(short, long),
        device=torch.device("cpu"),
    )

    assert len(short.tokens) < len(long.tokens)
    assert batch.category_ids.shape[:2] == (2, len(long.tokens))
    assert (
        batch.category_ids[0, len(short.tokens) :, :].count_nonzero()
        == 0
    )
    assert (
        batch.encoded_structure_coordinates[
            0, len(short.tokens) :, :
        ].count_nonzero()
        == 0
    )
    assert len(long.tokens) <= MAX_LOSSLESS_OBSERVATION_TOKENS


def test_card_multiplicity_is_a_universal_scalar() -> None:
    observation = _observation(
        player_hand=[
            card("spades", "A", 1),
            card("spades", "A", 2),
        ]
    )
    batch = tensorize_observation(
        observation=observation,
        device=torch.device("cpu"),
    )

    scalar_values = batch.scalar_values[0]
    assert 2.0 in tuple(
        float(scalar_values[index].item())
        for index in range(int(scalar_values.shape[0]))
    )
    candidate_counts = batch.candidate_counts[0]
    assert {
        float(candidate_counts[index].item())
        for index in range(int(candidate_counts.shape[0]))
    } == {1.0, 2.0}


def _observation(*, player_hand: list[Card]) -> Observation:
    return build_observation(
        viewer=0,
        snapshot=make_snapshot(player_hand=player_hand),
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )
