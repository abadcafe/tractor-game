"""Black-box tests for lossless typed observation tensorization."""

from __future__ import annotations

import torch

from server.game import Seat
from server.game.rules.cards import Card
from server.policy_model._schema.actions import CARD_CHOICE_COUNT
from server.policy_model._schema.encoding import (
    CATEGORY_COUNT,
)
from server.policy_model.observation import (
    Observation,
    build_observation,
)
from server.policy_model.observation.history import (
    ObservationMemoryView,
)
from server.policy_model.observation.packing import (
    MAX_LOSSLESS_OBSERVATION_TOKENS,
)
from server.policy_model.observation.structure import StructureAxis
from server.policy_model.observation.tensor_batch import (
    tensorize_observation,
    tensorize_observations,
)
from tests.support import card
from tests.support import snapshot as make_snapshot


def test_tensorize_observation_exposes_every_typed_column() -> None:
    observation = _observation(
        hand=[
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
    short = _observation(hand=[card("spades", "A", 1)])
    long = _observation(
        hand=[
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
        hand=[
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


def test_canonical_observation_has_exact_vocab_and_structure_rows() -> (
    None
):
    """Pin the model-facing vocabulary without an opaque hash."""
    observation = _observation(
        hand=[
            card("spades", "A", 1),
            card("spades", "A", 2),
        ]
    )
    batch = tensorize_observation(
        observation=observation,
        device=torch.device("cpu"),
    )

    expected_categories: tuple[tuple[int, ...], ...] = (
        (1, 1, 0, 13, 0, 0, 0, 0, 0, 0, 0),
        (2, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        (2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        (2, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        (2, 5, 0, 13, 0, 0, 0, 0, 0, 0, 0),
        (2, 6, 0, 13, 0, 0, 0, 0, 0, 0, 0),
        (2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (2, 9, 0, 0, 0, 0, 0, 2, 0, 0, 0),
        (2, 10, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        (2, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (2, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0),
        (2, 12, 3, 0, 0, 0, 0, 0, 0, 0, 0),
        (2, 12, 4, 0, 0, 0, 0, 0, 0, 0, 0),
        (5, 15, 0, 13, 2, 3, 0, 0, 0, 0, 1),
        (4, 14, 0, 0, 0, 0, 4, 8, 0, 1, 0),
    )
    expected_scalars: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        12.0,
        12.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
    )
    expected_structure: tuple[tuple[int, int, int], ...] = (
        *((0, 0, 0) for _ in range(15)),
        (0, 1, 1),
    )
    assert torch.equal(
        batch.category_ids[0],
        torch.tensor(expected_categories, dtype=torch.long),
    )
    assert torch.equal(
        batch.scalar_values[0],
        torch.tensor(expected_scalars, dtype=torch.float32),
    )
    assert torch.equal(
        batch.encoded_structure_coordinates[0],
        torch.tensor(expected_structure, dtype=torch.long),
    )
    assert torch.equal(
        batch.card_rule_values[0, 14],
        torch.tensor([0.0, 0.12], dtype=torch.float32),
    )
    assert int(batch.query_indices[0].item()) == 15


def _observation(*, hand: list[Card]) -> Observation:
    return build_observation(
        viewer=Seat.A,
        snapshot=make_snapshot(hand=hand),
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )
