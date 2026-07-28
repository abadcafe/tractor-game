"""Black-box tests for the typed observation tokenizer."""

from __future__ import annotations

from server.foundation.result import Ok
from server.game import Seat
from server.game.snapshots.tricks import (
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.policy_model.observation.history import (
    ObservationMemoryView,
)
from server.policy_model.observation.relative_state import (
    project_relative_observation,
)
from server.policy_model.observation.tokenization import (
    ActionToken,
    CardToken,
    GlobalToken,
    RoundToken,
    TokenFamily,
    TokenSequence,
    TrickToken,
    tokenize,
)
from tests.support import card, seat_values
from tests.support import snapshot as make_snapshot


def test_tokenize_emits_only_five_semantic_families() -> None:
    sequence = _sequence()

    assert {node.family for node in sequence.nodes} == {
        TokenFamily.GLOBAL_CONTEXT,
        TokenFamily.ROUND_CONTEXT,
        TokenFamily.TRICK_CONTEXT,
        TokenFamily.ACTION,
        TokenFamily.CARD,
    }
    assert all(
        isinstance(
            node.payload,
            (
                GlobalToken,
                RoundToken,
                TrickToken,
                ActionToken,
                CardToken,
            ),
        )
        for node in sequence.nodes
    )


def test_tokenize_binds_play_cards_by_address_not_card_fields() -> None:
    sequence = _sequence()
    play = next(
        node
        for node in sequence.nodes
        if isinstance(node.payload, ActionToken)
        and node.payload.occurrence == "fact"
        and node.payload.kind == "play"
    )
    played = next(
        node
        for node in sequence.nodes
        if isinstance(node.payload, CardToken)
        and node.address.payload_role == "played"
    )

    assert played.address.trick == play.address.trick
    assert played.address.play_position == (play.address.play_position)
    assert not hasattr(played.payload, "actor")
    assert not hasattr(played.payload, "trick_position")
    assert not hasattr(played.payload, "play_width")


def test_tokenize_card_count_is_shared_numeric_multiplicity() -> None:
    sequence = _sequence()
    hand = [
        node.payload
        for node in sequence.nodes
        if isinstance(node.payload, CardToken)
        and node.address.payload_role == "hand"
    ]

    assert [(token.face.rank.value, token.count) for token in hand] == [
        ("A", 2)
    ]
    assert all(
        not hasattr(token, "face_count_residual") for token in hand
    )


def test_tokenize_query_is_action_pooling_anchor() -> None:
    sequence = _sequence()

    query = sequence.nodes[sequence.query_index]
    assert isinstance(query.payload, ActionToken)
    assert query.payload.occurrence == "query"
    assert query.payload.kind == "play"
    assert query.payload.trick_position == "follow_1"


def _sequence() -> TokenSequence:
    first = card("spades", "A", 1)
    second = card("spades", "A", 2)
    trick = TrickSnapshot(
        lead_actor=Seat.B,
        current_actor=Seat.C,
        slots=(
            TrickSlotSnapshot(actor=Seat.A, cards=()),
            TrickSlotSnapshot(
                actor=Seat.B,
                cards=(card("hearts", "K"),),
            ),
            TrickSlotSnapshot(actor=Seat.C, cards=()),
            TrickSlotSnapshot(actor=Seat.D, cards=()),
        ),
    )
    projected = project_relative_observation(
        viewer=Seat.C,
        snapshot=make_snapshot(
            phase="PLAYING",
            awaiting_action="play",
            declarer=Seat.A,
            hand=[first, second],
            remaining_cards=seat_values(3, 4, 2, 5),
            trick=trick,
        ),
        memory=ObservationMemoryView(
            bid_actions=(), completed_tricks=()
        ),
    )
    assert isinstance(projected, Ok)
    return tokenize(projected.value)
