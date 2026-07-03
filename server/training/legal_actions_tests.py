"""Tests for rule-complete semantic legal action indexes."""

from __future__ import annotations

from server.player.test_helpers import card, make_snapshot
from server.protocol import (
    BidEventSnapshot,
    StirringStateSnapshot,
    TrickSlotSnapshot,
    TrickSnapshot,
)
from server.result import Ok, Rejected
from server.rules.card_faces import CardFace, FaceCount
from server.rules.cards import Card
from server.training.legal_actions import build_legal_action_index
from server.training.semantic_actions import (
    SemanticArgument,
    SemanticArgumentPrefix,
    SemanticArgumentTrace,
)


def test_build_legal_action_index_ignores_action_hints_for_follow() -> (
    None
):
    lead = card("hearts", "A", 1)
    heart = card("hearts", "3", 1)
    spade = card("spades", "K", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[heart, spade],
        action_hints=[],
        trick=_trick(
            lead_player=1,
            current_player=2,
            lead_cards=[lead],
        ),
    )

    legal_actions = build_legal_action_index(
        player_index=2,
        snapshot=snapshot,
    )

    allowed = legal_actions.allowed_next(
        SemanticArgumentPrefix(arguments=())
    )
    assert _select(heart, 1) in allowed
    assert _select(spade, 1) not in allowed


def test_follow_decode_accepts_only_full_rule_legal_play() -> None:
    lead = card("hearts", "A", 1)
    heart = card("hearts", "3", 1)
    spade = card("spades", "K", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[heart, spade],
        trick=_trick(
            lead_player=1,
            current_player=2,
            lead_cards=[lead],
        ),
    )
    legal_actions = build_legal_action_index(
        player_index=2,
        snapshot=snapshot,
    )

    decoded = legal_actions.decode(
        SemanticArgumentTrace(arguments=(_select(heart, 1),))
    )
    rejected = legal_actions.decode(
        SemanticArgumentTrace(arguments=(_select(spade, 1),))
    )

    assert isinstance(decoded, Ok)
    assert isinstance(rejected, Rejected)


def test_lead_mask_keeps_selected_cards_in_one_effective_suit() -> None:
    heart = card("hearts", "3", 1)
    spade = card("spades", "K", 1)
    snapshot = make_snapshot(
        phase="PLAYING",
        awaiting_action="play",
        player_hand=[heart, spade],
    )
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
    )

    prefix = SemanticArgumentPrefix(arguments=(_select(heart, 1),))
    allowed = legal_actions.allowed_next(prefix)

    assert SemanticArgument("stop") in allowed
    assert _select(spade, 1) not in allowed


def test_discard_auto_completes_at_exact_count_without_stop() -> None:
    first = card("hearts", "3", 1)
    second = card("spades", "K", 1)
    snapshot = make_snapshot(
        phase="STIRRING",
        awaiting_action="discard",
        player_hand=[first, second],
        stirring_state=StirringStateSnapshot(
            phase="EXCHANGING",
            trump_suit=None,
            current_player=0,
            declarer_player=0,
            exchanging_player=0,
            exchange_count=2,
        ),
    )
    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
    )

    trace = SemanticArgumentTrace(
        arguments=(_select(first, 1), _select(second, 1))
    )

    assert (
        legal_actions.allowed_next(
            SemanticArgumentPrefix(arguments=trace.arguments)
        )
        == ()
    )
    assert isinstance(legal_actions.decode(trace), Ok)


def test_bid_current_winner_can_only_pass() -> None:
    first = card("hearts", "2", 1)
    second = card("hearts", "2", 2)
    snapshot = make_snapshot(
        phase="DEAL_BID",
        awaiting_action="bid",
        trump_rank="2",
        player_hand=[first, second],
        bid_winner=BidEventSnapshot(
            player=0,
            cards=[first],
            kind="trump_rank",
            suit=first.suit,
            joker_type=None,
            count=1,
        ),
    )

    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
    )

    assert legal_actions.allowed_next(
        SemanticArgumentPrefix(arguments=())
    ) == (SemanticArgument("pass"),)


def test_stir_mask_uses_current_priority() -> None:
    heart_first = card("hearts", "2", 1)
    heart_second = card("hearts", "2", 2)
    spade_first = card("spades", "2", 1)
    spade_second = card("spades", "2", 2)
    diamond_first = card("diamonds", "2", 1)
    diamond_second = card("diamonds", "2", 2)
    snapshot = make_snapshot(
        phase="STIRRING",
        awaiting_action="stir",
        trump_rank="2",
        trump_suit="hearts",
        player_hand=[
            spade_first,
            spade_second,
            diamond_first,
            diamond_second,
        ],
        bid_winner=BidEventSnapshot(
            player=1,
            cards=[heart_first, heart_second],
            kind="trump_rank",
            suit=heart_first.suit,
            joker_type=None,
            count=2,
        ),
        stirring_state=StirringStateSnapshot(
            phase="WAITING",
            trump_suit=heart_first.suit,
            current_player=0,
            declarer_player=1,
            exchanging_player=None,
            exchange_count=None,
        ),
    )

    legal_actions = build_legal_action_index(
        player_index=0,
        snapshot=snapshot,
    )
    allowed = legal_actions.allowed_next(
        SemanticArgumentPrefix(arguments=())
    )

    assert SemanticArgument("pass") in allowed
    assert _select(spade_first, 2) in allowed
    assert _select(diamond_first, 2) not in allowed


def _trick(
    *,
    lead_player: int,
    current_player: int,
    lead_cards: list[Card],
) -> TrickSnapshot:
    return TrickSnapshot(
        lead_player=lead_player,
        current_player=current_player,
        slots=[
            TrickSlotSnapshot(player=0, cards=[]),
            TrickSlotSnapshot(
                player=lead_player,
                cards=list(lead_cards),
            ),
            TrickSlotSnapshot(player=2, cards=[]),
            TrickSlotSnapshot(player=3, cards=[]),
        ],
    )


def _select(card_value: Card, count: int) -> SemanticArgument:
    return SemanticArgument(
        "select_face_count",
        FaceCount(
            CardFace(card_value.suit, card_value.rank),
            count,
        ),
    )
