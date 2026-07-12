"""Tests for player action data containers."""

from __future__ import annotations

from typing import Literal

from server.game.room.actions import (
    BidAction,
    DiscardAction,
    NextRoundAction,
    PlayAction,
    SkipBidAction,
    SkipStirAction,
    StirAction,
)
from server.game.rules.cards import POINTS_MAP, Card, Rank, Suit


def _card(suit: Suit, rank: Rank, deck: Literal[1, 2] = 1) -> Card:
    return Card(
        id=f"D{deck}-{suit.value}-{rank.value}",
        suit=suit,
        rank=rank,
        points=POINTS_MAP[rank],
    )


def test_bid_action_fields() -> None:
    c1 = _card(Suit.HEARTS, Rank.TWO, 1)
    c2 = _card(Suit.HEARTS, Rank.TWO, 2)
    action = BidAction(cards=[c1, c2], count=2)
    assert action.cards == [c1, c2]
    assert action.count == 2


def test_play_action_fields() -> None:
    c1 = _card(Suit.SPADES, Rank.ACE, 1)
    action = PlayAction(cards=[c1])
    assert action.cards == [c1]


def test_stir_action_fields() -> None:
    c1 = _card(Suit.HEARTS, Rank.TWO, 1)
    c2 = _card(Suit.HEARTS, Rank.TWO, 2)
    action = StirAction(cards=[c1, c2])
    assert action.cards == [c1, c2]


def test_skip_bid_action_fields() -> None:
    action = SkipBidAction()
    assert isinstance(action, SkipBidAction)


def test_skip_stir_action_fields() -> None:
    action = SkipStirAction()
    assert isinstance(action, SkipStirAction)


def test_discard_action_fields() -> None:
    c1 = _card(Suit.DIAMONDS, Rank.THREE, 1)
    c2 = _card(Suit.CLUBS, Rank.FOUR, 1)
    c3 = _card(Suit.SPADES, Rank.FIVE, 1)
    action = DiscardAction(cards=[c1, c2, c3])
    assert action.cards == [c1, c2, c3]


def test_next_round_action_fields() -> None:
    action = NextRoundAction()
    assert isinstance(action, NextRoundAction)
