"""Deterministic card features derived from current round rules."""

from __future__ import annotations

from dataclasses import dataclass

from server.game.rules.card_faces import CardFace
from server.game.rules.cards import Rank, Suit
from server.game.rules.ordering import RANK_ORDER
from server.training.relative_state import TrumpMode
from server.training.relative_state.contexts import RoundContext


@dataclass(frozen=True, slots=True)
class CardRuleFeatures:
    """Known card semantics shared by input and output encoders."""

    effective_suit_id: int
    point_value: float
    relative_strength: float


def card_rule_features(
    face: CardFace, context: RoundContext
) -> CardRuleFeatures:
    """Return exact rule-derived features for one printed card face."""
    trump_suit = (
        context.trump.suit
        if context.trump.mode == TrumpMode.SUITED
        else None
    )
    is_trump = (
        face.suit == Suit.JOKER
        or face.rank == context.level_rank
        or (trump_suit is not None and face.suit == trump_suit)
    )
    effective_suit_id = (
        1 if is_trump else _printed_suit_id(face.suit) + 1
    )
    strength = _strength(face, trump_suit, context.level_rank)
    return CardRuleFeatures(
        effective_suit_id=effective_suit_id,
        point_value=float(face.points) / 10.0,
        relative_strength=float(strength) / 100.0,
    )


def _strength(
    face: CardFace,
    trump_suit: Suit | None,
    trump_rank: Rank,
) -> int:
    if face.rank == Rank.BIG_JOKER:
        return 100
    if face.rank == Rank.SMALL_JOKER:
        return 90
    if face.rank == trump_rank:
        if trump_suit is not None and face.suit == trump_suit:
            return 80
        return 70
    if trump_suit is not None and face.suit == trump_suit:
        return 45 + RANK_ORDER[face.rank]
    return RANK_ORDER[face.rank] - 2


def _printed_suit_id(suit: Suit) -> int:
    return (
        Suit.HEARTS,
        Suit.SPADES,
        Suit.DIAMONDS,
        Suit.CLUBS,
        Suit.JOKER,
    ).index(suit) + 1


__all__ = ("CardRuleFeatures", "card_rule_features")
